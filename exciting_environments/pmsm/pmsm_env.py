import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import chex
from functools import partial
import diffrax
from exciting_environments import CoreEnvironment
import exciting_environments as exc_envs
from typing import Callable
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator, griddata
from pathlib import Path
import os


t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])  # only for alpha/beta -> abc
t23 = 2 / 3 * jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]]).T  # only for abc -> alpha/beta

inverter_t_abc = jnp.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
)

ROTATION_MAP = np.ones((2, 2, 2), dtype=np.complex64)
ROTATION_MAP[1, 0, 1] = 0.5 * (1 + np.sqrt(3) * 1j)
ROTATION_MAP[1, 1, 0] = 0.5 * (1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 0] = 0.5 * (-1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 1] = -1
ROTATION_MAP[0, 0, 1] = 0.5 * (-1 + np.sqrt(3) * 1j)
ROTATION_MAP = jnp.array(ROTATION_MAP)


def t_dq_alpha_beta(eps):
    cos = jnp.cos(eps)
    sin = jnp.sin(eps)
    return jnp.column_stack((cos, sin, -sin, cos)).reshape(2, 2)


def dq2abc(u_dq, eps):
    u_abc = t32 @ dq2albet(u_dq, eps).T
    return u_abc.T


def dq2albet(u_dq, eps):
    q = t_dq_alpha_beta(-eps)
    u_alpha_beta = q @ u_dq.T

    return u_alpha_beta.T


def albet2dq(u_albet, eps):
    q_inv = t_dq_alpha_beta(eps)
    u_dq = q_inv @ u_albet.T

    return u_dq.T


def abc2dq(u_abc, eps):
    u_alpha_beta = t23 @ u_abc.T
    u_dq = albet2dq(u_alpha_beta.T, eps)
    return u_dq


def step_eps(eps, omega_el, tau, tau_scale=1.0):
    eps += omega_el * tau * tau_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def apply_hex_constraint(u_albet):
    """Clip voltages in alpha/beta coordinates into the voltage hexagon"""
    u_albet_c = u_albet[0] + 1j * u_albet[1]
    idx = (jnp.sin(jnp.angle(u_albet_c)[..., jnp.newaxis] - 2 / 3 * jnp.pi * jnp.arange(3)) >= 0).astype(int)
    rot_vec = ROTATION_MAP[idx[0], idx[1], idx[2]]
    # rotate sectors upwards
    u_albet_c = jnp.multiply(u_albet_c, rot_vec)
    u_albet_c = jnp.clip(u_albet_c.real, -2 / 3, 2 / 3) + 1j * u_albet_c.imag
    u_albet_c = u_albet_c.real + 1j * jnp.clip(u_albet_c.imag, 0, 2 / 3 * jnp.sqrt(3))
    u_albet_c = jnp.multiply(u_albet_c, jnp.conjugate(rot_vec))  # rotate back
    return jnp.column_stack([u_albet_c.real, u_albet_c.imag])


def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, tau):
    eps_advanced = step_eps(eps, omega_el, tau, 0.5)
    u_abc = dq2abc(u_dq, eps_advanced)
    # clip in abc coordinates
    u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
    u_dq = abc2dq(u_abc, eps)
    return u_dq


def switching_state_to_dq(switching_state, u_dc, eps):
    u_abc = inverter_t_abc[switching_state] * u_dc
    u_dq = abc2dq(u_abc, eps)
    return u_dq[0]


def currents_to_torque(i_d, i_q, p, psi_p, l_d, l_q):
    torque = 1.5 * p * (psi_p + (l_d - l_q) * i_d) * i_q
    return torque


# def currents_to_torque_saturated(Psi_d, Psi_q, i_d, i_q):
#     return Psi_d * i_q - Psi_q * i_d


def calc_max_torque(l_d, l_q, i_n, psi_p, p):
    i_d = jnp.where(l_d == l_q, 0, -psi_p / (4 * (l_d - l_q)) - jnp.sqrt((psi_p / (4 * (l_d - l_q))) ** 2 + i_n**2 / 2))
    i_q = jnp.sqrt(i_n**2 - i_d**2)
    max_torque = 1.5 * p * (psi_p + (l_d - l_q) * i_d) * i_q
    return max_torque


class PMSM_Physical:
    def __init__(
        self,
        batch_size: int = 8,
        params: dict = None,
        deadtime: int = 1,
        tau: float = 1e-4,
        solver: Callable = diffrax.Euler(),
        control_state="torque",
        saturated=False,
    ):

        if not params and saturated:
            params = {
                "p": 3,
                "r_s": 15e-3,
                "l_d": 0.37e-3,
                "l_q": 1.2e-3,
                "psi_p": 65.6e-3,
                "u_dc": 400,
                "i_n": 250,
                "max_omega_el": 1500 / 60 * 2 * jnp.pi,
            }

        if not params:
            params = {
                "p": 3,
                "r_s": 1,
                "l_d": 0.37e-3,
                "l_q": 1.2e-3,
                "psi_p": 65.6e-3,
                "u_dc": 400,
                "i_n": 250,
                "max_omega_el": 100 / 60 * 2 * jnp.pi,
            }

        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver

        self._action_description = ["u_d", "u_q"]

        if saturated:
            saturated_quants = [
                "L_dd",
                "L_dq",
                "L_qd",
                "L_qq",
                "Psi_d",
                "Psi_q",
            ]
            self.pmsm_lut = loadmat(os.path.dirname(exc_envs.pmsm.__file__) + "\\LUT_jax_grad.mat")  # "\\LUT_data.mat"
            for q in saturated_quants:
                qmap = self.pmsm_lut[q]
                x, y = np.indices(qmap.shape)
                nan_mask = np.isnan(qmap)
                qmap[nan_mask] = griddata(
                    (x[~nan_mask], y[~nan_mask]),  # points we know
                    qmap[~nan_mask],  # values we know
                    (x[nan_mask], y[nan_mask]),  # points to interpolate
                    method="nearest",
                )  # extrapolation can only do nearest
                self.pmsm_lut[q] = qmap

            i_max = params["i_n"]
            assert i_max == 250, "LUT_data was generated with i_max=250"

            n_grid_points_y, n_grid_points_x = self.pmsm_lut[saturated_quants[0]].shape
            x, y = np.linspace(-i_max, 0, n_grid_points_x), np.linspace(-i_max, i_max, n_grid_points_y)
            self.LUT_interpolators = {
                q: jax.scipy.interpolate.RegularGridInterpolator(
                    (x, y), self.pmsm_lut[q][:, :].T, method="linear", bounds_error=False, fill_value=None
                )
                for q in saturated_quants
            }

        params = self.PhysicalParams(**params)
        self.properties = self.Properties(
            control_state=control_state, physical_params=params, deadtime=deadtime, saturated=saturated
        )

        if deadtime > 0:
            self.initial_action_buffer = jnp.zeros((self.batch_size, deadtime, 2))
        else:
            self.initial_action_buffer = jnp.zeros((self.batch_size, 0, 2))

    @jdc.pytree_dataclass
    class PhysicalParams:
        """Dataclass containing the physical parameters of the environment."""

        p: jax.Array
        r_s: jax.Array
        l_d: jax.Array
        l_q: jax.Array
        psi_p: jax.Array
        u_dc: jax.Array
        i_n: jax.Array
        max_omega_el: jax.Array

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        action_buffer: jax.Array
        epsilon: jax.Array
        i_d: jax.Array
        i_q: jax.Array
        torque: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Properties:
        """Dataclass containing the properties of the physical pmsm environment."""

        saturated: bool
        deadtime: jax.Array
        control_state: jax.Array
        physical_params: jdc.pytree_dataclass

    @partial(jax.jit, static_argnums=0)
    def init_state(self):
        """Returns default initial state for all batches."""
        state = self.PhysicalState(
            action_buffer=self.initial_action_buffer,
            epsilon=jnp.zeros(self.batch_size),
            i_d=jnp.zeros(self.batch_size),
            i_q=jnp.zeros(self.batch_size),
            torque=jnp.zeros(self.batch_size),
            omega=jnp.zeros(self.batch_size),
        )
        return state

    def reset(self):
        return self.init_state()

    def ode_step(self, system_state, u_dq, properties):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_dq: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """

        omega_el = system_state.omega
        i_d = system_state.i_d
        i_q = system_state.i_q
        eps = system_state.epsilon

        args = (u_dq, properties.physical_params)
        if properties.saturated:

            def vector_field(t, y, args):
                i_d, i_q = y
                u_dq, _ = args

                J_k = jnp.array([[0, -1], [1, 0]])
                i_dq = jnp.array([i_d, i_q])
                p_d = {q: interp(jnp.array([i_d, i_q])) for q, interp in self.LUT_interpolators.items()}
                L_diff = jnp.column_stack([p_d[q] for q in ["L_dd", "L_dq", "L_qd", "L_qq"]]).reshape(2, 2)
                L_diff_inv = jnp.linalg.inv(L_diff)
                psi_dq = jnp.column_stack([p_d[psi] for psi in ["Psi_d", "Psi_q"]]).reshape(-1)
                di_dq_1 = jnp.einsum(
                    "ij,j->i",
                    (-L_diff_inv * properties.physical_params.r_s),
                    i_dq,
                )
                di_dq_2 = jnp.einsum("ik,k->i", L_diff_inv, u_dq)
                di_dq_3 = jnp.einsum("ij,jk,k->i", -L_diff_inv, J_k, psi_dq) * omega_el
                i_dq_diff = di_dq_1 + di_dq_2 + di_dq_3
                d_y = i_dq_diff[0], i_dq_diff[1]
                return d_y

        else:

            def vector_field(t, y, args):
                i_d, i_q = y
                u_dq, params = args
                u_d = u_dq[0]
                u_q = u_dq[1]
                l_d = params.l_d
                l_q = params.l_q
                psi_p = params.psi_p
                r_s = params.r_s
                i_d_diff = (u_d + omega_el * l_q * i_q - r_s * i_d) / l_d
                i_q_diff = (u_q - omega_el * (l_d * i_d + psi_p) - r_s * i_q) / l_q
                d_y = i_d_diff, i_q_diff  # [0]
                return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([i_d, i_q])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        i_d_k1 = y[0]
        i_q_k1 = y[1]

        with jdc.copy_and_mutate(system_state, validate=True) as system_state_next:
            system_state_next.epsilon = step_eps(eps, omega_el, self.tau, 1.0)
            system_state_next.i_d = i_d_k1
            system_state_next.i_q = i_q_k1
            system_state_next.torque = jnp.array(
                [
                    currents_to_torque(
                        i_d_k1,
                        i_q_k1,
                        properties.physical_params.p,
                        properties.physical_params.psi_p,
                        properties.physical_params.l_d,
                        properties.physical_params.l_q,
                    )
                ]
            )[0]

        return system_state_next

    def simulation_step(self, system_state, action, properties):
        """Computes state by simulating one step taking the deadtime into account.

        Args:
            system_state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """
        action_buffer = system_state.action_buffer

        if properties.deadtime > 0:

            future_u_dq = action

            updated_buffer = jnp.concatenate([action_buffer[1:, :], future_u_dq[None, :]], axis=0)
            u_dq = action_buffer[0, :]
        else:
            updated_buffer = action_buffer

            u_dq = action

        next_system_state = self.ode_step(system_state, u_dq, properties)
        with jdc.copy_and_mutate(next_system_state, validate=True) as next_system_state_update:
            next_system_state_update.action_buffer = updated_buffer

        return next_system_state_update

    @property
    def action_description(self):
        return self._action_description


class PMSM(CoreEnvironment):
    def __init__(
        self,
        pmsm_physical: PMSM_Physical,
        batch_size: int = 8,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            pmsm_physical(PMSM_Physical): Physical part of the pmsm environment.
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical state of the environment.
                action_buffer(float): Rotation angle. Default: jnp.pi
                epsilon(float): Angular velocity. Default: 10
                i_d(float):
                i_q(float):
                omega(float):
                torque(float):
            action_constraints(dict): Constraints of actions.
                u_d(float):
                u_q(float):
            static_params(dict): Parameters of environment which do not change during simulation.
                p_omega(float):
                p_reference(float):
                p_reset(float):
                i_lim_multiplier(float):
                constant_omega(bool):
                omega_ramp_min(float):
                omega_ramp_max(float):
                gamma(float):
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        self.pmsm_physical = pmsm_physical
        self.batch_size = batch_size
        assert self.batch_size == self.pmsm_physical.batch_size

        if not static_params:
            static_params = {
                "p_omega": 0.00005,
                "p_reference": 0.0002,
                "p_reset": 1.0,
                "i_lim_multiplier": 1.2,
                "constant_omega": False,
                "omega_ramp_min": 20000,
                "omega_ramp_max": 25000,
                "gamma": 0.85,
            }
            if self.pmsm_physical.properties.saturated:
                static_params["i_lim_multiplier"] = 1.0

        static_params = self.StaticParams(**static_params, physical_properties=self.pmsm_physical.properties)

        in_axes_phys_prop = self.create_in_axes_dataclass(static_params.physical_properties.physical_params)
        values, _ = tree_flatten(in_axes_phys_prop)
        if values:
            max_torque = jax.vmap(
                calc_max_torque,
                in_axes=(
                    in_axes_phys_prop.l_d,
                    in_axes_phys_prop.l_q,
                    in_axes_phys_prop.i_n,
                    in_axes_phys_prop.psi_p,
                    in_axes_phys_prop.p,
                ),
            )(
                static_params.physical_properties.physical_params.l_d,
                static_params.physical_properties.physical_params.l_q,
                static_params.physical_properties.physical_params.i_n,
                static_params.physical_properties.physical_params.psi_p,
                static_params.physical_properties.physical_params.p,
            )
        else:
            max_torque = calc_max_torque(
                static_params.physical_properties.physical_params.l_d,
                static_params.physical_properties.physical_params.l_q,
                static_params.physical_properties.physical_params.i_n,
                static_params.physical_properties.physical_params.psi_p,
                static_params.physical_properties.physical_params.p,
            )

        if not physical_constraints:
            physical_constraints = {
                "action_buffer": 2 * static_params.physical_properties.physical_params.u_dc / 3,
                "epsilon": jnp.pi,
                "i_d": static_params.physical_properties.physical_params.i_n * static_params.i_lim_multiplier,
                "i_q": static_params.physical_properties.physical_params.i_n * static_params.i_lim_multiplier,
                "omega": static_params.physical_properties.physical_params.max_omega_el,
                "torque": max_torque,
            }

        if not action_constraints:
            action_constraints = {
                "u_d": 2 * static_params.physical_properties.physical_params.u_dc / 3,
                "u_q": 2 * static_params.physical_properties.physical_params.u_dc / 3,
            }

        physical_constraints = self.pmsm_physical.PhysicalState(**physical_constraints)
        action_constraints = self.Actions(**action_constraints)

        if static_params.physical_properties.control_state == "currents":
            self._obs_description = [
                "i_d",
                "i_q",
                "cos_eps",
                "sin_eps",
                "omega_el",
                "torque",
                "i_d_ref",
                "i_q_ref",
                "action_buffer",
            ]

        elif static_params.physical_properties.control_state == "torque":
            self._obs_description = [
                "i_d",
                "i_q",
                "cos_eps",
                "sin_eps",
                "omega_el",
                "torque",
                "torque_ref",
                "action_buffer",
            ]

        env_properties = self.EnvProperties(
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

    @jdc.pytree_dataclass
    class Additions:
        """Dataclass containing additional information for simulation."""

        omega_add: jax.Array
        omega_count: jax.Array
        references: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        p_omega: jax.Array
        p_reference: jax.Array
        p_reset: jax.Array
        i_lim_multiplier: jax.Array
        constant_omega: jax.Array
        omega_ramp_min: jax.Array
        omega_ramp_max: jax.Array
        gamma: jax.Array
        physical_properties: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class Actions:
        """Dataclass containing the actions, that can be applied to the environment."""

        u_d: jax.Array
        u_q: jax.Array

    @jdc.pytree_dataclass
    class State:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        additions: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_constraints: jdc.pytree_dataclass
        action_constraints: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def init_state(self, random_key=None):
        """Returns default initial state for all batches."""
        physical_state = self.pmsm_physical.init_state()
        if random_key == None:
            random_seed = np.random.randint(0, 2**31)  # TODO
            random_key = jax.random.PRNGKey(seed=random_seed)
        keys = jax.random.split(jnp.uint32(random_key), self.batch_size)
        references = jnp.zeros((self.batch_size, 1))
        if self.env_properties.static_params.physical_properties.control_state == "currents":
            i_d_ref, keys = jax.vmap(self.update_reference, in_axes=(0, 0, None))(
                references, keys, self.env_properties.static_params.p_reset
            )
            i_q_ref, keys = jax.vmap(self.update_reference, in_axes=(0, 0, None))(
                references, keys, self.env_properties.static_params.p_reset
            )
            references = jnp.hstack((i_d_ref, i_q_ref))
        else:
            references, keys = jax.vmap(self.update_reference, in_axes=(0, 0, None))(
                references, keys, self.env_properties.static_params.p_reset
            )
        if self.env_properties.static_params.constant_omega:
            omega = jnp.ones(self.batch_size)
            omega_add = None
            omega_count = None
        else:
            omega = jnp.zeros((self.batch_size))
            omega_add = jnp.zeros((self.batch_size))
            omega_count = jnp.zeros((self.batch_size))
            omega, omega_add, omega_count, keys = jax.vmap(self.update_omega, in_axes=(0, 0, 0, 0, None))(
                omega, omega_add, omega_count, keys, self.env_properties.static_params.p_reset
            )

        with jdc.copy_and_mutate(physical_state, validate=True) as physical_state_upd:
            physical_state_upd.omega = omega * self.env_properties.physical_constraints.omega

        opt = self.Additions(omega_add=omega_add, omega_count=omega_count, references=references)
        system_state = self.State(physical_state=physical_state_upd, PRNGKey=jnp.float32(keys), additions=opt)

        return system_state

    def reset(self, random_key=None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.init_state()"
            system_state = initial_state
        else:
            system_state = self.init_state(random_key)
        observations = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties),
        )(system_state, self.env_properties)

        return observations, system_state

    def update_reference(self, reference, key, p):
        """Generates random reference values"""
        random_bool = jax.random.bernoulli(key, p=p)
        key, subkey = jax.random.split(key)
        new_reference = jnp.where(random_bool, jax.random.uniform(subkey, minval=-1.0, maxval=1.0), reference)
        key, subkey = jax.random.split(subkey)
        return new_reference, subkey

    def update_omega(self, omega, omega_add, omega_count, key, p):
        """Updates omega randomly in a smooth fashion and detached from the actual ODE calculations"""
        random_bool = jax.random.bernoulli(key, p=p)
        key, subkey = jax.random.split(key)

        # Add value to omega
        omega += omega_add

        # If new target omega has been reached stop adding values in the future
        omega_count = jnp.where(omega_count > 0, omega_count - 1, omega_count)
        omega_add = jnp.where(omega_count == 0, 0.0, omega_add)

        # Generate new omega targets and define the ramp
        key, subkey = jax.random.split(subkey)
        omega_new = jnp.where(
            random_bool & (omega_add == 0.0), jax.random.uniform(subkey, minval=-1.0, maxval=1.0), omega
        )

        key, subkey = jax.random.split(subkey)
        omega_count = jnp.where(
            omega_new != omega,
            jax.random.choice(
                subkey,
                jnp.arange(
                    self.env_properties.static_params.omega_ramp_min, self.env_properties.static_params.omega_ramp_max
                ),  # TODO change to passed parameter so vmappable
                replace=True,
                axis=0,
            ),
            omega_count,
        )

        omega_add += jnp.where(omega_new != omega, (omega_new - omega) / omega_count, 0.0)

        key, subkey = jax.random.split(subkey)

        return omega, omega_add, omega_count, subkey

    def constraint_denormalization(self, u_dq, system_state, env_properties):
        """Denormalizes the u_dq and clips it with respect to the hexagon."""
        u_albet_norm = dq2albet(
            u_dq,
            step_eps(
                system_state.physical_state.epsilon,
                self.env_properties.static_params.physical_properties.deadtime + 0.5,
                self.tau,
                system_state.physical_state.omega,
            ),
        )
        u_albet_norm_clip = apply_hex_constraint(u_albet_norm)
        u_dq = albet2dq(u_albet_norm_clip, system_state.physical_state.epsilon) * jnp.hstack(
            [env_properties.action_constraints.u_d, env_properties.action_constraints.u_q]
        )
        return u_dq[0]

    def step(self, system_state, actions_norm, env_properties):
        """Computes state and observation by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            action_norm: The normalized action to apply to the environment.
            env_properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            observation: The observation after the step.
            state: The computed state after the one step simulation.
        """
        actions = self.constraint_denormalization(actions_norm, system_state, env_properties)

        next_physical_state = self.pmsm_physical.simulation_step(
            system_state.physical_state, actions, env_properties.static_params.physical_properties
        )

        if not env_properties.static_params.constant_omega:
            omega, omega_add, omega_count, keys = self.update_omega(
                system_state.physical_state.omega / env_properties.physical_constraints.omega,
                system_state.additions.omega_add,
                system_state.additions.omega_count,
                jnp.uint32(system_state.PRNGKey),
                env_properties.static_params.p_omega,
            )
        else:
            omega = system_state.physical_state.omega / env_properties.physical_constraints.omega
            omega_add = system_state.additions.omega_add
            omega_count = system_state.additions.omega_count
            keys = jnp.uint32(system_state.PRNGKey)

        if env_properties.static_params.physical_properties.control_state == "currents":
            i_d_ref, keys = self.update_reference(
                system_state.additions.references[0], keys, env_properties.static_params.p_reference
            )
            i_q_ref, keys = self.update_reference(
                system_state.additions.references[1], keys, env_properties.static_params.p_reference
            )
            references = jnp.hstack((i_d_ref, i_q_ref))
        else:
            references, keys = self.update_reference(
                system_state.additions.references, keys, env_properties.static_params.p_reference
            )

        with jdc.copy_and_mutate(next_physical_state, validate=True) as next_physical_state_upd:
            next_physical_state_upd.omega = omega * self.env_properties.physical_constraints.omega

        opt = self.Additions(omega_add=omega_add, omega_count=omega_count, references=references)
        next_system_state = self.State(physical_state=next_physical_state_upd, PRNGKey=jnp.float32(keys), additions=opt)

        observations = self.generate_observation(next_system_state, env_properties)

        return observations, next_system_state

    def generate_observation(self, system_state, env_properties):
        """Returns observation for one batch."""
        physical_constraints = env_properties.physical_constraints
        eps = system_state.physical_state.epsilon
        cos_eps = jnp.cos(eps)
        sin_eps = jnp.sin(eps)
        obs = jnp.hstack(
            (
                system_state.physical_state.i_d / physical_constraints.i_d,
                system_state.physical_state.i_q / physical_constraints.i_q,
                system_state.physical_state.omega / physical_constraints.omega,
                system_state.physical_state.torque / physical_constraints.torque,
                cos_eps,
                sin_eps,
                system_state.additions.references,
                system_state.physical_state.action_buffer.reshape(-1) / physical_constraints.action_buffer,
            )
        )
        return obs

    def generate_truncated(self, system_state, env_properties):
        """Returns truncated information for one batch."""
        physical_constraints = env_properties.physical_constraints
        physical_state = system_state.physical_state
        i_d_norm = physical_state.i_d / physical_constraints.i_d
        i_q_norm = physical_state.i_q / physical_constraints.i_q
        i_s = jnp.sqrt(i_d_norm**2 + i_q_norm**2)
        return i_s > 1

    def generate_terminated(self, system_state, reward, env_properties):
        """Returns terminated information for one batch."""
        return self.generate_truncated(system_state, env_properties)

    def generate_reward(self, system_state, action, env_properties):
        """Returns reward for one batch."""
        physical_state = system_state.physical_state
        references = system_state.additions.references
        if env_properties.static_params.physical_properties.control_state == "currents":
            reward = self.current_reward_func(
                physical_state.i_d / env_properties.physical_constraints.i_d,
                physical_state.i_q / env_properties.physical_constraints.i_q,
                references[0],
                references[1],
                env_properties.static_params.gamma,
            )
        elif env_properties.static_params.physical_properties.control_state == "torque":
            reward = self.torque_reward_func(
                (physical_state.i_d * (env_properties.physical_constraints.i_d) ** -1),
                (physical_state.i_q * (env_properties.physical_constraints.i_q) ** -1),
                (physical_state.torque / (env_properties.physical_constraints.torque)),
                references,
                env_properties.static_params.i_lim_multiplier,
                env_properties.static_params.gamma,
            )

        return reward

    def current_reward_func(self, i_d, i_q, i_d_ref, i_q_ref, gamma):
        mse = 0.5 * (i_d - i_d_ref) ** 2 + 0.5 * (i_q - i_q_ref) ** 2
        return -1 * (mse * (1 - gamma))

    def torque_reward_func(self, i_d, i_q, torque, torque_ref, i_lim_multiplier, gamma):
        i_s = jnp.sqrt(i_d**2 + i_q**2)
        i_n = 1 / i_lim_multiplier
        i_d_plus = 0.2 * i_n
        torque_tol = 0.01
        rew = jnp.zeros_like(torque_ref)
        rew = jnp.where(i_s > 1, -1 * jnp.abs(i_s), rew)
        rew = jnp.where((i_s < 1.0) & (i_s > i_n), 0.5 * (1 - (i_s - i_n) / (1 - i_n)) - 1, rew)
        rew = jnp.where((i_s < i_n) & (i_d > i_d_plus), -0.5 * ((i_d - i_d_plus) / (i_n - i_d_plus)), rew)
        rew = jnp.where(
            (i_s < i_n) & (i_d < i_d_plus) & (jnp.abs(torque - torque_ref) > torque_tol),
            0.5 * (1 - jnp.abs((torque_ref - torque) / 2)),
            rew,
        )
        rew = jnp.where(
            (i_s < i_n) & (i_d < i_d_plus) & (jnp.abs(torque - torque_ref) < torque_tol), 1 - 0.5 * i_s, rew
        )
        return rew * (1 - gamma)

    @property
    def action_description(self):
        return self.pmsm._action_description

    @property
    def obs_description(self):
        return self._obs_description

    @property
    def control_state(self):
        return self.pmsm.control_state
