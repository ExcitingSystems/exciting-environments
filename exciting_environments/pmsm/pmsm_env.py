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
from dataclasses import fields

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


def calc_max_torque(l_d, l_q, i_n, psi_p, p):
    i_d = jnp.where(l_d == l_q, 0, -psi_p / (4 * (l_d - l_q)) - jnp.sqrt((psi_p / (4 * (l_d - l_q))) ** 2 + i_n**2 / 2))
    i_q = jnp.sqrt(i_n**2 - i_d**2)
    max_torque = 1.5 * p * (psi_p + (l_d - l_q) * i_d) * i_q
    return max_torque


class PMSM(CoreEnvironment):
    def __init__(
        self,
        batch_size: int = 8,
        saturated=False,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):

        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver

        if not physical_constraints:
            physical_constraints = {
                "u_d_buffer": 2 * 400 / 3,
                "u_q_buffer": 2 * 400 / 3,
                "epsilon": jnp.pi,
                "i_d": 250,
                "i_q": 250,
                "omega_el": 3000 / 60 * 2 * jnp.pi,
                "torque": 200,
            }

        if not action_constraints:
            action_constraints = {
                "u_d": 2 * 400 / 3,
                "u_q": 2 * 400 / 3,
            }

        saturated_quants = [
            "L_dd",
            "L_dq",
            "L_qd",
            "L_qq",
            "Psi_d",
            "Psi_q",
        ]
        if saturated:
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

            i_max = physical_constraints["i_d"]
            assert i_max == 250, "LUT_data was generated with i_max=250"

            n_grid_points_y, n_grid_points_x = self.pmsm_lut[saturated_quants[0]].shape
            x, y = np.linspace(-i_max, 0, n_grid_points_x), np.linspace(-i_max, i_max, n_grid_points_y)
            self.LUT_interpolators = {
                q: jax.scipy.interpolate.RegularGridInterpolator(
                    (x, y), self.pmsm_lut[q][:, :].T, method="linear", bounds_error=False, fill_value=None
                )
                for q in saturated_quants
            }
        else:
            self.LUT_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        if not static_params and saturated:
            static_params = {
                "p": 3,
                "r_s": 15e-3,
                "l_d": jnp.nan,
                "l_q": jnp.nan,
                "psi_p": jnp.nan,
                "deadtime": 1,
            }

        if not static_params:
            static_params = {
                "p": 3,
                "r_s": 1,
                "l_d": 0.37e-3,
                "l_q": 1.2e-3,
                "psi_p": 65.6e-3,
                "deadtime": 1,
            }

        if not control_state:
            control_state = []

        self.control_state = control_state

        static_params = self.StaticParams(**static_params)
        physical_constraints = self.PhysicalState(**physical_constraints)
        action_constraints = self.Action(**action_constraints)

        # in_axes_phys_prop = self.create_in_axes_dataclass(static_params)
        # values, _ = tree_flatten(in_axes_phys_prop)
        # if values:
        #     max_torque = jax.vmap(
        #         calc_max_torque,
        #         in_axes=(
        #             in_axes_phys_prop.l_d,
        #             in_axes_phys_prop.l_q,
        #             in_axes_phys_prop.i_n,
        #             in_axes_phys_prop.psi_p,
        #             in_axes_phys_prop.p,
        #         ),
        #     )(
        #         static_params.physical_properties.static_params.l_d,
        #         static_params.physical_properties.static_params.l_q,
        #         static_params.physical_properties.static_params.i_n,
        #         static_params.physical_properties.static_params.psi_p,
        #         static_params.physical_properties.static_params.p,
        #     )
        # else:
        #     max_torque = calc_max_torque(
        #         static_params.physical_properties.static_params.l_d,
        #         static_params.physical_properties.static_params.l_q,
        #         static_params.physical_properties.static_params.i_n,
        #         static_params.physical_properties.static_params.psi_p,
        #         static_params.physical_properties.static_params.p,
        #     )

        env_properties = self.EnvProperties(
            saturated=saturated,
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

        self._action_description = ["u_d", "u_q"]
        self._obs_description = [
            "i_d",
            "i_q",
            "cos_eps",
            "sin_eps",
            "omega_el",
            "torque",
            "u_d_buffer",
            "u_q_buffer",
        ]
        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the physical parameters of the environment."""

        p: jax.Array
        r_s: jax.Array
        l_d: jax.Array
        l_q: jax.Array
        psi_p: jax.Array
        deadtime: jax.Array

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        u_d_buffer: jax.Array
        u_q_buffer: jax.Array
        epsilon: jax.Array
        i_d: jax.Array
        i_q: jax.Array
        torque: jax.Array
        omega_el: jax.Array

    @jdc.pytree_dataclass
    class Additions:
        pass

    @jdc.pytree_dataclass
    class Action:
        """Dataclass containing the actions, that can be applied to the environment."""

        u_d: jax.Array
        u_q: jax.Array

    @jdc.pytree_dataclass
    class State:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        additions: jdc.pytree_dataclass
        reference: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        saturated: jax.Array
        physical_constraints: jdc.pytree_dataclass
        action_constraints: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def currents_to_torque(self, i_d, i_q, env_properties):
        torque = (
            1.5
            * env_properties.static_params.p
            * (
                env_properties.static_params.psi_p
                + (env_properties.static_params.l_d - env_properties.static_params.l_q) * i_d
            )
            * i_q
        )
        return torque

    def currents_to_torque_saturated(self, i_d, i_q, env_properties):
        Psi_d = self.LUT_interpolators["Psi_d"](jnp.array([i_d, i_q]))
        Psi_q = self.LUT_interpolators["Psi_q"](jnp.array([i_d, i_q]))
        return (Psi_d * i_q - Psi_q * i_d)[0]

    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default initial state for all batches."""
        if rng is None:
            phys = self.PhysicalState(
                u_d_buffer=0.0,
                u_q_buffer=0.0,
                epsilon=0.0,
                i_d=0.0,
                i_q=0.0,
                torque=0.0,
                omega_el=(env_properties.physical_constraints.omega_el),
            )
            subkey = jnp.nan
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(6,))
            i_d = state_norm[0] * env_properties.physical_constraints.i_d
            i_q = state_norm[1] * env_properties.physical_constraints.i_q
            torque = jax.lax.cond(
                env_properties.saturated,
                self.currents_to_torque_saturated,
                self.currents_to_torque,
                i_d,
                i_q,
                env_properties,
            )
            phys = self.PhysicalState(
                u_d_buffer=0.0,
                u_q_buffer=0.0,
                epsilon=state_norm[4] * env_properties.physical_constraints.epsilon,
                i_d=i_d,
                i_q=i_q,
                torque=torque,
                omega_el=jnp.abs(state_norm[5]) * env_properties.physical_constraints.omega_el,
            )
            key, subkey = jax.random.split(rng)
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(
            u_d_buffer=jnp.nan,
            u_q_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_d=jnp.nan,
            i_q=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)

    # @partial(jax.jit, static_argnums=0)
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0, 0))(
            self.env_properties, rng, jnp.ones(self.batch_size)
        )

    def ode_step(self, state, u_dq, properties):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_dq: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """
        system_state = state.physical_state
        omega_el = system_state.omega_el
        i_d = system_state.i_d
        i_q = system_state.i_q
        eps = system_state.epsilon

        args = (u_dq, properties.static_params)
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
                    (-L_diff_inv * properties.static_params.r_s),
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

        if properties.saturated:
            torque = jnp.array([self.currents_to_torque_saturated(i_d=i_d_k1, i_q=i_q_k1, env_properties=properties)])[
                0
            ]
        else:
            torque = jnp.array([self.currents_to_torque(i_d_k1, i_q_k1, properties)])[0]

        with jdc.copy_and_mutate(system_state, validate=False) as system_state_next:
            system_state_next.epsilon = step_eps(eps, omega_el, self.tau, 1.0)
            system_state_next.i_d = i_d_k1
            system_state_next.i_q = i_q_k1
            system_state_next.torque = torque  # [0]

        with jdc.copy_and_mutate(state, validate=False) as state_next:
            state_next.physical_state = system_state_next
        return state_next

    def constraint_denormalization(self, u_dq, system_state, env_properties):
        """Denormalizes the u_dq and clips it with respect to the hexagon."""
        u_albet_norm = dq2albet(
            u_dq,
            step_eps(
                system_state.physical_state.epsilon,
                self.env_properties.static_params.deadtime + 0.5,
                self.tau,
                system_state.physical_state.omega_el,
            ),
        )
        u_albet_norm_clip = apply_hex_constraint(u_albet_norm)
        u_dq = albet2dq(u_albet_norm_clip, system_state.physical_state.epsilon) * jnp.hstack(
            [env_properties.action_constraints.u_d, env_properties.action_constraints.u_q]
        )
        return u_dq[0]

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def _ode_solver_simulate_ahead(self, init_state, actions, properties, obs_stepsize, action_stepsize):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_dq: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """
        init_state_phys = init_state.physical_state
        omega_el = init_state_phys.omega_el
        i_d = init_state_phys.i_d
        i_q = init_state_phys.i_q
        eps = init_state_phys.epsilon

        args = (actions, properties.static_params)

        def force(t, args):
            actions = args
            return actions[jnp.array(t / action_stepsize, int), 0]

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
                    (-L_diff_inv * properties.static_params.r_s),
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
                eps_diff = omega_el
                d_y = i_d_diff, i_q_diff, eps_diff  # [0]
                return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = action_stepsize * actions.shape[0]
        y0 = tuple([i_d, i_q, eps])
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / obs_stepsize)))  #
        y = diffrax.diffeqsolve(term, self._solver, t0, t1, dt0=obs_stepsize, y0=y0, args=args, saveat=saveat)

        i_d_t = y.ys[0]
        i_q_t = y.ys[1]
        eps_t = y.ys[2]

        if properties.saturated:
            torque_t = jax.vmap(self.currents_to_torque_saturated)(i_d=i_d_t, i_q=i_q_t, env_properties=properties)

        else:
            torque_t = jax.vmap(self.currents_to_torque)(i_d_t, i_q_t, properties)

        phys = self.PhysicalState(
            u_d_buffer=jnp.zeros(9),
            u_q_buffer=jnp.zeros(9),
            epsilon=eps_t,
            i_d=i_d_t,
            i_q=i_q_t,
            torque=torque_t,
            omega_el=jnp.full(9, init_state_phys.omega_el),
        )
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(
            u_d_buffer=jnp.full(9, jnp.nan),
            u_q_buffer=jnp.full(9, jnp.nan),
            epsilon=jnp.full(9, jnp.nan),
            i_d=jnp.full(9, jnp.nan),
            i_q=jnp.full(9, jnp.nan),
            torque=jnp.full(9, jnp.nan),
            omega_el=jnp.full(9, jnp.nan),
        )
        return self.State(physical_state=phys, PRNGKey=init_state.PRNGKey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def sim_ahead(self, init_state, actions, env_properties, obs_stepsize, action_stepsize):
        """Computes multiple JAX-JIT compiled simulation steps for one batch.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[0] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Args:
            init_state: The initial state of the simulation
            actions: A set of actions to be applied to the environment, the value changes every
            action_stepsize (shape=(n_action_steps, action_dim))
            env_properties: The constant properties of the simulation
            obs_stepsize: The sampling time for the observations
            action_stepsize: The time between changes in the input/action
        """
        # TODO epsilon update for constraints
        actions = jax.vmap(self.constraint_denormalization)(actions, init_state, env_properties)

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state, actions, env_properties.static_params, obs_stepsize, action_stepsize
        )

        # generate observations for all timesteps
        observations = jax.vmap(self.generate_observation, in_axes=(0, self.in_axes_env_properties))(
            states, self.env_properties
        )

        # # delete first state because its initial state of simulation and not relevant for terminated
        states_flatten, struct = tree_flatten(states)
        # states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])

        # reward = jax.vmap(self.generate_reward, in_axes=(0, 0, self.in_axes_env_properties))(
        #     states_without_init_state,
        #     jnp.expand_dims(jnp.repeat(actions, int(action_stepsize / obs_stepsize)), 1),
        #     self.env_properties,
        # )
        # # reward = 0

        # # generate truncated
        # truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.in_axes_env_properties))(
        #     states, self.env_properties
        # )

        # # generate terminated

        # # get last state so that the simulation can be continued from the end point
        last_state = tree_unflatten(struct, jnp.array(states_flatten)[:, -1:])

        # terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, self.in_axes_env_properties))(
        #     states_without_init_state, reward, self.env_properties
        # )

        return observations, states, last_state

    def step(self, state, action, env_properties):
        """Computes state by simulating one step taking the deadtime into account.

        Args:
            system_state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """

        action = self.constraint_denormalization(action, state, env_properties)

        action_buffer = jnp.array([state.physical_state.u_d_buffer, state.physical_state.u_q_buffer])

        if env_properties.static_params.deadtime > 0:

            updated_buffer = jnp.array([action[0], action[1]])
            u_dq = action_buffer
        else:
            updated_buffer = action_buffer

            u_dq = action

        next_state = self.ode_step(state, u_dq, env_properties)
        with jdc.copy_and_mutate(next_state, validate=True) as next_state_update:
            next_state_update.physical_state.u_d_buffer = updated_buffer[0]
            next_state_update.physical_state.u_q_buffer = updated_buffer[1]

        observation = self.generate_observation(next_state_update, env_properties)
        return observation, next_state_update

    @property
    def action_description(self):
        return self._action_description

    @property
    def obs_description(self):
        return self._obs_description

    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.vmap_init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.vmap_init_state()"
            state = initial_state
        else:
            state = self.vmap_init_state(rng)

        obs = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties),
        )(state, self.env_properties)

        return obs, state

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
                system_state.physical_state.omega_el / physical_constraints.omega_el,
                system_state.physical_state.torque / physical_constraints.torque,
                cos_eps,
                sin_eps,
                system_state.physical_state.u_d_buffer / physical_constraints.u_d_buffer,
                system_state.physical_state.u_q_buffer / physical_constraints.u_q_buffer,
            )
        )
        for name in self.control_state:
            obs = jnp.hstack((obs, getattr(system_state.reference, name)))
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

    # def generate_reward(self, system_state, action, env_properties):
    #     # """Returns reward for one batch."""
    #     # physical_state = system_state.physical_state
    #     # references = system_state.additions.references
    #     # if env_properties.static_params.physical_properties.control_state == "currents":
    #     #     reward = self.current_reward_func(
    #     #         physical_state.i_d / env_properties.physical_constraints.i_d,
    #     #         physical_state.i_q / env_properties.physical_constraints.i_q,
    #     #         references[0],
    #     #         references[1],
    #     #         0.85,  # gamma
    #     #     )
    #     # elif env_properties.static_params.physical_properties.control_state == "torque":
    #     #     reward = self.torque_reward_func(
    #     #         (physical_state.i_d * (env_properties.physical_constraints.i_d) ** -1),
    #     #         (physical_state.i_q * (env_properties.physical_constraints.i_q) ** -1),
    #     #         (physical_state.torque / (env_properties.physical_constraints.torque)),
    #     #         references,
    #     #         env_properties.static_params.i_lim_multiplier,
    #     #         0.85,  # gamma
    #     #     )

    #     return 0  # reward

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        if "i_d" in self.control_state and "i_q" in self.control_state:
            reward += self.current_reward_func(
                state.physical_state.i_d, state.physical_state.i_q, state.reference.i_d, state.reference.i_q, 0.85
            )
        if "torque" in self.control_state:
            reward += self.torque_reward_func(
                state.physical_state.i_d,
                state.physical_state.i_q,
                state.physical_state.torque,
                state.reference.torque,
                1,
                0.85,
            )
        return jnp.array([reward])

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
