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


def default_paras(name):
    if name == "BRUSA":
        default_physical_constraints = {
            "u_d_buffer": 2 * 400 / 3,
            "u_q_buffer": 2 * 400 / 3,
            "epsilon": jnp.pi,
            "i_d": 250,
            "i_q": 250,
            "omega_el": 3 * 11000 * 2 * jnp.pi / 60,
            "torque": 200,
        }

        default_action_constraints = {
            "u_d": 2 * 400 / 3,
            "u_q": 2 * 400 / 3,
        }
        default_static_params = {
            "p": 3,
            "r_s": 17.932e-3,
            "l_d": 0.37e-3,
            "l_q": 1.2e-3,
            "psi_p": 65.65e-3,
            "deadtime": 1,
        }
        pmsm_lut = loadmat(Path(os.path.dirname(exc_envs.pmsm.__file__)) / Path("LUT_BRUSA_jax_grad.mat"))

    elif name == "SEW":
        default_physical_constraints = {
            "u_d_buffer": 2 * 550 / 3,
            "u_q_buffer": 2 * 550 / 3,
            "epsilon": jnp.pi,
            "i_d": 16,
            "i_q": 16,
            "omega_el": 4 * 2000 / 60 * 2 * jnp.pi,
            "torque": 15,
        }

        default_action_constraints = {
            "u_d": 2 * 550 / 3,
            "u_q": 2 * 550 / 3,
        }
        default_static_params = {
            "p": 4,
            "r_s": 208e-3,
            "l_d": 1.44e-3,
            "l_q": 1.44e-3,
            "psi_p": 122e-3,
            "deadtime": 1,
        }

        pmsm_lut = loadmat(Path(os.path.dirname(exc_envs.pmsm.__file__)) / Path("LUT_SEW_jax_grad.mat"))
    else:
        default_physical_constraints = None
        default_action_constraints = None
        default_static_params = None
        pmsm_lut = None
        raise Exception(f"Motor name {name} is not known.")

    return default_physical_constraints, default_action_constraints, default_static_params, pmsm_lut


class PMSM(CoreEnvironment):
    def __init__(
        self,
        batch_size: int = 8,
        saturated=False,
        LUT_motor_name: str = None,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size (int): Number of parallel environment simulations. Default: 8
            saturated (bool): Permanent magnet flux linkages and inductances are taken from LUT_motor_name specific LUTs. Default: False
            LUT_motor_name (str): Sets physical_constraints, action_constraints and static_params to default values for the passed motor name and stores associated LUTs for the possible saturated case. Needed if saturated==True.
            physical_constraints (dict): Constraints of the physical state of the environment.
                u_d_buffer (float): Direct share of the delayed action due to system deadtime. Default: 2 * 400 / 3
                u_q_buffer (float): Quadrature share of the delayed action due to system deadtime. Default: 2 * 400 / 3
                epsilon (float): Electrical rotation angle. Default: jnp.pi
                i_d (float): Direct share of the current in dq-coordinates. Default: 250
                i_q (float): Quadrature share of the current in dq-coordinates. Default: 250
                omega_el (float): Electrical angular velocity. Default: 3000 / 60 * 2 * jnp.pi
                torque (float): Torque caused by the current. Default: 200
            action_constraints (dict): Constraints of the input/action.
                u_d (float): Direct share of the voltage in dq-coordinates. Default: 2 * 400 / 3
                u_q (float): Quadrature share of the voltage in dq-coordinates. Default: 2 * 400 / 3
            static_params (dict): Parameters of environment which do not change during simulation.
                p (int): Pole pair number. Default: 3
                r_s (float): Stator resistance. Default: 15e-3
                l_d (float): Inductance in direct axes if motor not set to saturated. Default: 0.37e-3
                l_q (float): Inductance in direct if motor not set to saturated. Default: 65.6e-3,
                psi_p (float): Permanent magnet flux linkage if motor not set to saturated. Default: 122e-3,
                deadtime (int): Delay between passed and performed action on the system. Default: 1
            control_state: TODO
            solver (diffrax.solver): Solver used to compute state for next step.
            tau (float): Duration of one control/simulation step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver

        if LUT_motor_name is not None:
            default_physical_constraints, default_action_constraints, default_static_params, self.pmsm_lut = (
                default_paras(LUT_motor_name)
            )

            if saturated:
                saturated_quants = [
                    "L_dd",
                    "L_dq",
                    "L_qd",
                    "L_qq",
                    "Psi_d",
                    "Psi_q",
                ]
                i_d_max = np.max(self.pmsm_lut["i_d_vec"])
                i_q_max = np.max(self.pmsm_lut["i_q_vec"])
                i_d_min = np.min(self.pmsm_lut["i_d_vec"])
                i_q_min = np.min(self.pmsm_lut["i_q_vec"])
                i_d_stepsize = (i_d_max - i_d_min) / (self.pmsm_lut["i_d_vec"].shape[1] - 1)
                i_q_stepsize = (i_q_max - i_q_min) / (self.pmsm_lut["i_q_vec"].shape[1] - 1)
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

                    # repeat each border values to have the linear extrapolation create constant extrapolation
                    a = np.vstack([qmap[0, :], qmap, qmap[-1, :]])
                    b = np.hstack([a[:, :1], a, a[:, -1:]])

                    self.pmsm_lut[q] = b

                n_grid_points_y, n_grid_points_x = self.pmsm_lut[saturated_quants[0]].shape
                x, y = np.linspace(i_d_min - i_d_stepsize, i_d_max + i_d_stepsize, n_grid_points_x), np.linspace(
                    i_q_min - i_q_stepsize, i_q_max + i_q_stepsize, n_grid_points_y
                )
                self.LUT_interpolators = {
                    q: jax.scipy.interpolate.RegularGridInterpolator(
                        (x, y), self.pmsm_lut[q][:, :].T, method="linear", bounds_error=False, fill_value=None
                    )
                    for q in saturated_quants
                }
                default_static_params["l_d"] = jnp.nan
                default_static_params["l_q"] = jnp.nan
                default_static_params["psi_p"] = jnp.nan

            else:
                saturated_quants = [
                    "L_dd",
                    "L_dq",
                    "L_qd",
                    "L_qq",
                    "Psi_d",
                    "Psi_q",
                ]
                self.LUT_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        else:
            if saturated:
                raise Exception("LUT_motor_name is needed to load LUTs.")

            default_physical_constraints = {
                "u_d_buffer": 2 * 400 / 3,
                "u_q_buffer": 2 * 400 / 3,
                "epsilon": jnp.pi,
                "i_d": 250,
                "i_q": 250,
                "omega_el": 3000 / 60 * 2 * jnp.pi,
                "torque": 200,
            }

            default_action_constraints = {
                "u_d": 2 * 400 / 3,
                "u_q": 2 * 400 / 3,
            }
            default_static_params = {
                "p": 3,
                "r_s": 15e-3,
                "l_d": 0.37e-3,
                "l_q": 1.2e-3,
                "psi_p": 65.6e-3,
                "deadtime": 1,
            }
            saturated_quants = [
                "L_dd",
                "L_dq",
                "L_qd",
                "L_qq",
                "Psi_d",
                "Psi_q",
            ]
            self.LUT_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        if not static_params:
            static_params = default_static_params

        if not physical_constraints:
            physical_constraints = default_physical_constraints
        else:
            i_d_constr = physical_constraints["i_d"]
            i_q_constr = physical_constraints["i_q"]
            def_i_d_constr = default_physical_constraints["i_d"]
            def_i_q_constr = default_physical_constraints["i_q"]

            if i_d_constr > def_i_d_constr:
                print(
                    f"The defined permitted range of i_d (+-{i_d_constr}) exceeds the limits of the LUT (+-{def_i_d_constr}). Values outside this range are extrapolated."
                )
            if i_q_constr > def_i_q_constr:
                print(
                    f"The defined permitted range of i_q (+-{i_q_constr}) exceeds the limits of the LUT (+-{def_i_q_constr}). Values outside this range are extrapolated."
                )

        if not action_constraints:
            action_constraints = default_action_constraints

        if not control_state:
            control_state = []

        self.control_state = control_state

        static_params = self.StaticParams(**static_params)
        physical_constraints = self.PhysicalState(**physical_constraints)
        action_constraints = self.Action(**action_constraints)

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
        """Dataclass containing the action, that can be applied to the environment."""

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
                i_d=-env_properties.physical_constraints.i_d / 2,  # 0.0,
                i_q=0.0,
                torque=0.0,
                omega_el=(env_properties.physical_constraints.omega_el) / 2,
            )

            rng = jnp.nan
        else:
            rng, subkey = jax.random.split(rng)
            state_norm = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(2,))
            rng, subkey = jax.random.split(rng)
            i_dq_norm = jax.random.ball(subkey, 2)
            i_d = -jnp.abs(i_dq_norm[0] * env_properties.physical_constraints.i_d)
            i_q = i_dq_norm[1] * env_properties.physical_constraints.i_q
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
                epsilon=state_norm[0] * env_properties.physical_constraints.epsilon,
                i_d=i_d,
                i_q=i_q,
                torque=torque,
                omega_el=jnp.abs(state_norm[1]) * env_properties.physical_constraints.omega_el,
            )
        additions = None
        ref = self.PhysicalState(
            u_d_buffer=jnp.nan,
            u_q_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_d=jnp.nan,
            i_q=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        return self.State(physical_state=phys, PRNGKey=rng, additions=additions, reference=ref)

    def vmap_init_state(self, rng: chex.PRNGKey = None):
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0, 0))(
            self.env_properties, rng, jnp.ones(self.batch_size)
        )

    @partial(jax.jit, static_argnums=[0, 3])
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
                d_y = i_d_diff, i_q_diff
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

    @partial(jax.jit, static_argnums=[0, 3, 4, 5])
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
            return actions[jnp.array(t / action_stepsize, int)]

        if properties.saturated:

            def vector_field(t, y, args):
                actions, params = args
                i_d, i_q, eps = y
                u_dq = force(t, actions)
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
                eps_diff = omega_el
                d_y = i_dq_diff[0], i_dq_diff[1], eps_diff
                return d_y

        else:

            def vector_field(t, y, args):
                i_d, i_q, eps = y
                actions, params = args
                u_dq = force(t, actions)
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
        obs_len = i_d_t.shape[0]

        if properties.saturated:
            torque_t = jax.vmap(self.currents_to_torque_saturated, in_axes=(0, 0, None))(i_d_t, i_q_t, properties)

        else:
            torque_t = jax.vmap(self.currents_to_torque, in_axes=(0, 0, None))(i_d_t, i_q_t, properties)

        phys = self.PhysicalState(
            u_d_buffer=jnp.zeros(obs_len),
            u_q_buffer=jnp.zeros(obs_len),
            epsilon=eps_t,
            i_d=i_d_t,
            i_q=i_q_t,
            torque=torque_t,
            omega_el=jnp.full(obs_len, init_state_phys.omega_el),
        )
        additions = None
        ref = self.PhysicalState(
            u_d_buffer=jnp.full(obs_len, jnp.nan),
            u_q_buffer=jnp.full(obs_len, jnp.nan),
            epsilon=jnp.full(obs_len, jnp.nan),
            i_d=jnp.full(obs_len, jnp.nan),
            i_q=jnp.full(obs_len, jnp.nan),
            torque=jnp.full(obs_len, jnp.nan),
            omega_el=jnp.full(obs_len, jnp.nan),
        )
        return self.State(
            physical_state=phys, PRNGKey=jnp.full(obs_len, init_state.PRNGKey), additions=additions, reference=ref
        )

    def constraint_denormalization_ahead(self, actions, init_state, env_properties):
        act_len = actions.shape[0]
        phys = self.PhysicalState(
            u_d_buffer=jnp.full(act_len, init_state.physical_state.u_d_buffer),
            u_q_buffer=jnp.full(act_len, init_state.physical_state.u_q_buffer),
            epsilon=jnp.full(act_len, init_state.physical_state.epsilon)
            + jnp.linspace(0, self.tau * (act_len - 1), act_len) * init_state.physical_state.omega_el,
            i_d=jnp.full(act_len, init_state.physical_state.i_d),
            i_q=jnp.full(act_len, init_state.physical_state.i_q),
            torque=jnp.full(act_len, init_state.physical_state.torque),
            omega_el=jnp.full(act_len, init_state.physical_state.omega_el),
        )
        additions = None
        ref = self.PhysicalState(
            u_d_buffer=jnp.full(act_len, init_state.reference.u_d_buffer),
            u_q_buffer=jnp.full(act_len, init_state.reference.u_q_buffer),
            epsilon=jnp.full(act_len, init_state.reference.epsilon),
            i_d=jnp.full(act_len, init_state.reference.i_d),
            i_q=jnp.full(act_len, init_state.reference.i_q),
            torque=jnp.full(act_len, init_state.reference.torque),
            omega_el=jnp.full(act_len, init_state.reference.omega_el),
        )
        states = self.State(
            physical_state=phys, PRNGKey=jnp.full(act_len, init_state.PRNGKey), additions=additions, reference=ref
        )
        actions = jax.vmap(self.constraint_denormalization, in_axes=(0, 0, None))(actions, states, env_properties)
        return actions

    @partial(jax.jit, static_argnums=[0, 3, 4, 5])
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

        actions = self.constraint_denormalization_ahead(actions, init_state, env_properties)

        deadtime = env_properties.static_params.deadtime
        acts_buf = jnp.repeat(
            jnp.array([init_state.physical_state.u_d_buffer, init_state.physical_state.u_q_buffer])[None, :],
            deadtime,
            axis=0,
        )

        actions_dead = jnp.vstack([acts_buf, actions[: (actions.shape[0] - deadtime), :]])

        single_state_struct = tree_structure(init_state)

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state, actions_dead, env_properties, obs_stepsize, action_stepsize
        )

        with jdc.copy_and_mutate(states, validate=False) as states:
            acts_m = jnp.vstack([acts_buf, actions])
            acts_m = acts_m.repeat(int(obs_stepsize / action_stepsize), axis=0)
            if deadtime == 0:
                acts_m = jnp.zeros(((actions.shape[0] + 1), 2))
            states.physical_state.u_d_buffer = acts_m[:, 0]
            states.physical_state.u_q_buffer = acts_m[:, 1]

        # generate observations for all timesteps
        observations = jax.vmap(self.generate_observation, in_axes=(0, None))(states, env_properties)

        states_flatten, _ = tree_flatten(states)

        # get last state so that the simulation can be continued from the end point
        last_state = tree_unflatten(single_state_struct, jnp.array(states_flatten)[:, -1])

        return observations, states, last_state

    def generate_rew_trunc_term_ahead(self, states, actions, env_properties):
        """Computes reward, truncated and terminated for sim_ahead simulation for one batch."""
        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"
        deadtime = env_properties.static_params.deadtime

        states_flatten, struct = tree_flatten(states)
        states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])
        states_without_last_state = tree_unflatten(struct, jnp.array(states_flatten)[:, :-1])

        actions = jax.vmap(self.constraint_denormalization, in_axes=(0, 0, None))(
            actions, states_without_last_state, env_properties
        )

        deadtime = env_properties.static_params.deadtime
        acts_buf = jnp.repeat(
            jnp.array([states.physical_state.u_d_buffer[0], states.physical_state.u_q_buffer[0]])[None, :],
            deadtime,
            axis=0,
        )

        actions_dead = jnp.vstack([acts_buf, actions[: (actions.shape[0] - deadtime), :]])

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, None))(
            states_without_init_state,
            jnp.expand_dims(
                jnp.repeat(actions_dead, int((jnp.array(states_flatten).shape[1] - 1) / actions_dead.shape[0]), axis=0),
                1,
            ),
            env_properties,
        )
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, None))(states, env_properties)
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, None))(
            states_without_init_state, reward, env_properties
        )
        return reward, truncated, terminated

    @partial(jax.jit, static_argnums=[0, 3])
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

    def reset(
        self, env_properties, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None, vmap_helper=None
    ):
        """Resets one batch to default, random or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.init_state(env_properties)) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as init_state()"
            state = initial_state
        else:
            state = self.init_state(env_properties, rng)

        obs = self.generate_observation(state, env_properties)

        return obs, state

    def generate_observation(self, system_state, env_properties):
        """Returns observation for one batch."""
        physical_constraints = env_properties.physical_constraints
        eps = system_state.physical_state.epsilon
        cos_eps = jnp.cos(eps)
        sin_eps = jnp.sin(eps)
        obs = jnp.hstack(
            (
                (system_state.physical_state.i_d + (physical_constraints.i_d * 0.5))
                / (physical_constraints.i_d * 0.5),  #   system_state.physical_state.i_d/ physical_constraints.i_d,
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

    @partial(jax.jit, static_argnums=0)
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        physical_constraints = env_properties.physical_constraints
        phys = self.PhysicalState(
            u_d_buffer=obs[6] * (physical_constraints.u_d_buffer).astype(float),
            u_q_buffer=obs[7] * (physical_constraints.u_q_buffer).astype(float),
            epsilon=jnp.arctan2(obs[5], obs[4]),
            i_d=obs[0] * ((physical_constraints.i_d).astype(float) * 0.5)
            - ((physical_constraints.i_d).astype(float) * 0.5),
            i_q=obs[1] * (physical_constraints.i_q).astype(float),
            torque=obs[3] * (physical_constraints.torque).astype(float),
            omega_el=obs[2] * (env_properties.physical_constraints.omega_el).astype(float),
        )
        additions = None
        ref = self.PhysicalState(
            u_d_buffer=jnp.nan,
            u_q_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_d=jnp.nan,
            i_q=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        with jdc.copy_and_mutate(ref, validate=False) as new_ref:
            for name, pos in zip(self.control_state, range(len(self.control_state))):
                setattr(new_ref, name, obs[2 + pos] * (getattr(physical_constraints, name)).astype(float))
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)

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
