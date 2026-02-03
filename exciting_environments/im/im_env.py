from functools import partial
from typing import Callable
from types import MethodType
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, tree_map
import jax_dataclasses as jdc
import chex
import diffrax
from scipy.interpolate import griddata
from jax.scipy.linalg import expm
from dataclasses import fields

from copy import deepcopy

from exciting_environments import CoreEnvironment
from exciting_environments.im import MotorVariant

import numpy as np

# only for alpha/beta -> abc
t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])
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
    """Compute the transformation matrix for converting between DQ and Alpha-Beta reference frames."""
    cos = jnp.cos(eps)
    sin = jnp.sin(eps)
    return jnp.column_stack((cos, sin, -sin, cos)).reshape(2, 2)


def dq2abc(u_dq, eps):
    """Transform voltages from DQ coordinates to ABC (three-phase) coordinates."""
    u_abc = t32 @ dq2albet(u_dq, eps).T
    return u_abc.T


def dq2albet(u_dq, eps):
    """Transform voltages from DQ coordinates to Alpha-Beta coordinates."""
    q = t_dq_alpha_beta(-eps)
    u_alpha_beta = q @ u_dq.T

    return u_alpha_beta.T


def albet2dq(u_albet, eps):
    """Transform voltages from Alpha-Beta coordinates to DQ coordinates."""
    q_inv = t_dq_alpha_beta(eps)
    u_dq = q_inv @ u_albet.T

    return u_dq.T


def abc2dq(u_abc, eps):
    """Transform voltages from ABC (three-phase) coordinates to DQ coordinates."""
    u_alpha_beta = t23 @ u_abc.T
    u_dq = albet2dq(u_alpha_beta.T, eps)
    return u_dq


def step_eps(eps, omega_el, tau, tau_scale=1.0):
    """Update the electrical angle over a time step with optional scaling."""
    eps += omega_el * tau * tau_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def apply_hex_constraint(u_albet):
    """Clip voltages in alpha/beta coordinates into the voltage hexagon."""
    u_albet_c = u_albet[0] + 1j * u_albet[1]
    idx = (jnp.sin(jnp.angle(u_albet_c)[..., jnp.newaxis] - 2 / 3 * jnp.pi * jnp.arange(3)) >= 0).astype(int)
    rot_vec = ROTATION_MAP[idx[0], idx[1], idx[2]]
    # rotate sectors upwards
    u_albet_c = jnp.multiply(u_albet_c, rot_vec)
    u_albet_c = jnp.clip(u_albet_c.real, -2 / 3, 2 / 3) + 1j * u_albet_c.imag
    u_albet_c = u_albet_c.real + 1j * jnp.clip(u_albet_c.imag, 0, 2 / 3 * jnp.sqrt(3))
    u_albet_c = jnp.multiply(u_albet_c, jnp.conjugate(rot_vec))  # rotate back
    return jnp.column_stack([u_albet_c.real, u_albet_c.imag])


# def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, tau):
#     """Clip voltages in ABC (three-phase) coordinates and transform back to DQ coordinates."""
#     eps_advanced = step_eps(eps, omega_el, tau, 0.5)
#     u_abc = dq2abc(u_dq, eps_advanced)
#     # clip in abc coordinates
#     u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
#     u_dq = abc2dq(u_abc, eps)
#     return u_dq


class IM(CoreEnvironment):
    def __init__(
        self,
        batch_size: int = 8,
        saturated=False,
        motor_variant: MotorVariant = MotorVariant.DEFAULT,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Heun(),
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size (int): Number of parallel environment simulations. Default: 8
            saturated (bool): No saturated case implemented yet. Default: False
            motor_variant (MotorVariant): Sets physical_normalizations, action_normalizations, soft_constraints and static_params to default values for the passed motor variant and stores associated LUTs for the possible saturated case. Needed if saturated==True. Default: MotorVariant.DEFAULT
            physical_normalizations (dict): min-max normalization values of the physical state of the environment.
                u_alpha_buffer (MinMaxNormalization): Voltage in alpha axis of the delayed action due to system deadtime. Default: min=-2 * 560 / 3, max=2 * 560 / 3
                u_beta_buffer (MinMaxNormalization): Voltage in beta axis of the delayed action due to system deadtime. Default: min=-2 * 560 / 3, max=2 * 560 / 3
                epsilon (MinMaxNormalization): Electrical rotor rotation angle. Default: min=-jnp.pi, max=jnp.pi
                i_s_alpha(MinMaxNormalization): Stator current in alpha axis. Default: min=-20, max=20
                i_s_beta (MinMaxNormalization): Stator current in beta axis. Default: min=-20, max=20
                omega_el (MinMaxNormalization): Electrical rotor angular velocity. Default: min=0, max=2 * 5000 * 2 * jnp.pi / 60
                torque (MinMaxNormalization): Torque caused by the current. Default: min=-20, max=20
            action_normalizations (dict): min-max normalization values of the input/action.
                u_alpha(MinMaxNormalization): Voltage in alpha axis. Default: min=-2 * 560 / 3, max=2 * 560 / 3
                u_beta (MinMaxNormalization): Voltage in beta axis. Default: min=-2 * 560 / 3, max=2 * 560 / 3
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params (dict): Parameters of environment which do not change during simulation.
                p (int): Pole pair number. Default: 2
                r_s (float): Stator resistance. Default: 2.9338
                r_r (float): Rotor resistance. Default: 1.355
                l_m (float): Main inductance. Default: 143.75e-3
                l_sigs (float): Stator-side stray inductance. Default: 5.87e-3
                l_sigr (float): Rotor-side stray inductance. Default: 5.87e-3
                u_dc (float): DC link voltage. Default: 560
                deadtime (int): Delay between passed and performed action on the system. Default: 0
            control_state: Components of the physical state that are considered in reference tracking.
            solver (diffrax.solver): Solver used to compute state for next step.
            tau (float): Duration of one control/simulation step in seconds. Default: 1e-4.

        Note: Attributes of MinMaxNormalization of physical_normalizations and action_normalizations as well as static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver

        if motor_variant != MotorVariant.DEFAULT:
            motor_params = motor_variant.get_params()
            default_physical_normalizations = motor_params.physical_normalizations.__dict__
            default_action_normalizations = motor_params.action_normalizations.__dict__
            default_static_params = motor_params.static_params.__dict__
            default_soft_constraints = MethodType(motor_params.default_soft_constraints, self)
            LUT_predefined = motor_params.lut
            if saturated:
                raise NotImplementedError("Saturation case not implemented")

            else:
                saturated_quants = [
                    "l_m",
                    "l_sigs",
                    "l_sigr",
                ]
                self.LUT_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        else:
            if saturated:
                raise NotImplementedError("Saturation case not implemented")

            saturated_quants = [
                "l_m",
                "l_sigs",
                "l_sigr",
            ]

            motor_params = motor_variant.get_params()
            default_physical_normalizations = motor_params.physical_normalizations.__dict__
            default_action_normalizations = motor_params.action_normalizations.__dict__
            default_static_params = motor_params.static_params.__dict__
            default_soft_constraints = MethodType(motor_params.default_soft_constraints, self)
            LUT_predefined = motor_params.lut
            self.LUT = LUT_predefined
            self.LUT_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        if not static_params:
            static_params = default_static_params

        if not physical_normalizations:
            physical_normalizations = default_physical_normalizations
        else:
            i_s_alpha_lims = physical_normalizations["i_s_alpha"]
            i_s_beta_lims = physical_normalizations["i_s_beta"]
            def_i_s_alpha_lims = default_physical_normalizations["i_s_alpha"]
            def_i_s_beta_lims = default_physical_normalizations["i_s_beta"]

            if (i_s_alpha_lims.min < def_i_s_alpha_lims.min) or (i_s_alpha_lims.max > def_i_s_alpha_lims.max):
                print(
                    f"The defined permitted range of i_s_alpha ({i_s_alpha_lims}) exceeds the limits set in the motor parameters corresponding to motor_variant."
                )
            if (i_s_beta_lims.min < def_i_s_beta_lims.min) or (i_s_beta_lims.max > def_i_s_beta_lims.max):
                print(
                    f"The defined permitted range of i_s_beta ({i_s_beta_lims}) exceeds the limits set in the motor parameters corresponding to motor_variant."
                )

        if not action_normalizations:
            action_normalizations = default_action_normalizations

        if not control_state:
            control_state = []

        if not soft_constraints:
            soft_constraints = default_soft_constraints

        self.control_state = control_state
        self.soft_constraints = soft_constraints

        static_params = self.StaticParams(**static_params)
        physical_normalizations = self.PhysicalState(**physical_normalizations)
        action_normalizations = self.Action(**action_normalizations)

        env_properties = self.EnvProperties(
            saturated=saturated,
            physical_normalizations=physical_normalizations,
            action_normalizations=action_normalizations,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

        self._action_description = ["u_alpha", "u_beta"]
        self._obs_description = [
            "i_s_alpha",
            "i_s_beta",
            "psi_r_alpha",
            "psi_r_beta",
            "omega_el",
            "torque",
            "cos_eps",
            "sin_eps",
            "u_alpha_buffer",
            "u_beta_buffer",
        ]

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the physical parameters of the environment."""

        p: jax.Array
        r_s: jax.Array
        r_r: jax.Array
        l_m: jax.Array
        l_sigs: jax.Array
        l_sigr: jax.Array
        u_dc: jax.Array
        deadtime: jax.Array

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        u_alpha_buffer: jax.Array
        u_beta_buffer: jax.Array
        i_s_alpha: jax.Array
        i_s_beta: jax.Array
        psi_r_alpha: jax.Array
        psi_r_beta: jax.Array
        epsilon: jax.Array
        omega_el: jax.Array
        torque: jax.Array

    @jdc.pytree_dataclass
    class Additions:
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    @jdc.pytree_dataclass
    class Action:
        """Dataclass containing the action, that can be applied to the environment."""

        u_alpha: jax.Array
        u_beta: jax.Array

    @jdc.pytree_dataclass
    class EnvProperties:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        saturated: jax.Array
        physical_normalizations: jdc.pytree_dataclass
        action_normalizations: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def currents_to_torque(self, i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, env_properties):
        torque = (
            1.5
            * env_properties.static_params.p
            * env_properties.static_params.l_m
            / (env_properties.static_params.l_m + env_properties.static_params.l_sigr)
            * (psi_r_alpha * i_s_beta - psi_r_beta * i_s_alpha)
        )
        return torque

    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default initial state for all batches."""
        if rng is None:
            phys = self.PhysicalState(
                u_alpha_buffer=0.0,
                u_beta_buffer=0.0,
                epsilon=0.0,
                i_s_alpha=0.0,
                i_s_beta=0.0,
                psi_r_alpha=0.0,
                psi_r_beta=0.0,
                torque=0.0,
                omega_el=(
                    env_properties.physical_normalizations.omega_el.min
                    + env_properties.physical_normalizations.omega_el.max
                )
                / 2,
            )

            rng = jnp.nan
        else:
            rng, subkey = jax.random.split(rng)
            state_norm = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(2,))
            rng, subkey = jax.random.split(rng)
            i_s_alpha_beta = jax.random.ball(subkey, 2) * env_properties.physical_normalizations.i_s_alpha.max
            psi_r_alpha_beta = jax.random.ball(subkey, 2) * env_properties.physical_normalizations.psi_r_alpha.max

            torque = self.currents_to_torque(
                i_s_alpha_beta[0], i_s_alpha_beta[1], psi_r_alpha_beta[0], psi_r_alpha_beta[1], env_properties
            )

            phys = self.PhysicalState(
                u_alpha_buffer=0.0,
                u_beta_buffer=0.0,
                epsilon=(state_norm[0] + 1)
                / 2
                * (
                    env_properties.physical_normalizations.epsilon.max
                    - env_properties.physical_normalizations.epsilon.min
                )
                + env_properties.physical_normalizations.epsilon.min,
                i_s_alpha=i_s_alpha_beta[0],
                i_s_beta=i_s_alpha_beta[1],
                psi_r_alpha=psi_r_alpha_beta[0],
                psi_r_beta=psi_r_alpha_beta[1],
                torque=torque,
                omega_el=(state_norm[1] + 1)
                / 2
                * (
                    env_properties.physical_normalizations.omega_el.max
                    - env_properties.physical_normalizations.omega_el.min
                )
                + env_properties.physical_normalizations.omega_el.min,
            )

        def voltage(t):
            return jnp.array([0, 0])

        args = (env_properties.static_params, phys.omega_el)
        if env_properties.saturated:
            raise NotImplementedError("Saturated case not implemented yet.")
            vector_field = partial(self.saturated_ode, action=voltage)
        else:
            vector_field = partial(self.ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.i_s_alpha, phys.i_s_beta, phys.psi_r_alpha, phys.psi_r_beta, phys.epsilon])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = tree_map(lambda x: jnp.zeros_like(x, dtype=x.dtype), solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(
            u_alpha_buffer=jnp.nan,
            u_beta_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_s_alpha=jnp.nan,
            i_s_beta=jnp.nan,
            psi_r_alpha=jnp.nan,
            psi_r_beta=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        return self.State(physical_state=phys, PRNGKey=rng, additions=additions, reference=ref)

    def saturated_ode(self, t, y, args, action):
        raise NotImplementedError("")
        return

    def ode(self, t, y, args, action):
        i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, eps = y
        params, omega_el = args
        r_s = params.r_s
        r_r = params.r_r
        l_m = params.l_m
        l_r = params.l_sigr + l_m
        l_s = params.l_sigs + l_m
        sigma = (l_s * l_r - l_m**2) / (l_s * l_r)
        tau_r = l_r / r_r
        tau_sig = sigma * l_s / (r_s + r_r * (l_m**2) / (l_r**2))
        u_alpha_beta = action(t)
        u_alpha = u_alpha_beta[0]
        u_beta = u_alpha_beta[1]

        i_s_alpha_diff = (
            (-1 / tau_sig) * i_s_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_alpha
            + (l_m * omega_el / (sigma * l_r * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_alpha
        )
        i_s_beta_diff = (
            (-1 / tau_sig) * i_s_beta
            + (-l_m * omega_el / (sigma * l_r * l_s)) * psi_r_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_beta
        )
        psi_r_alpha_diff = (l_m / tau_r) * i_s_alpha + (-1 / tau_r) * psi_r_alpha + (-omega_el) * psi_r_beta

        psi_r_beta_diff = (l_m / tau_r) * i_s_beta + (omega_el) * psi_r_alpha + (-1 / tau_r) * psi_r_beta

        eps_diff = omega_el
        d_y = i_s_alpha_diff, i_s_beta_diff, psi_r_alpha_diff, psi_r_beta_diff, eps_diff
        return d_y

    @partial(jax.jit, static_argnums=[0, 3])
    def _ode_solver_step(self, state, u_alpha_beta, properties):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_alpha_beta: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """
        system_state = state.physical_state
        omega_el = system_state.omega_el
        i_s_alpha = system_state.i_s_alpha
        i_s_beta = system_state.i_s_beta
        psi_r_alpha = system_state.psi_r_alpha
        psi_r_beta = system_state.psi_r_beta
        eps = system_state.epsilon

        def voltage(t):
            return u_alpha_beta

        args = (properties.static_params, omega_el)
        if properties.saturated:
            raise NotImplementedError("Saturated case not implemented yet.")
            vector_field = partial(self.saturated_ode, action=voltage)
        else:
            vector_field = partial(self.ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, eps])

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)

        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        i_s_alpha_k1 = y[0]
        i_s_beta_k1 = y[1]
        psi_r_alpha_k1 = y[2]
        psi_r_beta_k1 = y[3]
        eps_k1 = y[4]

        eps_k1 = ((eps_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        if properties.saturated:
            raise NotImplementedError("Saturated case not implemented yet.")
            torque = jnp.array(
                [
                    self.currents_to_torque_saturated(
                        i_s_alpha=i_s_alpha_k1, i_s_beta=i_s_beta_k1, env_properties=properties
                    )
                ]
            )[0]
        else:
            torque = jnp.array(
                [self.currents_to_torque(i_s_alpha_k1, i_s_beta_k1, psi_r_alpha_k1, psi_r_beta_k1, properties)]
            )[0]

        with jdc.copy_and_mutate(system_state, validate=True) as system_state_next:
            system_state_next.epsilon = eps_k1
            system_state_next.i_s_alpha = i_s_alpha_k1
            system_state_next.i_s_beta = i_s_beta_k1
            system_state_next.psi_r_alpha = psi_r_alpha_k1
            system_state_next.psi_r_beta = psi_r_beta_k1
            system_state_next.torque = torque

        with jdc.copy_and_mutate(state, validate=True) as new_state:
            new_state.physical_state = system_state_next

        new_state = jdc.replace(
            new_state, additions=self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        )
        return new_state

    def constraint_denormalization(self, u_alpha_beta_norm, system_state, env_properties):
        """Denormalizes the u_alpha_beta and clips it with respect to the hexagon."""
        u_alpha_beta = self.denormalize_action(u_alpha_beta_norm, env_properties)
        # normalize to u_dc/2 for hexagon constraints
        u_alpha_beta_norm = u_alpha_beta * (1 / (env_properties.static_params.u_dc / 2))
        u_albet_norm_clip = apply_hex_constraint(u_alpha_beta_norm)
        u_alpha_beta = u_albet_norm_clip[0] * (env_properties.static_params.u_dc / 2)
        return u_alpha_beta

    @partial(jax.jit, static_argnums=[0, 3, 4, 5])
    def _ode_solver_simulate_ahead(self, init_state, actions, properties, obs_stepsize, action_stepsize):
        """Computes multiple simulation steps.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_alpha_beta: The action to apply to the environment.
            properties: Parameters and settings of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """
        init_state_phys = init_state.physical_state
        omega_el = init_state_phys.omega_el
        i_s_alpha = init_state_phys.i_s_alpha
        i_s_beta = init_state_phys.i_s_beta
        psi_r_alpha = init_state_phys.psi_r_alpha
        psi_r_beta = init_state_phys.psi_r_beta
        eps = init_state_phys.epsilon

        def voltage(t):
            return actions[jnp.array(t / action_stepsize, int)]

        args = (properties.static_params, omega_el)
        if properties.saturated:
            raise NotImplementedError("")
            vector_field = partial(self.saturated_ode, action=voltage)
        else:
            vector_field = partial(self.ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = action_stepsize * actions.shape[0]
        y0 = tuple([i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, eps])
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / obs_stepsize)))

        controller = diffrax.ConstantStepSize()

        y = diffrax.diffeqsolve(
            term,
            self._solver,
            t0,
            t1,
            dt0=obs_stepsize,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=controller,
        )

        i_s_alpha_t = y.ys[0]
        i_s_beta_t = y.ys[1]
        psi_r_alpha_t = y.ys[2]
        psi_r_beta_t = y.ys[3]
        eps_t = y.ys[4]
        # keep eps between -pi and pi
        eps_t = ((eps_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        obs_len = i_s_alpha_t.shape[0]

        if properties.saturated:
            raise NotImplementedError("")
            torque_t = jax.vmap(self.currents_to_torque_saturated, in_axes=(0, 0, 0, 0, None))(
                i_s_alpha_t, i_s_beta_t, psi_r_alpha_t, psi_r_beta_t, properties
            )

        else:
            torque_t = jax.vmap(self.currents_to_torque, in_axes=(0, 0, 0, 0, None))(
                i_s_alpha_t, i_s_beta_t, psi_r_alpha_t, psi_r_beta_t, properties
            )

        phys = self.PhysicalState(
            u_alpha_buffer=jnp.zeros(obs_len),
            u_beta_buffer=jnp.zeros(obs_len),
            epsilon=eps_t,
            i_s_alpha=i_s_alpha_t,
            i_s_beta=i_s_beta_t,
            psi_r_alpha=psi_r_alpha_t,
            psi_r_beta=psi_r_beta_t,
            torque=torque_t,
            omega_el=jnp.full(obs_len, init_state_phys.omega_el),
        )

        y0 = tuple([i_s_alpha_t[-1], i_s_beta_t[-1], psi_r_alpha_t[-1], psi_r_beta_t[-1], eps_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len), active_solver_state=jnp.full(obs_len, True)
        )
        ref = self.PhysicalState(
            u_alpha_buffer=jnp.full(obs_len, jnp.nan),
            u_beta_buffer=jnp.full(obs_len, jnp.nan),
            epsilon=jnp.full(obs_len, jnp.nan),
            i_s_alpha=jnp.full(obs_len, jnp.nan),
            i_s_beta=jnp.full(obs_len, jnp.nan),
            psi_r_alpha=jnp.full(obs_len, jnp.nan),
            psi_r_beta=jnp.full(obs_len, jnp.nan),
            torque=jnp.full(obs_len, jnp.nan),
            omega_el=jnp.full(obs_len, jnp.nan),
        )
        return self.State(
            physical_state=phys,
            PRNGKey=jnp.full(obs_len, init_state.PRNGKey),
            additions=additions,
            reference=ref,
        )

    def constraint_denormalization_ahead(self, actions, init_state, env_properties):
        act_len = actions.shape[0]
        with jdc.copy_and_mutate(init_state, validate=False) as states:
            for field in fields(states.physical_state):
                name = field.name
                setattr(
                    states.physical_state,
                    name,
                    self.repeat_values(getattr(states.physical_state, name), act_len),
                )
            states.physical_state.epsilon = (
                states.physical_state.epsilon
                + jnp.linspace(0, self.tau * (act_len - 1), act_len) * init_state.physical_state.omega_el
            )

            # extend state dimension to use vmapping across time
            for field in fields(states.reference):
                name = field.name
                setattr(
                    states.reference,
                    name,
                    self.repeat_values(getattr(states.reference, name), act_len),
                )

            for field in fields(states.additions):
                name = field.name
                setattr(
                    states.additions,
                    name,
                    self.repeat_values(getattr(states.additions, name), act_len),
                )

            states.PRNGKey = jnp.full(act_len, init_state.PRNGKey)

        actions = jax.vmap(self.constraint_denormalization, in_axes=(0, 0, None))(actions, states, env_properties)
        return actions

    @partial(jax.jit, static_argnums=[0, 3, 4, 5])
    def sim_ahead(self, init_state, actions, env_properties, obs_stepsize, action_stepsize):
        """Computes multiple JAX-JIT compiled simulation steps for one batch.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[0] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every
            action_stepsize (shape=(n_action_steps, action_dim)).
            env_properties: The constant properties of the simulation.
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.
        """

        actions = self.constraint_denormalization_ahead(actions, init_state, env_properties)

        deadtime = env_properties.static_params.deadtime
        acts_buf = jnp.repeat(
            jnp.array(
                [
                    init_state.physical_state.u_alpha_buffer,
                    init_state.physical_state.u_beta_buffer,
                ]
            )[None, :],
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
            states.physical_state.u_alpha_buffer = acts_m[:, 0]
            states.physical_state.u_beta_buffer = acts_m[:, 1]

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
            jnp.array(
                [
                    states.physical_state.u_alpha_buffer[0],
                    states.physical_state.u_beta_buffer[0],
                ]
            )[None, :],
            deadtime,
            axis=0,
        )

        actions_dead = jnp.vstack([acts_buf, actions[: (actions.shape[0] - deadtime), :]])

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, None))(
            states_without_init_state,
            jnp.expand_dims(
                jnp.repeat(
                    actions_dead,
                    int((jnp.array(states_flatten).shape[1] - 1) / actions_dead.shape[0]),
                    axis=0,
                ),
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

        action_buffer = jnp.array([state.physical_state.u_alpha_buffer, state.physical_state.u_beta_buffer])

        if env_properties.static_params.deadtime > 0:

            updated_buffer = jnp.array([action[0], action[1]])
            u_alpha_beta = action_buffer
        else:
            updated_buffer = action_buffer

            u_alpha_beta = action

        next_state = self._ode_solver_step(state, u_alpha_beta, env_properties)

        with jdc.copy_and_mutate(next_state, validate=True) as next_state_update:
            next_state_update.physical_state.u_alpha_buffer = updated_buffer[0]
            next_state_update.physical_state.u_beta_buffer = updated_buffer[1]

        observation = self.generate_observation(next_state_update, env_properties)
        return observation, next_state_update

    @property
    def action_description(self):
        return self._action_description

    @property
    def obs_description(self):
        return np.hstack(
            [
                np.array(self._obs_description),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    def generate_observation(self, system_state, env_properties):
        """Returns observation for one batch."""
        eps = system_state.physical_state.epsilon
        cos_eps = jnp.cos(eps)
        sin_eps = jnp.sin(eps)
        norm_state = self.normalize_state(system_state, env_properties)
        norm_state_phys = norm_state.physical_state
        obs = jnp.hstack(
            (
                norm_state_phys.i_s_alpha,
                norm_state_phys.i_s_beta,
                norm_state_phys.psi_r_alpha,
                norm_state_phys.psi_r_beta,
                norm_state_phys.omega_el,
                norm_state_phys.torque,
                cos_eps,
                sin_eps,
                norm_state_phys.u_alpha_buffer,
                norm_state_phys.u_beta_buffer,
            )
        )
        for name in self.control_state:
            obs = jnp.hstack((obs, getattr(norm_state.reference, name)))
        return obs

    @partial(jax.jit, static_argnums=[0, 2])
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        phys = self.PhysicalState(
            u_alpha_buffer=obs[8],
            u_beta_buffer=obs[9],
            epsilon=jnp.arctan2(obs[7], obs[6]) / jnp.pi,
            i_s_alpha=obs[0],
            i_s_beta=obs[1],
            psi_r_alpha=obs[2],
            psi_r_beta=obs[3],
            torque=obs[5],
            omega_el=obs[4],
        )

        def voltage(t):
            return jnp.array([0, 0])

        args = (env_properties.static_params, phys.omega_el)
        if env_properties.saturated:
            raise NotImplementedError("")
            vector_field = partial(self.saturated_ode, action=voltage)
        else:
            vector_field = partial(self.ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.i_s_alpha, phys.i_s_beta, phys.psi_r_alpha, phys.psi_r_beta, phys.epsilon])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = tree_map(lambda x: jnp.zeros_like(x, dtype=x.dtype), solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(
            u_alpha_buffer=jnp.nan,
            u_beta_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_s_alpha=jnp.nan,
            i_s_beta=jnp.nan,
            psi_r_alpha=jnp.nan,
            psi_r_beta=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        with jdc.copy_and_mutate(ref, validate=False) as new_ref:
            for name, pos in zip(self.control_state, range(len(self.control_state))):
                setattr(new_ref, name, obs[10 + pos])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state, env_properties)

    def generate_truncated(self, system_state, env_properties):
        """Returns truncated information for one batch."""
        state_norm = self.normalize_state(system_state, env_properties)
        physical_state_norm = state_norm.physical_state
        i_s_alpha_norm = physical_state_norm.i_s_alpha
        i_s_beta_norm = physical_state_norm.i_s_beta
        i_s = jnp.sqrt(i_s_alpha_norm**2 + i_s_beta_norm**2)
        return i_s[None] > 1

    def generate_terminated(self, system_state, reward, env_properties):
        """Returns terminated information for one batch."""
        return self.generate_truncated(system_state, env_properties)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        state_norm = self.normalize_state(state, env_properties)
        reward = 0
        if "i_s_alpha" in self.control_state and "i_s_beta" in self.control_state:
            reward += self.current_reward_func(
                state_norm.physical_state.i_s_alpha,
                state_norm.physical_state.i_s_beta,
                state_norm.reference.i_s_alpha,
                state_norm.reference.i_s_beta,
                0.85,
            )
        if "torque" in self.control_state:
            reward += self.torque_reward_func(
                state_norm.physical_state.i_s_alpha,
                state_norm.physical_state.i_s_beta,
                state_norm.physical_state.torque,
                state_norm.reference.torque,
                1,
                0.85,
            )
        return jnp.array([reward])

    def current_reward_func(self, i_s_alpha, i_s_beta, i_s_alpha_ref, i_s_beta_ref, gamma):
        mse = 0.5 * (i_s_alpha - i_s_alpha_ref) ** 2 + 0.5 * (i_s_beta - i_s_beta_ref) ** 2
        return -1 * (mse * (1 - gamma))

    def torque_reward_func(self, i_s_alpha, i_s_beta, torque, torque_ref, i_lim_multiplier, gamma):
        i_s = jnp.sqrt(i_s_alpha**2 + i_s_beta**2)
        i_n = 1 / i_lim_multiplier
        i_s_alpha_plus = 0.2 * i_n
        torque_tol = 0.01
        rew = jnp.zeros_like(torque_ref)
        rew = jnp.where(i_s > 1, -1 * jnp.abs(i_s), rew)
        rew = jnp.where((i_s < 1.0) & (i_s > i_n), 0.5 * (1 - (i_s - i_n) / (1 - i_n)) - 1, rew)
        rew = jnp.where(
            (i_s < i_n) & (i_s_alpha > i_s_alpha_plus),
            -0.5 * ((i_s_alpha - i_s_alpha_plus) / (i_n - i_s_alpha_plus)),
            rew,
        )
        rew = jnp.where(
            (i_s < i_n) & (i_s_alpha < i_s_alpha_plus) & (jnp.abs(torque - torque_ref) > torque_tol),
            0.5 * (1 - jnp.abs((torque_ref - torque) / 2)),
            rew,
        )
        rew = jnp.where(
            (i_s < i_n) & (i_s_alpha < i_s_alpha_plus) & (jnp.abs(torque - torque_ref) < torque_tol),
            1 - 0.5 * i_s,
            rew,
        )
        return rew * (1 - gamma)
