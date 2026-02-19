from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure, tree_map

import chex
import diffrax
import equinox as eqx
from dataclasses import fields

from exciting_environments import CoreEnvironment
from exciting_environments.utils import MinMaxNormalization


def cartpole_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize(state)
    physical_state_norm = state_norm.physical_state
    constrained_states = ["deflection", "velocity", "omega"]
    names = [f.name for f in fields(type(physical_state_norm))]
    values = [
        jax.nn.relu(jnp.abs(getattr(physical_state_norm, n)) - 1.0) if n in constrained_states else jnp.nan
        for n in names
    ]

    phys_soft_const = eqx.tree_unflatten(eqx.tree_structure(physical_state_norm), values)
    act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
    return phys_soft_const, act_soft_constr


class CartPole(CoreEnvironment):
    control_state: list = eqx.field(static=True)
    soft_constraints_logic: Callable = eqx.field(static=True)
    """
    State Variables
        ``['deflection', 'velocity', 'theta', 'omega']``

    Action Variable:
        ``['force']``

    Initial State:
        Unless chosen otherwise, deflection, omega and velocity is set to zero and theta is set to pi.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>>
        >>> # Create the environment
        >>> cartpole= excenv.CartPole(batch_size=5)
        >>>
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_cartpole=GymWrapper(env=cartpole)
        >>>
        >>> # Reset the environment with default initial values
        >>> gym_cartpole.reset()
        >>>
        >>> # Perform step
        >>> obs,reward,terminated,truncated = gym_cartpole.step(action=jnp.ones(5).reshape(-1,1))
        >>>

    """

    def __init__(
        self,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 2e-2,
    ):
        """
        Args:
            physical_normalizations(dict): min-max normalization values of the physical state of the environment.
                deflection(MinMaxNormalization): Deflection of the cart. Default: min=-10, max=10
                velocity(MinMaxNormalization): Velocity of the cart. Default: min=-10, max=10
                theta(MinMaxNormalization): Rotation angle of the pole. Default: min=-jnp.pi, max=jnp.pi
                omega(MinMaxNormalization): Angular velocity. Default: min=-10, max=10
            action_normalizations(dict): min-max normalization values of the input/action.
                force(MinMaxNormalization): Maximum torque that can be applied to the system as action. Default: min=-20, max=20
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params(dict): Parameters of environment which do not change during simulation.
                mu_p(float): Coefficient of friction of pole on cart. Default: 0.000002
                mu_c(float): Coefficient of friction of cart on track. Default: 0.0005
                l(float): Half-pole length. Default: 0.5
                m_p(float): Mass of the pole. Default: 0.1
                m_c(float): Mass of the cart. Default: 1
                g(float): Gravitational acceleration. Default: 9.81
            control_state (list): Components of the physical state that are considered in reference tracking.
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of MinMaxNormalization of physical_normalizations and action_normalizations as well as static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_normalizations:
            physical_normalizations = {
                "deflection": MinMaxNormalization(min=jnp.array(-2.4), max=jnp.array(2.4)),
                "velocity": MinMaxNormalization(min=jnp.array(-8), max=jnp.array(8)),
                "theta": MinMaxNormalization(min=jnp.array(-jnp.pi), max=jnp.array(jnp.pi)),
                "omega": MinMaxNormalization(min=jnp.array(-8), max=jnp.array(8)),
            }
        if not action_normalizations:
            action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-20), max=jnp.array(20))}

        if not static_params:
            static_params = {  # typical values from Source with DOI: 10.1109/TSMC.1983.6313077
                "mu_p": jnp.array(0.000002),
                "mu_c": jnp.array(0.0005),
                "l": jnp.array(0.5),
                "m_p": jnp.array(0.1),
                "m_c": jnp.array(1),
                "g": jnp.array(9.81),
            }

        if not control_state:
            control_state = []

        logic = soft_constraints if soft_constraints else cartpole_soft_constraints
        self.soft_constraints_logic = logic
        self.control_state = control_state

        physical_normalizations = self.PhysicalState(**physical_normalizations)
        action_normalizations = self.Action(**action_normalizations)
        static_params = self.StaticParams(**static_params)

        env_properties = self.EnvProperties(
            physical_normalizations=physical_normalizations,
            action_normalizations=action_normalizations,
            static_params=static_params,
        )
        super().__init__(env_properties=env_properties, tau=tau, solver=solver)

    class PhysicalState(eqx.Module):
        """Dataclass containing the physical state of the environment."""

        deflection: jax.Array
        velocity: jax.Array
        theta: jax.Array
        omega: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class StaticParams(eqx.Module):
        """Dataclass containing the static parameters of the environment."""

        mu_p: jax.Array
        mu_c: jax.Array
        l: jax.Array
        m_p: jax.Array
        m_c: jax.Array
        g: jax.Array

    class Action(eqx.Module):
        """Dataclass containing the action that can be applied to the environment."""

        force: jax.Array

    def _ode(self, t, y, args, action):
        deflection, velocity, theta, omega = y
        params = args
        d_omega = (
            params.g * jnp.sin(theta)
            + jnp.cos(theta)
            * (
                (-action(t)[0] - params.m_p * params.l * (omega**2) * jnp.sin(theta) + params.mu_c * jnp.sign(velocity))
                / (params.m_c + params.m_p)
            )
            - (params.mu_p * omega) / (params.m_p * params.l)
        ) / (params.l * (4 / 3 - (params.m_p * (jnp.cos(theta)) ** 2) / (params.m_c + params.m_p)))

        d_velocity = (
            action(t)[0]
            + params.m_p * params.l * ((omega**2) * jnp.sin(theta) - d_omega * jnp.cos(theta))
            - params.mu_c * jnp.sign(velocity)
        ) / (params.m_c + params.m_p)
        d_theta = omega
        d_deflection = velocity
        d_y = d_deflection, d_velocity, d_theta, d_omega
        return d_y

    @eqx.filter_jit
    def _ode_solver_step(self, state, action):
        """Computes state by simulating one step.

        Source DOI: 10.1109/TSMC.1983.6313077

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.

        Returns:
            state: The computed state after the one step simulation.
        """
        static_params = self.env_properties.static_params
        physical_state = state.physical_state
        args = static_params

        force = lambda t: action

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple(
            [
                physical_state.deflection,
                physical_state.velocity,
                physical_state.theta,
                physical_state.omega,
            ]
        )

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)
        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]
        theta_k1 = y[2]
        omega_k1 = y[3]
        theta_k1 = ((theta_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        new_physical_state = self.PhysicalState(
            deflection=deflection_k1,
            velocity=velocity_k1,
            theta=theta_k1,
            omega=omega_k1,
        )
        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = eqx.tree_at(lambda s: (s.physical_state, s.additions), state, (new_physical_state, new_additions))
        return new_state

    @eqx.filter_jit
    def _ode_solver_simulate_ahead(self, init_state, actions, obs_stepsize, action_stepsize):
        """Computes multiple simulation steps for one batch.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every.
            action_stepsize (shape=(n_action_steps, action_dim)).
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.

        Returns:
            next_states: The computed states during the multiple step simulation.
        """
        static_params = self.env_properties.static_params
        init_physical_state = init_state.physical_state
        args = static_params

        def force(t):
            return actions[jnp.array(t / action_stepsize, int)]

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = action_stepsize * actions.shape[0]
        init_physical_state_array, _ = tree_flatten(init_physical_state)
        y0 = tuple(init_physical_state_array)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / obs_stepsize)))  #
        sol = diffrax.diffeqsolve(
            term,
            self._solver,
            t0,
            t1,
            dt0=obs_stepsize,
            y0=y0,
            args=args,
            saveat=saveat,
        )

        deflection_t = sol.ys[0]
        velocity_t = sol.ys[1]
        theta_t = sol.ys[2]
        omega_t = sol.ys[3]
        obs_len = omega_t.shape[0]

        # keep theta between -pi and pi
        theta_t = ((theta_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        physical_states = self.PhysicalState(deflection=deflection_t, velocity=velocity_t, theta=theta_t, omega=omega_t)
        y0 = tuple([deflection_t[-1], velocity_t[-1], theta_t[-1], omega_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len), active_solver_state=jnp.full(obs_len, True)
        )
        PRNGKey = jnp.broadcast_to(jnp.asarray(init_state.PRNGKey), (obs_len,) + jnp.asarray(init_state.PRNGKey).shape)
        ref = self.PhysicalState(
            deflection=jnp.full(obs_len, init_state.reference.deflection),
            velocity=jnp.full(obs_len, init_state.reference.velocity),
            theta=jnp.full(obs_len, init_state.reference.theta),
            omega=jnp.full(obs_len, init_state.reference.omega),
        )
        return self.State(
            physical_state=physical_states,
            PRNGKey=PRNGKey,
            additions=additions,
            reference=ref,
        )

    @eqx.filter_jit
    def init_state(self, rng: chex.PRNGKey = None):
        """Returns default or random initial state for one batch."""
        env_properties = self.env_properties
        if rng is None:
            phys = self.PhysicalState(
                deflection=jnp.array(0.0),
                velocity=jnp.array(0.0),
                theta=jnp.array(1.0),
                omega=jnp.array(0.0),
            )
            subkey = jnp.array(jnp.nan)
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(4,))
            phys = self.PhysicalState(
                deflection=state_norm[0],
                velocity=state_norm[1],
                theta=state_norm[2],
                omega=state_norm[3],
            )
            key, subkey = jax.random.split(rng)

        force = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.deflection, phys.velocity, phys.theta, phys.omega])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan, theta=jnp.nan, omega=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_reward(self, state, action):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state)
        for name in self.control_state:
            if name == "theta":
                theta = getattr(state.physical_state, name)
                theta_ref = getattr(state.reference, name)
                reward += -((jnp.sin(theta) - jnp.sin(theta_ref)) ** 2 + (jnp.cos(theta) - jnp.cos(theta_ref)) ** 2)
            else:
                reward += -((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name)) ** 2)
        return jnp.array([reward])

    @eqx.filter_jit
    def generate_observation(self, state):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state)
        norm_state_phys = norm_state.physical_state
        obs = jnp.hstack(
            (
                norm_state_phys.deflection,
                norm_state_phys.velocity,
                norm_state_phys.theta,
                norm_state_phys.omega,
            )
        )
        for name in self.control_state:
            obs = jnp.hstack(
                (
                    obs,
                    getattr(norm_state.reference, name),
                )
            )
        return obs

    @eqx.filter_jit
    def generate_state_from_observation(self, obs, key=None):
        """Generates state from observation for one batch."""
        env_properties = self.env_properties
        phys = self.PhysicalState(
            deflection=obs[0],
            velocity=obs[1],
            theta=obs[2],
            omega=obs[3],
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan

        force = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.deflection, phys.velocity, phys.theta, phys.omega])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan, theta=jnp.nan, omega=jnp.nan)
        new_ref = ref
        for i, name in enumerate(self.control_state):
            new_ref = eqx.tree_at(lambda r: getattr(r, name), new_ref, obs[4 + i])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_truncated(self, state):
        """Returns truncated information for one batch."""
        obs = self.generate_observation(state)
        return jnp.abs(obs) > 1

    @eqx.filter_jit
    def generate_terminated(self, state, reward):
        """Returns terminated information for one batch."""
        return reward == 0

    @property
    def action_description(self):
        return np.array(["force"])

    @property
    def obs_description(self):
        return np.hstack(
            [
                np.array(["deflection", "velocity", "theta", "omega"]),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )
