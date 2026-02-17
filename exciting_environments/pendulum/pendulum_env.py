from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure, tree_map

from jax import lax
import diffrax
import equinox as eqx
import chex
from dataclasses import fields
from exciting_environments.utils import MinMaxNormalization


from exciting_environments import CoreEnvironment


class Pendulum(CoreEnvironment):
    """
    State Variables:
        ``['theta', 'omega']``

    Action Variable:
        ``['torque']``

    Initial State:
        Unless chosen otherwise, theta=pi and omega=0

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>>
        >>> # Create the environment
        >>> pend=excenvs.Pendulum(batch_size=4)
        >>>
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_pend=GymWrapper(env=pend)
        >>>
        >>> # Reset the environment with default initial values
        >>> gym_pend.reset()
        >>>
        >>> # Perform step
        >>> obs, reward, terminated,  truncated = gym_pend.step(action=jnp.ones(4).reshape(-1,1))
        >>>

    """

    def __init__(
        self,
        batch_size: int = 8,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size (int): Number of parallel environment simulations. Default: 8
            physical_normalizations (dict): Min and max values of the physical state of the environment for normalization.
                theta (MinMaxNormalization): Rotation angle. Default: min=-jnp.pi, max=jnp.pi
                omega (MinMaxNormalization): Angular velocity. Default: min=-10, max=10
            action_normalizations (dict): Min and max values of the input/action for normalization.
                torque (MinMaxNormalization): Maximum torque that can be applied to the system as an action. Default: min=-20, max=20
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params (dict): Parameters of environment which do not change during simulation.
                l (float): Length of the pendulum. Default: 1
                m (float): Mass of the pendulum tip. Default: 1
                g (float): Gravitational acceleration. Default: 9.81
            control_state (list): Components of the physical state that are considered in reference tracking.
            solver (diffrax.solver): Solver used to compute state for next step.
            tau (float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of MinMaxNormalization of physical_normalizations and action_normalizations as well as static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_normalizations:
            physical_normalizations = {
                "theta": MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
                "omega": MinMaxNormalization(min=-10, max=10),
            }

        if not action_normalizations:
            action_normalizations = {"torque": MinMaxNormalization(min=-20, max=20)}

        if not soft_constraints:
            soft_constraints = self.default_soft_constraints

        if not static_params:
            static_params = {"g": 9.81, "l": 2, "m": 1}

        if not control_state:
            control_state = []

        self.control_state = control_state
        self.soft_constraints = soft_constraints

        physical_normalizations = self.PhysicalState(**physical_normalizations)
        action_normalizations = self.Action(**action_normalizations)
        static_params = self.StaticParams(**static_params)

        env_properties = self.EnvProperties(
            physical_normalizations=physical_normalizations,
            action_normalizations=action_normalizations,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

    class PhysicalState(eqx.Module):
        """Dataclass containing the physical state of the environment."""

        theta: jax.Array
        omega: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class StaticParams(eqx.Module):
        """Dataclass containing the static parameters of the environment."""

        g: jax.Array
        l: jax.Array
        m: jax.Array

    class Action(eqx.Module):
        """Dataclass containing the action, that can be applied to the environment."""

        torque: jax.Array

    def _ode(self, t, y, args, action):
        theta, omega = y
        params = args
        d_omega = (action(t)[0] + params.l * params.m * params.g * jnp.sin(theta)) / (params.m * (params.l) ** 2)
        d_theta = omega
        d_y = d_theta, d_omega
        return d_y

    @partial(jax.jit, static_argnums=0)
    def _ode_solver_step(self, state, action, static_params):
        """Computes the next state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            next_state: The computed next state after the one step simulation.
        """

        physical_state = state.physical_state
        args = static_params

        torque = lambda t: action

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([physical_state.theta, physical_state.omega])

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)
        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        theta_k1 = y[0]
        omega_k1 = y[1]
        theta_k1 = ((theta_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        new_physical_state = self.PhysicalState(theta=theta_k1, omega=omega_k1)
        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = eqx.tree_at(lambda s: (s.physical_state, s.additions), state, (new_physical_state, new_additions))
        return new_state

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def _ode_solver_simulate_ahead(self, init_state, actions, static_params, obs_stepsize, action_stepsize):
        """Computes multiple simulation steps for one batch.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every.
            action_stepsize (shape=(n_action_steps, action_dim)).
            static_params: The constant properties of the simulation.
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.

        Returns:
            next_states: The computed states during the multiple step simulation.
        """

        init_physical_state = init_state.physical_state
        args = static_params

        def torque(t):
            return actions[jnp.array(t / action_stepsize, int)]

        vector_field = partial(self._ode, action=torque)

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

        theta_t = sol.ys[0]
        omega_t = sol.ys[1]
        obs_len = omega_t.shape[0]
        # keep theta between -pi and pi
        theta_t = ((theta_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        physical_states = self.PhysicalState(theta=theta_t, omega=omega_t)
        ref = self.PhysicalState(
            theta=jnp.full(obs_len, init_state.reference.theta),
            omega=jnp.full(obs_len, init_state.reference.omega),
        )
        y0 = tuple([theta_t[-1], omega_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len), active_solver_state=jnp.full(obs_len, True)
        )
        PRNGKey = jnp.full(obs_len, init_state.PRNGKey)
        return self.State(
            physical_state=physical_states,
            PRNGKey=PRNGKey,
            additions=additions,
            reference=ref,
        )

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default or random initial state for one batch."""
        if rng is None:
            phys = self.PhysicalState(
                theta=1.0,
                omega=0.0,
            )
            subkey = jnp.nan
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(2,))
            phys = self.PhysicalState(
                theta=state_norm[0],
                omega=state_norm[1],
            )
            key, subkey = jax.random.split(rng)

        torque = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.theta, phys.omega])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = tree_map(lambda x: x * jnp.nan, solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(theta=jnp.nan, omega=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state, env_properties)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state, env_properties)
        for name in self.control_state:
            if name == "theta":
                theta = getattr(state.physical_state, name)
                theta_ref = getattr(state.reference, name)
                reward += -((jnp.sin(theta) - jnp.sin(theta_ref)) ** 2 + (jnp.cos(theta) - jnp.cos(theta_ref)) ** 2)
            else:
                reward += -((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name)) ** 2)
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state, env_properties)
        norm_state_phys = norm_state.physical_state
        obs = jnp.hstack(
            (
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

    @partial(jax.jit, static_argnums=0)
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        phys = self.PhysicalState(
            theta=obs[0],
            omega=obs[1],
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan

        torque = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.theta, phys.omega])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = tree_map(lambda x: x * jnp.nan, solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)  # None
        ref = self.PhysicalState(theta=jnp.nan, omega=jnp.nan)
        new_ref = ref
        for i, name in enumerate(self.control_state):
            new_ref = eqx.tree_at(lambda r: getattr(r, name), new_ref, obs[2 + i])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state, env_properties)

    def default_soft_constraints(self, state, action_norm, env_properties):
        state_norm = self.normalize_state(state, env_properties)
        physical_state_norm = state_norm.physical_state
        phys_soft_const = jax.tree.map(lambda _: jnp.nan, physical_state_norm)
        phys_soft_const = eqx.tree_at(
            lambda s: s.omega, phys_soft_const, jax.nn.relu(jnp.abs(physical_state_norm.omega) - 1.0)
        )
        # define soft constraints for action
        act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
        return phys_soft_const, act_soft_constr

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, state, env_properties):
        """Returns truncated information for one batch."""
        obs = self.generate_observation(state, env_properties)
        return jnp.abs(obs) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information for one batch."""
        return reward == 0

    @property
    def obs_description(self):
        return np.hstack(
            [
                np.array(["theta", "omega"]),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    @property
    def action_description(self):
        return np.array(["torque"])
