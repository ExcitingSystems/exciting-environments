from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from dataclasses import fields
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, tree_leaves
import equinox as eqx
import diffrax
import chex


class CoreEnvironment(eqx.Module):
    tau: float = eqx.field(static=True)
    _solver: diffrax.AbstractSolver = eqx.field(static=True)
    env_properties: eqx.Module
    action_dim: int = eqx.field(static=True)
    physical_state_dim: int = eqx.field(static=True)
    _batch_tracer: jax.Array
    """
    Core Structure of provided Environments. Any new environments needs to inherit from this class
    and implement its abstract properties and methods.

    The simulations are all done with physical state space models. That means that the underlying description
    of the system is given through the differential equation describing the relationship between
    the change of the physical state x(t) w.r.t. the time as a function of the physical state and the
    input/action u(t) applied:

    dx(t)/dt = f(x(t), u(t)).

    The actual outputs of these simulations are discretized from this equation through the use of
    ODE solvers.

    NOTE: There is a difference between the state of the environment and the physical state x(t)
    of the underlying system. The former can also hold various helper variables such as PRNGKeys
    for stochastic environments, while the latter is reserved for the actual physical state of the
    ODE. The physical state is only a part of the full state.
    """

    def __init__(
        self,
        env_properties: eqx.Module,
        tau: float = 1e-4,
        solver=diffrax.Euler(),
    ):
        """Initialization of an environment.

        Args:
            tau (float): Duration of one control step in seconds. Default: 1e-4.
            solver (diffrax.solver): ODE solver used to approximate the ODE solution.
        """
        self.tau = tau
        self._batch_tracer = jnp.array(0.0)
        self._solver = solver
        self.env_properties = env_properties
        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    @abstractmethod
    class PhysicalState(eqx.Module):
        """The physical state x(t) of the underlying system and whose derivative
        w.r.t. time is described in the underlying ODE.

        The values stored in this dataclass are expected to be actual physical values
        that are unnormalized and given in SI units.
        """

        pass

    @abstractmethod
    class Additions(eqx.Module):
        """
        Stores additional environment state variables that may change over time.

        These variables do not directly belong to the physical state but are
        necessary for computations (e.g., internal buffers).
        """

        pass

    @abstractmethod
    class StaticParams(eqx.Module):
        """
        Holds static parameters of the environment that remain constant during simulation.

        Examples:
            - Length of a pendulum
            - Capacitance of a capacitor
            - Mass of an object
        """

        pass

    @abstractmethod
    class Action(eqx.Module):
        """
        Represents the input/action applied to the environment.

        The action influences the system dynamics through the function `f(x(t), u(t))`.
        """

        pass

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_solver_step(self, state, action):
        """
        Performs a single step of state evolution using the ODE solver.

        Args:
            state: The current state of the system.
            action: The action applied at the current step.

        Returns:
            state: The updated state after one simulation step.
        """
        return

    @partial(jax.jit, static_argnums=[0, 3, 4])
    @abstractmethod
    def _ode_solver_simulate_ahead(self, init_state, actions, obs_stepsize, action_stepsize):
        """
        Simulates a trajectory by applying a sequence of actions.

        Args:
            init_state: Initial state at the start of the trajectory.
            actions: Sequence of actions to be applied (shape=(n_action_steps, action_dim)).
            obs_stepsize: Sampling interval for observations.
            action_stepsize: Interval between consecutive action updates.

        Returns:
            states: Simulated trajectory states over time.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def init_state(self, rng: chex.PRNGKey = None, vmap_helper=None):
        """
        Generates an initial state for the environment.

        Args:
            rng (optional): Random key for random initialization.
            vmap_helper (optional): Helper variable for vectorized computation.

        Returns:
            state: The initial state.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, state):
        """
        Generates an observation from the given state.

        Args:
            state: Current state of the environment.

        Returns:
            observation: The computed observation.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_state_from_observation(self, obs, key=None):
        """
        Generates state from a given observation.

        Args:
            obs: The given observation.
            key (optional): Random key.

        Returns:
            state: Computed state.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_reward(self, state, action):
        """
        Computes the reward for a given state-action pair.

        Args:
            state: The current environment state.
            action: The action applied at the current step.

        Returns:
            reward: Computed reward value.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, state):
        """
        Computes truncated flag for given state.

        Args:
            state: The current environment state.

        Returns:
            truncated: Computed truncated flag.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, state, reward):
        """
        Computes terminated flag for given state and reward.

        Args:
            state: The current environment state.
            reward: The reward for current state-action pair.

        Returns:
            terminated: Computed terminated flag.
        """
        return

    class State(eqx.Module):
        """The state of the environment."""

        physical_state: eqx.Module
        PRNGKey: jax.Array
        additions: eqx.Module
        reference: eqx.Module

    class EnvProperties(eqx.Module):
        """The properties of the environment that stay constant during simulation."""

        physical_normalizations: eqx.Module
        action_normalizations: eqx.Module
        static_params: eqx.Module

    @eqx.filter_jit
    def normalize_state(self, state):
        """
        Normalizes the state using predefined normalization parameters.

        Args:
            state: Current environment state.

        Returns:
            norm_state: Normalized state.
        """
        env_properties = self.env_properties
        physical_normalizations = env_properties.physical_normalizations

        new_physical_state = jax.tree.map(
            lambda value, norm: norm.normalize(value),
            state.physical_state,
            physical_normalizations,
        )
        new_reference = jax.tree.map(
            lambda value, norm: norm.normalize(value),
            state.reference,
            physical_normalizations,
        )
        new_state = eqx.tree_at(
            lambda s: (s.physical_state, s.reference),
            state,
            (new_physical_state, new_reference),
        )

        return new_state

    @eqx.filter_jit
    def denormalize_state(self, norm_state):
        """
        Denormalizes a given normalized state.

        Args:
            norm_state: The normalized state to be converted back.

        Returns:
            state: The denormalized state.
        """
        env_properties = self.env_properties
        physical_normalizations = env_properties.physical_normalizations

        new_physical_state = jax.tree.map(
            lambda value, norm: norm.denormalize(value),
            norm_state.physical_state,
            physical_normalizations,
        )
        new_reference = jax.tree.map(
            lambda value, norm: norm.denormalize(value),
            norm_state.reference,
            physical_normalizations,
        )
        new_state = eqx.tree_at(
            lambda s: (s.physical_state, s.reference),
            norm_state,
            (new_physical_state, new_reference),
        )

        return new_state

    @eqx.filter_jit
    def denormalize_action(self, action_norm):
        """
        Denormalizes a given normalized action.

        Args:
            action_norm: The normalized action to be denormalized.

        Returns:
            action: The denormalized action.
        """
        env_properties = self.env_properties
        normalizations = env_properties.action_normalizations
        norm_objects = [getattr(normalizations, name) for name in normalizations.__annotations__]

        denorm_values = jnp.array([norm.denormalize(val) for norm, val in zip(norm_objects, action_norm)])

        return denorm_values

    def reset(
        self,
        rng: chex.PRNGKey = None,
        initial_state: eqx.Module = None,
    ):
        """
        Resets environment to default, random or passed initial state.

        Args:
            rng (optional): Random key for random initialization.
            initial_state (optional): The initial_state to which the environment will be reset.

        Returns:
            obs: Observation of initial state.
            state: The initial state.
        """
        if initial_state is not None:
            assert tree_structure(self.init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as init_state()"
            state = initial_state
        else:
            state = self.init_state(rng)
        obs = self.generate_observation(state)

        return obs, state

    @eqx.filter_jit
    def step(self, state, action_norm):
        """Computes one JAX-JIT compiled simulation step for one batch.

        Args:
            state: The current state of the simulation from which to calculate the next state.
            action: The action to apply to the environment.

        Returns:
            observation: The gathered observation.
            state: New state for the next step.
        """

        # denormalize action
        action = self.denormalize_action(action_norm)

        state = self._ode_solver_step(state, action)
        obs = self.generate_observation(state)

        return obs, state

    def repeat_values(self, x, n_repeat):
        """Repeats the values of x n_repeat times."""
        if x == None:
            return None
        elif isinstance(x, tuple):
            return tuple(self.repeat_values(i, n_repeat) for i in x)
        elif isinstance(x, jax.numpy.ndarray):
            return jnp.full(n_repeat, x)
        elif isinstance(x, float) or isinstance(x, bool):
            return jnp.full(n_repeat, x)
        else:
            raise ValueError(f"State needs to consist of jnp.array, tuple, float or bool, but {type(x)} is given.")

    @eqx.filter_jit
    def sim_ahead(
        self,
        init_state,
        actions,
        obs_stepsize=None,
        action_stepsize=None,
    ):
        """Computes multiple JAX-JIT compiled simulation steps for one batch.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[0] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Warning:
            Depending on the underlying ODE solver (e.g., Tsit5 or other higher-order solvers),
            intermediate evaluations during integration may internally access actions at future time steps.
            Therefore `sim_ahead` is not guaranteed to be numerically equivalent to repeated
            calls of `step`.


        Args:
            init_state: The initial state of the simulation
            actions: A set of actions to be applied to the environment, the value changes every
            action_stepsize (shape=(n_action_steps, action_dim))
            obs_stepsize: The sampling time for the observations
            action_stepsize: The time between changes in the input/action

        Returns:
            observations: The gathered observations.
            states: The computed states during the simulated trajectory.
            last_state: The last state of the simulations.
        """
        if not obs_stepsize:
            obs_stepsize = self.tau

        if not action_stepsize:
            action_stepsize = self.tau

        # denormalize actions
        actions = jax.vmap(self.denormalize_action, in_axes=(0))(actions)

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state,
            actions,
            obs_stepsize,
            action_stepsize,
        )

        observations = jax.vmap(self.generate_observation, in_axes=(0))(states)

        last_state = jax.tree.map(lambda x: x[-1], states)

        return observations, states, last_state

    @eqx.filter_jit
    def generate_rew_trunc_term_ahead(self, states, actions):
        """
        Computes rewards, truncated flags and terminated flags for data generated by `sim_ahead`.

        Args:
            states: A set of environment states over time, including the initial state.
            actions: A set of actions applied sequentially (shape=(n_action_steps, action_dim)).

        Returns:
            reward: Rewards computed for each step.
            truncated: Truncated flags at each step.
            terminated : Terminated flag at each step.
        """
        # denormalize actions
        actions = jax.vmap(self.denormalize_action, in_axes=(0))(actions)

        num_state_steps = jax.tree.leaves(states)[0].shape[0]

        states_without_init_state = jax.tree.map(lambda x: x[1:], states)

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0))(
            states_without_init_state,
            jnp.expand_dims(
                jnp.repeat(
                    actions,
                    int((num_state_steps - 1) / actions.shape[0]),
                ),
                1,
            ),
        )
        truncated = jax.vmap(self.generate_truncated, in_axes=(0))(states)
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0))(states_without_init_state, reward)
        return reward, truncated, terminated

    def soft_constraints(self, state, action_norm):
        return self.soft_constraints_logic(self, state, action_norm)

    @eqx.filter_jit
    def vmap_step(self, state, action):
        """Computes one JAX-JIT compiled simulation step for multiple (batch_size) batches.

        Args:
            state: The current state of the simulation from which to calculate the next
                state.
            action: The action to apply to the environment (shape=(batch_size, action_dim)).

        Returns:
            observation: The gathered observations.
            state: New state for the next step.
        """
        self._assert_batched()
        next_obs, next_state = jax.vmap(lambda e, s, a: e.step(s, a))(self, state, action)

        return next_obs, next_state

    @eqx.filter_jit
    def vmap_sim_ahead(self, init_state, actions, obs_stepsize=None, action_stepsize=None):
        """Computes multiple JAX-JIT compiled simulation steps for multiple (batch_size) batches.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[1] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every
            action_stepsize (shape=(batch_size, n_action_steps, action_dim)).
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.

        Returns:
            observations: The gathered observations.
            states: The computed states during the simulated trajectory.
            last_state: The last state of the simulations.
        """
        self._assert_batched()
        if not obs_stepsize:
            obs_stepsize = self.tau

        if not action_stepsize:
            action_stepsize = self.tau
        next_obs, next_states, last_state = jax.vmap(lambda e, s, a: e.sim_ahead(s, a))(self, init_state, actions)
        return next_obs, next_states, last_state

    @eqx.filter_jit
    def vmap_generate_rew_trunc_term_ahead(self, states, actions):
        """
         Computes reward, truncated, and terminated flags for multiple batches
         simulated by `vmap_sim_ahead`.


        Args:
            states: Environment states over time, including the initial state.
            actions: Actions applied sequentially (shape=(n_action_steps, action_dim)).

         Returns:
            reward: Rewards computed for each step for every batch.
            truncated: Truncated flags at each step for every batch.
            terminated : Terminated flag at each step for every batch.
        """
        self._assert_batched()
        reward, truncated, terminated = jax.vmap(lambda e, s, a: e.generate_rew_trunc_term_ahead(s, a))(
            self, states, actions
        )

        return reward, truncated, terminated

    @eqx.filter_jit
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        """
        Generates an initial state for all batches, either using default values or random initialization.

        Args:
            rng (optional): Random keys for random initializations.

        Returns:
            state: The initial state for all batches.
        """
        self._assert_batched()
        return jax.vmap(lambda e, k: e.init_state(k))(self, rng)

    @eqx.filter_jit
    def vmap_reset(self, rng: chex.PRNGKey = None, initial_state: eqx.Module = None):
        """
        Resets environment (all batches) to default, random or passed initial state.

        Args:
            rng (optional): Random keys for random initializations.
            initial_state (optional): initial_state to which the environment will be reset.

        Returns:
            obs: Observation of initial state for all batches.
            state: The initial state for all batches.
        """
        self._assert_batched()
        if initial_state is not None:
            assert tree_structure(self.vmap_init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.vmap_init_state()"

        obs, state = jax.vmap(lambda e, k: e.reset(k))(self, rng)
        return obs, state

    @eqx.filter_jit
    def vmap_generate_state_from_observation(self, obs, key=None):
        """
        Generates state for each batch from a given observation.

        Args:
            obs: The given observation of all batches.
            key (optional): Random keys.

        Returns:
            state: Computed state for each batch.
        """
        self._assert_batched()
        state = jax.vmap(lambda e, o, k: e.generate_state_from_observation(o, k))(self, obs, key)
        return state

    def _assert_batched(self):
        """Checks if the environment is batched by looking at the dummy leaf."""
        if jnp.ndim(self._batch_tracer) == 0:
            raise RuntimeError(
                "Calling a vmap method on a single-environment instance. Please use the 'make' function with a batch_size to create a batched environment or create it manually."
            )

    @property
    def batch_size(self):
        """Returns batch_size if environment is batched, else None."""
        if jnp.ndim(self._batch_tracer) > 0:
            return self._batch_tracer.shape[0]
        return None
