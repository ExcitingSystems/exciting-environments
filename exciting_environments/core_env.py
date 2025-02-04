from abc import ABC
from abc import abstractmethod
from functools import partial
from dataclasses import fields
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import diffrax
import chex


class CoreEnvironment(ABC):
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
        batch_size: int,
        env_properties: jdc.pytree_dataclass,
        tau: float = 1e-4,
        solver=diffrax.Euler(),
    ):
        """Initialization of an environment.

        Args:
            batch_size (int): Number of parallel environment simulations.
            env_properties(jdc.pytree_dataclass): All parameters and properties of the environment.
            tau (float): Duration of one control step in seconds. Default: 1e-4.
            solver (diffrax.solver): ODE solver used to approximate the ODE solution.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver
        self.env_properties = env_properties
        self.in_axes_env_properties = self.create_in_axes_dataclass(env_properties)
        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    @abstractmethod
    @jdc.pytree_dataclass
    class PhysicalState:
        """The physical state x(t) of the underlying system and whose derivative
        w.r.t. time is described in the underlying ODE.

        The values stored in this dataclass are expected to be actual physical values
        that are unnormalized.
        """

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Additions:
        """Additional information that can change from iteration to iteration and that is
        stored in the state of the system"""

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class StaticParams:
        """Static parameters of the environment that stay constant during simulation.
        This could be the length of a pendulum, the capacitance of a capacitor,
        the mass of a specific element and similar..
        """

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Action:
        """The input/action applied to the environment that is used to influence the
        dynamics from the outside.
        """

        pass

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_solver_step(self, state, action, static_params):
        """Computes state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """

        return

    @partial(jax.jit, static_argnums=[0, 4, 5])
    @abstractmethod
    def _ode_solver_simulate_ahead(self, init_state, actions, static_params, obs_stepsize, action_stepsize):
        """Computes states by simulating a trajectory with given actions.

        Args:
           init_state: The initial state of the simulation.
           actions: A set of actions to be applied to the environment, the value changes every
           action_stepsize (shape=(batch_size, n_action_steps, action_dim)).
           static_params: The static parameters of the environment.
           obs_stepsize: The sampling time for the observations.
           action_stepsize: The time between changes in the input/action.

           Returns:
            states: The computed states during the simulated trajectory.
        """
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default or random initial state."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, state, env_properties):
        """Returns observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_reward(self, state, action, env_properties):
        """Returns a reward for given state and action."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, state, env_properties):
        """Returns truncated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information."""
        return

    @jdc.pytree_dataclass
    class State:
        """The state of the environment."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        additions: jdc.pytree_dataclass
        reference: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """The properties of the environment that stay constant during simulation."""

        physical_normalizations: jdc.pytree_dataclass
        action_normalizations: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def create_in_axes_dataclass(self, dataclass):
        with jdc.copy_and_mutate(dataclass, validate=False) as dataclass_in_axes:
            for field in fields(dataclass_in_axes):
                name = field.name
                value = getattr(dataclass_in_axes, name)
                if value == None:
                    setattr(dataclass_in_axes, name, None)
                elif isinstance(value, list):
                    raise ValueError(
                        f'Passed env property "{name}" needs to be a jnp.array to have different setting per batch, but list is given.'
                    )
                elif jdc.is_dataclass(value):
                    setattr(dataclass_in_axes, name, self.create_in_axes_dataclass(value))
                elif jnp.isscalar(value):
                    setattr(dataclass_in_axes, name, None)
                elif isinstance(value, jax.numpy.ndarray):
                    if value.shape[0] == self.batch_size:
                        setattr(dataclass_in_axes, name, 0)
                    else:
                        setattr(dataclass_in_axes, name, None)
                else:
                    raise ValueError(
                        f'Passed env property "{name}" needs to be a scalar, jnp.array or jdc.pytree_dataclass, but {type(value)} is given.'
                    )
        return dataclass_in_axes

    @partial(jax.jit, static_argnums=0)
    def normalize_state(self, state, env_properties):
        physical_normalizations = env_properties.physical_normalizations
        with jdc.copy_and_mutate(state, validate=True) as norm_state:
            for field in fields(norm_state.physical_state):
                name = field.name
                norm_single_state = getattr(physical_normalizations, name).normalize(
                    getattr(state.physical_state, name)
                )
                norm_ref_single_state = getattr(physical_normalizations, name).normalize(getattr(state.reference, name))
                setattr(norm_state.physical_state, name, norm_single_state)
                setattr(norm_state.reference, name, norm_ref_single_state)
        return norm_state

    @partial(jax.jit, static_argnums=0)
    def denormalize_state(self, norm_state, env_properties):
        physical_normalizations = env_properties.physical_normalizations
        with jdc.copy_and_mutate(norm_state, validate=True) as state:
            for field in fields(state.physical_state):
                name = field.name
                single_state = getattr(physical_normalizations, name).denormalize(
                    getattr(norm_state.physical_state, name)
                )
                ref_single_state = getattr(physical_normalizations, name).denormalize(
                    getattr(norm_state.reference, name)
                )
                setattr(state.physical_state, name, single_state)
                setattr(state.reference, name, ref_single_state)
        return state

    @partial(jax.jit, static_argnums=0)
    def denormalize_action(self, action_norm, env_properties):
        normalizations = env_properties.action_normalizations
        action_denorm = jnp.zeros_like(action_norm)
        for i, field in enumerate(fields(normalizations)):
            norms = getattr(normalizations, field.name)
            action_denorm = action_denorm.at[i].set(norms.denormalize(action_norm[i]))
        return action_denorm

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

    @partial(jax.jit, static_argnums=0)
    def step(self, state, action_norm, env_properties):
        """Computes one JAX-JIT compiled simulation step for one batch.

        Args:
            state: The current state of the simulation from which to calculate the next state.
            action: The action to apply to the environment.
            env_properties: Contains action/state constraints and static parameters.

        Returns:
            observation: The gathered observation.
            state: New state for the next step.
        """

        assert action_norm.shape == (self.action_dim,), (
            f"The action needs to be of shape (action_dim,) which is "
            + f"{(self.action_dim,)}, but {action_norm.shape} is given"
        )

        physical_state_shape = jnp.array(tree_flatten(state.physical_state)[0]).T.shape

        assert physical_state_shape == (self.physical_state_dim,), (
            "The physical state needs to be of shape (physical_state_dim,) which is "
            + f"{(self.physical_state_dim,)}, but {physical_state_shape} is given"
        )

        # denormalize action
        action = self.denormalize_action(action_norm, env_properties)

        state = self._ode_solver_step(state, action, env_properties.static_params)
        obs = self.generate_observation(state, env_properties)

        return obs, state

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

        Returns:
            observations: The gathered observations.
            states: The computed states during the simulated trajectory.
            last_state: The last state of the simulations.
        """

        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        init_physical_state_shape = jnp.array(tree_flatten(init_state.physical_state)[0]).T.shape
        assert init_physical_state_shape == (self.physical_state_dim,), (
            "The initial physical state needs to be of shape (env.physical_state_dim,) which is "
            + f"{(self.physical_state_dim,)}, but {init_physical_state_shape} is given"
        )

        # denormalize actions
        actions = jax.vmap(self.denormalize_action, in_axes=(0, None))(actions, env_properties)

        single_state_struct = tree_structure(init_state)

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state, actions, env_properties.static_params, obs_stepsize, action_stepsize
        )

        # generate observations for all timesteps
        observations = jax.vmap(self.generate_observation, in_axes=(0, None))(states, env_properties)

        # get last state so that the simulation can be continued from the end point
        states_flatten, _ = tree_flatten(states)
        last_state = tree_unflatten(single_state_struct, jnp.array(states_flatten)[:, -1])

        return observations, states, last_state

    @partial(jax.jit, static_argnums=0)
    def generate_rew_trunc_term_ahead(self, states, actions, env_properties):
        """Computes reward,truncated and terminated for the data simulated by sim_ahead"""

        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        actions = jax.vmap(self.denormalize_action, in_axes=(0, None))(actions, env_properties)

        states_flatten, struct = tree_flatten(states)

        states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, None))(
            states_without_init_state,
            jnp.expand_dims(jnp.repeat(actions, int((jnp.array(states_flatten).shape[1] - 1) / actions.shape[0])), 1),
            env_properties,
        )
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, None))(states, env_properties)
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, None))(
            states_without_init_state, reward, env_properties
        )
        return reward, truncated, terminated

    @partial(jax.jit, static_argnums=0)
    def vmap_step(self, state, action):
        """Computes one JAX-JIT compiled simulation step for multiple (batch_size) batches.

        Args:
            state: The current state of the simulation from which to calculate the next
                state (shape=(batch_size, state_dim)).
            action: The action to apply to the environment (shape=(batch_size, action_dim)).

        Returns:
            observation: The gathered observations (shape=(batch_size,obs_dim)).
            state: New state for the next step.
        """
        assert action.shape == (
            self.batch_size,
            self.action_dim,
        ), (
            "The action needs to be of shape (batch_size, action_dim) which is "
            + f"{(self.batch_size, self.action_dim)}, but {action.shape} is given"
        )

        physical_state_shape = jnp.array(tree_flatten(state.physical_state)[0]).T.shape
        assert physical_state_shape == (
            (
                self.batch_size,
                self.physical_state_dim,
            )
        ), (
            "The physical state needs to be of shape (batch_size, physical_state_dim) which is "
            + f"{(self.batch_size, self.physical_state_dim)}, but {physical_state_shape} is given"
        )

        # vmap single operations
        obs, state = jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(
            state, action, self.env_properties
        )
        return obs, state

    @partial(jax.jit, static_argnums=[0, 3, 4])
    def vmap_sim_ahead(self, init_state, actions, obs_stepsize, action_stepsize):
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
        assert (
            obs_stepsize <= action_stepsize
        ), "The action stepsize should be greater or equal to the observation stepsize."
        assert actions.ndim == 3, "The actions need to have three dimensions: (batch_size, n_action_steps, action_dim)"
        assert (
            actions.shape[0] == self.batch_size
        ), f"The first dimension does not correspond to the batch size which is {self.batch_size}, but {actions.shape[0]} is given"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        init_physical_state_shape = jnp.array(tree_flatten(init_state.physical_state)[0]).T.shape
        assert init_physical_state_shape == (self.batch_size, self.physical_state_dim), (
            "The initial physical state needs to be of shape (batch_size, physical_state_dim,) which is "
            + f"{(self.batch_size, self.physical_state_dim)}, but {init_physical_state_shape} is given"
        )

        # vmap single operations
        observations, states, last_state = jax.vmap(
            self.sim_ahead, in_axes=(0, 0, self.in_axes_env_properties, None, None)
        )(init_state, actions, self.env_properties, obs_stepsize, action_stepsize)

        return observations, states, last_state

    @partial(jax.jit, static_argnums=0)
    def vmap_generate_rew_trunc_term_ahead(self, states, actions):
        """Computes reward,truncated and terminated for the data of multiple (batch_size) batches simulated by vmap_sim_ahead."""

        assert actions.ndim == 3, "The actions need to have three dimensions: (batch_size, n_action_steps, action_dim)"
        assert (
            actions.shape[0] == self.batch_size
        ), f"The first dimension does not correspond to the batch size which is {self.batch_size}, but {actions.shape[0]} is given"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"
        reward, truncated, terminated = jax.vmap(
            self.generate_rew_trunc_term_ahead, in_axes=(0, 0, self.in_axes_env_properties)
        )(states, actions, self.env_properties)

        return reward, truncated, terminated

    @partial(jax.jit, static_argnums=0)
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        """Returns default or random initial state for all batches."""
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0, 0))(
            self.env_properties, rng, jnp.ones(self.batch_size)
        )

    @partial(jax.jit, static_argnums=0)
    def vmap_reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment (all batches) to default, random or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.vmap_init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.vmap_init_state()"

        obs, state = jax.vmap(
            self.reset,
            in_axes=(self.in_axes_env_properties, 0, 0, 0),
        )(self.env_properties, rng, initial_state, jnp.ones(self.batch_size))

        return obs, state

    @partial(jax.jit, static_argnums=0)
    def vmap_generate_state_from_observation(self, obs, key=None):
        """Generates state from observation for all batches."""
        state = jax.vmap(self.generate_state_from_observation, in_axes=(0, self.in_axes_env_properties, 0))(
            obs, self.env_properties, key
        )
        return state
