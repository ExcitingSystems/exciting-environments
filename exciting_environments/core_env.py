from abc import ABC
from abc import abstractmethod
from functools import partial
from dataclasses import fields
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax_dataclasses as jdc
import diffrax
import chex


class CoreEnvironment(ABC):
    """
    Description:
        Core Structure of provided Environments.

    """

    def __init__(
        self,
        batch_size: int,
        physical_constraints,
        action_constraints,
        static_params,
        tau: float = 1e-4,
        solver=diffrax.Euler(),
        reward_func: Callable = None,
    ):
        """
        Args:
            batch_size (int): Number of parallel environment simulations.
            physical_constraints (jdc.pytree_dataclass): Constraints of the physical state of the environment.
            action_constraints (jdc.pytree_dataclass): Constraints of the input/action.
            static_params (jdc.pytree_dataclass): Parameters of environment which do not change during simulation.
            tau (float): Duration of one control step in seconds. Default: 1e-4.
            solver (diffrax.solver): ODE solver used to approximate the ODE solution.
            reward_func (Callable): Reward function for training. Needs observation vector, action and action_constraints as Parameters.
                Default: None (default_reward_func from class)
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver
        self.env_properties = self.EnvProperties(
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )
        self.in_axes_env_properties = self.create_in_axes_env_properties()
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    @property
    def default_reward_function(self):
        """Returns the default reward function for the given environment."""
        return self.default_reward_func

    @abstractmethod
    @jdc.pytree_dataclass
    class PhysicalState:
        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Optional:
        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class StaticParams:
        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Action:
        pass

    @jdc.pytree_dataclass
    class State:
        """Stores the state of the simulation."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        optional: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Stores properties of the environment that stay constant during simulation."""

        physical_constraints: jdc.pytree_dataclass
        action_constraints: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def create_in_axes_env_properties(self):
        axes = []
        """Returns Dataclass for in_axes to use jax.vmap."""
        for field in fields(self.env_properties):

            values = list(vars(getattr(self.env_properties, field.name)).values())
            names = list(vars(getattr(self.env_properties, field.name)).keys())
            in_axes_physical = []
            for v, n in zip(values, names):
                if jnp.isscalar(v):
                    in_axes_physical.append(None)
                else:
                    assert (
                        len(v) == self.batch_size
                    ), f"{n} in {field.name} is expected to be a scalar or a jnp.Array with len(jnp.Array)=batch_size={self.batch_size}"
                    in_axes_physical.append(0)

            axes.append(in_axes_physical)

        physical_axes = self.PhysicalState(*tuple(axes[0]))
        action_axes = self.Action(*tuple(axes[1]))
        param_axes = self.StaticParams(*tuple(axes[2]))

        return self.EnvProperties(physical_axes, action_axes, param_axes)

    @partial(jax.jit, static_argnums=0)
    def step(self, state, action, env_properties):
        """Computes one simulation step for one batch.

        Args:
            state: The current state of the simulation from which to calculate the next state.
            action: The action to apply to the environment.
            env_properties: Contains action/state constraints and static parameters.

        Returns:
            observation: The gathered observation.
            reward: Amount of reward received for the last step.
            terminated: Flag, indicating if Agent has reached the terminal state.
            truncated: Flag, e.g. indicating if state has gone out of bounds.
            state: New state for the next step.
        """
        assert action.shape == (self.action_dim,), (
            f"The action needs to be of shape (action_dim,) which is "
            + f"{(self.action_dim,)}, but {action.shape} is given"
        )

        physical_state_shape = jnp.array(tree_flatten(state.physical_state)[0]).T.shape
        assert physical_state_shape == (self.physical_state_dim,), (
            "The physical state needs to be of shape (physical_state_dim,) which is "
            + f"{(self.physical_state_dim,)}, but {physical_state_shape} is given"
        )

        state = self._ode_solver_step(state, action, env_properties.static_params)
        obs = self.generate_observation(state, env_properties.physical_constraints)
        reward = self.reward_func(obs, action, env_properties.action_constraints)
        terminated = self.generate_terminated(state, reward)

        # check constraints
        truncated = self.generate_truncated(state, env_properties.physical_constraints)

        return obs, reward, terminated, truncated, state

    @partial(jax.jit, static_argnums=0)
    def vmap_step(self, state, action):
        """JAX jit compiled and vmapped step for batch_size of environment.

        Args:
            state: The current state of the simulation from which to calculate the next
                state (shape=(batch_size, state_dim)).
            action: The action to apply to the environment (shape=(batch_size, action_dim)).
            env_properties: Contains action/state constraints and static parameters.

        Returns:
            observation: The gathered observations (shape=(batch_size,obs_dim)).
            reward: Amount of reward received for the last step (shape=(batch_size,1)).
            terminated: Flag, indicating if Agent has reached the terminal state (shape=(batch_size,1)).
            truncated: Flag, indicating if state has gone out of bounds (shape=(batch_size,state_dim)).
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
        obs, reward, terminated, truncated, state = jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(
            state, action, self.env_properties
        )
        return obs, reward, terminated, truncated, state

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def sim_ahead(self, init_state, actions, env_properties, obs_stepsize, action_stepsize):
        """ """
        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        init_physical_state_shape = jnp.array(tree_flatten(init_state.physical_state)[0]).T.shape
        assert init_physical_state_shape == (self.physical_state_dim,), (
            "The initial physical state needs to be of shape (env.physical_state_dim,) which is "
            + f"{(self.physical_state_dim,)}, but {init_physical_state_shape} is given"
        )

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state, actions, env_properties.static_params, obs_stepsize, action_stepsize
        )

        # generate observations for all timesteps
        observations = jax.vmap(
            self.generate_observation, in_axes=(0, self.in_axes_env_properties.physical_constraints)
        )(states, self.env_properties.physical_constraints)

        # generate rewards - use obs[1:] because obs[0] is observation for the initial state
        reward = jax.vmap(self.reward_func, in_axes=(0, 0, self.in_axes_env_properties.action_constraints))(
            observations[1:],
            jnp.expand_dims(jnp.repeat(actions, int(action_stepsize / obs_stepsize)), 1),
            self.env_properties.action_constraints,
        )

        # generate truncated
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.in_axes_env_properties.physical_constraints))(
            states, self.env_properties.physical_constraints
        )

        # generate terminated
        # delete first state because its initial state of simulation and not relevant for terminated
        states_flatten, struct = tree_flatten(states)
        states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0))(states_without_init_state, reward)

        return observations, reward, truncated, terminated

    @partial(jax.jit, static_argnums=[0, 3, 4])
    def vmap_sim_ahead(self, init_state, actions, obs_stepsize, action_stepsize):

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
        observations, rewards, truncated, terminated = jax.vmap(
            self.sim_ahead, in_axes=(0, 0, self.in_axes_env_properties, None, None)
        )(init_state, actions, self.env_properties, obs_stepsize, action_stepsize)

        return observations, rewards, truncated, terminated

    @property
    @abstractmethod
    def obs_description(self):
        """Returns a list of state names of all states in the observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def default_reward_func(self, obs, action):
        """Returns the default reward function of the environment."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, state):
        """Returns observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, state):
        """Returns truncated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, state, reward):
        """Returns terminated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_solver_step(self, state, action, static_params):
        """Computes the next state by simulating one step.

        Args:
            state: The current state of the simulation from which to calculate the next
                state with shape=(state_dim,).
            action: The action to apply to the environment with shape=(action_dim,).
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            next_state: The computed state after the one step simulation.
        """
        return

    @abstractmethod
    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial values."""
        return
