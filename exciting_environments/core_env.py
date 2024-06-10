import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import diffrax
import chex
from functools import partial
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces
from dataclasses import fields
from typing import Callable


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
            batch_size(int): Number of training examples utilized in one iteration.
            physical_constraints(jdc.pytree_dataclass): Constraints of physical states of the environment.
            action_constraints(jdc.pytree_dataclass): Constraints of actions.
            static_params(jdc.pytree_dataclass): Parameters of environment which do not change during simulation.
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            solver(diffrax.solver): Solver used to compute states for next step.
            reward_func(Callable): Reward function for training. Needs observation vector, action and action_constraints as Parameters.
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

    @property
    def default_reward_function(self):
        """Returns the default reward function for the given environment."""
        return self.default_reward_func

    @abstractmethod
    @jdc.pytree_dataclass
    class PhysicalStates:
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
    class Actions:
        pass

    @jdc.pytree_dataclass
    class States:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_states: jdc.pytree_dataclass
        PRNGKey: jax.Array
        optional: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Dataclass used for simulation which contains environment specific dataclasses."""

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

        physical_axes = self.PhysicalStates(*tuple(axes[0]))
        action_axes = self.Actions(*tuple(axes[1]))
        param_axes = self.StaticParams(*tuple(axes[2]))

        return self.EnvProperties(physical_axes, action_axes, param_axes)

    @partial(jax.jit, static_argnums=0)
    def step(self, states, action, env_properties):
        """Computes one simulation step for one batch.

        Args:
            states: The states from which to calculate states for the next step.
            action: The action to apply to the environment.
            env_properties: Contains action/state constraints and static parameter.

        Returns:
            Multiple Outputs:

            observation: The gathered observation.
            reward: Amount of reward received for the last step.
            terminated: Flag, indicating if Agent has reached the terminal state.
            truncated: Flag, e.g. indicating if state has gone out of bounds.
            states: New states for the next step.
        """

        # ode step
        states = self._ode_solver_step(states, action, env_properties.static_params)

        # observation
        obs = self.generate_observation(states, env_properties.physical_constraints)

        # reward
        reward = self.reward_func(obs, action, env_properties.action_constraints)

        # bound check
        truncated = self.generate_truncated(states, env_properties.physical_constraints)

        terminated = self.generate_terminated(states, reward)

        return obs, reward, terminated, truncated, states

    @partial(jax.jit, static_argnums=0)
    def vmap_step(self, action, states):
        """JAX jit compiled and vmapped step for batch_size of environment.

        Args:
            states: The states from which to calculate states for the next step.
            action: The action to apply to the environment.
            env_properties: Contains action/state constraints and static parameters.


        Returns:
            Multiple Outputs:

            observation: The gathered observations (shape=(batch_size,obs_dim)).
            reward: Amount of reward received for the last step (shape=(batch_size,1)).
            terminated: Flag, indicating if Agent has reached the terminal state (shape=(batch_size,1)).
            truncated: Flag, indicating if state has gone out of bounds (shape=(batch_size,states_dim)).
            states: New states for the next step.

        """
        # vmap single operations
        obs, reward, terminated, truncated, states = jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(
            states, action, self.env_properties
        )

        return obs, reward, terminated, truncated, states

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def sim_ahead(self, states, actions, env_properties, obs_stepsize, action_stepsize):

        # compute states trajectory for given actions
        states_t = self._ode_solver_simulate_ahead(
            states, actions, env_properties.static_params, obs_stepsize, action_stepsize
        )

        # generate observations for all timesteps
        obs = jax.vmap(self.generate_observation, in_axes=(0, self.in_axes_env_properties.physical_constraints))(
            states_t, self.env_properties.physical_constraints
        )

        # generate rewards - use obs[1:] because obs[0] is observation for the initial state
        reward = jax.vmap(self.reward_func, in_axes=(0, 0, self.in_axes_env_properties.action_constraints))(
            obs[1:],
            jnp.expand_dims(jnp.repeat(actions, int(action_stepsize / obs_stepsize)), 1),
            self.env_properties.action_constraints,
        )

        # generate truncated
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.in_axes_env_properties.physical_constraints))(
            states_t, self.env_properties.physical_constraints
        )

        # generate terminated
        # delete first state because its initial state of simulation and not relevant for terminated
        states_flatten, struct = tree_flatten(states_t)
        states_t_1 = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0))(states_t_1, reward)

        return obs, reward, truncated, terminated

    @partial(jax.jit, static_argnums=[0, 3, 4])
    def vmap_sim_ahead(self, states, actions, obs_stepsize, action_stepsize):

        # vmap single operations
        obs, rewards, truncated, terminated = jax.vmap(
            self.sim_ahead, in_axes=(0, 0, self.in_axes_env_properties, None, None)
        )(states, actions, self.env_properties, obs_stepsize, action_stepsize)

        return obs, rewards, truncated, terminated

    @property
    @abstractmethod
    def obs_description(self):
        """Returns a list of state names of all states in the observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def default_reward_func(self, obs, action):
        """Returns the default RewardFunction of the environment."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, states):
        """Returns observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, states):
        """Returns truncated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, states, reward):
        """Returns terminated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_solver_step(self, states_norm, action_norm, state_normalizer, action_normalizer, params):
        """Computes states by simulating one step.

        Args:
            states: The states from which to calculate states for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            states: The computed states after the one step simulation.
        """

        return

    @abstractmethod
    def reset(self, rng: chex.PRNGKey = None, initial_states: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial values."""
        return
