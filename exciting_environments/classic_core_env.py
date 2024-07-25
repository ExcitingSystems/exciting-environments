import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import tree_flatten
import diffrax
import chex
from functools import partial
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces
from dataclasses import fields
from typing import Callable
from exciting_environments import CoreEnvironment


class ClassicCoreEnvironment(CoreEnvironment):
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
        env_properties = self.EnvProperties(
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

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
        # denormalize action
        action = action * jnp.array(tree_flatten(env_properties.action_constraints)[0]).T

        # ode step
        states = self._ode_solver_step(states, action, env_properties.static_params)

        # observation
        obs = self.generate_observation(states, env_properties.physical_constraints)

        # # reward
        # reward = self.reward_func(obs, action, env_properties.action_constraints)

        # bound check
        truncated = self.generate_truncated(states, env_properties.physical_constraints)

        # terminated = self.generate_terminated(states, reward)

        return obs, truncated, states

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
