import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from abc import ABC
from abc import abstractmethod


class CoreEnvironment(ABC):
    """
    Description:
        Core Structure of provided Environments.

    State Variables:
        Each environment has got a list of state variables that are defined by the physical system represented.

        Example:
            ``['theta', 'omega']``

    Action Variable:
        Each environment has got an action which is applied to the physical system represented.

        Example:
            ``['torque']``

    Observation Space(State Space):
        Type: Box()
            The Observation Space is nothing but the State Space of the pyhsical system.
            This Space is a normalized, continious, multidimensional box in [-1,1].

    Action Space:
        Type: Box()
            The action space of the environments are the action spaces of the physical systems.
            This Space is a continious, multidimensional box. 


    Initial State:
        Initial state values depend on the physical system.

    """

    def __init__(self, batch_size: int, tau: float = 1e-4):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration.
            tau(float): Duration of one control step in seconds. Default: 1e-4.
        """
        self.batch_size = batch_size
        self.tau = tau

    @property
    def batch_size(self):
        """Returns the batch size of the environment setup."""
        return self._batch_size

    @property
    def default_reward_function(self):
        """Returns the default reward function for the given environment."""
        return self.default_reward_func

    @batch_size.setter
    def batch_size(self, batch_size):
        # If batchsize change, update the corresponding dimension
        self._batch_size = batch_size
        self.update_batch_dim()

    @partial(jax.jit, static_argnums=0)
    def _static_generate_observation(self, states):
        return states

    def generate_observation(self):
        """Returns the states of the environment."""
        return self.states

    def _test_rew_func(self, func):
        try:
            out = func(
                jnp.zeros([self.batch_size, int(len(self.obs_description))]))
        except:
            raise Exception(
                "Reward function should be using obs matrix as only parameter")
        try:
            if out.shape != (self.batch_size, 1):
                raise Exception(
                    "Reward function should be returning vector in shape (batch_size,1)")
        except:
            raise Exception(
                "Reward function should be returning vector in shape (batch_size,1)")
        return True

    def render(self, *_, **__):
        """
        Update the visualization of the environment.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")

    def step(self, action):
        """Perform one simulation step of the environment with an action of the action space.

        Args:
            action: Action to play on the environment.

        Returns:
            Multiple Outputs:

            observation(ndarray(float)): Observation/State Matrix (shape=(batch_size,states)).

            reward(ndarray(float)): Amount of reward received for the last step (shape=(batch_size,1)).

            terminated(bool): Flag, indicating if Agent has reached the terminal state.

            truncated(ndarray(bool)): Flag, indicating if state has gone out of bounds (shape=(batch_size,states)).

            {}: An empty dictionary for consistency with the OpenAi Gym interface.
        """

        obs, reward, terminated, truncated, self.states = self._step_static(
            self.states, action)

        return obs, reward, terminated, truncated, {}

    @partial(jax.jit, static_argnums=0)
    def _step_static(self, states, action_norm):
        """Addtional function in step execution to enable JAX jit"""
        # ode step
        states = self._ode_exp_euler_step(states, action_norm)

        # observation
        obs = self._static_generate_observation(states)

        # reward
        reward = self.reward_func(obs, action_norm)

        # bound check
        truncated = (jnp.abs(states) > 1)
        terminated = reward == 0

        return obs, reward, terminated, truncated, states

    def close(self):
        """Called when the environment is deleted.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")

    @property
    @abstractmethod
    def obs_description(self):
        """Returns a list of state names of all states in the observation (equal to state space)."""
        return self.states_description

    @property
    @abstractmethod
    def states_description(self):
        """Returns a list of state names of all states in the states space."""
        return np.array(["state1_name", "..."])

    @property
    @abstractmethod
    def action_description(self):
        """Returns the name of the action."""
        return np.array(["action_name"])

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def default_reward_func(self, obs, action):
        """Returns the default RewardFunction of the environment."""
        return

    @abstractmethod
    def update_batch_dim(self):
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_exp_euler_step(self, states_norm, action_norm):
        """Implementation of the system equations in the class with Explicit Euler.

        Returns:
            states(ndarray(float)): State Matrix (shape=(batch_size,states)).

        """
        return

    @abstractmethod
    def reset(self, random_key: chex.PRNGKey = False, initial_values: np.ndarray = None):
        """
            Reset the environment, return initial observation vector.
            Options:
                - Observation/State Space gets a random initial sample
                - Initial Observation/State Space is set to initial_values array

        """
        return
