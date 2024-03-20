import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces
from exciting_environments.core_env import CoreEnvironment
import diffrax
from exciting_environments.registration import make


class GymWrapper(ABC):

    def __init__(self, env):

        self.env = env
        self.states = jnp.tile(
            jnp.array(self.env.env_state_initials), (self.env.batch_size, 1))

    @classmethod
    def fromName(cls, env_id: str, **env_kwargs):
        env = make(env_id, **env_kwargs)
        return cls(env)

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

        obs, reward, terminated, truncated, self.states = self.env.step(
            action, self.states)

        return obs, reward, terminated, truncated, {}

    def reset(self, random_key: chex.PRNGKey = None, initial_values: jnp.ndarray = jnp.array([])):

        obs, self.states = self.env.reset(
            random_key=random_key, initial_values=initial_values)

        return obs, {}

    def render(self, *_, **__):
        """
        Update the visualization of the environment.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")

    def close(self):
        """Called when the environment is deleted.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")
