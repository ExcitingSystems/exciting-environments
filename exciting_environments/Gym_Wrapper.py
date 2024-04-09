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
from collections import OrderedDict


class GymWrapper(ABC):

    def __init__(self, env):

        self.env = env
        self.states = OrderedDict([(name, jnp.full((self.env.batch_size), init)) for name, init in zip(
            self.env.env_states_initials.keys(), self.env.env_states_initials.values())])

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.batch_size, len(list(self.env.env_max_actions.values()))), dtype=jnp.float32)

        self.env_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.batch_size, len(list(self.env.env_state_constraints.values()))), dtype=jnp.float32)

    @classmethod
    def fromName(cls, env_id: str, **env_kwargs):
        env = make(env_id, **env_kwargs)
        return cls(env)

    def step(self, actions):
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

        obs, reward, terminated, truncated, self.states = self.gym_step(
            actions, self.states)

        return obs, reward, terminated, truncated, {}

    @partial(jax.jit, static_argnums=0)
    def gym_step(self, actions, states):

        # denormalize action
        actions = actions*jnp.array(list(self.env.env_max_actions.values())).T

        # action shape from array to dict
        actions = {name: actions[:, idx] for name, idx in zip(
            self.env.env_actions_name, range(actions.shape[1]))}

        obs, reward, terminated, truncated, states = self.env.step(
            actions, states)

        return obs, reward, terminated, truncated, states

    def reset(self, random_key: chex.PRNGKey = None, initial_values: jnp.ndarray = jnp.array([])):

        if random_key != None:
            states_mat = self.env_observation_space.sample(
                random_key)*jnp.array(list(self.env.env_state_constraints.values())).T
            self.states = {name: states_mat[:, idx] for name, idx in zip(
                self.env.env_states_name, range(states_mat.shape[1]))}

        else:
            self.states = self.env.reset(initial_values=initial_values)

        obs = self.env.generate_observation(
            self.states, self.env.env_state_constraints)
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

    # def sim_paras(self, env_state_constraints, max_action):
    #     """Creates or updates parameters,variables,spaces,etc. to fit batch_size.

    #     Creates/Updates:
    #         action_space: Space for applied actions.
    #         observation_space: Space for system states.
    #         env_state_normalizer: Environment State normalizer to normalize and denormalize states of the environment to implement physical equations with actual values.
    #         action_normalizer: Action normalizer to normalize and denormalize actions to implement physical equations with actual values.
    #     """
    #     action_space = spaces.Box(
    #         low=-1.0, high=1.0, shape=(self.batch_size, len(max_action)), dtype=jnp.float32)

    #     env_observation_space = spaces.Box(
    #         low=-1.0, high=1.0, shape=(self.batch_size, len(env_state_constraints)), dtype=jnp.float32)

    #     return env_observation_space, action_space
