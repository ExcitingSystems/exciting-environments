from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import chex

from exciting_environments.registration import make


class GymWrapper:

    def __init__(self, env):

        self.env = env
        self.state = jnp.array(tree_flatten(self.env.init_state())[0]).T
        self.state_tree_struct = tree_structure(self.env.init_state())

        # TODO action and observation space for gym interface
        # self.action_space = spaces.Box(
        #     low=-1.0, high=1.0, shape=(self.env.batch_size, len(list(self.env.env_max_actions.values()))), dtype=jnp.float32)

        # self.env_observation_space = spaces.Box(
        #     low=-1.0, high=1.0, shape=(self.env.batch_size, len(list(self.env.env_state_constraints.values()))), dtype=jnp.float32)

    @classmethod
    def from_name(cls, env_id: str, **env_kwargs):
        """Creates GymWrapper with environment based on passed env_id."""
        env = make(env_id, **env_kwargs)
        return cls(env)

    def step(self, action):
        """Performs one simulation step of the environment with a given action.

        Args:
            action: Action to play on the environment.

        Returns:
            observation: The gathered observation (shape=(batch_size,obs_dim)).
            reward: Amount of reward received for the last step (shape=(batch_size,1)).
            terminated: Flag, indicating if Agent has reached the terminal state (shape=(batch_size,1)).
            truncated: Flag, indicating if state has gone out of bounds (shape=(batch_size,state)).
        """

        obs, reward, terminated, truncated, self.state = self.gym_step(action, self.state)

        return obs, reward, terminated, truncated

    @partial(jax.jit, static_argnums=0)
    def gym_step(self, action, state):
        """Jax Jit compiled simulation step using the step function provided by the environment.

        Args:
            action: The action to apply to the environment.
            state: The state from which to calculate state for the next step.

        Returns:
            observation: The gathered observations.
            reward: Amount of reward received for the last step.
            terminated: Flag, indicating if Agent has reached the terminal state.
            truncated: Flag, indicating if state has gone out of bounds.
            state: New state for the next step.
        """
        # transform array to dataclass defined in environment
        state = tree_unflatten(self.state_tree_struct, state.T)

        obs, reward, terminated, truncated, state = self.env.vmap_step(state, action)

        # transform dataclass to array
        state = jnp.array(tree_flatten(state)[0]).T

        return obs, reward, terminated, truncated, state

    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial state."""
        # TODO: rng

        if initial_state is not None:
            assert (
                jnp.array(tree_flatten(self.env.init_state())[0]).T.shape == initial_state.shape
            ), f"initial_state should have shape={jnp.array(tree_flatten(self.env.init_state())[0]).T.shape}"
            obs, state = self.env.reset(initial_state=tree_unflatten(self.state_tree_struct, initial_state.T))
        else:
            obs, state = self.env.reset()
        self.state = jnp.array(tree_flatten(state)[0]).T
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
