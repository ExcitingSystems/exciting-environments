import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
from functools import partial
import chex
from abc import ABC
from exciting_environments import spaces
from exciting_environments.registration import make


class GymWrapper(ABC):

    def __init__(self, env):

        self.env = env
        self.states = jnp.array(tree_flatten(self.env.init_states())[0]).T
        self.states_tree_struct = tree_structure(self.env.init_states())
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
            Multiple Outputs:

            observation: The gathered observation (shape=(batch_size,obs_dim)).
            reward: Amount of reward received for the last step (shape=(batch_size,1)).
            terminated: Flag, indicating if Agent has reached the terminal state (shape=(batch_size,1)).
            truncated: Flag, indicating if state has gone out of bounds (shape=(batch_size,states)).
        """

        obs, reward, terminated, truncated, self.states = self.gym_step(action, self.states)

        return obs, reward, terminated, truncated

    @partial(jax.jit, static_argnums=0)
    def gym_step(self, action, states):
        """Jax Jit compiled simulation step using the step function provided by the environment.

        Args:
            action: The action to apply to the environment.
            states: The states from which to calculate states for the next step.

        Returns:
            Multiple Outputs:

            observation: The gathered observations.
            reward: Amount of reward received for the last step.
            terminated: Flag, indicating if Agent has reached the terminal state.
            truncated: Flag, indicating if state has gone out of bounds.
            states: New states for the next step.
        """
        # denormalize action
        action = action * jnp.array(tree_flatten(self.env.env_properties.action_constraints)[0]).T

        # transform array to dataclass defined in environment
        states = tree_unflatten(self.states_tree_struct, states.T)

        obs, truncated, states = self.env.vmap_step(states, action)

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, self.env.in_axes_env_properties.action_constraints))(
            obs, action, self.env.env_properties.action_constraints
        )

        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0))(states, reward)
        # transform dataclass to array
        states = jnp.array(tree_flatten(states)[0]).T

        return obs, reward, terminated, truncated, states

    def generate_reward(self, obs, action, action_constraints):
        reward = self.env.reward_func(obs, action, action_constraints)
        return reward

    def generate_terminated(self, states, reward):
        terminated = self.env.generate_terminated(states, reward)
        return terminated

    def reset(self, rng: chex.PRNGKey = None, initial_states: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial states."""
        # TODO: rng

        if initial_states is not None:
            assert (
                jnp.array(tree_flatten(self.env.init_states())[0]).T.shape == initial_states.shape
            ), f"initial_states should have shape={jnp.array(tree_flatten(self.env.init_states())[0]).T.shape}"
            obs, states = self.env.reset(initial_states=tree_unflatten(self.states_tree_struct, initial_states.T))
        else:
            obs, states = self.env.reset()
        self.states = jnp.array(tree_flatten(states)[0]).T
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
