import jax
import jax.numpy as jnp
import numpy as np
import jax_dataclasses as jdc
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
from functools import partial
import chex
from abc import ABC
from exciting_environments import spaces
from exciting_environments.registration import make


class GymWrapper(ABC):

    def __init__(
        self,
        env,
        control_state=None,
        generate_reward=None,
        generate_terminated=None,
        generate_truncated=None,
        ref_params=None,
    ):

        self.env = env

        if control_state is None:
            print(f"No chosen control state in the GymWrapper. Control state is set to {self.env.control_state}.")
            self.control_state = self.env.control_state
        else:
            assert type(control_state) == list, f"Control state has to be a list."
            for i in control_state:
                assert i in list(
                    self.env.PhysicalState.__match_args__
                ), f"Given control state {i} is no valid physical state {list(self.env.PhysicalState.__match_args__)}."
            self.control_state = control_state
            self.env.control_state = control_state

        self.ref_gen = False

        _, init_state = self.env.vmap_reset()

        if not ref_params:
            self.ref_params = {
                "hold_steps_min": 10,
                "hold_steps_max": 1000,
            }
        self.reference_hold_steps = jnp.zeros((self.env.batch_size, 1))

        self.state = tree_flatten(init_state)[0]
        self.state_tree_struct = tree_structure(init_state)

        if not generate_reward:
            self.generate_reward = self.env.generate_reward
        if not generate_truncated:
            self.generate_truncated = self.env.generate_truncated
        if not generate_terminated:
            self.generate_terminated = self.env.generate_terminated

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
            truncated: Flag, indicating if state has gone out of bounds (shape=(batch_size,state)).
        """

        obs, reward, terminated, truncated, self.state, self.reference_hold_steps = self.gym_step(
            action, self.state, self.reference_hold_steps
        )

        return obs, reward, terminated, truncated

    @partial(jax.jit, static_argnums=0)
    def gym_step(self, action, state, reference_hold_steps):
        """Jax Jit compiled simulation step using the step function provided by the environment.

        Args:
            action: The action to apply to the environment.
            state: The state from which to calculate state for the next step.

        Returns:
            Multiple Outputs:

            observation: The gathered observations.
            reward: Amount of reward received for the last step.
            terminated: Flag, indicating if Agent has reached the terminal state.
            truncated: Flag, indicating if state has gone out of bounds.
            state: New state for the next step.
        """

        # transform array to dataclass defined in environment
        state = tree_unflatten(self.state_tree_struct, state)

        obs, state = self.env.vmap_step(state, action)

        # update reference
        if len(self.control_state) and self.ref_gen:
            state, reference_hold_steps = jax.vmap(self.update_ref, in_axes=(0, self.env.in_axes_env_properties, 0))(
                state, self.env.env_properties, reference_hold_steps
            )

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, self.env.in_axes_env_properties))(
            state, action, self.env.env_properties
        )

        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, self.env.in_axes_env_properties))(
            state, reward, self.env.env_properties
        )
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.env.in_axes_env_properties))(
            state, self.env.env_properties
        )
        # transform dataclass to array
        state = tree_flatten(state)[0]

        return obs, reward, terminated, truncated, state, reference_hold_steps

    def reset(
        self, rng_env: chex.PRNGKey = None, rng_ref: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None
    ):
        """Resets environment to random or passed initial state and can reset reference generator."""

        if initial_state is not None:
            try:
                _, _ = self.env.vmap_reset(initial_state=tree_unflatten(self.state_tree_struct, initial_state))
            except:
                print("initial_state should have the same structure as tree_flatten(self.env.vmap_init_state()")
            obs, state = self.env.vmap_reset(initial_state=tree_unflatten(self.state_tree_struct, initial_state))
        else:
            _, state = self.env.vmap_reset(rng_env)

        if rng_ref is not None:
            if len(rng_ref.shape) == 1:
                key = jax.random.split(rng_ref, num=self.env.batch_size)
            else:
                key = rng_ref
                assert rng_ref.shape[0] == self.env.batch_size

            with jdc.copy_and_mutate(state, validate=False) as state:
                state.PRNGKey = key

            self.ref_gen = True
            state, self.reference_hold_steps = jax.vmap(
                self.generate_new_ref, in_axes=(0, self.env.in_axes_env_properties, 0)
            )(state, self.env.env_properties, jnp.zeros(self.env.batch_size))
        else:
            self.ref_gen = False
            print("Since no PRNGKey for reference was provided, reference generation is deactivated.")

        self.state = tree_flatten(state)[0]
        obs = jax.vmap(self.env.generate_observation, in_axes=(0, self.env.in_axes_env_properties))(
            state, self.env.env_properties
        )
        return obs, {}

    def update_ref(self, state, env_properties, hold_steps):
        state, hold_steps = jax.lax.cond(
            hold_steps[0] == 0, self.generate_new_ref, lambda a, b, c: (a, c), state, env_properties, hold_steps
        )
        hold_steps += -1
        return state, hold_steps

    def generate_new_ref(self, state, env_properties, hold_steps):
        with jdc.copy_and_mutate(state, validate=False) as new_state:
            init = self.env.init_state(env_properties, state.PRNGKey)
            for name in self.control_state:
                setattr(new_state.reference, name, getattr(init.physical_state, name))

            key, subkey = jax.random.split(init.PRNGKey)

            hold_steps = jax.random.randint(
                subkey,
                minval=self.ref_params["hold_steps_min"],
                maxval=self.ref_params["hold_steps_max"],
                shape=(1,),
            )
            new_state.PRNGKey = key
        return new_state, hold_steps

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
