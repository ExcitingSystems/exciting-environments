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
        env_properties: jdc.pytree_dataclass,
        tau: float = 1e-4,
        solver=diffrax.Euler(),
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration.
            env_properties(jdc.pytree_dataclass): All parameters and properties of the environment.
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            solver(diffrax.solver): Solver used to compute states for next step.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver
        self.env_properties = env_properties
        self.in_axes_env_properties = self.create_in_axes_dataclass(self.env_properties)

    def create_in_axes_dataclass(self, dataclass):
        with jdc.copy_and_mutate(dataclass, validate=False) as dataclass_in_axes:
            for field in fields(dataclass_in_axes):
                name = field.name
                value = getattr(dataclass_in_axes, name)
                if jdc.is_dataclass(value):
                    setattr(dataclass_in_axes, name, self.create_in_axes_dataclass(value))
                elif jnp.isscalar(value):
                    setattr(dataclass_in_axes, name, None)
                else:
                    assert (
                        len(value) == self.batch_size
                    ), f"{name} is expected to be a scalar a pytree_dataclass or a jnp.Array with len(jnp.Array)=batch_size={self.batch_size}"
                    setattr(dataclass_in_axes, name, 0)
        return dataclass_in_axes

    @partial(jax.jit, static_argnums=0)
    def vmap_step(self, states, actions):
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
        return jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(states, actions, self.env_properties)
