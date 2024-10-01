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
    Core Structure of provided Environments. Any new environments needs to inherit from this class
    and implement its abstract properties and methods.

    The simulations are all done with physical state space models. That means that the underlying description
    of the system is given through the differential equation describing the relationship between
    the change of the physical state x(t) w.r.t. the time as a function of the physical state and the
    input/action u(t) applied:

    dx(t)/dt = f(x(t), u(t)).

    The actual outputs of these simulations are discretized from this equation through the use of
    ODE solvers.

    NOTE: There is a difference between the state of the environment and the physical state x(t)
    of the underlying system. The former can also hold various helper variables such as PRNGKeys
    for stochastic environments, while the latter is reserved for the actual physical state of the
    ODE. The physical state is only a part of the full state.
    """

    def __init__(
        self,
        batch_size: int,
        env_properties: jdc.pytree_dataclass,
        tau: float = 1e-4,
        solver=diffrax.Euler(),
    ):
        """Initialization of an environment.

        Args:
            batch_size (int): Number of parallel environment simulations.
            env_properties(jdc.pytree_dataclass): All parameters and properties of the environment.
            tau (float): Duration of one control step in seconds. Default: 1e-4.
            solver (diffrax.solver): ODE solver used to approximate the ODE solution.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver
        self.env_properties = env_properties
        self.in_axes_env_properties = self.create_in_axes_dataclass(env_properties)

    def create_in_axes_dataclass(self, dataclass):
        with jdc.copy_and_mutate(dataclass, validate=False) as dataclass_in_axes:
            for field in fields(dataclass_in_axes):
                name = field.name
                value = getattr(dataclass_in_axes, name)
                if value == None:
                    setattr(dataclass_in_axes, name, None)
                elif jdc.is_dataclass(value):
                    setattr(dataclass_in_axes, name, self.create_in_axes_dataclass(value))
                elif jnp.isscalar(value):
                    setattr(dataclass_in_axes, name, None)
                else:
                    assert (
                        len(value) == self.batch_size
                    ), f"{name} is expected to be a scalar, a pytree_dataclass or a jnp.Array with len(jnp.Array)=batch_size={self.batch_size}"
                    setattr(dataclass_in_axes, name, 0)
        return dataclass_in_axes

    @partial(jax.jit, static_argnums=0)
    def vmap_step(self, state, action):
        """Computes one JAX-JIT compiled simulation step for multiple (batch_size) batches.

        Args:
            state: The current state of the simulation from which to calculate the next
                state (shape=(batch_size, state_dim)).
            action: The action to apply to the environment (shape=(batch_size, action_dim)).

        Returns:
            observation: The gathered observations (shape=(batch_size,obs_dim)).
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
        obs, state = jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(
            state, action, self.env_properties
        )
        return obs, state

    @partial(jax.jit, static_argnums=[0, 3, 4])
    def vmap_sim_ahead(self, init_state, actions, obs_stepsize, action_stepsize):
        """Computes multiple JAX-JIT compiled simulation steps for multiple (batch_size) batches.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[1] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Args:
            init_state: The initial state of the simulation
            actions: A set of actions to be applied to the environment, the value changes every
            obs_stepsize: The sampling time for the observations
            action_stepsize: The time between changes in the input/action
        """
        assert (
            obs_stepsize <= action_stepsize
        ), "The action stepsize should be greater or equal to the observation stepsize."
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
        observations, states, last_state = jax.vmap(
            self.sim_ahead, in_axes=(0, 0, self.in_axes_env_properties, None, None)
        )(
            init_state, actions, self.env_properties, obs_stepsize, action_stepsize
        )  # rewards, truncated, terminated,

        return observations, states, last_state  # rewards, truncated, terminated,

    def vmap_generate_rew_trunc_term_ahead(self, states, actions):
        assert actions.ndim == 3, "The actions need to have three dimensions: (batch_size, n_action_steps, action_dim)"
        assert (
            actions.shape[0] == self.batch_size
        ), f"The first dimension does not correspond to the batch size which is {self.batch_size}, but {actions.shape[0]} is given"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"
        reward, truncated, terminated = jax.vmap(self.generate_rew_trunc_term_ahead, in_axes=(0, 0))(states, actions)

        return reward, truncated, terminated
