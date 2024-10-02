import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import tree_flatten, tree_unflatten
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
    ):
        """Initialization of an environment.

        Args:
            batch_size (int): Number of parallel environment simulations.
            physical_constraints (jdc.pytree_dataclass): Constraints of the physical state of the environment.
            action_constraints (jdc.pytree_dataclass): Constraints of the input/action.
            static_params (jdc.pytree_dataclass): Parameters of environment which do not change during simulation.
            tau (float): Duration of one control step in seconds. Default: 1e-4.
            solver (diffrax.solver): ODE solver used to approximate the ODE solution.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver
        env_properties = self.EnvProperties(
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )

        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)
        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    @abstractmethod
    @jdc.pytree_dataclass
    class PhysicalState:
        """The physical state x(t) of the underlying system and whose derivative
        w.r.t. time is described in the underlying ODE.

        The values stored in this dataclass are expected to be actual physical values
        that are unnormalized.
        """

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Additions:
        """Additional information that can change from iteration to iteration and that is
        stored in the state of the system"""

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class StaticParams:
        """Static parameters of the environment that stay constant during simulation.
        This could be the length of a pendulum, the capacitance of a capacitor,
        the mass of a specific element and similar..
        """

        pass

    @abstractmethod
    @jdc.pytree_dataclass
    class Action:
        """The input/action applied to the environment that is used to influence the
        dynamics from the outside.

        The values expected as input to the environment are to be normalized in [-1, 1]
        for each dimension.

        TODO: This kind of normalization is only really feasible for box constraints. This
        might be subject to change in the future.
        """

        pass

    @jdc.pytree_dataclass
    class State:
        """Stores the state of the environment."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        additions: jdc.pytree_dataclass
        reference: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Stores properties of the environment that stay constant during simulation."""

        physical_constraints: jdc.pytree_dataclass
        action_constraints: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    @partial(jax.jit, static_argnums=0)
    def step(self, state, action, env_properties):
        """Computes one JAX-JIT compiled simulation step for one batch.

        Args:
            state: The current state of the simulation from which to calculate the next state.
            action: The action to apply to the environment.
            env_properties: Contains action/state constraints and static parameters.

        Returns:
            observation: The gathered observation.
            state: New state for the next step.
        """
        # reward: Amount of reward received for the last step.
        # terminated: Flag, indicating if Agent has reached the terminal state.
        # truncated: Flag, e.g. indicating if state has gone out of bounds.

        assert action.shape == (self.action_dim,), (
            f"The action needs to be of shape (action_dim,) which is "
            + f"{(self.action_dim,)}, but {action.shape} is given"
        )

        physical_state_shape = jnp.array(tree_flatten(state.physical_state)[0]).T.shape

        if physical_state_shape[0] == 1:
            # allow batch_dim == 1
            physical_state_shape = physical_state_shape[1:]

        # assert physical_state_shape == (self.physical_state_dim,), (
        #     "The physical state needs to be of shape (physical_state_dim,) which is "
        #     + f"{(self.physical_state_dim,)}, but {physical_state_shape} is given"
        # ) -> TODO problems for FluidTank

        # denormalize action
        action = action * jnp.array(tree_flatten(env_properties.action_constraints)[0]).T

        state = self._ode_solver_step(state, action, env_properties.static_params)
        obs = self.generate_observation(state, env_properties)

        return obs, state

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def sim_ahead(self, init_state, actions, env_properties, obs_stepsize, action_stepsize):
        """Computes multiple JAX-JIT compiled simulation steps for one batch.

        The length of the set of inputs together with the action_stepsize determine the
        overall length of the simulation -> overall_time = actions.shape[0] * action_stepsize
        The actions are interpolated with zero order hold inbetween their values.

        Args:
            init_state: The initial state of the simulation
            actions: A set of actions to be applied to the environment, the value changes every
            action_stepsize (shape=(n_action_steps, action_dim))
            env_properties: The constant properties of the simulation
            obs_stepsize: The sampling time for the observations
            action_stepsize: The time between changes in the input/action
        """

        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        init_physical_state_shape = jnp.array(tree_flatten(init_state.physical_state)[0]).T.shape
        assert init_physical_state_shape == (self.physical_state_dim,), (
            "The initial physical state needs to be of shape (env.physical_state_dim,) which is "
            + f"{(self.physical_state_dim,)}, but {init_physical_state_shape} is given"
        )

        # denormalize actions
        actions = actions * jnp.array(tree_flatten(env_properties.action_constraints)[0]).T

        # compute states trajectory for given actions
        states = self._ode_solver_simulate_ahead(
            init_state, actions, env_properties.static_params, obs_stepsize, action_stepsize
        )

        # generate observations for all timesteps
        observations = jax.vmap(self.generate_observation, in_axes=(0, None))(states, env_properties)

        # delete first state because its initial state of simulation and not relevant for terminated
        states_flatten, struct = tree_flatten(states)

        states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])

        # reward = jax.vmap(self.generate_reward, in_axes=(0, 0, None))(
        #     states_without_init_state,
        #     jnp.expand_dims(jnp.repeat(actions, int(action_stepsize / obs_stepsize)), 1),
        #     env_properties,
        # )
        # reward = 0

        # generate truncated
        # truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.in_axes_env_properties))(
        #     states, self.env_properties
        # )

        # generate terminated

        # get last state so that the simulation can be continued from the end point
        last_state = tree_unflatten(struct, jnp.array(states_flatten)[:, -1:])

        # terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, self.in_axes_env_properties))(
        #     states_without_init_state, reward, self.env_properties
        # )

        return observations, states, last_state  # , reward, truncated, terminated

    def generate_rew_trunc_term_ahead(self, states, actions):

        assert actions.ndim == 2, "The actions need to have two dimensions: (n_action_steps, action_dim)"
        assert (
            actions.shape[-1] == self.action_dim
        ), f"The last dimension does not correspond to the action dim which is {self.action_dim}, but {actions.shape[-1]} is given"

        actions = actions * jnp.array(tree_flatten(self.env_properties.action_constraints)[0]).T

        states_flatten, struct = tree_flatten(states)

        states_without_init_state = tree_unflatten(struct, jnp.array(states_flatten)[:, 1:])

        reward = jax.vmap(self.generate_reward, in_axes=(0, 0, self.in_axes_env_properties))(
            states_without_init_state,
            jnp.expand_dims(
                jnp.repeat(actions, int((jnp.array(states_flatten).shape[1] - 1) / actions.shape[0])), 1
            ),  #
            self.env_properties,
        )
        truncated = jax.vmap(self.generate_truncated, in_axes=(0, self.in_axes_env_properties))(
            states, self.env_properties
        )
        terminated = jax.vmap(self.generate_terminated, in_axes=(0, 0, self.in_axes_env_properties))(
            states_without_init_state, reward, self.env_properties
        )
        return reward, truncated, terminated

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_reward(self, state, action, env_properties):
        """Returns the default RewardFunction of the environment."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, state, env_properties):
        """Returns observation."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, state, env_properties):
        """Returns truncated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_solver_step(self, state, action, static_params):
        """Computes state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """

        return

    @partial(jax.jit, static_argnums=[0, 4, 5])
    @abstractmethod
    def _ode_solver_simulate_ahead(self, init_state, actions, static_params, obs_stepsize, action_stepsize):
        """Computes states by simulating a trajectory with given actions."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def init_state(self):
        """Returns default initial state for all batches."""
        return

    @abstractmethod
    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial values."""
        return
