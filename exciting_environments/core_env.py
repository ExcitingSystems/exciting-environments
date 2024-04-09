import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces
import diffrax
from collections import OrderedDict


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

    def __init__(self, batch_size: int, tau: float = 1e-4, solver=diffrax.Euler(), reward_func=None):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration.
            tau(float): Duration of one control step in seconds. Default: 1e-4.
        """
        self.batch_size = batch_size
        self.tau = tau
        self._solver = solver

        if reward_func:
            if self._test_rew_func(reward_func):
                self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

    # @property
    # def batch_size(self):
    #     """Returns the batch size of the environment setup."""
    #     return self._batch_size

    @property
    def default_reward_function(self):
        """Returns the default reward function for the given environment."""
        return self.default_reward_func

    # @batch_size.setter
    # def batch_size(self, batch_size):
    #     # If batchsize change, update the corresponding dimension
    #     self._batch_size = batch_size

    def sim_paras(self, static_params_, env_state_constraints_, env_max_actions_):
        """Creates or updates static parameters to fit batch_size.

        Creates/Updates:
            params : Model Parameters.
        """
        static_params = static_params_.copy()
        for key, value in static_params.items():
            if jnp.isscalar(value):
                static_params[key] = jnp.full((self.batch_size), value)
                # self.static_para_dims[key] = None
            # elif jnp.all(value == value[0]):
            #     self.static_params[key] = jnp.full(
            #         (self.batch_size, 1), value[0])
            else:
                assert len(
                    value) == self.batch_size, f"{key} is expected to be a scalar or a list with len(list)=batch_size"
                static_params[key] = jnp.array(value)
                # self.static_para_dims[key] = 0

        env_state_constraints = env_state_constraints_.copy()
        for key, value in env_state_constraints.items():
            if jnp.isscalar(value):
                env_state_constraints[key] = jnp.full((self.batch_size), value)
                # self.static_para_dims[key] = None
            # elif jnp.all(value == value[0]):
            #     self.static_params[key] = jnp.full(
            #         (self.batch_size, 1), value[0])
            else:
                assert len(
                    value) == self.batch_size, f"Constraint of {key} is expected to be a scalar or a list with len(list)=batch_size"
                env_state_constraints[key] = jnp.array(value)
                # self.static_para_dims[key] = 0

        env_max_actions = env_max_actions_.copy()
        for key, value in env_max_actions.items():
            if jnp.isscalar(value):
                env_max_actions[key] = jnp.full((self.batch_size), value)
                # self.static_para_dims[key] = None
            # elif jnp.all(value == value[0]):
            #     self.static_params[key] = jnp.full(
            #         (self.batch_size, 1), value[0])
            else:
                assert len(
                    value) == self.batch_size, f"Constraint of {key} is expected to be a scalar or a list with len(list)=batch_size"
                env_max_actions[key] = jnp.array(value)
                # self.static_para_dims[key] = 0

        return static_params, OrderedDict(env_state_constraints), OrderedDict(env_max_actions)

    # def solver(self):
    #     """Returns the current solver of the environment setup."""
    #     return self._solver

    # @solver.setter
    # def solver(self, solver):
    #     # TODO:check if solver exists in diffrax ?
    #     self._solver = solver

    def _test_rew_func(self, func):
        """Checks if passed reward function is compatible with given environment.

        Args:
            func(function): Reward function to test.

        Returns:
            compatible(bool): Environment compatibility.
        """
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

    @partial(jax.jit, static_argnums=0)
    def step(self, action, states):
        """Addtional function in step execution to enable JAX jit.

        Args:
            states(ndarray(float)): State Matrix (shape=(batch_size,states)).
            action_norm(ndarray(float)): Action Matrix (shape=(batch_size,actions)).


        Returns:
            Multiple Outputs:

            observation(ndarray(float)): Observation/State Matrix (shape=(batch_size,states)).

            reward(ndarray(float)): Amount of reward received for the last step (shape=(batch_size,1)).

            terminated(bool): Flag, indicating if Agent has reached the terminal state.

            truncated(ndarray(bool)): Flag, indicating if state has gone out of bounds (shape=(batch_size,states)).

            {}: An empty dictionary for consistency with the OpenAi Gym interface.

        """
        # ode step
        states = jax.vmap(self._ode_exp_euler_step)(
            states, action, self.static_params)

        # observation
        # print(states)
        # print(self.env_state_constraints)
        obs = jax.vmap(self.generate_observation)(
            states, self.env_state_constraints)
        # reward
        reward = jax.vmap(self.reward_func)(
            obs, action, self.env_max_actions).reshape(-1, 1)

        # bound check
        truncated = jax.vmap(self.generate_truncated)(
            states, self.env_state_constraints)
        terminated = jax.vmap(self.generate_terminated)(states, reward)

        return obs, reward, terminated, truncated, states

    @property
    @abstractmethod
    def obs_description(self):
        """Returns a list of state names of all states in the observation (equal to state space)."""
        return self.states_description

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def default_reward_func(self, obs, action):
        """Returns the default RewardFunction of the environment."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_observation(self, states):
        """Returns states."""
        return states

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_truncated(self, states):
        """Returns states."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def generate_terminated(self, states, reward):
        """Returns states."""
        return

    @partial(jax.jit, static_argnums=0)
    @abstractmethod
    def _ode_exp_euler_step(self, states_norm, action_norm, state_normalizer,  action_normalizer, params):
        """Implementation of the system equations in the class with Explicit Euler.

        Args:
            states_norm(ndarray(float)): State Matrix (shape=(batch_size,states)).
            action_norm(ndarray(float)): Action Matrix (shape=(batch_size,actions)).


        Returns:
            states(ndarray(float)): State Matrix (shape=(batch_size,states)).

        """
        return

    @abstractmethod
    def reset(self, initial_values: jnp.ndarray = jnp.array([])):
        return
