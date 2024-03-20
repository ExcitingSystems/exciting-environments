import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces
import diffrax


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

    def sim_paras(self, static_params, env_state_constraints, max_action):
        """Creates or updates parameters,variables,spaces,etc. to fit batch_size.

        Creates/Updates:
            params : Model Parameters.
            action_space: Space for applied actions.
            observation_space: Space for system states.
            env_state_normalizer: Environment State normalizer to normalize and denormalize states of the environment to implement physical equations with actual values.
            action_normalizer: Action normalizer to normalize and denormalize actions to implement physical equations with actual values.
        """
        for key, value in static_params.items():
            if jnp.isscalar(value):
                static_params[key] = jnp.full((self.batch_size, 1), value)
                # self.static_para_dims[key] = None
            # elif jnp.all(value == value[0]):
            #     self.static_params[key] = jnp.full(
            #         (self.batch_size, 1), value[0])
            else:
                assert len(
                    value) == self.batch_size, f"{key} is expected to be a scalar or a list with len(list)=batch_size"
                static_params[key] = jnp.array(value).reshape(-1, 1)
                # self.static_para_dims[key] = 0

        env_state_normalizer = env_state_constraints.copy()
        for i in range(len(env_state_constraints)):
            if jnp.isscalar(env_state_constraints[i]):
                env_state_normalizer[i] = jnp.full(
                    self.batch_size, env_state_constraints[i])
            # elif jnp.all(self.state_constraints[i] == self.state_constraints[i][0]):
            #     self.state_normalizer[i] = jnp.full(
            #         self.batch_size, self.state_constraints[i][0])
            else:
                assert len(
                    self.env_state_constraints[i]) == self.batch_size, f"self.constraints entries are expected to be a scalar or a list with len(list)=batch_size"
        env_state_normalizer = jnp.array(env_state_normalizer).transpose()

        action_normalizer = max_action.copy()
        for i in range(len(max_action)):
            if jnp.isscalar(max_action[i]):
                action_normalizer[i] = jnp.full(
                    self.batch_size, max_action[i])
            # elif jnp.all(self.max_action[i] == self.max_action[i][0]):
            #     self.action_normalizer[i] = jnp.full(
            #         self.batch_size, self.max_action[i][0])
            else:
                assert len(
                    max_action[i]) == self.batch_size, f"self.max_action entries are expected to be a scalar or a list with len(list)=batch_size"
        action_normalizer = jnp.array(action_normalizer).transpose()

        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, len(max_action)), dtype=jnp.float32)

        env_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, len(env_state_constraints)), dtype=jnp.float32)

        return static_params, env_state_normalizer, action_normalizer, env_observation_space, action_space
    # @property
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
    def step(self, action_norm, states):
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
        states = jax.vmap(self._ode_exp_euler_step)(states, action_norm, self.env_state_normalizer,
                                                    self.action_normalizer, self.static_params)

        # observation
        obs = jax.vmap(self.generate_observation)(states)

        # reward
        reward = jax.vmap(self.reward_func)(obs, action_norm).reshape(-1, 1)

        # bound check
        truncated = jax.vmap(self.generate_truncated)(states)
        terminated = jax.vmap(self.generate_terminated)(states, reward)

        return obs, reward, terminated, truncated, states

    def reset(self, random_key: chex.PRNGKey = None, initial_values: jnp.ndarray = jnp.array([])):
        """Reset environment to chosen initial states. If no parameters are passed the states will be reset to defined default states.

        Args:
            random_key(chex.PRNGKey): If passed, environment states will be set to random initial states depending on the PRNGKey value.
            initial_values(ndarray): If passed, environment states will be set to passed initial_values.


        Returns:
            Multiple Outputs:

            observation(ndarray(float)): Observation/State Matrix (shape=(batch_size,states)).

            {}: An empty dictionary for consistency with the OpenAi Gym interface.
        """
        if random_key != None:
            states = self.env_observation_space.sample(random_key)
        elif initial_values.any() != False:
            assert initial_values.shape[
                0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(
                self.obs_description), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            assert self.env_observation_space.contains(
                initial_values), f"values of initial states are out of bounds"
            states = initial_values
        else:
            states = jnp.tile(
                jnp.array(self.env_state_initials), (self.batch_size, 1))

        obs = self.generate_observation(states)

        return obs, states

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
