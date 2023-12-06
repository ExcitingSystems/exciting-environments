import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from abc import ABC
from abc import abstractmethod
from exciting_environments import spaces


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

    def __init__(self, batch_size: int, tau: float = 1e-4, reward_func=None):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration.
            tau(float): Duration of one control step in seconds. Default: 1e-4.
        """
        self.batch_size = batch_size
        self.tau = tau
        if reward_func:
            if self._test_rew_func(reward_func):
                self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

    @property
    def batch_size(self):
        """Returns the batch size of the environment setup."""
        return self._batch_size

    @property
    def default_reward_function(self):
        """Returns the default reward function for the given environment."""
        return self.default_reward_func

    @batch_size.setter
    def batch_size(self, batch_size):
        # If batchsize change, update the corresponding dimension
        self._batch_size = batch_size
        self._update_batch_dim()

    def _update_batch_dim(self):
        """Creates or updates parameters,variables,spaces,etc. to fit batch_size.

        Creates/Updates:
            params : Model Parameters.
            action_space: Space for applied actions.
            observation_space: Space for system states.
            state_normalizer: State normalizer to normalize and denormalize states to implement physical equations with actual values.
            action_normalizer: Action normalizer to normalize and denormalize actions to implement physical equations with actual values.
            states: System states.

        """
        for key, value in self.params.items():
            if jnp.isscalar(value):
                self.params[key] = jnp.full((self.batch_size, 1), value)
            elif jnp.all(value == value[0]):
                self.params[key] = jnp.full((self.batch_size, 1), value[0])
            else:
                assert len(
                    value) == self.batch_size, f"{key} is expected to be a scalar or a list with len(list)=batch_size"
                self.params[key] = jnp.array(value).reshape(-1, 1)

        self.state_normalizer = self.state_constraints
        for i in range(len(self.state_constraints)):
            if jnp.isscalar(self.state_constraints[i]):
                self.state_normalizer[i] = jnp.full(
                    self.batch_size, self.state_constraints[i])
            elif jnp.all(self.state_constraints[i] == self.state_constraints[i][0]):
                self.state_normalizer[i] = jnp.full(
                    self.batch_size, self.state_constraints[i][0])
            else:
                assert len(
                    self.state_constraints[i]) == self.batch_size, f"self.constraints entries are expected to be a scalar or a list with len(list)=batch_size"
        self.state_normalizer = jnp.array(self.state_normalizer).transpose()

        self.action_normalizer = self.max_action
        for i in range(len(self.max_action)):
            if jnp.isscalar(self.max_action[i]):
                self.action_normalizer[i] = jnp.full(
                    self.batch_size, self.max_action[i])
            elif jnp.all(self.max_action[i] == self.max_action[i][0]):
                self.action_normalizer[i] = jnp.full(
                    self.batch_size, self.max_action[i][0])
            else:
                assert len(
                    self.max_action[i]) == self.batch_size, f"self.max_action entries are expected to be a scalar or a list with len(list)=batch_size"
        self.action_normalizer = jnp.array(self.action_normalizer).transpose()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, len(self.max_action)), dtype=jnp.float32)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, len(self.state_constraints)), dtype=jnp.float32)

        self.states = jnp.tile(
            jnp.array(self.state_initials), (self.batch_size, 1))

    @partial(jax.jit, static_argnums=0)
    def _static_generate_observation(self, states):
        """Returns states."""
        return states

    def generate_observation(self):
        """Returns the states of the environment."""
        return self.states

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

    def render(self, *_, **__):
        """
        Update the visualization of the environment.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")

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

        obs, reward, terminated, truncated, self.states = self._step_static(
            self.states, action)

        return obs, reward, terminated, truncated, {}

    @partial(jax.jit, static_argnums=0)
    def _step_static(self, states, action_norm):
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
        states = self._ode_exp_euler_step(states, action_norm)

        # observation
        obs = self._static_generate_observation(states)

        # reward
        reward = self.reward_func(obs, action_norm)

        # bound check
        truncated = (jnp.abs(states) > 1)
        terminated = reward == 0

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
            self.states = self.observation_space.sample(random_key)
        elif initial_values.any() != False:
            assert initial_values.shape[
                0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(
                self.obs_description), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            assert self.observation_space.contains(
                initial_values), f"values of initial states are out of bounds"
            self.states = initial_values
        else:
            self.states = jnp.tile(
                jnp.array(self.state_initials), (self.batch_size, 1))

        obs = self.generate_observation()

        return obs, {}

    def close(self):
        """Called when the environment is deleted.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")

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
    def _ode_exp_euler_step(self, states_norm, action_norm):
        """Implementation of the system equations in the class with Explicit Euler.

        Args:
            states_norm(ndarray(float)): State Matrix (shape=(batch_size,states)).
            action_norm(ndarray(float)): Action Matrix (shape=(batch_size,actions)).


        Returns:
            states(ndarray(float)): State Matrix (shape=(batch_size,states)).

        """
        return
