import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from exciting_environments import core_env, spaces


class Pendulum(core_env.CoreEnvironment):
    """
    State Variables:
        ``['theta' , 'omega']``

    Action Variable:
        ``['torque']''``

    Observation Space (State Space):
        Box(low=[-1, -1], high=[1, 1])    

    Action Space:
        Box(low=-1, high=1)

    Initial State:
        Unless chosen otherwise, theta equals 1(normalized to pi) and omega is set to zero.

    Example:
        >>> import jax
        >>> import exciting_environments as excenvs
        >>> 
        >>> # Create the environment
        >>> env= excenvs.make('Pendulum-v0',batch_size=2,l=2,m=4)
        >>> 
        >>> # Reset the environment with default initial values
        >>> env.reset()
        >>> 
        >>> # Sample a random action
        >>> action = env.action_space.sample(jax.random.PRNGKey(6))
        >>> 
        >>> # Perform step
        >>> obs,reward,terminated,truncated,info= env.step(action)
        >>> 

    """

    def __init__(self, batch_size: int = 8, l: float = 1, m: float = 1,  max_torque: float = 20, reward_func=None, g: float = 9.81, tau: float = 1e-4, constraints: list = [10]):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            l(float): Length of the pendulum. Default: 1
            m(float): Mass of the pendulum tip. Default: 1
            max_torque(float): Maximum torque that can be applied to the system as action. Default: 20 
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. 
                                    Default: None (default_reward_func from class) 
            g(float): Gravitational acceleration. Default: 9.81
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(list): Constraints for state ['omega'] (list with length 1). Default: [10]

        Note: l,m and max_torque can also be passed as lists with the length of the batch_size to set different parameters per batch. In addition to that constraints can also be passed as a list of lists with length 1 to set different constraints per batch.  
        """

        self.g = g
        self.l_values = l
        self.m_values = m
        self.max_torque_values = max_torque
        self.constraints = constraints
        super().__init__(batch_size=batch_size, tau=tau)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, 1), dtype=jnp.float32)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.batch_size, 2), dtype=jnp.float32)

        if reward_func:
            if self._test_rew_func(reward_func):
                self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

    def update_batch_dim(self):

        if isinstance(self.constraints, list) and not isinstance(self.constraints[0], list):
            assert len(
                self.constraints) == 1, f"constraints is expected to be a list with len(list)=1 or a list of lists with overall dimension (batch_size,1)"
            self.state_normalizer = jnp.concatenate(
                (jnp.array([jnp.pi]), jnp.array(self.constraints)), axis=0)
        else:
            assert jnp.array(
                self.constraints).shape[0] == self.batch_size, f"constraints is expected to be a list with len(list)=1 or a list of lists with overall dimension (batch_size,1)"
            self.state_normalizer = jnp.concatenate((jnp.full(
                (self.batch_size, 1), jnp.pi).reshape(-1, 1), jnp.array(self.constraints)), axis=1)

        if jnp.isscalar(self.l_values):
            self.l = jnp.full((self.batch_size, 1), self.l_values)
        else:
            assert len(
                self.l_values) == self.batch_size, f"l is expected to be a scalar or a list with len(list)=batch_size"
            self.l = jnp.array(self.l_values).reshape(-1, 1)

        if jnp.isscalar(self.m_values):
            self.m = jnp.full((self.batch_size, 1), self.m_values)
        else:
            assert len(
                self.m_values) == self.batch_size, f"m is expected to be a scalar or a list with len(list)=batch_size"
            self.m = jnp.array(self.m_values).reshape(-1, 1)

        if jnp.isscalar(self.max_torque_values):
            self.max_torque = jnp.full(
                (self.batch_size, 1), self.max_torque_values)
        else:
            assert len(
                self.max_torque_values) == self.batch_size, f"max_torque is expected to be a scalar or a list with len(list)=batch_size"
            self.max_torque = jnp.array(self.max_torque_values).reshape(-1, 1)

        theta = jnp.full((self.batch_size), 1).reshape(-1, 1)
        omega = jnp.zeros(self.batch_size).reshape(-1, 1)
        self.states = jnp.hstack((
            theta,
            omega,
        ))

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states_norm, torque_norm):

        torque = torque_norm*self.max_torque
        states = self.state_normalizer * states_norm
        theta = states[:, 0].reshape(-1, 1)
        omega = states[:, 1].reshape(-1, 1)

        dtheta = omega
        domega = (torque+self.l*self.m*self.g*jnp.sin(theta)) / \
            (self.m * (self.l)**2)

        theta_k1 = theta + self.tau * dtheta  # explicit Euler
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi
        omega_k1 = omega + self.tau * domega  # explicit Euler

        states_k1 = jnp.hstack((
            theta_k1,
            omega_k1,
        ))
        states_k1_norm = states_k1/self.state_normalizer

        return states_k1_norm

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action):
        return ((obs[:, 0])**2 + 0.1*(obs[:, 1])**2 + 0.1*(action[:, 0])**2).reshape(-1, 1)

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["theta", "omega"])

    @property
    def action_description(self):
        return np.array(["torque"])

    def reset(self, random_key: chex.PRNGKey = False, initial_values: jnp.ndarray = None):
        if random_key:
            self.states = self.observation_space.sample(random_key)
        elif initial_values != None:
            assert initial_values.shape[
                0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(
                self.obs_description), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            assert self.observation_space.contains(
                initial_values), f"values of initial states are out of bounds"
            self.states = initial_values
        else:
            self.states = self.states.at[:, 0:1].set(
                jnp.full(self.batch_size, 1).reshape(-1, 1))
            self.states = self.states.at[:, 1:2].set(
                jnp.zeros(self.batch_size).reshape(-1, 1))

        obs = self.generate_observation()

        return obs, {}
