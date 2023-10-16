import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from exciting_environments import core_env, spaces


class MassSpringDamper(core_env.CoreEnvironment):
    """

    State Variables:
        ``['deflection' , 'velocity']``

    Action Variable:
        ``['force']''``

    Observation Space (State Space):
        Box(low=[-1, -1], high=[1, 1])    

    Action Space:
        Box(low=-1, high=1)

    Initial State:
        Unless chosen otherwise, deflection and velocity is set to zero.

    Example:
        >>> import jax
        >>> import exciting_environments as excenvs
        >>> 
        >>> # Create the environment
        >>> env= excenvs.make('MassSpringDamper-v0',batch_size=2,d=2,k=0.5,max_force=10)
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

    def __init__(self, batch_size: int = 8, d: float = 1, k: float = 100, m: float = 1,  max_force: float = 20, reward_func=None, tau: float = 1e-4, constraints: list = [10, 10]):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            d(float): Damping constant. Default: 1
            k(float): Spring constant. Default: 100
            m(float): Mass of the oscillating object. Default: 1
            max_force(float): Maximum force that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(list): Constraints for states ['deflection','velocity'] (list with length 2). Default: [1000,10]

        Note: d,k,m and max_force can also be passed as lists with the length of the batch_size to set different parameters per batch. In addition to that constraints can also be passed as a list of lists with length 2 to set different constraints per batch.  
        """

        self.k_values = k
        self.d_values = d
        self.m_values = m
        self.max_force_values = max_force
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
                self.constraints) == 2, f"constraints is expected to be a list with len(list)=2 or a list of lists with overall dimension (batch_size,2)"
            self.state_normalizer = jnp.array(self.constraints)
        else:
            assert jnp.array(
                self.constraints).shape[0] == self.batch_size, f"constraints is expected to be a list with len(list)=1 or a list of lists with overall dimension (batch_size,1)"
            self.state_normalizer = jnp.array(self.constraints)

        if jnp.isscalar(self.d_values):
            self.d = jnp.full((self.batch_size, 1), self.d_values)
        else:
            assert len(
                self.d_values) == self.batch_size, f"d is expected to be a scalar or a list with len(list)=batch_size"
            self.d = jnp.array(self.d_values).reshape(-1, 1)

        if jnp.isscalar(self.k_values):
            self.k = jnp.full((self.batch_size, 1), self.k_values)
        else:
            assert len(
                self.k_values) == self.batch_size, f"k is expected to be a scalar or a list with len(list)=batch_size"
            self.k = jnp.array(self.k_values).reshape(-1, 1)

        if jnp.isscalar(self.m_values):
            self.m = jnp.full((self.batch_size, 1), self.m_values)
        else:
            assert len(
                self.m_values) == self.batch_size, f"m is expected to be a scalar or a list with len(list)=batch_size"
            self.m = jnp.array(self.m_values).reshape(-1, 1)

        if jnp.isscalar(self.max_force_values):
            self.max_force = jnp.full(
                (self.batch_size, 1), self.max_force_values)
        else:
            assert len(
                self.max_force_values) == self.batch_size, f"max_force is expected to be a scalar or a list with len(list)=batch_size"
            self.max_force = jnp.array(self.max_force_values).reshape(-1, 1)

        deflection = jnp.full((self.batch_size), 1).reshape(-1, 1)
        velocity = jnp.zeros(self.batch_size).reshape(-1, 1)
        self.states = jnp.hstack((
            deflection,
            velocity,
        ))

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states_norm, force_norm):

        force = force_norm*self.max_force
        states = self.state_normalizer * states_norm
        deflection = states[:, 0].reshape(-1, 1)
        velocity = states[:, 1].reshape(-1, 1)

        ddeflection = velocity
        dvelocity = (force - self.d * velocity - self.k*deflection)/self.m

        deflection_k1 = deflection + self.tau * ddeflection  # explicit Euler
        velocity_k1 = velocity + self.tau * dvelocity  # explicit Euler

        states_k1 = jnp.hstack((
            deflection_k1,
            velocity_k1,
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
        return np.array(["deflection", "velocity"])

    @property
    def action_description(self):
        return np.array(["force"])

    def reset(self, random_key: chex.PRNGKey = False, initial_values: jnp.ndarray = None):
        if random_key:
            self.states = self.observation_space.sample(random_key)
        elif initial_values != None:
            assert initial_values.shape[
                0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(
                self.obs_description), f"number of columns is expected to be amount of obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            assert self.observation_space.contains(
                initial_values), f"values of initial states are out of bounds"
            self.states = initial_values
        else:
            self.states = self.states.at[:, 0:1].set(
                jnp.zeros(self.batch_size).reshape(-1, 1))
            self.states = self.states.at[:, 1:2].set(
                jnp.zeros(self.batch_size).reshape(-1, 1))

        obs = self.generate_observation()

        return obs, {}
