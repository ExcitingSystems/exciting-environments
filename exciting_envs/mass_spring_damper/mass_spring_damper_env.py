import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import vector
from exciting_envs import spaces
from functools import partial
import chex


class MassSpringDamper:
    """
    Description:
        Environment to simulate a Mass-Spring-Damper System.

    State Variables:
        ``['deflection' , 'velocity']''``
        
    Action Variable:
        ``['force']''``
        
    Observation Space (State Space):
        Box(low=[-1, -1], high=[1, 1])    
        
    Action Space:
        Box(low=-1, high=1)

    Initial State:
        Unless chosen otherwise, deflection and velocity is set to zero.

    Example:
        >>> #TODO
        
    """
      

    def __init__(self, batch_size=8, d=1 , k=100, m=1,  max_force=20, reward_func=None, tau = 1e-4 , constraints= [10,10]):
        
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            d(float): Damping constant. Default: 1
            k(float): Spring constant. Default: 100
            m(float): Mass of the oscillating object. Default: 1
            max_force(float): Maximum force that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(array): Constraints for states ['deflection','velocity'] (array with length 2). Default: [1000,10]


        """

        
        self.tau = tau
        self.k_const = k
        self.d_const = d
        self.m_const = m
        self.max_force= max_force
        self.batch_size = batch_size
        
        self.state_normalizer = jnp.array(constraints)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size,1), dtype=jnp.float32)
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size,2), dtype=jnp.float32)
        
        if reward_func:
            if self.test_rew_func(reward_func):
                self.reward_func=reward_func
        else:
            self.reward_func=self.default_reward_func
        
        

    def update_batch_dim(self):
        if jnp.isscalar(self.d_const):
            self.d = jnp.full((self.batch_size,1), self.d_const)
        else:
            assert len(self.d_const)==self.batch_size, f"d is expected to be a scalar or a list with len(list)=batch_size"
            self.d= jnp.array(self.d_const).reshape(-1,1)
            
        if jnp.isscalar(self.k_const):
            self.k = jnp.full((self.batch_size,1), self.k_const)
        else:
            assert len(self.k_const)==self.batch_size, f"k is expected to be a scalar or a list with len(list)=batch_size"
            self.k= jnp.array(self.k_const).reshape(-1,1)
            
        if jnp.isscalar(self.m_const):
            self.m = jnp.full((self.batch_size,1), self.m_const)
        else:
            assert len(self.m_const)==self.batch_size, f"m is expected to be a scalar or a list with len(list)=batch_size"
            self.m= jnp.array(self.m_const).reshape(-1,1)
            
        deflection = jnp.full((self.batch_size),1).reshape(-1,1)
        velocity = jnp.zeros(self.batch_size).reshape(-1,1)
        self.states = jnp.hstack((
                    deflection,
                    velocity,
                ))
        
    def test_rew_func(self,func):
        try:
            out=func(jnp.zeros([self.batch_size,int(len(self.get_obs_description()))]))
        except:
            raise Exception("Reward function should be using obs matrix as only parameter")
        try:
            if out.shape != (self.batch_size,1):
                raise Exception("Reward function should be returning vector in shape (batch_size,1)")    
        except:
            raise Exception("Reward function should be returning vector in shape (batch_size,1)")
        return True
   
    @partial(jax.jit, static_argnums=0)
    def ode_exp_euler_step(self,states_norm,force_norm):
        
        force = force_norm*self.max_force
        states = self.state_normalizer * states_norm
        deflection = states[:,0].reshape(-1,1)
        velocity = states[:,1].reshape(-1,1)
        
        ddeflection = velocity
        dvelocity = (force - self.d* velocity- self.k*deflection)/self.m
    
        deflection_k1 = deflection + self.tau *ddeflection  # explicit Euler
        velocity_k1= velocity + self.tau *dvelocity # explicit Euler
        
        states_k1 = jnp.hstack((
                    deflection_k1,
                    velocity_k1,
                ))
        states_k1_norm = states_k1/self.state_normalizer
        
        return states_k1_norm

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter  
    def batch_size(self, batch_size):
        # If batchsize change, update the corresponding dimension
        self._batch_size = batch_size
        self.update_batch_dim()
         
    def generate_observation(self):
        return self.states
    
    @partial(jax.jit, static_argnums=0)
    def static_generate_observation(self,states):
        return states
        
    def get_def_reward_func(self):
        return self.default_reward_func
    
    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self,obs,action):
        return ((obs[:,0])**2 + 0.1*(obs[:,1])**2 + 0.1*(action[:,0])**2).reshape(-1,1)
    
    def get_obs_description(self):
        return self.get_states_description()
    
    def get_states_description(self):
        return np.array(["deflection","velocity"])
    
    def get_action_description(self):
        return np.array(["force"])
    
    def step(self, force_norm):
        #TODO Totzeit hinzufügen
        
        obs,reward,terminated,truncated,self.states= self.step_static(self.states,force_norm)
        
        return obs, reward, terminated, truncated, {}

    @partial(jax.jit, static_argnums=0)
    def step_static(self,states,force_norm):
        # ode step
        states = self.ode_exp_euler_step(states,force_norm)

        # observation
        obs = self.static_generate_observation(states)
        
        # reward
        reward = self.reward_func(obs,force_norm)

        #bound check
        truncated = (jnp.abs(states)> 1)
        terminated = reward == 0
        
        return obs, reward, terminated, truncated ,states
    
    def render(self):
        raise NotImplementedError("To be implemented!")
        
    def close(self):
        raise NotImplementedError("To be implemented!")
    
    def reset(self,random_key:chex.PRNGKey=False,initial_values:jnp.ndarray=None):
        if random_key:
            self.states=self.observation_space.sample(random_key)
        elif initial_values!=None:
            assert initial_values.shape[0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(self.get_obs_description()), f"number of columns is expected to be amount of obs_entries: {len(self.get_obs_description())}, got: {initial_values.shape[0]}"
            assert self.observation_space.contains(initial_values), f"values of initial states are out of bounds"
            self.states=initial_values
        else:
            self.states=self.states.at[:,0:1].set(jnp.zeros(self.batch_size).reshape(-1,1))
            self.states=self.states.at[:,1:2].set(jnp.zeros(self.batch_size).reshape(-1,1))
            
        obs = self.generate_observation()

        return obs,{}