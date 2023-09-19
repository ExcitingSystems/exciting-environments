import numpy as np
import jax
import jax.numpy as jnp
from exciting_environments import spaces
from gymnasium import vector
from functools import partial
import chex


class CartPole:
    """
    Description:
        Environment to simulate a Cartpole System.

    State Variables:
        ``['deflection' , 'velocity' , 'theta' , 'omega']''``
        
    Action Variable:
        ``['force']''``
        
    Observation Space (State Space):
        Box(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])    
        
    Action Space:
        Box(low=-1, high=1)

    Initial State:
        Unless chosen otherwise, deflection, omega and velocity is set to zero and theta is set to 1(normalized to pi).

    Example:
        >>> import jax
        >>> import exciting_environments as excenvs
        >>> 
        >>> # Create the environment
        >>> env= excenvs.make('CartPole-v0',batch_size=2,l=3,m_c=4,max_force=30)
        >>> 
        >>> # Reset the environment with default initial values
        >>> env.reset()
        >>> 
        >>> # Sample a random action
        >>> action = env.action_space.sample(jax.random.PRNGKey(6))
        >>> 
        >>> # Perform step
        >>> obs,reward,terminated, truncated,info= env.step(action)
        >>> 
        
    """
      

    def __init__(self, batch_size=8, my_p=0, my_c=0, l=1 , m_c=1, m_p=1,  max_force=20, reward_func=None, g=9.81,tau = 1e-4 , constraints= [10,10,10]):
        
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            my_p(float): Coefficient of friction of the articualtion connecting pole and cart. Default: 0
            my_c(float): Coefficient of friction between cart and track. Default: 0
            l(float): Length of the pendulum. Default: 1
            m_c(float): Mass of the cart. Default: 1
            m_p(float): Mass of the pendulum tip. Default: 1
            max_force(float): Maximum force that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            g(float): Gravitational acceleration. Default: 9.81
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(array): Constraints for states ['deflection','velocity','omega'] (array with length 3). Default: [10,10,10]

        Note: my_p, my_c, l, m_c, m_p and max_force can also be passed as lists with the length of the batch_size to set different parameters per batch.
        """

        
        self.tau = tau
        self.g = g
        self.my_p_values = my_p
        self.my_c_values = my_c
        self.m_c_values = m_c
        self.m_p_values = m_p
        self.l_values = l
        self.max_force_values= max_force
        self.batch_size = batch_size
        
        self.state_normalizer = jnp.concatenate((jnp.array(constraints[0:2]),jnp.array([jnp.pi]),jnp.array(constraints[2:3])), axis=0)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size,1), dtype=jnp.float32)
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size,4), dtype=jnp.float32)
        
        if reward_func:
            if self.test_rew_func(reward_func):
                self.reward_func=reward_func
        else:
            self.reward_func=self.default_reward_func
        
        

    def update_batch_dim(self):
        if jnp.isscalar(self.my_p_values):
            self.my_p = jnp.full((self.batch_size,1), self.my_p_values)
        else:
            assert len(self.my_p_values)==self.batch_size, f"my_p is expected to be a scalar or a list with len(list)=batch_size"
            self.my_p= jnp.array(self.my_p_values).reshape(-1,1)
            
        if jnp.isscalar(self.my_c_values):
            self.my_c = jnp.full((self.batch_size,1), self.my_c_values)
        else:
            assert len(self.my_c_values)==self.batch_size, f"my_c is expected to be a scalar or a list with len(list)=batch_size"
            self.my_c= jnp.array(self.my_c_values).reshape(-1,1)
            
        if jnp.isscalar(self.m_c_values):
            self.m_c = jnp.full((self.batch_size,1), self.m_c_values)
        else:
            assert len(self.m_c_values)==self.batch_size, f"m_c is expected to be a scalar or a list with len(list)=batch_size"
            self.m_c= jnp.array(self.m_c_values).reshape(-1,1)
            
        if jnp.isscalar(self.m_p_values):
            self.m_p = jnp.full((self.batch_size,1), self.m_p_values)
        else:
            assert len(self.m_p_values)==self.batch_size, f"m_p is expected to be a scalar or a list with len(list)=batch_size"
            self.m_p = jnp.array(self.m_p_values).reshape(-1,1)
        
        if jnp.isscalar(self.l_values):
            self.l = jnp.full((self.batch_size,1), self.l_values)
        else:
            assert len(self.l_values)==self.batch_size, f"l is expected to be a scalar or a list with len(list)=batch_size"
            self.l = jnp.array(self.l_values).reshape(-1,1)
        
        if jnp.isscalar(self.max_force_values):
            self.max_force = jnp.full((self.batch_size,1), self.max_force_values)
        else:
            assert len(self.max_force_values)==self.batch_size, f"max_force is expected to be a scalar or a list with len(list)=batch_size"
            self.max_force = jnp.array(self.max_force_values).reshape(-1,1)

        deflection = jnp.zeros(self.batch_size).reshape(-1,1)
        velocity = jnp.zeros(self.batch_size).reshape(-1,1)
        theta = jnp.full((self.batch_size),1).reshape(-1,1)
        omega = jnp.zeros(self.batch_size).reshape(-1,1)
        self.states = jnp.hstack((
                    deflection,
                    velocity,
                    theta,
                    omega,
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
        theta = states[:,2].reshape(-1,1)
        omega = states[:,3].reshape(-1,1)
        
        ddeflection = velocity
        dtheta = omega
        
        domega = (self.g*jnp.sin(theta)+jnp.cos(theta)*((-force-self.m_p*self.l*(omega**2)*jnp.sin(theta)+self.my_c*jnp.sign(velocity))/(self.m_c+self.m_p))-(self.my_p*omega)/(self.m_p*self.l))/(self.l*(4/3-(self.m_p*(jnp.cos(theta))**2)/(self.m_c+self.m_p)))
        
        dvelocity = (force + self.m_p*self.l*((omega**2)*jnp.sin(theta)-domega*jnp.cos(theta))- self.my_c* jnp.sign(velocity))/(self.m_c+self.m_p)
    
        deflection_k1 = deflection + self.tau *ddeflection  # explicit Euler
        velocity_k1= velocity + self.tau *dvelocity # explicit Euler
        
        theta_k1 = theta + self.tau *dtheta  # explicit Euler
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi
        omega_k1= omega + self.tau *domega # explicit Euler
        
        states_k1 = jnp.hstack((
                    deflection_k1,
                    velocity_k1,
                    theta_k1,
                    omega_k1,
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
        return ((0.01*obs[:,0])**2 + 0.1*(obs[:,1])**2 + (obs[:,2])**2 + 0.1*(obs[:,3])**2 + 0.1*(action[:,0])**2).reshape(-1,1)
    
    def get_obs_description(self):
        return self.get_states_description()
    
    def get_states_description(self):
        return np.array(["deflection","velocity","theta","omega"])
    
    def get_action_description(self):
        return np.array(["force"])
    
    def step(self, force_norm):
        
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
            self.states=self.states.at[:,2:3].set(jnp.full(self.batch_size,1).reshape(-1,1))
            self.states=self.states.at[:,3:4].set(jnp.zeros(self.batch_size).reshape(-1,1))
            
        obs = self.generate_observation()

        return obs,{}