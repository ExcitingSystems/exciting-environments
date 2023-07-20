import numpy as np
from gymnasium import spaces
from gymnasium import vector

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
        
        self.state_normalizer = constraints
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size,1), dtype=np.float32)
        
        single_obs_space=spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0 , 1.0]),shape=(2,) ,dtype=np.float32)
        self.observation_space= vector.utils.batch_space(single_obs_space, n=batch_size)
        
        if reward_func:
            if self.test_rew_func(reward_func):
                self.reward_func=reward_func
        else:
            self.reward_func=self.default_reward_func
        
        

    def update_batch_dim(self):
        self.d = np.full((self.batch_size,1), self.d_const)
        self.k = np.full((self.batch_size,1), self.k_const)
        self.m = np.full((self.batch_size,1), self.m_const)
        deflection = np.full((self.batch_size),1).reshape(-1,1)
        velocity = np.zeros(self.batch_size).reshape(-1,1)
        self.states = np.hstack((
                    deflection,
                    velocity,
                ))
        
    def test_rew_func(self,func):
        try:
            out=func(np.zeros([self.batch_size,int(len(self.get_obs_description()))]))
        except:
            raise Exception("Reward function should be using obs matrix as only parameter")
        try:
            if out.shape != (self.batch_size,1):
                raise Exception("Reward function should be returning vector in shape (batch_size,1)")    
        except:
            raise Exception("Reward function should be returning vector in shape (batch_size,1)")
        return True
            

    def ode_exp_euler_step(self,states_norm,force_norm):
        
        force = force_norm*self.max_force
        states = self.state_normalizer * states_norm
        deflection = states[:,0].reshape(-1,1)
        velocity = states[:,1].reshape(-1,1)
        
        ddeflection = velocity
        dvelocity = (force - self.d* velocity- self.k*deflection)/self.m
    
        deflection_k1 = deflection + self.tau *ddeflection  # explicit Euler
        velocity_k1= velocity + self.tau *dvelocity # explicit Euler
        
        states_k1 = np.hstack((
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
        return np.hstack((
            self.states,
            #force,
        ))

    @property   
    def def_reward_func(self):
        return self.default_reward_func
    
    def default_reward_func(self,obs,action):
        return ((obs[:,0])**2 + 0.1*(obs[:,1])**2 + 0.1*(action[:,0])**2).reshape(-1,1)
    
    @property 
    def obs_description(self):
        return self.states_description
    
    @property 
    def states_description(self):
        return np.array(["deflection","velocity"])
    
    @property 
    def action_description(self):
        return np.array(["force"])
    
    def step(self, force_norm):
        #TODO Totzeit hinzufügen
        
        
        # ode step
        self.states = self.ode_exp_euler_step(self.states,force_norm)

        # observation
        obs = self.generate_observation()
        
        # reward
        reward = self.reward_func(obs,force_norm)

        #bound check
        truncated = (np.abs(self.states)> 1)
        terminated = reward == 0
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError("To be implemented!")
        
    def close(self):
        raise NotImplementedError("To be implemented!")
    
    def reset(self,random_initial_values=False,initial_values:np.ndarray=None):
        if random_initial_values:
            self.states=self.observation_space.sample()
        elif initial_values!=None:
            assert initial_values.shape[0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(self.obs_description), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            self.states=initial_values
        else:
            self.states[:,0:1]=np.zeros(self.batch_size).reshape(-1,1)
            self.states[:,1:2]=np.zeros(self.batch_size).reshape(-1,1)
            
        obs = self.generate_observation()

        return obs,{}