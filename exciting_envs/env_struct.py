
import numpy as np
from gymnasium import spaces
from gymnasium import vector

class CoreEnvironment:
    """
    Description:
        Structure of provided Environments.

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
      

    def get_def_reward_function(self):
        """Returns the default RewardFunction of the environment."""
        return self.default_reward_func


    @property
    def batch_size(self):
        """Returns the batch size of the environment setup."""
        return self._batch_size


    def get_obs_description(self):
        """Returns a list of state names of all states in the observation (equal to state space)."""
        return self.get_states_description()
    
    def get_states_description(self):
        """Returns a list of state names of all states in the states space."""
        return np.array(["state1_name","..."])
    
    def get_action_description(self):
        """Returns the name of the action."""
        return np.array(["action_name"])

    def __init__(self, batch_size, physical_paras, max_action, reward_func=None, tau = 1e-4 , constraints= []):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration.
            physical_paras: Depending on environment there are multiple parameter for the physical system.
            max_action(float): Maximum action that can be applied to the system.
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(array): Constraints for states.


        """
        

            
    def reset(self,random_initial_values=False,initial_values:np.ndarray=None):
        """
            Reset the environment, return initial observation vector.
            Options:
                - Observation/State Space gets a random initial sample
                - Initial Observation/State Space is set to initial_values array
        
        """
        return

    def render(self, *_, **__):
        """
        Update the visualization of the motor.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")
    

    def step(self, action):
        """Perform one simulation step of the environment with an action of the action space.

        Args:
            action: Action to play on the environment.

        Returns:
            observation(ndarray(float)):
                Observation/State Matrix: (shape=(batch_size,states)).
            reward(ndarray(float)):
                Amount of reward received for the last step: (shape=(batch_size,1)).
            terminated(bool): 
                Flag, indicating if Agent has reached the terminal state.
            truncated(ndarray(bool)): 
                Flag, indicating if state has gone out of bounds: (shape=(batch_size,states)).
            {}: An empty dictionary for consistency with the OpenAi Gym interface.
        """
        return


    def close(self):
        """Called when the environment is deleted.

        NotImplemented
        """
        raise NotImplementedError("To be implemented!")