import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from exciting_environments import core_env


class Pendulum(core_env.CoreEnvironment):
    """
    State Variables:
        ``['theta', 'omega']``

    Action Variable:
        ``['torque']``

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

        self.params = {"g": g, "l": l, "m": m}
        self.state_constraints = [np.pi, constraints[0]]  # ["theta", "omega"]
        self.state_initials = [1, 0]
        self.max_action = [max_torque]

        super().__init__(batch_size=batch_size, tau=tau)

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states_norm, torque_norm, state_normalizer, action_normalizer, params):

        torque = torque_norm*action_normalizer
        states = state_normalizer * states_norm
        theta = states[0]
        omega = states[1]

        domega = (torque+params[1]*params[2]*params[0]
                  * jnp.sin(theta)) / (params[2] * (params[1])**2)

        omega_k1 = omega + self.tau * domega  # explicit Euler
        dtheta = omega_k1
        theta_k1 = theta + self.tau * dtheta  # explicit Euler
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi

        states_k1 = jnp.hstack((
            theta_k1,
            omega_k1,
        ))
        states_k1_norm = states_k1/state_normalizer
        return states_k1_norm

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action):
        return (obs[0])**2 + 0.1*(obs[1])**2 + 0.1*(action[0])**2

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["theta", "omega"])

    @property
    def action_description(self):
        return np.array(["torque"])
