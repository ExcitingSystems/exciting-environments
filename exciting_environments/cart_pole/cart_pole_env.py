import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from exciting_environments import core_env


class CartPole(core_env.CoreEnvironment):
    """
    State Variables
        ``['deflection', 'velocity', 'theta', 'omega']``

    Action Variable:
        ``['force']``

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
        >>> obs,reward,terminated,truncated,info= env.step(action)
        >>> 

    """

    def __init__(self, batch_size: float = 8, mu_p: float = 0, mu_c: float = 0, l: float = 1, m_c: float = 1, m_p: float = 1,  max_force: float = 20, reward_func=None, g: float = 9.81, tau: float = 1e-4, constraints: list = [10, 10, 10]):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            mu_p(float): Coefficient of friction of pole on cart. Default: 0
            mu_c(float): Coefficient of friction of cart on track. Default: 0
            l(float): Half-pole length. Default: 1
            m_c(float): Mass of the cart. Default: 1
            m_p(float): Mass of the pole. Default: 1
            max_force(float): Maximum force that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            g(float): Gravitational acceleration. Default: 9.81
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(list): Constraints for states ['deflection','velocity','omega'] (list with length 3). Default: [10,10,10]

        Note: mu_p, mu_c, l, m_c, m_p and max_force can also be passed as lists with the length of the batch_size to set different parameters per batch. In addition to that constraints can also be passed as a list of lists with length 3 to set different constraints per batch.
        """

        self.params = {"g": g, "mu_p": mu_p, "mu_c": mu_c,
                       "m_c": m_c, "m_p": m_p, "l": l}

        self.state_constraints = [constraints[0],
                                  constraints[1], np.pi, constraints[2]]  # ["deflection", "velocity", "theta", "omega"]
        self.state_initials = [0, 0, 1, 0]
        self.max_action = [max_force]

        super().__init__(batch_size=batch_size, tau=tau, reward_func=reward_func)

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states_norm, force_norm, state_normalizer, action_normalizer, params):

        force = force_norm*action_normalizer
        states = state_normalizer * states_norm
        deflection = states[0]
        velocity = states[1]
        theta = states[2]
        omega = states[3]

        ddeflection = velocity
        dtheta = omega

        domega = (params[0]*jnp.sin(theta)+jnp.cos(theta)*((-force-params[4]*params[5]*(omega**2)*jnp.sin(theta)+params[2]*jnp.sign(velocity)) /
                  (params[3]+params[4]))-(params[1]*omega)/(params[4]*params[5]))/(params[5]*(4/3-(params[4]*(jnp.cos(theta))**2)/(params[3]+params[4])))

        dvelocity = (force + params[4]*params[5]*((omega**2)*jnp.sin(theta)-domega *
                     jnp.cos(theta)) - params[2] * jnp.sign(velocity))/(params[3]+params[4])

        deflection_k1 = deflection + self.tau * ddeflection  # explicit Euler
        velocity_k1 = velocity + self.tau * dvelocity  # explicit Euler

        theta_k1 = theta + self.tau * dtheta  # explicit Euler
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi
        omega_k1 = omega + self.tau * domega  # explicit Euler

        states_k1 = jnp.hstack((
            deflection_k1,
            velocity_k1,
            theta_k1,
            omega_k1,
        ))
        states_k1_norm = states_k1/state_normalizer

        return states_k1_norm

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action):
        return ((0.01*obs[0])**2 + 0.1*(obs[1])**2 + (obs[2])**2 + 0.1*(obs[3])**2 + 0.1*(action[0])**2)

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["deflection", "velocity", "theta", "omega"])

    @property
    def action_description(self):
        return np.array(["force"])
