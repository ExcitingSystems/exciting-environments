import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from exciting_environments import core_env
import diffrax


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

    def __init__(self, batch_size: int = 8, l: float = 1, m: float = 1,  max_torque: float = 20, solver=diffrax.Euler(), reward_func=None, g: float = 9.81, tau: float = 1e-4, constraints: list = [10]):
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

        self.static_params = {"g": g, "l": l, "m": m}
        self.env_state_constraints = [
            np.pi, constraints[0]]  # ["theta", "omega"]
        self.env_state_initials = [1, 0]
        self.max_action = [max_torque]

        super().__init__(batch_size=batch_size, tau=tau, solver=solver)

        self.static_params, self.env_state_normalizer, self.action_normalizer, self.env_observation_space, self.action_space = self.sim_paras(
            self.static_params, self.env_state_constraints, self.max_action)

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states_norm, torque_norm, env_state_normalizer, action_normalizer, static_params):

        env_states_norm = states_norm
        torque = torque_norm*action_normalizer
        env_states = env_state_normalizer * env_states_norm
        args = (torque, static_params)

        def vector_field(t, y, args):
            theta, omega = y
            torque, params = args
            d_omega = (torque[0]+params["l"]*params["m"]*params["g"]
                       * jnp.sin(theta)) / (params["m"] * (params["l"])**2)
            d_theta = omega
            d_y = d_theta, d_omega[0]  # d_theta, d_omega
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple(env_states)
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(
            term, t0, t1, y0, args, env_state, made_jump=False)

        theta_k1 = y[0]
        omega_k1 = y[1]
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi

        env_states_k1 = jnp.hstack((
            theta_k1,
            omega_k1,
        ))

        env_states_k1_norm = env_states_k1/env_state_normalizer

        return env_states_k1_norm

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action):
        return (obs[0])**2 + 0.1*(obs[1])**2 + 0.1*(action[0])**2

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states):
        """Returns states."""
        return states

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states):
        """Returns states."""
        return jnp.abs(states) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, states, reward):
        """Returns states."""
        return reward == 0

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["theta", "omega"])

    @property
    def action_description(self):
        return np.array(["torque"])
