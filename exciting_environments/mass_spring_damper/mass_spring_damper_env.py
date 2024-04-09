import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from exciting_environments import core_env
import diffrax


class MassSpringDamper(core_env.CoreEnvironment):
    """

    State Variables:
        ``['deflection', 'velocity']``

    Action Variable:
        ``['force']``

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

    def __init__(self, batch_size: int = 8, k: int = 100, d: int = 1, m: int = 1,  env_max_actions: dict = {"force": 20}, solver=diffrax.Euler(), reward_func=None, tau: float = 1e-4, env_state_constraints: dict = {"deflection": 10, "velocity": 10}):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            d(float): Damping constant. Default: 1
            k(float): Spring constant. Default: 100
            m(float): Mass of the oscillating object. Default: 1
            max_force(float): Maximum force that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters. Default: None (default_reward_func from class) 
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(list): Constraints for states ['deflection','velocity'] (list with length 2). Default: [10,10]

        Note: d,k,m and max_force can also be passed as lists with the length of the batch_size to set different parameters per batch. In addition to that constraints can also be passed as a list of lists with length 2 to set different constraints per batch.  
        """

        self.env_states_name = ["deflection", "velocity"]
        self.env_actions_name = ["force"]

        self.env_states_initials = {"deflection": 0, "velocity": 0}

        super().__init__(batch_size=batch_size, tau=tau,
                         solver=solver, reward_func=reward_func)
        self.static_params, self.env_state_constraints, self.env_max_actions = self.sim_paras(
            {"k": k, "d": d, "m": m}, env_state_constraints, env_max_actions)

    @partial(jax.jit, static_argnums=0)
    def _ode_exp_euler_step(self, states, action, static_params):

        env_states = states
        args = (action, static_params)

        def vector_field(t, y, args):
            deflection, velocity = y
            action, params = args
            d_velocity = (action["force"] - params["d"]
                          * velocity - params["k"]*deflection)/params["m"]
            d_deflection = velocity
            d_y = d_deflection, d_velocity  # [0]
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple(env_states.values())
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(
            term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]

        # env_states_k1 = jnp.hstack((
        #     deflection_k1,
        #     velocity_k1,
        # ))
        env_states_k1 = {"deflection": deflection_k1, "velocity": velocity_k1}

        return env_states_k1

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action):
        return ((obs[0])**2 + 0.1*(obs[1])**2 + 0.1*(action["force"])**2)

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states, env_state_constraints):
        """Returns states."""
        return (jnp.array(list(states.values()))/jnp.array(list(env_state_constraints.values()))).T

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states, env_state_constraints):
        """Returns states."""
        return jnp.abs((jnp.array(list(states.values()))/jnp.array(list(env_state_constraints.values()))).T) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, states, reward):
        """Returns states."""
        return reward == 0

    @property
    def obs_description(self):
        return self.env_states_name

    def reset(self, initial_values: jnp.ndarray = jnp.array([])):
        # TODO
        # if initial_values.any() != False:
        #     assert initial_values.shape[
        #         0] == self.batch_size, f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
        #     assert initial_values.shape[1] == len(
        #         self.obs_description), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
        #     states = initial_values
        # else:
        #     states = jnp.tile(
        #         jnp.array(self.env_state_initials), (self.batch_size, 1))

        # obs = self.generate_observation(states)

        return  # obs, states
