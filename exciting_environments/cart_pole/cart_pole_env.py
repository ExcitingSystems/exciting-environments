import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import diffrax
from collections import OrderedDict
from exciting_environments import core_env
import jax_dataclasses as jdc
import chex
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure


class CartPole(core_env.CoreEnvironment):
    """
    State Variables
        ``['deflection', 'velocity', 'theta', 'omega']``

    Action Variable:
        ``['force']``

    Initial State:
        Unless chosen otherwise, deflection, omega and velocity is set to zero and theta is set to pi.

    Example:
        >>> import jax
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>> 
        >>> # Create the environment
        >>> cartpole= excenv.CartPole(batch_size=5)
        >>>
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_cartpole=GymWrapper(env=cartpole)
        >>>
        >>> # Reset the environment with default initial values
        >>> gym_cartpole.reset()
        >>> 
        >>> # Perform step
        >>> obs,reward,terminated,truncated,info= gym_cartpole.step(action=jnp.ones(5).reshape(-1,1))
        >>> 

    """

    def __init__(
            self,
            batch_size: int = 8,
            physical_constraints: dict = {
                "deflection": 10, "velocity": 10, "theta": jnp.pi, "omega": 10},
            action_constraints: dict = {"force": 20},
            static_params: dict = {"mu_p": 0, "mu_c": 0,
                                   "l": 1, "m_p": 1, "m_c": 1, "g": 9.81},
            solver=diffrax.Euler(),
            reward_func=None,
            tau: float = 1e-4,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(jdc.pytree_dataclass): Constraints of physical states of the environment.
                deflection(float): Deflection of the cart. Default: 10
                velocity(float): Velocity of the cart. Default: 10
                theta(float): Rotation angle of the pole. Default: jnp.pi
                omega(float): Angular velocity. Default: 10
            action_constraints(jdc.pytree_dataclass): Constraints of actions.
                force(float): Maximum torque that can be applied to the system as action. Default: 20 
            static_params(jdc.pytree_dataclass): Parameters of environment which do not change during simulation.
                mu_p(float): Coefficient of friction of pole on cart. Default: 0
                mu_c(float): Coefficient of friction of cart on track. Default: 0
                l(float): Half-pole length. Default: 1
                m_p(float): Mass of the pole. Default: 1
                m_c(float): Mass of the cart. Default: 1
                g(float): Gravitational acceleration. Default: 9.81
            solver(diffrax.solver): Solver used to compute states for next step.
            reward_func(function): Reward function for training. Needs observation vector, action and action_constraints as Parameters. 
                                    Default: None (default_reward_func from class)
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.  
        """

        physical_constraints = self.PhysicalStates(**physical_constraints)
        action_constraints = self.Actions(**action_constraints)
        static_params = self.StaticParams(**static_params)

        super().__init__(batch_size, physical_constraints, action_constraints, static_params, tau=tau,
                         solver=solver, reward_func=reward_func)

    @jdc.pytree_dataclass
    class PhysicalStates:
        deflection: jax.Array
        velocity: jax.Array
        theta: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Optional:
        something: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        mu_p: jax.Array
        mu_c: jax.Array
        l: jax.Array
        m_p: jax.Array
        m_c: jax.Array
        g: jax.Array

    @jdc.pytree_dataclass
    class Actions:
        force: jax.Array

    @partial(jax.jit, static_argnums=0)
    def _ode_solver_step(self, states, action, static_params):
        """Computes states by simulating one step.

        Args:
            states: The states from which to calculate states for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            states: The computed states after the one step simulation.
        """

        env_states = states.physical_states
        args = (action, static_params)

        def vector_field(t, y, args):
            deflection, velocity, theta, omega = y
            action, params = args
            d_omega = (params.g*jnp.sin(theta)+jnp.cos(theta)*((-action[0]-params.m_p*params.l*(omega**2)*jnp.sin(theta)+params.mu_c*jnp.sign(velocity)) /
                                                               (params.m_c+params.m_p))-(params.mu_p*omega)/(params.m_p*params.l))/(params.l*(4/3-(params.m_p*(jnp.cos(theta))**2)/(params.m_c+params.m_p)))

            d_velocity = (action[0] + params.m_p*params.l*((omega**2)*jnp.sin(theta)-d_omega *
                                                           jnp.cos(theta)) - params.mu_c * jnp.sign(velocity))/(params.m_c+params.m_p)
            d_theta = omega
            d_deflection = velocity
            d_y = d_deflection, d_velocity, d_theta, d_omega
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([env_states.deflection, env_states.velocity,
                   env_states.theta, env_states.omega])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(
            term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]
        theta_k1 = y[2]
        omega_k1 = y[3]
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi

        phys = self.PhysicalStates(deflection=deflection_k1, velocity=velocity_k1,
                                   theta=theta_k1, omega=omega_k1)
        opt = None  # Optional(something=...)
        return self.States(physical_states=phys, PRNGKey=None, optional=opt)

    @partial(jax.jit, static_argnums=0)
    def init_states(self):
        """Returns default initial states for all batches."""
        phys = self.PhysicalStates(deflection=jnp.zeros(self.batch_size), velocity=jnp.zeros(self.batch_size), theta=jnp.full(
            self.batch_size, jnp.pi), omega=jnp.zeros(self.batch_size))
        opt = None  # self.Optional(something=jnp.zeros(self.batch_size))
        return self.States(physical_states=phys, PRNGKey=None, optional=opt)

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action, action_constraints):
        """Returns reward for one batch."""
        reward = ((0.01*obs[0])**2 + 0.1*(obs[1])**2 +
                  (obs[2])**2 + 0.1*(obs[3])**2 + 0.1*(action[0]/action_constraints.force)**2)
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states, physical_constraints):
        """Returns observation for one batch."""
        obs = jnp.hstack((
            states.physical_states.deflection / physical_constraints.deflection,
            states.physical_states.velocity / physical_constraints.velocity,
            states.physical_states.theta / physical_constraints.theta,
            states.physical_states.omega / physical_constraints.omega,
        ))
        return obs

    @property
    def obs_description(self):
        return np.array(["deflection", "velocity", "theta", "omega"])

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states, physical_constraints):
        """Returns truncated information for one batch."""
        _states = jnp.hstack((
            states.physical_states.deflection / physical_constraints.deflection,
            states.physical_states.velocity / physical_constraints.velocity,
            states.physical_states.theta / physical_constraints.theta,
            states.physical_states.omega / physical_constraints.omega,
        ))
        return jnp.abs(_states) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, states, reward):
        """Returns terminated information for one batch."""
        return reward == 0

    def reset(self, rng: chex.PRNGKey = None, initial_states: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial states."""
        if initial_states is not None:
            assert tree_structure(self.init_states()) == tree_structure(
                initial_states), f"initial_states should have the same dataclass structure as self.init_states()"
            states = initial_states
        else:
            states = self.init_states()

        obs = jax.vmap(self.generate_observation, in_axes=(0, self.in_axes_env_properties.physical_constraints))(
            states, self.env_properties.physical_constraints)

        return obs, states
