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


class Pendulum(core_env.CoreEnvironment):
    """
    State Variables:
        ``['theta', 'omega']``

    Action Variable:
        ``['torque']``

    Initial State:
        Unless chosen otherwise, theta equals pi and omega is set to zero.

    Example:
        >>> import jax
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>> 
        >>> # Create the environment
        >>> pend=excenv.Pendulum(batch_size=4,action_constraints={"torque":10})
        >>> 
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_pend=GymWrapper(env=pend)
        >>> 
        >>> # Reset the environment with default initial values
        >>> gym_pend.reset()
        >>> 
        >>> # Perform step
        >>> obs,reward,terminated,truncated,info= gym_pend.step(action=jnp.ones(4).reshape(-1,1))
        >>> 

    """

    def __init__(
        self,
        batch_size: int = 8,
        physical_constraints: dict = {"theta": jnp.pi, "omega": 10},
        action_constraints: dict = {"torque": 20},
        static_params: dict = {"g": 9.81, "l": 2, "m": 1},
        solver=diffrax.Euler(),
        reward_func=None,
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(jdc.pytree_dataclass): Constraints of physical states of the environment.
                theta(float): Rotation angle. Default: jnp.pi
                omega(float): Angular velocity. Default: 10
            action_constraints(jdc.pytree_dataclass): Constraints of actions.
                torque(float): Maximum torque that can be applied to the system as action. Default: 20 
            static_params(jdc.pytree_dataclass): Parameters of environment which do not change during simulation.
                l(float): Length of the pendulum. Default: 1
                m(float): Mass of the pendulum tip. Default: 1
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
        theta: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Optional:
        something: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        g: jax.Array
        l: jax.Array
        m: jax.Array

    @jdc.pytree_dataclass
    class Actions:
        torque: jax.Array

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
            theta, omega = y
            action, params = args
            d_omega = (action[0]+params.l*params.m*params.g
                       * jnp.sin(theta)) / (params.m * (params.l)**2)
            d_theta = omega
            d_y = d_theta, d_omega
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([env_states.theta, env_states.omega])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(
            term, t0, t1, y0, args, env_state, made_jump=False)

        theta_k1 = y[0]
        omega_k1 = y[1]
        theta_k1 = ((theta_k1+jnp.pi) % (2*jnp.pi))-jnp.pi

        phys = self.PhysicalStates(
            theta=theta_k1, omega=omega_k1)
        opt = None  # Optional(something=...)
        return self.States(physical_states=phys, PRNGKey=None, optional=None)

    @partial(jax.jit, static_argnums=0)
    def init_states(self):
        """Returns default initial states for all batches."""
        phys = self.PhysicalStates(theta=jnp.full(
            self.batch_size, jnp.pi), omega=jnp.zeros(self.batch_size))
        opt = None  # self.Optional(something=jnp.zeros(self.batch_size))
        return self.States(physical_states=phys, PRNGKey=None, optional=opt)

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action, action_constraints):
        """Returns reward for one batch."""
        reward = (obs[0])**2 + 0.1*(obs[1])**2 + 0.1 * \
            (action[0]/action_constraints.torque)**2
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states, physical_constraints):
        """Returns observation for one batch."""
        obs = jnp.hstack((
            states.physical_states.theta / physical_constraints.theta,
            states.physical_states.omega / physical_constraints.omega,
        ))
        return obs

    @property
    def obs_description(self):
        return np.array(["theta", "omega"])

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states, physical_constraints):
        """Returns truncated information for one batch."""
        _states = jnp.hstack((
            states.physical_states.theta / physical_constraints.theta,
            states.physical_states.theta / physical_constraints.omega,
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
