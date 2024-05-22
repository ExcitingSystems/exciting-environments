import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import chex
from functools import partial
import diffrax
from exciting_environments import core_env


class MassSpringDamper(core_env.CoreEnvironment):
    """

    State Variables:
        ``['deflection', 'velocity']``

    Action Variable:
        ``['force']``

    Initial State:
        Unless chosen otherwise, deflection and velocity is set to zero.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>> 
        >>> # Create the environment
        >>> msd=excenv.MassSpringDamper(batch_size=4,action_constraints={"force":10})
        >>> 
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_msd=GymWrapper(env=msd)
        >>> 
        >>> # Reset the environment with default initial values
        >>> gym_msd.reset()
        >>> 
        >>> # Perform step
        >>> obs,reward,terminated,truncated,info= gym_msd.step(action=jnp.ones(4).reshape(-1,1))
        >>> 

    """

    def __init__(
        self,
        batch_size: int = 8,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        solver=diffrax.Euler(),
        reward_func=None,
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical states of the environment.
                deflection(float): Deflection of the mass. Default: 10
                velocity(float): Velocity of the mass. Default: 10
            action_constraints(dict): Constraints of actions.
                force(float): Maximum force that can be applied to the system as action. Default: 20 
            static_params(dict): Parameters of environment which do not change during simulation.
                d(float): Damping constant. Default: 1
                k(float): Spring constant. Default: 100
                m(float): Mass of the oscillating object. Default: 1
            solver(diffrax.solver): Solver used to compute states for next step.
            reward_func(function): Reward function for training. Needs observation vector, action and action_constraints as Parameters. 
                                    Default: None (default_reward_func from class)
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.  
        """

        if not physical_constraints:
            physical_constraints = {"deflection": 10, "velocity": 10}

        if not action_constraints:
            action_constraints = {"force": 20}

        if not static_params:
            static_params = {"k": 100, "d": 1, "m": 1}

        physical_constraints = self.PhysicalStates(**physical_constraints)
        action_constraints = self.Actions(**action_constraints)
        static_params = self.StaticParams(**static_params)

        super().__init__(batch_size, physical_constraints, action_constraints, static_params, tau=tau,
                         solver=solver, reward_func=reward_func)

    @jdc.pytree_dataclass
    class PhysicalStates:
        """Dataclass containing the physical states of the environment."""
        deflection: jax.Array
        velocity: jax.Array

    @jdc.pytree_dataclass
    class Optional:
        """Dataclass containing additional information for simulation."""
        something: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""
        d: jax.Array
        k: jax.Array
        m: jax.Array

    @jdc.pytree_dataclass
    class Actions:
        """Dataclass containing the actions, that can be applied to the environment."""
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
            deflection, velocity = y
            action, params = args
            d_velocity = (action[0] - params.d
                          * velocity - params.k*deflection)/params.m
            d_deflection = velocity
            d_y = d_deflection, d_velocity  # [0]
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([env_states.deflection, env_states.velocity])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(
            term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]

        phys = self.PhysicalStates(
            deflection=deflection_k1, velocity=velocity_k1)
        opt = None  # Optional(something=...)
        return self.States(physical_states=phys, PRNGKey=None, optional=None)

    @partial(jax.jit, static_argnums=0)
    def init_states(self):
        """Returns default initial states for all batches."""
        phys = self.PhysicalStates(deflection=jnp.zeros(
            self.batch_size), velocity=jnp.zeros(self.batch_size))
        opt = None  # self.Optional(something=jnp.zeros(self.batch_size))
        return self.States(physical_states=phys, PRNGKey=None, optional=opt)

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action, action_constraints):
        """Returns reward for one batch."""
        reward = ((obs[0])**2 + 0.1*(obs[1])**2
                  + 0.1 (action[0]/action_constraints.force)**2)
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states, physical_constraints):
        """Returns observation for one batch."""
        obs = jnp.hstack((
            states.physical_states.deflection / physical_constraints.deflection,
            states.physical_states.velocity / physical_constraints.velocity,
        ))
        return obs

    @property
    def obs_description(self):
        return np.array(["deflection", "velocity"])

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states, physical_constraints):
        """Returns truncated information for one batch."""
        _states = jnp.hstack((
            states.physical_states.deflection / physical_constraints.deflection,
            states.physical_states.velocity / physical_constraints.velocity,
        ))
        return jnp.abs(_states) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, states, reward):
        """Returns terminated information for one batch."""
        return reward == 0

    def reset(self, rng: chex.PRNGKey = None, initial_states: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial states."""
        if initial_states is not None:
            assert tree_structure(self.init_states()) == tree_structure(initial_states), (
                f"initial_states should have the same dataclass structure as self.init_states()"
            )
            states = initial_states
        else:
            states = self.init_states()

        obs = jax.vmap(self.generate_observation, in_axes=(0, self.in_axes_env_properties.physical_constraints))(
            states, self.env_properties.physical_constraints
        )

        return obs, states
