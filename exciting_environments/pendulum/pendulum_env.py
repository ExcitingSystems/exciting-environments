import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import chex
from functools import partial
import diffrax
from exciting_environments import classic_core_env
from typing import Callable


class Pendulum(classic_core_env.ClassicCoreEnvironment):
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
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical state of the environment.
                theta(float): Rotation angle. Default: jnp.pi
                omega(float): Angular velocity. Default: 10
            action_constraints(dict): Constraints of actions.
                torque(float): Maximum torque that can be applied to the system as action. Default: 20
            static_params(dict): Parameters of environment which do not change during simulation.
                l(float): Length of the pendulum. Default: 1
                m(float): Mass of the pendulum tip. Default: 1
                g(float): Gravitational acceleration. Default: 9.81
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_constraints:
            physical_constraints = {"theta": jnp.pi, "omega": 10}

        if not action_constraints:
            action_constraints = {"torque": 20}

        if not static_params:
            static_params = {"g": 9.81, "l": 2, "m": 1}

        physical_constraints = self.PhysicalState(**physical_constraints)
        action_constraints = self.Actions(**action_constraints)
        static_params = self.StaticParams(**static_params)

        super().__init__(
            batch_size,
            physical_constraints,
            action_constraints,
            static_params,
            tau=tau,
            solver=solver,
        )

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        theta: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Additions:
        """Dataclass containing additional information for simulation."""

        something: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        g: jax.Array
        l: jax.Array
        m: jax.Array

    @jdc.pytree_dataclass
    class Actions:
        """Dataclass containing the actions, that can be applied to the environment."""

        torque: jax.Array

    @partial(jax.jit, static_argnums=0)
    def _ode_solver_step(self, state, action, static_params):
        """Computes state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            state: The computed state after the one step simulation.
        """

        env_state = state.physical_state
        args = (action, static_params)

        def vector_field(t, y, args):
            theta, omega = y
            action, params = args
            d_omega = (action[0] + params.l * params.m * params.g * jnp.sin(theta)) / (params.m * (params.l) ** 2)
            d_theta = omega
            d_y = d_theta, d_omega
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([env_state.theta, env_state.omega])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        theta_k1 = y[0]
        omega_k1 = y[1]
        theta_k1 = ((theta_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        phys = self.PhysicalState(theta=theta_k1, omega=omega_k1)
        opt = None  # Optional(something=...)
        return self.State(physical_state=phys, PRNGKey=None, additions=None)

    @partial(jax.jit, static_argnums=0)
    def init_state(self):
        """Returns default initial state for all batches."""
        phys = self.PhysicalState(theta=jnp.full(self.batch_size, jnp.pi), omega=jnp.zeros(self.batch_size))
        opt = None  # self.Optional(something=jnp.zeros(self.batch_size))
        return self.State(physical_state=phys, PRNGKey=None, additions=opt)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        action_constraints = env_properties.action_constraints
        obs = self.generate_observation(state, env_properties)
        reward = (obs[0]) ** 2 + 0.1 * (obs[1]) ** 2 + 0.1 * (action[0] / action_constraints.torque) ** 2
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        """Returns observation for one batch."""
        physical_constraints = env_properties.physical_constraints
        obs = jnp.hstack(
            (
                state.physical_state.theta / physical_constraints.theta,
                state.physical_state.omega / physical_constraints.omega,
            )
        )
        return obs

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, state, env_properties):
        """Returns truncated information for one batch."""
        obs = self.generate_observation(state, env_properties)
        return jnp.abs(obs) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information for one batch."""
        return reward == 0

    @property
    def obs_description(self):
        return np.array(["theta", "omega"])

    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.init_state()"
            state = initial_state
        else:
            state = self.init_state()

        obs = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties),
        )(state, self.env_properties)

        return obs, state
