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
from dataclasses import fields


class MassSpringDamper(classic_core_env.ClassicCoreEnvironment):
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
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical state of the environment.
                deflection(float): Deflection of the mass. Default: 10
                velocity(float): Velocity of the mass. Default: 10
            action_constraints(dict): Constraints of actions.
                force(float): Maximum force that can be applied to the system as action. Default: 20
            static_params(dict): Parameters of environment which do not change during simulation.
                d(float): Damping constant. Default: 1
                k(float): Spring constant. Default: 100
                m(float): Mass of the oscillating object. Default: 1
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_constraints:
            physical_constraints = {"deflection": 10, "velocity": 10}

        if not action_constraints:
            action_constraints = {"force": 20}

        if not static_params:
            static_params = {"k": 100, "d": 1, "m": 1}

        if not control_state:
            control_state = ["deflection"]

        self.control_state = control_state

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

        deflection: jax.Array
        velocity: jax.Array

    @jdc.pytree_dataclass
    class Additions:
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
            deflection, velocity = y
            action, params = args
            d_velocity = (action[0] - params.d * velocity - params.k * deflection) / params.m
            d_deflection = velocity
            d_y = d_deflection, d_velocity  # [0]
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([env_state.deflection, env_state.velocity])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]

        with jdc.copy_and_mutate(state, validate=True) as new_state:
            new_state.physical_state = self.PhysicalState(deflection=deflection_k1, velocity=velocity_k1)
        return new_state

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None):
        """Returns default initial state for all batches."""
        if rng is None:
            phys = self.PhysicalState(
                deflection=0,
                velocity=0,
            )
            subkey = None
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(2,))
            phys = self.PhysicalState(
                deflection=state_norm[0] * env_properties.physical_constraints.deflection,
                velocity=state_norm[1] * env_properties.physical_constraints.velocity,
            )
            key, subkey = jax.random.split(rng)
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(deflection=None, velocity=None)
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=0)
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0))(self.env_properties, rng)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        for name in self.control_state:
            reward += (getattr(state.physical_state, name) - getattr(state.reference, name)) ** 2
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        """Returns observation for one batch."""
        physical_constraints = env_properties.physical_constraints
        obs = jnp.hstack(
            (
                state.physical_state.deflection / physical_constraints.deflection,
                state.physical_state.velocity / physical_constraints.velocity,
            )
        )
        for name in self.control_state:
            obs = jnp.hstack((obs, getattr(state.reference, name)))
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
        return np.hstack(
            [np.array(["deflection", "velocity"]), np.array([name + "_ref" for name in self.control_state])]
        )

    def reset(self, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        """Resets environment to default or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.vmap_init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.vmap_init_state()"
            state = initial_state
        else:
            state = self.vmap_init_state(rng)

        obs = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties),
        )(state, self.env_properties)

        return obs, state
