from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure
import jax_dataclasses as jdc
import chex
import diffrax

from exciting_environments import ClassicCoreEnvironment


class MassSpringDamper(ClassicCoreEnvironment):
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
        >>> obs,reward,terminated,truncated = gym_msd.step(action=jnp.ones(4).reshape(-1,1))
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
            physical_constraints(dict): Constraints of the physical state of the environment.
                deflection(float): Deflection of the mass. Default: 10
                velocity(float): Velocity of the mass. Default: 10
            action_constraints(dict): Constraints of the action.
                force(float): Maximum force that can be applied to the system as action. Default: 20
            static_params(dict): Parameters of environment which do not change during simulation.
                d(float): Damping constant. Default: 1
                k(float): Spring constant. Default: 100
                m(float): Mass of the oscillating object. Default: 1
            control_state (list): Components of the physical state that are considered in reference tracking.
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
            control_state = []

        self.control_state = control_state

        physical_constraints = self.PhysicalState(**physical_constraints)
        action_constraints = self.Action(**action_constraints)
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

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        d: jax.Array
        k: jax.Array
        m: jax.Array

    @jdc.pytree_dataclass
    class Action:
        """Dataclass containing the action, that can be applied to the environment."""

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

        physical_state = state.physical_state
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
        y0 = tuple([physical_state.deflection, physical_state.velocity])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]

        with jdc.copy_and_mutate(state, validate=True) as new_state:
            new_state.physical_state = self.PhysicalState(deflection=deflection_k1, velocity=velocity_k1)
        return new_state

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def _ode_solver_simulate_ahead(self, init_state, actions, static_params, obs_stepsize, action_stepsize):
        """Computes multiple simulation steps for one batch.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every.
            action_stepsize (shape=(n_action_steps, action_dim)).
            static_params: The constant properties of the simulation.
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.

        Returns:
            next_states: The computed states during the multiple step simulation.
        """

        init_physical_state = init_state.physical_state
        args = (actions, static_params)

        def force(t, args):
            actions = args
            return actions[jnp.array(t / action_stepsize, int), 0]

        def vector_field(t, y, args):
            deflection, velocity = y
            actions, params = args
            d_velocity = (force(t, actions) - params.d * velocity - params.k * deflection) / params.m
            d_deflection = velocity
            d_y = d_deflection, d_velocity
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = action_stepsize * actions.shape[0]
        init_physical_state_array, _ = tree_flatten(init_physical_state)
        y0 = tuple(init_physical_state_array)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / obs_stepsize)))  #
        sol = diffrax.diffeqsolve(term, self._solver, t0, t1, dt0=obs_stepsize, y0=y0, args=args, saveat=saveat)

        deflection_t = sol.ys[0]
        velocity_t = sol.ys[1]
        obs_len = velocity_t.shape[0]

        physical_states = self.PhysicalState(deflection=deflection_t, velocity=velocity_t)
        ref = self.PhysicalState(
            deflection=jnp.full(obs_len, init_state.reference.deflection),
            velocity=jnp.full(obs_len, init_state.reference.velocity),
        )
        additions = None
        PRNGKey = jnp.full(obs_len, init_state.PRNGKey)
        return self.State(physical_state=physical_states, PRNGKey=PRNGKey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default or random initial state for one batch."""
        if rng is None:
            phys = self.PhysicalState(
                deflection=0.0,
                velocity=0.0,
            )
            subkey = jnp.nan
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(2,))
            phys = self.PhysicalState(
                deflection=state_norm[0] * env_properties.physical_constraints.deflection,
                velocity=state_norm[1] * env_properties.physical_constraints.velocity,
            )
            key, subkey = jax.random.split(rng)
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan)
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        for name in self.control_state:
            reward += -(
                (
                    (getattr(state.physical_state, name) - getattr(state.reference, name))
                    / (getattr(env_properties.physical_constraints, name)).astype(float)
                )
                ** 2
            )
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
            obs = jnp.hstack(
                (
                    obs,
                    (getattr(state.reference, name) / (getattr(physical_constraints, name)).astype(float)),
                )
            )
        return obs

    @partial(jax.jit, static_argnums=0)
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        physical_constraints = env_properties.physical_constraints
        phys = self.PhysicalState(
            deflection=obs[0] * (physical_constraints.deflection).astype(float),
            velocity=obs[1] * (physical_constraints.velocity).astype(float),
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        additions = None
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan)
        with jdc.copy_and_mutate(ref, validate=False) as new_ref:
            for name, pos in zip(self.control_state, range(len(self.control_state))):
                setattr(new_ref, name, obs[2 + pos] * (getattr(physical_constraints, name)).astype(float))
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)

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

    @property
    def action_description(self):
        return np.array(["force"])

    def reset(
        self, env_properties, rng: chex.PRNGKey = None, initial_state: jdc.pytree_dataclass = None, vmap_helper=None
    ):
        """Resets one batch to default, random or passed initial state."""
        if initial_state is not None:
            assert tree_structure(self.init_state(env_properties)) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as init_state(env_properties)"
            state = initial_state
        else:
            state = self.init_state(env_properties, rng)

        obs = self.generate_observation(state, env_properties)

        return obs, state
