from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure, tree_map

import chex
import diffrax
import equinox as eqx
from dataclasses import fields

from exciting_environments import CoreEnvironment
from exciting_environments.utils import MinMaxNormalization


def massspringdamper_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize_state(state)
    physical_state_norm = state_norm.physical_state
    constrained_states = ["deflection", "velocity"]
    names = [f.name for f in fields(type(physical_state_norm))]
    values = [
        jax.nn.relu(jnp.abs(getattr(physical_state_norm, n)) - 1.0) if n in constrained_states else jnp.nan
        for n in names
    ]

    phys_soft_const = eqx.tree_unflatten(eqx.tree_structure(physical_state_norm), values)

    act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
    return phys_soft_const, act_soft_constr


class MassSpringDamper(CoreEnvironment):
    control_state: list = eqx.field(static=True)
    soft_constraints_logic: Callable = eqx.field(static=True)
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
        >>> msd=excenv.MassSpringDamper(batch_size=4)
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
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-4,
    ):
        """
        Args:
            physical_normalizations(dict): min-max normalization values of the physical state of the environment.
                deflection(MinMaxNormalization): Deflection of the mass. Default: min=-10, max=10
                velocity(MinMaxNormalization): Velocity of the mass. Default: min=-10, max=10
            action_normalizations(dict): min-max normalization values of the input/action.
                force(MinMaxNormalization): Maximum force that can be applied to the system as action. Default: min=-20, max=20
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params(dict): Parameters of environment which do not change during simulation.
                d(float): Damping constant. Default: 1
                k(float): Spring constant. Default: 100
                m(float): Mass of the oscillating object. Default: 1
            control_state (list): Components of the physical state that are considered in reference tracking.
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of MinMaxNormalization of physical_normalizations and action_normalizations as well as static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_normalizations:
            physical_normalizations = {
                "deflection": MinMaxNormalization(min=jnp.array(-10), max=jnp.array(10)),
                "velocity": MinMaxNormalization(min=jnp.array(-10), max=jnp.array(10)),
            }

        if not action_normalizations:
            action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-20), max=jnp.array(20))}

        if not static_params:
            static_params = {"k": jnp.array(100), "d": jnp.array(1), "m": jnp.array(1)}

        if not control_state:
            control_state = []

        logic = soft_constraints if soft_constraints else massspringdamper_soft_constraints
        self.soft_constraints_logic = logic
        self.control_state = control_state

        physical_normalizations = self.PhysicalState(**physical_normalizations)
        action_normalizations = self.Action(**action_normalizations)
        static_params = self.StaticParams(**static_params)

        env_properties = self.EnvProperties(
            physical_normalizations=physical_normalizations,
            action_normalizations=action_normalizations,
            static_params=static_params,
        )
        super().__init__(env_properties=env_properties, tau=tau, solver=solver)

    class PhysicalState(eqx.Module):
        """Dataclass containing the physical state of the environment."""

        deflection: jax.Array
        velocity: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class StaticParams(eqx.Module):
        """Dataclass containing the static parameters of the environment."""

        d: jax.Array
        k: jax.Array
        m: jax.Array

    class Action(eqx.Module):
        """Dataclass containing the action, that can be applied to the environment."""

        force: jax.Array

    def _ode(self, t, y, args, action):
        deflection, velocity = y
        params = args
        d_velocity = (action(t)[0] - params.d * velocity - params.k * deflection) / params.m
        d_deflection = velocity
        d_y = d_deflection, d_velocity  # [0]
        return d_y

    @eqx.filter_jit
    def _ode_solver_step(self, state, action):
        """Computes state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.

        Returns:
            state: The computed state after the one step simulation.
        """
        static_params = self.env_properties.static_params
        physical_state = state.physical_state
        args = static_params

        force = lambda t: action

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([physical_state.deflection, physical_state.velocity])

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)
        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]

        new_physical_state = self.PhysicalState(deflection=deflection_k1, velocity=velocity_k1)

        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = eqx.tree_at(lambda s: (s.physical_state, s.additions), state, (new_physical_state, new_additions))
        return new_state

    @eqx.filter_jit
    def _ode_solver_simulate_ahead(self, init_state, actions, obs_stepsize, action_stepsize):
        """Computes multiple simulation steps for one batch.

        Args:
            init_state: The initial state of the simulation.
            actions: A set of actions to be applied to the environment, the value changes every.
            action_stepsize (shape=(n_action_steps, action_dim)).
            obs_stepsize: The sampling time for the observations.
            action_stepsize: The time between changes in the input/action.

        Returns:
            next_states: The computed states during the multiple step simulation.
        """
        static_params = self.env_properties.static_params
        init_physical_state = init_state.physical_state
        args = static_params

        def force(t):
            return actions[jnp.array(t / action_stepsize, int)]

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = action_stepsize * actions.shape[0]
        init_physical_state_array, _ = tree_flatten(init_physical_state)
        y0 = tuple(init_physical_state_array)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / obs_stepsize)))  #
        sol = diffrax.diffeqsolve(
            term,
            self._solver,
            t0,
            t1,
            dt0=obs_stepsize,
            y0=y0,
            args=args,
            saveat=saveat,
        )

        deflection_t = sol.ys[0]
        velocity_t = sol.ys[1]
        obs_len = velocity_t.shape[0]

        physical_states = self.PhysicalState(deflection=deflection_t, velocity=velocity_t)
        ref = self.PhysicalState(
            deflection=jnp.full(obs_len, init_state.reference.deflection),
            velocity=jnp.full(obs_len, init_state.reference.velocity),
        )
        y0 = tuple([deflection_t[-1], velocity_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len), active_solver_state=jnp.full(obs_len, True)
        )
        PRNGKey = jnp.broadcast_to(jnp.asarray(init_state.PRNGKey), (obs_len,) + jnp.asarray(init_state.PRNGKey).shape)
        return self.State(
            physical_state=physical_states,
            PRNGKey=PRNGKey,
            additions=additions,
            reference=ref,
        )

    @eqx.filter_jit
    def init_state(self, rng: chex.PRNGKey = None):
        """Returns default or random initial state for one batch."""
        env_properties = self.env_properties
        if rng is None:
            phys = self.PhysicalState(
                deflection=jnp.array(0.0),
                velocity=jnp.array(0.0),
            )
            subkey = jnp.array(jnp.nan)
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(2,))
            phys = self.PhysicalState(
                deflection=state_norm[0],
                velocity=state_norm[1],
            )
            key, subkey = jax.random.split(rng)

        force = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.deflection, phys.velocity])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_reward(self, state, action):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state)
        for name in self.control_state:
            reward += -(((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name))) ** 2)
        return jnp.array([reward])

    @eqx.filter_jit
    def generate_observation(self, state):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state)
        norm_state_phys = norm_state.physical_state
        obs = jnp.hstack(
            (
                norm_state_phys.deflection,
                norm_state_phys.velocity,
            )
        )
        for name in self.control_state:
            obs = jnp.hstack(
                (
                    obs,
                    getattr(norm_state.reference, name),
                )
            )
        return obs

    @eqx.filter_jit
    def generate_state_from_observation(self, obs, key=None):
        """Generates state from observation for one batch."""
        env_properties = self.env_properties
        phys = self.PhysicalState(
            deflection=obs[0],
            velocity=obs[1],
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        force = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=force)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.deflection, phys.velocity])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan)
        new_ref = ref
        for i, name in enumerate(self.control_state):
            new_ref = eqx.tree_at(lambda r: getattr(r, name), new_ref, obs[2 + i])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_truncated(self, state):
        """Returns truncated information for one batch."""
        obs = self.generate_observation(state)
        return jnp.abs(obs) > 1

    @eqx.filter_jit
    def generate_terminated(self, state, reward):
        """Returns terminated information for one batch."""
        return reward == 0

    @property
    def obs_description(self):
        return np.hstack(
            [
                np.array(["deflection", "velocity"]),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    @property
    def action_description(self):
        return np.array(["force"])
