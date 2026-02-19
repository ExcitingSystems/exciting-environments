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


def fluidtank_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize(state)
    physical_state_norm = state_norm.physical_state
    phys_soft_const = jax.tree.map(lambda _: jnp.nan, physical_state_norm)

    # define soft constraints for action
    act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
    return phys_soft_const, act_soft_constr


class FluidTank(CoreEnvironment):
    control_state: list = eqx.field(static=True)
    soft_constraints_logic: Callable = eqx.field(static=True)
    """Fluid tank based on torricelli's principle.

    Based on ex. 7.3.2 on p. 355 of "System Dynamics" from Palm, William III.
    """

    def __init__(
        self,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-3,
    ):
        if not physical_normalizations:
            physical_normalizations = {"height": MinMaxNormalization(min=jnp.array(0), max=jnp.array(3))}

        if not action_normalizations:
            action_normalizations = {"inflow": MinMaxNormalization(min=jnp.array(0), max=jnp.array(0.2))}

        if not static_params:
            # c_d = 0.6 typical value for water [Palm2010]
            static_params = {
                "base_area": jnp.array(jnp.pi),
                "orifice_area": jnp.array(jnp.pi * 0.1**2),
                "c_d": jnp.array(0.6),
                "g": jnp.array(9.81),
            }

        if not control_state:
            control_state = []

        logic = soft_constraints if soft_constraints else fluidtank_soft_constraints
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

        height: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class StaticParams(eqx.Module):
        """Dataclass containing the static parameters of the environment."""

        base_area: jax.Array
        orifice_area: jax.Array
        c_d: jax.Array
        g: jax.Array

    class Action(eqx.Module):
        """Dataclass containing the action, that can be applied to the environment."""

        inflow: jax.Array

    def _ode(self, t, y, args, action):
        h = y[0]
        params = args

        h = jnp.clip(h, 0)

        dh_dt = action(t)[0] / params.base_area - params.c_d * params.orifice_area / params.base_area * jnp.sqrt(
            2 * params.g * h
        )
        return (dh_dt,)

    @eqx.filter_jit
    def _ode_solver_step(self, state, action):
        """Computes the next state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.

        Returns:
            next_state: The computed next state after the one step simulation.
        """
        static_params = self.env_properties.static_params
        physical_state = state.physical_state

        args = static_params

        inflow = lambda t: action

        vector_field = partial(self._ode, action=inflow)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = (physical_state.height,)

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)
        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        h_k1 = y[0]

        # clip to 0 because tank cannot be more empty than empty
        # necessary because of ODE solver approximation
        h_k1 = jnp.clip(h_k1, 0)

        new_physical_state = self.PhysicalState(height=h_k1)
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

        def inflow(t):
            return actions[jnp.array(t / action_stepsize, int)]

        vector_field = partial(self._ode, action=inflow)

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

        height_t = jnp.clip(sol.ys[0], 0)
        obs_len = height_t.shape[0]

        physical_states = self.PhysicalState(
            height=height_t,
        )
        y0 = tuple([height_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len), active_solver_state=jnp.full(obs_len, True)
        )
        PRNGKey = jnp.broadcast_to(jnp.asarray(init_state.PRNGKey), (obs_len,) + jnp.asarray(init_state.PRNGKey).shape)
        ref = self.PhysicalState(
            height=jnp.full(obs_len, init_state.reference.height),
        )
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
                height=jnp.array(0.0),
            )
            subkey = jnp.array(jnp.nan)
        else:
            state_norm = jax.random.uniform(rng, minval=0, maxval=1, shape=(1,))
            phys = self.PhysicalState(
                height=state_norm[0],
            )
            key, subkey = jax.random.split(rng)

        inflow = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=inflow)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.height])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(height=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_reward(self, state, action):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state)
        for name in self.control_state:
            reward += -((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name)) ** 2)
        return jnp.array([reward])

    @eqx.filter_jit
    def generate_observation(self, state):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state)
        norm_state_phys = norm_state.physical_state
        obs = (norm_state_phys.height)[None]
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
            height=obs[0],
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan

        inflow = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=inflow)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.height])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)

        ref = self.PhysicalState(height=jnp.nan)
        new_ref = ref
        for i, name in enumerate(self.control_state):
            new_ref = eqx.tree_at(lambda r: getattr(r, name), new_ref, obs[1 + i])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state)

    @eqx.filter_jit
    def generate_truncated(self, state):
        """Returns truncated information for one batch."""
        return jnp.array([0])

    @eqx.filter_jit
    def generate_terminated(self, state, reward):
        """Returns terminated information for one batch."""
        return jnp.array([False])

    @property
    def obs_description(self):
        return np.hstack(
            [
                self.states_description,
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    @property
    def states_description(self):
        return np.array(["fluid height"])

    @property
    def action_description(self):
        return np.array(["inflow"])
