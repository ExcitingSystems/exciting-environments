from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure
import jax_dataclasses as jdc
import chex
import diffrax
from dataclasses import fields

from exciting_environments import CoreEnvironment
from exciting_environments.utils import MinMaxNormalization


class FluidTank(CoreEnvironment):
    """Fluid tank based on torricelli's principle.

    Based on ex. 7.3.2 on p. 355 of "System Dynamics" from Palm, William III.
    """

    def __init__(
        self,
        batch_size: float = 1,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-3,
    ):
        if not physical_normalizations:
            physical_normalizations = {"height": MinMaxNormalization(min=0, max=3)}

        if not action_normalizations:
            action_normalizations = {"inflow": MinMaxNormalization(min=0, max=0.2)}

        if not soft_constraints:
            soft_constraints = self.default_soft_constraints

        if not static_params:
            # c_d = 0.6 typical value for water [Palm2010]
            static_params = {"base_area": jnp.pi, "orifice_area": jnp.pi * 0.1**2, "c_d": 0.6, "g": 9.81}

        if not control_state:
            control_state = []

        self.control_state = control_state
        self.soft_constraints = soft_constraints

        physical_normalizations = self.PhysicalState(**physical_normalizations)
        action_normalizations = self.Action(**action_normalizations)
        static_params = self.StaticParams(**static_params)

        env_properties = self.EnvProperties(
            physical_normalizations=physical_normalizations,
            action_normalizations=action_normalizations,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        height: jax.Array

    @jdc.pytree_dataclass
    class Additions:
        """Dataclass containing additional information for simulation."""

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        base_area: jax.Array
        orifice_area: jax.Array
        c_d: jax.Array
        g: jax.Array

    @jdc.pytree_dataclass
    class Action:
        """Dataclass containing the action, that can be applied to the environment."""

        inflow: jax.Array

    @partial(jax.jit, static_argnums=0)
    def _ode_solver_step(self, state, action, static_params):
        """Computes the next state by simulating one step.

        Args:
            state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
            static_params: Parameter of the environment, that do not change over time.

        Returns:
            next_state: The computed next state after the one step simulation.
        """
        physical_state = state.physical_state

        args = (action, static_params)

        def vector_field(t, y, args):
            h = y[0]
            inflow, params = args

            h = jnp.clip(h, 0)

            dh_dt = inflow[0] / params.base_area - params.c_d * params.orifice_area / params.base_area * jnp.sqrt(
                2 * params.g * h
            )
            return (dh_dt,)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = (physical_state.height,)

        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        h_k1 = y[0]

        # clip to 0 because tank cannot be more empty than empty
        # necessary because of ODE solver approximation
        h_k1 = jnp.clip(h_k1, 0)

        with jdc.copy_and_mutate(state, validate=True) as new_state:
            new_state.physical_state = self.PhysicalState(height=h_k1)
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
        raise NotImplementedError("To be implemented!")

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default or random initial state for one batch."""
        if rng is None:
            phys = self.PhysicalState(
                height=0.0,
            )
            subkey = jnp.nan
        else:
            state_norm = jax.random.uniform(rng, minval=0, maxval=1, shape=(1,))
            phys = self.PhysicalState(
                height=state_norm[0],
            )
            key, subkey = jax.random.split(rng)
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(height=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state, env_properties)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state, env_properties)
        for name in self.control_state:
            reward += -((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name)) ** 2)
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state, env_properties)
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

    @partial(jax.jit, static_argnums=0)
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        phys = self.PhysicalState(
            height=obs[0],
        )
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        additions = None
        ref = self.PhysicalState(height=jnp.nan)
        with jdc.copy_and_mutate(ref, validate=False) as new_ref:
            for name, pos in zip(self.control_state, range(len(self.control_state))):
                value = obs[1 + pos]
                setattr(new_ref, name, value)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state, env_properties)

    def default_soft_constraints(self, state, action_norm, env_properties):
        state_norm = self.normalize_state(state, env_properties)
        physical_state_norm = state_norm.physical_state
        with jdc.copy_and_mutate(physical_state_norm, validate=False) as phys_soft_const:
            for field in fields(phys_soft_const):
                name = field.name
                setattr(phys_soft_const, name, jnp.nan)

        # define soft constraints for action
        act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
        return phys_soft_const, act_soft_constr

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, state, env_properties):
        """Returns truncated information for one batch."""
        return jnp.array([0])

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information for one batch."""
        return jnp.array([False])

    @property
    def obs_description(self):
        return np.hstack([self.states_description, np.array([name + "_ref" for name in self.control_state])])

    @property
    def states_description(self):
        return np.array(["fluid height"])

    @property
    def action_description(self):
        return np.array(["inflow"])
