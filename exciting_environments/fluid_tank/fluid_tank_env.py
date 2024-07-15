from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure
import jax_dataclasses as jdc
import chex
import diffrax

from exciting_environments import core_env


class FluidTank(core_env.CoreEnvironment):
    """Fluid tank based on torricelli's principle.

    Based on ex. 7.3.2 on p. 355 of "System Dynamics" from Palm, William III.
    """

    def __init__(
        self,
        batch_size: float = 1,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        solver=diffrax.Euler(),
        reward_func: Callable = None,
        tau: float = 1e-3,
    ):
        if not physical_constraints:
            physical_constraints = {"height": 3}

        if not action_constraints:
            action_constraints = {"inflow": 0.2}

        if not static_params:
            # c_d = 0.6 typical value for water [Palm2010]
            static_params = {"base_area": jnp.pi, "orifice_area": jnp.pi * 0.1**2, "c_d": 0.6, "g": 9.81}

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
            reward_func=reward_func,
        )

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical state of the environment."""

        height: jax.Array

    @jdc.pytree_dataclass
    class Optional:
        """Dataclass containing additional information for simulation."""

        something: jax.Array

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
        physical_state = state.physical_state

        action = action / jnp.array(tree_flatten(self.env_properties.action_constraints)[0]).T
        action = (action + 1) / 2
        action = action * jnp.array(tree_flatten(self.env_properties.action_constraints)[0]).T

        args = (action, static_params)

        def vector_field(t, y, args):
            h = y[0]
            inflow, params = args

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
        h_k1 = jnp.clip(h_k1, min=0)

        phys = self.PhysicalState(height=h_k1)
        opt = None  # Optional(something=...)
        return self.State(physical_state=phys, PRNGKey=None, optional=None)

    @partial(jax.jit, static_argnums=0)
    def default_reward_func(self, obs, action, action_constraints):
        return 0

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, states, physical_constraints):
        return (states.physical_state.height - physical_constraints.height / 2) / (physical_constraints.height / 2)

    @partial(jax.jit, static_argnums=0)
    def generate_truncated(self, states, physical_constraints):
        return 0

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, states, reward):
        return False

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["fluid height"])

    @property
    def action_description(self):
        return np.array(["inflow"])

    @partial(jax.jit, static_argnums=0)
    def init_state(self):
        phys = self.PhysicalState(height=jnp.full(self.batch_size, self.env_properties.physical_constraints.height / 2))
        opt = None  # self.Optional(something=jnp.zeros(self.batch_size))
        return self.State(physical_state=phys, PRNGKey=None, optional=opt)

    def reset(self, rng: jax.random.PRNGKey = None, initial_state: jdc.pytree_dataclass = None):
        if initial_state is not None:
            assert tree_structure(self.init_state()) == tree_structure(
                initial_state
            ), f"initial_state should have the same dataclass structure as self.init_state()"
            state = initial_state
        else:
            state = self.init_state()

        obs = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties.physical_constraints),
        )(state, self.env_properties.physical_constraints)

        # TODO: this [None] looks off -> investigate
        return obs[None], state
