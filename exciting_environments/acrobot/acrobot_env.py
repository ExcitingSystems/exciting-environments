from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_structure, tree_map

import diffrax
import equinox as eqx
import chex
from dataclasses import fields
from exciting_environments.utils import MinMaxNormalization

from exciting_environments import CoreEnvironment


class Acrobot(CoreEnvironment):
    """
    State Variables:
        ``['theta_1', 'theta_2', 'omega_1', 'omega_2']``

    Action Variable:
        ``['torque']``

    Initial State:
        Unless chosen otherwise, theta_1=pi, theta_2=0 and omega_1=omega_2=0

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>>
        >>> # Create the environment
        >>> acrobot=excenvs.Acrobot(batch_size=4)
        >>>
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_acrobot=GymWrapper(env=acrobot)
        >>>
        >>> # Reset the environment with default initial values
        >>> gym_acrobot.reset()
        >>>
        >>> # Perform step
        >>> obs, reward, terminated,  truncated = gym_acrobot.step(action=jnp.ones(4).reshape(-1,1))
        >>>

    """

    def __init__(
        self,
        batch_size: int = 8,
        physical_normalizations: dict = None,
        action_normalizations: dict = None,
        soft_constraints: Callable = None,
        static_params: dict = None,
        control_state: list = None,
        solver=diffrax.Euler(),
        tau: float = 1e-3,
    ):
        """
        Args:
            batch_size (int): Number of parallel environment simulations. Default: 8
            physical_normalizations (dict): Min and max values of the physical state of the environment for normalization.
                theta1 (MinMaxNormalization): Rotation angle of the first/inner joint. Default: min=-jnp.pi, max=jnp.pi
                theta2 (MinMaxNormalization): Rotation angle relative to theta1 of second/outer joint. Default: min=-jnp.pi, max=jnp.pi
                omega_1 (MinMaxNormalization): Angular velocity of first joint. Default: min=-10, max=10
                omega_2 (MinMaxNormalization): Angular velocity of second joint. Default: min=-10, max=10
            action_normalizations (dict): Min and max values of the input/action for normalization.
                torque (MinMaxNormalization): Maximum torque that can be applied to the second joint as an action. Default: min=-20, max=20
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params (dict): Parameters of environment which do not change during simulation.
                g (float): Gravitational acceleration. Default: 9.81
                l_1 (float): Length of the first link. Default: 2
                l_2 (float): Length of the second link. Default: 2
                m_1 (float): Mass of the first link. Default: 1
                m_2 (float): Mass of the second link. Default: 1
                l_c1 (float): Distance from the base to the center of mass of the first link. Default: 1
                l_c2 (float): Distance from the first joint to the center of mass of the second link. Default: 1
                I_1 (float): Moment of inertia of the first link about its center of mass. Default: 1.3
                I_2 (float): Moment of inertia of the first link about its center of mass. Default: 1.3

            control_state (list): Components of the physical state that are considered in reference tracking.
            solver (diffrax.solver): Solver used to compute state for next step.
            tau (float): Duration of one control step in seconds. Default: 1e-3.

        Note: Attributes of MinMaxNormalization of physical_normalizations and action_normalizations as well as static_params can also be
            passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_normalizations:
            physical_normalizations = {
                "theta_1": MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
                "theta_2": MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
                "omega_1": MinMaxNormalization(min=-10, max=10),
                "omega_2": MinMaxNormalization(min=-10, max=10),
            }

        if not action_normalizations:
            action_normalizations = {"torque": MinMaxNormalization(min=-20, max=20)}

        if not soft_constraints:
            soft_constraints = self.default_soft_constraints

        if not static_params:
            static_params = {
                "g": 9.81,
                "l_1": 2,
                "l_2": 2,
                "m_1": 1,
                "m_2": 1,
                "l_c1": 1,
                "l_c2": 1,
                "I_1": 1.3,
                "I_2": 1.3,
            }

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

    class PhysicalState(eqx.Module):
        """Dataclass containing the physical state of the environment."""

        theta_1: jax.Array
        theta_2: jax.Array
        omega_1: jax.Array
        omega_2: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class StaticParams(eqx.Module):
        """Dataclass containing the static parameters of the environment."""

        g: jax.Array
        l_1: jax.Array
        l_2: jax.Array
        m_1: jax.Array
        m_2: jax.Array
        l_c1: jax.Array
        l_c2: jax.Array
        I_1: jax.Array
        I_2: jax.Array

    class Action(eqx.Module):
        """Dataclass containing the action, that can be applied to the environment."""

        torque: jax.Array

    def _ode(self, t, y, args, action):
        theta_1, theta_2, omega_1, omega_2 = y
        params = args
        d_11 = (
            params.m_1 * params.l_c1**2
            + params.m_2 * (params.l_1**2 + params.l_c2**2 + 2 * params.l_1 * params.l_c2 * jnp.cos(theta_2))
            + params.I_1
            + params.I_2
        )
        d_12 = params.m_2 * (params.l_c2**2 + params.l_1 * params.l_c2 * jnp.cos(theta_2)) + params.I_2
        d_22 = params.m_2 * params.l_c2**2 + params.I_2
        h_1 = (
            -params.m_2 * params.l_1 * params.l_c2 * jnp.sin(theta_2) * omega_2**2
            - 2 * params.m_2 * params.l_1 * params.l_c2 * jnp.sin(theta_2) * omega_1 * omega_2
        )
        h_2 = params.m_2 * params.l_1 * params.l_c2 * jnp.sin(theta_2) * omega_1**2
        phi_1 = (params.m_1 * params.l_c1 + params.m_2 * params.l_1) * params.g * jnp.cos(
            theta_1 + jnp.pi / 2
        ) + params.m_2 * params.l_c2 * params.g * jnp.cos(theta_1 + theta_2 + jnp.pi / 2)
        phi_2 = params.m_2 * params.l_c2 * params.g * jnp.cos(theta_1 + theta_2 + jnp.pi / 2)
        d_omega_1 = 1 / (d_12 - d_22 / d_12 * d_11) * (action(t)[0] + d_22 / d_12 * (h_1 + phi_1) - h_2 - phi_2)
        d_omega_2 = (-d_11 * d_omega_1 - h_1 - phi_1) / d_12
        d_theta_1 = omega_1
        d_theta_2 = omega_2
        d_y = d_theta_1, d_theta_2, d_omega_1, d_omega_2

        return d_y

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
        args = static_params

        torque = lambda t: action

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple(
            [
                physical_state.theta_1,
                physical_state.theta_2,
                physical_state.omega_1,
                physical_state.omega_2,
            ]
        )

        def false_fn(_):
            return self.Additions(
                solver_state=self._solver.init(term, t0, t1, y0, args),
                active_solver_state=True,
            )

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)
        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        theta_1_k1 = y[0]
        theta_2_k1 = y[1]
        omega_1_k1 = y[2]
        omega_2_k1 = y[3]
        theta_1_k1 = ((theta_1_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        theta_2_k1 = ((theta_2_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        with jdc.copy_and_mutate(state, validate=True) as new_state:
            new_state.physical_state = self.PhysicalState(
                theta_1=theta_1_k1,
                theta_2=theta_2_k1,
                omega_1=omega_1_k1,
                omega_2=omega_2_k1,
            )
        new_state = jdc.replace(
            new_state,
            additions=self.Additions(solver_state=solver_state_k1, active_solver_state=True),
        )
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
        args = static_params

        def torque(t):
            return actions[jnp.array(t / action_stepsize, int)]

        vector_field = partial(self._ode, action=torque)

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

        theta_1_t = sol.ys[0]
        theta_2_t = sol.ys[1]
        omega_1_t = sol.ys[2]
        omega_2_t = sol.ys[3]

        obs_len = theta_1_t.shape[0]
        # keep thetas between -pi and pi
        theta_1_t = ((theta_1_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        theta_2_t = ((theta_2_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        physical_states = self.PhysicalState(theta_1=theta_1_t, theta_2=theta_2_t, omega_1=omega_1_t, omega_2=omega_2_t)
        ref = self.PhysicalState(
            theta_1=jnp.full(obs_len, init_state.reference.theta_1),
            theta_2=jnp.full(obs_len, init_state.reference.theta_2),
            omega_1=jnp.full(obs_len, init_state.reference.omega_1),
            omega_2=jnp.full(obs_len, init_state.reference.omega_2),
        )
        y0 = tuple([theta_1_t[-1], theta_2_t[-1], omega_1_t[-1], omega_2_t[-1]])
        solver_state = self._solver.init(term, t1, t1 + self.tau, y0, args)
        additions = self.Additions(
            solver_state=self.repeat_values(solver_state, obs_len),
            active_solver_state=jnp.full(obs_len, True),
        )
        PRNGKey = jnp.full(obs_len, init_state.PRNGKey)
        return self.State(
            physical_state=physical_states,
            PRNGKey=PRNGKey,
            additions=additions,
            reference=ref,
        )

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default or random initial state for one batch."""
        if rng is None:
            phys = self.PhysicalState(
                theta_1=1.0,
                theta_2=0.0,
                omega_1=0.0,
                omega_2=0.0,
            )
            subkey = jnp.nan
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(4,))
            phys = self.PhysicalState(
                theta_1=state_norm[0],
                theta_2=state_norm[1],
                omega_1=state_norm[2],
                omega_2=state_norm[3],
            )
            key, subkey = jax.random.split(rng)

        torque = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.theta_1, phys.theta_2, phys.omega_1, phys.omega_2])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = tree_map(lambda x: x * jnp.nan, solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        ref = self.PhysicalState(theta_1=jnp.nan, theta_2=jnp.nan, omega_1=jnp.nan, omega_2=jnp.nan)
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)
        return self.denormalize_state(norm_state, env_properties)

    @partial(jax.jit, static_argnums=0)
    def generate_reward(self, state, action, env_properties):
        """Returns reward for one batch."""
        reward = 0
        norm_state = self.normalize_state(state, env_properties)
        for name in self.control_state:
            if name == "theta_1" or name == "theta_2":
                # For theta, we use the sine and cosine to avoid discontinuities at pi
                theta = getattr(state.physical_state, name)
                theta_ref = getattr(state.reference, name)
                reward += -((jnp.sin(theta) - jnp.sin(theta_ref)) ** 2 + (jnp.cos(theta) - jnp.cos(theta_ref)) ** 2)
            else:
                reward += -((getattr(norm_state.physical_state, name) - getattr(norm_state.reference, name)) ** 2)
        return jnp.array([reward])

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        """Returns observation for one batch."""
        norm_state = self.normalize_state(state, env_properties)
        norm_state_phys = norm_state.physical_state
        obs = jnp.hstack(
            (
                norm_state_phys.theta_1,
                norm_state_phys.theta_2,
                norm_state_phys.omega_1,
                norm_state_phys.omega_2,
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

    @partial(jax.jit, static_argnums=0)
    def generate_state_from_observation(self, obs, env_properties, key=None):
        """Generates state from observation for one batch."""
        phys = self.PhysicalState(
            theta_1=obs[0],
            theta_2=obs[1],
            omega_1=obs[2],
            omega_2=obs[3],
        )

        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan

        torque = lambda t: jnp.array([0])

        args = env_properties.static_params

        vector_field = partial(self._ode, action=torque)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([phys.theta_1, phys.theta_2, phys.omega_1, phys.omega_2])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = tree_map(lambda x: x * jnp.nan, solver_state)

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)  # None
        ref = self.PhysicalState(theta_1=jnp.nan, theta_2=jnp.nan, omega_1=jnp.nan, omega_2=jnp.nan)
        with jdc.copy_and_mutate(ref, validate=False) as new_ref:
            for name, pos in zip(self.control_state, range(len(self.control_state))):
                setattr(new_ref, name, obs[4 + pos])
        norm_state = self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=new_ref)
        return self.denormalize_state(norm_state, env_properties)

    def default_soft_constraints(self, state, action_norm, env_properties):
        state_norm = self.normalize_state(state, env_properties)
        physical_state_norm = state_norm.physical_state
        with jdc.copy_and_mutate(physical_state_norm, validate=False) as phys_soft_const:
            for field in fields(phys_soft_const):
                name = field.name
                setattr(phys_soft_const, name, jnp.nan)
            # define soft constraints for physical state
            soft_constr = jax.nn.relu(jnp.abs(getattr(physical_state_norm, "omega")) - 1.0)
            setattr(phys_soft_const, "omega", soft_constr)

        # define soft constraints for action
        act_soft_constr = jax.nn.relu(jnp.abs(action_norm) - 1.0)
        return phys_soft_const, act_soft_constr

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
            [
                np.array(["theta_1", "theta_2", "omega_1", "omega_2"]),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    @property
    def action_description(self):
        return np.array(["torque"])
