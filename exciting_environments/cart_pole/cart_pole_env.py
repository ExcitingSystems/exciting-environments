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


class CartPole(ClassicCoreEnvironment):
    """
    State Variables
        ``['deflection', 'velocity', 'theta', 'omega']``

    Action Variable:
        ``['force']``

    Initial State:
        Unless chosen otherwise, deflection, omega and velocity is set to zero and theta is set to pi.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import exciting_environments as excenvs
        >>> from exciting_environments import GymWrapper
        >>>
        >>> # Create the environment
        >>> cartpole= excenv.CartPole(batch_size=5)
        >>>
        >>> # Use GymWrapper for Simulation (optional)
        >>> gym_cartpole=GymWrapper(env=cartpole)
        >>>
        >>> # Reset the environment with default initial values
        >>> gym_cartpole.reset()
        >>>
        >>> # Perform step
        >>> obs,reward,terminated,truncated = gym_cartpole.step(action=jnp.ones(5).reshape(-1,1))
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
        tau: float = 2e-2,
    ):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical state of the environment.
                deflection(float): Deflection of the cart. Default: 10
                velocity(float): Velocity of the cart. Default: 10
                theta(float): Rotation angle of the pole. Default: jnp.pi
                omega(float): Angular velocity. Default: 10
            action_constraints(dict): Constraints of the action.
                force(float): Maximum torque that can be applied to the system as action. Default: 20
            static_params(dict): Parameters of environment which do not change during simulation.
                mu_p(float): Coefficient of friction of pole on cart. Default: 0
                mu_c(float): Coefficient of friction of cart on track. Default: 0
                l(float): Half-pole length. Default: 1
                m_p(float): Mass of the pole. Default: 1
                m_c(float): Mass of the cart. Default: 1
                g(float): Gravitational acceleration. Default: 9.81
            control_state: TODO
            solver(diffrax.solver): Solver used to compute state for next step.
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        if not physical_constraints:
            physical_constraints = {
                "deflection": 2.4,
                "velocity": 8,
                "theta": jnp.pi,
                "omega": 8,
            }
        if not action_constraints:
            action_constraints = {"force": 20}

        if not static_params:
            static_params = {  # typical values from Source with DOI: 10.1109/TSMC.1983.6313077
                "mu_p": 0.000002,
                "mu_c": 0.0005,
                "l": 0.5,
                "m_p": 0.1,
                "m_c": 1,
                "g": 9.81,
            }

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
        theta: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Additions:
        """Dataclass containing additional information for simulation."""

        something: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        mu_p: jax.Array
        mu_c: jax.Array
        l: jax.Array
        m_p: jax.Array
        m_c: jax.Array
        g: jax.Array

    @jdc.pytree_dataclass
    class Action:
        """Dataclass containing the action that can be applied to the environment."""

        force: jax.Array

    @partial(jax.jit, static_argnums=0)
    def _ode_solver_step(self, state, action, static_params):
        """Computes state by simulating one step.

        Source DOI: 10.1109/TSMC.1983.6313077

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
            deflection, velocity, theta, omega = y
            action, params = args
            d_omega = (
                params.g * jnp.sin(theta)
                + jnp.cos(theta)
                * (
                    (
                        -action[0]
                        - params.m_p * params.l * (omega**2) * jnp.sin(theta)
                        + params.mu_c * jnp.sign(velocity)
                    )
                    / (params.m_c + params.m_p)
                )
                - (params.mu_p * omega) / (params.m_p * params.l)
            ) / (params.l * (4 / 3 - (params.m_p * (jnp.cos(theta)) ** 2) / (params.m_c + params.m_p)))

            d_velocity = (
                action[0]
                + params.m_p * params.l * ((omega**2) * jnp.sin(theta) - d_omega * jnp.cos(theta))
                - params.mu_c * jnp.sign(velocity)
            ) / (params.m_c + params.m_p)
            d_theta = omega
            d_deflection = velocity
            d_y = d_deflection, d_velocity, d_theta, d_omega
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple(
            [
                physical_state.deflection,
                physical_state.velocity,
                physical_state.theta,
                physical_state.omega,
            ]
        )
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        deflection_k1 = y[0]
        velocity_k1 = y[1]
        theta_k1 = y[2]
        omega_k1 = y[3]
        theta_k1 = ((theta_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        with jdc.copy_and_mutate(state, validate=False) as new_state:
            new_state.physical_state = self.PhysicalState(
                deflection=deflection_k1,
                velocity=velocity_k1,
                theta=theta_k1,
                omega=omega_k1,
            )
        return new_state

    @partial(jax.jit, static_argnums=[0, 4, 5])
    def _ode_solver_simulate_ahead(self, init_state, actions, static_params, obs_stepsize, action_stepsize):
        """Computes states by simulating a trajectory with given actions."""

        init_physical_state = init_state.physical_state
        args = (actions, static_params)

        def force(t, args):
            actions = args
            return actions[jnp.array(t / action_stepsize, int), 0]

        def vector_field(t, y, args):
            deflection, velocity, theta, omega = y
            actions, params = args
            d_omega = (
                params.g * jnp.sin(theta)
                + jnp.cos(theta)
                * (
                    (
                        -force(t, actions)
                        - params.m_p * params.l * (omega**2) * jnp.sin(theta)
                        + params.mu_c * jnp.sign(velocity)
                    )
                    / (params.m_c + params.m_p)
                )
                - (params.mu_p * omega) / (params.m_p * params.l)
            ) / (params.l * (4 / 3 - (params.m_p * (jnp.cos(theta)) ** 2) / (params.m_c + params.m_p)))

            d_velocity = (
                force(t, actions)
                + params.m_p * params.l * ((omega**2) * jnp.sin(theta) - d_omega * jnp.cos(theta))
                - params.mu_c * jnp.sign(velocity)
            ) / (params.m_c + params.m_p)
            d_theta = omega
            d_deflection = velocity
            d_y = d_deflection, d_velocity, d_theta, d_omega
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
        theta_t = sol.ys[2]
        omega_t = sol.ys[3]

        # keep theta between -pi and pi
        theta_t = ((theta_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        physical_states = self.PhysicalState(deflection=deflection_t, velocity=velocity_t, theta=theta_t, omega=omega_t)
        additions = None
        PRNGKey = None
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan, theta=jnp.nan, omega=jnp.nan)
        return self.State(physical_state=physical_states, PRNGKey=PRNGKey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        """Returns default initial state for all batches."""
        if rng is None:
            phys = self.PhysicalState(
                deflection=0.0,
                velocity=0.0,
                theta=jnp.pi,
                omega=0.0,
            )
            subkey = None
        else:
            state_norm = jax.random.uniform(rng, minval=-1, maxval=1, shape=(4,))
            phys = self.PhysicalState(
                deflection=state_norm[0] * env_properties.physical_constraints.deflection,
                velocity=state_norm[1] * env_properties.physical_constraints.velocity,
                theta=state_norm[2] * env_properties.physical_constraints.theta,
                omega=state_norm[3] * env_properties.physical_constraints.omega,
            )
            key, subkey = jax.random.split(rng)
        additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
        ref = self.PhysicalState(deflection=jnp.nan, velocity=jnp.nan, theta=jnp.nan, omega=jnp.nan)
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)

    @partial(jax.jit, static_argnums=0)
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0, 0))(
            self.env_properties, rng, jnp.ones(self.batch_size)
        )

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
                state.physical_state.theta / physical_constraints.theta,
                state.physical_state.omega / physical_constraints.omega,
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
    def generate_truncated(self, state, env_properties):
        """Returns truncated information for one batch."""
        obs = self.generate_observation(state, env_properties)
        return jnp.abs(obs) > 1

    @partial(jax.jit, static_argnums=0)
    def generate_terminated(self, state, reward, env_properties):
        """Returns terminated information for one batch."""
        return reward == 0

    @property
    def action_description(self):
        return np.array(["force"])

    @property
    def obs_description(self):
        return np.hstack(
            [
                np.array(["deflection", "velocity", "theta", "omega"]),
                np.array([name + "_ref" for name in self.control_state]),
            ]
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
