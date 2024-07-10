import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
import jax_dataclasses as jdc
import chex
from functools import partial
import diffrax
from exciting_environments import CoreEnvironment
from typing import Callable


t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])  # only for alpha/beta -> abc
t23 = 2 / 3 * jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]]).T  # only for abc -> alpha/beta

inverter_t_abc = jnp.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
)


def t_dq_alpha_beta(eps):
    cos = jnp.cos(eps)
    sin = jnp.sin(eps)
    return jnp.column_stack((cos, sin, -sin, cos)).reshape(2, 2)


def dq2abc(u_dq, eps):
    u_abc = t32 @ dq2albet(u_dq, eps).T
    return u_abc.T


def dq2albet(u_dq, eps):
    q = t_dq_alpha_beta(-eps)
    u_alpha_beta = q @ u_dq.T

    return u_alpha_beta.T


def albet2dq(u_albet, eps):
    q_inv = t_dq_alpha_beta(eps)
    u_dq = q_inv @ u_albet.T

    return u_dq.T


def abc2dq(u_abc, eps):
    u_alpha_beta = t23 @ u_abc.T
    u_dq = albet2dq(u_alpha_beta.T, eps)
    return u_dq


def step_eps(eps, omega_el, tau, tau_scale=1.0):
    eps += omega_el * tau * tau_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, tau):
    eps_advanced = step_eps(eps, omega_el, tau, 0.5)
    u_abc = dq2abc(u_dq, eps_advanced)
    # clip in abc coordinates
    u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
    u_dq = abc2dq(u_abc, eps)
    return u_dq


def switching_state_to_dq(switching_state, u_dc, eps):
    u_abc = inverter_t_abc[switching_state] * u_dc
    u_dq = abc2dq(u_abc, eps)
    return u_dq[0]


def currents_to_torque(i_d, i_q, p, psi_p, l_d, l_q):
    torque = 1.5 * p * (psi_p + (l_d - l_q) * i_d) * i_q
    return torque


def calc_max_torque(l_d, l_q, i_n, psi_p, p):
    i_d = jnp.where(l_d == l_q, 0, -psi_p / (4 * (l_d - l_q)) - jnp.sqrt((psi_p / (4 * (l_d - l_q))) ** 2 + i_n**2 / 2))
    i_q = jnp.sqrt(i_n**2 - i_d**2)
    max_torque = 1.5 * p * (psi_p + (l_d - l_q) * i_d) * i_q
    return max_torque


class PMSM_Physical:
    def __init__(
        self,
        batch_size: int = 8,
        params: dict = None,
        deadtime: int = 1,
        tau: float = 1e-4,
        solver: Callable = diffrax.Euler(),
        control_state="torque",
        control_set="fcs",
    ):

        if not params:
            params = {
                "p": 3,
                "r_s": 1,
                "l_d": 0.37e-3,
                "l_q": 1.2e-3,
                "psi_p": 65.6e-3,
                "u_dc": 400,
                "i_n": 400,
                "omega_el": 100 / 60 * 2 * jnp.pi,
            }

        # self.params = params #TODO: In the future, params will be part of the state because they can change over time
        self.batch_size = batch_size  # params["p"].shape[0]
        self.tau = tau
        self._solver = solver

        if control_set == "ccs":
            self._action_description = ["u_d", "u_q"]
        elif control_set == "fcs":
            self._action_description = ["switching_state"]

        params = self.PhysicalParams(**params)
        self.properties = self.Properties(
            control_set=control_set, control_state=control_state, physical_params=params, deadtime=deadtime
        )

        if deadtime > 0:
            if control_set == "ccs":
                self.initial_action_buffer = jnp.zeros((self.batch_size, deadtime, 2))
            elif control_set == "fcs":
                initial_switching_state = jnp.zeros((self.batch_size, 1), dtype=int)
                initial_eps = jnp.zeros((self.batch_size, 1))
                initial_u_dq = jax.vmap(switching_state_to_dq, in_axes=(0, None, 0))(
                    initial_switching_state, params.u_dc, initial_eps
                )
                initial_u_dq_expanded = initial_u_dq[:, None, :]  # Reshape to (8, 1, 2)
                self.initial_action_buffer = jnp.tile(initial_u_dq_expanded, (1, deadtime, 1))
        else:
            self.initial_action_buffer = None

    @jdc.pytree_dataclass
    class PhysicalParams:
        """Dataclass containing the physical parameters of the environment."""

        p: jax.Array
        r_s: jax.Array
        l_d: jax.Array
        l_q: jax.Array
        psi_p: jax.Array
        u_dc: jax.Array
        i_n: jax.Array
        omega_el: jax.Array

    @jdc.pytree_dataclass
    class PhysicalState:
        """Dataclass containing the physical parameters of the environment."""

        action_buffer: jax.Array
        epsilon: jax.Array
        i_d: jax.Array
        i_q: jax.Array
        torque: jax.Array
        omega: jax.Array

    @jdc.pytree_dataclass
    class Properties:
        """Dataclass containing the physical parameters of the environment."""

        deadtime: jax.Array
        control_state: jax.Array
        control_set: jax.Array
        physical_params: jdc.pytree_dataclass

    @partial(jax.jit, static_argnums=0)
    def init_states(self):
        """Returns default initial states for all batches."""
        state = self.PhysicalState(
            action_buffer=self.initial_action_buffer,
            epsilon=jnp.zeros(self.batch_size),
            i_d=jnp.zeros(self.batch_size),
            i_q=jnp.zeros(self.batch_size),
            torque=jnp.zeros(self.batch_size),
            omega=jnp.zeros(self.batch_size),
        )
        return state

    def reset(self):
        return self.init_states()

    def ode_step(self, system_state, u_dq, static_params):

        omega_el = system_state.omega
        i_d = system_state.i_d
        i_q = system_state.i_q
        eps = system_state.epsilon

        args = (u_dq, static_params)

        def vector_field(t, y, args):
            i_d, i_q = y
            u_dq, params = args
            u_d = u_dq[0]
            u_q = u_dq[1]
            l_d = params.l_d
            l_q = params.l_q
            psi_p = params.psi_p
            r_s = params.r_s
            i_d_diff = (u_d + omega_el * l_q * i_q - r_s * i_d) / l_d
            i_q_diff = (u_q - omega_el * (l_d * i_d + psi_p) - r_s * i_q) / l_q
            d_y = i_d_diff, i_q_diff  # [0]
            return d_y

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.tau
        y0 = tuple([i_d, i_q])
        env_state = self._solver.init(term, t0, t1, y0, args)
        y, _, _, env_state, _ = self._solver.step(term, t0, t1, y0, args, env_state, made_jump=False)

        i_d_k1 = y[0]
        i_q_k1 = y[1]
        with jdc.copy_and_mutate(system_state, validate=True) as system_state_next:
            system_state_next.epsilon = step_eps(eps, omega_el, self.tau, 1.0)
            system_state_next.i_d = i_d_k1
            system_state_next.i_q = i_q_k1
            system_state_next.torque = jnp.array(
                [
                    currents_to_torque(
                        i_d_k1, i_q_k1, static_params.p, static_params.psi_p, static_params.l_d, static_params.l_q
                    )
                ]
            )[0]

        return system_state_next

    def simulation_step(self, system_state, action, properties):
        action_buffer = system_state.action_buffer
        eps = system_state.epsilon

        if properties.deadtime > 0:

            advanced_eps = step_eps(eps, system_state.omega, self.tau, tau_scale=properties.deadtime)

            if properties.control_set == "fcs":
                future_u_dq = switching_state_to_dq(action, properties.physical_params.u_dc, advanced_eps)
            else:
                future_u_dq = action

            updated_buffer = jnp.concatenate([action_buffer[1:, :], future_u_dq[None, :]], axis=0)
            u_dq = action_buffer[0, :]
        else:
            updated_buffer = action_buffer

            if properties.control_set == "fcs":
                u_dq = switching_state_to_dq(action, properties.physical_params.u_dc, eps)
            else:
                u_dq = action

        if properties.control_set == "ccs":
            u_dq = clip_in_abc_coordinates(
                u_dq=u_dq,
                u_dc=properties.physical_params.u_dc,
                omega_el=system_state.omega,
                eps=system_state.epsilon,
                tau=self.tau,
            )

        next_system_state = self.ode_step(system_state, u_dq, properties.physical_params)
        with jdc.copy_and_mutate(next_system_state, validate=True) as next_system_state_update:
            next_system_state_update.action_buffer = updated_buffer

        return next_system_state_update

    @property
    def action_description(self):
        return self._action_description


class PMSM(CoreEnvironment):
    def __init__(
        self,
        pmsm_physical: PMSM_Physical,
        gamma: float,
        batch_size: int = 8,
        physical_constraints: dict = None,
        action_constraints: dict = None,
        static_params: dict = None,
        solver=diffrax.Euler(),
        reward_func: Callable = None,
        tau: float = 1e-4,
    ):
        """
        Args: #TODO
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
            physical_constraints(dict): Constraints of physical states of the environment.
                theta(float): Rotation angle. Default: jnp.pi
                omega(float): Angular velocity. Default: 10
            action_constraints(dict): Constraints of actions.
                torque(float): Maximum torque that can be applied to the system as action. Default: 20
            static_params(dict): Parameters of environment which do not change during simulation.
                l(float): Length of the pendulum. Default: 1
                m(float): Mass of the pendulum tip. Default: 1
                g(float): Gravitational acceleration. Default: 9.81
            solver(diffrax.solver): Solver used to compute states for next step.
            reward_func(Callable): Reward function for training. Needs observation vector, action and action_constraints as Parameters.
                                    Default: None (default_reward_func from class)
            tau(float): Duration of one control step in seconds. Default: 1e-4.

        Note: Attributes of physical_constraints, action_constraints and static_params can also be passed as jnp.Array with the length of the batch_size to set different values per batch.
        """

        self.pmsm_physical = pmsm_physical
        self.gamma = gamma
        self.batch_size = batch_size
        assert self.batch_size == self.pmsm_physical.batch_size
        #         action_buffer: jax.Array
        # epsilon: jax.Array
        # i_d: jax.Array
        # i_q: jax.Array
        # torque: jax.Array
        # omega: jax.Array
        if not static_params:
            static_params = {
                "p_omega": 0.00005,
                "p_reference": 0.0002,
                "p_reset": 1.0,
                "i_lim_multiplier": 1.2,
                "omega_ramp_min": 20000,
                "omega_ramp_max": 25000,
            }
        static_params = self.StaticParams(**static_params, physical_properties=self.pmsm_physical.properties)

        in_axes_phys_prop = self.create_in_axes_dataclass(static_params.physical_properties.physical_params)
        values, _ = tree_flatten(in_axes_phys_prop)
        if values:
            max_torque = jax.vmap(
                calc_max_torque,
                in_axes=(
                    in_axes_phys_prop.l_d,
                    in_axes_phys_prop.l_q,
                    in_axes_phys_prop.i_n,
                    in_axes_phys_prop.psi_p,
                    in_axes_phys_prop.p,
                ),
            )(
                static_params.physical_properties.physical_params.l_d,
                static_params.physical_properties.physical_params.l_q,
                static_params.physical_properties.physical_params.i_n,
                static_params.physical_properties.physical_params.psi_p,
                static_params.physical_properties.physical_params.p,
            )
        else:
            max_torque = calc_max_torque(
                static_params.physical_properties.physical_params.l_d,
                static_params.physical_properties.physical_params.l_q,
                static_params.physical_properties.physical_params.i_n,
                static_params.physical_properties.physical_params.psi_p,
                static_params.physical_properties.physical_params.p,
            )

        if not physical_constraints:
            physical_constraints = {
                "action_buffer": 2 * static_params.physical_properties.physical_params.u_dc / 3,
                "epsilon": jnp.pi,
                "i_d": static_params.physical_properties.physical_params.i_n * static_params.i_lim_multiplier,
                "i_q": static_params.physical_properties.physical_params.i_n * static_params.i_lim_multiplier,
                "omega": static_params.physical_properties.physical_params.omega_el,
                "torque": max_torque,
            }

        if not action_constraints:
            action_constraints = {"u_dq": 2 * static_params.physical_properties.physical_params.u_dc / 3}

        physical_constraints = self.pmsm_physical.PhysicalState(**physical_constraints)
        action_constraints = self.Actions(**action_constraints)

        if static_params.physical_properties.control_state == "currents":
            self._obs_description = ["i_d", "i_q", "cos_eps", "sin_eps", "omega_el", "i_d_ref", "i_q_ref"]

        elif static_params.physical_properties.control_state == "torque":
            self._obs_description = ["i_d", "i_q", "cos_eps", "sin_eps", "omega_el", "torque_ref"]

        self.update_reference_vmap = jax.vmap(self.update_reference, in_axes=(0, 0, None))
        self.update_omegas_vmap = jax.vmap(self.update_omegas, in_axes=(0, 0, 0, 0, None))
        self.generate_observation_vmap = jax.vmap(self.generate_observation)

        env_properties = self.EnvProperties(
            physical_constraints=physical_constraints,
            action_constraints=action_constraints,
            static_params=static_params,
        )
        super().__init__(batch_size, env_properties=env_properties, tau=tau, solver=solver)

    @jdc.pytree_dataclass
    class Optional:
        """Dataclass containing additional information for simulation."""

        omega_add: jax.Array
        omega_count: jax.Array
        references: jax.Array

    @jdc.pytree_dataclass
    class StaticParams:
        """Dataclass containing the static parameters of the environment."""

        p_omega: jax.Array
        p_reference: jax.Array
        p_reset: jax.Array
        i_lim_multiplier: jax.Array
        omega_ramp_min: jax.Array
        omega_ramp_max: jax.Array
        physical_properties: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class Actions:
        """Dataclass containing the actions, that can be applied to the environment."""

        u_dq: jax.Array

    @jdc.pytree_dataclass
    class States:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_state: jdc.pytree_dataclass
        PRNGKey: jax.Array
        optional: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """Dataclass used for simulation which contains environment specific dataclasses."""

        physical_constraints: jdc.pytree_dataclass
        action_constraints: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    def reset(self, random_key):
        physical_state = self.pmsm_physical.reset()

        # As the physical system is not actually updating omegas I will pretend they are part of the environment instead
        omegas = jnp.zeros((self.batch_size, 1))
        omegas_add = jnp.zeros((self.batch_size, 1))
        omegas_count = jnp.zeros((self.batch_size, 1))

        keys = jax.random.split(random_key, self.batch_size)
        references = jnp.zeros((self.batch_size, 1))

        if self.env_properties.static_params.physical_properties.control_state == "currents":
            i_d_ref, keys = self.update_reference_vmap(references, keys, self.env_properties.static_params.p_reset)
            i_q_ref, keys = self.update_reference_vmap(references, keys, self.env_properties.static_params.p_reset)
            references = jnp.hstack((i_d_ref, i_q_ref))
        else:
            references, keys = self.update_reference_vmap(references, keys, self.env_properties.static_params.p_reset)

        omegas, omegas_add, omegas_count, keys = self.update_omegas_vmap(
            omegas, omegas_add, omegas_count, keys, self.env_properties.static_params.p_reset
        )

        with jdc.copy_and_mutate(physical_state, validate=True) as physical_state_upd:
            physical_state_upd.omega = (omegas * self.env_properties.physical_constraints.omega)[:, 0]

        # system_state = {
        #     "physical_state": physical_state,
        #     "omega_add": omegas_add,
        #     "omega_count": omegas_count,
        #     "keys": keys,
        #     "references": references,
        #     }
        opt = self.Optional(omega_add=omegas_add, omega_count=omegas_count, references=references)
        system_state = self.States(physical_state=physical_state_upd, PRNGKey=keys, optional=opt)

        observations = jax.vmap(
            self.generate_observation,
            in_axes=(0, self.in_axes_env_properties.physical_constraints),
        )(system_state, self.env_properties.physical_constraints)
        # observations = self.generate_observation_vmap(system_state, self.state_normalizer, self.action_normalizer)
        return observations, system_state

    def generate_observation(self, system_state, physical_constraints):
        eps = system_state.physical_state.epsilon
        cos_eps = jnp.cos(eps)
        sin_eps = jnp.sin(eps)
        obs = jnp.hstack(
            (
                system_state.physical_state.i_d / physical_constraints.i_d,
                system_state.physical_state.i_q / physical_constraints.i_q,
                system_state.physical_state.omega / physical_constraints.omega,
                cos_eps,
                sin_eps,
                system_state.optional.references,
                system_state.physical_state.action_buffer.reshape(-1) / physical_constraints.action_buffer,
            )
        )
        return obs

    def update_reference(self, reference, key, p):
        random_bool = jax.random.bernoulli(key, p=p)
        key, subkey = jax.random.split(key)
        new_reference = jnp.where(random_bool, jax.random.uniform(subkey, minval=-1.0, maxval=1.0), reference)
        key, subkey = jax.random.split(subkey)
        return new_reference, subkey

    def update_omegas(self, omegas, omegas_add, omegas_count, key, p):
        random_bool = jax.random.bernoulli(key, p=p)
        key, subkey = jax.random.split(key)

        # Add value to omegas
        omegas += omegas_add

        # If new target omega has been reached stop adding values in the future
        omegas_count = jnp.where(omegas_count > 0, omegas_count - 1, omegas_count)
        omegas_add = jnp.where(omegas_count == 0, 0.0, omegas_add)

        # Generate new omega targets and define the ramp
        key, subkey = jax.random.split(subkey)
        omegas_new = jnp.where(
            random_bool & (omegas_add == 0.0), jax.random.uniform(subkey, minval=-1.0, maxval=1.0), omegas
        )

        key, subkey = jax.random.split(subkey)
        omegas_count = jnp.where(
            omegas_new != omegas,
            jax.random.choice(
                subkey,
                jnp.arange(
                    self.env_properties.static_params.omega_ramp_min, self.env_properties.static_params.omega_ramp_max
                ),  # TODO change to passed parameter so vmappable
                replace=True,
                axis=0,
            ),
            omegas_count,
        )

        omegas_add += jnp.where(omegas_new != omegas, (omegas_new - omegas) / omegas_count, 0.0)

        key, subkey = jax.random.split(subkey)

        return omegas, omegas_add, omegas_count, subkey

    def step(self, system_state, actions, env_properties):
        if env_properties.static_params.physical_properties.control_set == "ccs":
            actions *= env_properties.action_constraints.u_dq
        next_physical_state = self.pmsm_physical.simulation_step(
            system_state.physical_state, actions, env_properties.static_params.physical_properties
        )
        omegas, omegas_add, omegas_count, keys = self.update_omegas(
            system_state.physical_state.omega / env_properties.physical_constraints.omega,
            system_state.optional.omega_add,
            system_state.optional.omega_count,
            system_state.PRNGKey,
            env_properties.static_params.p_reset,
        )
        # omegas = jnp.zeros((self.batch_size, 1)) + 0.2 #TODO: Remove
        if env_properties.static_params.physical_properties.control_set == "currents":
            i_d_ref, keys = self.update_reference(
                system_state.optional.reference[0], keys, env_properties.static_params.p_reference
            )
            i_q_ref, keys = self.update_reference(
                system_state.optional.reference[1], keys, env_properties.static_params.p_reference
            )
            references = jnp.hstack((i_d_ref, i_q_ref))
        else:
            references, keys = self.update_reference(
                system_state.optional.references, keys, env_properties.static_params.p_reference
            )

        # system_state = {
        #     "physical_state": physical_state,
        #     "omega_add": omegas_add,
        #     "omega_count": omegas_count,
        #     "keys": keys,
        #     "references": references,
        #     }
        with jdc.copy_and_mutate(next_physical_state, validate=True) as next_physical_state_upd:
            next_physical_state_upd.omega = omegas[0] * self.env_properties.physical_constraints.omega

        opt = self.Optional(omega_add=omegas_add, omega_count=omegas_count, references=references)
        next_system_state = self.States(physical_state=next_physical_state_upd, PRNGKey=keys, optional=opt)

        observations = self.generate_observation(next_system_state, env_properties.physical_constraints)
        rewards = self.calculate_reward(next_physical_state, system_state.optional.references, env_properties)
        dones = self.identify_system_limit_violations(next_physical_state, env_properties.physical_constraints)

        return next_system_state, observations, rewards, dones

    def identify_system_limit_violations(self, physical_state, physical_constraints):
        i_d_norm = physical_state.i_d / physical_constraints.i_d
        i_q_norm = physical_state.i_q / physical_constraints.i_q
        i_s = jnp.sqrt(i_d_norm**2 + i_q_norm**2)
        return i_s > 1

    def calculate_reward(self, physical_state, references, env_properties):
        if env_properties.static_params.physical_properties.control_state == "currents":
            reward = self.current_reward_func(
                physical_state.i_d / env_properties.physical_constraints.i_d,
                physical_state.i_q / env_properties.physical_constraints.i_q,
                references[0],
                references[1],
            )
        elif env_properties.static_params.physical_properties.control_state == "torque":
            reward = self.torque_reward_func(
                (physical_state.i_d * (env_properties.physical_constraints.i_d) ** -1),
                (physical_state.i_q * (env_properties.physical_constraints.i_q) ** -1),
                (physical_state.torque / (env_properties.physical_constraints.torque)),
                references,
                env_properties.static_params.i_lim_multiplier,
            )

        return reward

    def current_reward_func(self, i_d, i_q, i_d_ref, i_q_ref):
        mse = 0.5 * (i_d - i_d_ref) ** 2 + 0.5 * (i_q - i_q_ref) ** 2
        return -1 * (mse * (1 - self.gamma))

    def torque_reward_func(self, i_d, i_q, torque, torque_ref, i_lim_multiplier):
        i_s = jnp.sqrt(i_d**2 + i_q**2)
        i_n = 1 / i_lim_multiplier
        i_d_plus = 0.2 * i_n
        torque_tol = 0.01
        rew = jnp.zeros_like(torque_ref)
        rew = jnp.where(i_s > 1, -1 * jnp.abs(i_s), rew)
        rew = jnp.where((i_s < 1.0) & (i_s > i_n), 0.5 * (1 - (i_s - i_n) / (1 - i_n)) - 1, rew)
        rew = jnp.where((i_s < i_n) & (i_d > i_d_plus), -0.5 * ((i_d - i_d_plus) / (i_n - i_d_plus)), rew)
        rew = jnp.where(
            (i_s < i_n) & (i_d < i_d_plus) & (jnp.abs(torque - torque_ref) > torque_tol),
            0.5 * (1 - jnp.abs((torque_ref - torque) / 2)),
            rew,
        )
        rew = jnp.where(
            (i_s < i_n) & (i_d < i_d_plus) & (jnp.abs(torque - torque_ref) < torque_tol), 1 - 0.5 * i_s, rew
        )
        return rew * (1 - self.gamma)

    @property
    def action_description(self):
        return self.pmsm._action_description

    @property
    def obs_description(self):
        return self._obs_description

    @property
    def control_state(self):
        return self.pmsm.control_state
