from abc import ABC
from abc import abstractmethod
from functools import partial
from dataclasses import fields
from typing import Callable, Any, Dict, Type
import jax.numpy as jnp
import equinox as eqx

from exciting_environments.utils import MinMaxNormalization
import jax
import mujoco
import chex
from mujoco import mjx
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure


def dict_to_pytree(class_name: str, data: Dict[str, Any]):
    fields = {key: type(value) for key, value in data.items()}
    namespace = {"__annotations__": fields}
    DynamicClass = type(class_name, (eqx.Module,), namespace)
    return DynamicClass(**data), DynamicClass


qpos_names_type = {
    "0": [
        "body_position_x",
        "body_position_y",
        "body_position_z",
        "body_orientation_qw",
        "body_orientation_qx",
        "body_orientation_qy",
        "body_orientation_qz",
    ],
    "1": ["ball_orientation_qw", "ball_orientation_qx", "ball_orientation_qy", "ball_orientation_qz"],
    "2": ["position"],
    "3": ["angle"],
}
qvel_names_type = {
    "0": [
        "body_linear_velocity_x",
        "body_linear_velocity_y",
        "body_linear_velocity_z",
        "body_angular_velocity_x",
        "body_angular_velocity_y",
        "body_angular_velocity_z",
    ],
    "1": ["ball_angular_velocity_x", "ball_angular_velocity_y", "ball_angular_velocity_z"],
    "2": ["linear_velocity"],
    "3": ["angular_velocity"],
}

qpos_type_angle = {"0": [0, 0, 0, 1, 1, 1, 1], "1": [1, 1, 1, 1], "2": [0], "3": [1]}


class MujucoWrapper(eqx.Module):
    qpos_dim: int = eqx.field(static=True)
    qvel_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    sensor_dim: int = eqx.field(static=True)
    env_properties: eqx.Module
    mjx_model: eqx.Module
    tau: jax.Array
    action_description: list = eqx.field(static=True)
    obs_description: list = eqx.field(static=True)
    _batch_tracer: jax.Array

    def __init__(
        self,
        mujoco_model,
        physical_normalizations=None,
        action_normalization=None,
        tau: float = None,
    ):
        """
        A wrapper for batched simulation of MuJoCo environments with normalization support.

        Args:
            mujoco_model: A compiled MuJoCo model instance.
            physical_normalizations: A dataclass specifying min/max normalization for
                each physical state variable. If not provided, default values are generated from
                joint limits if given.
            action_normalization: A dataclass specifying min/max normalization for each
                action. If not provided, the models actuator limits are used if given.
            tau (float): Simulation step size. If not provided, defaults to the MuJoCo
                model's `opt.timestep`. If provided, it must match `opt.timestep`.
        """
        self.mjx_model = mjx.put_model(mujoco_model)
        if not tau:
            self.tau = mujoco_model.opt.timestep
        else:
            assert tau == mujoco_model.opt.timestep
            self.tau = tau

        self.qpos_dim = mujoco_model.nq
        self.qvel_dim = mujoco_model.nv
        self.action_dim = mujoco_model.nu
        self.sensor_dim = mujoco_model.nsensordata
        self._batch_tracer = jnp.array(0.0)

        action_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mujoco_model.nu)
        ]
        if not action_normalization:
            act_norm = self.generate_action_normalization_dataclasses(mujoco_model)
            if jnp.any(jnp.isnan(jnp.array(tree_flatten(act_norm)[0]))):
                raise ValueError(
                    f"The passing of action_normalization is necessary because the MuJoCo model does not provide all required normalizations for actuators. Call generate_action_normalization_dataclasses() to get current action_normalization dataclass and add missing values (jnp.nan)."
                )
        else:
            if jnp.any(jnp.isnan(jnp.array(tree_flatten(action_normalization)[0]))):
                raise ValueError(
                    f"Nan values in action_normalization. Call generate_action_normalization_dataclasses() to get current action_normalization dataclass and add missing values (jnp.nan)."
                )

        if not physical_normalizations:
            phys_norm = self.generate_physical_normalization_dataclasses(mujoco_model)
            if jnp.any(jnp.isnan(jnp.array(tree_flatten(phys_norm)[0]))):
                raise ValueError(
                    f"The passing of physical_normalizations is necessary because the MuJoCo model does not provide all required normalizations for qpos and qvel. Call generate_physical_normalization_dataclasses() to get current physical_normalization dataclass and add missing values (jnp.nan)."
                )
        else:
            if jnp.any(jnp.isnan(jnp.array(tree_flatten(physical_normalizations)[0]))):
                raise ValueError(
                    f"Nan values in physical_normalizations. Call generate_physical_normalization_dataclasses() to get current physical_normalization dataclass and add missing values (jnp.nan)."
                )
            phys_norm = physical_normalizations

        self.env_properties = self.EnvProperties(
            physical_normalizations=phys_norm, action_normalizations=action_normalization, static_params=None
        )

        self.action_description = action_names
        self.obs_description = list(self.env_properties.physical_normalizations.qpos.__dict__.keys()) + list(
            self.env_properties.physical_normalizations.qvel.__dict__.keys()
        )

    def generate_physical_normalization_dataclasses(self, model):
        q_pos = {}
        q_vel = {}
        is_angle = []
        for i in range(model.njnt):
            joint = model.joint(i)
            qpos_names = qpos_names_type[str(joint.type[0])]
            qvel_names = qvel_names_type[str(joint.type[0])]
            angle_flags = qpos_type_angle[str(joint.type[0])]
            is_angle += angle_flags
            qpos_names = [joint.name + "_" + pos_name for pos_name in qpos_names]
            qvel_names = [joint.name + "_" + vel_name for vel_name in qvel_names]
            q_pos.update(
                {
                    name: (
                        (
                            MinMaxNormalization(min=-jnp.pi, max=jnp.pi)
                            if angle_flags[i] == 1
                            else MinMaxNormalization(min=jnp.nan, max=jnp.nan)
                        )
                        if joint.limited[0] == 0
                        else MinMaxNormalization(min=joint.range[0], max=joint.range[1])
                    )
                    for i, name in enumerate(qpos_names)
                }
            )

            q_vel.update({name: MinMaxNormalization(min=jnp.nan, max=jnp.nan) for i, name in enumerate(qvel_names)})
        q_pos_pytree, _ = dict_to_pytree("qpos", q_pos)
        q_vel_pytree, _ = dict_to_pytree("qvel", q_vel)

        self.qpos_is_angle = is_angle

        return self.PhysicalNormalizations(qpos=q_pos_pytree, qvel=q_vel_pytree)

    def generate_action_normalization_dataclasses(self, model):
        action_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        action_ranges = model.actuator_ctrlrange
        action_limited = model.actuator_ctrllimited
        # action_normalization_data = {name: MinMaxNormalization(min=action_ranges[i, 0], max=action_ranges[i, 1]) for i, name in enumerate(action_names)}
        action_normalization_data = {
            name: (
                MinMaxNormalization(min=jnp.nan, max=jnp.nan)
                if action_limited[i] == 0
                else MinMaxNormalization(min=action_ranges[i, 0], max=action_ranges[i, 1])
            )
            for i, name in enumerate(action_names)
        }
        action_normalization, _ = dict_to_pytree("Action", action_normalization_data)
        return action_normalization

    class PhysicalNormalizations(eqx.Module):
        qpos: eqx.Module
        qvel: eqx.Module

    class EnvProperties(eqx.Module):
        """The properties of the environment that stay constant during simulation."""

        physical_normalizations: eqx.Module
        action_normalizations: eqx.Module
        static_params: eqx.Module

    @eqx.filter_jit
    def init_state(self, rng: chex.PRNGKey = None):
        # random qpos, qvel, act, external forces ...
        env_properties = self.env_properties
        mjx_data = mjx.make_data(self.mjx_model)
        if rng is not None:
            key, subkey = jax.random.split(rng)
            qpos_norm = jax.random.uniform(subkey, (self.qpos_dim,), minval=-1, maxval=1)
            qvel_norm = jax.random.uniform(subkey, (self.qvel_dim,), minval=-1, maxval=1)
            qpos = self.denormalize_components(qpos_norm, env_properties.physical_normalizations.qpos)
            qvel = self.denormalize_components(qvel_norm, env_properties.physical_normalizations.qvel)
            mjx_data = mjx_data.replace(qpos=qpos)
            mjx_data = mjx_data.replace(qvel=qvel)
        return mjx_data

    @eqx.filter_jit
    def generate_observation(self, state):
        # how to normalize has to be determined
        env_properties = self.env_properties
        qpos = jnp.where(jnp.array(self.qpos_is_angle), self.transform_angle(state.qpos), state.qpos)
        qpos_norm = self.normalize_components(qpos, env_properties.physical_normalizations.qpos)
        qvel_norm = self.normalize_components(state.qvel, env_properties.physical_normalizations.qvel)
        obs = jnp.hstack([qpos_norm, qvel_norm])
        return obs

    def transform_angle(self, theta):
        return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @eqx.filter_jit
    def normalize_components(self, array, normalizations):
        for i, field in enumerate(fields(normalizations)):
            name = field.name
            norm_value = getattr(normalizations, name).normalize(array[i])
            array = array.at[i].set(norm_value)
        return array

    @eqx.filter_jit
    def denormalize_components(self, array, normalizations):
        for i, field in enumerate(fields(normalizations)):
            name = field.name
            denorm_values = getattr(normalizations, name).denormalize(array[i])
            array = array.at[i].set(denorm_values)
        return array

    @eqx.filter_jit
    def denormalize_action(self, action_norm):
        """
        Denormalizes a given normalized action.

        Args:
            action_norm: The normalized action to be denormalized.
            env_properties: Environment properties containing normalization parameters.

        Returns:
            action: The denormalized action.
        """
        env_properties = self.env_properties
        normalizations = env_properties.action_normalizations
        action_denorm = jnp.zeros_like(action_norm)
        for i, field in enumerate(fields(normalizations)):
            norms = getattr(normalizations, field.name)
            action_denorm = action_denorm.at[i].set(norms.denormalize(action_norm[i]))
        return action_denorm

    def reset(self, rng: chex.PRNGKey = None, initial_qpos_qvel: eqx.Module = None, vmap_helper=None):
        """
        Resets environment to default, random or passed initial state.

        Args:
            env_properties: Environment properties.
            rng (optional): Random key for random initialization.
            initial_state (optional): The initial_state to which the environment will be reset.
            vmap_helper (optional): Helper variable for vectorized computation.

        Returns:
            obs: Observation of initial state.
            state: The initial state.
        """
        if initial_qpos_qvel is not None:
            data = mjx.make_data(self.mjx_model)
            data = data.replace(qpos=initial_qpos_qvel[0 : self.qpos_dim])
            data = data.replace(qvel=initial_qpos_qvel[self.qpos_dim :])
        else:
            data = self.init_state(rng)
        obs = self.generate_observation(data)
        return obs, data

    @eqx.filter_jit
    def step(self, mjx_data, action_norm):
        action = self.denormalize_action(action_norm)

        mjx_data_up = mjx_data.replace(ctrl=action)
        data = mjx.step(self.mjx_model, mjx_data_up)

        obs = self.generate_observation(data)

        return obs, data

    @eqx.filter_jit
    def vmap_step(self, mjx_data, action):
        """Computes one JAX-JIT compiled simulation step for multiple (batch_size) batches.

        Args:
            state: The current state of the simulation from which to calculate the next
                state.
            action: The action to apply to the environment (shape=(batch_size, action_dim)).

        Returns:
            observation: The gathered observations.
            state: New state for the next step.
        """
        self._assert_batched()
        obs, mjx_data = jax.vmap(lambda e, s, a: e.step(s, a))(self, mjx_data, action)
        return obs, mjx_data

    @eqx.filter_jit
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        """
        Generates an initial state for all batches, either using default values or random initialization.

        Args:
            rng (optional): Random keys for random initializations.

        Returns:
            state: The initial state for all batches.
        """
        self._assert_batched()
        return jax.vmap(lambda e, k: e.init_state(k))(self, rng)

    @eqx.filter_jit
    def vmap_reset(self, rng: chex.PRNGKey = None, initial_qpos_qvel: jax.Array = None):
        """
        Resets environment (all batches) to default, random or passed initial state.

        Args:
            rng (optional): Random keys for random initializations.
            initial_state (optional): initial_state to which the environment will be reset.

        Returns:
            obs: Observation of initial state for all batches.
            state: The initial state for all batches.
        """
        self._assert_batched()
        obs, state = jax.vmap(lambda e, k, s: e.reset(k, s))(self, rng, initial_qpos_qvel)

        return obs, state

    @eqx.filter_jit
    def vmap_generate_state_from_observation(self, obs, key=None):
        """
        Generates state for each batch from a given observation.

        Args:
            obs: The given observation of all batches.
            key (optional): Random keys.

        Returns:
            state: Computed state for each batch.
        """
        self._assert_batched()
        state = jax.vmap(lambda e, o, k: e.generate_state_from_observation(o, k))(self, obs, key)
        return state

    def _assert_batched(self):
        """Checks if the environment is batched by looking at the dummy leaf."""
        if jnp.ndim(self._batch_tracer) == 0:
            raise RuntimeError(
                "Calling a vmap method on a single-environment instance. Please use the 'make' function with a batch_size to create a batched environment or create it manually."
            )

    @property
    def batch_size(self):
        """Returns batch_size if environment is batched, else None."""
        if jnp.ndim(self._batch_tracer) > 0:
            return self._batch_tracer.shape[0]
        return None
