from abc import ABC
from abc import abstractmethod
from functools import partial
from dataclasses import fields
from typing import Callable, Any, Dict, Type
import jax.numpy as jnp
import jax_dataclasses as jdc
from exciting_environments.utils import MinMaxNormalization
import jax
import mujoco
import chex
from mujoco import mjx
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure


def dict_to_jdc_pytree(class_name: str, data: Dict[str, Any]):
    """Erstellt eine jdc.pytree_dataclass direkt aus einem Dictionary."""
    fields = {key: type(value) for key, value in data.items()}
    namespace = {"__annotations__": fields}
    DynamicClass = jdc.pytree_dataclass(type(class_name, (object,), namespace))
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


class MujucoWrapper(ABC):

    def __init__(
        self,
        mujoco_model,
        physical_normalizations=None,
        action_normalization=None,
        batch_size: int = 8,
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
            batch_size (int): Number of parallel simulations to run. Default is 8.
            tau (float): Simulation step size. If not provided, defaults to the MuJoCo
                model's `opt.timestep`. If provided, it must match `opt.timestep`.
        """
        self.mjx_model = mjx.put_model(mujoco_model)
        if not tau:
            self.tau = mujoco_model.opt.timestep
        else:
            assert tau == mujoco_model.opt.timestep
            self.tau = tau

        self.batch_size = batch_size
        self.qpos_dim = mujoco_model.nq
        self.qvel_dim = mujoco_model.nv
        self.action_dim = mujoco_model.nu
        self.sensor_dim = mujoco_model.nsensordata
        self.in_axes_env_properties = None
        self.mujoco_model = mujoco_model

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
        q_pos_jdc, _ = dict_to_jdc_pytree("qpos", q_pos)
        q_vel_jdc, _ = dict_to_jdc_pytree("qvel", q_vel)

        self.qpos_is_angle = is_angle

        return self.PhysicalNormalizations(qpos=q_pos_jdc, qvel=q_vel_jdc)

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
        action_normalization, _ = dict_to_jdc_pytree("Action", action_normalization_data)
        return action_normalization

    @jdc.pytree_dataclass
    class PhysicalNormalizations:
        qpos: jdc.pytree_dataclass
        qvel: jdc.pytree_dataclass

    @jdc.pytree_dataclass
    class EnvProperties:
        """The properties of the environment that stay constant during simulation."""

        physical_normalizations: jdc.pytree_dataclass
        action_normalizations: jdc.pytree_dataclass
        static_params: jdc.pytree_dataclass

    @partial(jax.jit, static_argnums=0)
    def init_state(self, env_properties, rng: chex.PRNGKey = None, vmap_helper=None):
        # random qpos, qvel, act, external forces ...
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

    @partial(jax.jit, static_argnums=0)
    def generate_observation(self, state, env_properties):
        # how to normalize has to be determined
        qpos = jnp.where(jnp.array(self.qpos_is_angle), self.transform_angle(state.qpos), state.qpos)
        qpos_norm = self.normalize_components(qpos, env_properties.physical_normalizations.qpos)
        qvel_norm = self.normalize_components(state.qvel, env_properties.physical_normalizations.qvel)
        obs = jnp.hstack([qpos_norm, qvel_norm])
        return obs

    def transform_angle(self, theta):
        return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @partial(jax.jit, static_argnums=0)
    def normalize_components(self, array, normalizations):
        for i, field in enumerate(fields(normalizations)):
            name = field.name
            norm_value = getattr(normalizations, name).normalize(array[i])
            array = array.at[i].set(norm_value)
        return array

    @partial(jax.jit, static_argnums=0)
    def denormalize_components(self, array, normalizations):
        for i, field in enumerate(fields(normalizations)):
            name = field.name
            denorm_values = getattr(normalizations, name).denormalize(array[i])
            array = array.at[i].set(denorm_values)
        return array

    @partial(jax.jit, static_argnums=0)
    def denormalize_action(self, action_norm, env_properties):
        """
        Denormalizes a given normalized action.

        Args:
            action_norm: The normalized action to be denormalized.
            env_properties: Environment properties containing normalization parameters.

        Returns:
            action: The denormalized action.
        """
        normalizations = env_properties.action_normalizations
        action_denorm = jnp.zeros_like(action_norm)
        for i, field in enumerate(fields(normalizations)):
            norms = getattr(normalizations, field.name)
            action_denorm = action_denorm.at[i].set(norms.denormalize(action_norm[i]))
        return action_denorm

    def reset(
        self, env_properties, rng: chex.PRNGKey = None, initial_qpos_qvel: jdc.pytree_dataclass = None, vmap_helper=None
    ):
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
            assert initial_qpos_qvel.shape[0] == self.qpos_dim + self.qvel_dim
            data = mjx.make_data(self.mjx_model)
            data = data.replace(qpos=initial_qpos_qvel[0 : self.qpos_dim])
            data = data.replace(qvel=initial_qpos_qvel[self.qpos_dim :])
        else:
            data = self.init_state(env_properties, rng)
        obs = self.generate_observation(data, env_properties)
        return obs, data

    @partial(jax.jit, static_argnums=0)
    def step(self, mjx_data, action_norm, env_properties):
        # action is not normalized jet -> need to define normalizations yourself... none given in mujoco

        assert action_norm.shape == (self.action_dim,), (
            f"The action needs to be of shape (action_dim,) which is "
            + f"{(self.action_dim,)}, but {action_norm.shape} is given"
        )

        # denormalize action
        action = self.denormalize_action(action_norm, env_properties)

        mjx_data_up = mjx_data.replace(ctrl=action)
        data = mjx.step(self.mjx_model, mjx_data_up)

        obs = self.generate_observation(data, env_properties)  # no distinction between obs and state yet

        return obs, data

    @partial(jax.jit, static_argnums=0)
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
        assert action.shape == (
            self.batch_size,
            self.action_dim,
        ), (
            "The action needs to be of shape (batch_size, action_dim) which is "
            + f"{(self.batch_size, self.action_dim)}, but {action.shape} is given"
        )
        obs, mjx_data = jax.vmap(self.step, in_axes=(0, 0, self.in_axes_env_properties))(
            mjx_data, action, self.env_properties
        )
        return obs, mjx_data

    @partial(jax.jit, static_argnums=0)
    def vmap_init_state(self, rng: chex.PRNGKey = None):
        """
        Generates an initial state for all batches, either using default values or random initialization.

        Args:
            rng (optional): Random keys for random initializations.

        Returns:
            state: The initial state for all batches.
        """
        return jax.vmap(self.init_state, in_axes=(self.in_axes_env_properties, 0, 0))(
            self.env_properties, rng, jnp.ones(self.batch_size)
        )

    @partial(jax.jit, static_argnums=0)
    def vmap_reset(self, rng: chex.PRNGKey = None, initial_qpos_qvel: jdc.pytree_dataclass = None):
        """
        Resets environment (all batches) to default, random or passed initial state.

        Args:
            rng (optional): Random keys for random initializations.
            initial_state (optional): initial_state to which the environment will be reset.

        Returns:
            obs: Observation of initial state for all batches.
            state: The initial state for all batches.
        """
        obs, state = jax.vmap(
            self.reset,
            in_axes=(self.in_axes_env_properties, 0, 0, 0),
        )(self.env_properties, rng, initial_qpos_qvel, jnp.ones(self.batch_size))

        return obs, state

    @partial(jax.jit, static_argnums=0)
    def vmap_generate_state_from_observation(self, obs, key=None):
        """
        Generates state for each batch from a given observation.

        Args:
            obs: The given observation of all batches.
            key (optional): Random keys.

        Returns:
            state: Computed state for each batch.
        """
        state = jax.vmap(self.generate_state_from_observation, in_axes=(0, self.in_axes_env_properties, 0))(
            obs, self.env_properties, key
        )
        return state
