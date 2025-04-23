import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from exciting_environments.utils import MinMaxNormalization, load_sim_properties_from_json
from pathlib import Path
import pickle
import os

jax.config.update("jax_enable_x64", True)

def test_default_initialization():
    """Ensure default static parameters and normalizations are not changed by accident."""
    batch_size = 4
    params = {"k": 100, "d": 1, "m": 1}
    action_normalizations = {"force": MinMaxNormalization(min=-20, max=20)}
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=-10, max=10),
        "velocity": MinMaxNormalization(min=-10, max=10),
    }
    env = excenvs.make("MassSpringDamper-v0", batch_size=batch_size)
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        if isinstance(value, jnp.ndarray) or isinstance(env_value, jnp.ndarray):
            assert jnp.array_equal(env_value, value), f"Default parameter {key} is different: {env_value} != {value}"
        else:
            assert env_value == value, f"Default parameter {key} is different: {env_value} != {value}"

    for key, norm in physical_normalizations.items():
        env_norm = getattr(env.env_properties.physical_normalizations, key)
        if isinstance(norm.min, jnp.ndarray) or isinstance(env_norm.min, jnp.ndarray):
            assert jnp.array_equal(
                norm.min, env_norm.min
            ), f"Default physical_normalization {key} is different: {env_norm.min} != {norm.min}"
        else:
            assert (
                env_norm.min == norm.min
            ), f"Default physical_normalization {key} is different: {env_norm.min} != {norm.min}"
        if isinstance(norm.max, jnp.ndarray) or isinstance(env_norm.max, jnp.ndarray):
            assert jnp.array_equal(
                norm.max, env_norm.max
            ), f"Default physical_normalization {key} is different: {env_norm.max} != {norm.max}"
        else:
            assert (
                env_norm.max == norm.max
            ), f"Default physical_normalization {key} is different: {env_norm.max} != {norm.max}"

    for key, norm in action_normalizations.items():
        env_norm = getattr(env.env_properties.action_normalizations, key)
        if isinstance(norm.min, jnp.ndarray) or isinstance(env_norm.min, jnp.ndarray):
            assert jnp.array_equal(
                norm.min, env_norm.min
            ), f"Default action_normalization {key} is different: {env_norm.min} != {norm.min}"
        else:
            assert (
                env_norm.min == norm.min
            ), f"Default action_normalization {key} is different: {env_norm.min} != {norm.min}"
        if isinstance(norm.max, jnp.ndarray) or isinstance(env_norm.max, jnp.ndarray):
            assert jnp.array_equal(
                norm.max, env_norm.max
            ), f"Default action_normalization {key} is different: {env_norm.max} != {norm.max}"
        else:
            assert (
                env_norm.max == norm.max
            ), f"Default action_normalization {key} is different: {env_norm.max} != {norm.max}"


def test_custom_initialization():
    """Ensure static parameters and normalizations are initialized correctly."""
    batch_size = 4
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=jnp.repeat(0, batch_size), max=130),
        "velocity": MinMaxNormalization(min=-105, max=10),
    }
    action_normalizations = {"force": MinMaxNormalization(min=-10, max=20)}
    params = {"k": jnp.repeat(10, batch_size), "m": 5, "d": 2}
    env = excenvs.make(
        "MassSpringDamper-v0",
        batch_size=batch_size,
        static_params=params,
        physical_normalizations=physical_normalizations,
        action_normalizations=action_normalizations,
    )
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        if isinstance(value, jnp.ndarray) or isinstance(env_value, jnp.ndarray):
            assert jnp.array_equal(env_value, value), f"Parameter {key} not set correctly: {env_value} != {value}"
        else:
            assert env_value == value, f"Parameter {key} not set correctly: {env_value} != {value}"

    for key, norm in physical_normalizations.items():
        env_norm = getattr(env.env_properties.physical_normalizations, key)
        if isinstance(norm.min, jnp.ndarray) or isinstance(env_norm.min, jnp.ndarray):
            assert jnp.array_equal(
                norm.min, env_norm.min
            ), f"Physical_normalization {key} not set correctly: {env_norm.min} != {norm.min}"
        else:
            assert (
                env_norm.min == norm.min
            ), f"Physical_normalization {key} not set correctly: {env_norm.min} != {norm.min}"
        if isinstance(norm.max, jnp.ndarray) or isinstance(env_norm.max, jnp.ndarray):
            assert jnp.array_equal(
                norm.max, env_norm.max
            ), f"Physical_normalization {key} not set correctly: {env_norm.max} != {norm.max}"
        else:
            assert (
                env_norm.max == norm.max
            ), f"Physical_normalization {key} not set correctly: {env_norm.max} != {norm.max}"

    for key, norm in action_normalizations.items():
        env_norm = getattr(env.env_properties.action_normalizations, key)
        if isinstance(norm.min, jnp.ndarray) or isinstance(env_norm.min, jnp.ndarray):
            assert jnp.array_equal(
                norm.min, env_norm.min
            ), f"Action_normalization {key} not set correctly: {env_norm.min} != {norm.min}"
        else:
            assert (
                env_norm.min == norm.min
            ), f"Action_normalization {key} not set correctly: {env_norm.min} != {norm.min}"
        if isinstance(norm.max, jnp.ndarray) or isinstance(env_norm.max, jnp.ndarray):
            assert jnp.array_equal(
                norm.max, env_norm.max
            ), f"Action_normalization {key} not set correctly: {env_norm.max} != {norm.max}"
        else:
            assert (
                env_norm.max == norm.max
            ), f"Action_normalization {key} not set correctly: {env_norm.max} != {norm.max}"


def test_step_results():
    data_dir = os.path.join(Path(__file__).parent, "data")
    file_path = os.path.join(data_dir, "sim_properties.json")
    loaded_params,loaded_action_normalizations,loaded_physical_normalizations,loaded_tau=load_sim_properties_from_json(file_path)
    env = excenvs.make(
        "MassSpringDamper-v0",
        tau=loaded_tau,
        solver=diffrax.Euler(),
        static_params=loaded_params,
        physical_normalizations=loaded_physical_normalizations,
        action_normalizations=loaded_action_normalizations,
    )

    stored_observations = jnp.load(str(Path(__file__).parent) + "/data/observations.npy")
    actions_data = jnp.load(str(Path(__file__).parent) + "/data/actions.npy")
    state = env.generate_state_from_observation(stored_observations[0], env.env_properties)
    generated_observations = []
    generated_observations.append(stored_observations[0])
    for i in range(10000):
        action = actions_data[i]
        obs, state = env.step(state, action, env.env_properties)
        generated_observations.append(obs)
    generated_observations = jnp.array(generated_observations)
    assert jnp.allclose(generated_observations, stored_observations, 1e-16), "Step function generates different data"
