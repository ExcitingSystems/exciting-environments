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
    params = {"k": jnp.array(100), "d": jnp.array(1), "m": jnp.array(1)}
    action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-20), max=jnp.array(20))}
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=jnp.array(-10), max=jnp.array(10)),
        "velocity": MinMaxNormalization(min=jnp.array(-10), max=jnp.array(10)),
    }
    env = excenvs.make("MassSpringDamper-v0")
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        assert env_value == value, f"Default parameter {key} is different: {env_value} != {value}"

    for key, norm in physical_normalizations.items():
        env_norm = getattr(env.env_properties.physical_normalizations, key)
        assert (
            env_norm.min == norm.min
        ), f"Default physical_normalization {key} is different: {env_norm.min} != {norm.min}"
        assert (
            env_norm.max == norm.max
        ), f"Default physical_normalization {key} is different: {env_norm.max} != {norm.max}"

    for key, norm in action_normalizations.items():
        env_norm = getattr(env.env_properties.action_normalizations, key)
        assert (
            env_norm.min == norm.min
        ), f"Default action_normalization {key} is different: {env_norm.min} != {norm.min}"

        assert (
            env_norm.max == norm.max
        ), f"Default action_normalization {key} is different: {env_norm.max} != {norm.max}"


def test_custom_initialization():
    """Ensure static parameters and normalizations are initialized correctly."""
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=jnp.array(0), max=jnp.array(130)),
        "velocity": MinMaxNormalization(min=jnp.array(-105), max=jnp.array(10)),
    }
    action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-10), max=jnp.array(20))}
    params = {"k": jnp.array(10), "m": jnp.array(5), "d": jnp.array(2)}
    env = excenvs.make(
        "MassSpringDamper-v0",
        static_params=params,
        physical_normalizations=physical_normalizations,
        action_normalizations=action_normalizations,
    )
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        assert env_value == value, f"Parameter {key} not set correctly: {env_value} != {value}"
    for key, norm in physical_normalizations.items():
        env_norm = getattr(env.env_properties.physical_normalizations, key)
        assert env_norm.min == norm.min, f"Physical_normalization {key} not set correctly: {env_norm.min} != {norm.min}"
        assert env_norm.max == norm.max, f"Physical_normalization {key} not set correctly: {env_norm.max} != {norm.max}"

    for key, norm in action_normalizations.items():
        env_norm = getattr(env.env_properties.action_normalizations, key)
        assert env_norm.min == norm.min, f"Action_normalization {key} not set correctly: {env_norm.min} != {norm.min}"

        assert env_norm.max == norm.max, f"Action_normalization {key} not set correctly: {env_norm.max} != {norm.max}"


def test_step_results():
    data_dir = os.path.join(Path(__file__).parent, "data")
    file_path = os.path.join(data_dir, "sim_properties.json")
    loaded_params, loaded_action_normalizations, loaded_physical_normalizations, loaded_tau = (
        load_sim_properties_from_json(file_path)
    )
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
    state = env.generate_state_from_observation(stored_observations[0])
    generated_observations = []
    generated_observations.append(stored_observations[0])
    for i in range(10000):
        action = actions_data[i]
        obs, state = env.step(state, action)
        generated_observations.append(obs)
    generated_observations = jnp.array(generated_observations)
    assert jnp.allclose(generated_observations, stored_observations, 1e-16), "Step function generates different data"
