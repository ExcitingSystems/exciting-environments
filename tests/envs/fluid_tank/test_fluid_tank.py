import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from exciting_environments import EnvironmentRegistry
from exciting_environments.utils import MinMaxNormalization, load_sim_properties_from_json
from pathlib import Path
import pickle
import os

jax.config.update("jax_enable_x64", True)


def test_default_initialization():
    """Ensure default static parameters and normalizations are not changed by accident."""
    params = {
        "base_area": jnp.array(jnp.pi),
        "orifice_area": jnp.array(jnp.pi * 0.1**2),
        "c_d": jnp.array(0.6),
        "g": jnp.array(9.81),
    }
    action_normalizations = {"inflow": MinMaxNormalization(min=jnp.array(0), max=jnp.array(0.2))}
    physical_normalizations = {"height": MinMaxNormalization(min=jnp.array(0), max=jnp.array(3))}
    env = EnvironmentRegistry.FLUID_TANK.make()
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
    params = {
        "base_area": jnp.array(jnp.pi),
        "orifice_area": jnp.array(jnp.pi * 0.2**2),
        "c_d": jnp.array(0.8),
        "g": jnp.array(9.81),
    }
    action_normalizations = {"inflow": MinMaxNormalization(min=jnp.array(0.02), max=jnp.array(0.3))}
    physical_normalizations = {"height": MinMaxNormalization(min=jnp.array(1), max=jnp.array(5))}
    env = EnvironmentRegistry.FLUID_TANK.make(
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
    env = EnvironmentRegistry.FLUID_TANK.make(
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
