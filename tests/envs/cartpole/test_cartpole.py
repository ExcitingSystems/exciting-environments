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
    params = {
        "mu_p": jnp.array(0.000002),
        "mu_c": jnp.array(0.0005),
        "l": jnp.array(0.5),
        "m_p": jnp.array(0.1),
        "m_c": jnp.array(1),
        "g": jnp.array(9.81),
    }
    action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-20), max=jnp.array(20))}
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=jnp.array(-2.4), max=jnp.array(2.4)),
        "velocity": MinMaxNormalization(min=jnp.array(-8), max=jnp.array(8)),
        "theta": MinMaxNormalization(min=jnp.array(-jnp.pi), max=jnp.array(jnp.pi)),
        "omega": MinMaxNormalization(min=jnp.array(-8), max=jnp.array(8)),
    }
    env = excenvs.make("CartPole-v0")
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


def test_static_parameters_initialization():
    """Ensure static parameters and normalizations are initialized correctly."""
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=jnp.array(-2.5), max=jnp.array(2.9)),
        "velocity": MinMaxNormalization(min=jnp.array(-1), max=jnp.array(2)),
        "theta": MinMaxNormalization(min=jnp.array(-jnp.pi / 2), max=jnp.array(jnp.pi / 3)),
        "omega": MinMaxNormalization(min=jnp.array(-3), max=jnp.array(76)),
    }
    action_normalizations = {"force": MinMaxNormalization(min=jnp.array(-21), max=jnp.array(10))}
    params = {
        "mu_p": jnp.array(0.000002),
        "mu_c": jnp.array(0.0005),
        "l": jnp.array(0.05),
        "m_p": jnp.array(0.1),
        "m_c": jnp.array(1),
        "g": jnp.array(35.81),
    }
    env = excenvs.make(
        "CartPole-v0",
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
        "CartPole-v0",
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
