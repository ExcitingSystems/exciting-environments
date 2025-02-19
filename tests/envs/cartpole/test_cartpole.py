import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from exciting_environments.utils import MinMaxNormalization
from pathlib import Path
import pickle
import os


def test_default_initialization():
    """Ensure default static parameters and normalizations are not changed by accident."""
    batch_size = 4
    params = {
        "mu_p": 0.000002,
        "mu_c": 0.0005,
        "l": 0.5,
        "m_p": 0.1,
        "m_c": 1,
        "g": 9.81,
    }
    action_normalizations = {"force": MinMaxNormalization(min=-20, max=20)}
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=-2.4, max=2.4),
        "velocity": MinMaxNormalization(min=-8, max=8),
        "theta": MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
        "omega": MinMaxNormalization(min=-8, max=8),
    }
    env = excenvs.make("CartPole-v0", batch_size=batch_size)
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


def test_static_parameters_initialization():
    """Ensure static parameters and normalizations are initialized correctly."""
    batch_size = 4
    physical_normalizations = {
        "deflection": MinMaxNormalization(min=-2.5, max=2.9),
        "velocity": MinMaxNormalization(min=-1, max=2),
        "theta": MinMaxNormalization(min=-jnp.pi / 2, max=jnp.pi / 3),
        "omega": MinMaxNormalization(min=-3, max=76),
    }
    action_normalizations = {"force": MinMaxNormalization(min=-21, max=10)}
    params = {
        "mu_p": 0.000002,
        "mu_c": 0.0005,
        "l": jnp.repeat(0.05, batch_size),
        "m_p": 0.1,
        "m_c": jnp.repeat(1, batch_size),
        "g": 35.81,
    }
    env = excenvs.make(
        "CartPole-v0",
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
    data_dir = os.path.join(Path(__file__).parent, "data")  # Use os.path.join
    file_path = os.path.join(data_dir, "sim_properties.pkl")
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    loaded_params = loaded_data["params"]
    loaded_action_normalizations = loaded_data["action_normalizations"]
    loaded_physical_normalizations = loaded_data["physical_normalizations"]
    loaded_tau = loaded_data["tau"]
    env = excenvs.make(
        "CartPole-v0",
        tau=loaded_tau,
        solver=diffrax.Euler(),
        static_params=loaded_params,
        physical_normalizations=loaded_physical_normalizations,
        action_normalizations=loaded_action_normalizations,
    )

    observations_data = jnp.load(str(Path(__file__).parent) + "/data/observations.npy")
    actions_data = jnp.load(str(Path(__file__).parent) + "/data/actions.npy")
    state = env.generate_state_from_observation(observations_data[0], env.env_properties)
    observations2 = []
    observations2.append(observations_data[0])
    for i in range(10000):
        action = actions_data[i][None]
        obs, state = env.step(state, action, env.env_properties)
        observations2.append(obs)
    observations2 = jnp.array(observations2)
    assert jnp.array_equal(observations2, observations_data), "Step function generates different data"
