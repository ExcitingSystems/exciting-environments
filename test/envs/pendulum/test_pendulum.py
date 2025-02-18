import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from exciting_environments.utils import MinMaxNormalization
from pathlib import Path


def test_default_initialization():
    """Ensure default static parameters and normalizations are not changed by accident."""
    batch_size = 4
    params = {"g": 9.81, "l": 2, "m": 1}
    action_normalizations = {"torque": MinMaxNormalization(min=-20, max=20)}
    physical_normalizations = {
        "theta": MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
        "omega": MinMaxNormalization(min=-10, max=10),
    }
    env = excenvs.make("Pendulum-v0", batch_size=batch_size)
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
        "theta": MinMaxNormalization(min=jnp.repeat(-jnp.pi / 2, batch_size), max=jnp.pi / 2),
        "omega": MinMaxNormalization(min=-5, max=3),
    }
    action_normalizations = {"torque": MinMaxNormalization(min=-10, max=10)}
    params = {"l": jnp.repeat(1, batch_size), "g": 9.81, "m": 1}
    env = excenvs.make(
        "Pendulum-v0",
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
    env = excenvs.make("Pendulum-v0", tau=1e-4, solver=diffrax.Euler())
    observations_data = jnp.load(str(Path(__file__).parent) + "\\data\\observations.npy")
    actions_data = jnp.load(str(Path(__file__).parent) + "\\data\\actions.npy")
    state = env.generate_state_from_observation(observations_data[0], env.env_properties)
    observations2 = []
    observations2.append(observations_data[0])
    for i in range(10000):
        action = actions_data[i][None]
        obs, state = env.step(state, action, env.env_properties)
        observations2.append(obs)
    observations2 = jnp.array(observations2)

    assert jnp.array_equal(observations2, observations_data), "Step function generates different data"
