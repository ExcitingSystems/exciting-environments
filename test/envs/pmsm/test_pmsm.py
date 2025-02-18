import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from exciting_environments.utils import MinMaxNormalization
from pathlib import Path
from motor_parameters import default_params

motor_names = ["BRUSA", "SEW", None]


@pytest.mark.parametrize("motor_name", motor_names)
def test_default_initialization(motor_name):
    """Ensure default static parameters and normalizations are not changed by accident."""
    motor_params = default_params(motor_name)
    physical_normalizations = motor_params.physical_normalizations.__dict__
    action_normalizations = motor_params.action_normalizations.__dict__
    params = motor_params.static_params.__dict__
    env = excenvs.make("PMSM-v0", LUT_motor_name=motor_name)
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
        "u_d_buffer": MinMaxNormalization(min=(-2 * 350 / 3), max=(2 * 26 / 3)),
        "u_q_buffer": MinMaxNormalization(min=(-2 * 320 / 3), max=(2 * 300 / 3)),
        "epsilon": MinMaxNormalization(min=jnp.repeat((-jnp.pi / 2), batch_size), max=(jnp.pi)),
        "i_d": MinMaxNormalization(min=(-30), max=(0)),
        "i_q": MinMaxNormalization(min=(-20), max=(250)),
        "omega_el": MinMaxNormalization(min=4, max=(3 * 1100 * 2 * jnp.pi / 60)),
        "torque": MinMaxNormalization(min=(-200), max=(2030)),
    }
    action_normalizations = {
        "u_d": MinMaxNormalization(min=(-2 * 350 / 3), max=(2 * 26 / 3)),
        "u_q": MinMaxNormalization(min=(-2 * 320 / 3), max=(2 * 300 / 3)),
    }
    params = {
        "p": jnp.repeat(3, batch_size),
        "r_s": 15e-3,
        "l_d": 0.37e-3,
        "l_q": 1.2e-3,
        "psi_p": 65.6e-3,
        "deadtime": 1,
    }
    env = excenvs.make(
        "PMSM-v0",
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
    env = excenvs.make("PMSM-v0", tau=1e-4, solver=diffrax.Euler())
    observations_data = jnp.load(str(Path(__file__).parent) + "\\data\\observations.npy")
    actions_data = jnp.load(str(Path(__file__).parent) + "\\data\\actions.npy")
    state = env.generate_state_from_observation(observations_data[0], env.env_properties)
    observations2 = []
    observations2.append(observations_data[0])
    for i in range(1000):
        action = actions_data[i]
        obs, state = env.step(state, action, env.env_properties)
        observations2.append(obs)
    observations2 = jnp.array(observations2)
    assert jnp.array_equal(observations2, observations_data), "Step function generates different data"
