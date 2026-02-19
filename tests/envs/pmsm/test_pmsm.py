import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpy as np
import diffrax
from exciting_environments import EnvironmentRegistry
from exciting_environments.utils import MinMaxNormalization, load_sim_properties_from_json
from pathlib import Path
from motor_parameters import MotorVariant
import pickle
import os
from pytest import approx

motor_variants = list(MotorVariant)


@pytest.mark.parametrize("motor_variant", motor_variants)
def test_default_initialization(motor_variant):
    """Ensure default static parameters and normalizations are not changed by accident."""
    motor_params = motor_variant.get_params()
    physical_normalizations = motor_params.physical_normalizations.__dict__
    action_normalizations = motor_params.action_normalizations.__dict__
    params = motor_params.static_params.__dict__
    env = EnvironmentRegistry.PMSM.make(motor_variant=motor_variant)
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        assert env_value == approx(float(value)), f"Default parameter {key} is different: {env_value} != {value}"

    for key, norm in physical_normalizations.items():
        env_norm = getattr(env.env_properties.physical_normalizations, key)
        assert env_norm.min == approx(
            norm.min
        ), f"Default physical_normalization {key} is different: {env_norm.min} != {norm.min}"
        assert env_norm.max == approx(
            norm.max
        ), f"Default physical_normalization {key} is different: {env_norm.max} != {norm.max}"

    for key, norm in action_normalizations.items():
        env_norm = getattr(env.env_properties.action_normalizations, key)
        assert env_norm.min == approx(
            norm.min
        ), f"Default action_normalization {key} is different: {env_norm.min} != {norm.min}"

        assert env_norm.max == approx(
            norm.max
        ), f"Default action_normalization {key} is different: {env_norm.max} != {norm.max}"


def test_custom_initialization():
    """Ensure static parameters and normalizations are initialized correctly."""
    physical_normalizations = {
        "u_d_buffer": MinMaxNormalization(min=jnp.array(-2 * 350 / 3), max=jnp.array(2 * 26 / 3)),
        "u_q_buffer": MinMaxNormalization(min=jnp.array(-2 * 320 / 3), max=jnp.array(2 * 300 / 3)),
        "epsilon": MinMaxNormalization(min=jnp.array(-jnp.pi / 2), max=jnp.array(jnp.pi)),
        "i_d": MinMaxNormalization(min=jnp.array(-30), max=jnp.array(0)),
        "i_q": MinMaxNormalization(min=jnp.array(-20), max=jnp.array(250)),
        "omega_el": MinMaxNormalization(min=jnp.array(4), max=jnp.array(3 * 1100 * 2 * jnp.pi / 60)),
        "torque": MinMaxNormalization(min=jnp.array(-200), max=jnp.array(2030)),
    }
    action_normalizations = {
        "u_d": MinMaxNormalization(min=jnp.array(-2 * 350 / 3), max=jnp.array(2 * 26 / 3)),
        "u_q": MinMaxNormalization(min=jnp.array(-2 * 320 / 3), max=jnp.array(2 * 300 / 3)),
    }
    params = {
        "p": jnp.array(3),
        "r_s": jnp.array(15e-3),
        "l_d": jnp.array(0.37e-3),
        "l_q": jnp.array(1.2e-3),
        "psi_p": jnp.array(65.6e-3),
        "u_dc": jnp.array(400),
        "deadtime": jnp.array(1),
    }
    env = EnvironmentRegistry.PMSM.make(
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
    env = EnvironmentRegistry.PMSM.make(
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
    for i in range(1000):
        action = actions_data[i]
        obs, state = env.step(state, action)
        generated_observations.append(obs)
    generated_observations = jnp.array(generated_observations)
    assert jnp.allclose(generated_observations, stored_observations, 1e-8), "Step function generates different data"
