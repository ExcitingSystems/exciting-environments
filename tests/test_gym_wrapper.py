import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax

from jax.tree_util import tree_flatten, tree_unflatten, tree_structure


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


env_ids = ["Pendulum-v0", "MassSpringDamper-v0", "CartPole-v0", "FluidTank-v0", "PMSM-v0"]


@pytest.mark.parametrize("env_id", env_ids)
def test_step_returns_correct_outputs(env_id):
    """Ensure step function returns outputs of expected type and shape."""
    env = excenvs.make(env_id, batch_size=4)
    gym_env = excenvs.GymWrapper(env=env)

    action = jnp.ones((env.batch_size, env.action_dim))

    _, state = env.vmap_reset()
    new_obs, state = env.vmap_step(state, action)

    _ = gym_env.reset()
    new_obs_gym, reward, terminated, truncated = gym_env.step(action)

    assert jnp.array_equal(new_obs, new_obs_gym), "gym_step generates different observation compared to standalone env"

    assert reward.shape == (4, 1), "Unexpected reward shape"
    assert terminated.shape == (4, 1), "Unexpected terminated shape"


@pytest.mark.parametrize("env_id", env_ids)
def test_gym_wrapper_ref_generation(env_id):
    env = excenvs.make(env_id, batch_size=4)
    gym_env = excenvs.GymWrapper(env=env)
    rng_env = jax.vmap(jax.random.PRNGKey)(jnp.array([0, 1, 2, 3]))
    rng_ref = jax.vmap(jax.random.PRNGKey)(jnp.array([0, 1, 2, 3]))
    obs, _ = gym_env.reset(rng_env=rng_env, rng_ref=rng_ref)

    assert gym_env.ref_gen == True
    assert gym_env.reference_hold_steps.shape == (gym_env.env.batch_size, 1)


###########################################################################################################################

###########################################################################################################################
