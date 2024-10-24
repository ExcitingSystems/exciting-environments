import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

env_ids = ["Pendulum-v0", "MassSpringDamper-v0", "CartPole-v0", "FluidTank-v0"]


@pytest.mark.parametrize("env_id", env_ids)
@pytest.mark.parametrize("tau", [1e-4, 1e-5])
def test_tau(env_id, tau):
    env = excenvs.make(env_id, tau=tau)
    assert env.tau == tau


@pytest.mark.parametrize("no_of_steps", [100])
@pytest.mark.parametrize("env_id", env_ids)
def test_execution(env_id, no_of_steps):
    env = excenvs.make(env_id)
    gym_env = excenvs.GymWrapper(env=env)
    terminated = False
    for _ in range(no_of_steps):
        if jnp.any(terminated):
            observation = gym_env.reset()
        action = jnp.ones(env.batch_size).reshape(-1, 1)
        observation, reward, terminated, truncated = gym_env.step(action)
        assert not jnp.any(jnp.isnan(observation)), "An invalid nan-value is in the state."
        # assert type(reward) in [float, np.float64, np.float32], 'The Reward is not a scalar floating point value.'
        assert not jnp.any(jnp.isnan(reward)), "Invalid nan-value as reward."
        # Only the shape is monitored here. The states and references may lay slightly outside of the specified space.
        # This happens if limits are violated or if some states are not observed to lay within their limits.
