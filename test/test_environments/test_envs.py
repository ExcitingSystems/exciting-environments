import pytest
import exciting_environments as excenvs
import numpy as np
import jax
import jax.numpy as jnp

env_ids = ['Pendulum-v0', 'MassSpringDamper-v0', 'CartPole-v0']


@pytest.mark.parametrize('env_id', env_ids)
@pytest.mark.parametrize('tau', [1e-4, 1e-5])
def test_tau(env_id, tau):
    env = excenvs.make(env_id, tau=tau)
    assert env.tau == tau


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('env_id', env_ids)
def test_execution(env_id, no_of_steps):
    env = excenvs.make(env_id)
    print(env_id)
    terminated = True
    for i in range(no_of_steps):
        if np.any(terminated):
            observation = env.reset()
        action = env.action_space.sample(jax.random.PRNGKey(i))
        assert env.action_space.contains(action)
        observation, reward, terminated, truncated, info = env.step(action)
        assert not np.any(
            np.isnan(observation[0])), 'An invalid nan-value is in the state.'
        assert not np.any(
            np.isnan(observation[1])), 'An invalid nan-value is in the reference.'
        assert info == {}
        # assert type(reward) in [float, np.float64, np.float32], 'The Reward is not a scalar floating point value.'
        assert not np.any(np.isnan(reward)), 'Invalid nan-value as reward.'
        # Only the shape is monitored here. The states and references may lay slightly outside of the specified space.
        # This happens if limits are violated or if some states are not observed to lay within their limits.
        assert observation.shape == env.observation_space.shape, 'The shape of the state is incorrect.'
