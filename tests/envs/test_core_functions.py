import pytest
import exciting_environments as excenvs
import jax
import jax.numpy as jnp
import numpy as np
import diffrax

from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
from exciting_environments import EnvironmentRegistry

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

envs_to_test = list(EnvironmentRegistry)
fully_observable_envs = envs_to_test


@pytest.mark.parametrize("env_type", envs_to_test)
@pytest.mark.parametrize("tau", [1e-4, 1e-5])
def test_tau(env_type, tau):
    env = env_type.make(tau=tau)
    assert env.tau == tau


@pytest.mark.parametrize("env_type", envs_to_test)
def test_reset(env_type):
    batch_size = 4
    single_env = env_type.make()
    batched_envs = env_type.make(batch_size=batch_size)
    key = jax.random.PRNGKey(seed=1234)
    keys = jax.random.split(key, num=batch_size)

    # single
    obs, state = single_env.reset(keys[0])
    assert obs.shape == single_env.obs_description.shape, f"Random reset returns different observation shape."
    assert type(state) == single_env.State, f"Random reset returns different state type."
    obs, state = single_env.reset()
    assert obs.shape == single_env.obs_description.shape, f"Default reset returns different observation shape."
    assert type(state) == single_env.State, f"Default reset returns different state type."

    # vmap
    obs, state = batched_envs.vmap_reset(keys)
    assert obs.shape == (
        batched_envs.batch_size,
        len(batched_envs.obs_description),
    ), f"Random vmap_reset returns different observation shape."
    assert type(state) == batched_envs.State, f"Random vmap_reset returns different state type."
    obs, state = batched_envs.vmap_reset()
    assert obs.shape == (
        batched_envs.batch_size,
        len(batched_envs.obs_description),
    ), f"Default vmap_reset returns different observation shape."
    assert type(state) == batched_envs.State, f"Default vmap_reset returns different state type."


@pytest.mark.parametrize("env_type", fully_observable_envs)
def test_gen_observation_gen_state(env_type):
    batch_size = 4
    single_env = env_type.make()
    batched_envs = env_type.make(batch_size=batch_size)

    # single
    obs, state = single_env.reset()
    assert jnp.array_equal(obs, single_env.generate_observation(state))
    assert jnp.array_equal(obs, single_env.generate_observation(single_env.generate_state_from_observation(obs)))

    # vmap
    obs, state = batched_envs.vmap_reset()
    # assert jnp.array_equal(obs, jax.vmap(batched_envs.generate_observation)(state))
    assert jnp.array_equal(obs, jax.vmap(lambda e, s: e.generate_observation(s))(batched_envs, state))

    assert jnp.array_equal(
        obs,
        jax.vmap(lambda e, s: e.generate_observation(s))(
            batched_envs, batched_envs.vmap_generate_state_from_observation(obs)
        ),
    )


@pytest.mark.parametrize("env_type", envs_to_test)
def test_step(env_type):
    batch_size = 4
    single_env = env_type.make()
    batched_envs = env_type.make(batch_size=batch_size)
    # single
    init_obs, state = single_env.reset()
    init_state_struct = tree_structure(state)
    for _ in range(100):
        action = jnp.ones(single_env.action_dim)
        obs, state = single_env.step(state, action)
    assert init_obs.shape == obs.shape, "Observation shape changes during simulation steps."
    assert init_state_struct == tree_structure(state), "State changes structure during simulation steps."

    # vmap
    init_obs, state = batched_envs.vmap_reset()
    init_state_struct = tree_structure(state)
    for _ in range(100):
        action = jnp.ones((batched_envs.batch_size, batched_envs.action_dim))
        obs, state = batched_envs.vmap_step(state, action)
    assert init_obs.shape == obs.shape, "Observation shape changes during vmapped simulation steps."
    assert init_state_struct == tree_structure(state), "State changes structure during vmapped simulation steps."


@pytest.mark.parametrize("env_type", envs_to_test)
def test_simulate_ahead(env_type):
    sim_steps = 10
    batch_size = 4
    single_env = env_type.make()
    batched_envs = env_type.make(batch_size=batch_size)
    # single
    obs, init_state = single_env.reset()
    acts = jnp.ones((sim_steps, single_env.action_dim))
    obs, states, last_state = single_env.sim_ahead(init_state, acts)
    assert obs.shape == (
        (sim_steps + 1),
        len(single_env.obs_description),
    ), "Observation changes shape during simulation ahead."
    assert tree_structure(init_state) == tree_structure(
        last_state
    ), "State changes structure during vmapped simulate ahead."

    # vmapped
    obs, init_state = batched_envs.vmap_reset()
    acts = jnp.ones((batch_size, sim_steps, batched_envs.action_dim))
    obs, states, last_state = batched_envs.vmap_sim_ahead(init_state, acts)
    assert obs.shape == (
        batch_size,
        (sim_steps + 1),
        len(batched_envs.obs_description),
    ), "Observation changes shape during vmapped simulation ahead."
    assert tree_structure(init_state) == tree_structure(
        last_state
    ), "State changes structure during vmapped simulate ahead."


@pytest.mark.parametrize("env_type", envs_to_test)
def test_similarity_step_sim_ahead_results(env_type):
    sim_steps = 10
    batch_size = 4
    single_env = env_type.make(solver=diffrax.Euler())

    # single
    obs, state = single_env.reset()
    acts = jnp.ones((sim_steps, single_env.action_dim))

    # sim ahead
    obs_ahead, states_ahead, last_state_ahead = single_env.sim_ahead(state, acts)
    last_obs_ahead = single_env.generate_observation(last_state_ahead)
    # steps
    for _ in range(sim_steps):
        action = jnp.ones(single_env.action_dim)
        obs_step, state = single_env.step(state, action)

    # compare final observations
    assert jnp.allclose(
        last_obs_ahead, obs_step, 1e-16
    ), "Simulate ahead and stepwise simulation return significantly deviating results for diffrax.Euler solver."
