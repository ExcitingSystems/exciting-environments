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
@pytest.mark.parametrize("tau", [1e-4, 1e-5])
def test_tau(env_id, tau):
    env = excenvs.make(env_id, tau=tau)
    assert env.tau == tau


@pytest.mark.parametrize("env_id", env_ids)
def test_reset(env_id):
    batch_size = 4
    env = excenvs.make(env_id, batch_size=batch_size)
    keys = jax.vmap(jax.random.PRNGKey)(np.random.randint(0, 2**31, size=(batch_size,)))

    # single
    obs, state = env.reset(env.env_properties, keys[0])
    assert obs.shape == env.obs_description.shape, f"Random reset returns different observation shape."
    assert type(state) == env.State, f"Random reset returns different state type."
    obs, state = env.reset(env.env_properties)
    assert obs.shape == env.obs_description.shape, f"Default reset returns different observation shape."
    assert type(state) == env.State, f"Default reset returns different state type."

    # vmap
    obs, state = env.vmap_reset(keys)
    assert obs.shape == (
        env.batch_size,
        len(env.obs_description),
    ), f"Random vmap_reset returns different observation shape."
    assert type(state) == env.State, f"Random vmap_reset returns different state type."
    obs, state = env.vmap_reset()
    assert obs.shape == (
        env.batch_size,
        len(env.obs_description),
    ), f"Default vmap_reset returns different observation shape."
    assert type(state) == env.State, f"Default vmap_reset returns different state type."


@pytest.mark.parametrize("env_id", env_ids)
def test_gen_observation_gen_state(env_id):
    batch_size = 4
    env = excenvs.make(env_id, batch_size=batch_size)

    # single
    obs, state = env.reset(env.env_properties)
    assert jnp.array_equal(obs, env.generate_observation(state, env.env_properties))
    assert jnp.array_equal(
        obs, env.generate_observation(env.generate_state_from_observation(obs, env.env_properties), env.env_properties)
    )

    # vmap
    obs, state = env.vmap_reset()
    assert jnp.array_equal(
        obs, jax.vmap(env.generate_observation, in_axes=(0, env.in_axes_env_properties))(state, env.env_properties)
    )
    assert jnp.array_equal(
        obs,
        jax.vmap(env.generate_observation, in_axes=(0, env.in_axes_env_properties))(
            env.vmap_generate_state_from_observation(obs), env.env_properties
        ),
    )


@pytest.mark.parametrize("env_id", env_ids)
def test_step(env_id):
    batch_size = 4
    env = excenvs.make(env_id, batch_size=batch_size)
    # single
    init_obs, state = env.reset(env.env_properties)
    init_state_struct = tree_structure(state)
    for _ in range(100):
        action = jnp.ones(env.action_dim)
        obs, state = env.step(state, action, env.env_properties)
    assert init_obs.shape == obs.shape, "Observation shape changes during simulation steps."
    assert init_state_struct == tree_structure(state), "State changes structure during simulation steps."

    # vmap
    init_obs, state = env.vmap_reset()
    init_state_struct = tree_structure(state)
    for _ in range(100):
        action = jnp.ones((env.batch_size, env.action_dim))
        obs, state = env.vmap_step(state, action)
    assert init_obs.shape == obs.shape, "Observation shape changes during vmapped simulation steps."
    assert init_state_struct == tree_structure(state), "State changes structure during vmapped simulation steps."


@pytest.mark.parametrize("env_id", env_ids)
def test_simulate_ahead(env_id):
    if env_id != "FluidTank-v0":
        sim_steps = 10
        batch_size = 4
        env = excenvs.make(env_id, batch_size=batch_size)
        # single
        obs, init_state = env.reset(env.env_properties)
        acts = jnp.ones((sim_steps, env.action_dim))
        obs, states, last_state = env.sim_ahead(init_state, acts, env.env_properties, env.tau, env.tau)
        assert obs.shape == (
            (sim_steps + 1),
            len(env.obs_description),
        ), "Observation changes shape during simulation ahead."
        assert tree_structure(init_state) == tree_structure(
            last_state
        ), "State changes structure during vmapped simulate ahead."

        # vmapped
        obs, init_state = env.vmap_reset()
        acts = jnp.ones((batch_size, sim_steps, env.action_dim))
        obs, states, last_state = env.vmap_sim_ahead(init_state, acts, env.tau, env.tau)
        assert obs.shape == (
            batch_size,
            (sim_steps + 1),
            len(env.obs_description),
        ), "Observation changes shape during vmapped simulation ahead."
        assert tree_structure(init_state) == tree_structure(
            last_state
        ), "State changes structure during vmapped simulate ahead."


@pytest.mark.parametrize("env_id", env_ids)
def test_similarity_step_sim_ahead_results(env_id):
    if env_id != "FluidTank-v0":
        sim_steps = 10
        batch_size = 4
        env = excenvs.make(env_id, batch_size=batch_size, solver=diffrax.Euler())

        # single
        obs, state = env.reset(env.env_properties)
        acts = jnp.ones((sim_steps, env.action_dim))

        # sim ahead
        obs_ahead, states_ahead, last_state_ahead = env.sim_ahead(state, acts, env.env_properties, env.tau, env.tau)
        last_obs_ahead = env.generate_observation(last_state_ahead, env.env_properties)
        # steps
        for _ in range(sim_steps):
            action = jnp.ones(env.action_dim)
            obs_step, state = env.step(state, action, env.env_properties)

        # compare final observations
        assert jnp.allclose(
            last_obs_ahead, obs_step, 1e-16
        ), "Simulate ahead and stepwise simulation return significantly deviating results for diffrax.Euler solver."
