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
def test_static_parameters_initialization(env_id):
    """Ensure static parameters are initialized correctly."""
    batch_size = 4
    if env_id == "Pendulum-v0":
        params = {"l": jnp.repeat(1, batch_size), "g": 9.81, "m": 1}
    elif env_id == "MassSpringDamper-v0":
        params = {"k": jnp.repeat(10, batch_size), "m": 5, "d": 2}
    elif env_id == "CartPole-v0":
        params = {
            "mu_p": 0.000002,
            "mu_c": 0.0005,
            "l": jnp.repeat(0.05, batch_size),
            "m_p": 0.1,
            "m_c": jnp.repeat(1, batch_size),
            "g": 9.81,
        }
    elif env_id == "FluidTank-v0":
        params = {"base_area": jnp.repeat(jnp.pi, batch_size), "orifice_area": jnp.pi * 0.1**2, "c_d": 0.6, "g": 9.81}
    elif env_id == "PMSM-v0":
        params = {
            "p": jnp.repeat(3, batch_size),
            "r_s": 15e-3,
            "l_d": 0.37e-3,
            "l_q": 1.2e-3,
            "psi_p": 65.6e-3,
            "deadtime": 1,
        }

    env = excenvs.make(env_id, batch_size=batch_size, static_params=params)
    for key, value in params.items():
        env_value = getattr(env.env_properties.static_params, key)
        if isinstance(value, jnp.ndarray) or isinstance(env_value, jnp.ndarray):
            assert jnp.array_equal(env_value, value), f"Parameter {key} not set correctly: {env_value} != {value}"
        else:
            assert env_value == value, f"Parameter {key} not set correctly: {env_value} != {value}"


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


# Gym Wrapper Tests
##########################


@pytest.mark.parametrize("no_of_steps", [100])
@pytest.mark.parametrize("env_id", env_ids)
def test_gym_execution(env_id, no_of_steps):
    env = excenvs.make(env_id)
    gym_env = excenvs.GymWrapper(env=env)
    terminated = False
    for _ in range(no_of_steps):
        if jnp.any(terminated):
            observation = gym_env.reset()
        action = jnp.ones((env.batch_size, env.action_dim))
        observation, reward, terminated, truncated = gym_env.step(action)
        assert not jnp.any(jnp.isnan(observation)), "An invalid nan-value is in the state."
        # assert type(reward) in [float, np.float64, np.float32], 'The Reward is not a scalar floating point value.'
        assert not jnp.any(jnp.isnan(reward)), "Invalid nan-value as reward."
        # Only the shape is monitored here. The states and references may lay slightly outside of the specified space.
        # This happens if limits are violated or if some states are not observed to lay within their limits.


# @pytest.mark.parametrize("env_class", [MassSpringDamper, Pendulum])
# def test_reset_initializes_states(env_class):
#     """Ensure reset initializes states correctly."""
#     env = env_class(batch_size=4)
#     gym_env = GymWrapper(env=env)

#     obs, state = gym_env.reset()

#     assert obs.shape == (4, env.observation_size), f"Observation shape mismatch after reset"
#     assert state.shape == (4, env.state_size), f"State shape mismatch after reset" # see if type -> env.State
#     assert jnp.all(jnp.isfinite(obs)), "Initial observations contain invalid values"
#     assert jnp.all(jnp.isfinite(state)), "Initial states contain invalid values"

# @pytest.mark.parametrize("env_class", [MassSpringDamper, Pendulum])
# def test_step_returns_correct_outputs(env_class):
#     """Ensure step function returns outputs of expected type and shape."""
#     env = env_class(batch_size=4)
#     gym_env = GymWrapper(env=env)

#     obs, state = gym_env.reset()
#     actions = jnp.zeros((4, 1))  # Zero actions

#     new_obs, reward, terminated, truncated = gym_env.step(actions)

#     assert new_obs.shape == obs.shape, "New observation shape mismatch"
#     assert reward.shape == (4,), "Reward shape mismatch"
#     assert terminated.shape == (4,), "Terminated flag shape mismatch"
#     assert truncated.shape == (4,), "Truncated flag shape mismatch"


###########################################################################################################################

###########################################################################################################################
