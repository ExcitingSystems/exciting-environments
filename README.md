# exciting-environments

## Overview
The exciting-environments package is a toolbox for the simulation of physical [differential equations](https://en.wikipedia.org/wiki/Differential_equation) wrapped into [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) inspired environments using [Jax](https://github.com/google/jax). Due to the just-in-time compilation native to JAX, this type of implementation offers great advantages in terms of simulation speed.

## Getting Started

A basic routine is as simple as:
```py
import jax.numpy as jnp
import exciting_environments as excenvs

env = excenvs.make("Pendulum-v0", batch_size=5, action_constraints={"torque": 15}, tau=2e-2) 
obs, state = env.reset()

actions = jnp.linspace(start=-1, stop=1, num=1000)[None, :, None]
actions = actions.repeat(env.batch_size, axis=0)

observations = []
observations.append(obs)

for idx in range(actions.shape[1]):
    obs, reward, terminated, truncated, state = env.vmap_step(
        state, actions[:, idx, :]
    )
    observations.append(obs)
observations = jnp.stack(observations, axis=1)

print("actions shape:", actions.shape)
print("observations shape:", observations.shape)
```

which produces $5$ identical trajectories in parallel:

![](https://github.com/ExcitingSystems/exciting-environments/blob/main/fig/excenvs_pendulum_simulation_example.png?raw=true)

alternatively, simulate full trajectories:

```py
import jax.numpy as jnp
import exciting_environments as excenvs
import diffrax

env = excenvs.make(
    "Pendulum-v0", solver=diffrax.Tsit5(), batch_size=5, action_constraints={"torque": 15}, tau=2e-2
) 
obs, state = env.reset()

actions = jnp.linspace(start=-1, stop=1, num=2000)[None, :, None]
actions = actions.repeat(env.batch_size, axis=0)

observations, rewards, terminations, truncations, last_state = env.vmap_sim_ahead(
    init_state=state,
    actions=actions,
    obs_stepsize=env.tau,
    action_stepsize=env.tau
)

print("actions shape:", actions.shape)
print("observations shape:", observations.shape)
```

which produces $5$ identical trajectories in parallel as well:

![](https://github.com/ExcitingSystems/exciting-environments/blob/main/fig/excenvs_pendulum_simulation_example_advanced.png?raw=true)

Note that in this case the Tsit5 ODE solver instead of the default explicit Euler is used.
All solvers used here are from the diffrax library (https://docs.kidger.site/diffrax/).