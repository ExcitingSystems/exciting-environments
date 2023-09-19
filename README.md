# exciting-environments

## Overview
The exctiting-environments package is a toolbox for the simulation of physical [differential equations](https://en.wikipedia.org/wiki/Differential_equation) wrapped into [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments using [Jax](https://github.com/google/jax). Due to the possible just-in-time compilation of JAX, this type of implementation offers great advantages in terms of simulation speed.

## Getting Started

A basic routine is as simple as:
```py
import exciting_environments as excenvs

if __name__ == '__main__':
    env = excenvs.make("Pendulum-v0") 
    env.reset()
    for _ in range(10000):
        obs_states, rewards, terminated, truncated, _ =\ 
        	env.step(env.action_space.sample(jax.random.PRNGKey(6)))  # pick random control action
        if done:
            obs_states, _ = env.reset()
    env.close()
```
