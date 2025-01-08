from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import jax_dataclasses as jdc

import abc


class ModelWrapper:
    """
    A base class for wrapping models, providing an interface for step-wise predictions,
    gradient computations, and simulation rollouts.
    """

    model: eqx.Module

    def __init__(self, model, **kwargs):
        self.model = model

    @abc.abstractmethod
    def step(self, obs, action, tau):
        """
        Perform a one-step prediction given the current observation, action, and time step.

        Args:
            obs: The current state or observation.
            action: The action taken at the current state.
            tau: The time step for the prediction.

        Returns:
            The predicted next observation after applying the action.
        """
        return

    @abc.abstractmethod
    def gradient(self, obs, action):
        """
        Compute the gradient of the state with respect to time.

        Args:
            obs: The current state or observation x(t).
            action: The action u(t) applied at the current state.

        Returns:
            The gradient dx(t)/dt = f(x(t), u(t)).
        """
        return

    @abc.abstractmethod
    def rollout(self, init_obs, actions, tau):
        """
        Simulate a trajectory starting from an initial observation and a sequence of actions.

        Args:
            init_obs: The initial state or observation.
            actions: A sequence of actions to apply over time.
            tau: The time step for each action in the trajectory.

        Returns:
            A sequence of observations representing the simulation rollout.
        """
        return


class MLP(eqx.Module):

    layers: list[eqx.nn.Linear]
    output_activation: Callable

    def __init__(self, layer_sizes, key, output_activation: Callable = lambda x: x):

        self.layers = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey))

        self.output_activation = output_activation

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = jax.nn.leaky_relu(layer(x))
        return self.output_activation(self.layers[-1](x))


class NeuralEulerODE(eqx.Module):
    func: MLP

    def __init__(self, layer_sizes, key, **kwargs):
        super().__init__(**kwargs)
        self.func = MLP(layer_sizes=layer_sizes, key=key)

    def step(self, obs, action, tau):
        obs_act = jnp.hstack([obs, action])
        next_obs = obs + tau * self.func(obs_act)
        return next_obs

    def __call__(self, init_obs, actions, tau):

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations
