from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import jax_dataclasses as jdc
from evaluations.utils import valid_space_grid
import abc


class PredictionComparison(eqx.Module):
    grid: jax.Array
    action_dim: float
    obs_dim: float

    def __init__(self, grid, action_dim, obs_dim):
        assert grid.shape[-1] == (
            action_dim + obs_dim
        ), f"The grid dimension does not fit the action_dim and obs_dim. Grid dimension should be action_dim+obs_dim, but {grid.shape[-1]} and {action_dim}+{obs_dim} are given"
        self.grid = grid
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def __call__(self, model, model_gt):
        pred = jax.vmap(model.gradient, in_axes=(0, 0))(self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :])
        pred_gt = jax.vmap(model_gt.gradient, in_axes=(0, 0))(
            self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :]
        )

        return (pred - pred_gt), jnp.mean((pred - pred_gt) ** 2)


class ModelEvaluator:
    def __init__(self, constraint_function, gt_model, obs_dim, act_dim, validation_points_per_dim):
        self.constraint_function = constraint_function
        self.gt_model = gt_model
        self.constraint_data_space_grid = valid_space_grid(
            constraint_function, obs_dim + act_dim, validation_points_per_dim, -1, 1
        )
        # create default metrics with default params
        self.default_metrics = {
            "pred_comp": PredictionComparison(
                grid=self.constraint_data_space_grid, action_dim=act_dim, obs_dim=obs_dim
            ),
        }
