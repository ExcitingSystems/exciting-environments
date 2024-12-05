from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import jax_dataclasses as jdc
from evaluations.density_estimation import (
    DensityEstimate,
    build_grid,
    update_density_estimate_multiple_observations,
)
from evaluations.metrics import (
    JSDLoss,
    audze_eglais,
    MC_uniform_sampling_distribution_approximation,
    kiss_space_filling_cost,
    blockwise_ksfc,
    blockwise_mcudsa,
)


class Evaluator(eqx.Module):

    constraint_function: Callable
    points_per_dim: int
    data_dim: int
    hypercube_grid: jax.Array
    constraint_data_space_grid: jax.Array
    metrics: jdc.pytree_dataclass

    def __init__(self, constraint_function, data_dim, points_per_dim):
        self.constraint_function = constraint_function
        self.points_per_dim = points_per_dim
        self.data_dim = data_dim
        hypercube_grid = build_grid(data_dim, -1, 1, points_per_dim)
        self.constraint_data_space_grid = self.valid_space_grid(hypercube_grid, constraint_function)
        self.hypercube_grid = hypercube_grid
        self.metrics = self.Metrics(
            jsd=self.default_jsd,
            ae=self.default_ae,
            mcudsa=self.default_mcudsa,
            ksfc=self.default_ksfc,
            constraints=self.constraint_compliances,
        )

    @jdc.pytree_dataclass
    class Metrics:
        """Dataclass containing the provided metric functions for data evaluation."""

        jsd: Callable
        ae: Callable
        mcudsa: Callable
        ksfc: Callable
        constraints: Callable

    def valid_space_grid(self, data_grid, constr_func):
        valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0
        constraint_data_grid = data_grid[jnp.where(valid_grid_point == True)]
        return constraint_data_grid

    def get_default_metrics(self, data_points, metrics=["jsd", "ae", "mcudsa", "ksfc", "constraints"]):
        metrics_results_nan = self.Metrics(jds=jnp.nan, ae=jnp.nan, mcudsa=jnp.nan, ksfc=jnp.nan, constraints=jnp.nan)
        with jdc.copy_and_mutate(metrics_results_nan, validate=False) as metrics_results:
            for met_name in metrics:
                metric_fun = getattr(self.metrics, met_name)
                metric_res = metric_fun(data_points)
                setattr(metrics_results, met_name, metric_res)
        return metrics_results

    def constraint_compliances(self, data_points):
        return jnp.sum(jax.vmap(self.constraint_function, in_axes=0)(data_points))

    def default_jsd(self, data_points, bandwidth=0.05):
        # losses use predefined constrained grid -> maybe option to choose different resolution for different metrics -> grid must be computed again
        n_grid_points = self.constraint_data_space_grid.shape[0]

        density_estimate = DensityEstimate(
            p=jnp.zeros([n_grid_points, 1]),
            x_g=self.constraint_data_space_grid,
            bandwidth=jnp.array([bandwidth]),
            n_observations=jnp.array([0]),
        )

        if data_points.shape[0] > 5000:
            # if there are too many datapoints at once, split them up and add
            # them in smaller chunks to the density estimate

            block_size = 1000

            for n in range(0, data_points.shape[0] + 1, block_size):
                density_estimate = update_density_estimate_multiple_observations(
                    density_estimate,
                    data_points[n : min(n + block_size, data_points.shape[0])],
                )
        else:
            density_estimate = update_density_estimate_multiple_observations(
                density_estimate,
                data_points,
            )

        # default uniform traget distribution #TODO extend to different options
        target_distribution = jnp.ones(density_estimate.p.shape)
        target_distribution /= jnp.sum(target_distribution)

        return JSDLoss(
            p=density_estimate.p / jnp.sum(density_estimate.p),
            q=target_distribution,
        )

    def default_ae(self, data_points):
        return audze_eglais(data_points)

    def default_mcudsa(self, data_points):
        # losses use predefined constrained grid -> maybe option to choose different resolution for different metrics -> grid must be computed again
        if self.data_dim > 2:
            return blockwise_mcudsa(data_points=data_points, support_points=self.constraint_data_space_grid)
        else:
            return MC_uniform_sampling_distribution_approximation(
                data_points=data_points, support_points=self.constraint_data_space_grid
            )

    def default_ksfc(self, data_points, variance=0.01, eps=1e-6):
        # losses use predefined constrained grid -> maybe option to choose different resolution for different metrics -> grid must be computed again
        if self.data_dim > 2:
            return blockwise_ksfc(
                data_points=data_points,
                support_points=self.constraint_data_space_grid,
                variances=jnp.ones([self.data_dim]) * variance,
                eps=eps,
            )
        else:
            return kiss_space_filling_cost(
                data_points=data_points,
                support_points=self.constraint_data_space_grid,
                variances=jnp.ones([self.data_dim]) * variance,
                eps=eps,
            )
