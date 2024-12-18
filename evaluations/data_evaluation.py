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


class JensenShannonDivergence(eqx.Module):
    bandwidth: float
    grid: jax.Array
    target_distribution: jax.Array

    def __init__(self, grid, bandwidth=0.05, target_distribution=None):
        self.bandwidth = bandwidth
        self.grid = grid
        if target_distribution is None:
            # generate uniform target distribution
            self.target_distribution = self.generate_distribution(grid, bandwidth)
        else:
            self.target_distribution = target_distribution

    def __call__(self, data_points):

        data_distribution = self.generate_distribution(data_points, self.bandwidth)
        return JSDLoss(
            p=data_distribution,
            q=self.target_distribution,
        )

    def generate_distribution(self, data_points, bandwidth):
        n_grid_points = self.grid.shape[0]

        density_estimate = DensityEstimate(
            p=jnp.zeros([n_grid_points, 1]),
            x_g=self.grid,
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

        return density_estimate.p / jnp.sum(density_estimate.p)


class ConstraintCompliances(eqx.Module):
    constraint_function: Callable

    def __init__(self, constraint_function):
        self.constraint_function = constraint_function

    def __call__(self, data_points):
        return jnp.sum(jax.vmap(self.constraint_function, in_axes=0)(data_points))


class AudzeEglaise(eqx.Module):

    def __call__(self, data_points):
        return audze_eglais(data_points)


class MCUniformSamplingDistributionApproximation(eqx.Module):
    grid: jax.Array

    def __init__(self, grid):
        self.grid = grid

    def __call__(self, data_points):
        if self.grid.shape[1] > 2:
            return blockwise_mcudsa(data_points=data_points, support_points=self.grid)
        else:
            return MC_uniform_sampling_distribution_approximation(data_points=data_points, support_points=self.grid)


class KissSpaceFillingCosts(eqx.Module):
    grid: jax.Array
    variance: float
    eps: float

    def __init__(self, grid, variance=0.01, eps=1e-6):
        self.grid = grid
        self.variance = variance
        self.eps = eps

    def __call__(self, data_points):
        # losses use predefined constrained grid -> maybe option to choose different resolution for different metrics -> grid must be computed again
        data_dim = self.grid.shape[1]
        if data_dim > 2:
            return blockwise_ksfc(
                data_points=data_points,
                support_points=self.grid,
                variances=jnp.ones([data_dim]) * self.variance,
                eps=self.eps,
            )
        else:
            return kiss_space_filling_cost(
                data_points=data_points,
                support_points=self.grid,
                variances=jnp.ones([data_dim]) * self.variance,
                eps=self.eps,
            )


class Evaluator(eqx.Module):

    constraint_function: Callable
    constraint_data_space_grid: jax.Array
    jsd: eqx.Module
    ae: eqx.Module
    mcudsa: eqx.Module
    ksfc: eqx.Module
    cc: eqx.Module

    def __init__(
        self, constraint_function, data_dim, points_per_dim, jsd=None, ae=None, mcudsa=None, ksfc=None, cc=None
    ):
        self.constraint_function = constraint_function
        hypercube_grid = build_grid(data_dim, -1, 1, points_per_dim)
        self.constraint_data_space_grid = self.valid_space_grid(hypercube_grid, constraint_function)

        # create metrics with default params
        # TODO tests?
        if type(jsd) is JensenShannonDivergence:
            self.jsd = jsd
        else:
            self.jsd = JensenShannonDivergence(grid=self.constraint_data_space_grid)

        if type(ae) is AudzeEglaise:
            self.ae = ae
        else:
            self.ae = AudzeEglaise()

        if type(mcudsa) is MCUniformSamplingDistributionApproximation:
            self.mcudsa = mcudsa
        else:
            self.mcudsa = MCUniformSamplingDistributionApproximation(grid=self.constraint_data_space_grid)

        if type(ksfc) is KissSpaceFillingCosts:
            self.ksfc = ksfc
        else:
            self.ksfc = KissSpaceFillingCosts(grid=self.constraint_data_space_grid)

        if type(cc) is ConstraintCompliances:
            self.cc = cc
        else:
            self.cc = ConstraintCompliances(constraint_function=constraint_function)

    def valid_space_grid(self, data_grid, constr_func):
        valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0
        constraint_data_grid = data_grid[jnp.where(valid_grid_point == True)]
        return constraint_data_grid

    def get_default_metrics(self, data_points, metrics=["jsd", "ae", "mcudsa", "ksfc", "cc"]):
        metrics_results = {}
        for met_name in metrics:
            metric_fun = getattr(self, met_name)
            metric_res = metric_fun(data_points)
            metrics_results[met_name] = metric_res
        return metrics_results
