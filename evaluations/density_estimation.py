import jax
import jax.numpy as jnp

import equinox as eqx


def select_bandwidth(
    delta_x: float,
    dim: int,
    n_g: int,
    percentage: float,
):
    """Select a bandwidth for the kernel density estimate by a rough heuristic.

    The bandwidth is designed so that the kernel is still at a given percentage of
    its maximum value when a step is taken in each dimension of the underlying
    grid.

    Args:
        delta_x: The size of the space in each dimension.
        dim: The dimension of the space.
        n_g: Number of grid points per dimension.
        percentage: The percentage of the maximum value of the kernel at the other
            grid point reached by stepping once in each dimension on the grid.
    """
    return delta_x * jnp.sqrt(dim) / (n_g * jnp.sqrt(-2 * jnp.log(percentage)))


@jax.jit
def gaussian_kernel(x: jax.Array, bandwidth: float) -> jax.Array:
    """Evaluates the Gaussian RBF kernel at x with given bandwidth. This can take arbitrary
    dimensions for 'x' and will compute the output by broadcasting. The last dimension of
    the input needs to be the dimension of the data which is reduced along.
    """
    data_dim = x.shape[-1]
    factor = bandwidth**data_dim * jnp.power(2 * jnp.pi, data_dim / 2)
    return 1 / factor * jnp.exp(-jnp.linalg.norm(x, axis=-1) ** 2 / (2 * bandwidth**2))


class DensityEstimate(eqx.Module):
    """Holds an estimation of the density of sampled data points.

    Args:
        p: The probability estimates at the grid points
        x_g: The grid points
        bandwidth: The bandwidth of the kernel density estimate
        n_observations: The number of observations that make up the current
            estimate
    """

    p: jax.Array
    x_g: jax.Array
    bandwidth: jax.Array
    n_observations: jax.Array

    @classmethod
    def from_estimate(cls, p, n_additional_observations, density_estimate):
        """Create a density estimate recursively from an existing estimate."""

        return cls(
            p=p,
            n_observations=(density_estimate.n_observations + n_additional_observations),
            x_g=density_estimate.x_g,
            bandwidth=density_estimate.bandwidth,
        )

    @classmethod
    def from_dataset(
        cls, observations, actions, use_actions=True, points_per_dim=30, x_min=-1, x_max=1, bandwidth=0.05
    ):
        """Create a fresh density estimate from gathered data."""

        if use_actions:
            dim = observations.shape[-1] + actions.shape[-1]
        else:
            dim = observations.shape[-1]
        n_grid_points = points_per_dim**dim

        density_estimate = cls(
            p=jnp.zeros([1, n_grid_points, 1]),
            x_g=build_grid(dim, x_min, x_max, points_per_dim),
            bandwidth=jnp.array([bandwidth]),
            n_observations=jnp.array([0]),
        )

        if observations.shape[0] == actions.shape[0] + 1:
            data_points = (
                jnp.concatenate([observations[0:-1, :], actions], axis=-1)[None] if use_actions else observations[None]
            )
        else:
            data_points = jnp.concatenate([observations, actions], axis=-1)[None] if use_actions else observations[None]

        density_estimate = jax.vmap(
            update_density_estimate_multiple_observations,
            in_axes=(DensityEstimate(0, None, None, None), 0),
            out_axes=(DensityEstimate(0, None, None, None)),
        )(
            density_estimate,
            data_points,
        )
        return density_estimate


@jax.jit
def update_density_estimate_single_observation(
    density_estimate: DensityEstimate,
    data_point: jax.Array,
) -> jax.Array:
    """Recursive update to the kernel density estimation (KDE) on a fixed grid.

    Args:
        density_estimate: The density estimate before the update
        data_point: The new data point

    Returns:
        The updated density estimate
    """
    kernel_value = gaussian_kernel(x=density_estimate.x_g - data_point, bandwidth=density_estimate.bandwidth)
    p_est = (
        1
        / (density_estimate.n_observations + 1)
        * (density_estimate.n_observations * density_estimate.p + kernel_value[..., None])
    )

    return DensityEstimate.from_estimate(p=p_est, n_additional_observations=1, density_estimate=density_estimate)


@jax.jit
def update_density_estimate_multiple_observations(
    density_estimate: DensityEstimate,
    data_points: jax.Array,
) -> jax.Array:
    """Add a new sequence of data points to the current data density estimate.

    Args:
        density_estimate: The density estimate before the update
        data_points: The sequence of data_points

    Returns:
        The updated values for the density estimate
    """

    def shifted_gaussian_kernel(x, observation, bandwidth):
        return gaussian_kernel(x - observation, bandwidth)

    new_sum_part = jax.vmap(shifted_gaussian_kernel, in_axes=(None, 0, None))(
        density_estimate.x_g, data_points, density_estimate.bandwidth
    )
    new_sum_part = jnp.sum(new_sum_part, axis=0)[..., None]
    p_est = (
        1
        / (density_estimate.n_observations + data_points.shape[0])
        * (density_estimate.n_observations * density_estimate.p + new_sum_part)
    )

    return DensityEstimate.from_estimate(
        p=p_est, n_additional_observations=data_points.shape[0], density_estimate=density_estimate
    )


def build_grid(dim, low, high, points_per_dim):
    """Build a uniform grid of points in the given dimension."""
    xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    x_g = jnp.meshgrid(*xs)
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, dim)

    assert x_g.shape[0] == points_per_dim**dim
    return x_g


def build_grid_2d(low, high, points_per_dim):
    return build_grid(2, low, high, points_per_dim)


def build_grid_3d(low, high, points_per_dim):
    return build_grid(3, low, high, points_per_dim)
