import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx


def build_grid(dim, low, high, points_per_dim):
    """Build a uniform grid of points in the given dimension."""
    xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    x_g = jnp.meshgrid(*xs)
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, dim)

    assert x_g.shape[0] == points_per_dim**dim
    return x_g


def get_valid_points(data_grid, constr_func):
    valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0

    # M = data_grid.shape[0]
    # block_size = 10_000
    # constraint_data_points = np.array([]).reshape(0, data_grid.shape[-1])

    # for m in range(0, M, block_size):
    #     end = min(m + block_size, M)
    #     # next block or until the end
    #     data = data_grid[m:end]
    #     val = valid_grid_point[m:end]
    #     constraint_data_points = jnp.concatenate([constraint_data_points, data[jnp.where(val == True)]])
    # constraint_data_points = jnp.array(constraint_data_points)

    constraint_data_points = data_grid[jnp.where(valid_grid_point == True)]
    return constraint_data_points


def valid_space_grid(constraint_function, data_dim, points_per_dim, min, max):
    hypercube_grid = build_grid(data_dim, min, max, points_per_dim)
    return get_valid_points(hypercube_grid, constraint_function)


# def get_valid_points2(data_grid, constr_func):
#     valid_grid_point = jax.lax.map(constr_func, data_grid) == 0
#     constraint_data_grid = data_grid[jnp.where(valid_grid_point == True)]
#     return constraint_data_grid


# def valid_space_grid2(constraint_function, data_dim, points_per_dim, min, max):
#     hypercube_grid = build_grid(data_dim, min, max, points_per_dim)
#     return get_valid_points2(hypercube_grid, constraint_function)
