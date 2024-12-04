from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx


class Evaluator(eqx.Module):

    constraint_function: Callable
    points_per_dim: int
    data_dim: int
    hypercube_grid: list
    data_space: jax.Array

    def __init__(self, constraint_function, data_dim, points_per_dim):
        self.constraint_function = constraint_function
        self.points_per_dim = points_per_dim
        self.data_dim = data_dim
        para = [jnp.linspace(-1, 1, points_per_dim) for _ in range(3)]
        hypercube_grid = jnp.meshgrid(*para)
        self.valid_data_space = constraint_function(hypercube_grid) == 0
        self.hypercube_grid = hypercube_grid
