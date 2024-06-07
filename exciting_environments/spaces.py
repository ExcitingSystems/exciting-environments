import chex
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Any


class Space:

    def sample(self, rng: chex.PRNGKey):
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        raise NotImplementedError


class Box(Space):
    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.dtype = dtype
        self.shape = shape

    @partial(jax.jit, static_argnums=0)
    def sample(self, rng: chex.PRNGKey):
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        in_range = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return in_range
