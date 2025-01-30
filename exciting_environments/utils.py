import jax_dataclasses as jdc
import equinox as eqx
from typing import Callable


@jdc.pytree_dataclass
class MinMaxNormalization:
    min: float
    max: float

    def normalize(self, denormalized_value):
        return 2 * (denormalized_value - self.min) / (self.max - self.min) - 1

    def denormalize(self, normalized_value):
        return (normalized_value + 1) / 2 * (self.max - self.min) + self.min
