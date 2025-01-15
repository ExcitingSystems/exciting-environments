import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Normalization:
    min: float
    max: float
