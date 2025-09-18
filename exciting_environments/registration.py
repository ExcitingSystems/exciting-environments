from exciting_environments import (
    CartPole,
    MassSpringDamper,
    Pendulum,
    FluidTank,
    PMSM,
    Acrobot,
)
from enum import Enum


class EnvironmentType(Enum):
    CART_POLE = "CartPole-v0"
    MASS_SPRING_DAMPER = "MassSpringDamper-v0"
    PENDULUM = "Pendulum-v0"
    FLUID_TANK = "FluidTank-v0"
    PMSM = "PMSM-v0"
    ACROBOT = "Acrobot-v0"

    def make(self, **env_kwargs):
        env_map = {
            EnvironmentType.CART_POLE: CartPole,
            EnvironmentType.MASS_SPRING_DAMPER: MassSpringDamper,
            EnvironmentType.PENDULUM: Pendulum,
            EnvironmentType.FLUID_TANK: FluidTank,
            EnvironmentType.PMSM: PMSM,
            EnvironmentType.ACROBOT: Acrobot,
        }
        cls = env_map.get(self)
        if cls is None:
            raise ValueError(f"Unknown environment: {self}")
        return cls(**env_kwargs)
