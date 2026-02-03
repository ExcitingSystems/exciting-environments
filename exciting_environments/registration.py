from exciting_environments import (
    CartPole,
    MassSpringDamper,
    Pendulum,
    FluidTank,
    PMSM,
    Acrobot,
)
from enum import Enum


class EnvironmentRegistry(Enum):
    CART_POLE = "CartPole-v0"
    MASS_SPRING_DAMPER = "MassSpringDamper-v0"
    PENDULUM = "Pendulum-v0"
    FLUID_TANK = "FluidTank-v0"
    PMSM = "PMSM-v0"
    ACROBOT = "Acrobot-v0"

    def make(self, **env_kwargs):
        env_map = {
            EnvironmentRegistry.CART_POLE: CartPole,
            EnvironmentRegistry.MASS_SPRING_DAMPER: MassSpringDamper,
            EnvironmentRegistry.PENDULUM: Pendulum,
            EnvironmentRegistry.FLUID_TANK: FluidTank,
            EnvironmentRegistry.PMSM: PMSM,
            EnvironmentRegistry.ACROBOT: Acrobot,
        }
        cls = env_map.get(self)
        if cls is None:
            raise ValueError(f"Unknown environment: {self}")
        return cls(**env_kwargs)
