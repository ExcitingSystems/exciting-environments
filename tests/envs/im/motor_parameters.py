import jax
import jax.numpy as jnp

from typing import Callable
from dataclasses import fields
from scipy.io import loadmat
from pathlib import Path
import os
import jax_dataclasses as jdc
from exciting_environments.utils import MinMaxNormalization
from copy import deepcopy

from enum import Enum


@jdc.pytree_dataclass
class PhysicalNormalizations:
    u_alpha_buffer: float
    u_beta_buffer: float
    epsilon: float
    i_s_alpha: float
    i_s_beta: float
    psi_r_alpha: float
    psi_r_beta: float
    omega_el: float
    torque: float


@jdc.pytree_dataclass
class ActionNormalizations:
    u_alpha: float
    u_beta: float


@jdc.pytree_dataclass
class StaticParams:
    p: int  # Number of pole pairs
    r_s: float  # Stator resistance
    r_r: float  # Rotor resistance
    l_m: float  # Main inductance
    l_sigs: float  # Stator-side stray inductance
    l_sigr: float  # Rotor-side stray inductance
    u_dc: float  # DC link voltage
    deadtime: int  # Deadtime compensation


@jdc.pytree_dataclass
class MotorParams:
    physical_normalizations: PhysicalNormalizations
    action_normalizations: ActionNormalizations
    static_params: StaticParams
    default_soft_constraints: Callable
    lut: dict


# Predefined motor configurations


def default_soft_constraints(self, state, action_norm, env_properties):
    state_norm = self.normalize_state(state, env_properties)
    physical_state_norm = state_norm.physical_state
    with jdc.copy_and_mutate(physical_state_norm, validate=False) as phys_soft_const:
        for field in fields(phys_soft_const):
            name = field.name
            setattr(
                phys_soft_const,
                name,
                jax.nn.relu(jnp.abs(getattr(physical_state_norm, name)) - 1.0),
            )
    return phys_soft_const, None


def torque_limit(p, l_m, l_sigr, i_s_max):
    return 1.5 * p * l_m / (l_m + l_sigr) * l_m * i_s_max * i_s_max / 2


DEFAULT = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_alpha_buffer=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
        u_beta_buffer=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
        epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
        i_s_alpha=MinMaxNormalization(min=(-5.5), max=(5.5)),
        i_s_beta=MinMaxNormalization(min=(-5.5), max=(5.5)),
        psi_r_alpha=MinMaxNormalization(min=(-0.8), max=(0.8)),  # i_s_max * l_m = 5.5 * 143.75e-3 = 0,790625
        psi_r_beta=MinMaxNormalization(min=(-0.8), max=(0.8)),
        omega_el=MinMaxNormalization(min=0, max=(2 * 4000 * 2 * jnp.pi / 60)),
        torque=MinMaxNormalization(
            min=(-torque_limit(2, 143.75e-3, 5.87e-3, 5.5)), max=(torque_limit(2, 143.75e-3, 5.87e-3, 5.5))
        ),  #
    ),
    action_normalizations=ActionNormalizations(
        u_alpha=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
        u_beta=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
    ),
    static_params=StaticParams(
        p=2,
        r_s=2.9338,
        r_r=1.355,
        l_m=143.75e-3,
        l_sigs=5.87e-3,
        l_sigr=5.87e-3,
        u_dc=560,
        deadtime=0,
    ),
    default_soft_constraints=default_soft_constraints,
    lut=None,
)


class MotorVariant(Enum):
    DEFAULT = "DEFAULT"

    def get_params(self):
        return deepcopy(DEFAULT)
