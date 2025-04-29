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


@jdc.pytree_dataclass
class PhysicalNormalizations:
    u_d_buffer: float
    u_q_buffer: float
    epsilon: float
    i_d: float
    i_q: float
    omega_el: float
    torque: float


@jdc.pytree_dataclass
class ActionNormalizations:
    u_d: float
    u_q: float


@jdc.pytree_dataclass
class StaticParams:
    p: int  # Number of pole pairs
    r_s: float  # Stator resistance
    l_d: float  # D-axis inductance
    l_q: float  # Q-axis inductance
    psi_p: float  # Permanent magnet flux linkage
    u_dc: float  # DC link voltage
    deadtime: int  # Deadtime compensation


@jdc.pytree_dataclass
class MotorParams:
    physical_normalizations: PhysicalNormalizations
    action_normalizations: ActionNormalizations
    static_params: StaticParams
    default_soft_constraints: Callable
    pmsm_lut: dict


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


BRUSA = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
        i_d=MinMaxNormalization(min=(-250), max=(0)),
        i_q=MinMaxNormalization(min=(-250), max=(250)),
        omega_el=MinMaxNormalization(min=0, max=(3 * 11000 * 2 * jnp.pi / 60)),
        torque=MinMaxNormalization(min=(-200), max=(200)),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
    ),
    static_params=StaticParams(
        p=3,
        r_s=17.932e-3,
        l_d=0.37e-3,
        l_q=1.2e-3,
        psi_p=65.65e-3,
        u_dc=400,
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    pmsm_lut=loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat")),
)

SEW = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
        u_q_buffer=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
        epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
        i_d=MinMaxNormalization(min=(-16), max=(0)),
        i_q=MinMaxNormalization(min=(-16), max=(16)),
        omega_el=MinMaxNormalization(min=0, max=(4 * 2000 / 60 * 2 * jnp.pi)),
        torque=MinMaxNormalization(min=(-15), max=(15)),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
        u_q=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
    ),
    static_params=StaticParams(
        p=4,
        r_s=208e-3,
        l_d=1.44e-3,
        l_q=1.44e-3,
        psi_p=122e-3,
        u_dc=550,
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    pmsm_lut=loadmat(Path(__file__).parent / Path("LUT_SEW_jax_grad.mat")),
)

DEFAULT = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
        i_d=MinMaxNormalization(min=(-250), max=(0)),
        i_q=MinMaxNormalization(min=(-250), max=(250)),
        omega_el=MinMaxNormalization(min=0, max=(3 * 11000 * 2 * jnp.pi / 60)),
        torque=MinMaxNormalization(min=(-200), max=(200)),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
    ),
    static_params=StaticParams(
        p=3,
        r_s=15e-3,
        l_d=0.37e-3,
        l_q=1.2e-3,
        psi_p=65.6e-3,
        u_dc=400,
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    pmsm_lut=None,
)


def default_params(name):
    """
    Returns default parameters for specified motor configurations.

    Args:
        name (str): Name of the motor ("BRUSA" or "SEW").

    Returns:
        MotorConfig: Configuration containing physical constraints, action constraints, static parameters, and LUT data.
    """
    if name is None:
        return deepcopy(DEFAULT)
    elif name == "BRUSA":
        return deepcopy(BRUSA)
    elif name == "SEW":
        return deepcopy(SEW)
    else:
        raise ValueError(f"Motor name {name} is not known.")
