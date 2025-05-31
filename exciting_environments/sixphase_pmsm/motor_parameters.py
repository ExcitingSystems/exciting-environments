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
    u_x_buffer: float
    u_y_buffer: float
    u_0p_buffer: float
    u_0m_buffer: float
    epsilon: float
    i_d: float
    i_q: float
    i_x: float
    i_y: float
    i_0p: float
    i_0m: float
    omega_el: float
    torque: float


@jdc.pytree_dataclass
class ActionNormalizations:
    u_d: float
    u_q: float
    u_x: float
    u_y: float
    u_0p: float
    u_0m: float


@jdc.pytree_dataclass
class StaticParams:
    p: int  # Number of pole pairs
    r_s: float  # Stator resistance
    l_d: float  # D-axis inductance
    l_q: float  # Q-axis inductance
    l_x: jax.Array  # X-axis inductance
    l_y: jax.Array  # Y-axis inductance
    l_d0: jax.Array  # D-axis zero-sequence inductance
    l_q0: jax.Array  # Q-axis zero-sequence inductance
    psi_p_dq: jax.Array  # Permanent magnet flux linkage in DQ frame
    psi_p_xy: jax.Array  # Permanent magnet flux linkage in XY frame
    u_dc: float  # DC link voltage
    deadtime: int  # Deadtime compensation


@jdc.pytree_dataclass
class MotorParams:
    physical_normalizations: PhysicalNormalizations
    action_normalizations: ActionNormalizations
    static_params: StaticParams
    default_soft_constraints: Callable
    pmsm_lut: dict


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


DEFAULT = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_x_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),  # TBD
        u_y_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),  # TBD
        u_0p_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),  # TBD
        u_0m_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),  # TBD
        epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
        i_d=MinMaxNormalization(min=(-250), max=(0)),
        i_q=MinMaxNormalization(min=(-250), max=(250)),  # TBD
        i_x=MinMaxNormalization(min=(-250), max=(0)),  # TBD
        i_y=MinMaxNormalization(min=(-250), max=(250)),  # TBD
        i_0p=MinMaxNormalization(min=(-250), max=(250)),  # TBD
        i_0m=MinMaxNormalization(min=(-250), max=(250)),  # TBD
        omega_el=MinMaxNormalization(min=0, max=(3 * 11000 * 2 * jnp.pi / 60)),
        torque=MinMaxNormalization(min=(-200), max=(200)),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_q=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_x=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_y=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_0p=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
        u_0m=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
    ),
    static_params=StaticParams(
        p=3,
        r_s=15e-3,
        l_d=0.37e-3,
        l_q=1.2e-3,
        l_x=0.37e-3,  # TBD
        l_y=1.2e-3,  # TBD
        l_d0=0.37e-3,  # TBD
        l_q0=1.2e-3,  # TBD
        psi_p_dq=0.65e-3,
        psi_p_xy=0,  # TBD
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
        name (str): Name of the motor.

    Returns:
        MotorConfig: Configuration containing physical constraints, action constraints, static parameters, and LUT data.
    """
    if name is None:
        return deepcopy(DEFAULT)
    else:
        raise ValueError(f"Motor name {name} is not known.")
