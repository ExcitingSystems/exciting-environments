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
    u_alpha_buffer: float
    u_beta_buffer: float
    epsilon: float
    i_s_alpha: float
    i_s_beta: float
    psi_r_alpha: float
    psi_r_beta: float
    omega_el: float
    torque: float
    i_sl_alpha: float
    i_sl_beta: float


@jdc.pytree_dataclass
class ActionNormalizations:
    u_alpha: float
    u_beta: float


@jdc.pytree_dataclass
class StaticParams:
    p: int
    r_fe: float
    l_m: float
    l_sigs: float
    l_sigr: float
    r_r: float
    r_s: float
    h_r: float
    h_s: float
    u_dc: float
    omega_rs_N: float
    psi_r_N: float
    deadtime: int


@jdc.pytree_dataclass
class SaturationParams:
    k1: float
    k2: float
    k3: float
    k4: float


@jdc.pytree_dataclass
class MotorParams:
    physical_normalizations: PhysicalNormalizations
    action_normalizations: ActionNormalizations
    static_params: StaticParams
    static_params_nonlinear: StaticParams
    default_soft_constraints: Callable
    saturation_params: SaturationParams


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


# BRUSA = MotorParams(
#     physical_normalizations=PhysicalNormalizations(
#         u_d_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
#         u_q_buffer=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
#         epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
#         i_d=MinMaxNormalization(min=(-250), max=(0)),
#         i_q=MinMaxNormalization(min=(-250), max=(250)),
#         omega_el=MinMaxNormalization(min=0, max=(3 * 11000 * 2 * jnp.pi / 60)),
#         torque=MinMaxNormalization(min=(-200), max=(200)),
#     ),
#     action_normalizations=ActionNormalizations(
#         u_d=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
#         u_q=MinMaxNormalization(min=(-2 * 400 / 3), max=(2 * 400 / 3)),
#     ),
#     static_params=StaticParams(
#         p=3,
#         r_s=17.932e-3,
#         l_d=0.37e-3,
#         l_q=1.2e-3,
#         psi_p=65.65e-3,
#         u_dc=400,
#         deadtime=1,
#     ),
#     default_soft_constraints=default_soft_constraints,
#     pmsm_lut=None,
# )

# SEW = MotorParams(
#     physical_normalizations=PhysicalNormalizations(
#         u_d_buffer=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
#         u_q_buffer=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
#         epsilon=MinMaxNormalization(min=(-jnp.pi), max=(jnp.pi)),
#         i_d=MinMaxNormalization(min=(-16), max=(0)),
#         i_q=MinMaxNormalization(min=(-16), max=(16)),
#         omega_el=MinMaxNormalization(min=0, max=(4 * 2000 / 60 * 2 * jnp.pi)),
#         torque=MinMaxNormalization(min=(-15), max=(15)),
#     ),
#     action_normalizations=ActionNormalizations(
#         u_d=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
#         u_q=MinMaxNormalization(min=(-2 * 550 / 3), max=(2 * 550 / 3)),
#     ),
#     static_params=StaticParams(
#         p=4,
#         r_s=208e-3,
#         l_d=1.44e-3,
#         l_q=1.44e-3,
#         psi_p=122e-3,
#         u_dc=550,
#         deadtime=1,
#     ),
#     default_soft_constraints=default_soft_constraints,
#     pmsm_lut=None,
# )

# Parameters from  DOI: 10.1109/EPEPEMC.2018.8522008
#           and DOI: 10.1109/TPEL.2021.3080129


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
        psi_r_beta=MinMaxNormalization(min=(-0.8), max=(0.8)),  # TODO
        omega_el=MinMaxNormalization(min=0, max=(2 * 4000 * 2 * jnp.pi / 60)),
        torque=MinMaxNormalization(
            min=(-torque_limit(2, 143.75e-3, 5.87e-3, 5.5)), max=(torque_limit(2, 143.75e-3, 5.87e-3, 5.5))
        ),  #
        i_sl_alpha=MinMaxNormalization(min=(-5.5), max=(5.5)),
        i_sl_beta=MinMaxNormalization(min=(-5.5), max=(5.5)),
    ),
    action_normalizations=ActionNormalizations(
        u_alpha=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
        u_beta=MinMaxNormalization(min=(-2 * 560 / 3), max=(2 * 560 / 3)),
    ),
    #  p: int
    # r_s: float
    # r_r: float
    # r_fe: float
    # l_m: float
    # l_sigs: float
    # l_sigr: float
    # r_dcr: float
    # r_dcs: float
    # h_r: float
    # h_s: float
    # u_dc: float
    # deadtime: int
    static_params=StaticParams(
        p=2,
        r_fe=700.4,
        r_r=1.355,
        r_s=2.9338,
        h_r=jnp.nan,
        h_s=jnp.nan,
        l_m=143.75e-3,
        l_sigs=5.87e-3,
        l_sigr=5.87e-3,
        omega_rs_N=3000 * 2 * 2 * jnp.pi / 60,
        psi_r_N=0.58,
        u_dc=560,  # estimation
        deadtime=0,
    ),
    static_params_nonlinear=StaticParams(
        p=2,
        r_fe=700.4,
        r_r=1.7297,
        r_s=1.6997,
        h_r=0.0029,
        h_s=0.6780,
        l_m=143.75e-3,
        l_sigs=0.0046,
        l_sigr=0.0101,
        omega_rs_N=3000 * 2 * 2 * jnp.pi / 60,
        psi_r_N=0.58,
        u_dc=560,  # estimation
        deadtime=0,
    ),
    default_soft_constraints=default_soft_constraints,
    saturation_params=SaturationParams(
        k1=0.1596,
        k2=0.0478,
        k3=39.4442,
        k4=0.4938,
    ),
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
    # elif name == "BRUSA":
    #     return deepcopy(BRUSA)
    # elif name == "SEW":
    #     return deepcopy(SEW)
    else:
        raise ValueError(f"Motor name {name} is not known.")
