import jax
import jax.numpy as jnp

from scipy.io import loadmat
from pathlib import Path
import os
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class PhysicalConstraints:
    u_d_buffer: float
    u_q_buffer: float
    epsilon: float
    i_d: float
    i_q: float
    omega_el: float
    torque: float


@jdc.pytree_dataclass
class ActionConstraints:
    u_d: float
    u_q: float


@jdc.pytree_dataclass
class StaticParams:
    p: int  # Number of pole pairs
    r_s: float  # Stator resistance
    l_d: float  # D-axis inductance
    l_q: float  # Q-axis inductance
    psi_p: float  # Permanent magnet flux linkage
    deadtime: int  # Deadtime compensation


@jdc.pytree_dataclass
class MotorParams:
    physical_constraints: PhysicalConstraints
    action_constraints: ActionConstraints
    static_params: StaticParams
    pmsm_lut: dict


# Predefined motor configurations
BRUSA = MotorParams(
    physical_constraints=PhysicalConstraints(
        u_d_buffer=2 * 400 / 3,
        u_q_buffer=2 * 400 / 3,
        epsilon=jnp.pi,
        i_d=250,
        i_q=250,
        omega_el=3 * 11000 * 2 * jnp.pi / 60,
        torque=200,
    ),
    action_constraints=ActionConstraints(
        u_d=2 * 400 / 3,
        u_q=2 * 400 / 3,
    ),
    static_params=StaticParams(
        p=3,
        r_s=17.932e-3,
        l_d=0.37e-3,
        l_q=1.2e-3,
        psi_p=65.65e-3,
        deadtime=1,
    ),
    pmsm_lut=loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat")),
)

SEW = MotorParams(
    physical_constraints=PhysicalConstraints(
        u_d_buffer=2 * 550 / 3,
        u_q_buffer=2 * 550 / 3,
        epsilon=jnp.pi,
        i_d=16,
        i_q=16,
        omega_el=4 * 2000 / 60 * 2 * jnp.pi,
        torque=15,
    ),
    action_constraints=ActionConstraints(
        u_d=2 * 550 / 3,
        u_q=2 * 550 / 3,
    ),
    static_params=StaticParams(
        p=4,
        r_s=208e-3,
        l_d=1.44e-3,
        l_q=1.44e-3,
        psi_p=122e-3,
        deadtime=1,
    ),
    pmsm_lut=loadmat(Path(__file__).parent / Path("LUT_SEW_jax_grad.mat")),
)

DEFAULT = MotorParams(
    physical_constraints=PhysicalConstraints(
        u_d_buffer=2 * 400 / 3,
        u_q_buffer=2 * 400 / 3,
        epsilon=jnp.pi,
        i_d=250,
        i_q=250,
        omega_el=3 * 1000 / 60 * 2 * jnp.pi,
        torque=200,
    ),
    action_constraints=ActionConstraints(
        u_d=2 * 400 / 3,
        u_q=2 * 400 / 3,
    ),
    static_params=StaticParams(
        p=4,
        r_s=15e-3,
        l_d=0.37e-3,
        l_q=1.2e-3,
        psi_p=65.6e-3,
        deadtime=1,
    ),
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
        return DEFAULT
    elif name == "BRUSA":
        return BRUSA
    elif name == "SEW":
        return SEW
    else:
        raise ValueError(f"Motor name {name} is not known.")
