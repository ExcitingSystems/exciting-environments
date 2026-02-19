import jax
import jax.numpy as jnp

from typing import Callable
from dataclasses import fields
from scipy.io import loadmat
from pathlib import Path
import os

import equinox as eqx
from exciting_environments.utils import MinMaxNormalization
from copy import deepcopy
import numpy as np
from scipy.interpolate import griddata


class PhysicalNormalizations(eqx.Module):
    u_d_buffer: jax.Array
    u_q_buffer: jax.Array
    epsilon: jax.Array
    i_d: jax.Array
    i_q: jax.Array
    omega_el: jax.Array
    torque: jax.Array


class ActionNormalizations(eqx.Module):
    u_d: jax.Array
    u_q: jax.Array


class StaticParams(eqx.Module):
    p: int  # Number of pole pairs
    r_s: jax.Array  # Stator resistance
    l_d: jax.Array  # D-axis inductance
    l_q: jax.Array  # Q-axis inductance
    psi_p: jax.Array  # Permanent magnet flux linkage
    u_dc: jax.Array  # DC link voltage
    deadtime: int  # Deadtime compensation


class MotorParams(eqx.Module):
    physical_normalizations: PhysicalNormalizations
    action_normalizations: ActionNormalizations
    static_params: StaticParams
    default_soft_constraints: Callable
    interpolators: dict


def generate_interpolators_and_lut(pmsm_lut):
    saturated_quants = [
        "L_dd",
        "L_dq",
        "L_qd",
        "L_qq",
        "Psi_d",
        "Psi_q",
    ]
    i_d_max = np.max(pmsm_lut["i_d_vec"])
    i_q_max = np.max(pmsm_lut["i_q_vec"])
    i_d_min = np.min(pmsm_lut["i_d_vec"])
    i_q_min = np.min(pmsm_lut["i_q_vec"])
    i_d_stepsize = (i_d_max - i_d_min) / (pmsm_lut["i_d_vec"].shape[1] - 1)
    i_q_stepsize = (i_q_max - i_q_min) / (pmsm_lut["i_q_vec"].shape[1] - 1)
    for q in saturated_quants:
        qmap = pmsm_lut[q]
        x, y = np.indices(qmap.shape)
        nan_mask = np.isnan(qmap)
        qmap[nan_mask] = griddata(
            (x[~nan_mask], y[~nan_mask]),  # points we know
            qmap[~nan_mask],  # values we know
            (x[nan_mask], y[nan_mask]),  # points to interpolate
            method="nearest",
        )  # extrapolation can only do nearest

        # repeat values ​​on the edge to have the linear extrapolation create constant extrapolation
        a = np.vstack([qmap[0, :], qmap, qmap[-1, :]])
        b = np.hstack([a[:, :1], a, a[:, -1:]])

        pmsm_lut[q] = b

    n_grid_points_y, n_grid_points_x = pmsm_lut[saturated_quants[0]].shape
    x, y = np.linspace(i_d_min - i_d_stepsize, i_d_max + i_d_stepsize, n_grid_points_x), np.linspace(
        i_q_min - i_q_stepsize, i_q_max + i_q_stepsize, n_grid_points_y
    )
    LUT_interpolators = {
        q: jax.scipy.interpolate.RegularGridInterpolator(
            (x, y),
            pmsm_lut[q][:, :].T,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        for q in saturated_quants
    }

    return LUT_interpolators, pmsm_lut


# Predefined motor configurations


def default_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize_state(state)
    physical_state_norm = state_norm.physical_state
    phys_soft_const = jax.tree.map(lambda x: jax.nn.relu(jnp.abs(x) - 1.0), physical_state_norm)
    return phys_soft_const, None


brusa_lut_raw = loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat"))

brusa_interpolators, brusa_lut = generate_interpolators_and_lut(brusa_lut_raw)


BRUSA = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        u_q_buffer=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        epsilon=MinMaxNormalization(min=(jnp.array(-jnp.pi)), max=(jnp.array(jnp.pi))),
        i_d=MinMaxNormalization(min=(jnp.array(-250)), max=(jnp.array(0))),
        i_q=MinMaxNormalization(min=(jnp.array(-250)), max=(jnp.array(250))),
        omega_el=MinMaxNormalization(min=jnp.array(0), max=(jnp.array(3 * 11000 * 2 * jnp.pi / 60))),
        torque=MinMaxNormalization(min=(jnp.array(-200)), max=(jnp.array(200))),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        u_q=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
    ),
    static_params=StaticParams(
        p=jnp.array(3),
        r_s=jnp.array(17.932e-3),
        l_d=jnp.array(0.37e-3),
        l_q=jnp.array(1.2e-3),
        psi_p=jnp.array(65.65e-3),
        u_dc=jnp.array(400),
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    interpolators=brusa_interpolators,
)

sew_lut_raw = loadmat(Path(__file__).parent / Path("LUT_SEW_jax_grad.mat"))

sew_interpolators, sew_lut = generate_interpolators_and_lut(sew_lut_raw)


SEW = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(jnp.array(-2 * 550 / 3)), max=(jnp.array(2 * 550 / 3))),
        u_q_buffer=MinMaxNormalization(min=(jnp.array(-2 * 550 / 3)), max=(jnp.array(2 * 550 / 3))),
        epsilon=MinMaxNormalization(min=(jnp.array(-jnp.pi)), max=(jnp.array(jnp.pi))),
        i_d=MinMaxNormalization(min=(jnp.array(-16)), max=(jnp.array(0))),
        i_q=MinMaxNormalization(min=(jnp.array(-16)), max=(jnp.array(16))),
        omega_el=MinMaxNormalization(min=jnp.array(0), max=(jnp.array(4 * 2000 / 60 * 2 * jnp.pi))),
        torque=MinMaxNormalization(min=(jnp.array(-15)), max=(jnp.array(15))),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(jnp.array(-2 * 550 / 3)), max=(jnp.array(2 * 550 / 3))),
        u_q=MinMaxNormalization(min=(jnp.array(-2 * 550 / 3)), max=(jnp.array(2 * 550 / 3))),
    ),
    static_params=StaticParams(
        p=jnp.array(4),
        r_s=jnp.array(208e-3),
        l_d=jnp.array(1.44e-3),
        l_q=jnp.array(1.44e-3),
        psi_p=jnp.array(122e-3),
        u_dc=jnp.array(550),
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    interpolators=sew_interpolators,
)

DEFAULT = MotorParams(
    physical_normalizations=PhysicalNormalizations(
        u_d_buffer=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        u_q_buffer=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        epsilon=MinMaxNormalization(min=(jnp.array(-jnp.pi)), max=(jnp.array(jnp.pi))),
        i_d=MinMaxNormalization(min=(jnp.array(-250)), max=(jnp.array(0))),
        i_q=MinMaxNormalization(min=(jnp.array(-250)), max=(jnp.array(250))),
        omega_el=MinMaxNormalization(min=jnp.array(0), max=(jnp.array(3 * 11000 * 2 * jnp.pi / 60))),
        torque=MinMaxNormalization(min=(jnp.array(-200)), max=(jnp.array(200))),
    ),
    action_normalizations=ActionNormalizations(
        u_d=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
        u_q=MinMaxNormalization(min=(jnp.array(-2 * 400 / 3)), max=(jnp.array(2 * 400 / 3))),
    ),
    static_params=StaticParams(
        p=jnp.array(3),
        r_s=jnp.array(15e-3),
        l_d=jnp.array(0.37e-3),
        l_q=jnp.array(1.2e-3),
        psi_p=jnp.array(65.6e-3),
        u_dc=jnp.array(400),
        deadtime=1,
    ),
    default_soft_constraints=default_soft_constraints,
    interpolators=brusa_interpolators,  # for sake of jax vmapping
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
