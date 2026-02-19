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
from enum import Enum


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


# Predefined motor configurations


def default_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize_state(state)
    physical_state_norm = state_norm.physical_state
    phys_soft_const = jax.tree.map(lambda x: jax.nn.relu(jnp.abs(x) - 1.0), physical_state_norm)
    return phys_soft_const, None


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
    interpolators=None,
)


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
    interpolators=None,
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
    interpolators=None,
)


class MotorVariant(Enum):
    DEFAULT = "DEFAULT"
    BRUSA = "BRUSA"
    SEW = "SEW"

    def get_params(self):
        if self is MotorVariant.BRUSA:
            return deepcopy(BRUSA)
        elif self is MotorVariant.SEW:
            return deepcopy(SEW)
        else:
            return deepcopy(DEFAULT)
