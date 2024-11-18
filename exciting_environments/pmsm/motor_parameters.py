import jax
import jax.numpy as jnp

from scipy.io import loadmat
from pathlib import Path
import os


def default_paras(name):
    if name == "BRUSA":
        default_physical_constraints = {
            "u_d_buffer": 2 * 400 / 3,
            "u_q_buffer": 2 * 400 / 3,
            "epsilon": jnp.pi,
            "i_d": 250,
            "i_q": 250,
            "omega_el": 3 * 11000 * 2 * jnp.pi / 60,
            "torque": 200,
        }

        default_action_constraints = {
            "u_d": 2 * 400 / 3,
            "u_q": 2 * 400 / 3,
        }
        default_static_params = {
            "p": 3,
            "r_s": 17.932e-3,
            "l_d": 0.37e-3,
            "l_q": 1.2e-3,
            "psi_p": 65.65e-3,
            "deadtime": 1,
        }
        pmsm_lut = loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat"))

    elif name == "SEW":
        default_physical_constraints = {
            "u_d_buffer": 2 * 550 / 3,
            "u_q_buffer": 2 * 550 / 3,
            "epsilon": jnp.pi,
            "i_d": 16,
            "i_q": 16,
            "omega_el": 4 * 2000 / 60 * 2 * jnp.pi,
            "torque": 15,
        }

        default_action_constraints = {
            "u_d": 2 * 550 / 3,
            "u_q": 2 * 550 / 3,
        }
        default_static_params = {
            "p": 4,
            "r_s": 208e-3,
            "l_d": 1.44e-3,
            "l_q": 1.44e-3,
            "psi_p": 122e-3,
            "deadtime": 1,
        }

        pmsm_lut = loadmat(Path(__file__).parent / Path("LUT_SEW_jax_grad.mat"))

    else:
        default_physical_constraints = None
        default_action_constraints = None
        default_static_params = None
        pmsm_lut = None
        raise Exception(f"Motor name {name} is not known.")

    return default_physical_constraints, default_action_constraints, default_static_params, pmsm_lut
