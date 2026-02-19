from exciting_environments import (
    CartPole,
    MassSpringDamper,
    Pendulum,
    FluidTank,
    PMSM,
    Acrobot,
)
import jax
import jax.numpy as jnp


def make(env_id: str, batch_size: int = None, **env_kwargs):
    def create_single_env():
        if env_id == "CartPole-v0":
            return CartPole(**env_kwargs)
        elif env_id == "MassSpringDamper-v0":
            return MassSpringDamper(**env_kwargs)
        elif env_id == "Pendulum-v0":
            return Pendulum(**env_kwargs)
        elif env_id == "FluidTank-v0":
            return FluidTank(**env_kwargs)
        elif env_id == "PMSM-v0":
            return PMSM(**env_kwargs)
        elif env_id == "Acrobot-v0":
            return Acrobot(**env_kwargs)
        else:
            raise ValueError(f"No existing environments got env_id ={env_id}")

    if batch_size is not None:
        envs_list = [create_single_env() for _ in range(batch_size)]
        batched_envs = jax.tree.map(lambda *args: jnp.stack(args), *envs_list)
        return batched_envs

    return create_single_env()
