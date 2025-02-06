from exciting_environments import CartPole, MassSpringDamper, Pendulum, FluidTank, PMSM


def make(env_id: str, **env_kwargs):
    if env_id == "CartPole-v0":
        env = CartPole(**env_kwargs)

    elif env_id == "MassSpringDamper-v0":
        env = MassSpringDamper(**env_kwargs)

    elif env_id == "Pendulum-v0":
        env = Pendulum(**env_kwargs)

    elif env_id == "FluidTank-v0":
        env = FluidTank(**env_kwargs)

    elif env_id == "PMSM-v0":
        env = PMSM(**env_kwargs)

    else:
        print(f"No existing environments got env_id ={env_id}")
        env = None

    return env
