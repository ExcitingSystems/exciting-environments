from cart_pole.cart_pole_env import CartPole
from mass_spring_damper.mass_spring_damper_env import MassSpringDamper
from pendulum.pendulum_env import Pendulum

def make(env_id: str, **env_kwargs):
    if env_id== 'CartPole-v0':
        env=CartPole(**env_kwargs)
    
    elif env_id=='MassSpringDamper-v0':
        env=MassSpringDamper(**env_kwargs)

    elif env_id=='Pendulum-v0':
        env=Pendulum(**env_kwargs)

    else:
        print(f"No existing environments got env_id ={env_id}")
        env=None

    return env