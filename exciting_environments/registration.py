from .cart_pole import CartPole
from .mass_spring_damper import MassSpringDamper
from .pendulum import Pendulum

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