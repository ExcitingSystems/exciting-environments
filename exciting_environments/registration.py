from cart_pole import CartPole
from mass_spring_damper import MassSpringDamper
from pendulum import Pendulum

def make(env_id: str, **env_kwargs):
    match env_id:
        case 'CartPole-v0':
            env=CartPole(**env_kwargs)
        
        case 'MassSpringDamper-v0':
            env=MassSpringDamper(**env_kwargs)

        case 'Pendulum-v0':
            env=Pendulum(**env_kwargs)

    return env