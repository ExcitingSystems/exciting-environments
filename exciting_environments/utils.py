import jax_dataclasses as jdc
import equinox as eqx
from typing import Callable

from dataclasses import asdict
import json

@jdc.pytree_dataclass
class MinMaxNormalization:
    min: float
    max: float

    def normalize(self, denormalized_value):
        return 2 * (denormalized_value - self.min) / (self.max - self.min) - 1

    def denormalize(self, normalized_value):
        return (normalized_value + 1) / 2 * (self.max - self.min) + self.min



def dump_sim_properties_to_json(params, action_normalizations, physical_normalizations, tau, filename):
    action_norm_serialized = {
    k: asdict(v) for k, v in action_normalizations.items()
    }
    physical_norm_serialized = {
            k: asdict(v) for k, v in physical_normalizations.items()
        }
    data = {
        "params": params,
        "action_normalizations": action_norm_serialized,
        "physical_normalizations": physical_norm_serialized,
        "tau": tau
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_sim_properties_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    params= data["params"]
    action_norm_serialized = data["action_normalizations"]
    physical_norm_serialized = data["physical_normalizations"]
    tau = data["tau"]
    action_normalizations = {
        key: MinMaxNormalization(**value)
        for key, value in action_norm_serialized.items()
    }
    physical_normaliztions = {
        key: MinMaxNormalization(**value)
        for key, value in physical_norm_serialized.items()
    }
    return params, action_normalizations, physical_normaliztions, tau