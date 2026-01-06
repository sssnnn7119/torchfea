import torch
from .hyperelastic import NeoHookean, LinearElastic
from .base import Materials_Base


__all__ = ['initialize_materials', 'Materials_Base', 'NeoHookean', 'LinearElastic']

def initialize_materials(materials_type: int,
                         materials_params: torch.Tensor):

    if materials_type == 1:
        mu = materials_params[0, 0]
        kappa = materials_params[0, 1]
        mat_now = NeoHookean(mu, kappa)
    elif materials_type == 2:
        E = materials_params[0, 0]
        nu = materials_params[0, 1]
        mat_now = LinearElastic(E, nu)
    else:
        raise Exception('Unknown material type')
    return mat_now
