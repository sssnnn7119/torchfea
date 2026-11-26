import torch

from .base import Materials_Base
from .hyperelastic import (
    ArrudaBoyce,
    Gent,
    MooneyRivlin,
    NeoHookean,
    NeoHookeanLnJ,
    Ogden,
    Yeoh,
)
from .linear import LinearElastic

__all__ = [
    "ArrudaBoyce",
    "Gent",
    "LinearElastic",
    "Materials_Base",
    "MooneyRivlin",
    "NeoHookean",
    "NeoHookeanLnJ",
    "Ogden",
    "Yeoh",
    "initialize_materials",
]


def initialize_materials(materials_type: int, materials_params: torch.Tensor):

    if materials_type == 1:
        mu = materials_params[0, 0]
        kappa = materials_params[0, 1]
        mat_now = NeoHookean(mu, kappa)
    elif materials_type == 2:
        E = materials_params[0, 0]
        nu = materials_params[0, 1]
        mat_now = LinearElastic(E, nu)
    else:
        raise Exception("Unknown material type")
    return mat_now
