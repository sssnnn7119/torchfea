import numpy as np
import torch

from . import materials
from .base import BaseElement
from .dimension3 import C3D4, C3D6, C3D8, C3D10, C3D15, C3D20, Element_3D, surfaces
from .dimension3.surfaces import Q4, Q8, T3, T6, BaseSurface, initialize_surfaces


def initialize_element(
    element_type: str, elems_index: torch.Tensor, elems: torch.Tensor, *args, **kwargs
) -> BaseElement:
    """
    Initialize the element based on the element type.

    Args:
        element_type (str): The type of the element to initialize.
        elems (np.ndarray): The elements array containing element connectivity.
        nodes (torch.Tensor): The nodes array containing node coordinates.
        materials (torch.Tensor): The materials array containing material properties.

    Returns:
        Element_Base: An instance of the specified element type.
    """
    class_names = [item.split(".")[-1] for item in BaseElement._subclasses.keys()]

    element_class = (
        list(BaseElement._subclasses.values())[
            np.where(np.array(class_names) == element_type)[0][0]
        ]
        if element_type in class_names
        else None
    )

    if element_class is None:
        raise ValueError(f"Element type '{element_type}' is not recognized.")

    result = element_class(elems_index=elems_index, elems=elems)

    return result
