from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly, Part

import numpy as np
import torch
from . import materials
from ...interfaces import Serializable

class BaseElement(Serializable):
    
    _serialized_attributes_exclude = ['_indices_matrix', '_indices_force', '_index_matrix_coalesce', 'gaussian_weight', 'shape_function_gaussian']

    shape_function: list[torch.Tensor]
    """
        the shape functions of the element
    """

    _num_gaussian: int
    """
        the number of guassian points
    """

    num_nodes_per_elem: int
    """ 
        the number of nodes per element
    """

    gaussian_weight_ref: torch.Tensor
    """        the weight of each guassian point"""

    gaussian_coordinates: torch.Tensor
    """the coordinates of gaussian points in the reference space"""


    def __init__(self, elems_index: torch.Tensor, elems: torch.Tensor) -> None:

        super().__init__()

        self.shape_function_gaussian: list[torch.Tensor]
        """
            the shape of shape_function at each guassian point
        """
        
        self.gaussian_weight: torch.Tensor
        """
        the weight of each guassian point
            [
                g, the num of guassian point
            ]
        """
        self._elems_index = elems_index
        """
            the index of the element
        """
        self._elems = elems
        """
            [elem, N]\n
            the element connectivity 
        """

        # Materials are managed as a dict so one element can aggregate multiple
        # material contributions in energy/force/stiffness calculations.
        self.materials: dict[str, materials.Materials_Base] = {}
        # Optional element-local material regions.  When populated, each
        # element contributes through exactly one named material instead of
        # summing every material in the block.  The UI uses this for Abaqus
        # style material-to-element-set assignments.
        self.material_element_indices: dict[str, np.ndarray] = {}

        self._density: torch.Tensor = None
        """
            the density of the element
        """

        cls = self.__class__
        cls.shape_function = [cls.shape_function[i].to(torch.get_default_device()).to(torch.get_default_dtype()) for i in range(len(cls.shape_function))]
        cls.gaussian_weight_ref = cls.gaussian_weight_ref.to(torch.get_default_device()).to(torch.get_default_dtype())
        cls.gaussian_coordinates = cls.gaussian_coordinates.to(torch.get_default_device()).to(torch.get_default_dtype())

    @property
    def density(self) -> torch.Tensor:
        return self._density
    
    @density.setter
    def density(self, value: np.ndarray | torch.Tensor):
        if isinstance(value, np.ndarray):
            value = torch.tensor(value)
        self._density = value

    def get_gaussian_points(self, nodes: torch.Tensor) -> torch.Tensor:
        """
            get the gaussian points of the element
        """
        raise NotImplementedError('The gaussian points of the element is not implemented yet')

    def initialize(self, nodes: torch.Tensor, *args, **kwargs):
        pass
    
    def get_mass_matrix(self,rotation_matrix:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            get the mass matrix of the element
        Returns:
            indices (torch.Tensor): the indices of the mass matrix
            M (torch.Tensor): the mass matrix of the element
        """
        raise NotImplementedError('The mass matrix of the element is not implemented yet')

    def potential_Energy(self, RGC: torch.Tensor):
        pass

    def structural_Force(self, RGC: torch.Tensor,rotation_matrix:torch.Tensor, if_onlyforce: bool = False, *args, **kwargs):
        pass

    def _iter_material_values(self):
        """Iterate the named materials assigned to this element block."""
        if not isinstance(self.materials, dict):
            raise TypeError("materials must be a dict[str, Materials_Base]")
        return self.materials.values()

    def _iter_material_contributions(self):
        """Yield ``(material, local element indices)`` contributions.

        Empty assignments apply every named material to the whole element
        block; populated assignments select element regions.
        """
        assignments = self.material_element_indices
        if not assignments:
            yield from ((material, None) for material in self._iter_material_values())
            return
        for name, material in self.materials.items():
            indices = np.asarray(assignments[name], dtype=int).ravel()
            if len(indices):
                yield material, indices

    def set_materials(self, mat: materials.Materials_Base | dict[str, materials.Materials_Base],
                      name=None, element_indices=None):
        """
            Set materials of the element.

            Args:
                mat: one material or a dict of materials
                name: key when setting a single material; auto-generated if None
                element_indices: optional local element ids assigned to this material
        """

        if isinstance(mat, dict):
            self.materials = dict(mat)
            self.material_element_indices = {}
            if element_indices is not None:
                raise ValueError("element_indices cannot accompany a material dictionary.")
            return

        if not isinstance(mat, materials.Materials_Base):
            raise TypeError("mat must be Materials_Base or dict[str, Materials_Base]")

        if name is None:
            name = f"material-{len(self.materials)}"

        self.materials[name] = mat
        if element_indices is not None:
            indices = np.asarray(element_indices, dtype=int).ravel()
            if np.any(indices < 0) or np.any(indices >= len(self._elems)):
                raise ValueError(f"Material '{name}' contains an out-of-range element index.")
            self.material_element_indices[name] = np.unique(indices)

    def delete_material(self, name: str | None = None):
        """Delete one material by name, or clear all when name is None."""
        mats = self.materials

        if name is None:
            mats.clear()
            self.material_element_indices.clear()
            return
        if name not in mats:
            raise ValueError(f"Material '{name}' not found in this element")
        del mats[name]
        self.material_element_indices.pop(name, None)
        
    def set_required_DoFs(
            self, RGC_remain_index: np.ndarray) -> np.ndarray:
        """
        Modify the RGC_remain_index
        """

    def refine_RGC(self, RGC: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
            refine the RGC of the element
        """
        return RGC
