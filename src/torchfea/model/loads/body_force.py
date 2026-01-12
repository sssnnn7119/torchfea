import numpy as np
import torch
from .base import BaseLoad
from ..elements.dimension3 import Element_3D
class BodyForce(BaseLoad):

    def __init__(self, instance_name: str, element_name: str, force_density: list[float] = [0.0, 0.0, -9.81e-6]) -> None:
        """
        Initialize the body force load.
        
        Args:
            force_density (list[float]): The body force density vector [fx, fy, fz]. (unit: force per unit volume)
        """
        super().__init__()
        self._parameters = torch.tensor(force_density, dtype=torch.float64)
        
        self._element_name = element_name

        self._instance_name = instance_name

        self._pdU_indices: torch.Tensor
        self._pdU_values: torch.Tensor

        self._element: Element_3D

        self._instance_RGC_index: int

    def initialize(self, assembly):
        super().initialize(assembly)
        
        # Collect all element sets and their elements
        self._element = assembly.get_instance(self._instance_name).elems[self._element_name]

        instance = assembly.get_instance(self._instance_name)    

        self._pdU_indices = torch.stack([self._element._elems*3, self._element._elems*3+1, self._element._elems*3+2], dim=-1).to(instance.nodes.device).to(torch.int64) + instance._index_start
        self._pdU_values = torch.einsum('i, ge, gea->eai', self.force_density, self._element.gaussian_weight, self._element.shape_function_d0_gaussian).flatten()
        self._instance_RGC_index = instance._RGC_index

    @property
    def force_density(self) -> torch.Tensor:
        """
        Get the body force density vector.
        """
        return self._parameters
    
    @force_density.setter
    def force_density(self, value: list[float] | torch.Tensor) -> None:
        """
        Set the body force density vector.
        """
        if isinstance(value, list):
            self._parameters = torch.tensor(value, dtype=torch.float64)
        else:
            self._parameters = value.to(torch.float64)
    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the body force vector. Body forces don't contribute to stiffness matrix.
        
        Args:
            RGC (list[torch.Tensor]): Current configuration.
            
        Returns:
            tuple: (F_indices, F_values, K_indices, K_values)
                - F_indices: Indices for force vector
                - F_values: Force values distributed from elements to nodes
                - K_indices: Empty tensor (body forces don't affect stiffness)
                - K_values: Empty tensor (body forces don't affect stiffness)
        """
        # Current node positions for volume calculation

        if if_onlyforce:
            return (self._pdU_indices.flatten(), self._pdU_values)

        return (self._pdU_indices.flatten(), 
                self._pdU_values, 
                torch.zeros([2, 0], dtype=torch.int64, device=self._pdU_values.device), 
                torch.zeros([0], device=self._pdU_values.device))

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Get the body force potential energy: U = -∫(f·r)dV
        
        Args:
            RGC (list[torch.Tensor]): Current configuration.
            
        Returns:
            torch.Tensor: Body force potential energy.
        """
        # Current node positions
        U = RGC[self._instance_RGC_index] + self._assembly.get_instance(self._instance_name).nodes
        elems = self._element
        displacement_gaussian = torch.zeros(elems._num_gaussian, elems._elems.shape[0], 3)
        for i in range(elems.num_nodes_per_elem):
            displacement_gaussian = displacement_gaussian + torch.einsum('ge, ei->gei', elems.shape_function_d0_gaussian[:, :, i], U[elems._elems[:, i]])

        potential_energy = torch.einsum('gei, i, ge->', displacement_gaussian, self.force_density, elems.gaussian_weight)

        return potential_energy

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Mark degrees of freedom that are affected by body forces.
        
        Args:
            RGC_remain_index (list[np.ndarray]): Current DOF activation flags.
            
        Returns:
            list[np.ndarray]: Updated DOF activation flags.
        """
        # Body forces affect nodes that belong to elements with volume
        RGC_remain_index[0][self._element._elems.flatten().cpu()] = True
        return RGC_remain_index
