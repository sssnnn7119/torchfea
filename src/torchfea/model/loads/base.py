from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import Assembly

import numpy as np
import torch
from .. import elements
from ..obj_base import BaseObj





class BaseLoad(BaseObj):

    def __init__(self) -> None:
        super().__init__()
        self._indices_matrix: torch.Tensor = torch.zeros([2, 0],
                                                        dtype=torch.int)
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = torch.zeros([0],
                                                            dtype=torch.int)
        """
            the start index of the stiffness matricx of structural stress
        """

        
        self._parameters: torch.Tensor = torch.zeros(0, dtype=torch.float64)
        """The parameters of this object.
        This is a 1D tensor containing all the parameters of this object.
        """

    def initialize(self, assembly: Assembly):
        super().initialize(assembly)
    
    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the stiffness matrix and force vector for the self-contact load.

        Args:
            RGC (list[torch.Tensor]): The global coordinates of the nodes.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The stiffness matrix and force vector for the self-contact load.
                - F_indices: The indices of the force vector.
                - F_values: The values of the force vector.
                - K_indices: The indices of the stiffness matrix.
                - K_values: The values of the stiffness matrix.
        """

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """Get the potential energy for the self-contact load."""
        raise NotImplementedError("get_potential_energy method not implemented")
    
    @staticmethod
    def get_F0():
        raise NotImplementedError("get_F0 method not implemented")
