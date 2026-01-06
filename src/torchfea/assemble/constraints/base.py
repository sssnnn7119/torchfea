import numpy as np
import torch
from ..obj_base import BaseObj

class BaseConstraint(BaseObj):
    """
    Constraints base class
    """

    def __init__(self) -> None:
        """
        Initialize the Constraints_Base class.
        """
        super().__init__()

    def initialize(self, assembly):
        super().initialize(assembly)


    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor = None, K_values: torch.Tensor = None, if_onlyforce: bool = False, *args, **kwargs):

        R = torch.sparse_coo_tensor(indices=[[]],
                                    values=[],
                                    size=[self._assembly.RGC_list_indexStart[-1]])
        if if_onlyforce:
            return R
        return R, torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])


    def modify_mass_matrix(self, mass_indices: torch.Tensor, mass_values: torch.Tensor, RGC: list[torch.Tensor]):
        return torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])