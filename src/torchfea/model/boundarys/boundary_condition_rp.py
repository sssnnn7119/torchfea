import numpy as np
import torch
from .base import BaseBoundary

class Boundary_Condition_RP(BaseBoundary):
    """
    Boundary condition for reference points (6 DoFs: UX, UY, UZ, RX, RY, RZ).
    """

    def __init__(self,
                 rp_name: str,
                 indexDoF: list[int] = [0, 1, 2, 3, 4, 5],
                 ) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.indexDoF = indexDoF
        self._constraint_index: int

    def initialize(self, assembly):
        super().initialize(assembly)
        self._constraint_index = self._assembly.get_reference_point(self.rp_name)._RGC_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        for i in self.indexDoF:
            RGC[self._constraint_index][i] = 0.0
        return RGC

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        for i in self.indexDoF:
            RGC_remain_index[self._constraint_index][i] = False
        return RGC_remain_index
