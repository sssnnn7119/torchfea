
import numpy as np
import torch
from .base import BaseLoad

class Concentrate_Force(BaseLoad):

    def __init__(self, rp_name: str, force: list[float]) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.rp_index: int = None
        self._parameters = torch.tensor(force, dtype=torch.float64)

    @property
    def force(self) -> torch.Tensor:
        return self._parameters
    
    @force.setter
    def force(self, value: list[float] | torch.Tensor) -> None:
        if isinstance(value, list):
            self._parameters = torch.tensor(value, dtype=torch.float64)
        else:
            self._parameters = value.to(torch.float64)

    def initialize(self, assembly):
        super().initialize(assembly)
        self.rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(assembly.RGC_list_indexStart[self.rp_index], assembly.RGC_list_indexStart[self.rp_index]+3)


    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if if_onlyforce:
            return self._indices_force, self.force
        
        return self._indices_force, self.force, torch.zeros([2, 0], dtype=torch.int), torch.zeros([0])

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.force) == list:
            self.force = torch.tensor(self.force)
        return (self.force * RGC[self.rp_index][:3]).sum()

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self.rp_index][:3] = True
        return RGC_remain_index
