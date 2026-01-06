import numpy as np
import torch
from .base import BaseBoundary

class Boundary_Condition(BaseBoundary):
    """
    Boundary condition (Dirichlet) for instances: fix selected DoFs on a set of nodes.
    """

    def __init__(self,
                 instance_name: str,
                 set_nodes_name: str,
                 indexDoF: list[int] = [0, 1, 2],
                 ) -> None:
        super().__init__()
        self.set_nodes_name = set_nodes_name
        self.instance_name = instance_name
        self.indexDoF = indexDoF
        self._constraint_index: int
        self.index_nodes: np.ndarray

    def initialize(self, assembly):
        super().initialize(assembly)
        self.index_nodes = self._assembly.get_instance(self.instance_name).set_nodes[self.set_nodes_name]
        self._constraint_index = self._assembly.get_instance(self.instance_name)._RGC_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply the boundary condition to the displacement vector
        """
        for i in self.indexDoF:
            RGC[self._constraint_index][self.index_nodes, i] = 0.0
        return RGC

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index by deactivating constrained DoFs
        """
        for i in self.indexDoF:
            RGC_remain_index[self._constraint_index][self.index_nodes, i] = False
        return RGC_remain_index
