import torch
import numpy as np
from .obj_base import BaseObj

class ReferencePoint(BaseObj):
    """
    ReferencePoints class for handling reference points in a finite element analysis (FEA) framework.
    This class is used to manage the coordinates of reference points, which can be used for various purposes such as boundary conditions, loads, or other constraints in the FEA model.
    """

    def __init__(self, node: list[float] | torch.Tensor = None) -> None:
        """
        Initialize the ReferencePoints class.

        Parameters:
        node (torch.Tensor): A tensor of shape (3) representing the coordinates of the reference point.
        """
        super().__init__()
        if isinstance(node, list):
            node = torch.tensor(node)
        elif isinstance(node, np.ndarray):
            node = torch.tensor(node.tolist())
        self.node = node
        self._RGC_requirements = 6

        