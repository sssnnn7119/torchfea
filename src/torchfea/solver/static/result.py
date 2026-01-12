import numpy as np
import torch
from ..baseresult import BaseResult

class StaticResult(BaseResult):
    """
    The result of a static finite element analysis (FEA) simulation.
    """

    def __init__(self, GC: torch.Tensor, load_params: dict[str, torch.Tensor] = None):
        super().__init__()
        self.GC = GC
        """ 
        Global displacements tensor
        """

        self.load_params: dict[str, torch.Tensor] = load_params
        """
        Load parameters used in the simulation
        """

    def save(self, path: str):
        """
        Save the static FEA result to a file.

        Args:
            path (str): The path to the file where the result will be saved.
        """
        # Implementation for saving static FEA results goes here
        load_params = {}
        if self.load_params is not None:
            for key, value in self.load_params.items():
                load_params[key] = value.cpu().numpy()

        np.savez_compressed(file=path, GC=self.GC.numpy(), load_params=load_params)
    
    @classmethod
    def load(cls, path: str) -> "StaticResult":
        """
        Load the static FEA result from a file.

        Args:
            path (str): The path to the file from which the result will be loaded.
        """
        # Implementation for loading static FEA results goes here
        data = np.load(path, allow_pickle=True)
        GC = torch.tensor(data['GC'])
        load_params_np = data['load_params'].item() if 'load_params' in data else None
        load_params = {}
        if load_params_np is not None:
            for key, value in load_params_np.items():
                load_params[key] = torch.tensor(value)
        return cls(GC=GC, load_params=load_params)