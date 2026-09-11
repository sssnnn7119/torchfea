

import torch


class BaseResult():
    """
    A class to represent the results of a finite element analysis (FEA) simulation.
    """

    def __init__(self):
        """
        Initialize the FEA result.
        """
        self.converged: bool = False
        """Whether the solution converged."""

        self.model_hash: str | None = None
        """Identity of the model used to produce this result."""

        self.GC: torch.Tensor = None
        """The generalized coordinates of the solution."""

    def save(self, path: str):
        """
        Save the FEA result to a file.

        Args:
            path (str): The path to the file where the result will be saved.
        """
        pass

    @classmethod
    def load(cls, path: str) -> "BaseResult":
        """
        Load the FEA result from a file.

        Args:
            path (str): The path to the file from which the result will be loaded.
        """
        pass

    

