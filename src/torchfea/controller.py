
from unittest.mock import Base
import numpy as np
import torch
from .model import Assembly
from .solver import BaseSolver


class FEAController():

    def __init__(self, maximum_iteration: int = 10000) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """
        
        self.assembly: Assembly = None
        """The assembly containing instances, elements, and reference points."""

        self.solver: BaseSolver = None
        """The solver used for finite element analysis."""

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """
        self.assembly.initialize(*args, **kwargs)
        self.solver.initialize(assembly=self.assembly, *args, **kwargs)

    def solve(self, GC0: torch.Tensor = None, if_initialize: bool = True, *args, **kwargs) -> bool:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        if if_initialize:
            self.initialize()
        result = self.solver.solve(GC0=GC0, *args, **kwargs)
        return result
