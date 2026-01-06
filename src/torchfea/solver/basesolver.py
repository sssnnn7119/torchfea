from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly

import torch

class BaseSolver:
    """
    Base class for all solvers in the FEA module.
    """
    def __init__(self) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.assembly: Assembly = None
        """ The assembly of the finite element model. """

        
        self.GC: torch.Tensor = None
        """ The generalized coordinates of the finite element model. """

    def initialize(self, assembly: Assembly, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            assembly (Assembly): The assembly of the finite element model.
        """
        self.assembly = assembly

    def solve(self, GC0: torch.Tensor = None, *args, **kwargs) -> bool:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        pass

