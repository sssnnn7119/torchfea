from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly
    from .baseresult import BaseResult
import torch
from ..interfaces import Serializable

class BaseSolver(Serializable):
    """
    Base class for all solvers in the FEA module.
    """
    _serialized_attributes_exclude = [
        'assembly', '_iter_now', 'K_solver', 'K_sp', 'if_factorized',
        '__low_alpha_count',
    ]
    def __init__(self) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.assembly: Assembly = None
        """ The assembly of the finite element model. """

    def initialize(self, assembly: Assembly, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            assembly (Assembly): The assembly of the finite element model.
        """
        self.assembly = assembly

    def solve(self, GC0: torch.Tensor = None, *args, **kwargs) -> BaseResult:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            BaseResult: The result of the finite element analysis. The result object should include convergence metadata (e.g. `converged`).
        """
        pass
