from __future__ import annotations

from typing import TYPE_CHECKING

import pypardiso

if TYPE_CHECKING:
    from ... import Assembly
    
from ...model.assembly import WorkCondition

import numpy as np
import torch

from ..baseresult import BaseResult
import scipy.sparse as sp

class StaticResult(BaseResult):
    """
    The result of a static finite element analysis (FEA) simulation.
    """

    def __init__(self, GC: torch.Tensor, work_conditions: WorkCondition,
                 jacobian: dict[str, torch.Tensor] = None, total_time: float = 0.0,
                 time_items: dict[str, list[float]] = None, converged: bool = True,
                 model_hash: str | None = None,
                 compatibility_hash: str | None = None):
        super().__init__()
        self.GC = GC.detach().clone()
        """ 
        Global displacements tensor
        """

        self.converged: bool = converged
        """Indicates whether the static solver converged."""

        self.model_hash: str | None = model_hash
        """Identity of the model used to produce this result."""

        self.compatibility_hash: str | None = compatibility_hash
        """Identity for viewing this result after load values change."""

        if jacobian is None:
            jacobian = {}
        else:
            jacobian = {k: v.detach().clone() for k, v in jacobian.items()}
        self.jacobian: dict[str, torch.Tensor] = jacobian
        """
        The Jacobian matrix (dGC/dLoadParams) of the static FEA result.
        """

        self.work_conditions: WorkCondition = work_conditions
        """
        The work conditions (loads, boundary conditions, etc.) used in the static FEA simulation.
        This is stored to allow for sensitivity analysis and to ensure that the same conditions can be applied when reusing the result.
        """

        self.K_solver: pypardiso.PyPardisoSolver = None
        """
        The Pardiso solver for the stiffness matrix
        """

        self.K_sp: sp.csr_matrix = None
        """
        The stiffness matrix in sparse CSR format
        """

        self.if_factorized: bool = False
        """
        Flag indicating whether the stiffness matrix has been factorized
        """

        self.total_time: float = total_time
        """Total time taken for the simulation"""

        self.time_items: dict[str, list[float]] = time_items if time_items is not None else {}
        """Dictionary to store time taken for different items in the simulation"""

    def __getstate__(self):
        """Exclude non-pickle-able objects (K_solver/K_sp) from serialized state."""
        state = self.__dict__.copy()
        state.pop('K_solver', None)
        state.pop('K_sp', None)
        return state

    def __setstate__(self, state):
        """Restore state and reset K_solver/K_sp to None after deserialization."""
        self.__dict__.update(state)
        self.K_solver = None
        self.K_sp = None

    def factorize_stiffness_matrix(self, assembly: 'Assembly'):
        """
        Factorize the stiffness matrix using the Pardiso solver.

        Args:
            K_idx (torch.Tensor): The indices of the stiffness matrix.
            K_val (torch.Tensor): The values of the stiffness matrix.
        """
        assembly.set_work_conditions(self.work_conditions)
        K_indices, K_values = assembly.assemble_Stiffness_Matrix(GC=self.GC)[1:]
        K_sp = sp.coo_matrix((K_values.detach().cpu().numpy(), (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())))
        self.K_sp = K_sp.tocsr()
        self.K_solver = pypardiso.PyPardisoSolver()
        self.K_solver.factorize(self.K_sp)
        self.if_factorized = True

    def remove_stored_factorization(self):
        """
        Remove the stored factorization of the stiffness matrix to free up memory.
        """
        if self.K_solver is not None:
            self.K_solver.free_memory(everything=True)
            self.K_solver = None
            self.K_sp = None
            self.if_factorized = False
            import gc
            gc.collect()

    def save(self, path: str):
        """
        Save the static FEA result to a file.

        Args:
            path (str): The path to the file where the result will be saved.
        """
        if not self.model_hash:
            raise ValueError("Cannot save a result without a model hash.")
        serialized_work_conditions = np.empty((), dtype=object)
        serialized_work_conditions[()] = self.work_conditions._serialize()

        archive = dict(
            GC=self.GC.cpu().numpy(),
            work_conditions=serialized_work_conditions,
            total_time=self.total_time,
            time_items=self.time_items,
            converged=self.converged,
            model_hash=self.model_hash,
        )
        if self.compatibility_hash:
            archive["compatibility_hash"] = self.compatibility_hash
        np.savez_compressed(file=path, **archive)
    
    @classmethod
    def load(cls, path: str) -> "StaticResult":
        """
        Load the static FEA result from a file.

        Args:
            path (str): The path to the file from which the result will be loaded.
        """
        with np.load(path, allow_pickle=True) as data:
            GC = torch.tensor(data['GC'])
            total_time = float(data['total_time'])
            time_items = data['time_items'].item()
            converged = bool(np.asarray(data['converged']).reshape(-1)[0])
            model_hash = str(np.asarray(data['model_hash']).reshape(-1)[0])
            compatibility_hash = (
                str(np.asarray(data['compatibility_hash']).reshape(-1)[0])
                if 'compatibility_hash' in data.files else None
            )
            serialized_work_conditions = data['work_conditions'].item()

        work_conditions = (
            WorkCondition._deserialize(serialized_work_conditions)
            if serialized_work_conditions is not None else None
        )

        return cls(GC=GC, work_conditions=work_conditions, total_time=total_time,
                   time_items=time_items, converged=converged, model_hash=model_hash,
                   compatibility_hash=compatibility_hash)
