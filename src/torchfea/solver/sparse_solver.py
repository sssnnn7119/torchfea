"""Sparse direct and conjugate-gradient solves for finite-element matrices."""
from __future__ import annotations

import logging
from functools import cache
from threading import Lock
from typing import Literal, Self, TypeVar
from weakref import finalize

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)
Backend = Literal["auto", "pardiso", "scipy", "cg"]
_RHS = TypeVar("_RHS", torch.Tensor, np.ndarray)


@cache
def _pardiso_solver_class():
    try:
        from pypardiso import PyPardisoSolver
    except Exception as error:
        logger.debug("PARDISO unavailable: %s", error)
        return None
    return PyPardisoSolver


class SparseSolver:
    """Solve one fixed square sparse matrix for one or more right-hand sides.

    Direct backends factorize on demand. Auto mode prefers PARDISO for float64
    and falls back to SciPy SuperLU. The cg backend iterates on a PyTorch CSR
    matrix and is intended for symmetric positive-definite systems.
    """

    def __init__(self, matrix: sp.spmatrix | torch.Tensor, *, backend: Backend = "auto"):
        if backend not in ("auto", "pardiso", "scipy", "cg"):
            raise ValueError(f"Unknown sparse solver backend: {backend!r}")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("SparseSolver requires a square matrix")
        if isinstance(matrix, torch.Tensor):
            if matrix.layout not in (torch.sparse_coo, torch.sparse_csr):
                raise ValueError("Expected a sparse COO or CSR tensor")
            matrix = matrix.detach().to_sparse_csr()
            if backend != "cg":
                csr = matrix.cpu()
                matrix = sp.csr_matrix((csr.values().numpy(), csr.col_indices().numpy(),
                                        csr.crow_indices().numpy()), shape=csr.shape)
        else:
            matrix = matrix.tocsr()
            if backend == "cg":
                # PyTorch CSR requires sorted, distinct columns in each row.
                if not matrix.has_canonical_format:
                    matrix = matrix.copy()
                    matrix.sum_duplicates()
                matrix = torch.sparse_csr_tensor(
                    matrix.indptr, matrix.indices, matrix.data,
                    size=matrix.shape, device="cpu", check_invariants=True,
                )
        self.matrix = matrix
        self._backend = backend
        self._lock = Lock()
        self._pardiso = None
        self._release = None
        self._superlu = None

    @classmethod
    def from_coo(cls, indices: torch.Tensor, values: torch.Tensor, size: int,
                 *, backend: Backend = "auto") -> Self:
        """Assemble a square matrix from PyTorch COO entries, summing duplicates."""
        if backend == "cg":
            matrix = torch.sparse_coo_tensor(
                indices, values.detach(), (size, size), device=values.device, check_invariants=True,
            )
        else:
            rows, cols = indices.detach().cpu().numpy()
            matrix = sp.csr_matrix((values.detach().cpu().numpy(), (rows, cols)), shape=(size, size))
        return cls(matrix, backend=backend)

    def factorize(self) -> None:
        """Prepare direct factors. CG uses its assembled CSR matrix as-is."""
        if self._backend != "cg":
            with self._lock:
                self._factorize()

    def _factorize(self):
        if self._pardiso is not None or self._superlu is not None:
            return
        if self._backend != "scipy":
            factory = _pardiso_solver_class() if self.matrix.dtype == np.float64 else None
            if factory is not None:
                try:
                    self._pardiso = factory()
                    self._release = finalize(self, self._pardiso.free_memory, everything=True)
                    self._pardiso.factorize(self.matrix)
                    return
                except Exception as error:
                    self._handle_pardiso_failure(error)
            elif self._backend == "pardiso":
                raise RuntimeError("PARDISO requires a working PyPardiso runtime and a float64 matrix")
        from scipy.sparse.linalg import splu
        self._superlu = splu(self.matrix.tocsc())

    def _handle_pardiso_failure(self, error: Exception):
        if self._release is not None:
            try:
                self._release()
            except Exception as cleanup_error:
                logger.warning("PARDISO cleanup failed: %s", cleanup_error)
        self._pardiso = None
        if self._backend == "pardiso":
            raise error
        logger.warning("PARDISO failed, retrying this matrix with SciPy SuperLU: %s", error)
        self._backend = "scipy"

    def solve(self, rhs: _RHS, *, x0: torch.Tensor | np.ndarray | None = None,
              tol: float = 1e-3, max_iter: int = 1500) -> _RHS:
        """Solve a vector or columns of RHS. CG stops at squared relative residual <= tol."""
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.matrix.shape[0]:
            raise ValueError("Right-hand side dimensions do not match the matrix")
        if self._backend == "cg":
            return self._solve_cg(rhs, x0, tol, max_iter)
        array = rhs.detach().cpu().numpy() if isinstance(rhs, torch.Tensor) else rhs
        if array.size:
            with self._lock:
                solution = self._solve_array(array)
        else:
            solution = np.empty(array.shape, dtype=self.matrix.dtype)
        if isinstance(rhs, torch.Tensor):
            result = torch.from_numpy(solution).to(rhs.device)
            if result.is_floating_point() and rhs.is_floating_point():
                return result.to(rhs.dtype)
            return result
        return solution

    @torch.no_grad()
    def _solve_cg(self, rhs, x0, tol, max_iter):
        if x0 is not None and x0.shape != rhs.shape:
            raise ValueError("CG initial guess must have the same shape as the right-hand side")
        if max_iter < 0 or tol < 0:
            raise ValueError("CG requires nonnegative max_iter and tol")
        device = rhs.device if isinstance(rhs, torch.Tensor) else self.matrix.device
        b = torch.as_tensor(rhs, device=device)
        dtype = torch.promote_types(self.matrix.dtype, b.dtype)
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.get_default_dtype()
        matrix = self.matrix.to(device=device, dtype=dtype)
        b = b.to(dtype)
        x = torch.zeros_like(b) if x0 is None else torch.as_tensor(x0, device=device, dtype=dtype).clone()
        if b.ndim == 1:
            x = self._cg_vector(matrix, b, x, tol, max_iter)
        else:
            for col in range(b.shape[1]):
                x[:, col] = self._cg_vector(matrix, b[:, col], x[:, col], tol, max_iter)
        if isinstance(rhs, torch.Tensor):
            return x.to(rhs.dtype) if rhs.is_floating_point() and x.is_floating_point() else x
        return x.cpu().numpy()

    @staticmethod
    def _cg_vector(matrix: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
                   tol: float, max_iter: int) -> torch.Tensor:
        norm_b_sq = torch.vdot(b, b).real
        if norm_b_sq == 0:
            return torch.zeros_like(b)
        # CSR times a dense column also has a native CPU kernel on builds without MKL.
        r = b - torch.mm(matrix, x.unsqueeze(1)).squeeze(1)
        p = r
        rsold = torch.vdot(r, r).real
        threshold = tol * norm_b_sq
        if rsold <= threshold:
            return x
        for _ in range(max_iter):
            Ap = torch.mm(matrix, p.unsqueeze(1)).squeeze(1)
            alpha = rsold / torch.vdot(p, Ap).real
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.vdot(r, r).real
            if rsnew <= threshold:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def _solve_array(self, rhs: np.ndarray) -> np.ndarray:
        self._factorize()
        if self._pardiso is not None:
            try:
                result = self._pardiso.solve(self.matrix, rhs)
                if not np.isfinite(result).all():
                    raise ArithmeticError("non-finite solution")
                return result
            except Exception as error:
                self._handle_pardiso_failure(error)
                self._factorize()
        return self._superlu.solve(rhs)
