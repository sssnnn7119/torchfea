from unittest.mock import Mock

import numpy as np
import pytest
import scipy.sparse.linalg as sparse_linalg
import torch
from scipy.sparse import csr_matrix

from torchfea.solver import SparseSolver, sparse_solver


@pytest.mark.parametrize("failure", [None, "unavailable", "init", "factorize", "solve", "nonfinite"])
def test_direct_solver_prefers_pardiso_and_reuses_fallback(monkeypatch, failure):
    matrix = csr_matrix([[4., 2.], [1., 3.]])
    rhs = np.array([[4., 6.], [1., 4.]])
    expected = np.array([[1., 1.], [0., 1.]])
    pardiso = Mock()
    factory = Mock(return_value=pardiso)
    if failure == "init":
        factory.side_effect = OSError("missing MKL")
    elif failure == "factorize":
        pardiso.factorize.side_effect = RuntimeError("factorization failed")
    elif failure == "solve":
        pardiso.solve.side_effect = RuntimeError("solve failed")
    elif failure == "nonfinite":
        pardiso.solve.return_value = np.full_like(rhs, np.nan)
    else:
        pardiso.solve.side_effect = lambda a, b: np.linalg.solve(a.toarray(), b)

    load = Mock(return_value=None if failure == "unavailable" else factory)
    monkeypatch.setattr(sparse_solver, "_pardiso_solver_class", load)
    superlu = Mock(wraps=sparse_linalg.splu)
    monkeypatch.setattr(sparse_linalg, "splu", superlu)

    factors = SparseSolver(matrix)
    assert factors.solve(np.empty((2, 0))).shape == (2, 0)
    load.assert_not_called()
    superlu.assert_not_called()
    np.testing.assert_allclose(factors.solve(rhs), expected)
    np.testing.assert_allclose(factors.solve(rhs[:, 0]), expected[:, 0])
    if failure is None:
        pardiso.factorize.assert_called_once()
        assert pardiso.solve.call_count == 2
        superlu.assert_not_called()
    else:
        superlu.assert_called_once()
        if failure in ("factorize", "solve", "nonfinite"):
            pardiso.free_memory.assert_called_once_with(everything=True)
        assert pardiso.solve.call_count == int(failure in ("solve", "nonfinite"))


def test_scipy_backend_solves_torch_coo_without_loading_pardiso(monkeypatch):
    load = Mock(side_effect=AssertionError("unexpected PARDISO import"))
    monkeypatch.setattr(sparse_solver, "_pardiso_solver_class", load)
    solver = SparseSolver.from_coo(
        torch.tensor([[0, 1, 0], [0, 1, 0]]),
        torch.tensor([1., 4., 2.]), 2, backend="scipy",
    )
    solver.factorize()
    rhs = torch.tensor([6., 4.], dtype=torch.float32)
    torch.testing.assert_close(solver.solve(rhs), rhs.new_tensor([2., 1.]))
    load.assert_not_called()


def test_explicit_pardiso_propagates_backend_failure(monkeypatch):
    native = Mock()
    native.solve.side_effect = RuntimeError("native solve failed")
    monkeypatch.setattr(sparse_solver, "_pardiso_solver_class", lambda: lambda: native)
    solver = SparseSolver(csr_matrix([[1.]]), backend="pardiso")
    with pytest.raises(RuntimeError, match="native solve failed"):
        solver.solve(np.ones(1))
    native.free_memory.assert_called_once_with(everything=True)
