import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from torchfea.solver import SparseSolver


def test_cg_reuses_device_matrix_and_solves_each_rhs(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CG requested a direct factorization")

    monkeypatch.setattr(SparseSolver, "_factorize", unexpected)
    indices = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 0, 1]], device="cpu")
    values = torch.tensor([2., 2., 1., 1., 3.], device="cpu", dtype=torch.float64)
    solver = SparseSolver.from_coo(indices, values, 2, backend="cg")
    assert solver.matrix.device == values.device
    solver.factorize()
    expected = values.new_tensor([[2., 1., 0.], [3., 0., 0.]])
    rhs = values.new_tensor([[11., 4., 0.], [11., 1., 0.]])
    guess = expected.clone()
    guess[:, 0] = 0.
    # One column needs iterations, another starts at the solution, and the last has zero RHS.
    torch.testing.assert_close(solver.solve(rhs, x0=guess, tol=1e-20, max_iter=2), expected)
    torch.testing.assert_close(solver.solve(rhs[:, 0], tol=1e-20), expected[:, 0])
    assert solver.solve(rhs[:, :0]).shape == (2, 0)


def test_cg_numpy_input_honors_tolerance_and_iteration_limit():
    matrix = csr_matrix([[4., 1.], [1., 3.]])
    solver = SparseSolver(matrix, backend="cg")
    rhs = np.array([1., 2.])
    # The first direction is b and the CG step length is (b.b) / (b.A.b).
    first = rhs * (rhs @ rhs) / (rhs @ (matrix @ rhs))
    np.testing.assert_allclose(solver.solve(rhs, tol=0, max_iter=1), first)
    np.testing.assert_allclose(solver.solve(rhs, tol=0.1), first)
    guess = np.array([1., 1.])
    np.testing.assert_array_equal(solver.solve(rhs, x0=guess, max_iter=0), guess)


def test_cg_accepts_scipy_csr_with_unsorted_duplicate_columns():
    matrix = csr_matrix(([1., 2., 2., 3., 1.], [1, 0, 0, 1, 0], [0, 3, 5]), shape=(2, 2))
    solver = SparseSolver(matrix, backend="cg")
    np.testing.assert_allclose(solver.solve(np.array([6., 7.]), tol=1e-12), [1., 2.])
    np.testing.assert_array_equal(matrix.indices, [1, 0, 0, 1, 0])
