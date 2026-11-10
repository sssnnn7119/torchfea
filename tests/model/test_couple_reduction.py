"""Numerical and autograd checks for ordinary COO index/value reduction."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torchfea.model.constraints import Couple


def make_couple():
    couple = Couple("instance", "nodes", "rp")
    # The instance starts at 5, deliberately not a multiple of three.
    couple._assembly = SimpleNamespace(_RGC_list_indexStart=[0, 5, 20, 26, 29])
    couple._instance_RGC_index = couple._couple_index = 1
    couple._rp_index = 2
    couple._indexNodes = np.array([0, 2, 4])
    couple._ref_location = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0.2, -0.3, 0.7]])
    node_dofs = (5 + torch.tensor(couple._indexNodes)[:, None] * 3 + torch.arange(3)).flatten()
    active = torch.ones(29, dtype=torch.bool)
    active[node_dofs] = False
    return couple, node_dofs, active


def dense_coo(indices, values):
    # Ordinary scatter_add preserves the graph and sums duplicate COO entries.
    return values.new_zeros(29 * 29).scatter_add(
        0, indices[0] * 29 + indices[1], values,
    ).reshape(29, 29)


def coordinates(q):
    return [q.new_zeros(5), q.new_zeros((5, 3)), q, q.new_zeros(3)]


@pytest.mark.parametrize("pattern", ["mixed", "external", "missing_nodes", "empty"])
def test_coo_reduction_matches_dense_chain_rule(pattern):
    couple, node_dofs, active = make_couple()
    generator = torch.Generator().manual_seed(137)
    indices = torch.randint(29, (2, 120), generator=generator)
    if pattern == "external":
        indices = torch.where(torch.isin(indices, node_dofs), 0, indices)
    elif pattern == "missing_nodes":
        # Only one of the three coupled nodes has stiffness entries.
        indices = torch.where(torch.isin(indices, node_dofs[3:]), 0, indices)
    elif pattern == "empty":
        indices = indices[:, :0]
    values = torch.randn(indices.shape[1], generator=generator)
    residual = torch.randn(29, generator=generator)
    q = torch.tensor([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    rgc = coordinates(q)
    force, out_indices, out_values = couple.modify_R_K(rgc, residual, indices, values)

    first, second = couple._calculate_Ydotz(rgc)
    jacobian = torch.cat((torch.eye(3).expand(3, -1, -1), first), dim=-1).reshape(9, 6)
    transform = torch.eye(29)
    transform[node_dofs] = 0
    transform[node_dofs, 20:26] = jacobian
    expected_force = transform.T @ residual
    expected_tangent = transform.T @ dense_coo(indices, values) @ transform
    expected_tangent[23:26, 23:26] += torch.einsum(
        "na,naij->ij", residual[node_dofs].reshape(3, 3), second,
    )
    actual_tangent = dense_coo(indices, values) + dense_coo(out_indices, out_values)
    assert torch.allclose((residual + force)[active], expected_force[active], atol=1e-12)
    assert torch.allclose(actual_tangent[active][:, active], expected_tangent[active][:, active], atol=1e-12)
    assert torch.allclose(couple.modify_R_K(rgc, residual, if_onlyforce=True), force)
    assert out_indices.dtype == torch.int64


@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_reduction_preserves_first_and_second_order_autograd(angle, monkeypatch):
    couple, _, active = make_couple()
    # Include duplicates, internal entries, and both directions of external
    # couplings; no symmetry assumption is imposed on the stiffness values.
    indices = torch.tensor([[5, 6, 11, 11, 1, 20, 23, 14], [6, 1, 11, 11, 5, 6, 23, 28]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Couple reduction must not construct sparse tensors")

    monkeypatch.setattr(torch, "sparse_coo_tensor", forbidden)

    def evaluate(q, residual, values, offsets):
        couple._ref_location = offsets
        force, out_indices, out_values = couple.modify_R_K(coordinates(q), residual, indices, values)
        matrix = dense_coo(indices, values) + dense_coo(out_indices, out_values)
        return (residual + force)[active], matrix[active][:, active]

    inputs = (
        torch.tensor([0.1, -0.2, 0.3, angle, -angle / 2, angle / 3], requires_grad=True),
        torch.linspace(-1., 1., 29, requires_grad=True),
        torch.linspace(0.1, 0.8, 8, requires_grad=True),
        couple._ref_location.clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(evaluate, inputs, fast_mode=True, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(evaluate, inputs, fast_mode=True, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("reverse", [False, True])
def test_disjoint_couples_on_one_instance_reduce_cross_couplings(reverse):
    first, _, _ = make_couple()
    second = Couple("instance", "other_nodes", "rp2")
    assembly = SimpleNamespace(_RGC_list_indexStart=[0, 5, 20, 26, 32, 35])
    first._assembly = second._assembly = assembly
    second._instance_RGC_index = second._couple_index = 1
    second._rp_index = 3
    second._indexNodes = np.array([1, 3])
    second._ref_location = torch.tensor([[0.5, -0.2, 0.1], [0.3, -0.7, 0.9]])
    q1 = torch.tensor([0., 0., 0., 0.3, -0.4, 0.5])
    q2 = torch.tensor([0., 0., 0., -0.2, 0.5, -0.6])
    rgc = [torch.zeros(5), torch.zeros((5, 3)), q1, q2, torch.zeros(3)]
    generator = torch.Generator().manual_seed(731)
    indices = torch.randint(35, (2, 200), generator=generator)
    values = torch.randn(200, generator=generator)
    residual = torch.randn(35, generator=generator)
    dense = values.new_zeros(35 * 35).scatter_add(0, indices[0] * 35 + indices[1], values).reshape(35, 35)
    transform = torch.eye(35)
    geometric = torch.zeros((35, 35))
    for couple in (first, second):
        node_dofs = (5 + torch.tensor(couple._indexNodes)[:, None] * 3 + torch.arange(3)).flatten()
        rp_start = assembly._RGC_list_indexStart[couple._rp_index]
        jacobian, hessian = couple._calculate_Ydotz(rgc)
        transform[node_dofs] = 0
        transform[node_dofs, rp_start:rp_start + 6] = torch.cat((
            torch.eye(3).expand(len(couple._indexNodes), -1, -1), jacobian,
        ), dim=-1).reshape(-1, 6)
        geometric[rp_start + 3:rp_start + 6, rp_start + 3:rp_start + 6] = torch.einsum(
            "na,naij->ij", residual[node_dofs].reshape(-1, 3), hessian,
        )
    actual_force = residual
    out_indices, out_values = indices, values
    for couple in ((second, first) if reverse else (first, second)):
        force, added_indices, added_values = couple.modify_R_K(rgc, actual_force, out_indices, out_values)
        actual_force = actual_force + force
        out_indices = torch.cat((out_indices, added_indices), dim=1)
        out_values = torch.cat((out_values, added_values))
    actual = out_values.new_zeros(35 * 35).scatter_add(
        0, out_indices[0] * 35 + out_indices[1], out_values,
    ).reshape(35, 35)
    active = torch.ones(35, dtype=torch.bool)
    active[5:20] = False
    expected = transform.T @ dense @ transform + geometric
    assert torch.allclose(actual_force[active], (transform.T @ residual)[active], atol=1e-12)
    assert torch.allclose(actual[active][:, active], expected[active][:, active], atol=1e-12)
