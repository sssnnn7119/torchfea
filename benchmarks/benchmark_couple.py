"""Compare the original Couple reduction with direct COO-value reduction.

Run: PYTHONPATH=src python benchmarks/benchmark_couple.py
Both implementations receive inputs requiring gradients; no sparse tensor
operations are used. The original implementation is retained only as a baseline.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.benchmark import Timer

from torchfea.model.constraints import Couple


def _make_case(total_dofs: int, node_count: int, neighbors: int = 12):
    generator = torch.Generator().manual_seed(20261110)
    node_dofs = node_count * 3
    outside_count = 24
    couple = Couple("instance", "nodes", "rp")
    couple._assembly = SimpleNamespace(
        _RGC_list_indexStart=[0, node_dofs, node_dofs + 6, total_dofs],
    )
    couple._instance_RGC_index = couple._couple_index = 0
    couple._rp_index = 1
    couple._indexNodes = np.arange(node_count)
    couple._ref_location = torch.randn((node_count, 3), generator=generator)
    rgc = [
        torch.zeros((node_count, 3)),
        torch.tensor([0.1, -0.2, 0.3, 0.2, -0.3, 0.4], requires_grad=True),
        torch.empty(0),
    ]
    residual = torch.randn(total_dofs, generator=generator, requires_grad=True)
    count = node_dofs * neighbors // 2
    rows = torch.randint(node_dofs, (count,), generator=generator)
    cols = torch.randint(node_dofs, (count,), generator=generator)
    boundary = torch.rand(count, generator=generator) < 0.1
    cols[boundary] = total_dofs - outside_count + torch.randint(
        outside_count, (int(boundary.sum()),), generator=generator,
    )
    # Symmetric COO entries, shuffled with duplicates; every node has a diagonal
    # so the legacy consecutive-node assumption is valid for this comparison.
    edges = torch.stack((rows, cols))
    diagonal = torch.arange(node_dofs)
    indices = torch.cat((edges, edges.flip(0), torch.stack((diagonal, diagonal))), dim=1)
    edge_values = torch.randn(count, generator=generator)
    values = torch.cat((edge_values, edge_values, torch.ones(node_dofs))).requires_grad_()
    active = torch.cat((
        torch.arange(node_dofs, node_dofs + 6),
        torch.arange(total_dofs - outside_count, total_dofs),
    ))
    return couple, (rgc, residual, indices, values), active


def _active_result(result, active):
    residual, indices, values = result
    keep = (indices[0] >= active[0]) & (indices[1] >= active[0])
    local = torch.searchsorted(active, indices[:, keep].contiguous())
    width = active.numel()
    matrix = values.new_zeros(width * width).scatter_add(
        0, local[0] * width + local[1], values[keep],
    ).reshape(width, width)
    return residual[active], matrix


def _forward_backward(function, couple, args, active):
    # The legacy implementation mirrors K's cross terms, so its derivative
    # is valid only for symmetric stiffness variations. Tie each mirrored
    # pair to the same parameter for the value/gradient/timing comparison.
    edge_count = (args[3].numel() - len(couple._indexNodes) * 3) // 2
    values = torch.cat((args[3][:edge_count], args[3][:edge_count], args[3][2 * edge_count:]))
    force, matrix = _active_result(function(couple, *args[:3], values), active)
    loss = force.square().sum() + matrix.square().mean()
    return torch.autograd.grad(loss, (args[0][1], args[1], args[3]))


def benchmark_couple(min_run_time: float = 0.2) -> list[dict]:
    """Check values/gradients, then time forward and backward on one CPU thread."""
    cases = [(300, 30), (30_000, 1_000), (300_000, 10_000), (3_000_000, 1_000)]
    results = []
    for total_dofs, node_count in cases:
        couple, args, active = _make_case(total_dofs, node_count)
        with torch.no_grad():
            legacy = _legacy_modify_R_K(couple, *args)
            current = couple.modify_R_K(*args)
            expected_force, expected_matrix = _active_result(legacy, active)
            actual_force, actual_matrix = _active_result(current, active)
            torch.testing.assert_close(actual_force, expected_force, atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(actual_matrix, expected_matrix, atol=1e-8, rtol=1e-9)
            old_count, new_count = legacy[2].numel(), current[2].numel()
            del legacy, current

        old = Timer(
            stmt="legacy(couple, *args)",
            globals={"legacy": _legacy_modify_R_K, "couple": couple, "args": args},
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        new = Timer(
            stmt="couple.modify_R_K(*args)", globals={"couple": couple, "args": args},
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        # Compare the same reduced objective, not the legacy's redundant entries.
        expected_gradients = _forward_backward(_legacy_modify_R_K, couple, args, active)
        actual_gradients = _forward_backward(Couple.modify_R_K, couple, args, active)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-8)
        del expected_gradients, actual_gradients
        old_backward = Timer(
            stmt="backward(function, couple, args, active)",
            globals={"backward": _forward_backward, "function": _legacy_modify_R_K,
                     "couple": couple, "args": args, "active": active},
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        new_backward = Timer(
            stmt="backward(function, couple, args, active)",
            globals={"backward": _forward_backward, "function": Couple.modify_R_K,
                     "couple": couple, "args": args, "active": active},
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        results.append({
            "total_dofs": total_dofs, "coupled_nodes": node_count,
            "input_nnz": args[2].shape[1], "old_ms": old.median * 1000,
            "new_ms": new.median * 1000, "speedup": old.median / new.median,
            "old_output_nnz": old_count, "new_output_nnz": new_count,
            "old_forward_backward_ms": old_backward.median * 1000,
            "new_forward_backward_ms": new_backward.median * 1000,
            "removed_dense_mib": 6 * total_dofs * args[3].element_size() / 2**20,
        })
    return results


def _legacy_modify_R_K(
    self,
    RGC: list[torch.Tensor],
    R0: torch.Tensor,
    K_indices: torch.Tensor = None,
    K_values: torch.Tensor = None,
    if_onlyforce: bool = False,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Modify the R and K

    Args:
        indexStart (list[int]): The starting indices for each node.
        U (list[torch.Tensor]): The displacement vector for each node.
        R (torch.Tensor): The global force vector.
        K (torch.Tensor): The global stiffness matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The modified R and K tensors.
    """

    if not if_onlyforce and (K_indices is None or K_values is None):
        raise ValueError(
            "K_indices and K_values must be provided when if_onlyforce is False"
        )

    R_now = R0[
        self._assembly._RGC_list_indexStart[
            self._instance_RGC_index
        ] : self._assembly._RGC_list_indexStart[self._instance_RGC_index + 1]
    ].view(-1, 3)
    Ydot, Ydot2 = self._calculate_Ydotz(RGC)

    # R
    # region
    Rrest = R_now[self._indexNodes]

    Edotv = Rrest.sum(dim=0)
    Edotz = torch.einsum("bj,bjp->p", Rrest, Ydot)

    R = torch.zeros(self._assembly._RGC_list_indexStart[-1])
    start_idx = self._assembly._RGC_list_indexStart[self._rp_index]
    R[start_idx : start_idx + 3] += Edotv
    R[start_idx + 3 : start_idx + 6] += Edotz
    # endregion

    if if_onlyforce:
        return R

    # K
    # region
    ## first, get the K of the rest part in index1

    # initial select the instance indices

    indice_max = self._assembly._RGC_list_indexStart[-1]
    indice_start = self._assembly._RGC_list_indexStart[self._couple_index]

    index = torch.where(
        torch.isin(
            ((K_indices[1] - indice_start) // 3),
            torch.tensor(self._indexNodes.tolist()),
        )
    )

    sort_index = torch.argsort((K_indices[1][index] - indice_start) // 3)

    index = index[0][sort_index]
    indice1 = K_indices[0][index]
    indice30 = K_indices[1][index] // 3
    indice3 = torch.unique_consecutive(indice30, return_inverse=True)[1]
    indice4 = K_indices[1][index] % 3

    Rdotv_indices = torch.stack([indice1, indice4], dim=0)
    Rdotv_indices_flatten = Rdotv_indices[0] * 3 + Rdotv_indices[1]
    Rdotv_values = K_values[index]
    Rdotv = (
        torch.zeros([indice_max * 3])
        .scatter_add_(0, Rdotv_indices_flatten, Rdotv_values)
        .reshape(indice_max, 3)
    )

    Rdotz_indices = torch.stack(
        [
            indice1.reshape(-1, 1).repeat(1, 3),
            torch.tensor([0, 1, 2]).reshape([1, 3]).repeat(indice1.shape[0], 1),
        ],
        dim=0,
    ).reshape([2, -1])
    Rdotz_indices_flatten = Rdotz_indices[0] * 3 + Rdotz_indices[1]
    Rdotz_values = (
        K_values[index].unsqueeze(-1) * Ydot.view(-1, 3)[indice4 + indice3 * 3]
    ).flatten()
    Rdotz = (
        torch.zeros([indice_max * 3])
        .scatter_add_(0, Rdotz_indices_flatten, Rdotz_values)
        .reshape(indice_max, 3)
    )

    index_remain_dim0 = np.vstack(
        [
            indice_start + self._indexNodes * 3,
            indice_start + self._indexNodes * 3 + 1,
            indice_start + self._indexNodes * 3 + 2,
        ]
    ).T

    Edotvv = Rdotv[index_remain_dim0].sum(dim=0)
    Edotzv = torch.einsum("biq,bip->pq", Rdotv[index_remain_dim0], Ydot)

    Edotzz = torch.einsum(
        "biq,bip->pq", Rdotz[index_remain_dim0], Ydot
    ) + torch.einsum("ai,aipq->pq", Rrest, Ydot2)
    # combine the indices and values
    indices = []
    values = []

    ## for Rv
    indice_Rv = Rdotv_indices
    index1 = indice_Rv[0]
    index2 = self._assembly._RGC_list_indexStart[self._rp_index] + indice_Rv[1]
    indices.append(torch.stack([index1, index2], dim=0))
    values.append(Rdotv_values)
    indices.append(torch.stack([index2, index1], dim=0))
    values.append(Rdotv_values)
    ## for Rz
    indice_Rz = Rdotz_indices
    index1 = indice_Rz[0]
    index2 = self._assembly._RGC_list_indexStart[self._rp_index] + indice_Rz[1] + 3
    indices.append(torch.stack([index1, index2], dim=0))
    values.append(Rdotz_values)
    indices.append(torch.stack([index2, index1], dim=0))
    values.append(Rdotz_values)
    ## for Edot2
    mat66 = torch.zeros([6, 6])
    mat66[:3, :3] = Edotvv
    mat66[3:, 3:] = Edotzz
    mat66[3:, :3] = Edotzv
    mat66[:3, 3:] = Edotzv.transpose(0, 1)

    indice_Edot2 = [
        torch.tensor([0, 1, 2, 3, 4, 5]).reshape(-1, 1).repeat(1, 6).flatten(),
        torch.tensor([0, 1, 2, 3, 4, 5]).reshape(1, -1).repeat(6, 1).flatten(),
    ]
    index1 = indice_Edot2[0] + self._assembly._RGC_list_indexStart[self._rp_index]
    index2 = indice_Edot2[1] + self._assembly._RGC_list_indexStart[self._rp_index]
    indices.append(torch.stack([index1, index2], dim=0))
    values.append(mat66.flatten())

    # combine the indices and values
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values, dim=0)
    # endregion

    return R, indices, values


if __name__ == "__main__":
    print("CPU, float64, 1 thread, gradients enabled; median forward time")
    print("DoFs       Nodes     Input nnz    Old ms    New ms    Speedup   Old/new output nnz   Removed dense MiB")
    for result in benchmark_couple():
        print(
            f"{result['total_dofs']:9,d} {result['coupled_nodes']:7,d} "
            f"{result['input_nnz']:11,d} {result['old_ms']:9.3f} "
            f"{result['new_ms']:9.3f} {result['speedup']:8.2f}x "
            f"{result['old_output_nnz']:10,d}/{result['new_output_nnz']:<10,d} "
            f"{result['removed_dense_mib']:9.2f}"
        )
        print(
            f"          Forward + backward: {result['old_forward_backward_ms']:.3f} -> "
            f"{result['new_forward_backward_ms']:.3f} ms "
            f"({result['old_forward_backward_ms'] / result['new_forward_backward_ms']:.2f}x)"
        )
