import numpy as np
import torch
from numpy.typing import NDArray

from ...utils import Rotation
from .base import BaseConstraint


class Couple(BaseConstraint):
    _serialized_attributes = ["instance_name", "set_nodes_name", "rp_name"]
    _serialized_attributes_exclude = ["_node_lookup"]

    def __init__(self, instance_name: str, set_nodes_name: str, rp_name: str) -> None:
        super().__init__()
        self.instance_name = instance_name
        self.set_nodes_name = set_nodes_name
        self.rp_name = rp_name
        self._node_lookup: torch.Tensor | None = None

        self._ref_location: torch.Tensor

        self._couple_index: int
        self._rp_index: int
        self._instance_RGC_index: int
        self._indexNodes: np.ndarray

    def initialize(self, assembly):
        super().initialize(assembly)
        self._node_lookup = None
        self._indexNodes = np.sort(
            self._assembly.get_instance(self.instance_name).set_nodes[
                self.set_nodes_name
            ]
        )
        self._rp_index = self._assembly.get_reference_point(self.rp_name)._RGC_index
        self._couple_index = self._assembly.get_instance(self.instance_name)._RGC_index

        instance = self._assembly.get_instance(self.instance_name)
        self._instance_RGC_index = instance._RGC_index

        index_global = instance.nodes[self._indexNodes]
        self._ref_location = (
            index_global - self._assembly.get_reference_point(self.rp_name).node
        )

    def get_input_dofs(self) -> NDArray[np.int64]:
        """The six reference-point coordinates drive this rigid coupling."""
        start = self._assembly._RGC_list_indexStart[self._rp_index]
        return np.arange(start, start + 6, dtype=np.int64)

    def get_output_dofs(self) -> NDArray[np.int64]:
        """Identify only the coupled nodes for dependency/ownership checks."""
        start = self._assembly._RGC_list_indexStart[self._instance_RGC_index]
        return (
            start + self._indexNodes[:, None] * 3 + np.arange(3, dtype=np.int64)
        ).reshape(-1)

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the couple constraint to the displacement vector
        """
        RGC[self._couple_index] = RGC[self._couple_index].clone()
        RGC[self._couple_index][self._indexNodes] = (
            RGC[self._rp_index][:3]
            + Rotation.from_rotvec(RGC[self._rp_index][3:]).apply(self._ref_location)
            - self._ref_location
        )

        return RGC

    def modify_mass_matrix(self, mass_indices, mass_values, RGC: list[torch.Tensor]):
        v = RGC[self._rp_index][:3]
        z = RGC[self._rp_index][3:]

        y = v + Rotation.from_rotvec(z).apply(self._ref_location) - self._ref_location

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._couple_index][self._indexNodes] = False
        RGC_remain_index[self._rp_index][:] = True
        return RGC_remain_index

    def modify_R_K(
        self,
        RGC: list[torch.Tensor],
        R0: torch.Tensor,
        K_indices: torch.Tensor | None = None,
        K_values: torch.Tensor | None = None,
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return additive residual and COO tangent contributions for elimination.

        With nodal Jacobian J and nodal Hessian H, the RP block is
        J.T @ K_nodes @ J + sum(R_nodes * H). Only cross terms with DOFs
        outside this node set survive elimination and need to be emitted.
        Index/value tensors remain ordinary dense tensors, preserving the
        graph through RGC, R0 and K_values without sparse tensor operations.
        """
        if not if_onlyforce and (K_indices is None or K_values is None):
            raise ValueError("K_indices and K_values must be provided when if_onlyforce is False")

        starts = self._assembly._RGC_list_indexStart
        node_start = int(starts[self._instance_RGC_index])
        rp_start = int(starts[self._rp_index])
        nodes = torch.as_tensor(self._indexNodes, device=R0.device, dtype=torch.int64)
        node_dofs = node_start + nodes[:, None] * 3 + torch.arange(3, device=R0.device)
        residual = torch.zeros_like(R0)
        if nodes.numel() == 0:
            if if_onlyforce:
                return residual
            return residual, K_indices.new_empty((2, 0)), K_values.new_empty(0)

        first, second = self._calculate_Ydotz(RGC)
        nodal_force = R0[node_dofs]
        residual[rp_start:rp_start + 6] = torch.cat((
            nodal_force.sum(dim=0),
            torch.einsum("na,nai->i", nodal_force, first),
        ))
        if if_onlyforce:
            return residual

        # Cache only the static instance-node -> coupled-node row mapping.
        # Its size is the instance node count, not the global DoF count.
        node_count = (int(starts[self._instance_RGC_index + 1]) - node_start) // 3
        lookup = self._node_lookup
        if lookup is None or lookup.device != R0.device or lookup.numel() != node_count:
            lookup = torch.full((node_count,), -1, device=R0.device, dtype=torch.int64)
            lookup[nodes] = torch.arange(nodes.numel(), device=R0.device)
            self._node_lookup = lookup
        local_dofs = K_indices - node_start
        local_nodes = torch.div(local_dofs, 3, rounding_mode="floor")
        positions = lookup[local_nodes.clamp(0, node_count - 1)]
        coupled = (local_nodes >= 0) & (local_nodes < node_count) & (positions >= 0)
        components = local_dofs.remainder(3)
        row_coupled, col_coupled = coupled.unbind(0)
        boundary = row_coupled ^ col_coupled
        internal = row_coupled & col_coupled

        # Both orientations of a cross term use the same four coefficients:
        # one matching translation and three rotational derivatives.
        map_row = row_coupled[boundary]
        node_row = torch.where(map_row, positions[0, boundary], positions[1, boundary])
        component = torch.where(map_row, components[0, boundary], components[1, boundary])
        weights = torch.cat((
            K_values.new_ones((component.numel(), 1)), first[node_row, component],
        ), dim=1)
        rp_targets = rp_start + torch.cat((
            component[:, None],
            torch.arange(3, 6, device=K_indices.device).expand(component.numel(), -1),
        ), dim=1)
        outside = torch.where(map_row, K_indices[1, boundary], K_indices[0, boundary])[:, None]
        cross_indices = torch.stack((
            torch.where(map_row[:, None], rp_targets, outside).reshape(-1),
            torch.where(map_row[:, None], outside, rp_targets).reshape(-1),
        ))
        cross_values = (K_values[boundary, None] * weights).reshape(-1)

        jacobian = torch.cat((
            torch.eye(3, device=first.device, dtype=first.dtype).expand(nodes.numel(), -1, -1), first,
        ), dim=-1)
        row_jacobian = jacobian[positions[0, internal], components[0, internal]]
        col_jacobian = jacobian[positions[1, internal], components[1, internal]]
        block = row_jacobian.T @ (K_values[internal, None] * col_jacobian)
        geometric = torch.einsum("na,naij->ij", nodal_force, second)
        block = block + torch.nn.functional.pad(geometric, (3, 0, 3, 0))
        rp_dofs = rp_start + torch.arange(6, device=K_indices.device)
        block_indices = torch.stack((rp_dofs.repeat_interleave(6), rp_dofs.repeat(6)))
        # Keep zero-valued coefficients: they can have nonzero derivatives.
        return (
            residual,
            torch.cat((cross_indices, block_indices), dim=1),
            torch.cat((cross_values, block.reshape(-1))),
        )

    def _calculate_Ydotz(
        self, RGC: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First and second derivatives of rotated reference offsets."""
        z = RGC[self._rp_index][3:]
        _, first, second = Rotation.matrix_derivatives(z)
        offsets = self._ref_location.to(z)
        return (
            torch.einsum("abi,nb->nai", first, offsets).contiguous(),
            torch.einsum("abij,nb->naij", second, offsets),
        )
