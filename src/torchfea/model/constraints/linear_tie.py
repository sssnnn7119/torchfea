from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import torch
from numpy.typing import NDArray

from .base import BaseConstraint

if TYPE_CHECKING:
    from ..assembly import Assembly
    from ..elements.dimension3.surfaces import BaseSurface


class LinearTie(BaseConstraint):
    """Eliminate secondary translations using a fixed sparse interpolation.

    Subclasses build node-level COO entries, rather than a dense C matrix.
    Each entry contains a secondary-row index, a main-node index and a
    scalar weight. All three translation components share these weights.
    The displacement Jacobian is constant and its second derivative is zero.
    """

    _serialized_attributes: list[str] = [
        "main_instance_name", "main_surface_name",
        "secondary_instance_name", "secondary_surface_name",
        "position_tolerance", "allow_unmatched", "enabled",
    ]

    def __init__(
        self,
        main_instance_name: str,
        main_surface_name: str,
        secondary_instance_name: str,
        secondary_surface_name: str,
        position_tolerance: float,
        allow_unmatched: bool = False,
    ) -> None:
        super().__init__()
        # Persist only the model definition; all interpolation and DoF maps
        # below are rebuilt from the current reference mesh at initialize().
        self.main_instance_name: str = main_instance_name
        self.main_surface_name: str = main_surface_name
        self.secondary_instance_name: str = secondary_instance_name
        self.secondary_surface_name: str = secondary_surface_name
        self.position_tolerance: float = float(position_tolerance)
        self.allow_unmatched: bool = allow_unmatched

        self._main_index: int
        self._secondary_index: int
        # Node-level COO: each weight links one secondary row to a main node.
        # Rows can have different lengths; all three translations share them.
        self._secondary_nodes: torch.Tensor
        self._mapping_rows: torch.Tensor
        self._main_nodes: torch.Tensor
        self._weights: torch.Tensor
        self._unmatched_nodes: torch.Tensor
        self._projection_distances: torch.Tensor

        # Scalar-DoF CSR map used to expand only affected sparse matrix entries.
        self._secondary_dofs: torch.Tensor
        self._dof_rows: torch.Tensor
        self._dof_targets: torch.Tensor
        self._dof_weights: torch.Tensor
        self._dof_ptr: torch.Tensor

    def initialize(self, assembly: Assembly) -> None:
        super().initialize(assembly)
        if not np.isfinite(self.position_tolerance) or self.position_tolerance < 0:
            raise ValueError("Tie position_tolerance must be finite and nonnegative.")

        # Resolve instance-space geometry before checking surface connectivity;
        # projection must include the instances' reference translations/rotations.
        main = assembly.get_instance(self.main_instance_name)
        secondary = assembly.get_instance(self.secondary_instance_name)
        self._main_index = main._RGC_index
        self._secondary_index = secondary._RGC_index
        main_coordinates = main.nodes
        secondary_coordinates = secondary.nodes
        if not main_coordinates.is_floating_point() or not secondary_coordinates.is_floating_point():
            raise ValueError("Tie reference coordinates must be floating-point tensors.")
        if (main_coordinates.dtype != secondary_coordinates.dtype
                or main_coordinates.device != secondary_coordinates.device):
            raise ValueError("Tied instances must use the same dtype and device.")

        main_faces = main.surfaces.get_elements(self.main_surface_name)
        secondary_faces = secondary.surfaces.get_elements(self.secondary_surface_name)
        if not main_faces or not any(face._elems.numel() for face in main_faces):
            raise ValueError(f"Main surface '{self.main_surface_name}' is missing or empty.")
        if not secondary_faces or not any(face._elems.numel() for face in secondary_faces):
            raise ValueError(
                f"Secondary surface '{self.secondary_surface_name}' is missing or empty.")
        if not (torch.isfinite(main_coordinates).all()
                and torch.isfinite(secondary_coordinates).all()):
            raise ValueError("Tie reference coordinates must be finite.")
        for faces, coordinates in [
            (main_faces, main_coordinates), (secondary_faces, secondary_coordinates),
        ]:
            for face in faces:
                ids = face._elems
                if ids.numel() and (ids.min() < 0 or ids.max() >= len(coordinates)):
                    raise ValueError("Tie surface connectivity contains an out-of-bounds node.")

        # Subclasses generate weights; validate their common contract before
        # preparing the shared displacement and force/matrix transfer kernels.
        self._build_interpolation_map(
            main_faces, secondary_faces, main_coordinates, secondary_coordinates)
        self._validate_interpolation_map(main_coordinates, secondary_coordinates)
        self._initialize_dof_map()

    def _build_interpolation_map(
        self,
        main_faces: list[BaseSurface],
        secondary_faces: list[BaseSurface],
        main_coordinates: torch.Tensor,
        secondary_coordinates: torch.Tensor,
    ) -> None:
        """Populate the secondary nodes and node-level COO interpolation entries."""
        raise NotImplementedError("A LinearTie subclass must construct its interpolation.")

    def _validate_interpolation_map(
        self, main_coordinates: torch.Tensor, secondary_coordinates: torch.Tensor,
    ) -> None:
        """Check COO indexing, partition of unity and absence of self-dependence."""
        rows, nodes, weights = self._mapping_rows, self._main_nodes, self._weights
        if rows.ndim != 1 or nodes.shape != rows.shape or weights.shape != rows.shape:
            raise ValueError("Tie interpolation entries must be equally sized vectors.")
        if self._secondary_nodes.ndim != 1:
            raise ValueError("Tie secondary node indices must be a vector.")
        if (self._secondary_nodes.numel() > 1
                and not torch.all(self._secondary_nodes[1:] > self._secondary_nodes[:-1])):
            raise ValueError("Tie secondary node indices must be sorted and unique.")
        if self._secondary_nodes.numel() and (
            self._secondary_nodes.min() < 0
            or self._secondary_nodes.max() >= len(secondary_coordinates)
        ):
            raise ValueError("Tie secondary node index is out of bounds.")
        if rows.numel() and (
            rows.min() < 0 or rows.max() >= len(self._secondary_nodes)
            or nodes.min() < 0 or nodes.max() >= len(main_coordinates)
        ):
            raise ValueError("Tie interpolation index is out of bounds.")
        if not torch.isfinite(weights).all():
            raise ValueError("Tie interpolation weights must be finite.")
        # Partition of unity preserves a uniform translation. Higher-order
        # shape functions can be negative, so positivity is not required.
        row_sums = weights.new_zeros(len(self._secondary_nodes))
        row_sums.index_add_(0, rows, weights)
        tolerance = 100 * torch.finfo(weights.dtype).eps
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance, rtol=tolerance):
            raise ValueError("Tie interpolation must reproduce constant translations.")
        if (self._main_index == self._secondary_index
                and torch.isin(nodes, self._secondary_nodes).any()):
            raise ValueError("A Tie cannot use its dependent nodes as main nodes.")

    def _initialize_dof_map(self) -> None:
        """Expand compact node weights into a CSR map for scalar DoFs."""
        device = self._weights.device
        axes = torch.arange(3, device=device)
        starts = self._assembly._RGC_list_indexStart
        # Replicate each node coefficient over x/y/z without forming C kron I3.
        self._secondary_dofs = (
            starts[self._secondary_index] + self._secondary_nodes[:, None] * 3 + axes
        ).reshape(-1)
        rows = (self._mapping_rows[:, None] * 3 + axes).reshape(-1)
        targets = (
            starts[self._main_index] + self._main_nodes[:, None] * 3 + axes
        ).reshape(-1)
        # Group associations by dependent scalar DoF so CSR offsets can find
        # all replacement columns during sparse row/column expansion.
        order = torch.argsort(rows, stable=True)
        self._dof_rows = rows[order]
        self._dof_targets = targets[order]
        self._dof_weights = self._weights.repeat_interleave(3)[order]
        counts = torch.bincount(rows, minlength=self._secondary_dofs.numel())
        self._dof_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    def get_input_dofs(self) -> NDArray[np.int64]:
        return self._dof_targets.detach().cpu().numpy()

    def get_output_dofs(self) -> NDArray[np.int64]:
        return self._secondary_dofs.detach().cpu().numpy()

    def set_required_DoFs(
        self, RGC_remain_index: list[NDArray[np.bool_]],
    ) -> list[NDArray[np.bool_]]:
        # Reverse constraint order first activates required sources, then lets
        # their upstream producers eliminate them when resolving a Tie chain.
        main_nodes = self._main_nodes.detach().cpu().numpy()
        secondary_nodes = self._secondary_nodes.detach().cpu().numpy()
        RGC_remain_index[self._main_index][main_nodes] = True
        RGC_remain_index[self._secondary_index][secondary_nodes] = False
        return RGC_remain_index

    def _interpolate(self, main_displacements: torch.Tensor) -> torch.Tensor:
        """Apply C without constructing a matrix or a dense Jacobian."""
        result = main_displacements.new_zeros((len(self._secondary_nodes), 3))
        contributions = (
            self._weights.to(main_displacements)[:, None] * main_displacements[self._main_nodes]
        )
        return result.index_add_(0, self._mapping_rows, contributions)

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        # Clone the destination block so gathering sources remains valid even
        # for disjoint main/secondary surfaces belonging to the same instance.
        secondary = RGC[self._secondary_index].clone()
        secondary[self._secondary_nodes] = self._interpolate(RGC[self._main_index])
        RGC[self._secondary_index] = secondary
        return RGC

    @overload
    def modify_R_K(
        self, RGC: list[torch.Tensor] | None, R0: torch.Tensor,
        K_indices: torch.Tensor, K_values: torch.Tensor,
        if_onlyforce: Literal[False] = False, *args: object, **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def modify_R_K(
        self, RGC: list[torch.Tensor] | None, R0: torch.Tensor,
        K_indices: torch.Tensor | None = None, K_values: torch.Tensor | None = None,
        *, if_onlyforce: Literal[True], **kwargs: object,
    ) -> torch.Tensor: ...

    @overload
    def modify_R_K(
        self, RGC: list[torch.Tensor] | None, R0: torch.Tensor,
        K_indices: torch.Tensor | None = None, K_values: torch.Tensor | None = None,
        if_onlyforce: bool = False, *args: object, **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def modify_R_K(
        self, RGC: list[torch.Tensor] | None, R0: torch.Tensor,
        K_indices: torch.Tensor | None = None, K_values: torch.Tensor | None = None,
        if_onlyforce: bool = False, *args: object, **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return H.T R - R and H.T K H - K for Assembly's additive hooks.

        H replaces each secondary displacement by its interpolation. The
        chain rule is analytic: H is constant, so no Hessian term is needed.
        """
        # Remove secondary residuals and transfer their virtual work to the
        # independent main DoFs using exactly the displacement weights.
        residual = torch.zeros_like(R0)
        residual[self._secondary_dofs] = -R0[self._secondary_dofs]
        residual.index_add_(
            0, self._dof_targets,
            self._dof_weights.to(R0) * R0[self._secondary_dofs[self._dof_rows]],
        )
        if if_onlyforce:
            return residual
        if K_indices is None or K_values is None:
            raise ValueError("K_indices and K_values are required for Tie stiffness reduction.")
        indices, values = self._transform_matrix_delta(K_indices, K_values)
        return residual, indices, values

    def _transform_matrix_delta(
        self, indices: torch.Tensor, values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse congruence, expanding only entries touching dependent DoFs."""
        if not self._secondary_dofs.numel() or not values.numel():
            return indices[:, :0], values[:0]

        indices = indices.to(device=values.device, dtype=torch.int64)

        # Leave independent-independent entries untouched. Only entries with
        # a dependent row or column need the congruence transformation.
        dofs = self._secondary_dofs.to(indices.device)
        positions = torch.searchsorted(dofs, indices.contiguous())
        positions = positions.clamp_max(len(dofs) - 1)
        dependent = dofs[positions] == indices
        affected = dependent.any(dim=0)
        original_indices = indices[:, affected]
        original_values = values[affected]
        positions = positions[:, affected]
        dependent = dependent[:, affected]
        if not original_values.numel():
            return indices[:, :0], values[:0]

        # Each affected COO entry expands over its row and column source
        # associations; CSR offsets support different stencil lengths per row.
        ptr = self._dof_ptr.to(indices.device)
        targets = self._dof_targets.to(indices.device)
        counts = torch.where(
            dependent, ptr[positions + 1] - ptr[positions], 1)
        entry_counts = counts.prod(dim=0)
        source = torch.repeat_interleave(
            torch.arange(len(original_values), device=indices.device), entry_counts)
        offsets = torch.arange(len(source), device=indices.device) - torch.repeat_interleave(
            entry_counts.cumsum(0) - entry_counts, entry_counts)
        choices = torch.stack([
            offsets // counts[1, source], offsets % counts[1, source],
        ])
        slots = ptr[positions[:, source]] + choices
        mapped_indices = torch.where(
            dependent[:, source], targets[slots], original_indices[:, source])
        weights = self._dof_weights.to(device=values.device, dtype=values.dtype)
        factors = torch.where(dependent[:, source], weights[slots], 1).prod(dim=0)
        mapped_values = original_values[source] * factors

        # Assembly adds our return value: subtract the original affected
        # entries, then coalesce replacement contributions and cancellations.
        delta_indices = torch.cat([mapped_indices, original_indices], dim=1)
        delta_values = torch.cat([mapped_values, -original_values])
        size = self._assembly._RGC_list_indexStart[-1]
        delta = torch.sparse_coo_tensor(
            delta_indices, delta_values, (size, size)).coalesce()
        nonzero = delta.values() != 0
        return delta.indices()[:, nonzero], delta.values()[nonzero]

    def modify_mass_matrix(
        self, mass_indices: torch.Tensor, mass_values: torch.Tensor,
        RGC: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same constant-coordinate congruence to the mass matrix."""
        return self._transform_matrix_delta(mass_indices, mass_values)
