from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import BaseConstraint


class Connector(BaseConstraint):
    """Kinematic joint between two reference points.

    ``connection_point`` is a point in the initial global coordinate system.
    If it is omitted, the initial position of the master reference point is
    used.  The constraint is imposed on this point, expressed in the local
    coordinate systems of the master and slave reference points.

    ``translation_free_directions`` contains the directions in which the
    connection point is allowed to translate relative to the master point.
    The relative translation is the linear combination of these vectors and
    the corresponding local free coordinates.  If the list is empty, the
    connection point has no relative translation.

    ``rotation_free_directions`` has the same meaning for relative rotation.
    The directions are interpreted in the master reference point's initial
    body-fixed frame.

    The joint keeps the master reference point's six coordinates and one
    redundant-coordinate block containing the relative translation and
    rotation free coordinates.  The slave reference point is reconstructed
    as::

        R_slave = R_master @ Exp(relative_rotation)

    where both relative motions are linear combinations of their respective
    free-direction bases.
    """

    _serialized_attributes = [
        "master_rp_name",
        "slave_rp_name",
        "connection_point",
        "kinetic_directions",
    ]

    def __init__(
        self,
        master_rp_name: str,
        slave_rp_name: str,
        kinetic_directions: list[Sequence[float]] | None = None,
        connection_point: Sequence[float] | None = None,
    ) -> None:
        """Initialize a Connector constraint.

        Args:
            master_rp_name: Name of the master reference point.
            slave_rp_name: Name of the slave reference point.
            kinetic_directions: List of 6D kinetic vectors. Each vector has
                the form ``[tx, ty, tz, rx, ry, rz]``. The sequence of the
                vectors determines the order of the kinetic coordinates.
            connection_point: Optional 3D point ``[x, y, z]`` in the initial
                global coordinate system where the joint is applied. If not
                provided, the initial position of the master reference point
                is used.
        """

        super().__init__()

        self.master_rp_name = master_rp_name
        self.slave_rp_name = slave_rp_name
        self.connection_point = connection_point
        self.kinetic_directions = kinetic_directions

        # Runtime state, rebuilt after deserialization in initialize().
        self._master_rp_index: int | None = None
        self._slave_rp_index: int | None = None
        self._master_connection_offset: torch.Tensor | None = None
        self._slave_connection_offset: torch.Tensor | None = None

    @property
    def _RGC_requirements(self) -> tuple[int, ...]:
        """Shape of the connector's coordinate block at runtime."""
        count = (
            len(self.kinetic_directions)
            if self.kinetic_directions is not None
            else 0
        )
        return (count,)
 
    def initialize(self, assembly):
        if self.master_rp_name == self.slave_rp_name:
            raise ValueError(
                "Joint master_rp_name and slave_rp_name must be different."
            )

        super().initialize(assembly)

        master = self._assembly.get_reference_point(self.master_rp_name)
        slave = self._assembly.get_reference_point(self.slave_rp_name)

        self._master_rp_index = master._RGC_index
        self._slave_rp_index = slave._RGC_index

        connection_point = (
            master.node
            if self.connection_point is None
            else torch.as_tensor(
                self.connection_point,
                dtype=master.node.dtype,
                device=master.node.device,
            )
        )
        self._master_connection_offset = connection_point.to(
            self._assembly.device
        ) - master.node.to(self._assembly.device)

        self._slave_connection_offset = connection_point.to(
            self._assembly.device
        ) - slave.node.to(self._assembly.device)

    def get_input_blocks(self) -> set:
        return {
            self._master_rp_index,
            self._RGC_index,
        }

    def get_output_blocks(self) -> set:
        return {self._slave_rp_index}

    def set_required_DoFs(self, RGC_remain_index):
        # The master RP is an independent rigid-body coordinate block.
        RGC_remain_index[self._master_rp_index][:] = True

        # The joint block stores the relative translation and rotation free
        # coordinates.
        RGC_remain_index[self._RGC_index].reshape(-1)[:] = True

        # The slave RP is reconstructed by modify_RGC().
        RGC_remain_index[self._slave_rp_index][:] = False
        return RGC_remain_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        """Reconstruct the slave reference point's coordinates from the master and the joint's relative coordinates."""

        return RGC

    def modify_R_K(
        self,
        RGC: list[torch.Tensor],
        R0: torch.Tensor,
        K_indices: torch.Tensor = None,
        K_values: torch.Tensor = None,
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        if if_onlyforce:
            jacobian, _ = self._kinematic_derivatives(
                RGC,
                need_hessian=False,
            )
        else:
            if K_indices is None or K_values is None:
                raise ValueError(
                    "K_indices and K_values must be provided when "
                    "if_onlyforce is False."
                )
            jacobian, hessian = self._kinematic_derivatives(
                RGC,
                need_hessian=True,
            )

        slave_rgc_start = self._assembly._RGC_list_indexStart[self._slave_rp_index]
        slave_rgc_indices = torch.arange(
            slave_rgc_start,
            slave_rgc_start + 6,
            dtype=torch.int64,
            device=R0.device,
        )
        active_rgc_indices = self._global_active_indices(R0.device)

        slave_force = R0[slave_rgc_indices]
        mapped_force = jacobian.transpose(0, 1) @ slave_force
        R = torch.zeros_like(R0).scatter_add_(
            0,
            active_rgc_indices,
            mapped_force.to(device=R0.device, dtype=R0.dtype),
        )

        if if_onlyforce:
            return R

        mapped_indices, mapped_values = self._map_sparse_entries(
            K_indices,
            K_values,
            jacobian,
            active_rgc_indices,
            slave_rgc_indices,
        )

        geometric = torch.einsum(
            "i,ijk->jk",
            slave_force.to(device=hessian.device, dtype=hessian.dtype),
            hessian,
        )
        geometric_rows = active_rgc_indices.repeat_interleave(
            active_rgc_indices.numel()
        )
        geometric_cols = active_rgc_indices.repeat(active_rgc_indices.numel())
        geometric_indices = torch.stack(
            [
                geometric_rows.to(K_indices.dtype),
                geometric_cols.to(K_indices.dtype),
            ]
        )

        return (
            R,
            torch.cat([mapped_indices, geometric_indices], dim=1),
            torch.cat([mapped_values, geometric.reshape(-1)]),
        )

    def modify_mass_matrix(
        self,
        mass_indices: torch.Tensor,
        mass_values: torch.Tensor,
        RGC: list[torch.Tensor],
    ):
        jacobian, _ = self._kinematic_derivatives(
            RGC,
            need_hessian=False,
        )

        active_indices = self._global_active_indices(mass_indices.device)
        slave_indices = self._global_slave_indices(mass_indices.device)

        return self._map_sparse_entries(
            mass_indices,
            mass_values,
            jacobian,
            active_indices,
            slave_indices,
        )
