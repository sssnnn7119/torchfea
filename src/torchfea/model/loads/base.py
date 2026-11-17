from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import Assembly

import torch

from ..obj_base import BaseObj


class BaseLoad(BaseObj):
    _serialized_attributes_exclude = [
        "_indices_matrix",
        "_indices_force",
        "_index_matrix_coalesce",
        "_continuation_factor",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._indices_matrix: torch.Tensor = torch.zeros([2, 0], dtype=torch.int)
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = torch.zeros([0], dtype=torch.int)
        """
            the start index of the stiffness matricx of structural stress
        """

        self._parameters: torch.Tensor = torch.zeros(0, dtype=torch.float64)
        """The parameters of this object.
        This is a 1D tensor containing all the parameters of this object.
        """

    def initialize(self, assembly: Assembly):
        super().initialize(assembly)

    def get_stiffness(
        self,
        RGC: list[torch.Tensor],
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        """Public load response entry point.

        Derived loads implement :meth:`_get_stiffness`; this method is kept as
        a stable wrapper so load strategies can decorate force and tangent
        responses without changing every assembly call site.
        """
        return self._get_stiffness(RGC=RGC, if_onlyforce=if_onlyforce, *args, **kwargs)

    def _get_stiffness(
        self,
        RGC: list[torch.Tensor],
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        """Get the stiffness matrix and force vector for the self-contact load.

        Args:
            RGC (list[torch.Tensor]): The global coordinates of the nodes.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The stiffness matrix and force vector for the self-contact load.
                - F_indices: The indices of the force vector.
                - F_values: The values of the force vector.
                - K_indices: The indices of the stiffness matrix.
                - K_values: The values of the stiffness matrix.
        """
        raise NotImplementedError("_get_stiffness method not implemented")

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """Public potential-energy entry point for load strategies."""
        return self._get_potential_energy(RGC=RGC)

    def _get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """Get the potential energy for the load."""
        raise NotImplementedError("get_potential_energy method not implemented")

    @staticmethod
    def get_F0():
        raise NotImplementedError("get_F0 method not implemented")


class ContinuousLoad(BaseLoad):
    """Load wrapper with a scalar continuation factor.

    The factor multiplies the load residual, tangent and potential energy at
    the common public wrapper layer.  This keeps the underlying load
    implementation unchanged and makes rollback a solver-level operation:
    restore the last accepted configuration and factor, then retry.
    """

    def __init__(self) -> None:
        super().__init__()
        self._continuation_factor: float | torch.Tensor = 1.0

    @property
    def continuation_factor(self) -> float | torch.Tensor:
        """Current continuation factor; one means the full load."""
        return self._continuation_factor

    @continuation_factor.setter
    def continuation_factor(self, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("continuation_factor must be a scalar")
            self._continuation_factor = value
        else:
            self._continuation_factor = float(value)

    def get_stiffness(
        self,
        RGC: list[torch.Tensor],
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        response = super().get_stiffness(
            RGC=RGC, if_onlyforce=if_onlyforce, *args, **kwargs
        )
        if if_onlyforce:
            force_indices, force_values = response
            factor = self._factor_like(force_values)
            return force_indices, force_values * factor

        force_indices, force_values, K_indices, K_values = response
        factor = self._factor_like(force_values)
        return (
            force_indices,
            force_values * factor,
            K_indices,
            K_values * factor,
        )

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        energy = super().get_potential_energy(RGC=RGC)
        return energy * self._factor_like(energy)

    def _factor_like(self, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(self.continuation_factor, torch.Tensor):
            return self.continuation_factor.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        return reference.new_tensor(self.continuation_factor)


# Both names are useful in user code; keep one implementation.
ContinuationLoad = ContinuousLoad
