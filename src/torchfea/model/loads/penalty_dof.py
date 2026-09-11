from __future__ import annotations

import numpy as np
import torch

from .base import BaseLoad


class Penalty_DoF(BaseLoad):
    """
    Penalize one DoF in one object's RGC segment to track a target value using quadratic penalty energy.

    Parameters:
        obj_name: object name in assembly
        s: local flattened DoF index in the selected object's RGC segment
        target: desired value for the selected DoF
        k: penalty coefficient
        obj_type: one of {'auto', 'instance', 'rp', 'load', 'constraint'}
    """

    _serialized_attributes = ['obj_name', 's', 'target', 'k', '_parameters', 'obj_type']

    def __init__(
        self,
        obj_name: str,
        s: int,
        target: float,
        k: float,
        obj_type: str = "auto",
    ) -> None:
        super().__init__()

        self.obj_name = obj_name
        self.s = int(s)
        self.obj_type = obj_type

        # [k, target]
        self._parameters = torch.tensor([k, target], dtype=torch.float64)
        self._indices_force: torch.Tensor | None = None

        # Located in initialize from object + local index s
        self._rgc_list_index: int | None = None
        self._rgc_local_flat_index: int | None = None
        self._global_s: int | None = None



    @property
    def k(self) -> torch.Tensor:
        return self._parameters[0]

    @k.setter
    def k(self, value: float) -> None:
        self._parameters[0] = value

    @property
    def target(self) -> torch.Tensor:
        return self._parameters[1]

    @target.setter
    def target(self, value: float) -> None:
        self._parameters[1] = value

    def _locate_object_local_s(self):
        obj = self._assembly.get_object(self.obj_name, self.obj_type)
        rgc_idx = obj._RGC_index

        if rgc_idx is None:
            raise ValueError(f"Object '{self.obj_name}' does not have a valid _RGC_index.")

        seg_size = int(np.prod(self._assembly._RGC_size[rgc_idx]))
        if self.s < 0 or self.s >= seg_size:
            raise ValueError(
                f"local s={self.s} out of range for object '{self.obj_name}', segment size={seg_size}"
            )

        global_s = int(self._assembly._RGC_list_indexStart[rgc_idx]) + self.s
        return rgc_idx, self.s, global_s

    def initialize(self, assembly):
        super().initialize(assembly)

        rgc_idx, local_flat, global_s = self._locate_object_local_s()
        self._rgc_list_index = rgc_idx
        self._rgc_local_flat_index = local_flat
        self._global_s = global_s

        self._indices_force = torch.tensor([self._global_s], dtype=torch.int64, device=assembly.device)

    def _get_params_like(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k.to(device=ref.device, dtype=ref.dtype)
        target = self.target.to(device=ref.device, dtype=ref.dtype)
        return k, target

    def _get_s_now(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return RGC[self._rgc_list_index].reshape(-1)[self._rgc_local_flat_index]

    def get_stiffness(
        self,
        RGC: list[torch.Tensor],
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        s_now = self._get_s_now(RGC)
        k, target = self._get_params_like(s_now)

        # f is defined like spring loads so Assembly residual uses -f.
        f = k * (target - s_now)
        F_indices = self._indices_force
        F_values = f.reshape(1)

        if if_onlyforce:
            return F_indices, F_values

        # Tangent of f wrt selected DoF: df/ds = -k
        K_indices = torch.stack([F_indices, F_indices], dim=0)
        K_values = (-k).reshape(1)
        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        s_now = self._get_s_now(RGC)
        k, target = self._get_params_like(s_now)
        # Negative sign so Assembly._total_Potential_Energy adds penalty energy.
        return -0.5 * k * (s_now - target) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        arr = RGC_remain_index[self._rgc_list_index].reshape(-1)
        arr[self._rgc_local_flat_index] = True
        return RGC_remain_index
