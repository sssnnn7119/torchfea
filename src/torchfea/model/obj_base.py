from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly import Assembly

import numpy as np
import torch

from ..interfaces import Serializable


class BaseObj(Serializable):
    _serialized_attributes_exclude = [
        "_assembly",
        "_RGC_index",
        "_index_start",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._RGC_requirements: list[int] = [0]
        """Number of redundant generalized coordinates required by this object."""
        self._RGC_index: int = None
        """Index of this object's redundant generalized coordinates."""
        self._index_start: int = None
        """Start index of this object's redundant generalized coordinates."""
        self._assembly: Assembly = None
        """Assembly containing this object."""
        self.enabled: bool = True
        """Whether this object participates in analysis."""

    @property
    def serialized_attributes(self):
        """Return persistent object attributes."""
        serialized_attrs = super().serialized_attributes
        serialized_attrs = [attr for attr in serialized_attrs if attr != "_assembly"]
        serialized_attrs.append("_RGC_requirements")
        return serialized_attrs

    def set_RGC_index(self, index: int) -> None:
        """Set the redundant-coordinate block index."""
        self._RGC_index = index

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """Update the reduced-coordinate mask for this object."""
        return RGC_remain_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return RGC

    def initialize(self, assembly: Assembly):
        self._assembly = assembly
        self._index_start = assembly._RGC_list_indexStart[self._RGC_index]

    def initialize_dynamic(self):
        pass

    def reinitialize(self, RGC: list[torch.Tensor]):
        pass
