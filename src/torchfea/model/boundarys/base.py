import numpy as np
import torch
from ..obj_base import BaseObj


class BaseBoundary(BaseObj):
	"""
	Boundary base class. Provides the same surface API as constraints where relevant,
	but typically only modifies RGC and required DoFs (Dirichlet conditions).
	"""

	def __init__(self) -> None:
		super().__init__()

	def initialize(self, assembly):
		super().initialize(assembly)

	def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
		"""
		Modify the RGC_remain_index to deactivate constrained DoFs.
		"""
		return RGC_remain_index

	def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
		"""
		Apply boundary conditions directly to the RGC values (Dirichlet), if needed.
		"""
		return RGC

	# For compatibility with Assembly constraint hooks, provide no-op stubs
	def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
				   K_indices: torch.Tensor = None, K_values: torch.Tensor = None,
				   if_onlyforce: bool = False, *args, **kwargs):
		if if_onlyforce:
			return torch.zeros(self._assembly.RGC_list_indexStart[-1])
		return (torch.zeros(self._assembly.RGC_list_indexStart[-1]),
				torch.zeros([2, 0], dtype=torch.int64),
				torch.zeros([0]))

