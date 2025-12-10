import torch
import numpy as np
from .base import BaseLoad


def _spring_tangent_block(d: torch.Tensor, k: float, L0: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the 3x3 tangent block for a geometrically exact axial spring.

    Parameters:
    - d: vector from node 1 to node 2 (x2 - x1)
    - k: stiffness
    - L0: rest length

    Returns:
    - K (3x3): df1/dd, such that
        [K11 K12; K21 K22] with K12 = K, K11 = -K, K21 = -K, K22 = K
      for the two-point spring case.
    For RP-to-point, the effective stiffness wrt the RP is K11 = -K.
    """
    l = torch.linalg.norm(d)
    if l.item() < eps:
        # Degenerate: no direction; return zero block to avoid NaNs
        return torch.zeros((3, 3), dtype=d.dtype, device=d.device)

    n = d / l
    I = torch.eye(3, dtype=d.dtype, device=d.device)
    # Consistent tangent commonly used for axial spring linearization
    # K = k * [ n n^T + ((l - L0)/l) * (I - n n^T) ]
    K = k * (torch.outer(n, n) + ((l - L0) / l) * (I - torch.outer(n, n)))
    return K


class Spring_RP_RP(BaseLoad):
    """
    A nonlinear axial spring connecting two reference points (RP-RP).

    Parameters:
        rp_name1: name of first reference point
        rp_name2: name of second reference point
        k: spring stiffness
        rest_length (optional): rest length L0; defaults to initial distance between RPs
    """

    def __init__(self, rp_name1: str, rp_name2: str, k: float, rest_length: float | None = None) -> None:
        super().__init__()
        self.rp_name1 = rp_name1
        self.rp_name2 = rp_name2
        
        rl = rest_length if rest_length is not None else -1.0
        self._parameters = torch.tensor([k, rl], dtype=torch.float64)

    @property
    def k(self) -> float:
        return self._parameters[0]
    
    @k.setter
    def k(self, value: float) -> None:
        self._parameters[0] = value

    @property
    def rest_length(self) -> float:
        return self._parameters[1]
    
    @rest_length.setter
    def rest_length(self, value: float) -> None:
        self._parameters[1] = value

        # indices cache
        self._rp_index1: int | None = None
        self._rp_index2: int | None = None
        self._idx_tr1: torch.Tensor | None = None
        self._idx_tr2: torch.Tensor | None = None

    def initialize(self, assembly):
        super().initialize(assembly)
        rp1 = assembly.get_reference_point(self.rp_name1)
        rp2 = assembly.get_reference_point(self.rp_name2)
        self._rp_index1 = rp1._RGC_index
        self._rp_index2 = rp2._RGC_index

        s1 = assembly.RGC_list_indexStart[self._rp_index1]
        s2 = assembly.RGC_list_indexStart[self._rp_index2]
        self._idx_tr1 = torch.arange(s1, s1 + 3, device=assembly.device, dtype=torch.int64)
        self._idx_tr2 = torch.arange(s2, s2 + 3, device=assembly.device, dtype=torch.int64)

        # Default rest length from initial geometry if not provided
        if self.rest_length < 0:
            p1 = rp1.node.to(assembly.device).to(torch.get_default_dtype())
            p2 = rp2.node.to(assembly.device).to(torch.get_default_dtype())
            self.rest_length = torch.linalg.norm(p2 - p1).item()

    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs):
        # Current positions of the two RPs
        rp1 = self._assembly.get_reference_point(self.rp_name1)
        rp2 = self._assembly.get_reference_point(self.rp_name2)
        x1 = rp1.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index1][:3]
        x2 = rp2.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index2][:3]

        d = x2 - x1
        l = torch.linalg.norm(d)
        eps = 1e-16
        if l.item() < eps:
            # No defined direction; no force or stiffness
            F_indices = torch.cat([self._idx_tr1, self._idx_tr2], dim=0)
            F_values = torch.zeros(6, dtype=x1.dtype, device=x1.device)
            if if_onlyforce:
                return F_indices, F_values
            K_indices = torch.zeros((2, 0), dtype=torch.int64, device=x1.device)
            K_values = torch.zeros(0, dtype=x1.dtype, device=x1.device)
            return F_indices, F_values, K_indices, K_values

        n = d / l
        f = self.k * (l - self.rest_length) * n
        f1 = f
        f2 = -f

        F_indices = torch.cat([self._idx_tr1, self._idx_tr2], dim=0)
        F_values = torch.cat([f1, f2], dim=0)

        if if_onlyforce:
            return F_indices, F_values

        # Tangent blocks (3x3)
        K_block = _spring_tangent_block(d, self.k, self.rest_length)
        # Build COO indices for 6x6 blocks
        rows11 = self._idx_tr1.repeat_interleave(3)
        cols11 = self._idx_tr1.repeat(3)
        rows12 = self._idx_tr1.repeat_interleave(3)
        cols12 = self._idx_tr2.repeat(3)
        rows21 = self._idx_tr2.repeat_interleave(3)
        cols21 = self._idx_tr1.repeat(3)
        rows22 = self._idx_tr2.repeat_interleave(3)
        cols22 = self._idx_tr2.repeat(3)

        K_indices = torch.stack([
            torch.cat([rows11, rows12, rows21, rows22], dim=0),
            torch.cat([cols11, cols12, cols21, cols22], dim=0)
        ], dim=0)

        K_values = torch.cat([
            (-K_block).reshape(-1),  # K11
            (K_block).reshape(-1),   # K12
            (K_block).reshape(-1),   # K21
            (-K_block).reshape(-1)   # K22
        ], dim=0)

        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        rp1 = self._assembly.get_reference_point(self.rp_name1)
        rp2 = self._assembly.get_reference_point(self.rp_name2)
        x1 = rp1.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index1][:3]
        x2 = rp2.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index2][:3]
        l = torch.linalg.norm(x2 - x1)
        # Negative sign so Assembly._total_Potential_Energy adds spring energy (internal-like)
        return -0.5 * self.k * (l - self.rest_length) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        RGC_remain_index[self._rp_index1][:3] = True
        RGC_remain_index[self._rp_index2][:3] = True
        return RGC_remain_index


class Spring_RP_Point(BaseLoad):
    """
    A nonlinear axial spring connecting one reference point to a fixed point in space (RP-Point).

    Parameters:
        rp_name: name of the reference point
        point: [x, y, z] fixed spatial point
        k: spring stiffness
        rest_length (optional): rest length L0; defaults to initial distance between RP and point
    """

    def __init__(self, rp_name: str, point: list[float], k: float, rest_length: float = None) -> None:
        
        super().__init__()
        self.rp_name = rp_name

        self.point = torch.tensor(point)
        self.k = torch.tensor(k)
        self.rest_length = None if rest_length is None else torch.tensor(rest_length)

        self._rp_index: int | None = None
        self._idx_tr: torch.Tensor | None = None

    def initialize(self, assembly):
        super().initialize(assembly)
        rp = assembly.get_reference_point(self.rp_name)
        self._rp_index = rp._RGC_index
        s = assembly.RGC_list_indexStart[self._rp_index]
        self._idx_tr = torch.arange(s, s + 3, device=assembly.device, dtype=torch.int64)
        # Materialize _P on the correct device/dtype via property setter
        self.point = torch.tensor(self.point)

        # Default rest length from initial geometry if not provided
        if self.rest_length is None:
            x0 = rp.node.to(assembly.device).to(torch.get_default_dtype())
            self.rest_length = torch.linalg.norm(self.point - x0)

    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs):

        if type(self.point) != torch.Tensor:
            self.point = torch.tensor(self.point)
        

        rp = self._assembly.get_reference_point(self.rp_name)
        x = rp.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index][:3]
        d = self.point - x
        l = torch.linalg.norm(d)

        eps = 1e-16
        if l.item() < eps:
            F_indices = self._idx_tr
            F_values = torch.zeros(3, dtype=x.dtype, device=x.device)
            if if_onlyforce:
                return F_indices, F_values
            K_indices = torch.zeros((2, 0), dtype=torch.int64, device=x.device)
            K_values = torch.zeros(0, dtype=x.dtype, device=x.device)
            return F_indices, F_values, K_indices, K_values

        n = d / l
        f = self.k * (l - self.rest_length) * n
        F_indices = self._idx_tr
        F_values = f

        if if_onlyforce:
            return F_indices, F_values

        K_block = _spring_tangent_block(d, self.k, self.rest_length)
        rows = self._idx_tr.repeat_interleave(3)
        cols = self._idx_tr.repeat(3)
        K_indices = torch.stack([rows, cols], dim=0)
        K_values = (-K_block).reshape(-1)  # derivative wrt RP coords
        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.point) != torch.Tensor:
            self.point = torch.tensor(self.point)
            
        rp = self._assembly.get_reference_point(self.rp_name)
        x = rp.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index][:3]
        l = torch.linalg.norm(self.point - x)
        # Negative sign so Assembly._total_Potential_Energy adds spring energy (internal-like)
        return -0.5 * self.k * (l - self.rest_length) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        RGC_remain_index[self._rp_index][:3] = True
        return RGC_remain_index
