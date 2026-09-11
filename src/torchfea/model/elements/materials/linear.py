import torch

from .base import Materials_Base


class LinearElastic(Materials_Base):
    """
    Saint Venant--Kirchhoff material written in the existing material API.

    The constitutive law is linear in the Green--Lagrange strain, but the
    kinematics are finite-strain:

    ``C = F.T @ F``, ``E = 1/2 * (C - I)``,
    ``S = lambda * tr(E) * I + 2 * mu * E``.

    ``Element_3D`` contracts the material stress with reference gradients and
    expects the transpose of the first Piola stress.  Consequently this class
    returns ``P.T`` and the consistent ``d(P.T)/dF`` tangent.
    """

    def __init__(self, E: torch.Tensor | float,
                 nu: torch.Tensor | float) -> None:
        """
        Initialize a linear elastic material for large deformation.
        
        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        super().__init__()

        self.type = 2  # Material type 2 for linear elasticity

        # Normalize scalar inputs so integer and NumPy scalar parameters also
        # support the ``.to(device, dtype)`` conversion used during assembly.
        if not torch.is_tensor(E):
            E = torch.tensor([E], dtype=torch.get_default_dtype())
        if not torch.is_tensor(nu):
            nu = torch.tensor([nu], dtype=torch.get_default_dtype())

        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Pre-compute Lamé parameters
        self.lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus (second Lamé parameter)

    def _broadcast_param(self, x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Broadcast material parameter to [g, e] shape."""
        g, e = F.shape[0], F.shape[1]
        if x.dim() == 0 or x.numel() == 1:
            return x.reshape(1, 1).expand(g, e)
        if x.dim() == 1:
            if x.shape[0] == g:
                return x.view(g, 1).expand(g, e)
            if x.shape[0] == e:
                return x.view(1, e).expand(g, e)
            raise ValueError(f"Cannot broadcast parameter with shape {tuple(x.shape)} to [{g}, {e}]")
        if x.dim() == 2 and x.shape[0] == g and x.shape[1] == e:
            return x
        raise ValueError(f"Unsupported parameter shape {tuple(x.shape)}")

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None,
                                 I1: torch.Tensor = None,
                                 J: torch.Tensor = None):
        """
        Compute the strain energy density for large deformation linear elasticity.
        
        For large deformations, we use the Saint Venant-Kirchhoff model:
        W = (lambda/2)(tr(E))^2 + mu*tr(E^2)
        where E = 1/2*(F^T*F - I) is the Green-Lagrange strain tensor
        
        Args:
            F: Deformation gradient
            I1: First invariant (optional)
            J: Jacobian determinant (optional)
            
        Returns:
            Strain energy density
        """
        # Green-Lagrange strain tensor E = 1/2*(C - I)
        C = torch.matmul(F.transpose(-1, -2), F)
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3)
        E = 0.5 * (C - I_tensor)

        # Trace of E: tr(E)
        tr_E = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1)

        # Compute E^2
        E_squared = torch.matmul(E, E)

        # Trace of E^2: tr(E^2)
        tr_E_squared = torch.diagonal(E_squared, dim1=-2, dim2=-1).sum(-1)

        # Broadcast Lamé parameters
        lambda_ = self._broadcast_param(self.lambda_.to(F.device, F.dtype), F)
        mu = self._broadcast_param(self.mu.to(F.device, F.dtype), F)

        # W = (lambda/2)(tr(E))^2 + mu*tr(E^2)
        W = 0.5 * lambda_ * tr_E**2 + mu * tr_E_squared

        return W

    def material_Constitutive_C3(self, F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):
        """
        Constitutive response for the ``Element_3D`` total-Lagrangian form.

        The extra keyword arguments match the hyperelastic material interface
        used by ``Element_3D``.  The
        Saint-Venant--Kirchhoff formulation only needs ``F`` and computes its
        own right Cauchy--Green tensor.

        Args:
            F: deformation gradient, shape [g, e, 3, 3]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``P.T`` with shape [g, e, 3, 3]
                - ``d(P.T)/dF`` with shape [g, e, 3, 3, 3, 3]
        """
        batch_size, elem_size = F.shape[0], F.shape[1]

        # Right Cauchy-Green deformation tensor
        C = torch.matmul(F.transpose(-1, -2), F)

        # Identity tensor
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3)

        # Green-Lagrange strain tensor
        E = 0.5 * (C - I_tensor)

        # Trace of Green-Lagrange strain tensor
        tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1)

        # Broadcast Lamé parameters
        lambda_ = self._broadcast_param(self.lambda_.to(F.device, F.dtype), F)
        mu = self._broadcast_param(self.mu.to(F.device, F.dtype), F)

        lambda4 = lambda_.view(batch_size, elem_size, 1, 1)
        mu4 = mu.view(batch_size, elem_size, 1, 1)

        # 2nd Piola stress S = lambda tr(E) I + 2 mu E
        S = lambda4 * tr_E.view(batch_size, elem_size, 1, 1) * I_tensor + 2.0 * mu4 * E

        # First Piola stress P = F S.  Element_3D uses reference gradients
        # with the opposite index order, so return P.T as ``s``.
        P = torch.matmul(F, S)
        stress = P.transpose(-1, -2)

        # Material tangent dS/dE.  The explicit identity products keep the
        # tensor index order visible: Cmat[A,B,C,D] = dS[A,B]/dE[C,D].
        identity = torch.eye(3, device=F.device, dtype=F.dtype)
        lambda4 = lambda_.view(batch_size, elem_size, 1, 1, 1, 1)
        mu4 = mu.view(batch_size, elem_size, 1, 1, 1, 1)
        Cmat = (
            lambda4 * torch.einsum('ab,cd->abcd', identity, identity)
            + mu4 * (
                torch.einsum('ac,bd->abcd', identity, identity)
                + torch.einsum('ad,bc->abcd', identity, identity)))

        # A[i,A,k,L] = dP[i,A]/dF[k,L]
        #            = delta[i,k] S[L,A]
        #              + F[i,B] Cmat[B,A,L,D] F[k,D].
        geometric = torch.einsum('ik,geLA->geiAkL', identity, S)
        material = torch.einsum('geiB,geBALD,gekD->geiAkL', F, Cmat, F)
        tangent_P = geometric + material

        # Match the residual/stiffness index convention in C3base.py:
        # returned tangent[A,i,k,L] = d(P[i,A])/dF[k,L].
        tangent = tangent_P.permute(0, 1, 3, 2, 4, 5).contiguous()
        return stress, tangent
