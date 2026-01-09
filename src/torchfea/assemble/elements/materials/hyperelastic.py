import torch

from .base import Materials_Base


class NeoHookean(Materials_Base):

    def __init__(self, mu: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:

        super().__init__()

        self.type = 1

        # if mu is scalar, then flag it as a constant
        if type(mu) == float:
            mu = torch.tensor([mu], dtype=torch.float32)

        if type(kappa) == float:
            kappa = torch.tensor([kappa], dtype=torch.float32)

        self.mu = mu
        self.kappa = kappa

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None):

        
        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)
        W = self.mu    / 2 * (I1 - 3) + \
            self.kappa / 2 * (J  - 1)**2
        return W

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):

        if J is None:
            invF = F.inverse()
            J = F.det()
            Jneg = J**(-2 / 3)
            I1 = (F**2).sum([-1, -2]) * Jneg

        J = J.view(J.shape[0], J.shape[1], 1, 1)
        Jneg = Jneg.view(J.shape[0], J.shape[1], 1, 1)
        I1 = I1.view(J.shape[0], J.shape[1], 1, 1)

        if self.mu.dim() == 0 or self.mu.shape[0] == 1:
            mu = self.mu.view(1, 1, 1, 1)
            kappa = self.kappa.view(1, 1, 1, 1)
        else:
            mu = self.mu.view(self.mu.shape[0], self.mu.shape[1], 1, 1)
            kappa = self.kappa.view(self.kappa.shape[0], self.kappa.shape[1], 1, 1)

        muJneg = mu * Jneg
        FtMuJneg = F.transpose(-1, -2) * muJneg
        muI1invF = mu * I1 * invF
        kappaJinvF = kappa * J * invF

        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3])

        s = FtMuJneg + (-1 / 3 * muI1invF + kappaJinvF * (J - 1))

        C = torch.einsum(
            'geij,gelk->geijkl',
            -2 / 3 * FtMuJneg + kappaJinvF * (2 * J - 1) + 2 / 9 * muI1invF,
            invF)

        C += torch.einsum('geij,gekl->geijkl',
                                         -2 / 3 * muJneg * invF, F)
        C += torch.einsum('geik,gelj->geijkl',
                                         (1 / 3 * muI1invF - kappaJinvF *
                                          (J - 1)), invF)

        for m in range(3):
            for n in range(3):
                C[:, :, m, n, n, m] += (muJneg).squeeze()

        return s, C


class LinearElastic(Materials_Base):
    """
    Linear elastic material model adapted for large deformation.
    
    This model uses Young's modulus (E) and Poisson's ratio (nu) as inputs
    and implements a hyperelastic formulation based on linear elasticity that
    can handle large deformations.
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

        # Convert scalar inputs to tensors if needed
        if type(E) == float:
            E = torch.tensor([E], dtype=torch.float32)

        if type(nu) == float:
            nu = torch.tensor([nu], dtype=torch.float32)

        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Pre-compute Lamé parameters
        self.lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus (second Lamé parameter)

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
        batch_size, elem_size = F.shape[0], F.shape[1]
        
        if J is None:
            J = F.det()
            
        if I1 is None:
            # Compute right Cauchy-Green deformation tensor C = F^T·F
            C = torch.einsum('geij,gejk->geik', F, F)
            I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)
        
        # Create identity tensor with appropriate dimensions
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3).expand(batch_size, elem_size, 3, 3)
        
        # Green-Lagrange strain tensor E = 1/2*(C - I)
        C = torch.einsum('geij,gejk->geik', F, F)
        E = 0.5 * (C - I_tensor)
        
        # Trace of E: tr(E)
        tr_E = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1)
        
        # Compute E^2
        E_squared = torch.einsum('geij,gejk->geik', E, E)
        
        # Trace of E^2: tr(E^2)
        tr_E_squared = torch.diagonal(E_squared, dim1=-2, dim2=-1).sum(-1)
        
        # Expand Lamé parameters for broadcasting
        lambda_ = self.lambda_.view(-1, 1).expand(batch_size, elem_size)
        mu = self.mu.view(-1, 1).expand(batch_size, elem_size)
        
        # W = (lambda/2)(tr(E))^2 + mu*tr(E^2)
        W = 0.5 * lambda_ * tr_E**2 + mu * tr_E_squared
        
        return W

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):
        """
        Compute the stress and elasticity tensor for large deformation linear elasticity.
        
        Uses the Saint Venant-Kirchhoff model with 2nd Piola-Kirchhoff stress tensor.
        
        Args:
            F: Deformation gradient
            J: Jacobian determinant (optional)
            Jneg: J^(-2/3) (optional)
            invF: Inverse of F (optional)
            I1: First invariant (optional)
            
        Returns:
            Tuple of (stress tensor, elasticity tensor)
        """
        batch_size, elem_size = F.shape[0], F.shape[1]
        
        # Calculate necessary quantities if not provided
        if J is None:
            J = F.det().view(F.shape[0], F.shape[1], 1, 1)
        else:
            J = J.view(F.shape[0], F.shape[1], 1, 1)
        
        if invF is None:
            invF = F.inverse()
            
        # Right Cauchy-Green deformation tensor
        C = torch.einsum('geij,gejk->geik', F, F)
        
        # Identity tensor
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3).expand(batch_size, elem_size, 3, 3)
        
        # Green-Lagrange strain tensor
        E = 0.5 * (C - I_tensor)
        
        # Trace of Green-Lagrange strain tensor
        tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1).view(batch_size, elem_size, 1, 1)
        
        # Reshape Lamé parameters for broadcasting
        lambda_ = self.lambda_.view(-1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        mu = self.mu.view(-1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        
        # 2nd Piola-Kirchhoff stress tensor for Saint Venant-Kirchhoff model
        # S = lambda * tr(E) * I + 2 * mu * E
        S = lambda_ * tr_E * I_tensor + 2 * mu * E
        
        # Elasticity tensor (material tangent) in reference configuration
        # C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
        C_mat = torch.zeros([batch_size, elem_size, 3, 3, 3, 3], device=F.device, dtype=F.dtype)
        
        # Create efficient vectorized implementation for the elasticity tensor
        # First term: lambda * delta_ij * delta_kl
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        delta_ij = 1.0 if i == j else 0.0
                        delta_kl = 1.0 if k == l else 0.0
                        delta_ik = 1.0 if i == k else 0.0
                        delta_jl = 1.0 if j == l else 0.0
                        delta_il = 1.0 if i == l else 0.0
                        delta_jk = 1.0 if j == k else 0.0
                        
                        C_mat[..., i, j, k, l] = lambda_.squeeze() * delta_ij * delta_kl + \
                                             mu.squeeze() * (delta_ik * delta_jl + delta_il * delta_jk)
        
        return S, C_mat


