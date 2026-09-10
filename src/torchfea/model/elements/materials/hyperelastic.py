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

        self._mu = mu
        self._kappa = kappa

    @property
    def mu(self) -> torch.Tensor:
        return self._mu
    
    @mu.setter
    def mu(self, value: torch.Tensor | float) -> None:
        if type(value) == float:
            value = torch.tensor([value], dtype=torch.float32)
        self._mu = value

    @property
    def kappa(self) -> torch.Tensor:
        return self._kappa
    
    @kappa.setter
    def kappa(self, value: torch.Tensor | float) -> None:
        if type(value) == float:
            value = torch.tensor([value], dtype=torch.float32)
        self._kappa = value

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

        if self.mu.dim() == 0 or self.mu.numel() == 1:
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
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3],
                         device=F.device, dtype=F.dtype)

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
                C[:, :, m, n, n, m] += muJneg[:, :, 0, 0]

        return s, C


class NeoHookeanLnJ(NeoHookean):

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None):

        
        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)
        W = self.mu    / 2 * (I1 - 3) + \
            self.kappa / 2 * (torch.log(J))**2
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

        if self.mu.dim() == 0 or self.mu.numel() == 1:
            mu = self.mu.view(1, 1, 1, 1)
            kappa = self.kappa.view(1, 1, 1, 1)
        else:
            mu = self.mu.view(self.mu.shape[0], self.mu.shape[1], 1, 1)
            kappa = self.kappa.view(self.kappa.shape[0], self.kappa.shape[1], 1, 1)

        muJneg = mu * Jneg
        FtMuJneg = F.transpose(-1, -2) * muJneg
        muI1invF = mu * I1 * invF

        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3],
                         device=F.device, dtype=F.dtype)

        s = FtMuJneg + -1 / 3 * muI1invF + kappa * torch.log(J) * invF

        C = torch.einsum(
            'geij,gelk->geijkl',
            -2 / 3 * FtMuJneg + kappa * invF + 2 / 9 * muI1invF,
            invF)

        C += torch.einsum('geij,gekl->geijkl',
                                         -2 / 3 * muJneg * invF, F)
        C += torch.einsum('geik,gelj->geijkl',
                                         (1 / 3 * muI1invF - kappa * torch.log(J) *
                                          invF), invF)

        for m in range(3):
            for n in range(3):
                C[:, :, m, n, n, m] += muJneg[:, :, 0, 0]

        return s, C
