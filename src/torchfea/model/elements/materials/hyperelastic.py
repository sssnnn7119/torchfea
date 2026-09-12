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


class _InvariantHyperelastic(Materials_Base):
    """Base for compressible isotropic models using modified invariants.

    The energy is written as ``W(I1bar, I2bar, J)`` where ``I1bar`` and
    ``I2bar`` are the first and second invariants of the isochoric right
    Cauchy--Green tensor.  Subclasses only provide the energy and its first
    and second derivatives with respect to these three scalar arguments.

    The implementation returns the same quantities as the legacy material
    classes: ``s = P.T`` and the material/geometric tangent in the index order
    expected by ``C3base.py``.  The tangent is analytical and therefore works
    inside the solver's ``torch.no_grad()`` region.
    """

    type = 1

    @staticmethod
    def _as_parameter(value: torch.Tensor | float) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.tensor([value], dtype=torch.get_default_dtype())

    @staticmethod
    def _parameter_field(value: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar, Gaussian-wise, or element-wise parameter."""
        value = value.to(device=F.device, dtype=F.dtype)
        g, e = F.shape[:2]
        if value.dim() == 0 or value.numel() == 1:
            return value.reshape(1, 1).expand(g, e)
        if value.dim() == 1:
            if value.shape[0] == g:
                return value.reshape(g, 1).expand(g, e)
            if value.shape[0] == e:
                return value.reshape(1, e).expand(g, e)
        if value.shape == (g, e):
            return value
        raise ValueError(
            f"Cannot broadcast parameter with shape {tuple(value.shape)} "
            f"to deformation field [{g}, {e}]"
        )

    def _energy_derivatives(self,
                            I1bar: torch.Tensor,
                            I2bar: torch.Tensor,
                            J: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return W and its first/second invariant derivatives.

        The return order is ``W, W1, W2, WJ, W11, W12, W22, W1J, W2J,
        WJJ``.  ``W1J`` and ``W2J`` are included so that models with coupled
        isochoric/volumetric terms can be added without changing the tangent
        implementation.
        """
        raise NotImplementedError

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None,
                                 I1: torch.Tensor = None,
                                 J: torch.Tensor = None):
        if J is None:
            J = F.det()

        C = torch.matmul(F.transpose(-1, -2), F)
        trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        I2_C = 0.5 * (trace_C**2 - torch.einsum('...ij,...ji->...', C, C))
        I1bar = trace_C * J**(-2 / 3)
        I2bar = I2_C * J**(-4 / 3)
        return self._energy_derivatives(I1bar, I2bar, J)[0]

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):
        if J is None:
            J = F.det()
        if invF is None:
            invF = F.inverse()

        C = torch.matmul(F.transpose(-1, -2), F)
        Cinv = torch.matmul(invF, invF.transpose(-1, -2))
        trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        I2_C = 0.5 * (trace_C**2 - torch.einsum('...ij,...ji->...', C, C))
        q = J**(-2 / 3)
        q2 = q**2
        I1bar = trace_C * q
        I2bar = I2_C * q2

        (W, W1, W2, WJ, W11, W12, W22, W1J, W2J,
         WJJ) = self._energy_derivatives(I1bar, I2bar, J)

        eye = torch.eye(3, device=F.device, dtype=F.dtype)
        eye22 = eye.reshape(1, 1, 3, 3)

        # Derivatives of the three scalar invariants with respect to C.
        I1_C = q[..., None, None] * (
            eye22 - trace_C[..., None, None] * Cinv / 3.0)
        I2_C_tensor = q2[..., None, None] * (
            trace_C[..., None, None] * eye22 - C
            - 2.0 * I2_C[..., None, None] * Cinv / 3.0)
        J_C = 0.5 * J[..., None, None] * Cinv

        A = (
            W1[..., None, None] * I1_C
            + W2[..., None, None] * I2_C_tensor
            + WJ[..., None, None] * J_C
        )
        s = 2.0 * torch.matmul(A, F.transpose(-1, -2))

        # Differentiate C, J, and C^{-1} in every F_{kL} direction.
        dC = (
            torch.einsum('al,gekb->geklab', eye, F)
            + torch.einsum('geka,bl->geklab', F, eye)
        )
        dJ = J[..., None, None] * invF.transpose(-1, -2)
        dtrace_C = torch.diagonal(dC, dim1=-2, dim2=-1).sum(-1)
        dI2_C = torch.einsum(
            'geab,geklab->gekl',
            trace_C[..., None, None] * eye22 - C,
            dC,
        )
        dq = -2.0 / 3.0 * q[..., None, None] * dJ / J[..., None, None]
        dq2 = -4.0 / 3.0 * q2[..., None, None] * dJ / J[..., None, None]
        dI1bar = q[..., None, None] * dtrace_C + trace_C[..., None, None] * dq
        dI2bar = q2[..., None, None] * dI2_C + I2_C[..., None, None] * dq2
        dCinv = -torch.einsum('geam,geklmn,genb->geklab', Cinv, dC, Cinv)

        # Insert the two F-derivative axes before the C-tensor axes when
        # combining a [g,e,3,3] tensor with a [g,e,k,l,3,3] derivative.
        eye_kl = eye.reshape(1, 1, 1, 1, 3, 3)
        Cinv_kl = Cinv[..., None, None, :, :]

        dI1_C = (
            dq[..., None, None] * (
                eye22 - trace_C[..., None, None] * Cinv / 3.0
            )[..., None, None, :, :]
            + q[..., None, None, None, None] * (
                -dtrace_C[..., None, None] * Cinv_kl / 3.0
                - trace_C[..., None, None, None, None] * dCinv / 3.0
            )
        )
        dI2_C_tensor = (
            dq2[..., None, None] * (
                trace_C[..., None, None] * eye22 - C
                - 2.0 * I2_C[..., None, None] * Cinv / 3.0
            )[..., None, None, :, :]
            + q2[..., None, None, None, None] * (
                dtrace_C[..., None, None] * eye_kl - dC
                - 2.0 * dI2_C[..., None, None] * Cinv_kl / 3.0
                - 2.0 * I2_C[..., None, None, None, None] * dCinv / 3.0
            )
        )
        dJ_C = (
            0.5 * dJ[..., None, None] * Cinv[..., None, None, :, :]
            + 0.5 * J[..., None, None, None, None] * dCinv
        )

        dW1 = (
            W11[..., None, None] * dI1bar
            + W12[..., None, None] * dI2bar
            + W1J[..., None, None] * dJ
        )
        dW2 = (
            W12[..., None, None] * dI1bar
            + W22[..., None, None] * dI2bar
            + W2J[..., None, None] * dJ
        )
        dWJ = (
            W1J[..., None, None] * dI1bar
            + W2J[..., None, None] * dI2bar
            + WJJ[..., None, None] * dJ
        )
        dA = (
            dW1[..., None, None] * I1_C[..., None, None, :, :]
            + W1[..., None, None, None, None] * dI1_C
            + dW2[..., None, None] * I2_C_tensor[..., None, None, :, :]
            + W2[..., None, None, None, None] * dI2_C_tensor
            + dWJ[..., None, None] * J_C[..., None, None, :, :]
            + WJ[..., None, None, None, None] * dJ_C
        )

        # dA has [g,e,k,L,A,B]. Convert d(2 A F.T)/dF to [g,e,A,i,k,L].
        dA_Ft = torch.einsum('geklab,geib->geklai', dA, F)
        dA_Ft = dA_Ft.permute(0, 1, 4, 5, 2, 3)
        direct = torch.einsum('geab,bl,ik->geaikl', A, eye, eye)
        C_tangent = 2.0 * (dA_Ft + direct)
        return s, C_tangent


class MooneyRivlin(_InvariantHyperelastic):
    """Compressible two-parameter Mooney--Rivlin material.

    Parameters follow the standard form

    ``W = c10*(I1bar - 3) + c01*(I2bar - 3) + kappa/2*(J - 1)**2``.
    ``c10`` and ``c01`` have stress units, while ``kappa`` is the bulk
    modulus used by the existing Neo-Hookean implementation.
    """

    def __init__(self, c10: torch.Tensor | float,
                 c01: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:
        super().__init__()
        self.c10 = self._as_parameter(c10)
        self.c01 = self._as_parameter(c01)
        self.kappa = self._as_parameter(kappa)

    def _energy_derivatives(self, I1bar, I2bar, J):
        c10 = self._parameter_field(self.c10, I1bar[..., None, None])
        c01 = self._parameter_field(self.c01, I1bar[..., None, None])
        kappa = self._parameter_field(self.kappa, I1bar[..., None, None])
        zero = torch.zeros_like(I1bar)
        return (
            c10 * (I1bar - 3.0) + c01 * (I2bar - 3.0)
            + 0.5 * kappa * (J - 1.0)**2,
            c10, c01, kappa * (J - 1.0),
            zero, zero, zero, zero, zero, kappa,
        )


class Yeoh(_InvariantHyperelastic):
    """Compressible Yeoh third-order model.

    ``W = c1*x + c2*x**2 + c3*x**3 + kappa/2*(J - 1)**2`` with
    ``x = I1bar - 3``.
    """

    def __init__(self, c1: torch.Tensor | float,
                 c2: torch.Tensor | float,
                 c3: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:
        super().__init__()
        self.c1 = self._as_parameter(c1)
        self.c2 = self._as_parameter(c2)
        self.c3 = self._as_parameter(c3)
        self.kappa = self._as_parameter(kappa)

    def _energy_derivatives(self, I1bar, I2bar, J):
        c1 = self._parameter_field(self.c1, I1bar[..., None, None])
        c2 = self._parameter_field(self.c2, I1bar[..., None, None])
        c3 = self._parameter_field(self.c3, I1bar[..., None, None])
        kappa = self._parameter_field(self.kappa, I1bar[..., None, None])
        x = I1bar - 3.0
        zero = torch.zeros_like(I1bar)
        return (
            c1 * x + c2 * x**2 + c3 * x**3
            + 0.5 * kappa * (J - 1.0)**2,
            c1 + 2.0 * c2 * x + 3.0 * c3 * x**2,
            zero,
            kappa * (J - 1.0),
            2.0 * c2 + 6.0 * c3 * x,
            zero,
            zero,
            zero,
            zero,
            kappa,
        )


class Gent(_InvariantHyperelastic):
    """Compressible Gent model with a limiting-chain extensibility ``Jm``."""

    def __init__(self, mu: torch.Tensor | float,
                 Jm: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:
        super().__init__()
        self.mu = self._as_parameter(mu)
        self.Jm = self._as_parameter(Jm)
        self.kappa = self._as_parameter(kappa)

    def _energy_derivatives(self, I1bar, I2bar, J):
        mu = self._parameter_field(self.mu, I1bar[..., None, None])
        Jm = self._parameter_field(self.Jm, I1bar[..., None, None])
        kappa = self._parameter_field(self.kappa, I1bar[..., None, None])
        x = I1bar - 3.0
        remaining = 1.0 - x / Jm
        zero = torch.zeros_like(I1bar)
        return (
            -0.5 * mu * Jm * torch.log(remaining)
            + 0.5 * kappa * (J - 1.0)**2,
            0.5 * mu / remaining,
            zero,
            kappa * (J - 1.0),
            0.5 * mu / (Jm * remaining**2),
            zero,
            zero,
            zero,
            zero,
            kappa,
        )


class ArrudaBoyce(_InvariantHyperelastic):
    """Compressible five-term Arruda--Boyce eight-chain approximation.

    ``N`` is the number of statistical segments per chain.  The polynomial
    approximation is valid below the locking stretch; users should choose
    ``N`` sufficiently large for the expected deformation range.
    """

    def __init__(self, mu: torch.Tensor | float,
                 N: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:
        super().__init__()
        self.mu = self._as_parameter(mu)
        self.N = self._as_parameter(N)
        self.kappa = self._as_parameter(kappa)

    def _energy_derivatives(self, I1bar, I2bar, J):
        mu = self._parameter_field(self.mu, I1bar[..., None, None])
        N = self._parameter_field(self.N, I1bar[..., None, None])
        kappa = self._parameter_field(self.kappa, I1bar[..., None, None])
        coefficients = (
            0.5,
            1.0 / (20.0 * N),
            11.0 / (1050.0 * N**2),
            19.0 / (7000.0 * N**3),
            519.0 / (673750.0 * N**4),
        )
        c1, c2, c3, c4, c5 = coefficients
        zero = torch.zeros_like(I1bar)
        return (
            mu * (
                c1 * (I1bar - 3.0)
                + c2 * (I1bar**2 - 3.0**2)
                + c3 * (I1bar**3 - 3.0**3)
                + c4 * (I1bar**4 - 3.0**4)
                + c5 * (I1bar**5 - 3.0**5)
            )
            + 0.5 * kappa * (J - 1.0)**2,
            mu * (
                c1 + 2.0 * c2 * I1bar + 3.0 * c3 * I1bar**2
                + 4.0 * c4 * I1bar**3 + 5.0 * c5 * I1bar**4
            ),
            zero,
            kappa * (J - 1.0),
            mu * (
                2.0 * c2 + 6.0 * c3 * I1bar + 12.0 * c4 * I1bar**2
                + 20.0 * c5 * I1bar**3
            ),
            zero,
            zero,
            zero,
            zero,
            kappa,
        )


class Ogden(_InvariantHyperelastic):
    """Compressible Ogden material in principal isochoric stretches.

    The energy uses the common ``(mu_p, alpha_p)`` expansion

    ``W = sum_p 2*mu_p/alpha_p**2 *
    (sum_i lambda_bar_i**alpha_p - 3) + kappa/2*(J - 1)**2``.

    ``mu`` and ``alpha`` may be scalars or one-dimensional tensors containing
    the terms of the expansion.  The tangent uses spectral divided
    differences and their repeated-eigenvalue limit, so it remains finite at
    the reference configuration.
    """

    def __init__(self, mu: torch.Tensor | float,
                 alpha: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:
        super().__init__()
        self.mu = self._as_parameter(mu).reshape(-1)
        self.alpha = self._as_parameter(alpha).reshape(-1)
        self.kappa = self._as_parameter(kappa)
        if self.mu.numel() != self.alpha.numel():
            raise ValueError("Ogden mu and alpha must have the same number of terms")
        if torch.any(self.alpha == 0):
            raise ValueError("Ogden alpha terms must be non-zero")

    @staticmethod
    def _symmetric_eigenvalues_3x3(C: torch.Tensor) -> torch.Tensor:
        """Return eigenvalues of a symmetric 3x3 tensor without a LAPACK call.

        The implementation is the closed-form solution of the depressed
        cubic.  ``C`` is expected to have shape ``[..., 3, 3]``.  Sorting the
        three roots makes their order consistent with the eigensystem helper.
        """
        C = 0.5 * (C + C.transpose(-1, -2))
        dtype = C.dtype
        eye = torch.eye(3, device=C.device, dtype=dtype)
        trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        mean = trace / 3.0
        centered = C - mean[..., None, None] * eye
        p = torch.sqrt((centered * centered).sum(dim=(-1, -2)) / 6.0)

        scale = torch.maximum(
            C.abs().amax(dim=(-1, -2)),
            torch.ones_like(mean),
        )
        repeated = p <= (100.0 * torch.finfo(dtype).eps * scale)
        p_safe = torch.where(repeated, torch.ones_like(p), p)
        b00, b01, b02 = centered[..., 0, 0], centered[..., 0, 1], centered[..., 0, 2]
        b10, b11, b12 = centered[..., 1, 0], centered[..., 1, 1], centered[..., 1, 2]
        b20, b21, b22 = centered[..., 2, 0], centered[..., 2, 1], centered[..., 2, 2]
        determinant = (
            b00 * (b11 * b22 - b12 * b21)
            - b01 * (b10 * b22 - b12 * b20)
            + b02 * (b10 * b21 - b11 * b20)
        )
        r = determinant / (2.0 * p_safe**3)
        phi = torch.acos(r.clamp(-1.0, 1.0)) / 3.0

        roots = torch.stack((
            mean + 2.0 * p * torch.cos(phi),
            mean + 2.0 * p * torch.cos(phi + 2.0 * torch.pi / 3.0),
            mean + 2.0 * p * torch.cos(phi + 4.0 * torch.pi / 3.0),
        ), dim=-1)
        roots = torch.sort(roots, dim=-1).values
        return torch.where(
            repeated[..., None], mean[..., None].expand_as(roots), roots)

    @staticmethod
    def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        tiny = torch.finfo(vector.dtype).tiny
        norm = torch.sqrt((vector * vector).sum(dim=-1, keepdim=True))
        return vector / norm.clamp_min(tiny)

    @classmethod
    def _symmetric_eigensystem_3x3(cls, C: torch.Tensor):
        """Return an analytic eigensystem for a symmetric 3x3 tensor.

        For distinct roots, eigenvectors are null vectors constructed from
        cross products of rows of ``C - lambda*I``.  When two roots coincide,
        a deterministic orthonormal basis is selected in the degenerate
        eigenspace.  This avoids the large and variable CUDA workspace used
        by a general batched eigensolver.
        """
        C = 0.5 * (C + C.transpose(-1, -2))
        eigenvalues = cls._symmetric_eigenvalues_3x3(C)
        eye = torch.eye(3, device=C.device, dtype=C.dtype)
        matrix = (
            C.unsqueeze(-3)
            - eigenvalues[..., :, None, None] * eye
        )
        row0 = matrix[..., 0, :]
        row1 = matrix[..., 1, :]
        row2 = matrix[..., 2, :]
        candidates = torch.stack((
            torch.cross(row0, row1, dim=-1),
            torch.cross(row0, row2, dim=-1),
            torch.cross(row1, row2, dim=-1),
        ), dim=-2)
        candidate_norms = (candidates * candidates).sum(dim=-1)
        best_candidate = candidate_norms.argmax(dim=-1)
        gather_index = best_candidate[..., None, None].expand(
            *best_candidate.shape, 1, 3)
        vectors = candidates.gather(-2, gather_index).squeeze(-2)
        vectors = cls._normalize_vector(vectors)

        scale = torch.maximum(
            C.abs().amax(dim=(-1, -2)),
            torch.ones_like(eigenvalues[..., 0]),
        )
        tolerance = 100.0 * torch.finfo(C.dtype).eps * scale
        low_double = (
            (eigenvalues[..., 1] - eigenvalues[..., 0]).abs() <= tolerance)
        high_double = (
            (eigenvalues[..., 2] - eigenvalues[..., 1]).abs() <= tolerance)
        triple = low_double & high_double

        # lambda_2 is the unique root for a low double eigenvalue.
        unique_high = vectors[..., 2, :]
        axis_high = unique_high.abs().argmin(dim=-1)
        base_high = eye[axis_high]
        low_v0 = cls._normalize_vector(
            torch.cross(unique_high, base_high, dim=-1))
        low_v1 = cls._normalize_vector(
            torch.cross(unique_high, low_v0, dim=-1))
        low_basis = torch.stack((low_v0, low_v1, unique_high), dim=-1)

        # lambda_0 is the unique root for a high double eigenvalue.
        unique_low = vectors[..., 0, :]
        axis_low = unique_low.abs().argmin(dim=-1)
        base_low = eye[axis_low]
        high_v1 = cls._normalize_vector(
            torch.cross(unique_low, base_low, dim=-1))
        high_v2 = cls._normalize_vector(
            torch.cross(unique_low, high_v1, dim=-1))
        high_basis = torch.stack((unique_low, high_v1, high_v2), dim=-1)

        eigensystem = vectors.transpose(-1, -2)
        eigensystem = torch.where(
            low_double[..., None, None], low_basis, eigensystem)
        eigensystem = torch.where(
            high_double[..., None, None], high_basis, eigensystem)
        eigensystem = torch.where(
            triple[..., None, None], eye, eigensystem)
        return eigenvalues, eigensystem

    def _spectral_quantities(self, F: torch.Tensor, J: torch.Tensor):
        C = torch.matmul(F.transpose(-1, -2), F)
        eigenvalues, eigenvectors = self._symmetric_eigensystem_3x3(C)

        alpha = self.alpha.to(device=F.device, dtype=F.dtype)
        exponent = alpha.reshape(1, 1, 1, -1) / 2.0
        powers = eigenvalues[..., None]**exponent
        stretch_power_sum = powers.sum(dim=-2)
        q = J[..., None]**(-alpha.reshape(1, 1, -1) / 3.0)

        # H_p = d tr(C**(alpha_p/2))/dC in the principal basis.
        h = exponent * (eigenvalues[..., None]**(exponent - 1.0))
        h = h.permute(0, 1, 3, 2)
        H = torch.einsum('geai,gemi,gebi->gemab',
                         eigenvectors, h, eigenvectors)
        return C, eigenvalues, eigenvectors, stretch_power_sum, q, H

    def strain_energy_density_C3(self, F: torch.Tensor = None,
                                 I1: torch.Tensor = None,
                                 J: torch.Tensor = None):
        if J is None:
            J = F.det()
        C = torch.matmul(F.transpose(-1, -2), F)
        eigenvalues = self._symmetric_eigenvalues_3x3(C)
        alpha = self.alpha.to(device=F.device, dtype=F.dtype)
        exponent = alpha.reshape(1, 1, 1, -1) / 2.0
        stretch_power_sum = (
            eigenvalues[..., None]**exponent
        ).sum(dim=-2)
        q = J[..., None]**(-alpha.reshape(1, 1, -1) / 3.0)
        mu = self.mu.to(device=F.device, dtype=F.dtype)
        coefficients = 2.0 * mu / alpha**2
        kappa = self._parameter_field(self.kappa, F)
        return (
            (coefficients.reshape(1, 1, -1)
             * (q * stretch_power_sum - 3.0)).sum(dim=-1)
            + 0.5 * kappa * (J - 1.0)**2
        )

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):
        if J is None:
            J = F.det()
        if invF is None:
            invF = F.inverse()
        kappa = self._parameter_field(self.kappa, F)
        return self._material_Constitutive_C3_core(
            F=F, J=J, invF=invF, kappa=kappa)

    def _material_Constitutive_C3_core(self,
                                       F: torch.Tensor,
                                       J: torch.Tensor,
                                       invF: torch.Tensor,
                                       kappa: torch.Tensor):

        (C, eigenvalues, eigenvectors, stretch_power_sum, q,
         H) = self._spectral_quantities(F, J)
        Cinv = torch.matmul(invF, invF.transpose(-1, -2))
        mu = self.mu.to(device=F.device, dtype=F.dtype)
        alpha = self.alpha.to(device=F.device, dtype=F.dtype)
        coefficients = 2.0 * mu / alpha**2

        # A_iso = dW_iso/dC, computed term-by-term in the principal basis.
        alpha_modes = alpha.reshape(1, 1, -1, 1, 1)
        B_modes = (
            H - alpha_modes * stretch_power_sum[..., :, None, None]
            * Cinv[:, :, None, :, :] / 6.0
        )
        A_modes = coefficients.reshape(1, 1, -1, 1, 1) * q[..., :, None, None] * B_modes
        A = A_modes.sum(dim=-3)

        J_C = 0.5 * J[..., None, None] * Cinv
        WJ = kappa * (J - 1.0)
        A = A + WJ[..., None, None] * J_C
        s = 2.0 * torch.matmul(A, F.transpose(-1, -2))

        eye = torch.eye(3, device=F.device, dtype=F.dtype)
        dC = (
            torch.einsum('al,gekb->geklab', eye, F)
            + torch.einsum('geka,bl->geklab', F, eye)
        )
        dJ = J[..., None, None] * invF.transpose(-1, -2)
        dCinv = -torch.einsum('geam,geklmn,genb->geklab', Cinv, dC, Cinv)

        # Spectral derivative dH_p/dC.  The off-diagonal divided difference
        # is replaced by h'_p at coincident eigenvalues.
        eig_i = eigenvalues[..., :, None]
        eig_j = eigenvalues[..., None, :]
        eig_difference = eig_i - eig_j
        mode_p = (alpha / 2.0).reshape(1, 1, -1, 1, 1)
        h_modes = (
            (alpha / 2.0).reshape(1, 1, 1, -1)
            * eigenvalues[..., None]**(
                (alpha / 2.0).reshape(1, 1, 1, -1) - 1.0)
        ).permute(0, 1, 3, 2)
        h_i = h_modes[..., :, None]
        h_j = h_modes[..., None, :]
        midpoint = 0.5 * (eig_i + eig_j)
        h_prime_midpoint = (
            mode_p * (mode_p - 1.0)
            * midpoint[:, :, None, :, :]**(mode_p - 2.0)
        )
        scale = torch.maximum(
            eigenvalues.amax(dim=-1),
            torch.ones_like(eigenvalues[..., 0]),
        )
        repeated = eig_difference.abs() <= (100.0 * torch.finfo(F.dtype).eps * scale[..., None, None])
        safe_difference = torch.where(
            repeated, torch.ones_like(eig_difference), eig_difference)
        divided_difference = torch.where(
            repeated[..., None, :, :],
            h_prime_midpoint,
            (h_i - h_j) / safe_difference[..., None, :, :],
        )

        dC_eigen = torch.einsum(
            'geai,geklab,gebj->geklij', eigenvectors, dC, eigenvectors)
        dH_eigen = divided_difference[..., None, None, :, :] * dC_eigen[:, :, None, ...]
        dH = torch.einsum(
            'geai,gemklij,gebj->gemklab', eigenvectors, dH_eigen, eigenvectors)
        dH = dH.permute(0, 1, 3, 4, 2, 5, 6)
        dS = torch.einsum('gemab,geklab->gemkl', H, dC).permute(0, 1, 3, 4, 2)
        dq = (
            -alpha.reshape(1, 1, -1) / 3.0
            * q[..., None, None, :] * dJ[..., None]
            / J[..., None, None, None]
        )
        alpha_derivative = alpha.reshape(1, 1, 1, 1, -1, 1, 1)
        dB = (
            dH
            - alpha_derivative / 6.0 * (
                dS[..., None, None]
                * Cinv[:, :, None, None, None, :, :]
                + stretch_power_sum[:, :, None, None, :, None, None]
                * dCinv[:, :, :, :, None, :, :]
            )
        )
        dA_iso = (
            coefficients.reshape(1, 1, 1, 1, -1, 1, 1)
            * (
                dq[..., None, None] * B_modes[:, :, None, None, :, :, :]
                + q[:, :, None, None, :, None, None] * dB
            )
        ).sum(dim=-3)

        dJ_C = (
            0.5 * dJ[..., None, None] * Cinv[..., None, None, :, :]
            + 0.5 * J[..., None, None, None, None] * dCinv
        )
        dA = (
            dA_iso
            + (kappa[..., None, None] * dJ)[..., None, None]
            * J_C[..., None, None, :, :]
            + WJ[..., None, None, None, None] * dJ_C
        )

        dA_Ft = torch.einsum('geklab,geib->geklai', dA, F)
        dA_Ft = dA_Ft.permute(0, 1, 4, 5, 2, 3)
        direct = torch.einsum('geab,bl,ik->geaikl', A, eye, eye)
        C_tangent = 2.0 * (dA_Ft + direct)
        return s, C_tangent
