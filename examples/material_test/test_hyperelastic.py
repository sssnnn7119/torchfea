import torch

from torchfea.model.elements.materials import (
    ArrudaBoyce,
    Gent,
    MooneyRivlin,
    Ogden,
    Yeoh,
)


def _models():
    return (
        MooneyRivlin(c10=0.2, c01=0.1, kappa=4.5),
        Yeoh(c1=0.3, c2=0.01, c3=0.001, kappa=4.5),
        Gent(mu=0.6, Jm=20.0, kappa=4.5),
        ArrudaBoyce(mu=0.6, N=20.0, kappa=4.5),
        Ogden(mu=torch.tensor([0.3, 0.1]), alpha=torch.tensor([2.0, 5.0]), kappa=4.5),
    )


def test_common_hyperelastic_models_and_tangents():
    F0 = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
    F = torch.tensor(
        [[[[1.08, 0.03, 0.00],
           [0.00, 0.96, 0.02],
           [0.00, 0.00, 1.04]]]],
        dtype=torch.float64,
    )

    for material in _models():
        W0 = material.strain_energy_density_C3(F0)
        s0, C0 = material.material_Constitutive_C3(F0)
        s, C = material.material_Constitutive_C3(F)

        assert torch.allclose(W0, torch.zeros_like(W0))
        assert torch.allclose(s0, torch.zeros_like(s0))
        assert C0.shape == (1, 1, 3, 3, 3, 3)
        assert torch.isfinite(s).all()
        assert torch.isfinite(C).all()

        if isinstance(material, Ogden):
            numerical_C0 = torch.zeros_like(C0)
            epsilon = 1e-6
            for k in range(3):
                for l in range(3):
                    perturbation = torch.zeros_like(F0)
                    perturbation[..., k, l] = epsilon
                    stress_plus = material.material_Constitutive_C3(
                        F0 + perturbation)[0]
                    stress_minus = material.material_Constitutive_C3(
                        F0 - perturbation)[0]
                    numerical_C0[..., :, :, k, l] = (
                        stress_plus - stress_minus) / (2.0 * epsilon)
            assert torch.allclose(C0, numerical_C0, rtol=1e-5, atol=1e-7)

        # Check the analytical tangent against a central finite difference.
        numerical_C = torch.zeros_like(C)
        epsilon = 1e-6
        for k in range(3):
            for l in range(3):
                perturbation = torch.zeros_like(F)
                perturbation[..., k, l] = epsilon
                stress_plus = material.material_Constitutive_C3(
                    F + perturbation)[0]
                stress_minus = material.material_Constitutive_C3(
                    F - perturbation)[0]
                numerical_C[..., :, :, k, l] = (
                    stress_plus - stress_minus) / (2.0 * epsilon)

        assert torch.allclose(C, numerical_C, rtol=1e-5, atol=1e-7)

        # Also exercise the [gaussian, element, ...] broadcasting used by
        # C3D4 and the other 3-D elements.
        F_batch = F.expand(2, 3, -1, -1).clone()
        s_batch, C_batch = material.material_Constitutive_C3(F_batch)
        assert s_batch.shape == (2, 3, 3, 3)
        assert C_batch.shape == (2, 3, 3, 3, 3, 3)
        assert torch.isfinite(s_batch).all()
        assert torch.isfinite(C_batch).all()
