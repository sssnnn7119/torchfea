"""Adjoint derivatives retain small nonzero load-Jacobian contributions."""
import pickle
from types import SimpleNamespace

import pytest
import torch

from torchfea.solver.static import StaticImplicitSolver, StaticResult


class CubicSpring:
    """One equilibrium DOF with R = d*u + u**3 - p."""

    def __init__(self, design: torch.Tensor) -> None:
        self.design = design
        self._loads = {'p': SimpleNamespace(_parameters=design.new_zeros(1))}

    def initialize(self) -> None:
        pass

    def set_work_conditions(self, pressure: float) -> None:
        self._loads['p']._parameters = self.design.new_tensor([pressure])

    def assemble_force(self, GC: torch.Tensor) -> torch.Tensor:
        return self.design * GC + GC**3 - self._loads['p']._parameters

    def assemble_Stiffness_Matrix(self, GC: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.tensor([[0], [0]], dtype=torch.int64)
        return self.assemble_force(GC), indices, self.design + 3 * GC**2


def equilibrium(design: float, pressure: float) -> float:
    u = pressure / design
    for _ in range(25):
        u -= (design*u + u**3 - pressure) / (design + 3*u*u)
    return u


def test_factorized_result_solves_multiple_rhs_and_rebuilds_after_pickle():
    assembly = CubicSpring(torch.tensor([2.0]))
    result = StaticResult(GC=torch.tensor([1.0]), work_conditions=3.0)
    rhs = torch.tensor([[5.0, 10.0]]).numpy()
    result.factorize_stiffness_matrix(assembly)
    assert (result.K_solver.solve(rhs) == [[1.0, 2.0]]).all()

    restored = pickle.loads(pickle.dumps(result))
    assert not restored.if_factorized
    restored.factorize_stiffness_matrix(assembly)
    assert (restored.K_solver.solve(rhs[:, :1]) == [[1.0]]).all()
    restored.remove_stored_factorization()
    assert restored.K_solver is restored.K_sp is None


def sensitivity(scale: float, pressures: list[float], *, kind: str = "jacobian") -> float:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        design = torch.tensor([2.0])
        assembly = CubicSpring(design)
        solver = StaticImplicitSolver()
        solver.assembly = assembly
        results = []
        for pressure in pressures:
            u = equilibrium(2.0, pressure)
            results.append(StaticResult(
                GC=torch.tensor([u]), work_conditions=pressure,
                jacobian={'p': torch.tensor([[1 / (2 + 3*u*u)]])},
            ))

        def apply(model: CubicSpring, values: torch.Tensor) -> None:
            model.design = values

        def objective(values: list[StaticResult], model: CubicSpring) -> torch.Tensor:
            terms = []
            for result in values:
                displacement = result.GC.sum()
                jacobian = result.jacobian['p'].sum()
                terms.append(displacement if kind == 'displacement' else
                             displacement * jacobian if kind == 'mixed' else jacobian)
            return scale * sum(terms)

        actual = solver.get_jacobian_sensitivity_multistep(
            fe_results=results, design_vars=design, load_names=['p'],
            apply_func=apply, compute_objective_funcs=objective,
        )
        return actual.item()
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('scale', [1.0, 1e-7, 2e-8, 1e-12, 1e-16])
@pytest.mark.parametrize('pressures', [[0.864], [0.5, 1.5]])
def test_load_jacobian_gradient_scales_with_the_objective(scale, pressures):
    expected = 0.0
    for pressure in pressures:
        u = equilibrium(2.0, pressure)
        stiffness = 2 + 3*u*u
        expected -= (2 - 3*u*u) / stiffness**3
    actual = sensitivity(scale, pressures)
    assert actual / scale == pytest.approx(expected, rel=1e-10, abs=0)
    epsilon = 1e-5
    def value(design):
        return sum(1 / (design + 3*equilibrium(design, p)**2) for p in pressures)
    finite_difference = (value(2+epsilon) - value(2-epsilon)) / (2*epsilon)
    assert actual / scale == pytest.approx(finite_difference, rel=1e-8, abs=0)


@pytest.mark.parametrize('scale', [1.0, 1e-12])
def test_displacement_objective_keeps_exactly_zero_jacobian_partials_optional(scale):
    pressure = 0.864
    u = equilibrium(2.0, pressure)
    actual = sensitivity(scale, [pressure], kind='displacement')
    assert actual / scale == pytest.approx(-u / (2+3*u*u), rel=1e-10, abs=0)


@pytest.mark.parametrize('scale', [1.0, 1e-12])
@pytest.mark.parametrize('pressures', [[0.864], [0.5, 1.5]])
def test_mixed_displacement_jacobian_objective_matches_finite_differences(scale, pressures):
    expected = sum(-4 * equilibrium(2.0, p) / (2 + 3*equilibrium(2.0, p)**2)**3
                   for p in pressures)
    actual = sensitivity(scale, pressures, kind='mixed')
    assert actual / scale == pytest.approx(expected, rel=1e-10, abs=0)
    epsilon = 1e-5
    def value(design):
        return sum(equilibrium(design, p) / (design + 3*equilibrium(design, p)**2)
                   for p in pressures)
    finite_difference = (value(2+epsilon) - value(2-epsilon)) / (2*epsilon)
    assert actual / scale == pytest.approx(finite_difference, rel=1e-8, abs=0)
