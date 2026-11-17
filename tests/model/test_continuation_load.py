from types import MethodType, SimpleNamespace

import torch

from torchfea.model.loads import ContinuousLoad, Penalty_DoF
from torchfea.solver.static import StaticImplicitSolver


class ToyContinuousLoad(ContinuousLoad):
    def _get_stiffness(self, RGC, if_onlyforce=False, *args, **kwargs):
        force_indices = torch.tensor([0], dtype=torch.int64)
        force_values = torch.tensor([4.0], dtype=RGC[0].dtype)
        if if_onlyforce:
            return force_indices, force_values
        K_indices = torch.tensor([[0], [0]], dtype=torch.int64)
        K_values = torch.tensor([7.0], dtype=RGC[0].dtype)
        return force_indices, force_values, K_indices, K_values

    def _get_potential_energy(self, RGC):
        return torch.as_tensor(3.0, dtype=RGC[0].dtype)


def test_continuous_load_scales_force_tangent_and_energy_together():
    load = ToyContinuousLoad()
    rgc = [torch.zeros(1, dtype=torch.float64)]

    load.continuation_factor = 0.25
    force_indices, force_values, K_indices, K_values = load.get_stiffness(rgc)
    assert force_indices.tolist() == [0]
    assert K_indices.tolist() == [[0], [0]]
    assert torch.equal(force_values, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(K_values, torch.tensor([1.75], dtype=torch.float64))
    assert load.get_potential_energy(rgc).item() == 0.75


def test_penalty_dof_is_a_continuous_load():
    assert issubclass(Penalty_DoF, ContinuousLoad)


def test_legacy_solver_archive_gets_continuation_defaults_during_load():
    solver = StaticImplicitSolver()
    serialized, class_name = solver._serialize()
    for name in (
        "initial_increment",
        "maximum_increment",
        "minimum_increment",
        "cutback_factor",
        "growth_factor",
        "maximum_cutbacks",
    ):
        serialized.pop(name, None)

    loaded = StaticImplicitSolver._deserialize((serialized, class_name))
    assert loaded.initial_increment == 1.0
    assert loaded.maximum_increment == 1.0
    assert loaded.minimum_increment == 1e-4
    assert loaded.cutback_factor == 0.5
    assert loaded.growth_factor == 1.5
    assert loaded.maximum_cutbacks == 12


def test_cutback_retries_from_last_accepted_state_without_history_output():
    load = ToyContinuousLoad()
    solver = StaticImplicitSolver(
        initial_increment=1.0,
        maximum_increment=1.0,
        minimum_increment=0.25,
        cutback_factor=0.5,
        growth_factor=1.5,
        maximum_cutbacks=4,
    )
    solver.assembly = SimpleNamespace(_loads={"load": load})
    calls = []

    def fake_iteration(self, GC, tol_error):
        factor = float(load.continuation_factor)
        calls.append((factor, GC.clone()))
        if len(calls) == 1:
            return GC + 100.0, 0.0, {}, False
        return GC + factor, 0.0, {}, True

    solver._solve_iteration = MethodType(fake_iteration, solver)
    output = solver._solve_with_cutback(torch.zeros(1), tol_error=1e-6)
    GC, _, _, converged, factor = output

    assert converged
    assert factor == 1.0
    assert [factor for factor, _ in calls] == [1.0, 0.5, 1.0]
    # The final retry starts from the accepted state produced at factor 0.5,
    # not from the discarded failed trial.
    assert calls[-1][1].item() == 0.5
    assert GC.item() == 1.5
