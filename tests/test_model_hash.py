import numpy as np
import pytest
import torch

from torchfea.interfaces import Serializable
from torchfea.model.loads.pressure import Pressure
from torchfea.model_hash import (
    controller_model_hash,
    controller_result_compatibility_hash,
)
from torchfea.model.assembly import WorkCondition
from torchfea.solver.static.result import StaticResult


class _HashFixtureController(Serializable):
    def __init__(self, pressure=10.0, surface_set="top", mesh_signature=(0, 1)):
        self.loads = {
            "pressure": Pressure(
                instance_name="Part-1-1",
                surface_set=surface_set,
                pressure=pressure,
            )
        }
        self.mesh_signature = np.asarray(mesh_signature, dtype=np.int64)


def test_compatibility_hash_ignores_load_parameters_but_strict_hash_does_not():
    original = _HashFixtureController(pressure=10.0)
    changed_pressure = _HashFixtureController(pressure=25.0)

    assert controller_model_hash(original) != controller_model_hash(changed_pressure)
    assert controller_result_compatibility_hash(original) == (
        controller_result_compatibility_hash(changed_pressure)
    )


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [("surface_set", "bottom"), ("mesh_signature", (0, 1, 2))],
)
def test_compatibility_hash_keeps_load_target_and_model_structure(
    changed_field, changed_value
):
    original = _HashFixtureController()
    changed = _HashFixtureController()
    if changed_field == "surface_set":
        changed.loads["pressure"].surface_set = changed_value
    else:
        changed.mesh_signature = np.asarray(changed_value, dtype=np.int64)

    assert controller_result_compatibility_hash(original) != (
        controller_result_compatibility_hash(changed)
    )


def test_static_result_round_trips_compatibility_hash(tmp_path):
    result = StaticResult(
        GC=torch.tensor([0.0, 1.0]),
        work_conditions=WorkCondition(),
        model_hash="strict-hash",
        compatibility_hash="compatible-hash",
    )
    result_path = tmp_path / "result.npz"

    result.save(str(result_path))
    loaded = StaticResult.load(str(result_path))

    assert loaded.model_hash == "strict-hash"
    assert loaded.compatibility_hash == "compatible-hash"
