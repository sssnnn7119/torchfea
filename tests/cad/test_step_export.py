from pathlib import Path
import math

import gmsh
import pytest

from torchfea.cad import Box, CADHistory, Cut, Cylinder, Translate, export_step


@pytest.mark.parametrize("suffix", [".stp", ".step"])
def test_step_export_round_trips_final_boolean_and_transformed_solids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    history = CADHistory([
        Box("block", dx=10, dy=8, dz=3),
        Cylinder("tool", x=5, y=4, radius=1, dz=3),
        Cut("hole", object_id="block", tool_id="tool"),
        Translate("move", input_id="hole", dx=20, dy=-2, dz=5),
        Box("separate", x=40, dx=2, dy=3, dz=4),
    ])
    output = tmp_path / f"零件{suffix}"

    # A CAD-only export must work before meshing. Fail if an FE or preview
    # mesh is requested, then read the STEP through the real OCC importer.
    def reject_meshing(*args: object, **kwargs: object) -> None:
        raise AssertionError("STEP export must not generate a mesh.")

    monkeypatch.setattr(gmsh.model.mesh, "generate", reject_meshing)
    export_step(history, str(output))
    assert output.read_text().startswith("ISO-10303-21;")
    assert not gmsh.isInitialized()

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("step-round-trip")
        gmsh.model.occ.importShapes(str(output))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        assert len(volumes) == 2
        masses = sorted(gmsh.model.occ.getMass(3, tag) for _, tag in volumes)
        assert masses == pytest.approx([24, 240 - 3 * math.pi])
        moved = max(volumes, key=lambda entity: gmsh.model.occ.getMass(*entity))
        assert gmsh.model.occ.getCenterOfMass(*moved) == pytest.approx([25, 2, 6.5])
    finally:
        gmsh.finalize()
