import math
from pathlib import Path

import gmsh
import pytest

from torchfea import FEAController, load_model
from torchfea.cad import CADHistory, StepImport, export_step
from torchfea.model import Assembly, Instance
from torchfea.model.elements.materials import LinearElastic


@pytest.mark.parametrize("suffix", [".stp", ".step"])
def test_imported_feature_survives_model_save_and_participates_in_boolean(
    tmp_path: Path, suffix: str,
) -> None:
    # Create a real external solid, then append it after an existing feature
    # using the controller's public CAD API.
    source = CADHistory()
    source.add_cylinder(x=5, y=4, radius=1, dz=3)
    step_path = tmp_path / f"孔工具{suffix}"
    export_step(source, str(step_path))
    controller = FEAController()
    history = controller.cad.add_part("plate")
    block = history.add_box(dx=10, dy=8, dz=3)
    imported = controller.cad.import_step("plate", str(step_path))
    cut = history.cut(block.id, imported.id)
    history.translate(cut.id, dx=20, dy=-2, dz=5)
    assert history.operations[1] is imported

    # The native model must contain the STEP bytes, rather than depending on
    # a path that happens to exist on the machine where it was imported.
    step_path.unlink()
    part = history.mesh(2)
    for element in part.elems.values():
        element.set_materials(LinearElastic(1000, 0.3))
        element.density = 1.0
    controller.assembly = Assembly()
    controller.assembly.add_part(part, "plate")
    controller.assembly.add_instance(Instance("plate"), "plate-1")
    model_path = tmp_path / "model.npz"
    controller.save_model(str(model_path), if_save_source_code=False)
    restored = load_model(str(model_path)).cad.get_part("plate")
    restored_import = restored.operations[1]
    assert isinstance(restored_import, StepImport)
    assert restored_import.filename == imported.filename
    assert restored_import.data == imported.data
    output = tmp_path / "result.stp"
    export_step(restored, str(output))

    # Check the rebuilt solid's geometry after import, cut, transform and
    # save/load, including the absence of an extra copy of the cutting tool.
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("imported-feature")
        gmsh.model.occ.importShapes(str(output))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        assert len(volumes) == 1
        assert gmsh.model.occ.getMass(*volumes[0]) == pytest.approx(240 - 3 * math.pi)
        assert gmsh.model.occ.getCenterOfMass(*volumes[0]) == pytest.approx([25, 2, 6.5])
    finally:
        gmsh.finalize()
