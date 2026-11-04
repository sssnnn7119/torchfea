"""Native CAD meshing reuses CPU topology without per-face Torch transfers."""

import numpy as np
import pytest

from torchfea.cad import CADHistory
from torchfea.model.elements import Element_3D


@pytest.mark.parametrize("shape", ["box", "cylinder"])
def test_exterior_matches_connectivity_without_torch_face_scan(shape, monkeypatch):
    history = CADHistory()
    if shape == "box":
        history.add_box(dx=3, dy=2, dz=2)
    else:
        history.add_cylinder(radius=2, dz=3)
    reference_extract = Element_3D.extract_boundary_surface_set

    def reject_scan(self):
        pytest.fail("CAD meshing must reuse its CPU face index.")

    monkeypatch.setattr(Element_3D, "extract_boundary_surface_set", reject_scan)
    part = history.mesh(1.0)
    element = part.elems["C3D4"]
    expected = reference_extract(element)
    actual = part.surfaces["extern"]
    assert [side for ids, side in actual] == [side for ids, side in expected]
    for (actual_ids, side), (expected_ids, expected_side) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_ids, expected_ids)
    exterior = {(int(element_id), side) for ids, side in actual for element_id in ids}
    cad_faces = {
        (int(element_id), side)
        for groups in part._gmsh_surface_sets.values()
        for ids, side in groups for element_id in ids
    }
    assert exterior == cad_faces
