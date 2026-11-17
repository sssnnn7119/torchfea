from pathlib import Path

import numpy as np
import pytest

import torchfea
from torchfea.inp import FEA_INP, RawRecord, Surface, SurfaceEntry, Boundary


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_INPUTS = sorted((ROOT / "examples").rglob("*.inp"))


@pytest.mark.parametrize("source", EXAMPLE_INPUTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_all_existing_inputs_round_trip_byte_for_byte(source, tmp_path):
    inp = FEA_INP()
    inp.read_inp(source)
    output = tmp_path / "copy.inp"
    inp.write_inp(output)
    assert output.read_bytes() == source.read_bytes()


def test_unknown_keywords_comments_encoding_and_line_endings_survive(tmp_path):
    original = (
        b"\xef\xbb\xbf** Comment with undecodable legacy byte \xff\r\n"
        b"*hEaDiNg\r\nA custom model\n\r\n"
        b"*Material, name=UserMaterial\r\n"
        b"*User Material, constants=10,\r\n unsymm\r\n"
        b"1,2,3,4,5,6,7,8\r\n9,10\r\n"
        b"** Preserve this comment\r\n"
        b"*Include, input=\"external, mesh.inp\"\r\n"
        b"*Step, name=Load\r\n*Static\r\n.1,1,1e-8,1\r\n"
        b"*Boundary\r\nRP,1,1,.02\r\n"
        b"*Unsupported Vendor Keyword, Flag, arbitrary=yes\r\n"
        b"not necessarily numeric, <PARAMETER>\r\n"
        b"*Output, field\r\n*Node Output\r\nU,RF\r\n*End Step"
    )
    source = tmp_path / "original.inp"
    source.write_bytes(original)
    inp = FEA_INP()
    inp.read_inp(source)
    assert isinstance(inp.find("USER MATERIAL")[0], RawRecord)
    assert 'input="external, mesh.inp"' in inp.find("INCLUDE")[0].text
    assert inp.steps[0].name == "Load"
    condition = inp.steps[0].boundaries[0].values[0]
    assert condition.target == "RP"
    assert condition.value == .02
    output = tmp_path / "copy.inp"
    inp.write_inp(output)
    assert output.read_bytes() == original


def test_editing_a_keyword_preserves_unrelated_unsupported_physics():
    text = "** keep\r\n*Step, name=one\r\n*Static\r\n.1,1\r\n*Cload\r\n1,1,4\r\n** load\r\n*Exotic\r\nopaque data\r\n*End Step"
    inp = FEA_INP.from_text(text)
    inp.steps[0].concentrated_loads[0].values[0].magnitude = 9.
    inp.steps[0].definition.nlgeom = True
    result = inp.to_text()
    assert result == text.replace("*Step, name=one", "*STEP, name=one, nlgeom=YES, inc=100").replace("*Cload", "*CLOAD").replace("1,1,4", "1, 1, 9")
    assert FEA_INP.from_text(result).steps[0].definition.nlgeom


def test_sparse_labels_repeated_blocks_sections_and_direct_surface_labels():
    inp = FEA_INP.from_text(
        "*Part, name=Block\n"
        "*Node, nset=first\n10,0,0,0\n30,1,0,0\n"
        "*Node, nset=second\n50,0,1,0\n90,0,0,1\n"
        "*Element, type=C3D4, elset=solid\n1001,10,30,50,90\n"
        "*Element, type=C3D4, elset=solid\n2001,10,30,50,90\n"
        "*Solid Section, material=Steel, elset=solid\n,\n"
        "*Surface, name=face, type=ELEMENT\n2001,S2\n"
        "*End Part\n*Material, name=Steel\n*Elastic\n1000,.3\n"
    )
    part = inp.part[0]
    assert part.name == "Block"
    assert part.node_set_labels("second").tolist() == [50, 90]
    assert part.element_set_labels("solid").tolist() == [1001, 2001]
    assert part.sections[0].material == "Steel"
    assert part.surfaces[0].entries[0].target == "2001"
    assert inp.material[0].elastic.young_modulus == 1000.
    controller = torchfea.from_inp(inp)
    block = controller.assembly._parts["Block"].elems["C3D4"]
    assert block._elems.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert block._elems_index.tolist() == [1000, 2000]


def test_parameterized_mesh_can_be_read_and_written_without_numeric_evaluation():
    text = "*Parameter\nE=200000\n*Part,name=P\n*Node\n1,<X>,0,0\n*Nset,nset=generated,generate\n<START>,<END>,1\n*End Part\n*Material,name=M\n*Elastic\n<E>,.3\n"
    model = FEA_INP.from_text(text)
    assert model.to_text() == text
    assert isinstance(model.find("NODE")[0], RawRecord)
    assert "<X>" in model.find("NODE")[0].text


def test_new_keyword_models_use_the_same_writer_and_quotes():
    model = FEA_INP()
    model.records.append(Surface(name="a,b", entries=[SurfaceEntry("2", "S1")]))
    model.refresh_views()
    assert model.mesh.surfaces[0].name == "a,b"
    assert FEA_INP.from_text(model.to_text()).mesh.surfaces[0].name == "a,b"
