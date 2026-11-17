from pathlib import Path
import shutil
import subprocess
import runpy

import numpy as np
import pytest
import torch

import torchfea
from torchfea.model import boundarys, constraints, loads, materials
from torchfea.model.elements.dimension3 import C3D4


def controller_fixture():
    controller = torchfea.FEAController()
    assembly = controller.assembly = torchfea.Assembly()
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [0., 0., 1.], [1., 1., 1.]])
    part = torchfea.Part(nodes)
    block = C3D4(torch.tensor([10, 42]),
                  torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    block.set_materials(materials.LinearElastic(1000., .3))
    block.density = torch.tensor([2., 3.])
    part.add_element(block, "solid")
    part.set_nodes["base"] = np.array([0, 1, 2])
    part.set_nodes["tip"] = np.array([4])
    part.set_elements["all"] = np.array([10, 42])
    part.surfaces["loaded"] = [(np.array([42]), 2)]
    assembly.add_part(part, "part")
    assembly.add_instance(torchfea.Instance("part"), "instance")
    assembly.add_reference_point(torchfea.ReferencePoint([1., 1., 1.]), "rp")
    assembly.add_constraint(constraints.Couple("instance", "tip", "rp"), "couple")
    assembly.add_boundary(boundarys.Boundary_Condition("instance", "base"), "base")
    assembly.add_boundary(boundarys.Boundary_Condition_RP("rp", [3, 4, 5]), "rotation")
    assembly.add_load(loads.Concentrate_Force("rp", [3., 0., 0.]), "force1")
    assembly.add_load(loads.Concentrate_Force("rp", [2., 0., 0.]), "force2")
    assembly.add_load(loads.Pressure("instance", "loaded", 4.), "pressure")
    assembly.add_load(loads.BodyForce("instance", "solid", [0., -6., 0.]), "body")
    assembly.add_load(loads.Penalty_DoF("rp", 1, .2, 50.), "penalty")
    controller.solver = torchfea.solver.StaticImplicitSolver()
    return controller


def records(text):
    """Independent small keyword reader, including empty spring data lines."""
    result = []
    for line in text.splitlines():
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            fields = [field.strip().lower() for field in line[1:].split(",")]
            params = dict(field.split("=", 1) for field in fields[1:] if "=" in field)
            result.append((fields[0], params, []))
        elif result:
            result[-1][2].append([x.strip() for x in line.split(",")])
    return result


def of_kind(deck, keyword):
    return [(params, data) for kind, params, data in deck if kind == keyword]


def test_complete_job_export_without_initialization_or_model_mutation(tmp_path):
    controller = controller_fixture()
    before = controller.get_model_hash()
    path = tmp_path / "job.inp"
    controller.export_inp(path, nlgeom=False)
    deck = records(path.read_text())
    assert controller.get_model_hash() == before
    assert controller.assembly._instances["instance"]._RGC_index is None
    assert of_kind(deck, "heading")
    assert of_kind(deck, "step")[0][0] == {"name": "static", "nlgeom": "no", "inc": "1000"}
    assert of_kind(deck, "static")[0][1] == [["0.1", "1", "1e-06", "0.1"]]
    assert len(of_kind(deck, "solid section")) == 2  # different densities
    assert len(of_kind(deck, "material")) == 2
    assert [float(data[0][0]) for _, data in of_kind(deck, "density")] == [2., 3.]
    assert of_kind(deck, "kinematic")[0][1] == [["1", "3"]]
    assert of_kind(deck, "end step")
    assert of_kind(deck, "output")
    assert not of_kind(deck, "user material")

    nodes = {int(row[0]): np.array(row[1:], dtype=float)
             for _, data in of_kind(deck, "node") for row in data}
    assert len(nodes) == 6
    assert np.allclose(nodes[6], [1, 1, 1])
    solids = of_kind(deck, "element")[0][1]
    assert [int(row[0]) for row in solids] == [1, 2]
    assert all(int(node) in nodes for row in solids for node in row[1:])
    # Original element 42 becomes global element 2; face remains Abaqus S3.
    element_surfaces = [data for params, data in of_kind(deck, "surface")
                        if params["type"] == "element"]
    assert element_surfaces == [[["2", "S3"]]]
    cload = {tuple(map(int, row[:2])): float(row[2])
             for _, data in of_kind(deck, "cload") for row in data}
    assert cload == {(6, 1): 5., (6, 2): 10.}
    spring = of_kind(deck, "spring")[0][1]
    assert spring == [["2"], ["50"]]
    # SPRING1 + k*target has exactly the original final penalty residual/tangent.
    for q in [-.7, 0., .3]:
        assert 50 * q - cload[6, 2] == pytest.approx(50 * (q - .2))
    assert [row for _, data in of_kind(deck, "dload") for row in data] == [
        ["1", "BY", "-6"], ["2", "BY", "-6"]]
    assert of_kind(deck, "dsload")[0][1][0][1:] == ["P", "4"]


def test_controller_conversion_returns_the_shared_inp_model_and_round_trips(tmp_path):
    controller = controller_fixture()
    model = controller.to_inp(nlgeom=False)
    assert isinstance(model, torchfea.FEA_INP)
    assert len(model.steps) == 1
    assert len(model.steps[0].concentrated_loads) == 1
    assert any(block.element_type == "SPRING1" for block in model.mesh.elements)
    assert model.mesh.nodes.shape == (6, 4)
    settings = model.steps[0].static
    settings.initial_increment = .05
    settings.minimum_increment = 1e-7
    settings.maximum_increment = .2
    path = tmp_path / "converted.inp"
    model.write_inp(path)
    loaded = torchfea.FEA_INP()
    loaded.read_inp(path)
    copy = tmp_path / "copy.inp"
    loaded.write_inp(copy)
    assert copy.read_bytes() == path.read_bytes()
    assert loaded.steps[0].static.initial_increment == .05
    assert loaded.steps[0].static.minimum_increment == 1e-7
    assert loaded.steps[0].static.maximum_increment == .2


def test_reused_part_instances_have_unique_labels_and_transformed_coordinates(tmp_path):
    controller = controller_fixture()
    controller.assembly.add_instance(
        torchfea.Instance("part", translation=[3., 2., 1.], rotation=[0., 0., np.pi/2]),
        "second instance with unicode 名字",
    )
    path = tmp_path / "two.inp"
    controller.export_inp(path, nlgeom=False)
    deck = records(path.read_text())
    nodes = {int(row[0]): np.array(row[1:], dtype=float)
             for _, data in of_kind(deck, "node") for row in data}
    assert len(nodes) == 11
    assert np.allclose(nodes[7], [3., 3., 1.])
    ids = [int(row[0]) for _, data in of_kind(deck, "element") for row in data]
    assert ids == [1, 2, 3, 4, 5]
    names = [name for _, params, _ in deck for key, name in params.items()
             if key == "name"]
    assert len(names) == len(set(names))
    path.read_bytes().decode("ascii")


@pytest.mark.parametrize("mat,keyword,constants", [
    (materials.NeoHookean(6., 100.), "neo hooke", [3., .02]),
    (materials.MooneyRivlin(2., 1., 100.), "mooney-rivlin", [2., 1., .02]),
    (materials.Yeoh(2., .3, .1, 100.), "yeoh", [2., .3, .1, .02, 0., 0.]),
    (materials.Ogden(torch.tensor([3., 2., 1.]), torch.tensor([2., 4., 6.]), 100.),
     "ogden", [3., 2., 2., 4., 1., 6., .02, 0., 0.]),
])
def test_hyperelastic_constants_and_multiline_material_data(tmp_path, mat, keyword, constants):
    controller = controller_fixture()
    block = controller.assembly._parts["part"].elems["solid"]
    block.set_materials({"rubber": mat})
    path = tmp_path / "rubber.inp"
    controller.export_inp(path)
    for params, data in of_kind(records(path.read_text()), "hyperelastic"):
        assert f"*HYPERELASTIC, {keyword.upper()}" in path.read_text().upper()
        assert np.allclose([float(x) for row in data for x in row], constants)
        assert all(len(row) <= 8 for row in data)


def test_material_regions_and_elementwise_constants(tmp_path):
    controller = controller_fixture()
    block = controller.assembly._parts["part"].elems["solid"]
    block.set_materials({})
    block.set_materials(materials.LinearElastic(100., .25), "one", [0])
    block.set_materials(materials.NeoHookean(4., 200.), "two", [1])
    path = tmp_path / "regions.inp"
    controller.export_inp(path, nlgeom=False)
    deck = records(path.read_text())
    assert len(of_kind(deck, "elastic")) == 1
    assert len(of_kind(deck, "hyperelastic")) == 1
    assert len(of_kind(deck, "solid section")) == 2


def test_unsupported_enabled_physics_does_not_overwrite_existing_file(tmp_path):
    controller = controller_fixture()
    controller.assembly.add_reference_point(torchfea.ReferencePoint([2., 0., 0.]), "second")
    joint = loads.SphericalJoint("rp", "second", connection_point=[1., 1., 1.], k=1e5)
    controller.assembly.add_load(joint, "custom_joint")
    path = tmp_path / "existing.inp"
    path.write_text("existing contents")
    with pytest.raises(NotImplementedError, match="custom_joint.*SphericalJoint"):
        controller.export_inp(path, nlgeom=False)
    assert path.read_text() == "existing contents"
    joint.enabled = False
    controller.export_inp(path, nlgeom=False)
    assert "*End Step" in path.read_text()


@pytest.mark.parametrize("joint_type,direction,connector", [
    (loads.SphericalJoint, None, "JOIN"),
    (loads.SingleDoFJoint, [0, 0, 0, 0, 1, 1], "HINGE"),
    (loads.SingleDoFJoint, [1, 2, 0, 0, 0, 0], "TRANSLATOR"),
])
def test_native_joint_opt_in_keeps_anchor_offsets_and_free_axis(
    tmp_path, joint_type, direction, connector
):
    controller = controller_fixture()
    assembly = controller.assembly
    assembly.add_reference_point(torchfea.ReferencePoint([2., 3., 4.]), "second")
    kwargs = {} if direction is None else {"free_direction": direction}
    joint = joint_type("rp", "second", connection_point=[.3, .5, .7], k=1e5, **kwargs)
    assembly.add_load(joint, "joint")
    output = tmp_path / "joint.inp"
    with pytest.raises(NotImplementedError, match="joint"):
        controller.export_inp(output, nlgeom=False)
    with pytest.warns(UserWarning, match="ideal"):
        controller.export_inp(output, nlgeom=False, native_interactions=True)
    deck = records(output.read_text())
    section = of_kind(deck, "connector section")[0][1]
    assert section[0] == [connector]
    mpcs = of_kind(deck, "mpc")
    assert len(mpcs) == 2
    nodes = {int(row[0]): np.array(row[1:], dtype=float)
             for _, data in of_kind(deck, "node") for row in data}
    for _, data in mpcs:
        assert data[0][0] == "BEAM"
        assert np.allclose(nodes[int(data[0][1])], [.3, .5, .7])
        assert int(data[0][2]) in {6, 7}
    if direction is not None:
        orientation = of_kind(deck, "orientation")[0][1]
        first = np.array(orientation[0][:3], dtype=float)
        second = np.array(orientation[0][3:], dtype=float)
        expected = np.array(direction[3:] if connector == "HINGE" else direction[:3])
        assert np.allclose(first, expected / np.linalg.norm(expected))
        assert np.dot(first, second) == pytest.approx(0., abs=1e-15)
        assert np.linalg.norm(second) == pytest.approx(1.)


@pytest.mark.parametrize("self_contact", [False, True])
@pytest.mark.parametrize("native_interactions", [False, True])
def test_contact_defaults_to_frictionless_hard_contact(
    tmp_path, self_contact, native_interactions
):
    controller = controller_fixture()
    if not self_contact:
        controller.assembly.add_instance(
            torchfea.Instance("part", translation=[0., 0., 2.]), "second")
    contact = (loads.ContactSelf("instance", "loaded") if self_contact else
               loads.Contact("instance", "second", "loaded", "loaded"))
    controller.assembly.add_load(contact, "contact")
    path = tmp_path / "contact.inp"
    with pytest.warns(UserWarning, match="hard general contact"):
        controller.export_inp(path, nlgeom=False, native_interactions=native_interactions)
    text = path.read_text()
    deck = records(text)
    assert of_kind(deck, "surface behavior")[0][0]["pressure-overclosure"] == "hard"
    pair = of_kind(deck, "contact inclusions")[0][1][0]
    assert (pair[0] == pair[1]) == self_contact
    # The interaction defines only normal behavior: Abaqus defaults to frictionless.
    assert not of_kind(deck, "friction")
    assert of_kind(deck, "contact property assignment")[0][1][0][:2] == pair
    assert text.upper().index("*CONTACT PROPERTY ASSIGNMENT") < text.upper().index("*STEP")


def test_joint_chain_exports_all_five_native_hinges(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/joint_chain_test/joint_chain_gravity.py"
    namespace = runpy.run_path(str(example))
    controller, _ = namespace["build_chain_controller"]()
    path = tmp_path / "chain.inp"
    with pytest.warns(UserWarning):
        controller.export_inp(path, nlgeom=False, native_interactions=True)
    deck = records(path.read_text())
    assert len(of_kind(deck, "coupling")) == 5
    sections = of_kind(deck, "connector section")
    assert len(sections) == 5
    assert all(data[0] == ["HINGE"] for _, data in sections)
    assert len(of_kind(deck, "dload")[0][1]) > 5
    assert of_kind(deck, "end step")


def test_export_uses_final_loads_and_undeformed_geometry(tmp_path):
    controller = controller_fixture()
    first = tmp_path / "first.inp"
    second = tmp_path / "second.inp"
    controller.export_inp(first, nlgeom=False)
    controller.initialize()
    for load in controller.assembly._loads.values():
        if isinstance(load, loads.ContinuousLoad):
            load.continuation_factor = .17
    controller.assembly._GC += .25
    controller.export_inp(second, nlgeom=False)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("element_type", ["C3D4", "C3D6", "C3D8", "C3D10", "C3D15", "C3D20"])
def test_existing_element_examples_export_valid_connectivity(tmp_path, element_type):
    path = Path(__file__).resolve().parents[1] / "examples/element_test" / f"{element_type}.inp"
    original = torchfea.FEA_INP()
    original.read_inp(path)
    controller = torchfea.from_inp(original)
    output = tmp_path / "solid.inp"
    controller.export_inp(output, nlgeom=False)
    deck = records(output.read_text())
    node_count = sum(len(data) for _, data in of_kind(deck, "node"))
    expected = int(element_type[3:])
    for params, rows in of_kind(deck, "element"):
        assert params["type"] == element_type.lower()
        current = []
        connectivity = []
        for row in rows:
            current += [int(x) for x in row if x]
            if row[-1]:
                connectivity.append(current)
                current = []
        assert not current
        assert all(len(row) == expected + 1 for row in connectivity)
        assert all(1 <= n <= node_count for row in connectivity for n in row[1:])


@pytest.mark.skipif(shutil.which("abaqus") is None, reason="Abaqus executable is not installed")
def test_abaqus_datacheck(tmp_path):
    controller = controller_fixture()
    path = tmp_path / "torchfea_export.inp"
    controller.export_inp(path, nlgeom=False)
    result = subprocess.run(
        [shutil.which("abaqus"), "job=torchfea_export", f"input={path}", "datacheck", "interactive"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = (tmp_path / "torchfea_export.dat").read_text(errors="replace")
    assert "***ERROR" not in data.upper(), data
