from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from torchfea.model import Assembly, Instance, Part
from torchfea.model.boundarys import Boundary_Condition
from torchfea.model.constraints import Couple, LinearTie, NodeToSurfaceTie
from torchfea.model.constraints.node_to_surface_tie import _SurfaceProjection
from torchfea.model.elements import C3D4
from torchfea.model.elements.dimension3.surfaces import BaseSurface, Q4, Q8, T3, T6
from torchfea.model.elements.materials import LinearElastic
from torchfea.model.reference_points import ReferencePoint


@pytest.fixture(autouse=True)
def double_precision() -> Iterator[None]:
    """Keep numerical derivative comparisons independent of caller defaults."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def face_nodes(face_type: type[BaseSurface]) -> torch.Tensor:
    """Regular/curved reference faces with known polynomial coordinates."""
    nodes: list[list[float]]
    if face_type in (T3, T6):
        nodes = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        if face_type is T6:
            nodes += [[0.5, 0, 0.15], [0.5, 0.5, -0.05], [0, 0.5, 0.1]]
    else:
        nodes = [[-1, -1, 0], [1, -1, 0.1], [1, 1, 0.2], [-1, 1, -0.1]]
        if face_type is Q8:
            nodes += [[0, -1, 0.2], [1, 0, 0.15], [0, 1, 0.4], [-1, 0, 0.1]]
    return torch.tensor(nodes, dtype=torch.get_default_dtype())


def make_face(face_type: type[BaseSurface], nodes: torch.Tensor) -> BaseSurface:
    surface = face_type(torch.arange(len(nodes)).reshape(1, -1))
    # Surface shape evaluation only needs the reference nodes, so this unit
    # fixture deliberately omits unrelated volume-element/material setup.
    surface.initialize(cast(Part, SimpleNamespace(nodes=nodes)))
    return surface


def make_tie(
    face_type: type[BaseSurface] = T3,
    main_nodes: torch.Tensor | None = None,
    secondary_nodes: torch.Tensor | None = None,
    main_faces: list[BaseSurface] | None = None,
    *, position_tolerance: float = 1e-8, allow_unmatched: bool = False,
) -> tuple[NodeToSurfaceTie, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Build just the geometry and block offsets required by the Tie kernels."""
    main_nodes = face_nodes(face_type) if main_nodes is None else main_nodes
    face = main_faces[0] if main_faces else make_face(face_type, main_nodes)
    if secondary_nodes is None:
        local = torch.tensor([[0.13, 0.2], [0.65, 0.1], [0.1, 0.7]])
        secondary_nodes = face.evaluate_shape_functions(local)[0] @ main_nodes
    secondary_face = make_face(T3, secondary_nodes)

    def main_elements(name: str) -> list[BaseSurface]:
        return (main_faces or [face]) if name == "interface" else []

    def secondary_elements(name: str) -> list[BaseSurface]:
        return [secondary_face] if name == "interface" else []

    main = SimpleNamespace(
        nodes=main_nodes, _RGC_index=0,
        surfaces=SimpleNamespace(get_elements=main_elements),
    )
    secondary = SimpleNamespace(
        nodes=secondary_nodes, _RGC_index=1,
        surfaces=SimpleNamespace(get_elements=secondary_elements),
    )

    def get_instance(name: str) -> SimpleNamespace:
        return {"main": main, "secondary": secondary}[name]

    size = (len(main_nodes) + len(secondary_nodes)) * 3
    assembly = SimpleNamespace(
        _RGC_list_indexStart=[0, len(main_nodes) * 3, size, size],
        get_instance=get_instance,
    )
    tie = NodeToSurfaceTie(
        "main", "interface", "secondary", "interface",
        position_tolerance, allow_unmatched=allow_unmatched,
    )
    tie.set_RGC_index(2)
    tie.initialize(cast(Assembly, assembly))
    return tie, main, secondary, assembly


@pytest.mark.parametrize("face_type", [T3, T6, Q4, Q8])
def test_surface_shape_first_and_second_derivatives_by_finite_difference(face_type: type[BaseSurface]) -> None:
    # Difference N and dN independently to verify both analytic derivative
    # orders without using automatic differentiation as the reference.
    surface = make_face(face_type, face_nodes(face_type))
    local = torch.tensor([0.23, 0.31])
    shape, first, second = surface.evaluate_shape_functions(local)
    step = 1e-6
    for axis in range(2):
        shift = torch.eye(2)[axis] * step
        plus = surface.evaluate_shape_functions(local + shift)
        minus = surface.evaluate_shape_functions(local - shift)
        torch.testing.assert_close(first[axis], (plus[0] - minus[0]) / (2 * step), atol=2e-10, rtol=2e-9)
        torch.testing.assert_close(second[:, axis], (plus[1] - minus[1]) / (2 * step), atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(shape.sum(), torch.tensor(1.0))
    torch.testing.assert_close(first.sum(-1), torch.zeros(2), atol=1e-14, rtol=0)
    torch.testing.assert_close(second.sum(-1), torch.zeros(2, 2), atol=1e-14, rtol=0)


@pytest.mark.parametrize("face_type", [T3, T6, Q4, Q8])
def test_projection_objective_has_analytic_gradient_and_hessian(face_type: type[BaseSurface]) -> None:
    # Verify the distance objective itself, independently of optimizer
    # convergence, so an incorrect curvature term cannot hide in a projection.
    nodes = face_nodes(face_type)
    projection = _SurfaceProjection(make_face(face_type, nodes))
    coordinates = nodes.numpy()
    point = np.array([0.17, 0.14, 0.8])
    local = np.array([0.23, 0.31])
    _, first, second = projection.objective(local, coordinates, point)
    step = 1e-6
    for axis in range(2):
        shift = np.eye(2)[axis] * step
        plus = projection.objective(local + shift, coordinates, point)
        minus = projection.objective(local - shift, coordinates, point)
        np.testing.assert_allclose(first[axis], (plus[0] - minus[0]) / (2 * step), atol=3e-10, rtol=2e-9)
        np.testing.assert_allclose(second[:, axis], (plus[1] - minus[1]) / (2 * step), atol=3e-10, rtol=2e-9)
    np.testing.assert_allclose(second, second.T, atol=1e-14)


@pytest.mark.parametrize("face_type", [T3, T6, Q4, Q8])
def test_projection_retains_original_fe_interpolation_and_rigid_motion(face_type: type[BaseSurface]) -> None:
    tie, main, secondary, _ = make_tie(face_type)
    # At most 3/4/6/8 coefficients per secondary node, never a dense C.
    assert tie._weights.ndim == 1
    assert tie._weights.numel() <= len(secondary.nodes) * len(main.nodes)
    affine = torch.tensor([[0.2, -0.3, 0.1], [0.4, -0.1, 0.3], [0.3, 0.5, 0.2]])
    translation = torch.tensor([0.1, -0.4, 0.2])
    actual = tie._interpolate(main.nodes @ affine.T + translation)
    expected = secondary.nodes @ affine.T + translation
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    angle = torch.tensor(1.3)
    rotation = torch.tensor([
        [angle.cos(), -angle.sin(), 0],
        [angle.sin(), angle.cos(), 0], [0, 0, 1],
    ])
    actual = tie._interpolate(main.nodes @ rotation.T + translation - main.nodes)
    expected = secondary.nodes @ rotation.T + translation - secondary.nodes
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("face_type", [T6, Q8])
def test_curved_face_search_bounds_include_bulges_beyond_all_nodes(face_type: type[BaseSurface]) -> None:
    nodes = face_nodes(face_type)
    nodes[:, 2] = 0
    nodes[3 if face_type is T6 else 4:, 2] = 1
    surface = make_face(face_type, nodes)
    local = torch.tensor([[0.3, 0.3], [0.4, 0.2], [0.2, 0.4]]) if face_type is T6 else torch.tensor([
        [0, 0], [0.1, 0], [0, 0.1],
    ])
    points = surface.evaluate_shape_functions(local)[0] @ nodes
    assert points[:, 2].min() > nodes[:, 2].max()
    tie, _, _, _ = make_tie(face_type, main_nodes=nodes, secondary_nodes=points)
    torch.testing.assert_close(tie._interpolate(nodes), points, atol=1e-10, rtol=1e-9)


def test_closest_point_on_triangle_edge_and_tolerance_policy() -> None:
    points = torch.tensor([[0.6, 0.6, 0], [0.4, 0.1, 0], [0.1, 0.4, 0]])
    with pytest.raises(ValueError, match="outside position_tolerance"):
        make_tie(secondary_nodes=points)
    tie, main, _, _ = make_tie(secondary_nodes=points, position_tolerance=0.15)
    torch.testing.assert_close(tie._interpolate(main.nodes)[0], torch.tensor([0.5, 0.5, 0.0]))
    partial, _, _, _ = make_tie(secondary_nodes=points, allow_unmatched=True)
    assert partial._unmatched_nodes.tolist() == [0]
    assert partial._secondary_nodes.tolist() == [1, 2]


def test_empty_partial_tie_is_an_identity_and_does_not_remove_dofs() -> None:
    tie, main, secondary, assembly = make_tie(
        secondary_nodes=torch.tensor([[2., 0, 0], [2, 1, 0], [3, 0, 0]]),
        allow_unmatched=True,
    )
    assert tie.get_output_dofs().size == 0
    mask = [np.ones(main.nodes.shape, bool), np.ones(secondary.nodes.shape, bool)]
    assert all(m.all() for m in tie.set_required_DoFs(mask))
    size = assembly._RGC_list_indexStart[-1]
    residual = torch.arange(float(size))
    delta = tie.modify_R_K(None, residual, if_onlyforce=True)
    assert not delta.any()
    indices, values = tie.modify_mass_matrix(torch.eye(size).to_sparse().indices(), torch.ones(size), None)
    assert indices.shape == (2, 0) and values.numel() == 0


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="position_tolerance"):
        make_tie(position_tolerance=tolerance)


def expand_displacements(tie: LinearTie, main_displacements: torch.Tensor) -> torch.Tensor:
    """Evaluate physical displacements to difference energy in independent DoFs."""
    blocks = [main_displacements.reshape(-1, 3), torch.zeros((len(tie._secondary_nodes), 3)), torch.empty(0)]
    return torch.cat([block.flatten() for block in tie.modify_RGC(blocks)])


def test_displacement_jacobian_and_zero_hessian_by_finite_difference() -> None:
    # Compare each independent-coordinate direction with the constant map;
    # a centered second difference must vanish for this linear kinematics.
    tie, main, _, _ = make_tie()
    generator = torch.Generator().manual_seed(12)
    displacement = torch.randn(main.nodes.numel(), generator=generator)
    step = 1e-5
    for direction in torch.eye(len(displacement)):
        analytic = expand_displacements(tie, direction)
        plus = expand_displacements(tie, displacement + step * direction)
        minus = expand_displacements(tie, displacement - step * direction)
        value = expand_displacements(tie, displacement)
        torch.testing.assert_close((plus - minus) / (2 * step), analytic, atol=3e-11, rtol=1e-9)
        torch.testing.assert_close((plus - 2 * value + minus) / step**2, torch.zeros_like(value), atol=1e-5, rtol=0)


def test_nonlinear_energy_gradient_and_hessian_after_tie_reduction() -> None:
    # Dense cross-instance coupling and a quartic energy produce a nontrivial
    # Hessian even though the Tie's own displacement Hessian is zero.
    tie, main, _, assembly = make_tie()
    size = assembly._RGC_list_indexStart[-1]
    generator = torch.Generator().manual_seed(42)
    matrix = torch.randn((size, size), generator=generator)
    matrix = matrix.T @ matrix + torch.eye(size)
    q = torch.randn(main.nodes.numel(), generator=generator) * 0.1

    def energy(coordinates: torch.Tensor) -> torch.Tensor:
        u = expand_displacements(tie, coordinates)
        return 0.5 * u @ matrix @ u + 0.05 * u.pow(4).sum()

    def derivatives(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = expand_displacements(tie, coordinates)
        residual = matrix @ u + 0.2 * u.pow(3)
        stiffness = matrix + torch.diag(0.6 * u.pow(2))
        sparse = stiffness.to_sparse().coalesce()
        # Exercise COO duplicates, as element assembly accumulates them.
        indices = sparse.indices().repeat(1, 2)
        values = sparse.values().repeat(2) * 0.5
        delta, ki, kv = tie.modify_R_K(None, residual, indices, values)
        transformed = stiffness + torch.sparse_coo_tensor(ki, kv, (size, size)).to_dense()
        force_only = tie.modify_R_K(None, residual, if_onlyforce=True)
        torch.testing.assert_close(force_only, delta)
        return (residual + delta)[:len(q)], transformed[:len(q), :len(q)]

    force, stiffness = derivatives(q)
    step = 1e-6
    # Difference energy for the force and difference force for the tangent;
    # these independent checks catch missing row/column transfer terms.
    for axis, direction in enumerate(torch.eye(len(q))):
        numerical_force = (energy(q + step * direction) - energy(q - step * direction)) / (2 * step)
        numerical_stiffness = (derivatives(q + step * direction)[0] - derivatives(q - step * direction)[0]) / (2 * step)
        torch.testing.assert_close(force[axis], numerical_force, atol=2e-9, rtol=1e-8)
        torch.testing.assert_close(stiffness[:, axis], numerical_stiffness, atol=2e-9, rtol=1e-8)
    torch.testing.assert_close(stiffness, stiffness.T)


@pytest.mark.parametrize("matrix_kind", ["mass", "nonsymmetric", "main_only", "empty"])
def test_sparse_matrix_transfer_equals_dense_congruence(matrix_kind: str) -> None:
    tie, _, _, assembly = make_tie()
    size = assembly._RGC_list_indexStart[-1]
    if matrix_kind == "mass":
        matrix = torch.diag(torch.arange(1., size + 1))
    elif matrix_kind == "nonsymmetric":
        matrix = torch.arange(float(size * size)).reshape(size, size)
    elif matrix_kind == "main_only":
        matrix = torch.zeros(size, size)
        matrix[:9, :9] = 2
    else:
        matrix = torch.zeros(size, size)
    # A dense reference transform is deliberately limited to this tiny test.
    transform = torch.eye(size)
    transform[tie._secondary_dofs] = 0
    transform[tie._secondary_dofs[tie._dof_rows], tie._dof_targets] += tie._dof_weights
    sparse = matrix.to_sparse().coalesce()
    indices, values = tie.modify_mass_matrix(sparse.indices(), sparse.values(), None)
    actual = matrix + torch.sparse_coo_tensor(indices, values, (size, size)).to_dense()
    torch.testing.assert_close(actual, transform.T @ matrix @ transform)


def make_assembly_chain() -> Assembly:
    """Use real volume elements to exercise Assembly's complete chain rule."""
    assembly = Assembly()
    nodes = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    part = Part(nodes)
    element = C3D4(torch.tensor([0]), torch.tensor([[0, 1, 2, 3]]))
    element.set_materials(LinearElastic(100., 0.25))
    element.density = torch.tensor([2.])
    part.add_element(element, "solid")
    part.surfaces["interface"] = [(np.array([0]), 0)]
    part.set_nodes["interface"] = np.array([0, 1, 2])
    assembly._parts["part"] = part
    assembly._instances = {name: Instance("part") for name in ["a", "b", "c"]}
    # Consumer sorts before producer by name; the dependency planner must fix it.
    assembly.add_constraint(NodeToSurfaceTie("b", "interface", "c", "interface", 1e-8), "first")
    assembly.add_constraint(NodeToSurfaceTie("a", "interface", "b", "interface", 1e-8), "last")
    return assembly


@pytest.mark.parametrize("with_couple", [False, True])
def test_real_assembly_chain_residual_and_tangent_match_energy_derivatives(with_couple: bool) -> None:
    assembly = make_assembly_chain()
    if with_couple:
        assembly.add_reference_point(ReferencePoint([0., 0, 0]), "rp")
        assembly.add_constraint(Couple("a", "interface", "rp"), "couple")
    assembly.initialize()
    assert [c.main_instance_name for c in assembly._constraints_forward if isinstance(c, LinearTie)] == ["a", "b"]
    q = torch.linspace(-0.02, 0.04, assembly._GC.numel())
    residual, indices, values = assembly.assemble_Stiffness_Matrix(GC=q)
    stiffness = torch.sparse_coo_tensor(indices, values, (len(q), len(q))).to_dense()
    blocks = assembly._GC2RGC(q)
    torch.testing.assert_close(blocks[0][:3], blocks[1][:3])
    torch.testing.assert_close(blocks[1][:3], blocks[2][:3])
    step = 1e-6
    for axis, direction in enumerate(torch.eye(len(q))):
        plus, minus = q + step * direction, q - step * direction
        gradient = (assembly._total_Potential_Energy(GC=plus) - assembly._total_Potential_Energy(GC=minus)) / (2 * step)
        tangent = (assembly.assemble_Stiffness_Matrix(GC=plus)[0] - assembly.assemble_Stiffness_Matrix(GC=minus)[0]) / (2 * step)
        torch.testing.assert_close(residual[axis], gradient, atol=2e-8, rtol=1e-6)
        torch.testing.assert_close(stiffness[:, axis], tangent, atol=2e-7, rtol=1e-6)


def test_duplicate_ties_are_rejected() -> None:
    assembly = make_assembly_chain()
    assembly.add_constraint(NodeToSurfaceTie("a", "interface", "b", "interface", 1e-8), "duplicate")
    with pytest.raises(ValueError, match="both constraints"):
        assembly.initialize()


def test_couple_and_tie_dependency_and_duplicate_output_detection() -> None:
    assembly = make_assembly_chain()
    assembly.add_reference_point(ReferencePoint([0., 0, 0]), "rp")
    assembly.add_constraint(Couple("a", "interface", "rp"), "couple")
    assembly.initialize()
    assert isinstance(assembly._constraints_forward[0], Couple)
    assembly.add_constraint(Couple("b", "interface", "rp"), "duplicate")
    with pytest.raises(ValueError, match="both constraints"):
        assembly.initialize()


def test_serialization_excludes_mapping_and_rebuilds_it_on_initialize() -> None:
    tie, _, _, assembly = make_tie(T6)
    wire = tie._serialize()
    assert wire[0]["_weights"] == (None, "NoneType")
    loaded = NodeToSurfaceTie._deserialize(wire)
    assert isinstance(loaded, LinearTie)
    loaded.set_RGC_index(2)
    loaded.initialize(cast(Assembly, assembly))
    torch.testing.assert_close(loaded._weights, tie._weights)
    np.testing.assert_array_equal(loaded.get_output_dofs(), tie.get_output_dofs())


def test_zero_position_tolerance_accepts_coincident_surface_with_roundoff() -> None:
    tie, _, _, _ = make_tie(position_tolerance=0)
    assert tie._secondary_nodes.tolist() == [0, 1, 2]


def test_dynamic_assembly_chain_mass_matches_full_coordinate_transform() -> None:
    assembly = make_assembly_chain()
    assembly.initialize()
    assembly.initialize_dynamic()
    q = assembly._GC.clone()
    size = assembly._RGC_list_indexStart[-1]
    mass = torch.sparse_coo_tensor(
        assembly._mass_matrix_indices, assembly._mass_matrix_values, (size, size),
    ).to_dense()
    transform = torch.stack([
        torch.cat([block.reshape(-1) for block in assembly._GC2RGC(direction)])
        for direction in torch.eye(len(q))
    ], dim=1)
    indices, values = assembly.assemble_mass_matrix(q)
    actual = torch.sparse_coo_tensor(indices, values, (len(q), len(q))).to_dense()
    torch.testing.assert_close(actual, transform.T @ mass @ transform)


def test_tie_cycle_is_rejected_and_disabling_tie_restores_secondary_dofs() -> None:
    assembly = make_assembly_chain()
    assembly.add_constraint(NodeToSurfaceTie("c", "interface", "a", "interface", 1e-8), "cycle")
    with pytest.raises(ValueError, match="dependency cycle"):
        assembly.initialize()
    del assembly._constraints["cycle"]
    assembly.initialize()
    original_size = assembly._GC.numel()
    assembly._constraints["first"].enabled = False
    assembly.define_required_DoFs()
    assert assembly._GC.numel() == original_size + 9
    assert len(assembly._constraints_forward) == 1


def test_boundary_on_main_nodes_is_allowed() -> None:
    assembly = make_assembly_chain()
    assembly._boundarys["fixed-main"] = Boundary_Condition("a", "interface")
    assembly.initialize()
    q = torch.linspace(0.01, 0.02, len(assembly._GC))
    blocks = assembly._GC2RGC(q)
    for block in blocks[:3]:
        torch.testing.assert_close(block[:3], torch.zeros(3, 3))


def test_instance_transforms_are_used_in_projection_without_mutating_part() -> None:
    coordinates = face_nodes(T3)
    main_part = Part(coordinates)
    surface = make_face(T3, coordinates)
    main_part.surfaces._surface_elements["interface"] = [surface]
    main = Instance("main-part", translation=[5., -2, 0.4], rotation=[0., 0, 0.6])
    main.part = main_part
    main.set_RGC_index(0)
    local = torch.tensor([[0.1, 0.2], [0.6, 0.1], [0.1, 0.6]])
    points = surface.evaluate_shape_functions(local)[0] @ main.nodes
    secondary_part = Part(points - torch.tensor([1., 2, 3]))
    secondary_part.surfaces._surface_elements["interface"] = [
        make_face(T3, secondary_part.nodes)]
    secondary = Instance("secondary-part", translation=[1., 2, 3])
    secondary.part = secondary_part
    secondary.set_RGC_index(1)
    assembly = SimpleNamespace(
        _RGC_list_indexStart=[0, 9, 18, 18],
        get_instance=lambda name: {"main": main, "secondary": secondary}[name],
    )
    tie = NodeToSurfaceTie("main", "interface", "secondary", "interface", 1e-8)
    tie.set_RGC_index(2)
    tie.initialize(cast(Assembly, assembly))
    torch.testing.assert_close(tie._interpolate(main.nodes), secondary.nodes)
    torch.testing.assert_close(main_part.nodes, coordinates)


def test_degenerate_main_face_is_rejected() -> None:
    nodes = torch.tensor([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    points = torch.tensor([[0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]])
    with pytest.raises(ValueError, match="Degenerate main facet"):
        make_tie(main_nodes=nodes, secondary_nodes=points)


def test_mixed_main_faces_use_ragged_rows_for_displacement_and_matrix_transfer() -> None:
    nodes = torch.tensor([
        [0., 0, 0], [1, 0, 0], [0, 1, 0],
        [3, -1, 0], [5, -1, 0], [5, 1, 0], [3, 1, 0],
    ])
    triangle = T3(torch.tensor([[0, 1, 2]]))
    quadrilateral = Q4(torch.tensor([[3, 4, 5, 6]]))
    for surface in [triangle, quadrilateral]:
        surface.initialize(cast(Part, SimpleNamespace(nodes=nodes)))
    points = torch.tensor([[0.2, 0.2, 0], [4, 0, 0], [4.4, 0.3, 0]])
    tie, _, _, assembly = make_tie(
        main_nodes=nodes, secondary_nodes=points,
        main_faces=[triangle, quadrilateral],
    )
    assert torch.bincount(tie._mapping_rows).tolist() == [3, 4, 4]
    torch.testing.assert_close(tie._interpolate(nodes), points)
    size = assembly._RGC_list_indexStart[-1]
    transform = torch.eye(size)
    transform[tie._secondary_dofs] = 0
    transform[tie._secondary_dofs[tie._dof_rows], tie._dof_targets] += tie._dof_weights
    matrix = torch.arange(float(size * size)).reshape(size, size)
    sparse = matrix.to_sparse().coalesce()
    indices, values = tie.modify_mass_matrix(sparse.indices(), sparse.values(), None)
    actual = matrix + torch.sparse_coo_tensor(indices, values, (size, size)).to_dense()
    torch.testing.assert_close(actual, transform.T @ matrix @ transform)


def test_bvh_stops_projecting_facets_beyond_the_best_actual_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All plane boxes fit the initial tolerance. After projecting the first
    # plane at zero distance, farther planes must not invoke the narrow phase.
    count = 32
    base = face_nodes(T3)
    nodes = torch.cat([base + torch.tensor([0., 0, float(i)]) for i in range(count)])
    surface = T3(torch.arange(count * 3).reshape(-1, 3))
    surface.initialize(cast(Part, SimpleNamespace(nodes=nodes)))
    points = torch.tensor([[0.1, 0.1, 0], [0.5, 0.1, 0], [0.1, 0.5, 0]])
    calls = 0
    original = _SurfaceProjection.closest_point

    def counted_projection(
        projection: _SurfaceProjection,
        point: NDArray[np.float64],
        coordinates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        nonlocal calls
        calls += 1
        return original(projection, point, coordinates)

    monkeypatch.setattr(_SurfaceProjection, "closest_point", counted_projection)
    tie, _, _, _ = make_tie(
        main_nodes=nodes, secondary_nodes=points, main_faces=[surface],
        position_tolerance=float(count),
    )
    assert calls == len(points)
    assert tie._main_facets.tolist() == [0, 0, 0]
    torch.testing.assert_close(tie._interpolate(nodes), points)
