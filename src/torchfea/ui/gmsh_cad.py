"""Native Gmsh OCC feature history, BREP preview, and finite-element meshing.

The workbench has two deliberately separate paths:

* :func:`preview_brep` tessellates OCC *faces only* for interactive CAD
  display. It never produces a torchfea ``Part`` or a 3-D FE mesh.
* :func:`create_part` applies one global mesh size, creates the final C3D4
  volume mesh, and maps Gmsh surface/volume entities back to torchfea sets.

No function in this module calls ``torchfea.model.cad``.
"""

from __future__ import annotations

from collections import defaultdict
import os
import tempfile
from typing import Iterable

import numpy as np
import torch

from .logging_utils import LOGGER


_PRIMITIVES = {"box", "wedge", "cylinder", "sphere", "cone", "torus"}
_BOOLEANS = {"fuse", "cut", "intersect", "fragment"}
_TRANSFORMS = {"translate", "rotate", "dilate", "mirror"}


def primitive_defaults(kind: str) -> dict[str, float]:
    """Default native OCC parameters for a newly inserted feature."""
    defaults = {
        "box": {"x": 0., "y": 0., "z": 0., "dx": 20., "dy": 10., "dz": 5.},
        "wedge": {"x": 0., "y": 0., "z": 0., "dx": 20., "dy": 10., "dz": 5., "ltx": 0.},
        "cylinder": {"x": 0., "y": 0., "z": 0., "dx": 0., "dy": 0., "dz": 20., "radius": 5.},
        "sphere": {"x": 0., "y": 0., "z": 0., "radius": 5.},
        "cone": {"x": 0., "y": 0., "z": 0., "dx": 0., "dy": 0., "dz": 20., "radius1": 5., "radius2": 1.},
        "torus": {"x": 0., "y": 0., "z": 0., "major_radius": 8., "minor_radius": 2.},
    }
    return dict(defaults[kind])


def transform_defaults(kind: str) -> dict[str, float]:
    """Default values for an in-history OCC transformation feature."""
    defaults = {
        "translate": {"dx": 5., "dy": 0., "dz": 0.},
        "rotate": {"x": 0., "y": 0., "z": 0., "ax": 0., "ay": 0., "az": 1., "angle": 1.5707963267948966},
        "dilate": {"x": 0., "y": 0., "z": 0., "sx": 1., "sy": 1., "sz": 1.},
        # ax + by + cz + d = 0, defaulting to mirror in the x=0 plane.
        "mirror": {"a": 1., "b": 0., "c": 0., "d": 0.},
    }
    return dict(defaults[kind])


def preview_brep(operations: Iterable) -> list[dict]:
    """Return face-wise display geometry for the final native OCC solid.

    Temporary two-dimensional Gmsh triangles are visualization tessellation,
    not the analysis mesh. Each record keeps its BREP surface and volume tags
    so selections are stable until volume meshing.
    """
    import gmsh

    operations = list(operations)
    LOGGER.info("BREP preview start: operations=%d kinds=%s", len(operations),
                [getattr(operation, "kind", "?") for operation in operations])

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear(); gmsh.model.add("torchfea_brep_preview")
        volumes = _build_occ(gmsh, operations)
        LOGGER.debug("BREP OCC history built: final_volumes=%s", volumes)
        _set_preview_size(gmsh, volumes)
        volumes = _mesh_with_topology_retry(gmsh, volumes, 2)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points, local = _local_nodes(node_tags, node_coords)
        vertex_tags, vertex_points = _occ_vertices(gmsh)
        records: list[dict] = []
        for _dimension, surface_tag in _boundary_surfaces(gmsh, volumes):
            triangles = _surface_triangles(gmsh, surface_tag, local)
            if not len(triangles):
                raise RuntimeError(
                    f"Gmsh produced no triangular display mesh for BREP surface {surface_tag}.")
            owners = [tag for tag in volumes if (2, surface_tag) in gmsh.model.getBoundary([(3, tag)], False, False)]
            if not owners:
                raise RuntimeError(
                    f"BREP surface {surface_tag} is not owned by a final volume.")
            edge_lines = []
            for _edge_dimension, edge_tag in _surface_curves(gmsh, surface_tag):
                lines = _curve_lines(gmsh, edge_tag, local)
                if len(lines):
                    edge_lines.append({"edge_tag": int(edge_tag), "lines": lines})
            records.append({"points": points, "triangles": triangles, "surface_tag": int(surface_tag),
                            "volume_tags": [int(tag) for tag in owners],
                            "edge_lines": edge_lines,
                            "vertex_tags": vertex_tags, "vertex_points": vertex_points})
        LOGGER.info("BREP preview complete: faces=%d display_nodes=%d", len(records), len(points))
        return records
    except Exception:
        LOGGER.exception("BREP preview failed")
        raise
    finally:
        gmsh.finalize()


def create_part(operations: Iterable, mesh_size: float):
    """Mesh final OCC volumes once and adapt them to a C3D4 torchfea Part."""
    import gmsh

    from torchfea.model import Part
    from torchfea.model.elements import initialize_element

    operations = list(operations)
    mesh_size = float(mesh_size)
    LOGGER.info("FE mesh start: operations=%d mesh_size=%r (%s)", len(operations), mesh_size, type(mesh_size).__name__)
    if mesh_size <= 0:
        raise ValueError("Global mesh size must be positive.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear(); gmsh.model.add("torchfea_analysis_mesh")
        volumes = _build_occ(gmsh, operations)
        LOGGER.debug("FE OCC history built: final_volumes=%s", volumes)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        LOGGER.debug("Gmsh 3D mesh.generate(3) begin")
        volumes = _mesh_with_topology_retry(gmsh, volumes, 3)
        LOGGER.debug("Gmsh 3D mesh.generate(3) complete")
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points, local = _local_nodes(node_tags, node_coords)
        vertex_nodes = _vertex_mesh_nodes(gmsh, local)
        connectivity, volume_elements = _volume_tetrahedra(gmsh, volumes, local)
        surface_triangles = {tag: _surface_triangles(gmsh, tag, local)
                             for _dim, tag in _boundary_surfaces(gmsh, volumes)}
        missing_surfaces = [tag for tag, triangles in surface_triangles.items() if not len(triangles)]
        if missing_surfaces:
            raise RuntimeError(f"Gmsh produced no triangular boundary mesh for surfaces: {missing_surfaces}")
        surface_nodes = {tag: np.unique(triangles.ravel())
                         for tag, triangles in surface_triangles.items()}
    except Exception:
        LOGGER.exception("FE mesh failed inside Gmsh/OCC stage")
        raise
    finally:
        gmsh.finalize()

    LOGGER.info("FE mesh raw result: nodes=%d C3D4=%d volumes=%d surfaces=%d",
                len(points), len(connectivity), len(volume_elements), len(surface_triangles))
    if not len(connectivity):
        raise RuntimeError("Native Gmsh meshing did not produce C3D4 tetrahedra.")
    part = Part(torch.tensor(points, dtype=torch.get_default_dtype()))
    element = initialize_element("C3D4", torch.arange(int(len(connectivity)), dtype=torch.long),
                                 torch.tensor(connectivity, dtype=torch.long))
    part.add_element(element, "C3D4")
    part.add_surface_set("extern", element.extract_boundary_surface_set())
    # These are analysis-wide system collections.  They are deliberately
    # created outside the editable UI definition so every freshly meshed
    # Part has stable names for BCs, exports, and post-processing.
    part.add_node_set("all", np.arange(len(points), dtype=int))
    part.add_element_set("all", np.arange(len(connectivity), dtype=int))
    all_surface_nodes = (np.unique(np.concatenate(list(surface_nodes.values())))
                         if surface_nodes else np.empty(0, dtype=int))
    if not len(all_surface_nodes):
        raise RuntimeError("Gmsh produced no nodes on the Part exterior surface.")
    part.add_node_set("surface_all", all_surface_nodes)
    face_lookup = _tet_face_lookup(connectivity, element.surfaceid_map)
    gmsh_surface_sets: dict[int, list[tuple[np.ndarray, int]]] = {}
    for tag, triangles in surface_triangles.items():
        grouped: dict[int, list[int]] = defaultdict(list)
        unmatched = 0
        for triangle in triangles:
            matched = face_lookup.get(tuple(sorted(int(node) for node in triangle)))
            if matched is not None:
                elem_id, side = matched
                grouped[side].append(elem_id)
            else:
                unmatched += 1
        if unmatched:
            raise RuntimeError(
                f"Gmsh surface {tag} has {unmatched} triangles not present on any C3D4 boundary face.")
        groups = [(np.asarray(ids, dtype=int), side) for side, ids in grouped.items()]
        if groups:
            gmsh_surface_sets[tag] = groups
            part.add_surface_set(f"gmsh_surface_{tag}", groups)
            part.add_node_set(f"gmsh_nodes_{tag}", surface_nodes[tag])
    # Public UI metadata: Gmsh entity tag -> native torchfea selection.
    part._gmsh_surface_sets = gmsh_surface_sets
    part._gmsh_surface_nodes = surface_nodes
    part._gmsh_volume_elements = volume_elements
    part._gmsh_vertex_nodes = vertex_nodes
    part.exterior_surface = "extern"
    LOGGER.info("FE mesh complete: torchfea nodes=%s elements=%s generated_sets=%s",
                tuple(part.nodes.shape),
                {name: tuple(element._elems.shape) for name, element in part.elems.items()},
                {"node": ["all", "surface_all"],
                 "element": ["all"]})
    return part


def _build_occ(gmsh, operations: Iterable) -> list[int]:
    """Replay ordered OCC primitive and boolean features, returning final volumes."""
    occ = gmsh.model.occ
    operations = list(operations)
    if not operations:
        raise ValueError("A Part needs at least one OCC feature.")
    outputs: dict[str, list[tuple[int, int]]] = {}
    available: set[str] = set()
    seen: set[str] = set()
    alive: list[tuple[int, int]] = []
    for operation in operations:
        kind, values, op_id = operation.kind, operation.values, str(operation.id).strip()
        inputs = list(operation.inputs or [])
        if not op_id:
            raise ValueError("Every OCC feature needs a non-empty id.")
        if op_id in seen:
            raise ValueError(f"Duplicate OCC feature id '{op_id}'.")
        unknown = [name for name in inputs if name not in seen]
        if unknown:
            raise ValueError(
                f"OCC feature '{op_id}' references unknown or future feature(s): {unknown}.")
        unavailable = [name for name in inputs if name not in available]
        if unavailable:
            raise ValueError(
                f"OCC feature '{op_id}' references consumed feature(s): {unavailable}.")
        if len(inputs) != len(set(inputs)):
            raise ValueError(f"OCC feature '{op_id}' contains duplicate inputs.")
        LOGGER.debug("OCC operation: id=%s kind=%s inputs=%s values=%s", op_id, kind,
                     inputs, values)
        if kind in _PRIMITIVES:
            if inputs:
                raise ValueError(f"Primitive feature '{op_id}' cannot have inputs.")
            output = [(3, _add_primitive(occ, kind, values))]
            alive.extend(output)
        elif kind in _BOOLEANS:
            if len(inputs) != 2:
                raise ValueError(f"Boolean feature '{op_id}' needs exactly two input features.")
            object_entities, tool_entities = outputs[inputs[0]], outputs[inputs[1]]
            if not object_entities or not tool_entities:
                raise ValueError(f"Boolean feature '{op_id}' has unavailable inputs.")
            if kind == "fuse":
                output, _ = occ.fuse(object_entities, tool_entities, removeObject=True, removeTool=True)
            elif kind == "cut":
                output, _ = occ.cut(object_entities, tool_entities, removeObject=True, removeTool=True)
            elif kind == "intersect":
                output, _ = occ.intersect(object_entities, tool_entities, removeObject=True, removeTool=True)
            else:
                output, _ = occ.fragment(object_entities, tool_entities, removeObject=True, removeTool=True)
            output = [entity for entity in output if entity[0] == 3]
            consumed = set(object_entities + tool_entities)
            alive = [entity for entity in alive if entity not in consumed]
            alive.extend(output)
        elif kind in _TRANSFORMS:
            if len(operation.inputs) != 1:
                raise ValueError(f"Transform feature '{op_id}' needs exactly one input feature.")
            output = outputs[inputs[0]]
            if not output:
                raise ValueError(f"Transform feature '{op_id}' has an unavailable input.")
            _transform(occ, kind, output, values)
            # The transform intentionally changes the selected OCC entities
            # in place.  Replacing the alive entries keeps a transformed
            # feature as the current leaf in the history graph.
            consumed = set(output)
            alive = [entity for entity in alive if entity not in consumed]
            alive.extend(output)
        else:
            raise ValueError(f"Unsupported OCC feature '{kind}'.")
        occ.synchronize()
        outputs[op_id] = output
        seen.add(op_id)
        for input_id in inputs:
            available.remove(input_id)
        available.add(op_id)
    if not alive:
        raise ValueError("A Part needs at least one native OCC feature.")
    return [tag for dimension, tag in dict.fromkeys(alive) if dimension == 3]


def _add_primitive(occ, kind: str, values: dict[str, float | int]) -> int:
    def value(key: str) -> float:
        return float(values[key])
    if kind == "box":
        if min(value("dx"), value("dy"), value("dz")) <= 0:
            raise ValueError("Box dimensions must be positive.")
        return occ.addBox(value("x"), value("y"), value("z"),
                          value("dx"), value("dy"), value("dz"))
    if kind == "wedge":
        if min(value("dx"), value("dy"), value("dz")) <= 0:
            raise ValueError("Wedge dimensions must be positive.")
        # Gmsh's seventh positional argument is ``tag``; ``ltx`` is the
        # eighth one.  Passing ltx as argument seven makes non-zero values
        # look like an integer entity tag and causes Wedge creation/preview
        # to fail.  Keep the automatic tag explicit and pass the top extent
        # in its documented slot.
        return occ.addWedge(value("x"), value("y"), value("z"), value("dx"),
                            value("dy"), value("dz"), -1, value("ltx"))
    if kind == "cylinder":
        axis = np.array([value("dx"), value("dy"), value("dz")])
        if value("radius") <= 0 or np.linalg.norm(axis) == 0:
            raise ValueError("Cylinder radius and axis must be positive.")
        return occ.addCylinder(value("x"), value("y"), value("z"), *axis, value("radius"))
    if kind == "sphere":
        if value("radius") <= 0:
            raise ValueError("Sphere radius must be positive.")
        return occ.addSphere(value("x"), value("y"), value("z"), value("radius"))
    if kind == "cone":
        axis = np.array([value("dx"), value("dy"), value("dz")])
        if min(value("radius1"), value("radius2")) < 0 or np.linalg.norm(axis) == 0:
            raise ValueError("Cone radii and axis must be valid.")
        return occ.addCone(value("x"), value("y"), value("z"), *axis,
                           value("radius1"), value("radius2"))
    if kind == "torus":
        if min(value("major_radius"), value("minor_radius")) <= 0:
            raise ValueError("Torus radii must be positive.")
        return occ.addTorus(value("x"), value("y"), value("z"),
                            value("major_radius"), value("minor_radius"))
    raise ValueError(f"Unknown primitive '{kind}'.")


def _transform(occ, kind: str, entities: list[tuple[int, int]],
               values: dict[str, float | int]) -> None:
    """Apply one native OCC transformation to a feature's live entities."""
    def value(key: str) -> float:
        return float(values[key])
    if kind == "translate":
        occ.translate(entities, value("dx"), value("dy"), value("dz"))
    elif kind == "rotate":
        axis = np.asarray([value("ax"), value("ay"), value("az")])
        if np.linalg.norm(axis) == 0:
            raise ValueError("Rotate transform needs a non-zero axis.")
        occ.rotate(entities, value("x"), value("y"), value("z"), *axis, value("angle"))
    elif kind == "dilate":
        if min(abs(value("sx")), abs(value("sy")), abs(value("sz"))) == 0:
            raise ValueError("Scale factors must be non-zero.")
        occ.dilate(entities, value("x"), value("y"), value("z"),
                   value("sx"), value("sy"), value("sz"))
    elif kind == "mirror":
        normal = np.asarray([value("a"), value("b"), value("c")])
        if np.linalg.norm(normal) == 0:
            raise ValueError("Mirror transform needs a non-zero plane normal.")
        occ.mirror(entities, *normal, value("d"))
    else:
        raise ValueError(f"Unknown OCC transform '{kind}'.")


def _local_nodes(node_tags, node_coords):
    tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    order = np.argsort(tags)
    tags, coords = tags[order], coords[order]

    def to_local_indices(global_tags):
        return np.searchsorted(tags, np.asarray(global_tags, dtype=np.int64))

    return coords, to_local_indices


def _occ_vertices(gmsh):
    tags, points = [], []
    for _dimension, tag in gmsh.model.getEntities(0):
        value = np.asarray(gmsh.model.getValue(0, tag, []), dtype=float)
        if value.size == 3:
            tags.append(int(tag)); points.append(value)
    return tags, np.asarray(points, dtype=float).reshape(-1, 3) if points else np.empty((0, 3))


def _vertex_mesh_nodes(gmsh, local):
    result = {}
    for _dimension, tag in gmsh.model.getEntities(0):
        node_tags, _coords, _params = gmsh.model.mesh.getNodes(0, tag)
        if len(node_tags):
            result[int(tag)] = local(node_tags).astype(int)
    return result


def _volume_tetrahedra(gmsh, volumes: list[int], local):
    blocks, by_volume, offset = [], {}, 0
    missing = []
    for tag in volumes:
        types, _tags, node_blocks = gmsh.model.mesh.getElements(3, tag)
        tetra = next((local(np.asarray(nodes).reshape(-1, 4)) for typ, nodes in zip(types, node_blocks) if typ == 4), None)
        if tetra is None:
            missing.append((int(tag), [int(typ) for typ in types]))
            continue
        count = int(len(tetra)); start = int(offset)
        blocks.append(tetra)
        by_volume[int(tag)] = np.arange(start, start + count, dtype=np.int64)
        offset = start + count
        LOGGER.debug("volume tetra block: tag=%s count=%d offset_end=%d", tag, len(tetra), offset)
    if missing:
        raise RuntimeError(f"Gmsh produced no C3D4 tetrahedra for volumes: {missing}")
    return (np.concatenate(blocks, axis=0) if blocks else np.empty((0, 4), dtype=int)), by_volume


def _surface_triangles(gmsh, tag: int, local):
    types, _tags, node_blocks = gmsh.model.mesh.getElements(2, tag)
    blocks = [local(np.asarray(nodes).reshape(-1, 3)) for typ, nodes in zip(types, node_blocks) if typ == 2]
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, 3), dtype=int)


def _surface_curves(gmsh, surface_tag: int) -> list[tuple[int, int]]:
    """Return the OCC 1-D boundary curves of one BREP surface."""
    return sorted(set(gmsh.model.getBoundary(
        [(2, int(surface_tag))], oriented=False, recursive=False)))


def _curve_lines(gmsh, tag: int, local):
    """Return local two-node display segments for an OCC boundary curve."""
    types, _tags, node_blocks = gmsh.model.mesh.getElements(1, int(tag))
    blocks = [local(np.asarray(nodes).reshape(-1, 2))
              for typ, nodes in zip(types, node_blocks) if int(typ) == 1]
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, 2), dtype=int)


def _boundary_surfaces(gmsh, volumes: list[int]) -> list[tuple[int, int]]:
    return sorted(set(gmsh.model.getBoundary([(3, tag) for tag in volumes], oriented=False, recursive=False)))


def _tet_face_lookup(connectivity: np.ndarray, face_map: dict[int, list[int]]):
    result = {}
    for side, local_nodes in face_map.items():
        for element_id, nodes in enumerate(connectivity[:, local_nodes]):
            key = tuple(sorted(int(node) for node in nodes))
            if key in result:
                # An interior face occurs twice and must not be assigned to a
                # surface set.  Keep an explicit sentinel so a non-manifold
                # face cannot silently resolve to whichever element happened
                # to be visited last.
                result[key] = None
            else:
                result[key] = (element_id, side)
    return result


def _set_preview_size(gmsh, volumes: list[int]) -> None:
    if not volumes:
        raise RuntimeError("Cannot tessellate a BREP preview without a final OCC volume.")
    boxes = [gmsh.model.getBoundingBox(3, tag) for tag in volumes]
    if not boxes:
        raise RuntimeError("Cannot determine a BREP preview bounding box.")
    extent = max(max(box[3] - box[0], box[4] - box[1], box[5] - box[2]) for box in boxes)
    size = max(extent / 28.0, 1e-3)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)


def _mesh_with_topology_retry(gmsh, volumes: list[int], dimension: int) -> list[int]:
    """Generate a mesh, retrying through a topology-preserving STEP import.

    Some curved OCC booleans (notably a torus intersecting a box face) leave
    duplicate seam edges.  ``occ.healShapes()`` can repair the wire, but it
    may also reverse/remove a valid portion of a fused solid (turning a union
    into a difference).  A STEP export/import lets OpenCASCADE rebuild the
    topology while preserving the solid mass and boolean semantics.
    """
    try:
        gmsh.model.mesh.generate(int(dimension))
        return volumes
    except Exception as exc:
        if "closed loop" not in str(exc).lower():
            raise
        LOGGER.warning("Gmsh %d-D mesh hit an open OCC wire; rebuilding topology via STEP", dimension)
        source_mass = sum(float(gmsh.model.occ.getMass(3, int(tag))) for tag in volumes)
        gmsh.model.mesh.clear()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="torchfea_occ_", suffix=".step", delete=False) as handle:
                temp_path = handle.name
            gmsh.write(temp_path)
            gmsh.clear()
            gmsh.model.add("torchfea_occ_step_retry")
            imported = gmsh.model.occ.importShapes(temp_path)
            gmsh.model.occ.synchronize()
            retry_volumes = [int(tag) for dim, tag in imported if int(dim) == 3]
            if not retry_volumes:
                retry_volumes = [int(tag) for dim, tag in gmsh.model.getEntities(3)]
            if not retry_volumes:
                raise RuntimeError("STEP topology rebuild produced no OCC volume.")
            retry_mass = sum(float(gmsh.model.occ.getMass(3, tag)) for tag in retry_volumes)
            if source_mass and abs(retry_mass - source_mass) > max(1e-7, abs(source_mass) * 1e-7):
                raise RuntimeError(
                    f"STEP topology rebuild changed OCC volume ({source_mass:g} -> {retry_mass:g}).")
            gmsh.model.mesh.generate(int(dimension))
            LOGGER.info("OCC STEP topology retry succeeded: dimension=%d volumes=%s mass=%g",
                        dimension, retry_volumes, retry_mass)
            return retry_volumes
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError as exc:
                    LOGGER.warning("could not remove temporary OCC STEP file %s: %s", temp_path, exc)
