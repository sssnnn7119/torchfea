"""Native Gmsh OCC feature history, BREP preview, and finite-element meshing.

The workbench has two deliberately separate paths:

* :func:`preview_brep` tessellates OCC *faces only* for interactive CAD
  display. It never produces a torchfea ``Part`` or a 3-D FE mesh.
* :func:`create_part` applies one global mesh size, creates the final C3D4
  volume mesh, and maps Gmsh surface/volume entities back to torchfea sets.

Each operation owns its OCC implementation in ``operations.py``;
``OCCBuilder`` owns history dependency and entity-lifetime management. The UI
calls this kernel module and contains no CAD execution implementation.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import logging
import os
import threading
from typing import Iterable, Iterator

import numpy as np
import torch

from .builder import OCCBuilder
from .history import CADHistory
from .operations import CADOperation

LOGGER = logging.getLogger("torchfea.cae")


_GMSH_LOCK = threading.RLock()


@contextmanager
def occ_session(name: str) -> Iterator[OCCBuilder]:
    """Own one serialized Gmsh/OCC session for geometry, preview or meshing."""
    import gmsh

    with _GMSH_LOCK:
        if gmsh.isInitialized():
            raise RuntimeError("A managed OCC session cannot replace an active Gmsh model.")
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(name)
            yield OCCBuilder(gmsh)
        finally:
            gmsh.finalize()


def _operation_list(source: Iterable[CADOperation] | CADHistory) -> list[CADOperation]:
    if isinstance(source, CADHistory):
        return list(source.operations)
    return list(source)


def preview_brep(operations: Iterable[CADOperation] | CADHistory) -> list[dict]:
    """Return face-wise display geometry for the final native OCC solid.

    Native OpenCASCADE triangles are visualization tessellation,
    not the analysis mesh. Each record keeps its BREP surface and volume tags
    so selections are stable until volume meshing.
    """
    import gmsh

    operations = _operation_list(operations)
    LOGGER.info("BREP preview start: operations=%d kinds=%s", len(operations),
                [operation.kind for operation in operations])

    _GMSH_LOCK.acquire()
    try:
        # Gmsh's default interrupt handler registration uses signal.signal(),
        # which Python only permits in the main thread.  BREP previews can be
        # requested from the Qt thread, while FE generation runs in the solve
        # worker, so disable that optional handler and serialize the global
        # Gmsh API between the two paths.
        gmsh.initialize(interruptible=False)
    except Exception:
        _GMSH_LOCK.release()
        raise
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear(); gmsh.model.add("torchfea_brep_preview")
        volumes = OCCBuilder(gmsh).replay(operations)
        LOGGER.debug("BREP OCC history built: final_volumes=%s", volumes)
        linear_deflection = _preview_deflection(gmsh, volumes)
        from .tessellation import triangulate_faces

        surfaces = _boundary_surfaces(gmsh, volumes)
        face_edges = {tag: [edge_tag for _, edge_tag in _surface_curves(gmsh, tag)]
                      for _, tag in surfaces}
        points, face_triangles, edge_polylines = triangulate_faces(
            gmsh, face_edges, linear_deflection,
        )
        vertex_tags, vertex_points = _occ_vertices(gmsh)
        records: list[dict] = []
        for _dimension, surface_tag in surfaces:
            triangles = face_triangles[surface_tag]
            if not len(triangles):
                raise RuntimeError(
                    f"OpenCASCADE produced no display triangles for BREP surface {surface_tag}.")
            owners = [tag for tag in volumes if (2, surface_tag) in gmsh.model.getBoundary([(3, tag)], False, False)]
            if not owners:
                raise RuntimeError(
                    f"BREP surface {surface_tag} is not owned by a final volume.")
            edge_lines = []
            for edge_tag in face_edges[surface_tag]:
                lines = edge_polylines[edge_tag]
                if len(lines):
                    edge_lines.append({"edge_tag": int(edge_tag), "lines": lines})
            records.append({"points": points, "triangles": triangles, "surface_tag": int(surface_tag),
                            "volume_tags": [int(tag) for tag in owners],
                            "edge_lines": edge_lines,
                            "vertex_tags": vertex_tags, "vertex_points": vertex_points})
        LOGGER.info("BREP preview complete: faces=%d display_nodes=%d", len(records), len(points))
        return records
    except Exception as exc:
        LOGGER.exception("BREP preview failed")
        raise RuntimeError(f"CAD preview failed: {exc}") from exc
    finally:
        try:
            gmsh.finalize()
        finally:
            _GMSH_LOCK.release()


def preview_discrete_part(part) -> list[dict]:
    """Return display faces for an already-meshed native Part.

    A discrete Part has no OCC topology to tessellate. Its boundary element
    faces are therefore used directly as the viewport surface. The records
    carry ``discrete=True`` so the UI never treats them as CAD faces.
    """
    points = part.nodes.detach().cpu().numpy()
    records: list[dict] = []
    surface_tag = 0
    for element_name, element in part.elems.items():
        extractor = element.extract_boundary_surface_set
        face_map = element.surfaceid_map
        for element_ids, side in extractor():
            ids = np.asarray(element_ids, dtype=int).reshape(-1)
            if not len(ids):
                continue
            local_nodes = face_map[int(side)]
            face_nodes = element._elems[ids][:, local_nodes].detach().cpu().numpy()
            if face_nodes.shape[1] == 3:
                triangles = face_nodes
            elif face_nodes.shape[1] > 3:
                triangles = np.concatenate([
                    np.column_stack((face_nodes[:, 0], face_nodes[:, index],
                                     face_nodes[:, index + 1]))
                    for index in range(1, face_nodes.shape[1] - 1)
                ], axis=0)
            else:
                raise ValueError(f"Element block '{element_name}' contains a degenerate boundary face.")
            edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]), axis=0)
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            records.append({
                "points": points,
                "triangles": triangles,
                "surface_tag": surface_tag,
                "volume_tags": [],
                "edge_lines": [{"edge_tag": surface_tag, "lines": edges}],
                "vertex_tags": [],
                "vertex_points": np.empty((0, 3), dtype=float),
                "surface_name": str(part.exterior_surface),
                "discrete": True,
            })
            surface_tag += 1
    if not records:
        raise RuntimeError("Discrete Part has no supported boundary faces to display.")
    LOGGER.info("discrete preview complete: faces=%d display_nodes=%d", len(records), len(points))
    return records


def export_step(operations: Iterable[CADOperation] | CADHistory, path: str) -> None:
    """Replay a Part's OCC history and export its final solids as STEP.

    STEP export intentionally does not call ``mesh.generate``: the file is a
    CAD/BREP artifact and is independent of the FE mesh size.
    """
    import gmsh

    output_path = os.fspath(path)
    if not output_path:
        raise ValueError("STEP export path cannot be empty.")
    operations = _operation_list(operations)
    LOGGER.info("STEP export start: path=%s operations=%d", output_path, len(operations))

    _GMSH_LOCK.acquire()
    try:
        gmsh.initialize(interruptible=False)
    except Exception:
        _GMSH_LOCK.release()
        raise
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("torchfea_step_export")
        volumes = OCCBuilder(gmsh).replay(operations)
        if not volumes:
            raise RuntimeError("Part has no final OCC volume to export.")
        gmsh.write(output_path)
        LOGGER.info("STEP export complete: path=%s volumes=%s", output_path, volumes)
    except Exception:
        LOGGER.exception("STEP export failed: path=%s", output_path)
        raise
    finally:
        try:
            gmsh.finalize()
        finally:
            _GMSH_LOCK.release()


def create_part(operations: Iterable[CADOperation] | CADHistory, mesh_size: float):
    """Mesh final OCC volumes and resolve any CAD history set definitions."""
    import gmsh

    from ..model.part import Part
    from ..model.elements import initialize_element

    history = operations if isinstance(operations, CADHistory) else None
    operations = _operation_list(operations)
    mesh_size = float(mesh_size)
    LOGGER.info("FE mesh start: operations=%d mesh_size=%r (%s)", len(operations), mesh_size, type(mesh_size).__name__)
    if mesh_size <= 0:
        raise ValueError("Global mesh size must be positive.")
    _GMSH_LOCK.acquire()
    try:
        # See preview_brep(): this call may run in the Qt solve worker.
        gmsh.initialize(interruptible=False)
    except Exception:
        _GMSH_LOCK.release()
        raise
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear(); gmsh.model.add("torchfea_analysis_mesh")
        volumes = OCCBuilder(gmsh).replay(operations)
        LOGGER.debug("FE OCC history built: final_volumes=%s", volumes)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        LOGGER.debug("Gmsh 3D mesh.generate(3) begin")
        gmsh.model.mesh.generate(3)
        LOGGER.debug("Gmsh 3D mesh.generate(3) complete")
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points, local = _local_nodes(node_tags, node_coords)
        vertex_nodes = _vertex_mesh_nodes(gmsh, local)
        curve_nodes = {
            int(tag): local(gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)[0]).astype(int)
            for _, tag in gmsh.model.getEntities(1)
        }
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
        try:
            gmsh.finalize()
        finally:
            _GMSH_LOCK.release()

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
    # Runtime maps used to resolve CADHistory selectors.  They are excluded
    # from Part serialization; the CAD history is the persistent definition.
    part._gmsh_surface_sets = gmsh_surface_sets
    part._gmsh_surface_nodes = surface_nodes
    part._gmsh_volume_elements = volume_elements
    part._gmsh_vertex_nodes = vertex_nodes
    part._gmsh_curve_nodes = curve_nodes
    if history is not None:
        _apply_history_sets(part, history)
    else:
        part.exterior_surface = "extern"
    LOGGER.info("FE mesh complete: torchfea nodes=%s elements=%s generated_sets=%s",
                tuple(part.nodes.shape),
                {name: tuple(element._elems.shape) for name, element in part.elems.items()},
                {"node": ["all", "surface_all"],
                 "element": ["all"]})
    return part


def _apply_history_sets(part, history: CADHistory) -> None:
    """Resolve CAD topology selections onto the freshly generated FE mesh."""
    external_name = str(getattr(history, "exterior_surface", "extern")).strip() or "extern"
    if external_name != "extern":
        if external_name in part.surfaces:
            raise ValueError(
                f"CAD exterior surface '{external_name}' conflicts with an existing surface.")
        part.surfaces[external_name] = part.surfaces["extern"]
        del part.surfaces._surface_dict["extern"]
    part.exterior_surface = external_name

    def resolve(selector: str, kind: str):
        fields = str(selector).split(":", 2)
        if len(fields) == 2 and fields[0] == "cad" and fields[1] == "all":
            if kind == "surface":
                return part.surfaces[part.exterior_surface]
            if kind == "node":
                return np.arange(len(part.nodes), dtype=int)
            return np.arange(sum(len(element._elems) for element in part.elems.values()), dtype=int)
        if len(fields) != 3 or fields[0] != "cad":
            raise ValueError(f"Invalid CAD selector '{selector}'.")
        selector_kind, raw_tag = fields[1], fields[2]
        try:
            tag = int(raw_tag)
        except ValueError as exc:
            raise ValueError(f"Invalid CAD selector '{selector}'.") from exc
        maps = {
            ("surface", "surface"): part._gmsh_surface_sets,
            ("node", "surface"): part._gmsh_surface_nodes,
            ("node", "vertex"): part._gmsh_vertex_nodes,
            ("node", "curve"): part._gmsh_curve_nodes,
            ("element", "volume"): part._gmsh_volume_elements,
        }
        try:
            return maps[(kind, selector_kind)][tag]
        except KeyError as exc:
            raise ValueError(
                f"CAD selector '{selector}' does not identify a final {kind} entity.") from exc

    for selection in getattr(history, "sets", []):
        values = resolve(selection.selector, selection.kind)
        if selection.kind == "surface":
            if selection.name in part.surfaces:
                raise ValueError(f"CAD surface set '{selection.name}' already exists.")
            part.add_surface_set(selection.name, values)
            face_nodes = []
            for element_ids, side in values:
                element = next(iter(part.elems.values()))
                face_nodes.append(
                    element._elems[np.asarray(element_ids, dtype=int)][:,
                    element.surfaceid_map[int(side)]].detach().cpu().numpy().ravel())
            generated_name = f"surface_{selection.name}_nodes"
            if generated_name in part.set_nodes:
                raise ValueError(
                    f"CAD surface set '{selection.name}' conflicts with its generated node set.")
            part.add_node_set(
                generated_name,
                np.unique(np.concatenate(face_nodes)),
            )
        elif selection.kind == "node":
            if selection.name in {"all", "surface_all"} or selection.name in part.set_nodes:
                raise ValueError(f"CAD node set '{selection.name}' is reserved or duplicated.")
            part.add_node_set(selection.name, np.asarray(values, dtype=int).ravel())
        else:
            if selection.name == "all" or selection.name in part.set_elements:
                raise ValueError(f"CAD element set '{selection.name}' is reserved or duplicated.")
            part.add_element_set(selection.name, np.asarray(values, dtype=int).ravel())


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


def _preview_deflection(gmsh, volumes: list[int]) -> float:
    """Return the native display chord tolerance from the CAD extent."""
    if not volumes:
        raise RuntimeError("Cannot tessellate a BREP preview without a final OCC volume.")
    boxes = [gmsh.model.getBoundingBox(3, tag) for tag in volumes]
    if not boxes:
        raise RuntimeError("Cannot determine a BREP preview bounding box.")
    extent = max(max(box[3] - box[0], box[4] - box[1], box[5] - box[2]) for box in boxes)
    return max(extent / 1000.0, 1e-8)
