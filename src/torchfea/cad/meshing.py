"""Geometry-preserving periodic-face preparation for Gmsh volume meshing."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
from OCP.BRepTools import BRepTools
from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopoDS import TopoDS_Shape

from .native import _shape_signature, _subshapes, read_native_geometry

_KINDS = (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID)
EntityMap = dict[tuple[int, int], tuple[int, ...]]


def _match_entities(gmsh: ModuleType, shape: TopoDS_Shape) -> dict[tuple[int, int], TopoDS_Shape]:
    """Associate exact OCC subshapes with current session entity tags."""
    result = {}
    for dimension, kind in enumerate(_KINDS):
        shapes = _subshapes(shape, kind)
        signatures = np.array([_shape_signature(item, dimension) for item in shapes])
        used = set()
        for _, tag in gmsh.model.getEntities(dimension):
            signature = (gmsh.model.getValue(0, tag, []) if dimension == 0 else [
                gmsh.model.occ.getMass(dimension, tag),
                *gmsh.model.occ.getCenterOfMass(dimension, tag),
                *gmsh.model.occ.getMatrixOfInertia(dimension, tag),
            ])
            matches = np.flatnonzero(np.all(np.isclose(
                signatures, signature, rtol=1e-7, atol=1e-8,
            ), axis=1))
            index = next((int(index) for index in matches if int(index) not in used), None)
            if index is None:
                raise RuntimeError(f"Cannot match CAD entity ({dimension}, {tag}) for meshing.")
            used.add(index)
            result[dimension, int(tag)] = shapes[index]
    return result


def split_closed_faces(gmsh: ModuleType, volumes: list[int]) -> tuple[list[int], EntityMap]:
    """Split periodic CAD faces for meshing and retain every original entity group.

    Splitting changes face topology only. Result sets are grouped back under
    their source tags so CAD face/edge/vertex selections remain unchanged.
    """
    shape = read_native_geometry(gmsh)
    divider = ShapeUpgrade_ShapeDivideClosed(shape)
    divider.SetNbSplitPoints(1)
    if not divider.Perform():
        return volumes, {}
    original = _match_entities(gmsh, shape)
    prepared = divider.Result()
    context = divider.GetContext()
    with TemporaryDirectory(prefix="torchfea_meshing_") as directory:
        path = Path(directory) / "partitioned.brep"
        BRepTools.Write_s(prepared, str(path))
        gmsh.clear()
        gmsh.model.add("torchfea_analysis_mesh")
        imported = gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
    modified = {}
    for dimension in range(4):
        entities = gmsh.model.getEntities(dimension)
        signatures = [
            gmsh.model.getValue(0, tag, []) if dimension == 0 else [
                gmsh.model.occ.getMass(dimension, tag),
                *gmsh.model.occ.getCenterOfMass(dimension, tag),
                *gmsh.model.occ.getMatrixOfInertia(dimension, tag),
            ] for _, tag in entities
        ]
        modified[dimension] = ([tag for _, tag in entities], np.array(signatures))
    mapping: EntityMap = {}
    for (dimension, tag), item in original.items():
        replacement = context.Apply(item)
        children = [] if replacement.IsNull() else _subshapes(replacement, _KINDS[dimension])
        targets = set()
        tags, signatures = modified[dimension]
        for child in children:
            matches = np.flatnonzero(np.all(np.isclose(
                signatures, _shape_signature(child, dimension), rtol=1e-7, atol=1e-8,
            ), axis=1))
            if not len(matches):
                raise RuntimeError(f"Cannot restore CAD entity ({dimension}, {tag}) after splitting.")
            targets.update(tags[int(index)] for index in matches)
        mapping[dimension, tag] = tuple(sorted(targets))
    return [int(tag) for dimension, tag in imported if dimension == 3], mapping


def restore_entity_arrays(
    arrays: dict[int, np.ndarray], dimension: int, mapping: EntityMap,
) -> dict[int, np.ndarray]:
    """Aggregate mesh data from split entities under their original CAD tags."""
    if not mapping:
        return arrays
    template = next(iter(arrays.values()))
    result = {}
    for (kind, tag), targets in mapping.items():
        if kind != dimension:
            continue
        blocks = [arrays[target] for target in targets if target in arrays]
        result[tag] = (np.unique(np.concatenate(blocks), axis=0) if blocks
                       else np.empty((0, *template.shape[1:]), dtype=template.dtype))
    return result
