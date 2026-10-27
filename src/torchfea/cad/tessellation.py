"""Native OCC display triangulation, independent of finite-element meshing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape


def _shape_signature(shape: TopoDS_Shape, dimension: int) -> np.ndarray:
    """Use the same exact mass properties as Gmsh's OCC topology identities."""
    properties = GProp_GProps()
    if dimension == 2:
        BRepGProp.SurfaceProperties_s(shape, properties)
    else:
        BRepGProp.LinearProperties_s(shape, properties)
    center = properties.CentreOfMass()
    inertia = properties.MatrixOfInertia()
    return np.array([
        properties.Mass(), center.X(), center.Y(), center.Z(),
        *(inertia.Value(row, column) for row in range(1, 4) for column in range(1, 4)),
    ])


def _subshapes(shape: TopoDS_Shape, kind: TopAbs_ShapeEnum) -> list[TopoDS_Shape]:
    """List unique topological objects, retaining their original orientation."""
    shapes: dict[TopoDS_Shape, TopoDS_Shape] = {}
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        item = explorer.Current()
        shapes.setdefault(item.Oriented(TopAbs_FORWARD), item)
        explorer.Next()
    return list(shapes.values())


def triangulate_faces(
    gmsh: ModuleType,
    face_edges: Mapping[int, Sequence[int]],
    linear_deflection: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build native display faces and edges while retaining CAD entity tags.

    Gmsh transfers exact BREP geometry and topology identities only. All
    display coordinates, triangles and boundary polylines come from OCC.
    """
    with TemporaryDirectory(prefix="torchfea_preview_") as directory:
        path = Path(directory) / "geometry.brep"
        gmsh.write(str(path))
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise RuntimeError("Cannot read the native CAD preview geometry.")

    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.15, True)
    if not mesher.IsDone():
        raise RuntimeError("OpenCASCADE could not triangulate the CAD preview.")

    faces = [TopoDS.Face(item) for item in _subshapes(shape, TopAbs_FACE)]
    signatures = np.array([_shape_signature(face, 2) for face in faces])
    used: set[int] = set()
    point_blocks: list[np.ndarray] = []
    offset = 0
    triangles: dict[int, np.ndarray] = {}
    lines: dict[int, np.ndarray] = {}
    for tag, edge_tags in face_edges.items():
        signature = np.array([
            gmsh.model.occ.getMass(2, tag),
            *gmsh.model.occ.getCenterOfMass(2, tag),
            *gmsh.model.occ.getMatrixOfInertia(2, tag),
        ])
        matches = np.flatnonzero(np.all(np.isclose(
            signatures, signature, rtol=1e-7, atol=1e-8,
        ), axis=1))
        index = next((int(index) for index in matches if int(index) not in used), None)
        if index is None:
            raise RuntimeError(f"Cannot associate display triangles with CAD face {tag}.")
        used.add(index)
        face = faces[index]
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is None or mesh.NbTriangles() == 0:
            raise RuntimeError(f"CAD face {tag} has no display triangles.")
        transform = location.Transformation()
        coordinates = []
        for node in range(1, mesh.NbNodes() + 1):
            point = mesh.Node(node).Transformed(transform)
            coordinates.append((point.X(), point.Y(), point.Z()))
        cells = np.array([mesh.Triangle(cell).Get()
                          for cell in range(1, mesh.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            cells = cells[:, [0, 2, 1]]
        triangles[int(tag)] = cells + offset
        point_blocks.append(np.asarray(coordinates, dtype=float))

        edges = [TopoDS.Edge(item) for item in _subshapes(face, TopAbs_EDGE)
                 if not BRep_Tool.Degenerated_s(TopoDS.Edge(item))]
        edge_signatures = np.array([_shape_signature(edge, 1) for edge in edges])
        for edge_tag in edge_tags:
            if edge_tag in lines:
                continue
            mass = gmsh.model.occ.getMass(1, edge_tag)
            if mass == 0.0:
                lines[edge_tag] = np.empty((0, 2), dtype=np.int64)
                continue
            signature = np.array([
                mass, *gmsh.model.occ.getCenterOfMass(1, edge_tag),
                *gmsh.model.occ.getMatrixOfInertia(1, edge_tag),
            ])
            matches = np.flatnonzero(np.all(np.isclose(
                edge_signatures, signature, rtol=1e-7, atol=1e-8,
            ), axis=1))
            if not len(matches):
                raise RuntimeError(f"Cannot associate display polyline with CAD edge {edge_tag}.")
            polygon = BRep_Tool.PolygonOnTriangulation_s(edges[int(matches[0])], mesh, location)
            if polygon is None:
                raise RuntimeError(f"CAD edge {edge_tag} has no display polyline.")
            nodes = np.array([polygon.Node(node) for node in range(1, polygon.NbNodes() + 1)],
                             dtype=np.int64) - 1 + offset
            lines[edge_tag] = np.column_stack((nodes[:-1], nodes[1:]))
        offset += len(coordinates)
    return np.concatenate(point_blocks), triangles, lines
