"""OCC boundary topology and geometry-preserving face conversion.

Functions operate inside an active ``occ_session``. Entity tags are session
handles; names and geometric signatures are the persistent identities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gmsh
import numpy as np


@dataclass(frozen=True)
class BREPVertex:
    name: str
    coordinates: tuple[float, float, float]
    edges: tuple[str, ...]


@dataclass(frozen=True)
class BREPEdge:
    name: str
    geometry_type: str
    vertices: tuple[str, ...]
    faces: tuple[str, ...]
    signature: tuple[float, ...]
    parameter_bounds: tuple[float, float]


@dataclass(frozen=True)
class BREPFace:
    name: str
    geometry_type: str
    loops: tuple[tuple[str, ...], ...]
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]]
    signature: tuple[float, ...]

    @property
    def is_nurbs(self) -> bool:
        return "bspline" in self.geometry_type.casefold()


@dataclass(frozen=True)
class BREPTopology:
    faces: dict[str, BREPFace]
    edges: dict[str, BREPEdge]
    vertices: dict[str, BREPVertex]
    face_tags: dict[str, int]
    edge_tags: dict[str, int]
    vertex_tags: dict[str, int]


def entity_signature(dimension: int, tag: int) -> tuple[float, ...]:
    """Describe a point or a geometric curve/face independently of OCC tags."""
    if dimension == 0:
        return tuple(map(float, gmsh.model.getValue(0, tag, [])))
    return (
        float(gmsh.model.occ.getMass(dimension, tag)),
        *map(float, gmsh.model.occ.getCenterOfMass(dimension, tag)),
        *map(float, gmsh.model.occ.getMatrixOfInertia(dimension, tag)),
    )


def match_entity_names(
    dimension: int,
    tags: Sequence[int],
    signatures: Mapping[str, Sequence[float]],
    prefix: str,
) -> dict[int, str]:
    """Restore a bijective name map for unchanged geometry, or assign names."""
    tags = sorted(set(map(int, tags)))
    if not signatures:
        return {tag: f"{prefix}_{index}" for index, tag in enumerate(tags)}
    if len(tags) != len(signatures):
        raise ValueError(f"{prefix} topology differs from its saved name map.")
    values = np.asarray([entity_signature(dimension, tag) for tag in tags])
    names: dict[int, str] = {}
    for name, signature in signatures.items():
        matches = np.flatnonzero(np.all(np.isclose(values, signature, rtol=1e-7, atol=1e-8), axis=1))
        if len(matches) != 1 or tags[int(matches[0])] in names:
            raise ValueError(f"Cannot uniquely identify {name} in the imported BREP.")
        names[tags[int(matches[0])]] = name
    return names


def read_brep_topology(
    face_tags: Sequence[int],
    face_map: Mapping[str, Sequence[float]] | None = None,
    edge_map: Mapping[str, Sequence[float]] | None = None,
    vertex_map: Mapping[str, Sequence[float]] | None = None,
    face_loops: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> BREPTopology:
    """Read face wires, shared edges and vertices without generating a mesh."""
    face_names = match_entity_names(2, face_tags, {} if face_loops else face_map or {}, "face")
    loops: dict[int, tuple[tuple[int, ...], ...]] = {}
    for tag in face_names:
        _, boundaries = gmsh.model.occ.getCurveLoops(tag)
        loops[tag] = tuple(tuple(map(int, boundary)) for boundary in boundaries)
    edge_tags = sorted({abs(tag) for boundaries in loops.values() for loop in boundaries for tag in loop})
    edge_names = match_entity_names(1, edge_tags, edge_map or {}, "edge")
    if face_loops:
        # Rational reparameterization changes OCC's default mass quadrature.
        # The unchanged boundary graph identifies faces without those integrals.
        def boundary_key(wires: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
            return tuple(sorted(tuple(sorted(wire)) for wire in wires))

        boundaries = {
            tag: boundary_key([[edge_names[abs(edge)] for edge in wire]
                               for wire in loops[tag]]) for tag in face_names
        }
        restored: dict[int, str] = {}
        for name, wires in face_loops.items():
            candidates = [tag for tag, key in boundaries.items() if key == boundary_key(wires)]
            if len(candidates) != 1 or candidates[0] in restored:
                raise ValueError(f"Cannot uniquely identify boundary of {name} after conversion.")
            restored[candidates[0]] = name
        if len(restored) != len(face_names):
            raise ValueError("Face topology changed during NURBS conversion.")
        face_names = restored
    edge_vertices = {
        tag: tuple(abs(int(vertex)) for _, vertex in gmsh.model.getBoundary(
            [(1, tag)], combined=False, oriented=True, recursive=False,
        )) for tag in edge_tags
    }
    vertex_tags = sorted({tag for values in edge_vertices.values() for tag in values})
    vertex_names = match_entity_names(0, vertex_tags, vertex_map or {}, "vertex")
    faces: dict[str, BREPFace] = {}
    for tag, name in face_names.items():
        lower, upper = gmsh.model.getParametrizationBounds(2, tag)
        faces[name] = BREPFace(
            name, gmsh.model.getType(2, tag),
            tuple(tuple(edge_names[abs(edge)] for edge in loop) for loop in loops[tag]),
            ((float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))),
            entity_signature(2, tag),
        )
    edges = {
        name: BREPEdge(
            name, gmsh.model.getType(1, tag),
            tuple(vertex_names[vertex] for vertex in edge_vertices[tag]),
            tuple(face_name for face_tag, face_name in face_names.items()
                  if any(abs(edge) == tag for loop in loops[face_tag] for edge in loop)),
            entity_signature(1, tag),
            tuple(float(bound[0]) for bound in gmsh.model.getParametrizationBounds(1, tag)),
        ) for tag, name in edge_names.items()
    }
    vertices = {
        name: BREPVertex(
            name, entity_signature(0, tag),
            tuple(edge_names[edge] for edge in edge_tags if tag in edge_vertices[edge]),
        ) for tag, name in vertex_names.items()
    }
    return BREPTopology(
        faces, edges, vertices,
        {name: tag for tag, name in face_names.items()},
        {name: tag for tag, name in edge_names.items()},
        {name: tag for tag, name in vertex_names.items()},
    )


def _plane_to_nurbs(tag: int) -> int:
    """Create an exact bilinear support plane with the original trimmed wires."""
    lower, upper = gmsh.model.getParametrizationBounds(2, tag)
    corners = [(lower[0], lower[1]), (upper[0], lower[1]),
               (lower[0], upper[1]), (upper[0], upper[1])]
    points = [gmsh.model.occ.addPoint(*map(float, gmsh.model.getValue(2, tag, uv))) for uv in corners]
    wires, boundaries = gmsh.model.occ.getCurveLoops(tag)
    origin = np.asarray(gmsh.model.getValue(2, tag, [0.0, 0.0]))
    du, dv = np.asarray(gmsh.model.getDerivative(2, tag, [0.0, 0.0])).reshape(2, 3)
    transform = np.eye(4)
    transform[:3, :3] = np.linalg.inv(np.column_stack((du, dv, np.cross(du, dv))))
    transform[:3, 3] = -transform[:3, :3] @ origin
    # Identify the outer wire by its enclosed planar area; wire order in a
    # STEP file does not specify which loop is exterior.
    areas: list[tuple[float, int]] = []
    for wire, edges in zip(wires, boundaries, strict=True):
        temporary = gmsh.model.occ.addPlaneSurface([int(wire)])
        area = float(gmsh.model.occ.getMass(2, temporary))
        gmsh.model.occ.remove([(2, temporary)], recursive=False)
        # Map exact planar curves into UV. Projecting 3D circles onto a
        # B-spline support approximates them and can open the sewn shell.
        copies = gmsh.model.occ.copy([(1, int(edge)) for edge in edges])
        gmsh.model.occ.affineTransform(copies, transform.ravel())
        wire_uv = gmsh.model.occ.addWire([edge for _, edge in copies], checkClosed=True)
        areas.append((area, wire_uv))
    ordered = [wire for _, wire in sorted(areas, reverse=True)]
    result = gmsh.model.occ.addBSplineSurface(
        points, numPointsU=2, degreeU=1, degreeV=1,
        weights=[1.0] * 4, knotsU=[float(lower[0]), float(upper[0])],
        knotsV=[float(lower[1]), float(upper[1])],
        multiplicitiesU=[2, 2], multiplicitiesV=[2, 2],
        wireTags=ordered, wire3D=False,
    )
    gmsh.model.occ.remove([(2, tag)], recursive=False)
    return int(result)


def convert_faces_to_nurbs(face_tags: Sequence[int]) -> tuple[int, ...]:
    """Convert detached faces, preserving their outer wires and holes.

    Owners must remove their volumes before conversion and reassemble the
    returned faces into a sewn shell afterwards. OCC's native conversion keeps
    planes analytic, so planar faces receive an explicit B-spline support.
    """
    converted: list[int] = []
    for raw_tag in face_tags:
        tag = int(raw_tag)
        geometry_type = gmsh.model.getType(2, tag).casefold()
        if geometry_type == "plane":
            tag = _plane_to_nurbs(tag)
        elif "bspline" not in geometry_type:
            gmsh.model.occ.convertToNURBS([(2, tag)])
        gmsh.model.occ.synchronize()
        if "bspline" not in gmsh.model.getType(2, tag).casefold():
            raise ValueError(f"OCC did not convert face {raw_tag} to a NURBS surface.")
        converted.append(tag)
    return tuple(converted)


def remove_unused_brep_entities(volume: int) -> None:
    """Keep one solid and its complete boundary graph in a private OCC model."""
    keep: dict[int, set[int]] = {3: {volume}}
    for dimension in (2, 1, 0):
        keep[dimension] = {
            abs(int(tag)) for _, tag in gmsh.model.getBoundary(
                [(dimension + 1, tag) for tag in keep[dimension + 1]],
                combined=False, oriented=False, recursive=False,
            )
        }
    for dimension in (3, 2, 1, 0):
        unused = [(dim, tag) for dim, tag in gmsh.model.occ.getEntities(dimension)
                  if tag not in keep[dimension]]
        if unused:
            gmsh.model.occ.remove(unused, recursive=False)
    gmsh.model.occ.synchronize()
