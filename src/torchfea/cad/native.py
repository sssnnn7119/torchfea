"""Native OCC geometry transfer, subshapes and exact geometric signatures."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FORWARD, TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape


def read_native_geometry(gmsh: ModuleType) -> TopoDS_Shape:
    """Transfer exact geometry from the current Gmsh OCC model."""
    with TemporaryDirectory(prefix="torchfea_native_") as directory:
        path = Path(directory) / "geometry.brep"
        gmsh.write(str(path))
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise RuntimeError("Cannot read the native CAD geometry.")
    return shape


def _shape_signature(shape: TopoDS_Shape, dimension: int) -> np.ndarray:
    """Use the same exact mass properties as Gmsh's OCC topology identities."""
    properties = GProp_GProps()
    if dimension == 0:
        point = BRep_Tool.Pnt_s(TopoDS.Vertex(shape))
        return np.array([point.X(), point.Y(), point.Z()])
    if dimension == 3:
        BRepGProp.VolumeProperties_s(shape, properties)
    elif dimension == 2:
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

