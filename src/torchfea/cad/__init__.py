"""Kernel CAD construction API.

The public API is intentionally usable without the CAE UI::

    from torchfea import FEAController
    from torchfea.cad import Box, Cylinder, Fuse

    fe = FEAController()
    cad = fe.cad.add_part("Part-1")
    cad.add(Box("Box-1", dx=10, dy=5, dz=2))
    cad.add(Cylinder("Cylinder-1", x=5, radius=1))
    cad.add(Fuse("Fuse-1", "Box-1", "Cylinder-1"))
    part = fe.cad.mesh_part("Part-1", mesh_size=1.0)

Each feature is an explicit serializable class. ``FEAController.cad`` owns
the named histories used to construct the finite-element assembly.
"""

from .operations import (
    BooleanOperation,
    Box,
    CADOperation,
    Cone,
    Cut,
    Cylinder,
    Dilate,
    Fragment,
    Fuse,
    Intersect,
    Mirror,
    PrimitiveOperation,
    Rotate,
    Sphere,
    StepImport,
    Torus,
    TransformOperation,
    Translate,
    Wedge,
)
from .history import CADModel, CADHistory, CADSetDefinition
from .occ import (
    create_part,
    export_step,
    preview_discrete_part,
    preview_brep,
)
__all__ = [
    "CADModel",
    "CADHistory",
    "CADSetDefinition",
    "CADOperation",
    "PrimitiveOperation",
    "BooleanOperation",
    "TransformOperation",
    "Box",
    "Wedge",
    "Cylinder",
    "Sphere",
    "Cone",
    "Torus",
    "StepImport",
    "Fuse",
    "Cut",
    "Intersect",
    "Fragment",
    "Translate",
    "Rotate",
    "Dilate",
    "Mirror",
    "create_part",
    "export_step",
    "preview_discrete_part",
    "preview_brep",
]
