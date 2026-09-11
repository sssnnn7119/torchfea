"""Kernel CAD construction API.

The public API is intentionally usable without the CAE UI::

    from torchfea import FEAController

    fe = FEAController()
    cad = fe.cad.add_part("Part-1")
    box = cad.add_box(dx=10, dy=5, dz=2)
    cylinder = cad.add_cylinder(x=5, radius=1)
    cad.fuse(box.id, cylinder.id)
    part = fe.cad.mesh_part("Part-1", mesh_size=1.0)

``CADHistory`` and ``CADOperation`` are serializable discrete construction
records.  ``FEAController.cad`` owns the named histories used to construct
the finite-element assembly.
"""

from .history import CADModel, CADHistory, CADOperation
from .occ import (
    create_part,
    export_step,
    preview_discrete_part,
    preview_brep,
    primitive_defaults,
    transform_defaults,
)
__all__ = [
    "CADModel",
    "CADHistory",
    "CADOperation",
    "create_part",
    "export_step",
    "preview_discrete_part",
    "preview_brep",
    "primitive_defaults",
    "transform_defaults",
]
