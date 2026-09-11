"""Serializable CAD feature histories used by the finite-element kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ..interfaces import Serializable


@dataclass
class CADOperation(Serializable):
    """One native OCC operation in an ordered Part history.

    The operation graph is deliberately data-only.  It can therefore be
    authored from Python, edited by the CAE UI, serialized with the normal
    torchfea object serializer, and replayed by the kernel's Gmsh adapter.
    """

    id: str
    kind: str
    values: dict[str, object] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


class CADHistory(Serializable):
    """Ordered, replayable OCC history for one discrete Part definition."""

    _serialized_attributes = ["operations", "mesh_size"]

    def __init__(self, operations: Iterable[CADOperation] | None = None,
                 mesh_size: float | None = None):
        self.operations: list[CADOperation] = list(operations or [])
        # This is the last global FE mesh size used for this history.  It is
        # metadata rather than a CAD feature, but retaining it makes a saved
        # Part's geometry-and-meshing provenance explicit.
        self.mesh_size = None if mesh_size is None else float(mesh_size)

    def add(self, operation: CADOperation) -> CADOperation:
        if not isinstance(operation, CADOperation):
            raise TypeError("CAD history entries must be CADOperation objects.")
        self.operations.append(operation)
        return operation

    def _next_id(self, kind: str) -> str:
        prefix = kind.title()
        used = {operation.id for operation in self.operations}
        index = 1
        while f"{prefix}-{index}" in used:
            index += 1
        return f"{prefix}-{index}"

    def add_primitive(self, kind: str, **values) -> CADOperation:
        from .occ import primitive_defaults

        try:
            defaults = primitive_defaults(kind)
        except KeyError as exc:
            raise ValueError(f"Unsupported CAD primitive '{kind}'.") from exc
        defaults.update(values)
        return self.add(CADOperation(self._next_id(kind), kind, defaults))

    def add_box(self, x=0.0, y=0.0, z=0.0, dx=20.0, dy=10.0, dz=5.0, **values):
        return self.add_primitive("box", x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, **values)

    def add_wedge(self, x=0.0, y=0.0, z=0.0, dx=20.0, dy=10.0, dz=5.0, ltx=0.0, **values):
        return self.add_primitive("wedge", x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, ltx=ltx, **values)

    def add_cylinder(self, x=0.0, y=0.0, z=0.0, dx=0.0, dy=0.0, dz=20.0,
                     radius=5.0, **values):
        return self.add_primitive(
            "cylinder", x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, radius=radius, **values)

    def add_sphere(self, x=0.0, y=0.0, z=0.0, radius=5.0, **values):
        return self.add_primitive("sphere", x=x, y=y, z=z, radius=radius, **values)

    def add_cone(self, x=0.0, y=0.0, z=0.0, dx=0.0, dy=0.0, dz=20.0,
                 radius1=5.0, radius2=1.0, **values):
        return self.add_primitive(
            "cone", x=x, y=y, z=z, dx=dx, dy=dy, dz=dz,
            radius1=radius1, radius2=radius2, **values)

    def add_torus(self, x=0.0, y=0.0, z=0.0, major_radius=8.0,
                  minor_radius=2.0, **values):
        return self.add_primitive(
            "torus", x=x, y=y, z=z,
            major_radius=major_radius, minor_radius=minor_radius, **values)

    def import_step(self, path: str) -> CADOperation:
        """Embed a STEP file as an import feature in this Part history."""
        from .occ import step_file_values

        return self.add(CADOperation(
            self._next_id("import"), "step_import", step_file_values(path)))

    def add_boolean(self, kind: str, object_id: str, tool_id: str) -> CADOperation:
        if kind not in {"fuse", "cut", "intersect", "fragment"}:
            raise ValueError(f"Unsupported CAD boolean '{kind}'.")
        if object_id == tool_id:
            raise ValueError("A CAD boolean needs two different input features.")
        return self.add(CADOperation(
            self._next_id(kind), kind, {}, [str(object_id), str(tool_id)]))

    def fuse(self, object_id: str, tool_id: str) -> CADOperation:
        return self.add_boolean("fuse", object_id, tool_id)

    def cut(self, object_id: str, tool_id: str) -> CADOperation:
        return self.add_boolean("cut", object_id, tool_id)

    def intersect(self, object_id: str, tool_id: str) -> CADOperation:
        return self.add_boolean("intersect", object_id, tool_id)

    def fragment(self, object_id: str, tool_id: str) -> CADOperation:
        return self.add_boolean("fragment", object_id, tool_id)

    def add_transform(self, kind: str, input_id: str, **values) -> CADOperation:
        from .occ import transform_defaults

        try:
            defaults = transform_defaults(kind)
        except KeyError as exc:
            raise ValueError(f"Unsupported CAD transform '{kind}'.") from exc
        defaults.update(values)
        return self.add(CADOperation(self._next_id(kind), kind, defaults, [str(input_id)]))

    def translate(self, input_id: str, dx=5.0, dy=0.0, dz=0.0, **values):
        return self.add_transform("translate", input_id, dx=dx, dy=dy, dz=dz, **values)

    def rotate(self, input_id: str, x=0.0, y=0.0, z=0.0, ax=0.0,
               ay=0.0, az=1.0, angle=1.5707963267948966, **values):
        return self.add_transform(
            "rotate", input_id, x=x, y=y, z=z, ax=ax, ay=ay, az=az,
            angle=angle, **values)

    def dilate(self, input_id: str, x=0.0, y=0.0, z=0.0, sx=1.0,
               sy=1.0, sz=1.0, **values):
        return self.add_transform(
            "dilate", input_id, x=x, y=y, z=z, sx=sx, sy=sy, sz=sz, **values)

    def mirror(self, input_id: str, a=1.0, b=0.0, c=0.0, d=0.0, **values):
        return self.add_transform("mirror", input_id, a=a, b=b, c=c, d=d, **values)

    def preview(self) -> list[dict]:
        from .occ import preview_brep

        return preview_brep(self)

    def mesh(self, mesh_size: float):
        from .occ import create_part

        mesh_size = float(mesh_size)
        part = create_part(self, mesh_size)
        self.mesh_size = mesh_size
        return part

    def final_feature_ids(self) -> list[str]:
        consumed = {input_id for operation in self.operations for input_id in operation.inputs}
        return [operation.id for operation in self.operations if operation.id not in consumed]


class CADModel(Serializable):
    """Collection of named CAD histories owned by an FEA controller."""

    _serialized_attributes = ["parts", "mesh_size"]

    def __init__(self, mesh_size: float | None = None):
        self.parts: dict[str, CADHistory] = {}
        self.mesh_size = None if mesh_size is None else float(mesh_size)

    def add_part(self, name: str, history: CADHistory | None = None) -> CADHistory:
        name = str(name).strip()
        if not name:
            raise ValueError("CAD Part name cannot be empty.")
        if name in self.parts:
            raise ValueError(f"CAD Part '{name}' already exists.")
        history = history or CADHistory()
        if not isinstance(history, CADHistory):
            raise TypeError("CAD Part history must be a CADHistory object.")
        self.parts[name] = history
        return history

    def set_part(self, name: str, history: CADHistory) -> CADHistory:
        name = str(name).strip()
        if not name:
            raise ValueError("CAD Part name cannot be empty.")
        if not isinstance(history, CADHistory):
            raise TypeError("CAD Part history must be a CADHistory object.")
        self.parts[name] = history
        return history

    def get_part(self, name: str) -> CADHistory:
        try:
            return self.parts[str(name)]
        except KeyError as exc:
            raise KeyError(f"CAD Part '{name}' is not defined.") from exc

    def delete_part(self, name: str) -> None:
        del self.parts[str(name)]

    def preview_part(self, name: str) -> list[dict]:
        return self.get_part(name).preview()

    def mesh_part(self, name: str, mesh_size: float | None = None):
        size = self.mesh_size if mesh_size is None else float(mesh_size)
        if size is None:
            raise ValueError("A mesh size is required to mesh a CAD Part.")
        self.mesh_size = float(size)
        return self.get_part(name).mesh(size)

    def import_step(self, name: str, path: str) -> CADOperation:
        """Embed a STEP file in a named Part history."""
        return self.get_part(name).import_step(path)

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)
