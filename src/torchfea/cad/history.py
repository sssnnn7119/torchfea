"""Serializable CAD histories used by the finite-element kernel."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Iterator, TypeVar

from ..interfaces import Serializable
from .operations import (
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
    Rotate,
    Sphere,
    StepImport,
    Torus,
    Translate,
    Wedge,
)


OperationType = TypeVar("OperationType", bound=CADOperation)


@dataclass
class CADSetDefinition(Serializable):
    """A named CAD-topology selection stored with the feature history.

    ``selector`` addresses OCC topology (for example
    ``cad:surface:7``), never FE node or element indices.  The selector is
    resolved only after the history has been replayed and the analysis mesh
    has been generated.
    """

    name: str
    kind: str
    selector: str

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.kind = str(self.kind).strip().casefold()
        self.selector = str(self.selector).strip()
        if not self.name:
            raise ValueError("A CAD set needs a non-empty name.")
        if self.kind not in {"node", "element", "surface"}:
            raise ValueError(f"Unknown CAD set kind '{self.kind}'.")
        if not self.selector.startswith("cad:"):
            raise ValueError(
                "CAD set selectors must address topology with the 'cad:' prefix.")


class CADHistory(Serializable):
    """Ordered OCC features and topology selections for one Part."""

    _serialized_attributes = ["operations", "sets", "mesh_size", "exterior_surface"]

    def __init__(
        self,
        operations: Iterable[CADOperation] | None = None,
        mesh_size: float | None = None,
        sets: Iterable[CADSetDefinition] | None = None,
        exterior_surface: str = "extern",
    ) -> None:
        self.operations: list[CADOperation] = []
        for operation in operations or ():
            self.add(operation)
        self.sets: list[CADSetDefinition] = []
        for selection in sets or ():
            self.add_set(selection)
        self.mesh_size = None if mesh_size is None else float(mesh_size)
        self.exterior_surface = str(exterior_surface).strip() or "extern"

    def add(self, operation: OperationType) -> OperationType:
        """Append one explicitly typed CAD operation."""
        if not isinstance(operation, CADOperation):
            raise TypeError("CAD history entries must inherit CADOperation.")
        if any(existing.id == operation.id for existing in self.operations):
            raise ValueError(f"CAD operation id '{operation.id}' already exists.")
        self.operations.append(operation)
        return operation

    def add_set(self, selection: CADSetDefinition) -> CADSetDefinition:
        """Add one named topology selection to this Part history."""
        if not isinstance(selection, CADSetDefinition):
            raise TypeError("CAD history sets must be CADSetDefinition objects.")
        if any(item.kind == selection.kind and item.name == selection.name
               for item in self.sets):
            raise ValueError(
                f"CAD {selection.kind} set '{selection.name}' already exists.")
        self.sets.append(selection)
        return selection

    def set_set(self, name: str, kind: str, selector: str | int) -> CADSetDefinition:
        """Create or replace one named topology selection."""
        kind = str(kind).strip().casefold()
        if selector == "all":
            selector = "cad:all"
        elif isinstance(selector, Integral):
            entity_kind = {"surface": "surface", "node": "vertex", "element": "volume"}[kind]
            selector = f"cad:{entity_kind}:{selector}"
        replacement = CADSetDefinition(name, kind, selector)
        self.sets = [item for item in self.sets
                     if not (item.kind == replacement.kind and item.name == replacement.name)]
        self.sets.append(replacement)
        return replacement

    def add_surface_set(self, name: str, selector: str | int) -> CADSetDefinition:
        return self.set_set(name, "surface", selector)

    def add_node_set(self, name: str, selector: str | int) -> CADSetDefinition:
        return self.set_set(name, "node", selector)

    def add_element_set(self, name: str, selector: str | int) -> CADSetDefinition:
        return self.set_set(name, "element", selector)

    def remove_set(self, name: str, kind: str) -> None:
        target = CADSetDefinition(name, kind, "cad:all")
        self.sets = [item for item in self.sets
                     if not (item.kind == target.kind and item.name == target.name)]

    def remove_surface_set(self, name: str) -> None:
        self.remove_set(name, "surface")

    def remove_node_set(self, name: str) -> None:
        self.remove_set(name, "node")

    def remove_element_set(self, name: str) -> None:
        self.remove_set(name, "element")

    def _next_id(self, operation_type: type[CADOperation]) -> str:
        prefix = operation_type.__name__
        used = {operation.id for operation in self.operations}
        index = 1
        while f"{prefix}-{index}" in used:
            index += 1
        return f"{prefix}-{index}"

    def add_box(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: float = 20.0,
        dy: float = 10.0,
        dz: float = 5.0,
    ) -> Box:
        return self.add(Box(self._next_id(Box), x, y, z, dx, dy, dz))

    def add_wedge(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: float = 20.0,
        dy: float = 10.0,
        dz: float = 5.0,
        ltx: float = 0.0,
    ) -> Wedge:
        return self.add(Wedge(self._next_id(Wedge), x, y, z, dx, dy, dz, ltx))

    def add_cylinder(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 20.0,
        radius: float = 5.0,
    ) -> Cylinder:
        return self.add(Cylinder(
            self._next_id(Cylinder), x, y, z, dx, dy, dz, radius))

    def add_sphere(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        radius: float = 5.0,
    ) -> Sphere:
        return self.add(Sphere(self._next_id(Sphere), x, y, z, radius))

    def add_cone(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 20.0,
        radius1: float = 5.0,
        radius2: float = 1.0,
    ) -> Cone:
        return self.add(Cone(
            self._next_id(Cone), x, y, z, dx, dy, dz, radius1, radius2))

    def add_torus(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        major_radius: float = 8.0,
        minor_radius: float = 2.0,
    ) -> Torus:
        return self.add(Torus(
            self._next_id(Torus), x, y, z, major_radius, minor_radius))

    def import_step(self, path: str) -> StepImport:
        return self.add(StepImport.from_file(self._next_id(StepImport), path))

    def fuse(self, object_id: str, tool_id: str) -> Fuse:
        return self.add(Fuse(self._next_id(Fuse), object_id, tool_id))

    def cut(self, object_id: str, tool_id: str) -> Cut:
        return self.add(Cut(self._next_id(Cut), object_id, tool_id))

    def intersect(self, object_id: str, tool_id: str) -> Intersect:
        return self.add(Intersect(self._next_id(Intersect), object_id, tool_id))

    def fragment(self, object_id: str, tool_id: str) -> Fragment:
        return self.add(Fragment(self._next_id(Fragment), object_id, tool_id))

    def translate(
        self,
        input_id: str,
        dx: float = 5.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> Translate:
        return self.add(Translate(self._next_id(Translate), input_id, dx, dy, dz))

    def rotate(
        self,
        input_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        ax: float = 0.0,
        ay: float = 0.0,
        az: float = 1.0,
        angle: float = 1.5707963267948966,
    ) -> Rotate:
        return self.add(Rotate(
            self._next_id(Rotate), input_id, x, y, z, ax, ay, az, angle))

    def dilate(
        self,
        input_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        sz: float = 1.0,
    ) -> Dilate:
        return self.add(Dilate(
            self._next_id(Dilate), input_id, x, y, z, sx, sy, sz))

    def mirror(
        self,
        input_id: str,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
    ) -> Mirror:
        return self.add(Mirror(self._next_id(Mirror), input_id, a, b, c, d))

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
        consumed = {
            input_id
            for operation in self.operations
            for input_id in operation.input_ids
        }
        return [
            operation.id
            for operation in self.operations
            if operation.id not in consumed
        ]


class CADModel(Serializable):
    """Collection of named CAD histories owned by an FEA controller."""

    _serialized_attributes = ["parts", "mesh_size"]

    def __init__(self, mesh_size: float | None = None) -> None:
        self.parts: dict[str, CADHistory] = {}
        self.mesh_size = None if mesh_size is None else float(mesh_size)

    def add_part(
        self,
        name: str,
        history: CADHistory | None = None,
    ) -> CADHistory:
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

    def import_step(self, name: str, path: str) -> StepImport:
        return self.get_part(name).import_step(path)

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)
