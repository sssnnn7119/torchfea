"""Explicit, serializable CAD feature definitions."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
import base64
import copy
from dataclasses import dataclass, field, fields
import math
import os
import tempfile
from typing import Any, ClassVar, Mapping, Protocol

from ..interfaces import Serializable


DimTag = tuple[int, int]


class OCCContext(Protocol):
    """OCC services available while replaying one CAD history."""

    occ: Any

    def entities(self, operation_id: str) -> list[DimTag]: ...


@dataclass
class CADOperation(Serializable, metaclass=ABCMeta):
    """Base class for one named operation in an OCC feature history."""

    id: str

    kind: ClassVar[str] = ""
    non_parameter_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise TypeError("CADOperation is an abstract operation definition.")
        self.id = str(self.id).strip()
        if not self.id:
            raise ValueError("A CAD operation needs a non-empty id.")
        for parameter_name in self.parameter_names:
            value = getattr(self, parameter_name)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{type(self).__name__}.{parameter_name} must be numeric.") from exc
            setattr(self, parameter_name, value)

    @property
    def input_ids(self) -> tuple[str, ...]:
        """Identifiers of features consumed by this operation."""
        return ()

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the operation's explicit numeric parameters."""
        excluded = {"id", *self.non_parameter_fields}
        return tuple(
            operation_field.name
            for operation_field in fields(self)
            if operation_field.name not in excluded
        )

    def copy_with_mapped_ids(
        self,
        operation_id: str,
        input_mapping: Mapping[str, str],
    ) -> "CADOperation":
        """Copy an operation while remapping its own id and input ids."""
        operation = copy.deepcopy(self)
        operation.id = str(operation_id)
        operation._map_input_ids(input_mapping)
        return operation

    def _map_input_ids(self, input_mapping: Mapping[str, str]) -> None:
        if self.input_ids:
            raise NotImplementedError(
                f"{type(self).__name__} must implement input-id mapping.")

    @abstractmethod
    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        """Create or modify this feature's OCC volume entities."""


@dataclass
class PrimitiveOperation(CADOperation):
    """Base class for a solid created without history inputs."""


@dataclass
class Box(PrimitiveOperation):
    kind: ClassVar[str] = "box"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 20.0
    dy: float = 10.0
    dz: float = 5.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        dimensions = float(self.dx), float(self.dy), float(self.dz)
        if min(dimensions) <= 0:
            raise ValueError("Box dimensions must be positive.")
        tag = context.occ.addBox(
            float(self.x), float(self.y), float(self.z), *dimensions)
        return [(3, tag)]


@dataclass
class Wedge(PrimitiveOperation):
    kind: ClassVar[str] = "wedge"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 20.0
    dy: float = 10.0
    dz: float = 5.0
    ltx: float = 0.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        dimensions = float(self.dx), float(self.dy), float(self.dz)
        if min(dimensions) <= 0:
            raise ValueError("Wedge dimensions must be positive.")
        tag = context.occ.addWedge(
            float(self.x), float(self.y), float(self.z), *dimensions,
            -1, float(self.ltx))
        return [(3, tag)]


@dataclass
class Cylinder(PrimitiveOperation):
    kind: ClassVar[str] = "cylinder"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 20.0
    radius: float = 5.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        axis = float(self.dx), float(self.dy), float(self.dz)
        if float(self.radius) <= 0 or math.hypot(*axis) == 0:
            raise ValueError("Cylinder radius and axis must be positive.")
        tag = context.occ.addCylinder(
            float(self.x), float(self.y), float(self.z),
            *axis, float(self.radius))
        return [(3, tag)]


@dataclass
class Sphere(PrimitiveOperation):
    kind: ClassVar[str] = "sphere"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 5.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        if float(self.radius) <= 0:
            raise ValueError("Sphere radius must be positive.")
        tag = context.occ.addSphere(
            float(self.x), float(self.y), float(self.z), float(self.radius))
        return [(3, tag)]


@dataclass
class Cone(PrimitiveOperation):
    kind: ClassVar[str] = "cone"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 20.0
    radius1: float = 5.0
    radius2: float = 1.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        axis = float(self.dx), float(self.dy), float(self.dz)
        radii = float(self.radius1), float(self.radius2)
        if min(radii) < 0 or math.hypot(*axis) == 0:
            raise ValueError("Cone radii and axis must be valid.")
        tag = context.occ.addCone(
            float(self.x), float(self.y), float(self.z), *axis, *radii)
        return [(3, tag)]


@dataclass
class Torus(PrimitiveOperation):
    kind: ClassVar[str] = "torus"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    major_radius: float = 8.0
    minor_radius: float = 2.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        radii = float(self.major_radius), float(self.minor_radius)
        if min(radii) <= 0:
            raise ValueError("Torus radii must be positive.")
        tag = context.occ.addTorus(
            float(self.x), float(self.y), float(self.z), *radii)
        return [(3, tag)]


@dataclass
class StepImport(PrimitiveOperation):
    """A STEP solid embedded in the history as Base64 data."""

    kind: ClassVar[str] = "step_import"
    non_parameter_fields: ClassVar[tuple[str, ...]] = ("filename", "data")

    filename: str
    data: str = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.filename = str(self.filename).strip()
        if not self.filename:
            raise ValueError("A STEP import needs a filename.")
        if not isinstance(self.data, str) or not self.data:
            raise ValueError("A STEP import needs embedded file data.")

    @classmethod
    def from_file(cls, operation_id: str, path: str) -> "StepImport":
        source_path = os.fspath(path)
        if not source_path:
            raise ValueError("STEP import path cannot be empty.")
        try:
            with open(source_path, "rb") as source:
                content = source.read()
        except OSError as exc:
            raise ValueError(
                f"Cannot read STEP file '{source_path}': {exc}") from exc
        if not content:
            raise ValueError(f"STEP file '{source_path}' is empty.")
        return cls(
            operation_id,
            filename=os.path.basename(source_path),
            data=base64.b64encode(content).decode("ascii"),
        )

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        try:
            content = base64.b64decode(self.data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"STEP import feature '{self.id}' contains invalid file data.") from exc

        with tempfile.NamedTemporaryFile(
            prefix="torchfea_step_",
            suffix=".step",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
        try:
            imported = context.occ.importShapes(temporary_path)
        finally:
            os.unlink(temporary_path)

        volumes = [
            (int(dimension), int(tag))
            for dimension, tag in imported
            if int(dimension) == 3
        ]
        if not volumes:
            raise ValueError(
                f"STEP import feature '{self.id}' contains no solid volume.")
        return volumes


@dataclass
class BooleanOperation(CADOperation):
    """Base class for a two-input Boolean feature."""

    non_parameter_fields: ClassVar[tuple[str, ...]] = ("object_id", "tool_id")

    object_id: str
    tool_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        self.object_id = str(self.object_id).strip()
        self.tool_id = str(self.tool_id).strip()
        if not self.object_id or not self.tool_id:
            raise ValueError("A CAD Boolean operation needs two input ids.")
        if self.object_id == self.tool_id:
            raise ValueError("A CAD Boolean operation needs two different inputs.")

    @property
    def input_ids(self) -> tuple[str, str]:
        return self.object_id, self.tool_id

    def _map_input_ids(self, input_mapping: Mapping[str, str]) -> None:
        self.object_id = input_mapping[self.object_id]
        self.tool_id = input_mapping[self.tool_id]

    def _apply_boolean(self, context: OCCContext, boolean_method) -> list[DimTag]:
        output, _ = boolean_method(
            context.entities(self.object_id),
            context.entities(self.tool_id),
            removeObject=True,
            removeTool=True,
        )
        return [
            (int(dimension), int(tag))
            for dimension, tag in output
            if int(dimension) == 3
        ]


@dataclass
class Fuse(BooleanOperation):
    kind: ClassVar[str] = "fuse"

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        return self._apply_boolean(context, context.occ.fuse)


@dataclass
class Cut(BooleanOperation):
    kind: ClassVar[str] = "cut"

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        return self._apply_boolean(context, context.occ.cut)


@dataclass
class Intersect(BooleanOperation):
    kind: ClassVar[str] = "intersect"

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        return self._apply_boolean(context, context.occ.intersect)


@dataclass
class Fragment(BooleanOperation):
    kind: ClassVar[str] = "fragment"

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        return self._apply_boolean(context, context.occ.fragment)


@dataclass
class TransformOperation(CADOperation):
    """Base class for an in-place transform of one live feature."""

    non_parameter_fields: ClassVar[tuple[str, ...]] = ("input_id",)

    input_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        self.input_id = str(self.input_id).strip()
        if not self.input_id:
            raise ValueError("A CAD transform needs an input id.")

    @property
    def input_ids(self) -> tuple[str]:
        return (self.input_id,)

    def _map_input_ids(self, input_mapping: Mapping[str, str]) -> None:
        self.input_id = input_mapping[self.input_id]


@dataclass
class Translate(TransformOperation):
    kind: ClassVar[str] = "translate"

    dx: float = 5.0
    dy: float = 0.0
    dz: float = 0.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        entities = context.entities(self.input_id)
        context.occ.translate(
            entities, float(self.dx), float(self.dy), float(self.dz))
        return entities


@dataclass
class Rotate(TransformOperation):
    kind: ClassVar[str] = "rotate"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 1.0
    angle: float = 1.5707963267948966

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        axis = float(self.ax), float(self.ay), float(self.az)
        if math.hypot(*axis) == 0:
            raise ValueError("Rotate transform needs a non-zero axis.")
        entities = context.entities(self.input_id)
        context.occ.rotate(
            entities,
            float(self.x), float(self.y), float(self.z),
            *axis, float(self.angle),
        )
        return entities


@dataclass
class Dilate(TransformOperation):
    kind: ClassVar[str] = "dilate"

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        scales = float(self.sx), float(self.sy), float(self.sz)
        if min(map(abs, scales)) == 0:
            raise ValueError("Scale factors must be non-zero.")
        entities = context.entities(self.input_id)
        context.occ.dilate(
            entities,
            float(self.x), float(self.y), float(self.z),
            *scales,
        )
        return entities


@dataclass
class Mirror(TransformOperation):
    kind: ClassVar[str] = "mirror"

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    def apply_occ(self, context: OCCContext) -> list[DimTag]:
        normal = float(self.a), float(self.b), float(self.c)
        if math.hypot(*normal) == 0:
            raise ValueError("Mirror transform needs a non-zero plane normal.")
        entities = context.entities(self.input_id)
        context.occ.mirror(entities, *normal, float(self.d))
        return entities
