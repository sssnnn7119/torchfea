"""Typed Abaqus INP models with lossless read/write for unedited records.

Known entities have fixed dataclass fields. Unknown keywords use RawRecord
with an explicit text field so round trips retain all input without pretending
that the finite-element solver implements every Abaqus keyword.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING, TextIO
import re

import numpy as np

from .utils import Rotation

if TYPE_CHECKING:
    from .controller import FEAController


def _fields(text: str) -> list[str]:
    return [value.strip() for value in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', text)]


def _option(fields: list[str], name: str, default: str | None = None) -> str | None:
    for token in fields[1:]:
        key, separator, value = token.partition("=")
        if key.strip().lower() == name:
            return value.strip().strip('"') if separator else ""
    return default


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _numeric(values) -> str:
    return ", ".join(format(float(value), ".16g") for value in values)


@dataclass(slots=True)
class InpRecord:
    keyword: ClassVar[str] = ""
    known_parameters: ClassVar[tuple[str, ...]] = ()
    _source_header: str = field(default="", init=False, repr=False)
    _source_data: list[str] = field(default_factory=list, init=False, repr=False)
    _original_render: str = field(default="", init=False, repr=False)

    def header_values(self) -> list[tuple[str, str | None]]:
        return []

    def data_lines(self) -> list[str]:
        return []

    def _render(self) -> str:
        newline = "\r\n" if "\r\n" in self._source_header else "\n"
        fields = _fields("".join(self._source_header.splitlines()).lstrip()[1:]) if self._source_header else []
        extra = [token for token in fields[1:]
                 if token.partition("=")[0].strip().lower() not in self.known_parameters]
        parameters = []
        for name, value in self.header_values():
            if value is None:
                parameters.append(name)
            else:
                quoted = f'"{value}"' if "," in value else value
                parameters.append(f"{name}={quoted}")
        header = ", ".join(["*" + self.keyword, *parameters, *extra]) + newline
        return header + "".join(line + newline for line in self.data_lines())

    def bind_source(self, header: str, data: list[str]) -> None:
        self._source_header = header
        self._source_data = list(data)
        self._original_render = self._render()

    def to_text(self) -> str:
        rendered = self._render()
        if self._source_header and rendered == self._original_render:
            return self._source_header + "".join(self._source_data)
        comments = [line for line in self._source_data if line.lstrip().startswith("**")]
        return rendered + "".join(comments)


@dataclass(slots=True)
class RawRecord(InpRecord):
    """Opaque, fully preserved input for a keyword without a typed implementation."""
    keyword_name: str = ""
    text: str = ""

    @property
    def keyword(self) -> str:
        return self.keyword_name

    def to_text(self) -> str:
        return self.text


@dataclass(slots=True)
class Nodes(InpRecord):
    keyword: ClassVar[str] = "NODE"
    known_parameters: ClassVar[tuple[str, ...]] = ("nset",)
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    coordinates: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    node_set: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [] if self.node_set is None else [("nset", self.node_set)]

    def data_lines(self) -> list[str]:
        return [f"{label}, {_numeric(point)}" for label, point in zip(self.labels, self.coordinates)]


@dataclass(slots=True)
class Elements(InpRecord):
    keyword: ClassVar[str] = "ELEMENT"
    known_parameters: ClassVar[tuple[str, ...]] = ("type", "elset")
    element_type: str = "C3D4"
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    connectivity: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.int64))
    element_set: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("type", self.element_type)] + ([] if self.element_set is None else [("elset", self.element_set)])

    def data_lines(self) -> list[str]:
        result = []
        for label, nodes in zip(self.labels, self.connectivity):
            entries = [str(label), *map(str, nodes)]
            for start in range(0, len(entries), 16):
                result.append(", ".join(entries[start:start+16]) + ("," if start+16 < len(entries) else ""))
        return result


@dataclass(slots=True)
class NodeSet(InpRecord):
    keyword: ClassVar[str] = "NSET"
    known_parameters: ClassVar[tuple[str, ...]] = ("nset", "generate", "instance")
    name: str = ""
    members: list[str] = field(default_factory=list)
    generate: bool = False
    instance: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        result = [(self.keyword.lower(), self.name)]
        if self.generate:
            result.append(("generate", None))
        if self.instance is not None:
            result.append(("instance", self.instance))
        return result

    def data_lines(self) -> list[str]:
        size = 3 if self.generate else 16
        return [", ".join(self.members[i:i+size]) for i in range(0, len(self.members), size)]

    def numeric_labels(self) -> np.ndarray:
        if self.generate:
            result = [np.arange(int(self.members[i]), int(self.members[i+1])+1,
                                int(self.members[i+2])) for i in range(0, len(self.members), 3)]
            return np.unique(np.concatenate(result)) if result else np.empty(0, dtype=int)
        return np.unique(np.asarray(self.members, dtype=np.int64))


@dataclass(slots=True)
class ElementSet(NodeSet):
    keyword: ClassVar[str] = "ELSET"
    known_parameters: ClassVar[tuple[str, ...]] = ("elset", "generate", "instance")


@dataclass(slots=True)
class SurfaceEntry:
    target: str
    face: str = ""
    weight: float | None = None


@dataclass(slots=True)
class Surface(InpRecord):
    keyword: ClassVar[str] = "SURFACE"
    known_parameters: ClassVar[tuple[str, ...]] = ("name", "type")
    name: str = ""
    surface_type: str = "ELEMENT"
    entries: list[SurfaceEntry] = field(default_factory=list)

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("type", self.surface_type), ("name", self.name)]

    def data_lines(self) -> list[str]:
        return [entry.target + (f", {entry.face}" if entry.face else "")
                + (f", {entry.weight:.16g}" if entry.weight is not None else "")
                for entry in self.entries]


@dataclass(slots=True)
class SolidSection(InpRecord):
    keyword: ClassVar[str] = "SOLID SECTION"
    known_parameters: ClassVar[tuple[str, ...]] = ("elset", "material")
    element_set: str = ""
    material: str = ""
    thickness: float | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("elset", self.element_set), ("material", self.material)]

    def data_lines(self) -> list[str]:
        return ["," if self.thickness is None else format(self.thickness, ".16g")]


@dataclass(slots=True)
class NamedRecord(InpRecord):
    known_parameters: ClassVar[tuple[str, ...]] = ("name",)
    name: str = ""

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("name", self.name)]


@dataclass(slots=True)
class PartDefinition(NamedRecord):
    keyword: ClassVar[str] = "PART"


@dataclass(slots=True)
class MaterialDefinition(NamedRecord):
    keyword: ClassVar[str] = "MATERIAL"


@dataclass(slots=True)
class AssemblyDefinition(NamedRecord):
    keyword: ClassVar[str] = "ASSEMBLY"


@dataclass(slots=True)
class StepDefinition(NamedRecord):
    keyword: ClassVar[str] = "STEP"
    known_parameters: ClassVar[tuple[str, ...]] = ("name", "nlgeom", "inc")
    nlgeom: bool = False
    maximum_increments: int = 100

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("name", self.name), ("nlgeom", "YES" if self.nlgeom else "NO"),
                ("inc", str(self.maximum_increments))]


@dataclass(slots=True)
class Elastic(InpRecord):
    keyword: ClassVar[str] = "ELASTIC"
    young_modulus: float = 1.0
    poisson_ratio: float = 0.3

    def data_lines(self) -> list[str]:
        return [_numeric([self.young_modulus, self.poisson_ratio])]


@dataclass(slots=True)
class Hyperelastic(InpRecord):
    keyword: ClassVar[str] = "HYPERELASTIC"
    known_parameters: ClassVar[tuple[str, ...]] = (
        "neo hooke", "mooney-rivlin", "yeoh", "ogden", "arruda-boyce",
        "polynomial", "reduced polynomial", "n")
    model: str = "NEO HOOKE"
    order: int = 1
    constants: list[float] = field(default_factory=list)

    def header_values(self) -> list[tuple[str, str | None]]:
        return [(self.model.lower(), None)] + ([("n", str(self.order))]
                if self.model.upper() in {"OGDEN", "POLYNOMIAL", "REDUCED POLYNOMIAL"} else [])

    def data_lines(self) -> list[str]:
        return [_numeric(self.constants[i:i+8]) for i in range(0, len(self.constants), 8)]


@dataclass(slots=True)
class Density(InpRecord):
    keyword: ClassVar[str] = "DENSITY"
    value: float = 0.0

    def data_lines(self) -> list[str]:
        return [format(self.value, ".16g")]


@dataclass(slots=True)
class Instance(InpRecord):
    keyword: ClassVar[str] = "INSTANCE"
    known_parameters: ClassVar[tuple[str, ...]] = ("name", "part")
    name: str = ""
    part_name: str = ""
    translation: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float, float, float, float, float] | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("name", self.name), ("part", self.part_name)]

    def data_lines(self) -> list[str]:
        rows = []
        if self.translation is not None or self.rotation is not None:
            rows.append(_numeric(self.translation if self.translation is not None else (0., 0., 0.)))
        if self.rotation is not None:
            rows.append(_numeric(self.rotation))
        return rows

    def transform_matrix(self) -> np.ndarray:
        transform = np.eye(4)
        if self.translation is not None:
            transform[:3, 3] = self.translation
        if self.rotation is not None:
            values = np.asarray(self.rotation)
            origin, axis = values[:3], values[3:6] - values[:3]
            matrix = Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(values[6])).as_matrix().cpu().numpy()
            transform[:3, :3] = matrix
            transform[:3, 3] = matrix @ transform[:3, 3] + origin - matrix @ origin
        return transform


@dataclass(slots=True)
class Static(InpRecord):
    keyword: ClassVar[str] = "STATIC"
    initial_increment: float = 0.1
    time_period: float = 1.0
    minimum_increment: float | None = None
    maximum_increment: float | None = None

    def data_lines(self) -> list[str]:
        values = [self.initial_increment, self.time_period]
        if self.minimum_increment is not None:
            values.append(self.minimum_increment)
        if self.maximum_increment is not None:
            values.append(self.maximum_increment)
        return [_numeric(values)]


@dataclass(slots=True)
class BoundaryValue:
    target: str
    first_dof: int
    last_dof: int
    value: float = 0.0


@dataclass(slots=True)
class Boundary(InpRecord):
    keyword: ClassVar[str] = "BOUNDARY"
    values: list[BoundaryValue] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [f"{v.target}, {v.first_dof}, {v.last_dof}, {v.value:.16g}" for v in self.values]


@dataclass(slots=True)
class NodalForce:
    target: str
    dof: int
    magnitude: float


@dataclass(slots=True)
class ConcentratedLoad(InpRecord):
    keyword: ClassVar[str] = "CLOAD"
    values: list[NodalForce] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [f"{v.target}, {v.dof}, {v.magnitude:.16g}" for v in self.values]


@dataclass(slots=True)
class DistributedForce:
    target: str
    load_type: str
    magnitude: float


@dataclass(slots=True)
class DistributedLoad(InpRecord):
    keyword: ClassVar[str] = "DLOAD"
    values: list[DistributedForce] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [f"{v.target}, {v.load_type}, {v.magnitude:.16g}" for v in self.values]


@dataclass(slots=True)
class SurfaceLoad(DistributedLoad):
    keyword: ClassVar[str] = "DSLOAD"


@dataclass(slots=True)
class Coupling(InpRecord):
    keyword: ClassVar[str] = "COUPLING"
    known_parameters: ClassVar[tuple[str, ...]] = ("constraint name", "ref node", "surface")
    name: str = ""
    reference_node: str = ""
    surface: str = ""

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("constraint name", self.name), ("ref node", self.reference_node), ("surface", self.surface)]


@dataclass(slots=True)
class Kinematic(InpRecord):
    keyword: ClassVar[str] = "KINEMATIC"
    first_dof: int = 1
    last_dof: int = 3

    def data_lines(self) -> list[str]:
        return [f"{self.first_dof}, {self.last_dof}"]


@dataclass(slots=True)
class Spring(InpRecord):
    keyword: ClassVar[str] = "SPRING"
    known_parameters: ClassVar[tuple[str, ...]] = ("elset",)
    element_set: str = ""
    stiffness: float = 1.0
    first_dof: int | None = None
    second_dof: int | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("elset", self.element_set)]

    def data_lines(self) -> list[str]:
        dofs = "" if self.first_dof is None else str(self.first_dof)
        if self.second_dof is not None:
            dofs += f", {self.second_dof}"
        return [dofs, format(self.stiffness, ".16g")]


@dataclass(slots=True)
class ConnectorSection(InpRecord):
    keyword: ClassVar[str] = "CONNECTOR SECTION"
    known_parameters: ClassVar[tuple[str, ...]] = ("elset",)
    element_set: str = ""
    connection_type: str = "JOIN"
    first_orientation: str | None = None
    second_orientation: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("elset", self.element_set)]

    def data_lines(self) -> list[str]:
        result = [self.connection_type]
        if self.first_orientation is not None:
            result.append(f"{self.first_orientation}, {self.second_orientation or self.first_orientation}")
        return result


@dataclass(slots=True)
class Orientation(InpRecord):
    keyword: ClassVar[str] = "ORIENTATION"
    known_parameters: ClassVar[tuple[str, ...]] = ("name", "system")
    name: str = ""
    first_axis: tuple[float, float, float] = (1., 0., 0.)
    second_axis: tuple[float, float, float] = (0., 1., 0.)
    rotation_axis: int = 1
    additional_angle: float = 0.

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("name", self.name), ("system", "RECTANGULAR")]

    def data_lines(self) -> list[str]:
        return [_numeric([*self.first_axis, *self.second_axis]),
                f"{self.rotation_axis}, {self.additional_angle:.16g}"]


@dataclass(slots=True)
class MpcValue:
    constraint_type: str
    nodes: list[str]


@dataclass(slots=True)
class MultiPointConstraint(InpRecord):
    keyword: ClassVar[str] = "MPC"
    values: list[MpcValue] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [", ".join([value.constraint_type, *value.nodes]) for value in self.values]


@dataclass(slots=True)
class Output(InpRecord):
    keyword: ClassVar[str] = "OUTPUT"
    known_parameters: ClassVar[tuple[str, ...]] = ("field", "history", "frequency")
    output_type: str = "FIELD"
    frequency: int | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [(self.output_type.lower(), None)] + (
            [] if self.frequency is None else [("frequency", str(self.frequency))])


@dataclass(slots=True)
class NodeOutput(InpRecord):
    keyword: ClassVar[str] = "NODE OUTPUT"
    known_parameters: ClassVar[tuple[str, ...]] = ("nset",)
    variables: list[str] = field(default_factory=lambda: ["U", "RF"])
    node_set: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [] if self.node_set is None else [("nset", self.node_set)]

    def data_lines(self) -> list[str]:
        return [", ".join(self.variables)]


@dataclass(slots=True)
class ElementOutput(InpRecord):
    keyword: ClassVar[str] = "ELEMENT OUTPUT"
    known_parameters: ClassVar[tuple[str, ...]] = ("elset",)
    variables: list[str] = field(default_factory=lambda: ["S", "E"])
    element_set: str | None = None

    def header_values(self) -> list[tuple[str, str | None]]:
        return [] if self.element_set is None else [("elset", self.element_set)]

    def data_lines(self) -> list[str]:
        return [", ".join(self.variables)]


@dataclass(slots=True)
class SurfaceInteraction(NamedRecord):
    keyword: ClassVar[str] = "SURFACE INTERACTION"


@dataclass(slots=True)
class SurfaceBehavior(InpRecord):
    keyword: ClassVar[str] = "SURFACE BEHAVIOR"
    known_parameters: ClassVar[tuple[str, ...]] = ("pressure-overclosure",)
    pressure_overclosure: str = "HARD"

    def header_values(self) -> list[tuple[str, str | None]]:
        return [("pressure-overclosure", self.pressure_overclosure)]


@dataclass(slots=True)
class GeneralContact(InpRecord):
    keyword: ClassVar[str] = "CONTACT"


@dataclass(slots=True)
class ContactPair:
    first_surface: str
    second_surface: str


@dataclass(slots=True)
class ContactInclusions(InpRecord):
    keyword: ClassVar[str] = "CONTACT INCLUSIONS"
    pairs: list[ContactPair] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [f"{pair.first_surface}, {pair.second_surface}" for pair in self.pairs]


@dataclass(slots=True)
class ContactProperty:
    first_surface: str
    second_surface: str
    interaction: str


@dataclass(slots=True)
class ContactPropertyAssignment(InpRecord):
    keyword: ClassVar[str] = "CONTACT PROPERTY ASSIGNMENT"
    assignments: list[ContactProperty] = field(default_factory=list)

    def data_lines(self) -> list[str]:
        return [f"{value.first_surface}, {value.second_surface}, {value.interaction}"
                for value in self.assignments]


def _parse_record(header: str, data: list[str]) -> InpRecord:
    fields = _fields("".join(header.splitlines()).strip()[1:])
    word = fields[0].upper()
    rows = [_fields(line.strip()) for line in data
            if line.strip() and not line.lstrip().startswith("**")]
    record: InpRecord
    try:
        if word == "NODE" and rows:
            numbers = [[_float(v) for v in row if v] for row in rows]
            record = Nodes(np.array([int(row[0]) for row in numbers]),
                           np.array([(row[1:4]+[0.]*3)[:3] for row in numbers]), _option(fields, "nset"))
        elif word == "ELEMENT" and rows:
            entries, pending = [], []
            for row in rows:
                pending.extend(int(v) for v in row if v)
                if row[-1]:
                    entries.append(pending)
                    pending = []
            if pending or len({len(row) for row in entries}) != 1:
                raise ValueError("Unsupported connectivity continuation.")
            values = np.array(entries, dtype=np.int64)
            record = Elements(_option(fields, "type", "").upper(), values[:, 0], values[:, 1:],
                              _option(fields, "elset"))
        elif word in {"NSET", "ELSET"}:
            members = [v for row in rows for v in row if v]
            if word == "NSET":
                record = NodeSet(_option(fields, "nset", ""), members,
                                 _option(fields, "generate") is not None, _option(fields, "instance"))
            else:
                record = ElementSet(_option(fields, "elset", ""), members,
                                    _option(fields, "generate") is not None, _option(fields, "instance"))
        elif word == "SURFACE":
            record = Surface(_option(fields, "name", ""), _option(fields, "type", "ELEMENT").upper(),
                             [SurfaceEntry(row[0], row[1] if len(row)>1 else "",
                                           _float(row[2]) if len(row)>2 and row[2] else None) for row in rows])
        elif word == "SOLID SECTION":
            record = SolidSection(_option(fields, "elset", ""), _option(fields, "material", ""),
                                  _float(rows[0][0]) if rows and rows[0][0] else None)
        elif word == "PART":
            record = PartDefinition(_option(fields, "name", "Part"))
        elif word == "MATERIAL":
            record = MaterialDefinition(_option(fields, "name", "Material"))
        elif word == "ASSEMBLY":
            record = AssemblyDefinition(_option(fields, "name", "Assembly"))
        elif word == "STEP":
            record = StepDefinition(_option(fields, "name", "Step"),
                                    _option(fields, "nlgeom", "NO").upper() == "YES",
                                    int(_option(fields, "inc", "100")))
        elif word == "ELASTIC" and len(rows) == 1 and len(rows[0]) == 2 and _option(fields, "type", "ISOTROPIC").upper() == "ISOTROPIC":
            record = Elastic(_float(rows[0][0]), _float(rows[0][1]))
        elif word == "HYPERELASTIC":
            model = next(token.strip().upper() for token in fields[1:]
                         if "=" not in token and token.strip())
            record = Hyperelastic(model, int(_option(fields, "n", "1")),
                                  [_float(v) for row in rows for v in row if v])
        elif word == "DENSITY" and len(rows) == 1:
            record = Density(_float(rows[0][0]))
        elif word == "INSTANCE":
            translation, rotation = None, None
            if rows and len(rows[0]) == 3:
                translation = tuple(map(_float, rows.pop(0)))
            if rows:
                if len(rows) != 1 or len(rows[0]) != 7:
                    raise ValueError("Unsupported instance definition.")
                rotation = tuple(map(_float, rows[0]))
            record = Instance(_option(fields, "name", ""), _option(fields, "part", ""), translation, rotation)
        elif word == "STATIC" and rows:
            values = [_float(v) if v else None for v in rows[0]]
            values += [None] * (4-len(values))
            record = Static(values[0] if values[0] is not None else .1,
                            values[1] or 1., values[2], values[3])
        elif word == "BOUNDARY":
            record = Boundary([BoundaryValue(row[0], int(row[1]),
                              int(row[2]) if len(row)>2 and row[2] else int(row[1]),
                              _float(row[3]) if len(row)>3 and row[3] else 0.) for row in rows])
        elif word == "CLOAD":
            record = ConcentratedLoad([NodalForce(row[0], int(row[1]), _float(row[2])) for row in rows])
        elif word in {"DLOAD", "DSLOAD"}:
            values = [DistributedForce(row[0], row[1], _float(row[2])) for row in rows]
            record = DistributedLoad(values) if word == "DLOAD" else SurfaceLoad(values)
        elif word == "COUPLING":
            record = Coupling(_option(fields, "constraint name", ""), _option(fields, "ref node", ""),
                              _option(fields, "surface", ""))
        elif word == "KINEMATIC" and rows:
            record = Kinematic(int(rows[0][0]), int(rows[0][1]) if len(rows[0])>1 else int(rows[0][0]))
        elif word == "SPRING" and len(rows) == 1 and _option(fields, "nonlinear") is None:
            # Blank DOF row for SPRINGA is not present in the nonblank rows.
            record = Spring(_option(fields, "elset", ""), _float(rows[0][0]))
        elif word == "SPRING" and len(rows) == 2 and _option(fields, "nonlinear") is None:
            record = Spring(_option(fields, "elset", ""), _float(rows[1][0]), int(rows[0][0]),
                            int(rows[0][1]) if len(rows[0])>1 and rows[0][1] else None)
        elif word == "CONNECTOR SECTION" and rows:
            record = ConnectorSection(_option(fields, "elset", ""), ", ".join(rows[0]),
                                      rows[1][0] if len(rows)>1 else None,
                                      rows[1][1] if len(rows)>1 and len(rows[1])>1 else None)
        elif word == "ORIENTATION" and rows and len(rows[0]) == 6 and _option(fields, "system", "RECTANGULAR").upper() == "RECTANGULAR":
            record = Orientation(_option(fields, "name", ""),
                                 tuple(map(_float, rows[0][:3])), tuple(map(_float, rows[0][3:6])),
                                 int(rows[1][0]) if len(rows)>1 and rows[1][0] else 1,
                                 _float(rows[1][1]) if len(rows)>1 and len(rows[1])>1 and rows[1][1] else 0.)
        elif word == "MPC":
            record = MultiPointConstraint([MpcValue(row[0], row[1:]) for row in rows])
        elif word == "OUTPUT":
            frequency = _option(fields, "frequency")
            record = Output("HISTORY" if _option(fields, "history") is not None else "FIELD",
                            int(frequency) if frequency is not None else None)
        elif word == "NODE OUTPUT":
            record = NodeOutput([v for row in rows for v in row if v], _option(fields, "nset"))
        elif word == "ELEMENT OUTPUT":
            record = ElementOutput([v for row in rows for v in row if v], _option(fields, "elset"))
        elif word == "SURFACE INTERACTION":
            record = SurfaceInteraction(_option(fields, "name", ""))
        elif word == "SURFACE BEHAVIOR" and not rows:
            record = SurfaceBehavior(_option(fields, "pressure-overclosure", "HARD"))
        elif word == "CONTACT":
            record = GeneralContact()
        elif word == "CONTACT INCLUSIONS":
            record = ContactInclusions([ContactPair(row[0], row[1] if len(row)>1 else "") for row in rows])
        elif word == "CONTACT PROPERTY ASSIGNMENT" and all(len(row) == 3 for row in rows):
            record = ContactPropertyAssignment([ContactProperty(*row) for row in rows])
        else:
            return RawRecord(word, header + "".join(data))
        record.bind_source(header, data)
        return record
    except (ValueError, IndexError, TypeError, OverflowError, StopIteration):
        return RawRecord(word, header + "".join(data))


@dataclass(slots=True)
class Parts:
    name: str = ""
    node_blocks: list[Nodes] = field(default_factory=list)
    elements: list[Elements] = field(default_factory=list)
    node_sets: list[NodeSet] = field(default_factory=list)
    element_sets: list[ElementSet] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
    sections: list[SolidSection] = field(default_factory=list)

    @property
    def nodes(self) -> np.ndarray:
        return np.concatenate([np.column_stack((block.labels-1, block.coordinates))
                               for block in self.node_blocks]) if self.node_blocks else np.empty((0, 4))

    def node_set_labels(self, name: str) -> np.ndarray:
        return self._set_labels(name, self.node_sets)

    def element_set_labels(self, name: str) -> np.ndarray:
        return self._set_labels(name, self.element_sets)

    @staticmethod
    def _set_labels(name: str, sets: list[NodeSet], visiting: tuple[str, ...] = ()) -> np.ndarray:
        if name in visiting:
            raise ValueError(f"Cyclic set reference: {name}")
        members = []
        for block in sets:
            if block.name != name:
                continue
            if block.generate:
                members.extend(block.numeric_labels())
            else:
                for member in block.members:
                    if member.isdigit():
                        members.append(int(member))
                    else:
                        members.extend(Parts._set_labels(member, sets, (*visiting, name)))
        return np.unique(np.asarray(members, dtype=np.int64))


@dataclass(slots=True)
class Materials:
    name: str = ""
    elastic: Elastic | None = None
    hyperelastic: Hyperelastic | None = None
    density: Density | None = None


@dataclass(slots=True)
class Assembly:
    name: str = "Assembly"
    instances: list[Instance] = field(default_factory=list)
    node_blocks: list[Nodes] = field(default_factory=list)
    node_sets: list[NodeSet] = field(default_factory=list)
    element_sets: list[ElementSet] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)

    def read(self, origin_data: list[str], ind: int) -> None:
        text = "".join(line if line.endswith(("\n", "\r")) else line+"\n"
                       for line in origin_data[ind:])
        model = FEA_INP.from_text(text)
        self.name = model.assemble.name
        self.instances = model.assemble.instances
        self.node_blocks = model.assemble.node_blocks
        self.node_sets = model.assemble.node_sets
        self.element_sets = model.assemble.element_sets
        self.surfaces = model.assemble.surfaces

    def write_inp(self, stream: TextIO) -> None:
        model = FEA_INP()
        model.records.append(AssemblyDefinition(self.name))
        for instance in self.instances:
            model.records.extend([instance, RawRecord("END INSTANCE", "*End Instance\n")])
        model.records.extend(self.node_blocks)
        model.records.extend(self.node_sets)
        model.records.extend(self.element_sets)
        model.records.extend(self.surfaces)
        model.records.append(RawRecord("END ASSEMBLY", "*End Assembly\n"))
        stream.write(model.to_text())


@dataclass(slots=True)
class Steps:
    name: str = ""
    definition: StepDefinition | None = None
    static: Static | None = None
    boundaries: list[Boundary] = field(default_factory=list)
    concentrated_loads: list[ConcentratedLoad] = field(default_factory=list)
    distributed_loads: list[DistributedLoad] = field(default_factory=list)
    surface_loads: list[SurfaceLoad] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    node_outputs: list[NodeOutput] = field(default_factory=list)
    element_outputs: list[ElementOutput] = field(default_factory=list)
    other_records: list[InpRecord] = field(default_factory=list)


class FEA_INP:
    __slots__ = ("part", "material", "assemble", "steps", "mesh", "records",
                 "preamble", "source_path")
    Parts = Parts
    Materials = Materials
    Assembly = Assembly
    Steps = Steps

    def __init__(self) -> None:
        self.part: list[Parts] = []
        self.material: list[Materials] = []
        self.assemble: Assembly = Assembly()
        self.steps: list[Steps] = []
        self.mesh: Parts = Parts("Global")
        self.records: list[InpRecord] = []
        self.preamble: str = ""
        self.source_path: Path | None = None

    @classmethod
    def from_text(cls, text: str) -> FEA_INP:
        model = cls()
        model.read_text(text)
        return model

    def read_text(self, text: str) -> None:
        self.records = []
        self.preamble = ""
        lines = text.splitlines(keepends=True)
        if lines and lines[0].startswith("\ufeff"):
            self.preamble = "\ufeff"
            lines[0] = lines[0][1:]
        index = 0
        while index < len(lines):
            if not lines[index].lstrip().startswith("*") or lines[index].lstrip().startswith("**"):
                self.preamble += lines[index]
                index += 1
                continue
            header = lines[index]
            index += 1
            while header.rstrip().endswith(",") and index < len(lines) and lines[index].lstrip()[:1].isalpha():
                header += lines[index]
                index += 1
            data = []
            while index < len(lines):
                line = lines[index].lstrip()
                if line.startswith("*") and not line.startswith("**"):
                    break
                data.append(lines[index])
                index += 1
            self.records.append(_parse_record(header, data))
        self.refresh_views()

    def append_line(self, line: str) -> None:
        """Internal incremental construction; promoted to fixed record classes on refresh."""
        if line.lstrip().startswith("*") and not line.lstrip().startswith("**"):
            self.records.append(RawRecord(line.lstrip()[1:].split(",", 1)[0].strip().upper(), line))
        elif self.records:
            last = self.records[-1]
            if not isinstance(last, RawRecord):
                if line.lstrip().startswith("**"):
                    self.records.append(RawRecord("COMMENT", line))
                else:
                    raise TypeError("Data lines must follow a keyword record.")
            else:
                last.text += line
        else:
            self.preamble += line

    def refresh_views(self) -> None:
        promoted = []
        for record in self.records:
            if isinstance(record, RawRecord):
                if record.keyword_name == "COMMENT":
                    promoted.append(record)
                    continue
                lines = record.text.splitlines(keepends=True)
                header = lines.pop(0)
                while header.rstrip().endswith(",") and lines and lines[0].lstrip()[:1].isalpha():
                    header += lines.pop(0)
                record = _parse_record(header, lines)
            promoted.append(record)
        self.records = promoted
        self.part, self.material, self.steps = [], [], []
        self.assemble = Assembly()
        self.mesh = Parts("Global")
        part: Parts | None = None
        material: Materials | None = None
        step: Steps | None = None
        in_assembly = False
        for record in self.records:
            if isinstance(record, PartDefinition):
                part = Parts(record.name)
                self.part.append(part)
                material = None
            elif record.keyword == "END PART":
                part = None
            elif isinstance(record, AssemblyDefinition):
                self.assemble.name = record.name
                in_assembly = True
                material = None
            elif record.keyword == "END ASSEMBLY":
                in_assembly = False
            elif isinstance(record, MaterialDefinition):
                material = Materials(record.name)
                self.material.append(material)
            elif isinstance(record, StepDefinition):
                step = Steps(record.name, record)
                self.steps.append(step)
                material = None
            elif record.keyword == "END STEP":
                step = None
            elif isinstance(record, Elastic) and material is not None:
                material.elastic = record
            elif isinstance(record, Hyperelastic) and material is not None:
                material.hyperelastic = record
            elif isinstance(record, Density) and material is not None:
                material.density = record
            elif isinstance(record, Instance):
                self.assemble.instances.append(record)
            elif step is not None:
                if isinstance(record, Static):
                    step.static = record
                elif isinstance(record, Boundary):
                    step.boundaries.append(record)
                elif isinstance(record, ConcentratedLoad):
                    step.concentrated_loads.append(record)
                elif isinstance(record, SurfaceLoad):
                    step.surface_loads.append(record)
                elif isinstance(record, DistributedLoad):
                    step.distributed_loads.append(record)
                elif isinstance(record, Output):
                    step.outputs.append(record)
                elif isinstance(record, NodeOutput):
                    step.node_outputs.append(record)
                elif isinstance(record, ElementOutput):
                    step.element_outputs.append(record)
                else:
                    step.other_records.append(record)
            else:
                target = part if part is not None else self.mesh
                if isinstance(record, Nodes):
                    target.node_blocks.append(record)
                    if in_assembly:
                        self.assemble.node_blocks.append(record)
                elif isinstance(record, Elements):
                    target.elements.append(record)
                elif isinstance(record, ElementSet):
                    target.element_sets.append(record)
                    if in_assembly:
                        self.assemble.element_sets.append(record)
                elif isinstance(record, NodeSet):
                    target.node_sets.append(record)
                    if in_assembly:
                        self.assemble.node_sets.append(record)
                elif isinstance(record, Surface):
                    target.surfaces.append(record)
                    if in_assembly:
                        self.assemble.surfaces.append(record)
                elif isinstance(record, SolidSection):
                    target.sections.append(record)
        # Inline sets on *NODE and *ELEMENT are visible as typed set records.
        for target in [*self.part, self.mesh]:
            for block in target.node_blocks:
                if block.node_set:
                    target.node_sets.append(NodeSet(block.node_set, list(map(str, block.labels))))
            for block in target.elements:
                if block.element_set:
                    target.element_sets.append(ElementSet(block.element_set, list(map(str, block.labels))))

    def find(self, keyword: str) -> list[InpRecord]:
        return [record for record in self.records if record.keyword == keyword.upper()]

    def to_text(self) -> str:
        return self.preamble + "".join(record.to_text() for record in self.records)

    def read_inp(self, path: str | Path) -> None:
        path = Path(path)
        self.read_text(path.read_bytes().decode("utf-8", errors="surrogateescape"))
        self.source_path = path.resolve()

    def write_inp(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_text().encode("utf-8", errors="surrogateescape"))

    @classmethod
    def from_controller(
        cls,
        controller: FEAController,
        *,
        nlgeom: bool = True,
        initial_increment: float = 0.1,
        minimum_increment: float = 1e-6,
        maximum_increment: float = 0.1,
        maximum_increments: int = 1000,
        native_interactions: bool = False,
    ) -> FEA_INP:
        from .converters import controller_to_inp
        return controller_to_inp(
            controller, nlgeom=nlgeom,
            initial_increment=initial_increment,
            minimum_increment=minimum_increment,
            maximum_increment=maximum_increment,
            maximum_increments=maximum_increments,
            native_interactions=native_interactions,
        )

