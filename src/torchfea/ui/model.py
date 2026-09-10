"""Persistent CAE definition and its conversion to native torchfea objects.

The UI keeps a compact JSON-safe definition rather than serialising Qt state.
``build_controller`` is the only boundary where this definition becomes a
``Part``/``Assembly``/``FEAController`` ready for the existing solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

import numpy as np
import torch

from .logging_utils import LOGGER


def parse_indices(text: str) -> np.ndarray:
    """Parse ``0, 3:6, 12`` as zero-based inclusive integer ids."""
    values: list[int] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start, stop = (int(v.strip()) for v in token.split(":", 1))
            step = 1 if stop >= start else -1
            values.extend(range(start, stop + step, step))
        else:
            values.append(int(token))
    return np.unique(np.asarray(values, dtype=int))


def parse_faces(text: str) -> list[tuple[np.ndarray, int]]:
    """Parse ``0:1, 4:6`` to the (element ids, zero-based face id) API."""
    grouped: dict[int, list[int]] = {}
    for element, side in re.findall(r"([^,:\s]+)\s*:\s*(\d+)", text):
        face = int(side) - 1
        if face < 0:
            raise ValueError("Surface side numbers begin at 1.")
        grouped.setdefault(face, []).extend(parse_indices(element).tolist())
    if not grouped:
        raise ValueError("Use element:side pairs, for example 0:1, 2:3.")
    return [(np.unique(ids), face) for face, ids in grouped.items()]


@dataclass
class CADOperation:
    """One editable native Gmsh OCC feature in a Part history."""

    id: str
    kind: str  # OCC primitive | boolean | translate | rotate | dilate | mirror
    values: dict[str, float | int] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


@dataclass
class GeometryDefinition:
    kind: str = "box"  # legacy/base feature kind
    # box: xmin xmax ymin ymax zmin zmax nx ny nz
    # cylinder: x y z dx dy dz radius mesh_size
    values: dict[str, float | int] = field(default_factory=lambda: {
        "xmin": 0.0, "xmax": 20.0, "ymin": 0.0, "ymax": 10.0,
        "zmin": 0.0, "zmax": 5.0, "nx": 9, "ny": 5, "nz": 3,
        "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0,
        "dz": 20.0, "radius": 5.0, "mesh_size": 1.5,
    })
    operations: list[CADOperation] = field(default_factory=list)

    def ensure_operations(self) -> list[CADOperation]:
        """Promote a legacy one-primitive geometry to an OCC feature list."""
        if not self.operations:
            LOGGER.info("promoting legacy GeometryDefinition to OCC history: kind=%s", self.kind)
            if self.kind == "box" and "xmin" in self.values:
                values = {
                    "x": float(self.values["xmin"]), "y": float(self.values["ymin"]), "z": float(self.values["zmin"]),
                    "dx": float(self.values["xmax"]) - float(self.values["xmin"]),
                    "dy": float(self.values["ymax"]) - float(self.values["ymin"]),
                    "dz": float(self.values["zmax"]) - float(self.values["zmin"]),
                }
            else:
                values = {key: value for key, value in self.values.items() if key != "mesh_size"}
            self.operations = [CADOperation("Base-1", self.kind, values)]
        return self.operations


@dataclass
class NamedSet:
    name: str
    selection: str


def surface_node_set_name(surface_name: str) -> str:
    """Stable name for the generated node set belonging to one surface."""
    return f"surface_{surface_name}_nodes"


def auto_node_set_names(part: "PartDefinition") -> set[str]:
    """Names reserved for generated node collections on a Part."""
    return {
        "all",
        "surface_all",
        *(surface_node_set_name(item.name) for item in part.surface_sets),
    }


@dataclass
class MaterialDefinition:
    """A Part material and the element set to which it is assigned."""

    name: str = "Material-1"
    kind: str = "linear_elastic"  # linear_elastic | neo_hookean | neo_hookean_lnj
    young: float = 210000.0
    poisson: float = 0.3
    mu: float = 1.0
    kappa: float = 1.0
    element_set: str = "all"


@dataclass
class PartDefinition:
    name: str = "Part-1"
    geometry: GeometryDefinition = field(default_factory=GeometryDefinition)
    young: float = 210000.0
    poisson: float = 0.3
    node_sets: list[NamedSet] = field(default_factory=list)
    element_sets: list[NamedSet] = field(default_factory=list)
    surface_sets: list[NamedSet] = field(default_factory=list)
    # The complete exterior boundary belongs to the Part, not to an Instance.
    # It is always present at runtime and may only be renamed.
    exterior_surface: str = "extern"
    # Keep this field after the legacy fields so positional construction of a
    # PartDefinition from older scripts still maps node/element/surface sets
    # to the same parameters.
    materials: list[MaterialDefinition] = field(default_factory=lambda: [MaterialDefinition()])
    # Density belongs to the Part material collection and is shared by all
    # material assignments on this Part.  Keep it after ``materials`` for
    # positional compatibility with older UI scripts.
    density: float = 1.0

    def __post_init__(self):
        # Preserve the legacy constructor ``PartDefinition(young=..., poisson=...)``
        # when no explicit material collection was supplied.
        if not self.materials:
            self.materials = [MaterialDefinition(
                young=float(self.young), poisson=float(self.poisson))]
        elif (len(self.materials) == 1
              and self.materials[0].name == "Material-1"
              and self.materials[0].kind == "linear_elastic"
              and self.materials[0].element_set == "all"
              and self.materials[0].young == 210000.0
              and self.materials[0].poisson == .3
              and (self.young != 210000.0 or self.poisson != .3)):
            self.materials[0].young = float(self.young)
            self.materials[0].poisson = float(self.poisson)


@dataclass
class InstanceDefinition:
    name: str = "Part-1-1"
    part_name: str = "Part-1"
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Compatibility field for old JSON.  The UI/controller no longer use it.
    exterior: str = ""


@dataclass
class ReferencePointDefinition:
    name: str = "RP-1"
    point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class BoundaryDefinition:
    name: str = "BC-1"
    instance: str = "Part-1-1"
    node_set: str = "nodes_xmin"
    dofs: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class LoadDefinition:
    name: str = "Load-1"
    kind: str = "pressure"  # pressure | body_force | concentrated_force
    instance: str = "Part-1-1"
    target: str = "surface_zmax"  # surface / element name / reference point
    values: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class CAEProject:
    label: str = "untitled"
    parts: list[PartDefinition] = field(default_factory=lambda: [PartDefinition()])
    instances: list[InstanceDefinition] = field(default_factory=lambda: [InstanceDefinition()])
    reference_points: list[ReferencePointDefinition] = field(default_factory=list)
    boundaries: list[BoundaryDefinition] = field(default_factory=list)
    loads: list[LoadDefinition] = field(default_factory=list)
    solver_maximum_iteration: int = 100
    solver_tolerance: float = 1e-5
    mesh_size: float = 1.5

    def to_dict(self) -> dict:
        data = asdict(self)
        # Do not persist the obsolete instance-level exterior-surface field;
        # ``from_dict`` still accepts it so older projects remain readable.
        for instance in data.get("instances", []):
            instance.pop("exterior", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CAEProject":
        def load_named_sets(raw_sets):
            return [NamedSet(**set_data) for set_data in raw_sets]

        part_definitions = []
        for part_data in data.get("parts", []):
            geometry_data = part_data.get("geometry", {})
            geometry = GeometryDefinition(
                kind=geometry_data.get("kind", "box"),
                values=geometry_data.get("values", GeometryDefinition().values),
                operations=[CADOperation(**operation_data)
                            for operation_data in geometry_data.get("operations", [])],
            )
            # Older projects stored one material as Part.young/poisson.
            # Promote that pair into the new Part-owned material collection.
            raw_materials = part_data.get("materials")
            if raw_materials:
                material_definitions = [MaterialDefinition(**material_data)
                                        for material_data in raw_materials]
            else:
                material_definitions = [MaterialDefinition(
                    name="Material-1", kind="linear_elastic",
                    young=float(part_data.get("young", 210000.0)),
                    poisson=float(part_data.get("poisson", .3)),
                    element_set="all")]
            part_definitions.append(PartDefinition(
                name=part_data.get("name", "Part-1"), geometry=geometry,
                young=part_data.get("young", 210000.0),
                poisson=part_data.get("poisson", .3),
                materials=material_definitions,
                density=float(part_data.get("density", 1.0)),
                node_sets=load_named_sets(part_data.get("node_sets", [])),
                element_sets=load_named_sets(part_data.get("element_sets", [])),
                surface_sets=load_named_sets(part_data.get("surface_sets", [])),
                exterior_surface=part_data.get("exterior_surface", "extern")))
        if not part_definitions:
            LOGGER.warning("CAE definition contains no Parts; creating default Part-1")
        project = cls(
            label=data.get("label", "untitled"),
            parts=part_definitions or [PartDefinition()],
            instances=[InstanceDefinition(**item) for item in data.get("instances", [])],
            reference_points=[ReferencePointDefinition(**item)
                             for item in data.get("reference_points", [])],
            boundaries=[BoundaryDefinition(**item)
                        for item in data.get("boundaries", [])],
            loads=[LoadDefinition(**item) for item in data.get("loads", [])],
            # JSON producers and older UI versions may serialize integral
            # controls as ``100.0``.  Keep those values numeric but integral
            # before they reach Python ``range``/TorchScript APIs.
            solver_maximum_iteration=int(round(float(data.get("solver_maximum_iteration", 100)))),
            solver_tolerance=float(data.get("solver_tolerance", 1e-5)),
            mesh_size=float(data.get("mesh_size", 1.5)))
        if not project.instances:
            LOGGER.warning("CAE definition contains no Instances; creating default instance")
            project.instances = [InstanceDefinition(part_name=project.parts[0].name)]
        return project

    def save(self, path: str) -> str:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output)

    @classmethod
    def load(cls, path: str) -> "CAEProject":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_controller(project: CAEProject):
    """Build a solver-ready native model from the UI definition.

    This intentionally calls the public CAD and Assembly APIs, so a model
    created here has the same semantics as one authored directly in Python.
    """
    from torchfea import FEAController
    from torchfea.model import Assembly, Instance, ReferencePoint, loads, boundarys
    from torchfea.model.elements.materials import LinearElastic, NeoHookean, NeoHookeanLnJ
    from torchfea.solver.static import StaticImplicitSolver
    from .gmsh_cad import create_part

    _validate_project_references(project)
    LOGGER.info("controller build start: label=%r parts=%d instances=%d mesh_size=%r (%s)",
                project.label, len(project.parts), len(project.instances),
                project.mesh_size, type(project.mesh_size).__name__)
    assembly = Assembly()
    for definition in project.parts:
        operations = definition.geometry.ensure_operations()
        LOGGER.info("building Part %s: operations=%d node_sets=%d element_sets=%d surface_sets=%d",
                    definition.name, len(operations),
                    len(definition.node_sets), len(definition.element_sets), len(definition.surface_sets))
        LOGGER.info("Part %s material density=%r", definition.name, definition.density)
        part = create_part(operations, project.mesh_size)
        exterior_name = str(definition.exterior_surface).strip()
        if not exterior_name:
            raise ValueError(f"Part '{definition.name}' exterior surface name cannot be empty.")
        if exterior_name != "extern":
            if exterior_name in part.surfaces:
                raise ValueError(
                    f"Part '{definition.name}' exterior surface name '{exterior_name}' is already in use.")
            part.surfaces[exterior_name] = part.surfaces["extern"]
            del part.surfaces._surface_dict["extern"]
        part.exterior_surface = exterior_name
        LOGGER.debug("Part %s exterior surface=%s (all boundary faces)", definition.name, exterior_name)
        LOGGER.debug("Part %s meshed: nodes=%d element_blocks=%s", definition.name,
                     len(part.nodes), {name: len(element._elems) for name, element in part.elems.items()})
        density = torch.full(
            (1,), float(definition.density), dtype=torch.get_default_dtype(),
            device=part.nodes.device)
        for element in part.elems.values():
            element.density = density.expand(len(element._elems)).clone()
        LOGGER.debug("Part %s element density assigned: %g", definition.name, float(definition.density))

        for node_set in definition.node_sets:
            LOGGER.debug("resolving node set %s selection=%r", node_set.name, node_set.selection)
            if node_set.selection.startswith("gmsh:vertex:"):
                tag = int(node_set.selection.rsplit(":", 1)[1])
                try:
                    selected_node_indices = part._gmsh_vertex_nodes[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Node set '{node_set.name}' references missing Gmsh vertex tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            elif node_set.selection.startswith("gmsh:surface:"):
                tag = int(node_set.selection.rsplit(":", 1)[1])
                try:
                    selected_node_indices = part._gmsh_surface_nodes[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Node set '{node_set.name}' references missing Gmsh surface tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            else:
                selected_node_indices = (
                    np.arange(len(part.nodes))
                    if node_set.selection.strip().casefold() == "all"
                    else parse_indices(node_set.selection))
            if node_set.name in auto_node_set_names(definition):
                raise ValueError(f"Node set name '{node_set.name}' on Part '{definition.name}' is reserved.")
            if node_set.name in {f"gmsh_nodes_{tag}" for tag in part._gmsh_surface_nodes}:
                raise ValueError(f"Node set name '{node_set.name}' on Part '{definition.name}' is reserved.")
            selected_node_indices = np.asarray(selected_node_indices, dtype=int).ravel()
            if (not len(selected_node_indices)
                    or np.any(selected_node_indices < 0)
                    or np.any(selected_node_indices >= len(part.nodes))):
                raise ValueError(
                    f"Node set '{node_set.name}' on Part '{definition.name}' contains no valid node indices.")
            part.add_node_set(node_set.name, selected_node_indices)
        for element_set in definition.element_sets:
            LOGGER.debug("resolving element set %s selection=%r", element_set.name, element_set.selection)
            if element_set.selection.startswith("gmsh:volume:"):
                tag = int(element_set.selection.rsplit(":", 1)[1])
                try:
                    selected_element_indices = part._gmsh_volume_elements[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Element set '{element_set.name}' references missing Gmsh volume tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            else:
                selected_element_indices = parse_indices(element_set.selection)
            element_count = len(part.elems["C3D4"]._elems)
            selected_element_indices = np.asarray(selected_element_indices, dtype=int).ravel()
            if (not len(selected_element_indices)
                    or np.any(selected_element_indices < 0)
                    or np.any(selected_element_indices >= element_count)):
                raise ValueError(
                    f"Element set '{element_set.name}' on Part '{definition.name}' contains no valid element indices.")
            if element_set.name == "all":
                raise ValueError(f"Element set name 'all' on Part '{definition.name}' is reserved.")
            part.add_element_set(element_set.name, selected_element_indices)
        for surface_set in definition.surface_sets:
            LOGGER.debug("resolving surface set %s selection=%r", surface_set.name, surface_set.selection)
            if surface_set.selection.startswith("gmsh:surface:"):
                tag = int(surface_set.selection.rsplit(":", 1)[1])
                try:
                    selected_surface_faces = part._gmsh_surface_sets[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Surface set '{surface_set.name}' references missing Gmsh surface tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            else:
                selected_surface_faces = parse_faces(surface_set.selection)
            if surface_set.name in {f"gmsh_surface_{tag}" for tag in part._gmsh_surface_sets}:
                raise ValueError(f"Surface set name '{surface_set.name}' on Part '{definition.name}' is reserved.")
            if not selected_surface_faces:
                raise ValueError(f"Surface set '{surface_set.name}' on Part '{definition.name}' is empty.")
            for element_ids, side in selected_surface_faces:
                element_ids = np.asarray(element_ids, dtype=int).ravel()
                if (not len(element_ids) or np.any(element_ids < 0)
                        or np.any(element_ids >= len(part.elems["C3D4"]._elems))
                        or int(side) not in range(4)):
                    raise ValueError(
                        f"Surface set '{surface_set.name}' on Part '{definition.name}' contains an invalid element face.")
            part.add_surface_set(surface_set.name, selected_surface_faces)
            # Keep a node-set companion for every user-defined surface.  The
            # surface grammar stores element ids plus a local tetra face id;
            # deriving nodes here guarantees that the companion remains in
            # sync with the exact surface selection used by pressure loads.
            face_nodes = []
            element_data = part.elems["C3D4"]
            for element_ids, side in selected_surface_faces:
                indices = np.asarray(element_ids, dtype=int).ravel()
                local_nodes = element_data.surfaceid_map[int(side)]
                face_nodes.append(
                    element_data._elems[indices][:, local_nodes].detach().cpu().numpy().ravel())
            generated_name = surface_node_set_name(surface_set.name)
            if generated_name in part.set_nodes:
                raise ValueError(
                    f"Surface set '{surface_set.name}' generated node set '{generated_name}' already exists.")
            part.add_node_set(generated_name, np.unique(np.concatenate(face_nodes)))
            LOGGER.debug("generated surface node set %s: nodes=%d", generated_name,
                         len(part.set_nodes[generated_name]))
        LOGGER.info("Part %s generated analysis sets: node_sets=%s element_sets=%s",
                    definition.name, sorted(part.set_nodes), sorted(part.set_elements))
        material_definitions = definition.materials or [MaterialDefinition(
            name="Material-1", kind="linear_elastic", young=definition.young,
            poisson=definition.poisson, element_set="all")]
        element_count = len(part.elems["C3D4"]._elems)
        assigned = np.full(element_count, -1, dtype=int)
        for material_index, material_definition in enumerate(material_definitions):
            target_set = str(material_definition.element_set).strip()
            material_element_indices = np.asarray(
                part.set_elements.get(target_set, []), dtype=int).ravel()
            if target_set == "all":
                material_element_indices = np.arange(element_count, dtype=int)
            if (not len(material_element_indices)
                    or np.any(material_element_indices < 0)
                    or np.any(material_element_indices >= element_count)):
                raise ValueError(
                    f"Material '{material_definition.name}' on Part '{definition.name}' "
                    f"references an invalid element set '{target_set}'.")
            if np.any(assigned[material_element_indices] >= 0):
                overlap = material_element_indices[assigned[material_element_indices] >= 0]
                raise ValueError(
                    f"Material '{material_definition.name}' on Part '{definition.name}' "
                    f"overlaps another material on element indices {overlap.tolist()}.")
            kind = str(material_definition.kind).strip().casefold()
            if kind == "linear_elastic":
                material = LinearElastic(
                    torch.tensor([float(material_definition.young)], dtype=torch.get_default_dtype()),
                    torch.tensor([float(material_definition.poisson)], dtype=torch.get_default_dtype()))
            elif kind == "neo_hookean":
                material = NeoHookean(
                    torch.tensor([float(material_definition.mu)], dtype=torch.get_default_dtype()),
                    torch.tensor([float(material_definition.kappa)], dtype=torch.get_default_dtype()))
            elif kind == "neo_hookean_lnj":
                material = NeoHookeanLnJ(
                    torch.tensor([float(material_definition.mu)], dtype=torch.get_default_dtype()),
                    torch.tensor([float(material_definition.kappa)], dtype=torch.get_default_dtype()))
            else:
                raise ValueError(
                    f"Unknown material type '{material_definition.kind}' on Part '{definition.name}'.")
            for element in part.elems.values():
                element.set_materials(
                    material, material_definition.name,
                    element_indices=material_element_indices)
            assigned[material_element_indices] = material_index
            LOGGER.debug("assigned material %s kind=%s to Part %s element_set=%s elements=%d",
                         material_definition.name, kind, definition.name,
                         target_set, len(material_element_indices))
        if np.any(assigned < 0):
            missing = np.flatnonzero(assigned < 0).tolist()
            raise ValueError(
                f"Part '{definition.name}' has no material assigned to element indices {missing}.")
        LOGGER.info("Part %s materials=%s", definition.name,
                    [item.name for item in material_definitions])
        assembly.add_part(part, definition.name)

    for definition in project.instances:
        part_definition = next(item for item in project.parts if item.name == definition.part_name)
        exterior = str(part_definition.exterior_surface).strip()
        LOGGER.debug("adding instance %s part=%s translation=%s rotation=%s",
                     definition.name, definition.part_name, definition.translation,
                     definition.rotation)
        part = assembly.get_part(definition.part_name)
        if exterior not in part.surfaces:
            raise ValueError(
                f"Part '{definition.part_name}' exterior surface '{exterior}' is missing "
                f"on Part '{definition.part_name}'.")
        assembly.add_instance(Instance(
            definition.part_name, definition.translation, definition.rotation), definition.name)
    for definition in project.reference_points:
        LOGGER.debug("adding reference point %s point=%s", definition.name, definition.point)
        assembly.add_reference_point(ReferencePoint(definition.point), definition.name)
    for definition in project.boundaries:
        LOGGER.debug("adding boundary %s instance=%s node_set=%s dofs=%s", definition.name,
                     definition.instance, definition.node_set, definition.dofs)
        assembly.add_boundary(boundarys.Boundary_Condition(
            definition.instance, definition.node_set, definition.dofs), definition.name)
    for definition in project.loads:
        LOGGER.debug("adding load %s kind=%s instance=%s target=%s values=%s", definition.name,
                     definition.kind, definition.instance, definition.target, definition.values)
        if definition.kind == "pressure":
            load = loads.Pressure(definition.instance, definition.target, definition.values[0])
        elif definition.kind == "body_force":
            load = loads.BodyForce(definition.instance, definition.target, definition.values[:3])
        elif definition.kind == "concentrated_force":
            load = loads.Concentrate_Force(definition.target, definition.values[:3])
        else:
            raise ValueError(f"Unknown load type: {definition.kind}")
        assembly.add_load(load, definition.name)

    controller = FEAController()
    controller.assembly = assembly
    maximum_iteration = int(round(float(project.solver_maximum_iteration)))
    tolerance = float(project.solver_tolerance)
    LOGGER.debug("solver settings normalized: maximum_iteration=%d tolerance=%r", maximum_iteration, tolerance)
    controller.solver = StaticImplicitSolver(maximum_iteration, tolerance)
    LOGGER.info("controller build complete: parts=%d instances=%d loads=%d boundaries=%d",
                len(assembly._parts), len(assembly._instances), len(assembly._loads), len(assembly._boundarys))
    return controller


def _validate_project_references(project: CAEProject) -> None:
    """Reject references that the native Assembly API would otherwise defer.

    Assembly's ``add_*`` methods intentionally accept lightweight objects and
    overwrite duplicate names.  That is useful for low-level scripting but is
    a dangerous fallback for a UI definition: a typo can survive model build
    and only fail later during solver initialization.  Validate the complete
    project graph at the UI-to-model boundary instead.
    """
    if not project.parts:
        raise ValueError("CAE project needs at least one Part.")
    if not project.instances:
        raise ValueError("CAE project needs at least one Instance.")

    def unique_names(items, kind: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in items:
            name = str(getattr(item, "name", "")).strip()
            if not name:
                raise ValueError(f"{kind} name cannot be empty.")
            if name in result:
                raise ValueError(f"Duplicate {kind} name '{name}'.")
            result[name] = item
        return result

    parts = unique_names(project.parts, "Part")
    instances = unique_names(project.instances, "Instance")
    reference_points = unique_names(project.reference_points, "Reference point")
    unique_names(project.boundaries, "Boundary")
    unique_names(project.loads, "Load")
    for part in project.parts:
        if not np.isfinite(float(part.density)) or float(part.density) <= 0:
            raise ValueError(f"Material density on Part '{part.name}' must be positive and finite.")
        unique_names(part.materials, f"material on Part '{part.name}'")
        unique_names(part.node_sets, f"node set on Part '{part.name}'")
        unique_names(part.element_sets, f"element set on Part '{part.name}'")
        unique_names(part.surface_sets, f"surface set on Part '{part.name}'")
        material_element_sets = {"all", *(item.name for item in part.element_sets)}
        material_targets = []
        for material in part.materials:
            kind = str(material.kind).strip().casefold()
            if kind not in {"linear_elastic", "neo_hookean", "neo_hookean_lnj"}:
                raise ValueError(
                    f"Material '{material.name}' on Part '{part.name}' has unknown type '{material.kind}'.")
            target = str(material.element_set).strip()
            if target not in material_element_sets:
                raise ValueError(
                    f"Material '{material.name}' on Part '{part.name}' references missing element set '{target}'.")
            material_targets.append(target)
            if kind == "linear_elastic":
                if not np.isfinite(float(material.young)) or float(material.young) <= 0:
                    raise ValueError(f"Material '{material.name}' Young's modulus must be positive.")
                if not np.isfinite(float(material.poisson)) or not 0 <= float(material.poisson) < .5:
                    raise ValueError(f"Material '{material.name}' Poisson ratio must be in [0, 0.5).")
            else:
                if not np.isfinite(float(material.mu)) or float(material.mu) <= 0:
                    raise ValueError(f"Material '{material.name}' mu must be positive.")
                if not np.isfinite(float(material.kappa)) or float(material.kappa) <= 0:
                    raise ValueError(f"Material '{material.name}' kappa must be positive.")
        if len(material_targets) != len(set(material_targets)):
            raise ValueError(f"Material element sets on Part '{part.name}' overlap by name.")
        exterior_name = str(part.exterior_surface).strip()
        if not exterior_name:
            raise ValueError(f"Part '{part.name}' exterior surface name cannot be empty.")
        if exterior_name.startswith("gmsh_surface_"):
            raise ValueError(
                f"Part '{part.name}' exterior surface name '{exterior_name}' is reserved.")
        if any(item.name == exterior_name for item in part.surface_sets):
            raise ValueError(
                f"Part '{part.name}' exterior surface '{exterior_name}' must be unique.")
        reserved_node_names = auto_node_set_names(part)
        if any(item.name in reserved_node_names for item in part.node_sets):
            raise ValueError(
                f"Node set names {sorted(reserved_node_names)} on Part '{part.name}' are reserved.")
        if any(item.name == "all" for item in part.element_sets):
            raise ValueError(f"Element set name 'all' on Part '{part.name}' is reserved.")

    for instance in instances.values():
        if instance.part_name not in parts:
            raise ValueError(
                f"Instance '{instance.name}' references missing Part '{instance.part_name}'.")
        if len(instance.translation) != 3 or len(instance.rotation) != 3:
            raise ValueError(f"Instance '{instance.name}' translation and rotation need 3 values.")

    for point in reference_points.values():
        if len(point.point) != 3:
            raise ValueError(f"Reference point '{point.name}' needs 3 coordinates.")

    if int(round(float(project.solver_maximum_iteration))) <= 0:
        raise ValueError("Solver maximum iteration must be positive.")
    if not np.isfinite(float(project.solver_tolerance)) or float(project.solver_tolerance) <= 0:
        raise ValueError("Solver tolerance must be a positive finite number.")
    if not np.isfinite(float(project.mesh_size)) or float(project.mesh_size) <= 0:
        raise ValueError("Global mesh size must be a positive finite number.")

    for boundary in project.boundaries:
        instance = instances.get(boundary.instance)
        if instance is None:
            raise ValueError(
                f"Boundary '{boundary.name}' references missing Instance '{boundary.instance}'.")
        part = parts[instance.part_name]
        node_set_names = {item.name for item in part.node_sets} | auto_node_set_names(part)
        if boundary.node_set not in node_set_names:
            raise ValueError(
                f"Boundary '{boundary.name}' references missing node set '{boundary.node_set}' "
                f"on Part '{part.name}'.")
        if not boundary.dofs or any(int(dof) not in (0, 1, 2) for dof in boundary.dofs):
            raise ValueError(f"Boundary '{boundary.name}' DOFs must be a non-empty subset of 0, 1, 2.")

    for load in project.loads:
        if load.instance not in instances and load.kind != "concentrated_force":
            raise ValueError(
                f"Load '{load.name}' references missing Instance '{load.instance}'.")
        if load.kind == "pressure":
            if len(load.values) < 1:
                raise ValueError(f"Pressure load '{load.name}' needs one scalar value.")
            instance = instances[load.instance]
            part = parts[instance.part_name]
            surface_names = {str(part.exterior_surface).strip(), *(item.name for item in part.surface_sets)}
            if load.target not in surface_names:
                raise ValueError(
                    f"Pressure load '{load.name}' references missing surface set '{load.target}' "
                    f"on Part '{part.name}'.")
        elif load.kind == "body_force":
            if len(load.values) < 3:
                raise ValueError(f"Body force '{load.name}' needs three components.")
            instance = instances[load.instance]
            part = parts[instance.part_name]
            # Native Gmsh import currently creates one C3D4 block.  Keep the
            # check against the actual element block names so this remains
            # correct if additional element types are introduced later.
            element_names = {"C3D4"}
            if load.target not in element_names:
                raise ValueError(
                    f"Body force '{load.name}' references unknown element block '{load.target}' "
                    f"on Part '{part.name}'.")
        elif load.kind == "concentrated_force":
            if len(load.values) < 3:
                raise ValueError(f"Concentrated force '{load.name}' needs three components.")
            if load.target not in reference_points:
                raise ValueError(
                    f"Concentrated force '{load.name}' references missing reference point '{load.target}'.")
        else:
            raise ValueError(f"Unknown load type: {load.kind}")
