"""CAE editor definitions and conversion to/from native torchfea objects.

The native model is the only persisted representation.  ``build_controller``
stores the CAD history, assembled discrete model, sets, materials, loads and
constraints in the normal torchfea object graph.  ``from_controller`` reads
that graph back into the editable Qt-facing dataclasses; no UI-side JSON or
metadata blob is written to disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

import numpy as np
import torch

from torchfea.cad import CADOperation, primitive_defaults
from torchfea.interfaces import Serializable


DEFAULT_MESH_SIZE = 1.5


def enabled_value(value: bool | None) -> bool:
    """Convert the native enabled flag using Assembly's default semantics."""
    if value is None:
        return True
    if not isinstance(value, bool):
        raise TypeError(f"enabled must be bool or None, got {type(value).__name__}.")
    return value


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
class GeometryDefinition:
    """Geometry source used by one Part.

    ``cad`` Parts are replayed from OCC operations.  ``discrete`` Parts are
    already-meshed native :class:`torchfea.model.Part` objects, typically
    imported from an INP file and saved in the normal controller ``.npz``.
    The native object is kept at runtime; ``discrete_data`` is only an
    in-memory undo/redo snapshot and is not a second model file.
    """

    operations: list[CADOperation] = field(
        default_factory=lambda: [CADOperation("Base-1", "box", primitive_defaults("box"))])
    source: str = "cad"
    discrete_data: object = field(default=None, repr=False, compare=False)
    runtime_part: object = field(default=None, repr=False, compare=False)


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
    node_sets: list[NamedSet] = field(default_factory=list)
    element_sets: list[NamedSet] = field(default_factory=list)
    surface_sets: list[NamedSet] = field(default_factory=list)
    # The complete exterior boundary belongs to the Part, not to an Instance.
    # It is always present at runtime and may only be renamed.
    exterior_surface: str = "extern"
    materials: list[MaterialDefinition] = field(default_factory=lambda: [MaterialDefinition()])
    density: float = 1.0


@dataclass
class InstanceDefinition:
    name: str = "Part-1-1"
    part_name: str = "Part-1"
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    enabled: bool = True


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
    # ``nodes`` uses an instance node set; ``reference_point`` constrains the
    # six rigid-body degrees of freedom of one assembly reference point.
    kind: str = "nodes"
    reference_point: str = ""
    enabled: bool = True


@dataclass
class ConstraintDefinition:
    """An assembly kinematic constraint backed by torchfea.constraints."""

    name: str = "Constraint-1"
    kind: str = "couple"
    instance: str = "Part-1-1"
    node_set: str = "surface_all"
    reference_point: str = "RP-1"
    enabled: bool = True


@dataclass
class LoadDefinition:
    name: str = "Load-1"
    kind: str = "pressure"  # pressure | body_force | concentrated_force
    instance: str = "Part-1-1"
    target: str = "surface_zmax"  # surface / element name / reference point
    values: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    # Optional second target/instance support for pair-wise contacts and
    # springs.
    instance2: str = ""
    target2: str = ""
    enabled: bool = True


@dataclass
class CAEProject:
    label: str = "untitled"
    parts: list[PartDefinition] = field(default_factory=lambda: [PartDefinition()])
    instances: list[InstanceDefinition] = field(default_factory=lambda: [InstanceDefinition()])
    reference_points: list[ReferencePointDefinition] = field(default_factory=list)
    boundaries: list[BoundaryDefinition] = field(default_factory=list)
    loads: list[LoadDefinition] = field(default_factory=list)
    constraints: list[ConstraintDefinition] = field(default_factory=list)
    solver_maximum_iteration: int = 100
    solver_tolerance: float = 1e-5
    mesh_size: float = DEFAULT_MESH_SIZE

    def to_dict(self) -> dict:
        # ``runtime_part`` is deliberately excluded: it contains live solver
        # state and is already represented by ``discrete_data`` when an undo
        # snapshot needs to recreate a discrete Part.
        return {
            "label": self.label,
            "parts": [
                {
                    "name": part.name,
                    "geometry": {
                        "source": part.geometry.source,
                        "operations": [asdict(operation) for operation in part.geometry.operations],
                        "discrete_data": part.geometry.discrete_data,
                    },
                    "node_sets": [asdict(item) for item in part.node_sets],
                    "element_sets": [asdict(item) for item in part.element_sets],
                    "surface_sets": [asdict(item) for item in part.surface_sets],
                    "exterior_surface": part.exterior_surface,
                    "materials": [asdict(item) for item in part.materials],
                    "density": part.density,
                }
                for part in self.parts
            ],
            "instances": [asdict(item) for item in self.instances],
            "reference_points": [asdict(item) for item in self.reference_points],
            "boundaries": [asdict(item) for item in self.boundaries],
            "loads": [asdict(item) for item in self.loads],
            "constraints": [asdict(item) for item in self.constraints],
            "solver_maximum_iteration": self.solver_maximum_iteration,
            "solver_tolerance": self.solver_tolerance,
            "mesh_size": self.mesh_size,
        }

    @classmethod
    def from_controller(cls, controller) -> "CAEProject":
        """Read the complete CAE definition from a native FEA model.

        CAD operations come from ``FEAController.cad``.  The remaining CAE
        objects are already represented by the native ``Assembly`` and its
        ``Part``/``Instance``/load/constraint objects, so this conversion is
        deliberately a reader rather than a second persistence format.
        """
        assembly = controller.assembly
        cad_model = controller.cad

        def tensor_list(value) -> list[float]:
            if torch.is_tensor(value):
                value = value.detach().cpu().reshape(-1).tolist()
            return [float(item) for item in value]

        def native_indices(value) -> str:
            values = sorted({int(item) for item in np.asarray(value, dtype=int).reshape(-1)})
            ranges: list[str] = []
            start = previous = None
            for item in values:
                if start is None:
                    start = previous = item
                elif item == previous + 1:
                    previous = item
                else:
                    ranges.append(str(start) if start == previous else f"{start}:{previous}")
                    start = previous = item
            if start is not None:
                ranges.append(str(start) if start == previous else f"{start}:{previous}")
            return ", ".join(ranges)

        def native_surface(value) -> str:
            tokens = []
            for element_ids, side in value:
                tokens.extend(
                    f"{int(element)}:{int(side) + 1}"
                    for element in np.asarray(element_ids, dtype=int).reshape(-1))
            return ", ".join(tokens)

        def ensure_native_exterior(native_part):
            """Make imported defaults explicit without mutating the controller.

            ``from_controller`` is a UI projection and must not alter the
            native model that was just loaded.  In particular, adding the
            generated ``surface_all``/surface-node sets directly to that
            Part used to change its serialized model hash before result
            validation.  Rehydrate a lightweight serialized copy and add the
            editor-only defaults there instead.
            """
            native_part = Serializable._deserialize(native_part._serialize())
            exterior = str(native_part.exterior_surface)
            if exterior not in native_part.surfaces:
                boundary = []
                for element in native_part.elems.values():
                    boundary.extend(element.extract_boundary_surface_set())
                if boundary:
                    native_part.surfaces[exterior] = boundary
                else:
                    raise ValueError(
                        f"Discrete Part has no boundary surface: '{type(native_part).__name__}'.")
            native_part.exterior_surface = exterior
            native_part.set_nodes.setdefault("all", np.arange(len(native_part.nodes), dtype=int))
            if native_part.surfaces[exterior]:
                face_nodes = []
                for element_ids, side in native_part.surfaces[exterior]:
                    element_ids = np.asarray(element_ids, dtype=int).reshape(-1)
                    element = next(iter(native_part.elems.values()))
                    face_nodes.append(
                        element._elems[element_ids][:, element.surfaceid_map[int(side)]]
                        .detach().cpu().numpy().reshape(-1))
                native_part.set_nodes.setdefault("surface_all", np.unique(np.concatenate(face_nodes)))
            for name, element in native_part.elems.items():
                native_part.set_elements.setdefault(
                    name if name != "C3D4" else "all",
                    np.arange(len(element._elems), dtype=int))
            return native_part

        def geometry_for(part_name: str, native_part) -> GeometryDefinition:
            history = cad_model.parts.get(part_name)
            operations = list(history.operations) if history is not None else []
            if operations:
                return GeometryDefinition(operations=operations, source="cad")
            native_part = ensure_native_exterior(native_part)
            return GeometryDefinition(
                operations=[], source="discrete",
                discrete_data=native_part._serialize(), runtime_part=native_part)

        def material_definitions(native_part):
            from torchfea.model.elements.materials import LinearElastic, NeoHookean, NeoHookeanLnJ

            element_sets = {
                name: np.asarray(indices, dtype=int).reshape(-1)
                for name, indices in native_part.set_elements.items()
            }
            material_objects = {}
            material_regions = {}
            density_values = []
            for element in native_part.elems.values():
                if element.density is not None:
                    density_values.extend(tensor_list(element.density))
                for name, material in element.materials.items():
                    material_objects.setdefault(name, material)
                for name, indices in element.material_element_indices.items():
                    material_regions.setdefault(name, np.asarray(indices, dtype=int).reshape(-1))
            count = sum(len(element._elems) for element in native_part.elems.values())
            definitions = []
            derived_sets = []
            for name, material in material_objects.items():
                if isinstance(material, LinearElastic):
                    kind = "linear_elastic"
                    values = {"young": float(material.E.reshape(-1)[0]),
                              "poisson": float(material.nu.reshape(-1)[0])}
                elif isinstance(material, NeoHookeanLnJ):
                    kind = "neo_hookean_lnj"
                    values = {"mu": float(material.mu.reshape(-1)[0]),
                              "kappa": float(material.kappa.reshape(-1)[0])}
                elif isinstance(material, NeoHookean):
                    kind = "neo_hookean"
                    values = {"mu": float(material.mu.reshape(-1)[0]),
                              "kappa": float(material.kappa.reshape(-1)[0])}
                else:
                    raise ValueError(
                        f"Unsupported native material '{type(material).__name__}' on Part.")
                region = material_regions.get(name)
                target = "all"
                if region is not None and len(region) != count:
                    for set_name, set_indices in element_sets.items():
                        if np.array_equal(np.sort(region), np.sort(set_indices)):
                            target = set_name
                            break
                    else:
                        target = f"material_{name}"
                        derived_sets.append(NamedSet(target, native_indices(region)))
                definitions.append(MaterialDefinition(name=name, kind=kind,
                                                       element_set=target, **values))
            if not definitions:
                raise ValueError(f"Part '{type(native_part).__name__}' has no materials.")
            if not density_values:
                raise ValueError(f"Part '{type(native_part).__name__}' has no density.")
            density = float(np.mean(density_values))
            return definitions, density, derived_sets

        part_definitions = []
        for name, native_part in assembly._parts.items():
            materials, density, derived_sets = material_definitions(native_part)
            exterior = str(native_part.exterior_surface)
            surfaces = []
            for surface_name in native_part.surfaces.keys():
                surface_name = str(surface_name)
                if surface_name == exterior or surface_name == "extern" or surface_name.startswith("gmsh_surface_"):
                    continue
                surfaces.append(NamedSet(surface_name, native_surface(native_part.surfaces[surface_name])))
            nodes = []
            for set_name, indices in native_part.set_nodes.items():
                set_name = str(set_name)
                if (set_name in {"all", "surface_all"}
                        or set_name.startswith("gmsh_nodes_")
                        or set_name.startswith("surface_") and set_name.endswith("_nodes")):
                    continue
                nodes.append(NamedSet(set_name, native_indices(indices)))
            elements = [NamedSet(str(set_name), native_indices(indices))
                        for set_name, indices in native_part.set_elements.items()
                        if str(set_name) != "all"]
            elements.extend(derived_sets)
            part_definitions.append(PartDefinition(
                name=str(name), geometry=geometry_for(str(name), native_part),
                materials=materials, density=density,
                node_sets=nodes, element_sets=elements,
                surface_sets=surfaces, exterior_surface=exterior))

        instances = []
        for name, instance in assembly._instances.items():
            instances.append(InstanceDefinition(
                name=str(name), part_name=str(instance.part_name),
                translation=tensor_list(instance._translation),
                rotation=tensor_list(instance._rotation),
                enabled=enabled_value(instance.enabled)))

        reference_points = [ReferencePointDefinition(
            name=str(name), point=tensor_list(point.node))
            for name, point in assembly._reference_points.items()]

        boundaries = []
        from torchfea.model import boundarys, constraints, loads
        for name, boundary in assembly._boundarys.items():
            if isinstance(boundary, boundarys.Boundary_Condition_RP):
                boundaries.append(BoundaryDefinition(
                    name=str(name), kind="reference_point",
                    reference_point=str(boundary.rp_name),
                    dofs=[int(item) for item in boundary.indexDoF],
                    enabled=enabled_value(boundary.enabled)))
            elif isinstance(boundary, boundarys.Boundary_Condition):
                boundaries.append(BoundaryDefinition(
                    name=str(name), kind="nodes",
                    instance=str(boundary.instance_name),
                    node_set=str(boundary.set_nodes_name),
                    dofs=[int(item) for item in boundary.indexDoF],
                    enabled=enabled_value(boundary.enabled)))
            else:
                raise ValueError(f"Unsupported native boundary '{type(boundary).__name__}'.")

        project_constraints = []
        for name, constraint in assembly._constraints.items():
            if isinstance(constraint, constraints.Couple):
                project_constraints.append(ConstraintDefinition(
                    name=str(name), kind="couple",
                    instance=str(constraint.instance_name),
                    node_set=str(constraint.set_nodes_name),
                    reference_point=str(constraint.rp_name),
                    enabled=enabled_value(constraint.enabled)))
            else:
                raise ValueError(f"Unsupported native constraint '{type(constraint).__name__}'.")

        project_loads = []
        for name, load in assembly._loads.items():
            values = tensor_list(load._parameters)
            if isinstance(load, loads.Pressure):
                item = LoadDefinition(str(name), "pressure", load.instance_name,
                                      target=load.surface_set, values=values[:1],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.BodyForce):
                item = LoadDefinition(str(name), "body_force", load._instance_name,
                                      target=load._element_name, values=values[:3],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Concentrate_Force):
                item = LoadDefinition(str(name), "concentrated_force",
                                      target=load.rp_name, values=values[:3],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Moment):
                item = LoadDefinition(str(name), "moment", target=load.rp_name,
                                      values=values[:3], enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Spring_RP_RP):
                item = LoadDefinition(str(name), "spring_rp_rp", target=load.rp_name1,
                                      target2=load.rp_name2, values=values[:2],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Spring_RP_Point):
                item = LoadDefinition(str(name), "spring_rp_point", target=load.rp_name,
                                      values=tensor_list(load.point) + [float(load.k),
                                          -1.0 if load.rest_length is None else float(load.rest_length)],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Penalty_DoF):
                item = LoadDefinition(str(name), "penalty_dof", target=load.obj_name,
                                      target2=load.obj_type,
                                      values=[float(load.s), float(load.target), float(load.k)],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.ContactSelf):
                item = LoadDefinition(str(name), "contact_self", load.instance_name,
                                      target=load.surface_name,
                                      values=[float(load._ignore_min_normal),
                                              float(load._ignore_max_normal),
                                              float(load._initial_detact_ratio), *values[:6],
                                              float(load.mesh_size)],
                                      enabled=enabled_value(load.enabled))
            elif isinstance(load, loads.Contact):
                item = LoadDefinition(str(name), "contact", load.instance_name1,
                                      target=load.surface_name1,
                                      instance2=load.instance_name2,
                                      target2=load.surface_name2,
                                      values=[*values[:6],
                                              float(load.mesh_size)],
                                      enabled=enabled_value(load.enabled))
            else:
                raise ValueError(f"Unsupported native load '{type(load).__name__}'.")
            project_loads.append(item)

        solver = controller.solver
        mesh_size = cad_model.mesh_size
        if mesh_size is None:
            mesh_size = next(
                (history.mesh_size for history in cad_model.parts.values()
                 if history.mesh_size is not None),
                None,
            )
        if mesh_size is None:
            # Discrete Parts already contain their FE mesh and therefore have
            # no native CAD mesh size.  Keep the editor usable; this value is
            # only used if the user later creates or remeshes a CAD Part.
            mesh_size = DEFAULT_MESH_SIZE
        return cls(
            label="untitled",
            parts=part_definitions,
            instances=instances,
            reference_points=reference_points,
            boundaries=boundaries,
            constraints=project_constraints,
            loads=project_loads,
            solver_maximum_iteration=int(solver.maximum_iteration),
            solver_tolerance=float(solver.tol_error),
            mesh_size=float(mesh_size),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "CAEProject":
        def load_named_sets(raw_sets):
            return [NamedSet(**set_data) for set_data in raw_sets]

        part_definitions = []
        for part_data in data["parts"]:
            geometry_data = part_data["geometry"]
            source = geometry_data["source"].strip().casefold()
            if source not in {"cad", "discrete"}:
                raise ValueError(f"Unknown geometry source '{source}'.")
            discrete_data = geometry_data["discrete_data"]
            runtime_part = None
            if source == "discrete":
                if discrete_data is None:
                    raise ValueError(f"Discrete Part '{part_data['name']}' has no native mesh.")
                runtime_part = Serializable._deserialize(discrete_data)
            geometry = GeometryDefinition(
                operations=[CADOperation(**operation_data)
                            for operation_data in geometry_data["operations"]],
                source=source,
                discrete_data=discrete_data,
                runtime_part=runtime_part,
            )
            material_definitions = [MaterialDefinition(**material_data)
                                    for material_data in part_data["materials"]]
            part_definitions.append(PartDefinition(
                name=part_data["name"], geometry=geometry,
                materials=material_definitions,
                density=float(part_data["density"]),
                node_sets=load_named_sets(part_data["node_sets"]),
                element_sets=load_named_sets(part_data["element_sets"]),
                surface_sets=load_named_sets(part_data["surface_sets"]),
                exterior_surface=part_data["exterior_surface"]))
        project = cls(
            label=data["label"],
            parts=part_definitions,
            instances=[InstanceDefinition(**item) for item in data["instances"]],
            reference_points=[ReferencePointDefinition(**item)
                             for item in data["reference_points"]],
            boundaries=[BoundaryDefinition(**item)
                        for item in data["boundaries"]],
            constraints=[ConstraintDefinition(**item)
                         for item in data["constraints"]],
            loads=[LoadDefinition(**item) for item in data["loads"]],
            # Keep integral controls numeric before they reach Python
            # ``range``/TorchScript APIs.
            solver_maximum_iteration=int(round(float(data["solver_maximum_iteration"]))),
            solver_tolerance=float(data["solver_tolerance"]),
            mesh_size=float(data["mesh_size"]))
        return project

def build_controller(project: CAEProject):
    """Build a solver-ready native model from the UI definition.

    This intentionally calls the public CAD and Assembly APIs, so a model
    created here has the same semantics as one authored directly in Python.
    """
    from torchfea import FEAController
    from torchfea.model import Assembly, Instance, ReferencePoint, loads, boundarys, constraints
    from torchfea.model.elements.materials import LinearElastic, NeoHookean, NeoHookeanLnJ
    from torchfea.solver.static import StaticImplicitSolver
    from torchfea.cad import CADHistory, CADModel, create_part

    _validate_project_references(project)
    assembly = Assembly()
    cad_model = CADModel(mesh_size=project.mesh_size)
    for definition in project.parts:
        geometry = definition.geometry
        source = geometry.source.strip().casefold()
        if source not in {"cad", "discrete"}:
            raise ValueError(f"Unknown geometry source '{source}'.")
        operations = list(geometry.operations)
        if source == "discrete":
            serialized_part = geometry.discrete_data
            if serialized_part is not None:
                part = Serializable._deserialize(serialized_part)
            else:
                part = geometry.runtime_part
            if part is None:
                raise ValueError(
                    f"Discrete Part '{definition.name}' has no serialized native mesh.")
        else:
            cad_model.set_part(
                definition.name,
                CADHistory(operations, mesh_size=project.mesh_size),
            )
            part = create_part(operations, project.mesh_size)
        exterior_name = str(definition.exterior_surface).strip()
        if not exterior_name:
            raise ValueError(f"Part '{definition.name}' exterior surface name cannot be empty.")
        if source == "discrete":
            current_exterior = str(part.exterior_surface)
            if exterior_name != current_exterior:
                if exterior_name in part.surfaces:
                    raise ValueError(
                        f"Part '{definition.name}' exterior surface name '{exterior_name}' is already in use.")
                if current_exterior not in part.surfaces:
                    raise ValueError(
                        f"Part '{definition.name}' exterior surface '{current_exterior}' is missing.")
                part.surfaces[exterior_name] = part.surfaces[current_exterior]
                del part.surfaces._surface_dict[current_exterior]
        elif exterior_name != "extern":
            if exterior_name in part.surfaces:
                raise ValueError(
                    f"Part '{definition.name}' exterior surface name '{exterior_name}' is already in use.")
            part.surfaces[exterior_name] = part.surfaces["extern"]
            del part.surfaces._surface_dict["extern"]
        part.exterior_surface = exterior_name
        density = torch.full(
            (1,), float(definition.density), dtype=torch.get_default_dtype(),
            device=part.nodes.device)
        for element in part.elems.values():
            element.density = density.expand(len(element._elems)).clone()

        if source == "cad":
            gmsh_vertex_nodes = part._gmsh_vertex_nodes
            gmsh_surface_nodes = part._gmsh_surface_nodes
            gmsh_surface_sets = part._gmsh_surface_sets
            gmsh_volume_elements = part._gmsh_volume_elements
        else:
            gmsh_vertex_nodes = {}
            gmsh_surface_nodes = {}
            gmsh_surface_sets = {}
            gmsh_volume_elements = {}
        element_data = next(iter(part.elems.values()), None)
        if element_data is None:
            raise ValueError(f"Part '{definition.name}' has no element blocks.")
        element_count = len(element_data._elems)
        for node_set in definition.node_sets:
            if node_set.selection.startswith("gmsh:vertex:"):
                tag = int(node_set.selection.rsplit(":", 1)[1])
                try:
                    selected_node_indices = gmsh_vertex_nodes[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Node set '{node_set.name}' references missing Gmsh vertex tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            elif node_set.selection.startswith("gmsh:surface:"):
                tag = int(node_set.selection.rsplit(":", 1)[1])
                try:
                    selected_node_indices = gmsh_surface_nodes[tag]
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
            if node_set.name in {f"gmsh_nodes_{tag}" for tag in gmsh_surface_nodes}:
                raise ValueError(f"Node set name '{node_set.name}' on Part '{definition.name}' is reserved.")
            selected_node_indices = np.asarray(selected_node_indices, dtype=int).ravel()
            if (not len(selected_node_indices)
                    or np.any(selected_node_indices < 0)
                    or np.any(selected_node_indices >= len(part.nodes))):
                raise ValueError(
                    f"Node set '{node_set.name}' on Part '{definition.name}' contains no valid node indices.")
            part.add_node_set(node_set.name, selected_node_indices)
        for element_set in definition.element_sets:
            if element_set.selection.startswith("gmsh:volume:"):
                tag = int(element_set.selection.rsplit(":", 1)[1])
                try:
                    selected_element_indices = gmsh_volume_elements[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Element set '{element_set.name}' references missing Gmsh volume tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            else:
                selected_element_indices = parse_indices(element_set.selection)
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
            if surface_set.selection.startswith("gmsh:surface:"):
                tag = int(surface_set.selection.rsplit(":", 1)[1])
                try:
                    selected_surface_faces = gmsh_surface_sets[tag]
                except KeyError as exc:
                    raise ValueError(
                        f"Surface set '{surface_set.name}' references missing Gmsh surface tag {tag} "
                        f"on Part '{definition.name}'.") from exc
            else:
                selected_surface_faces = parse_faces(surface_set.selection)
            if surface_set.name in {f"gmsh_surface_{tag}" for tag in gmsh_surface_sets}:
                raise ValueError(f"Surface set name '{surface_set.name}' on Part '{definition.name}' is reserved.")
            if not selected_surface_faces:
                raise ValueError(f"Surface set '{surface_set.name}' on Part '{definition.name}' is empty.")
            for element_ids, side in selected_surface_faces:
                element_ids = np.asarray(element_ids, dtype=int).ravel()
                if (not len(element_ids) or np.any(element_ids < 0)
                        or np.any(element_ids >= element_count)
                        or int(side) not in element_data.surfaceid_map):
                    raise ValueError(
                        f"Surface set '{surface_set.name}' on Part '{definition.name}' contains an invalid element face.")
            part.add_surface_set(surface_set.name, selected_surface_faces)
            # Keep a node-set companion for every user-defined surface.  The
            # surface grammar stores element ids plus a local tetra face id;
            # deriving nodes here guarantees that the companion remains in
            # sync with the exact surface selection used by pressure loads.
            face_nodes = []
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
        material_definitions = definition.materials
        assigned = np.full(element_count, -1, dtype=int)
        for material_index, material_definition in enumerate(material_definitions):
            target_set = str(material_definition.element_set).strip()
            if target_set == "all":
                material_element_indices = np.arange(element_count, dtype=int)
            else:
                material_element_indices = np.asarray(
                    part.set_elements[target_set], dtype=int).ravel()
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
        if np.any(assigned < 0):
            missing = np.flatnonzero(assigned < 0).tolist()
            raise ValueError(
                f"Part '{definition.name}' has no material assigned to element indices {missing}.")
        assembly.add_part(part, definition.name)

    for definition in project.instances:
        part_definition = next(item for item in project.parts if item.name == definition.part_name)
        exterior = str(part_definition.exterior_surface).strip()
        part = assembly.get_part(definition.part_name)
        if exterior not in part.surfaces:
            raise ValueError(
                f"Part '{definition.part_name}' exterior surface '{exterior}' is missing "
                f"on Part '{definition.part_name}'.")
        instance = Instance(definition.part_name, definition.translation, definition.rotation)
        instance.enabled = bool(definition.enabled)
        assembly.add_instance(instance, definition.name)
    for definition in project.reference_points:
        assembly.add_reference_point(ReferencePoint(definition.point), definition.name)
    for definition in project.boundaries:
        if definition.kind == "reference_point":
            boundary = boundarys.Boundary_Condition_RP(
                definition.reference_point, definition.dofs)
        elif definition.kind == "nodes":
            boundary = boundarys.Boundary_Condition(
                definition.instance, definition.node_set, definition.dofs)
        else:
            raise ValueError(f"Unknown boundary type: {definition.kind}")
        boundary.enabled = bool(definition.enabled)
        assembly.add_boundary(boundary, definition.name)
    for definition in project.constraints:
        if definition.kind != "couple":
            raise ValueError(f"Unknown constraint type: {definition.kind}")
        constraint = constraints.Couple(
            definition.instance, definition.node_set, definition.reference_point)
        constraint.enabled = bool(definition.enabled)
        assembly.add_constraint(constraint, definition.name)
    for definition in project.loads:
        values = [float(value) for value in definition.values]
        if definition.kind == "pressure":
            load = loads.Pressure(definition.instance, definition.target, values[0])
        elif definition.kind == "body_force":
            load = loads.BodyForce(definition.instance, definition.target, values[:3])
        elif definition.kind == "concentrated_force":
            load = loads.Concentrate_Force(definition.target, values[:3])
        elif definition.kind == "moment":
            load = loads.Moment(definition.target, values[:3])
        elif definition.kind == "spring_rp_rp":
            load = loads.Spring_RP_RP(
                definition.target, definition.target2, values[0],
                None if values[1] < 0 else values[1])
        elif definition.kind == "spring_rp_point":
            load = loads.Spring_RP_Point(
                definition.target, values[:3], values[3],
                None if values[4] < 0 else values[4])
        elif definition.kind == "penalty_dof":
            load = loads.Penalty_DoF(
                definition.target, int(round(values[0])), values[1], values[2],
                definition.target2)
        elif definition.kind == "contact_self":
            load = loads.ContactSelf(definition.instance, definition.target, *values)
        elif definition.kind == "contact":
            load = loads.Contact(
                definition.instance, definition.instance2, definition.target,
                definition.target2, *values)
        else:
            raise ValueError(f"Unknown load type: {definition.kind}")
        load.enabled = bool(definition.enabled)
        assembly.add_load(load, definition.name)

    controller = FEAController()
    controller.assembly = assembly
    controller.cad = cad_model
    maximum_iteration = int(round(float(project.solver_maximum_iteration)))
    tolerance = float(project.solver_tolerance)
    controller.solver = StaticImplicitSolver(maximum_iteration, tolerance)
    return controller


def _validate_project_references(project: CAEProject) -> None:
    """Reject references that the native Assembly API would otherwise defer.

    Validate the complete project graph before it reaches Assembly.
    """
    if not project.parts:
        raise ValueError("CAE project needs at least one Part.")
    if not project.instances:
        raise ValueError("CAE project needs at least one Instance.")

    def unique_names(items, kind: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in items:
            name = str(item.name).strip()
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
    unique_names(project.constraints, "Constraint")
    unique_names(project.loads, "Load")
    for part in project.parts:
        geometry_source = part.geometry.source.strip().casefold()
        if geometry_source not in {"cad", "discrete"}:
            raise ValueError(
                f"Part '{part.name}' has unknown geometry source '{part.geometry.source}'.")
        if geometry_source == "discrete" and not (
                part.geometry.discrete_data is not None
                or part.geometry.runtime_part is not None):
            raise ValueError(f"Discrete Part '{part.name}' has no native mesh.")
        if not np.isfinite(float(part.density)) or float(part.density) <= 0:
            raise ValueError(f"Material density on Part '{part.name}' must be positive and finite.")
        if not part.materials:
            raise ValueError(f"Part '{part.name}' needs at least one material.")
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
        if boundary.kind == "reference_point":
            if boundary.reference_point not in reference_points:
                raise ValueError(
                    f"Boundary '{boundary.name}' references missing reference point "
                    f"'{boundary.reference_point}'.")
            allowed_dofs = (0, 1, 2, 3, 4, 5)
        elif boundary.kind == "nodes":
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
            allowed_dofs = (0, 1, 2)
        else:
            raise ValueError(f"Boundary '{boundary.name}' has unknown type '{boundary.kind}'.")
        if not boundary.dofs or any(int(dof) not in allowed_dofs for dof in boundary.dofs):
            raise ValueError(
                f"Boundary '{boundary.name}' DOFs must be a non-empty subset of {allowed_dofs}.")

    for constraint in project.constraints:
        if constraint.kind != "couple":
            raise ValueError(f"Constraint '{constraint.name}' has unknown type '{constraint.kind}'.")
        instance = instances.get(constraint.instance)
        if instance is None:
            raise ValueError(
                f"Constraint '{constraint.name}' references missing Instance '{constraint.instance}'.")
        part = parts[instance.part_name]
        node_set_names = {item.name for item in part.node_sets} | auto_node_set_names(part)
        if constraint.node_set not in node_set_names:
            raise ValueError(
                f"Constraint '{constraint.name}' references missing node set '{constraint.node_set}' "
                f"on Part '{part.name}'.")
        if constraint.reference_point not in reference_points:
            raise ValueError(
                f"Constraint '{constraint.name}' references missing reference point "
                f"'{constraint.reference_point}'.")

    for load in project.loads:
        values = [float(value) for value in load.values]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Load '{load.name}' contains non-finite parameters.")
        required_values = {
            "pressure": 1,
            "body_force": 3,
            "concentrated_force": 3,
            "moment": 3,
            "spring_rp_rp": 2,
            "spring_rp_point": 5,
            "penalty_dof": 3,
            "contact_self": 10,
            "contact": 7,
        }
        if load.kind in required_values and len(values) != required_values[load.kind]:
            raise ValueError(
                f"Load '{load.name}' needs {required_values[load.kind]} values.")

        def require_instance(instance_name: str):
            instance = instances.get(instance_name)
            if instance is None:
                raise ValueError(
                    f"Load '{load.name}' references missing Instance '{instance_name}'.")
            return instance, parts[instance.part_name]

        def require_reference_point(point_name: str):
            if point_name not in reference_points:
                raise ValueError(
                    f"Load '{load.name}' references missing reference point '{point_name}'.")

        if load.kind == "pressure":
            instance, part = require_instance(load.instance)
            surface_names = {str(part.exterior_surface).strip(),
                             *(item.name for item in part.surface_sets)}
            if load.target not in surface_names:
                raise ValueError(
                    f"Pressure load '{load.name}' references missing surface set '{load.target}' "
                    f"on Part '{part.name}'.")
        elif load.kind == "body_force":
            _instance, part = require_instance(load.instance)
            if load.target != "C3D4":
                raise ValueError(
                    f"Body force '{load.name}' references unknown element block '{load.target}' "
                    f"on Part '{part.name}'.")
        elif load.kind in {"concentrated_force", "moment"}:
            require_reference_point(load.target)
        elif load.kind == "spring_rp_rp":
            require_reference_point(load.target)
            require_reference_point(load.target2)
            if values[0] <= 0:
                raise ValueError(f"Spring '{load.name}' needs a positive stiffness.")
        elif load.kind == "spring_rp_point":
            require_reference_point(load.target)
            if values[3] <= 0:
                raise ValueError(f"Spring '{load.name}' needs point coordinates and positive stiffness.")
        elif load.kind == "penalty_dof":
            object_maps = {
                "instance": instances,
                "rp": reference_points,
                "load": {item.name: item for item in project.loads},
                "constraint": {item.name: item for item in project.constraints},
                "boundary": {item.name: item for item in project.boundaries},
            }
            object_type = str(load.target2 or "auto")
            if object_type == "auto":
                if not any(load.target in mapping for mapping in object_maps.values()):
                    raise ValueError(f"Penalty load '{load.name}' references unknown object '{load.target}'.")
            elif object_type not in object_maps or load.target not in object_maps[object_type]:
                raise ValueError(f"Penalty load '{load.name}' references unknown {object_type} '{load.target}'.")
            if int(round(values[0])) < 0 or values[2] <= 0:
                raise ValueError(f"Penalty load '{load.name}' needs s, target and positive stiffness.")
        elif load.kind == "contact_self":
            _instance, part = require_instance(load.instance)
            surface_names = {str(part.exterior_surface).strip(),
                             *(item.name for item in part.surface_sets)}
            if load.target not in surface_names:
                raise ValueError(f"Contact '{load.name}' references missing surface '{load.target}'.")
        elif load.kind == "contact":
            _instance1, part1 = require_instance(load.instance)
            _instance2, part2 = require_instance(load.instance2)
            surfaces1 = {str(part1.exterior_surface).strip(), *(item.name for item in part1.surface_sets)}
            surfaces2 = {str(part2.exterior_surface).strip(), *(item.name for item in part2.surface_sets)}
            if load.target not in surfaces1 or load.target2 not in surfaces2:
                raise ValueError(f"Contact '{load.name}' references a missing surface.")
        else:
            raise ValueError(f"Unknown load type: {load.kind}")
