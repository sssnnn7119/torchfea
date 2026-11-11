"""Conversions between native controllers and the typed INP model.

This module maps model objects without reading or writing files. Static-job
conversion uses a flat assembly with globally unique labels and includes
instance transforms in the undeformed coordinates. The reverse conversion
supports the solid/material subset documented by inp_to_controller.
"""

from __future__ import annotations

from collections import defaultdict
from textwrap import wrap
from typing import TYPE_CHECKING
import warnings

import numpy as np
import torch

from .. import inp as inp_types
from ..inp import FEA_INP
from ..model import (
    Assembly,
    Instance,
    Part,
    boundarys,
    constraints,
    elements,
    loads,
    materials,
)
from ..model.elements.dimension3 import C3D4, C3D6, C3D8, C3D10, C3D15, C3D20
from ..solver.static import StaticImplicitSolver
from ..utils import Rotation

if TYPE_CHECKING:
    from ..controller import FEAController
    from ..model.elements.base import BaseElement


_SOLIDS = {C3D4, C3D6, C3D8, C3D10, C3D15, C3D20}


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError("Abaqus export requires finite numeric values.")
    return result


def _scalar(value, description: str) -> float:
    array = _array(value)
    if array.size != 1:
        raise NotImplementedError(f"{description}: a uniform scalar is required.")
    return float(array.reshape(-1)[0])


def _numbers(values) -> str:
    return ", ".join(format(float(value), ".16g") for value in values)


class _ControllerToInp:
    def __init__(
        self, controller: FEAController, nlgeom: bool, native_interactions: bool
    ) -> None:
        self.assembly = controller.assembly
        self.nlgeom = nlgeom
        self.native_interactions = native_interactions
        self.model = FEA_INP()
        self.next_node = 1
        self.next_element = 1
        self.name_count = 0
        self.node_ids: dict[str, np.ndarray] = {}
        self.element_ids: dict[str, dict[int, int]] = {}
        self.block_sets: dict[tuple[str, str], str] = {}
        self.node_sets: dict[tuple[str, str], str] = {}
        self.surfaces: dict[tuple[str, str], str] = {}
        self.rp_ids: dict[str, int] = {}
        self.fixed: dict[tuple[int, int], float] = {}
        self.cload: dict[tuple[int, int], float] = defaultdict(float)
        self.dload: dict[tuple[int, str], float] = defaultdict(float)
        self.dsload: dict[str, float] = defaultdict(float)
        self.material_records: list[inp_types.InpRecord] = []
        self.notices: set[str] = set()
        self.coupled_nodes: set[int] = set()
        self.contact_pairs: set[tuple[str, str]] = set()

    def write(self, *lines: str) -> None:
        for line in lines:
            self.model.append_line(line + "\n")

    def name(self, prefix: str) -> str:
        self.name_count += 1
        return f"TF_{prefix}_{self.name_count}"

    def comment(self, text: str) -> None:
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.encode("ascii", errors="backslashreplace").decode("ascii")
        self.write(*("** " + line for line in wrap(text, width=240)))

    def set(self, kind: str, ids: np.ndarray, name: str | None = None) -> str:
        ids = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
        if not ids.size:
            raise ValueError(f"Cannot export an empty {kind}.")
        name = self.name(kind) if name is None else name
        members = list(map(str, ids))
        if kind.lower() == "nset":
            self.model.records.append(inp_types.NodeSet(name, members))
        elif kind.lower() == "elset":
            self.model.records.append(inp_types.ElementSet(name, members))
        else:
            raise ValueError(f"Unsupported set type '{kind}'.")
        return name

    def add_node(self, point) -> int:
        point = _array(point).reshape(-1)
        if point.size != 3:
            raise ValueError("Node coordinates must have three components.")
        label = self.next_node
        self.next_node += 1
        self.model.records.append(inp_types.Nodes(np.array([label]), point.reshape(1, 3)))
        return label

    def part(self, instance_name: str) -> Part:
        instance = self.assembly._instances[instance_name]
        if instance.enabled is False:
            raise ValueError(f"Object references disabled instance '{instance_name}'.")
        return self.assembly._parts[instance.part_name]

    def nodes_in_set(self, instance_name: str, set_name: str) -> np.ndarray:
        part = self.part(instance_name)
        indices = _array(part.set_nodes[set_name])
        if indices.size == 0:
            raise ValueError(f"Empty node set '{instance_name}.{set_name}'.")
        indices = indices.astype(np.int64).reshape(-1)
        nodes = self.node_ids[instance_name]
        if np.any(indices < 0) or np.any(indices >= nodes.size):
            raise ValueError(f"Invalid node set '{instance_name}.{set_name}'.")
        return nodes[indices]

    def _field(self, value, block: BaseElement, description: str) -> np.ndarray:
        """Allow homogeneous or element-wise fields, reject Gauss variation."""
        array = _array(value)
        count = len(block._elems)
        gauss = block._num_gaussian
        if array.size == 1:
            return np.full(count, float(array.reshape(-1)[0]))
        if array.ndim == 1 and array.size == gauss:
            # Match the material API, which gives Gaussian axes precedence.
            if not np.all(array == array[0]):
                raise NotImplementedError(f"{description}: Gauss-point variation.")
            return np.full(count, float(array[0]))
        if array.shape in {(count,), (1, count)}:
            return array.reshape(count)
        if array.shape == (gauss, count) and np.all(array == array[:1]):
            return array[0]
        raise NotImplementedError(f"{description}: unsupported parameter field {array.shape}.")

    def _material(self, material, block: BaseElement):
        """Return keyword and per-element Abaqus constants, without density."""
        if type(material) is materials.LinearElastic:
            if self.nlgeom:
                self.notices.add(
                    "LinearElastic is exported as Abaqus *ELASTIC (same E and nu); "
                    "its finite-strain law differs from torchfea's St Venant-Kirchhoff law."
                )
            fields = [self._field(material.E, block, "E"),
                      self._field(material.nu, block, "nu")]
            if np.any(fields[0] <= 0) or np.any(fields[1] <= -1) or np.any(fields[1] >= .5):
                raise ValueError("Elastic material requires E > 0 and -1 < nu < 0.5.")
            return "*Elastic", np.column_stack(fields)
        if type(material) is materials.NeoHookean:
            mu = self._field(material.mu, block, "mu")
            bulk = self._field(material.kappa, block, "kappa")
            fields = [mu / 2, self._compressibility(bulk)]
            return "*Hyperelastic, neo hooke", np.column_stack(fields)
        if type(material) is materials.MooneyRivlin:
            fields = [self._field(material.c10, block, "c10"),
                      self._field(material.c01, block, "c01"),
                      self._compressibility(self._field(material.kappa, block, "kappa"))]
            return "*Hyperelastic, mooney-rivlin", np.column_stack(fields)
        if type(material) is materials.Yeoh:
            fields = [self._field(material.c1, block, "c1"),
                      self._field(material.c2, block, "c2"),
                      self._field(material.c3, block, "c3"),
                      self._compressibility(self._field(material.kappa, block, "kappa")),
                      np.zeros(len(block._elems)), np.zeros(len(block._elems))]
            return "*Hyperelastic, yeoh", np.column_stack(fields)
        if type(material) is materials.Ogden:
            mu = _array(material.mu).reshape(-1)
            alpha = _array(material.alpha).reshape(-1)
            if mu.size != alpha.size or not 1 <= mu.size <= 6:
                raise NotImplementedError("Abaqus supports one to six Ogden terms.")
            bulk = self._field(material.kappa, block, "kappa")
            columns = [np.full(len(block._elems), x)
                       for pair in zip(mu, alpha) for x in pair]
            columns += [self._compressibility(bulk)]
            columns += [np.zeros(len(block._elems)) for _ in range(mu.size - 1)]
            return f"*Hyperelastic, ogden, n={mu.size}", np.column_stack(columns)
        raise NotImplementedError(
            f"Material '{type(material).__name__}' has no supported Abaqus mapping."
        )

    @staticmethod
    def _compressibility(bulk: np.ndarray) -> np.ndarray:
        if np.any(bulk <= 0):
            raise ValueError("Hyperelastic bulk modulus must be positive.")
        return 2.0 / bulk

    def sections(self, block: BaseElement, ids: np.ndarray) -> None:
        count = ids.size
        owners = np.full(count, -1, dtype=int)
        assignments = block.material_element_indices
        if not assignments and len(block.materials) != 1:
            raise NotImplementedError(
                "Abaqus export requires one material per element; summed materials "
                "cannot be exported as separate solid sections."
            )
        density = (np.zeros(count) if block.density is None else
                   self._density(block.density, count))
        for index, (mat_name, mat) in enumerate(block.materials.items()):
            local = (np.asarray(assignments[mat_name], dtype=int).reshape(-1)
                     if assignments else np.arange(count))
            if not local.size:
                continue
            if np.any(local < 0) or np.any(local >= count) or np.any(owners[local] != -1):
                raise ValueError("Material assignments overlap or contain invalid indices.")
            owners[local] = index
            _keyword, constants = self._material(mat, block)
            rows = np.column_stack((constants[local], density[local]))
            unique, inverse = np.unique(rows, axis=0, return_inverse=True)
            for region, row in enumerate(unique):
                material_name = self.name("MAT")
                region_name = self.set("Elset", ids[local[inverse == region]])
                self.comment(f"Material {mat_name}: {type(mat).__name__}")
                self.model.records.append(inp_types.SolidSection(region_name, material_name))
                self.material_records.append(inp_types.MaterialDefinition(material_name))
                if type(mat) is materials.LinearElastic:
                    self.material_records.append(inp_types.Elastic(row[0], row[1]))
                else:
                    if type(mat) is materials.NeoHookean:
                        model, order = "NEO HOOKE", 1
                    elif type(mat) is materials.MooneyRivlin:
                        model, order = "MOONEY-RIVLIN", 1
                    elif type(mat) is materials.Yeoh:
                        model, order = "YEOH", 1
                    else:
                        model, order = "OGDEN", int(mat.mu.numel())
                    self.material_records.append(inp_types.Hyperelastic(model, order, row[:-1].tolist()))
                if row[-1] > 0:
                    self.material_records.append(inp_types.Density(row[-1]))
        if np.any(owners == -1):
            raise ValueError("Every exported solid element needs a material assignment.")

    @staticmethod
    def _density(value, count: int) -> np.ndarray:
        array = _array(value).reshape(-1)
        if array.size == 1:
            array = np.full(count, array[0])
        if array.size != count or np.any(array < 0):
            raise ValueError("Density must be non-negative and scalar or element-wise.")
        return array

    def mesh(self) -> None:
        for instance_name, instance in sorted(self.assembly._instances.items()):
            if instance.enabled is False:
                continue
            part = self.part(instance_name)
            self.comment(f"Instance {instance_name}; part {instance.part_name}")
            nodes = _array(part.nodes)
            if nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes):
                raise ValueError(f"Invalid nodes in part '{instance.part_name}'.")
            matrix = Rotation.from_rotvec(instance._rotation).as_matrix()
            points = nodes @ _array(matrix).T + _array(instance._translation)
            labels = np.arange(self.next_node, self.next_node + len(nodes))
            self.next_node += len(nodes)
            self.node_ids[instance_name] = labels
            self.model.records.append(inp_types.Nodes(labels, points))
            self.set("Nset", labels)
            mapping: dict[int, int] = {}
            side_counts: dict[int, int] = {}
            for block_name, block in sorted(part.elems.items()):
                if type(block) not in _SOLIDS:
                    raise NotImplementedError(f"Unsupported element '{type(block).__name__}'.")
                connectivity = _array(block._elems).astype(np.int64)
                original_ids = _array(block._elems_index).astype(np.int64).reshape(-1)
                if connectivity.shape != (original_ids.size, block.num_nodes_per_elem):
                    raise ValueError(f"Invalid connectivity in '{block_name}'.")
                if np.any(connectivity < 0) or np.any(connectivity >= len(nodes)):
                    raise ValueError(f"Invalid node index in '{block_name}'.")
                if np.any(original_ids < 0) or len(np.unique(original_ids)) != len(original_ids):
                    raise ValueError(f"Invalid or duplicate element IDs in '{block_name}'.")
                if set(map(int, original_ids)) & mapping.keys():
                    raise ValueError("Element IDs must be unique within a part.")
                ids = np.arange(self.next_element, self.next_element + len(connectivity))
                self.next_element += len(connectivity)
                mapping.update(zip(map(int, original_ids), map(int, ids)))
                side_counts.update((int(e), block.num_surfaces) for e in original_ids)
                if not len(ids):
                    continue
                elset = self.name("BLOCK")
                self.block_sets[instance_name, block_name] = elset
                self.model.records.append(inp_types.Elements(
                    type(block).__name__, ids, labels[connectivity], elset))
                self.sections(block, ids)
            if not mapping:
                raise ValueError(f"Instance '{instance_name}' contains no solid elements.")
            self.element_ids[instance_name] = mapping
            for set_name in sorted(part.set_nodes):
                if _array(part.set_nodes[set_name]).size:
                    self.comment(f"Node set {instance_name}.{set_name}")
                    self.node_sets[instance_name, set_name] = self.set(
                        "Nset", self.nodes_in_set(instance_name, set_name))
            for set_name, element_indices in sorted(part.set_elements.items()):
                if _array(element_indices).size:
                    self.comment(f"Element set {instance_name}.{set_name}")
                    self.set("Elset", [mapping[int(i)] for i in _array(element_indices).reshape(-1)])
            for surface_name in sorted(part.surfaces.keys()):
                items = part.surfaces[surface_name]
                if not items:
                    continue
                entries = set()
                for element_indices, side in items:
                    for eid in _array(element_indices).reshape(-1):
                        eid = int(eid)
                        if eid not in mapping or not 0 <= int(side) < side_counts[eid]:
                            raise ValueError(f"Invalid surface '{instance_name}.{surface_name}'.")
                        entries.add((mapping[eid], int(side) + 1))
                if entries:
                    name = self.name("SURF")
                    self.surfaces[instance_name, surface_name] = name
                    self.comment(f"Surface {instance_name}.{surface_name}")
                    self.model.records.append(inp_types.Surface(
                        name, "ELEMENT", [inp_types.SurfaceEntry(str(eid), f"S{side}")
                                          for eid, side in sorted(entries)]))
        if not self.node_ids:
            raise ValueError("No enabled solid instances to export.")
        for name, rp in sorted(self.assembly._reference_points.items()):
            self.comment(f"Reference point {name}")
            self.rp_ids[name] = self.add_node(rp.node)
            self.set("Nset", [self.rp_ids[name]])

    def couple(self, node_ids: np.ndarray, rp_id: int) -> None:
        overlap = self.coupled_nodes.intersection(map(int, node_ids))
        if overlap:
            raise ValueError("Abaqus kinematic couplings cannot share dependent nodes.")
        self.coupled_nodes.update(map(int, node_ids))
        nset = self.set("Nset", node_ids)
        surface = self.name("COUPLING_SURF")
        self.model.records.append(inp_types.Surface(
            surface, "NODE", [inp_types.SurfaceEntry(nset, "1.")]))
        name = self.name("COUPLING")
        self.model.records.extend([inp_types.Coupling(name, str(rp_id), surface),
                                   inp_types.Kinematic(1, 3)])

    def constraints(self) -> None:
        for name, constraint in sorted(self.assembly._constraints.items()):
            if constraint.enabled is False:
                continue
            self.comment(f"Constraint {name}")
            if type(constraint) is not constraints.Couple:
                raise NotImplementedError(f"Unsupported constraint '{type(constraint).__name__}'.")
            self.couple(self.nodes_in_set(constraint.instance_name, constraint.set_nodes_name),
                        self.rp_ids[constraint.rp_name])

    def boundary(self) -> None:
        for name, boundary in sorted(self.assembly._boundarys.items()):
            if boundary.enabled is False:
                continue
            if type(boundary) is boundarys.Boundary_Condition_RP:
                nodes = [self.rp_ids[boundary.rp_name]]
                upper = 6
            elif type(boundary) is boundarys.Boundary_Condition:
                nodes = self.nodes_in_set(boundary.instance_name, boundary.set_nodes_name)
                if self.coupled_nodes.intersection(map(int, nodes)):
                    raise ValueError(
                        f"Boundary '{name}' acts on kinematically coupled nodes; "
                        "apply it to the reference point for Abaqus export."
                    )
                upper = 3
            else:
                raise NotImplementedError(f"Unsupported boundary '{type(boundary).__name__}'.")
            for dof in boundary.indexDoF:
                if not 0 <= dof < upper:
                    raise ValueError(f"Invalid boundary DoF in '{name}'.")
                for node in nodes:
                    self.fixed[int(node), dof + 1] = 0.0

    def spring(self, nodes: list[int], k: float, dof: int | None = None) -> None:
        if k <= 0:
            raise ValueError("Spring/penalty stiffness must be positive.")
        eid = self.next_element
        self.next_element += 1
        name = self.name("SPRING")
        spring_type = "SPRING1" if dof is not None else "SPRINGA"
        self.model.records.extend([
            inp_types.Elements(spring_type, np.array([eid]), np.array([nodes]), name),
            inp_types.Spring(name, k, dof),
        ])

    def penalty(self, load: loads.Penalty_DoF) -> None:
        obj = self.assembly.get_object(load.obj_name, load.obj_type)
        if load.obj_name in self.assembly._reference_points and obj is self.assembly._reference_points[load.obj_name]:
            if not 0 <= load.s < 6:
                raise ValueError("RP penalty DoF must be in [0, 5].")
            node, dof = self.rp_ids[load.obj_name], load.s + 1
        elif load.obj_name in self.assembly._instances and obj is self.assembly._instances[load.obj_name]:
            labels = self.node_ids[load.obj_name]
            if not 0 <= load.s < labels.size * 3:
                raise ValueError("Instance penalty DoF is out of range.")
            node, dof = int(labels[load.s // 3]), load.s % 3 + 1
        else:
            raise NotImplementedError("Penalty_DoF export supports instances and reference points.")
        if node in self.coupled_nodes:
            raise ValueError("Penalty_DoF on coupled nodes must be applied at the RP for export.")
        k = _scalar(load.k, "penalty stiffness")
        self.spring([node], k, dof)
        self.cload[node, dof] += k * _scalar(load.target, "penalty target")
        if dof > 3 and self.nlgeom:
            self.notices.add("Abaqus rotational springs differ from a quadratic rotation-vector penalty at finite rotations.")

    def axial_spring(self, load) -> None:
        a = self.rp_ids[load.rp_name1] if type(load) is loads.Spring_RP_RP else self.rp_ids[load.rp_name]
        if type(load) is loads.Spring_RP_RP:
            b = self.rp_ids[load.rp_name2]
            x1 = _array(self.assembly._reference_points[load.rp_name1].node)
            x2 = _array(self.assembly._reference_points[load.rp_name2].node)
        else:
            x1 = _array(self.assembly._reference_points[load.rp_name].node)
            x2 = _array(load.point)
            b = self.add_node(x2)
            for dof in range(1, 4):
                self.fixed[b, dof] = 0.
        length = float(np.linalg.norm(x2 - x1))
        rest = length if load.rest_length is None else _scalar(load.rest_length, "rest length")
        if rest < 0:
            rest = length
        if length <= 0 or not np.isclose(rest, length, rtol=1e-10, atol=1e-12):
            raise NotImplementedError("SPRINGA export requires a nonzero, initially stress-free spring.")
        self.spring([a, b], _scalar(load.k, "spring stiffness"))

    def joint(self, load) -> None:
        """Map allowed motions to ideal Abaqus connectors after explicit opt-in."""
        connection = "JOIN"
        orientation = None
        if type(load) is loads.SingleDoFJoint:
            direction = _array(load.free_direction)
            translation = np.linalg.norm(direction[:3])
            rotation = np.linalg.norm(direction[3:])
            if translation > 0 and rotation > 0:
                raise NotImplementedError(
                    "Mixed translation/rotation SingleDoFJoint directions need "
                    "a screw connector; no native mapping is implemented."
                )
            connection = "HINGE" if rotation > 0 else "TRANSLATOR"
            axis = direction[3:] if rotation > 0 else direction[:3]
            axis = axis / np.linalg.norm(axis)
            second = np.eye(3)[np.argmin(np.abs(axis))]
            second -= np.dot(second, axis) * axis
            second /= np.linalg.norm(second)
            orientation = self.name("ORIENT")
            self.model.records.append(inp_types.Orientation(
                orientation, tuple(axis), tuple(second)))
        point = _array(load.connection_point)
        nodes = []
        for rp_name in (load.rp_name1, load.rp_name2):
            rp = self.assembly._reference_points[rp_name]
            node = self.rp_ids[rp_name]
            if not np.array_equal(_array(rp.node), point):
                anchor = self.add_node(point)
                # First BEAM MPC node is dependent; RP remains the master.
                self.model.records.append(inp_types.MultiPointConstraint([
                    inp_types.MpcValue("BEAM", [str(anchor), str(node)])]))
                node = anchor
            nodes.append(node)
        eid = self.next_element
        self.next_element += 1
        elset = self.name("JOINT")
        self.model.records.extend([
            inp_types.Elements("CONN3D2", np.array([eid]), np.array([nodes]), elset),
            inp_types.ConnectorSection(elset, connection, orientation, orientation),
        ])
        self.notices.add(
            "native_interactions=True: penalty joints are exported as ideal "
            "JOIN/HINGE/TRANSLATOR connectors; finite penalty stiffness is not retained."
        )

    def contact(self, load) -> None:
        if type(load) is loads.ContactSelf:
            first = self.surfaces[load.instance_name, load.surface_name]
            second = first
        else:
            first = self.surfaces[load.instance_name1, load.surface_name1]
            second = self.surfaces[load.instance_name2, load.surface_name2]
        self.contact_pairs.add(tuple(sorted((first, second))))
        self.notices.add(
            "Contact and ContactSelf are exported as frictionless hard general "
            "contact. Penalty parameters and self-contact "
            "initial exclusion filters are not retained."
        )

    def loads(self) -> None:
        for name, load in sorted(self.assembly._loads.items()):
            if load.enabled is False:
                continue
            self.comment(f"Load {name}: {type(load).__name__}")
            if type(load) in (loads.Concentrate_Force, loads.Moment):
                node = self.rp_ids[load.rp_name]
                offset = 1 if type(load) is loads.Concentrate_Force else 4
                values = _array(load.force if offset == 1 else load.moment).reshape(-1)
                if values.size != 3:
                    raise ValueError("Force and moment loads require three components.")
                for dof, value in enumerate(values, offset):
                    self.cload[node, dof] += float(value)
            elif type(load) is loads.Pressure:
                self.dsload[self.surfaces[load.instance_name, load.surface_set]] += _scalar(load.pressure, "pressure")
            elif type(load) is loads.BodyForce:
                part = self.part(load._instance_name)
                block = part.elems[load._element_name]
                mapping = self.element_ids[load._instance_name]
                values = _array(load.force_density).reshape(-1)
                if values.size != 3:
                    raise ValueError("Body force requires three components.")
                for eid in _array(block._elems_index).reshape(-1):
                    for kind, value in zip(("BX", "BY", "BZ"), values):
                        self.dload[mapping[int(eid)], kind] += float(value)
            elif type(load) is loads.Penalty_DoF:
                self.penalty(load)
            elif type(load) in (loads.Spring_RP_RP, loads.Spring_RP_Point):
                self.axial_spring(load)
            elif self.native_interactions and type(load) in (loads.SingleDoFJoint, loads.SphericalJoint):
                self.joint(load)
            elif type(load) in (loads.Contact, loads.ContactSelf):
                self.contact(load)
            else:
                raise NotImplementedError(
                    f"Load '{name}' ({type(load).__name__}) has no supported Abaqus mapping. "
                    "The export was cancelled; the load was not silently omitted."
                )

    def step(self, initial: float, minimum: float, maximum: float, increments: int) -> None:
        self.set("Elset", [eid for mapping in self.element_ids.values()
                           for eid in mapping.values()], name="TF_SOLIDS")
        self.model.records.extend(self.material_records)
        for notice in sorted(self.notices):
            self.comment(notice)
        if self.contact_pairs:
            interaction = self.name("CONTACT_PROPERTY")
            self.model.records.extend([
                inp_types.SurfaceInteraction(interaction),
                inp_types.SurfaceBehavior(),
                inp_types.GeneralContact(),
                inp_types.ContactInclusions([
                    inp_types.ContactPair(first, second) for first, second in sorted(self.contact_pairs)]),
                inp_types.ContactPropertyAssignment([
                    inp_types.ContactProperty(first, second, interaction)
                    for first, second in sorted(self.contact_pairs)]),
            ])
        self.model.records.extend([
            inp_types.StepDefinition("Static", self.nlgeom, increments),
            inp_types.Static(initial, 1., minimum, maximum),
        ])
        if self.fixed:
            self.model.records.append(inp_types.Boundary([
                inp_types.BoundaryValue(str(node), dof, dof, value)
                for (node, dof), value in sorted(self.fixed.items())]))
        concentrated = [(key, value) for key, value in sorted(self.cload.items()) if value != 0]
        if concentrated:
            self.model.records.append(inp_types.ConcentratedLoad([
                inp_types.NodalForce(str(node), dof, value) for (node, dof), value in concentrated]))
        distributed = [(key, value) for key, value in sorted(self.dload.items()) if value != 0]
        if distributed:
            self.model.records.append(inp_types.DistributedLoad([
                inp_types.DistributedForce(str(eid), kind, value) for (eid, kind), value in distributed]))
        pressure = [(key, value) for key, value in sorted(self.dsload.items()) if value != 0]
        if pressure:
            self.model.records.append(inp_types.SurfaceLoad([
                inp_types.DistributedForce(surface, "P", value) for surface, value in pressure]))
        self.model.records.extend([
            inp_types.Output(),
            inp_types.NodeOutput(),
            inp_types.ElementOutput(element_set="TF_SOLIDS"),
            inp_types.RawRecord("END STEP", "*End Step\n"),
        ])


def controller_to_inp(
    controller: FEAController,
    *,
    nlgeom: bool = True,
    initial_increment: float = 0.1,
    minimum_increment: float = 1e-6,
    maximum_increment: float = 0.1,
    maximum_increments: int = 1000,
    native_interactions: bool = False,
) -> FEA_INP:
    """Convert a controller to the common INP keyword model.

    Exports the undeformed model and final load parameters, independent of
    initialization, solved coordinates, and temporary continuation factors.
    Contact and ContactSelf always map to frictionless hard general contact.
    native_interactions explicitly enables ideal connector mappings for joints.
    LinearElastic maps to Abaqus *ELASTIC, whose finite-strain law differs.
    """
    if controller.assembly is None:
        raise ValueError("The controller has no assembly.")
    if controller.solver is not None and type(controller.solver) is not StaticImplicitSolver:
        raise NotImplementedError("INP job export currently supports static implicit analyses.")
    if not 0 < minimum_increment <= initial_increment <= maximum_increment <= 1:
        raise ValueError("Require 0 < minimum <= initial <= maximum increment <= 1.")
    if not isinstance(maximum_increments, int) or maximum_increments < 1:
        raise ValueError("maximum_increments must be a positive integer.")
    deck = _ControllerToInp(controller, nlgeom, native_interactions)
    deck.write("*Heading", "torchfea static analysis",
               "** Units are inherited from the model; no unit conversion is performed.",
               "** Flat assembly: instance transforms are included in nodal coordinates.",
               "*Preprint, echo=NO, model=NO, history=NO, contact=NO")
    deck.mesh()
    deck.constraints()
    deck.boundary()
    deck.loads()
    deck.step(initial_increment, minimum_increment, maximum_increment, maximum_increments)
    for notice in sorted(deck.notices):
        warnings.warn(notice, UserWarning, stacklevel=2)
    deck.model.refresh_views()
    return deck.model


def inp_to_controller(inp: FEA_INP, create_instance: bool = True) -> FEAController:
    """Import the supported solid/material subset of a typed INP model.

    INP round trips retain all records. Solver import intentionally supports a
    narrower subset and reports missing materials or unsupported definitions.
    """
    from ..controller import FEAController

    assembly_now = Assembly()
    if not inp.part:
        raise NotImplementedError("The solver importer needs part-based solid meshes.")
    for source in inp.part:
        nodes = source.nodes
        if not len(nodes):
            raise ValueError(f"Part '{source.name}' has no numeric nodes.")
        labels = nodes[:, 0].astype(np.int64)
        order = np.argsort(labels)
        sorted_labels = labels[order]

        def node_rows(original_labels: np.ndarray) -> np.ndarray:
            positions = np.searchsorted(sorted_labels, original_labels)
            if np.any(positions >= len(labels)) or np.any(
                sorted_labels[positions] != original_labels
            ):
                raise ValueError(f"Unknown node label in part '{source.name}'.")
            return order[positions]

        part_now = Part(
            torch.tensor(
                nodes[:, 1:],
                device=torch.get_default_device(),
                dtype=torch.get_default_dtype(),
            )
        )
        for node_set in source.node_sets:
            part_now.set_nodes[node_set.name] = node_rows(
                source.node_set_labels(node_set.name) - 1
            )
        for element_set in source.element_sets:
            part_now.set_elements[element_set.name] = (
                source.element_set_labels(element_set.name) - 1
            )
        assembly_now.add_part(part_now, name=source.name)
        if create_instance:
            assembly_now.add_instance(Instance(source.name), name=source.name)
        # Repeated blocks of the same type are merged, retaining original labels.
        for element_type in sorted({block.element_type for block in source.elements}):
            blocks = [
                block for block in source.elements if block.element_type == element_type
            ]
            element_labels = np.concatenate([block.labels for block in blocks])
            connectivity = np.concatenate([block.connectivity for block in blocks])
            element = elements.initialize_element(
                element_type=element_type,
                elems_index=torch.tensor(element_labels - 1, dtype=torch.int64),
                elems=torch.tensor(node_rows(connectivity - 1), dtype=torch.int64),
                part=part_now,
            )
            assigned = np.zeros(len(element_labels), dtype=bool)
            density = np.zeros(len(element_labels))
            for region, section in enumerate(source.sections):
                indices = np.flatnonzero(
                    np.isin(
                        element_labels, source.element_set_labels(section.element_set)
                    )
                )
                if not len(indices):
                    continue
                if np.any(assigned[indices]):
                    raise ValueError(f"Overlapping sections in part '{source.name}'.")
                material = next(
                    mat for mat in inp.material if mat.name == section.material
                )
                if material.elastic is not None:
                    mat = materials.LinearElastic(
                        material.elastic.young_modulus, material.elastic.poisson_ratio
                    )
                elif (
                    material.hyperelastic is not None
                    and material.hyperelastic.model == "NEO HOOKE"
                ):
                    c10, d1 = material.hyperelastic.constants[:2]
                    if d1 <= 0:
                        raise NotImplementedError(
                            "Solver import needs a compressible Neo-Hookean material."
                        )
                    mat = materials.NeoHookean(2 * c10, 2 / d1)
                else:
                    raise NotImplementedError(
                        f"Unsupported solver material '{material.name}'."
                    )
                element.set_materials(
                    mat, name=f"material-{region}", element_indices=indices
                )
                if material.density is not None:
                    density[indices] = material.density.value
                assigned[indices] = True
            if not np.all(assigned):
                raise ValueError(f"Unassigned solid section in part '{source.name}'.")
            element.density = density
            part_now.add_element(element, name=element_type)
        for surface in source.surfaces:
            if surface.surface_type != "ELEMENT":
                continue
            faces = []
            for entry in surface.entries:
                if not entry.face.upper().startswith("S"):
                    raise NotImplementedError(
                        "Solver import requires numbered solid faces."
                    )
                ids = (
                    np.array([int(entry.target)])
                    if entry.target.isdigit()
                    else source.element_set_labels(entry.target)
                )
                faces.append((ids - 1, int(entry.face[1:]) - 1))
            part_now.add_surface_set(surface.name, faces)
    controller = FEAController()
    controller.assembly = assembly_now
    return controller
