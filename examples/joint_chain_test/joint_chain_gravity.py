"""Multi-link hinged chain built with the torchfea CAD and Joint APIs.

The links are elastic C3D4 solids with a high Young's modulus, so they behave
like stiff bars while still using the finite-element solid formulation.  Each
link is coupled exactly once, as a whole, to one reference point.  Adjacent
link reference points are connected by a one-degree-of-freedom hinge whose
free rotation is about the global z axis.

The initial chain lies in the x-y plane.  The left end is hinged to a wall,
the right end receives a constant horizontal force, and its y/z translations
are fixed.  Gravity acts in the negative y direction.
"""

from __future__ import annotations

import numpy as np
import torch

import torchfea
from torchfea.model.elements.materials import LinearElastic
from torchfea.utils.maths import rotation_matrix_from_vector

LINK_COUNT = 5
LINK_LENGTH = 0.40
LINK_RADIUS = 0.025
MESH_SIZE = 0.04

YOUNG_MODULUS = 2.0e10
POISSON_RATIO = 0.30
DENSITY = 780.0
GRAVITY = 9.81
HORIZONTAL_FORCE = 40.0
TIP_PENALTY_STIFFNESS = 1.0e12


def _link_points(link_count: int, link_length: float) -> np.ndarray:
    """Return the initial hinge locations for a horizontal chain."""
    return np.array(
        [[index * link_length, 0.0, 0.0] for index in range(link_count + 1)],
        dtype=float,
    )


def _add_cad_link(
    controller: torchfea.FEAController,
    name: str,
    start: np.ndarray,
    length: float,
    radius: float,
    mesh_size: float,
):
    """Create and mesh one cylindrical link through the CAD history API."""
    history = controller.cad.add_part(name)
    cylinder = history.add_cylinder(
        x=0.0,
        y=0.0,
        z=0.0,
        dx=length,
        dy=0.0,
        dz=0.0,
        radius=radius,
    )
    # The local cylinder is already aligned with +x.  Keeping the transform
    # explicit makes it easy to change this example to angled initial links.
    translated = history.translate(
        cylinder.id,
        dx=float(start[0]),
        dy=float(start[1]),
        dz=float(start[2]),
    )
    assert translated.id in history.final_feature_ids()

    part = controller.cad.mesh_part(name, mesh_size=mesh_size)

    # The CAD mesher produces a C3D4 block named C3D4 for each link.
    material = LinearElastic(YOUNG_MODULUS, POISSON_RATIO)
    density = torch.tensor(DENSITY, dtype=torch.get_default_dtype())
    for element in part.elems.values():
        element.set_materials(material)
        element.density = density

    return part


def _rp_current_connection_position(
    assembly,
    rgc: list[torch.Tensor],
    rp_name: str,
    connection_point: np.ndarray,
) -> torch.Tensor:
    """Return a reference point's current position at a connection point."""
    rp = assembly.get_reference_point(rp_name)
    coordinates = rgc[rp._RGC_index]
    node = rp.node.to(device=coordinates.device, dtype=coordinates.dtype)
    point = torch.as_tensor(
        connection_point,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    rotation = rotation_matrix_from_vector(coordinates[3:])
    return node + coordinates[:3] + rotation @ (point - node)


def build_chain_controller() -> tuple[torchfea.FEAController, dict[str, object]]:
    """Build the CAD chain model and return it before solving."""
    controller = torchfea.FEAController()
    assembly = controller.assembly = torchfea.Assembly()
    points = _link_points(LINK_COUNT, LINK_LENGTH)

    link_names = []
    for link_index in range(LINK_COUNT):
        part_name = f"link-part-{link_index}"
        instance_name = f"link-{link_index}"
        part = _add_cad_link(
            controller,
            name=part_name,
            start=points[link_index],
            length=LINK_LENGTH,
            radius=LINK_RADIUS,
            mesh_size=MESH_SIZE,
        )
        assembly.add_part(part, name=part_name)
        assembly.add_instance(
            torchfea.Instance(part_name=part_name),
            name=instance_name,
        )
        link_names.append((part_name, instance_name))

    # One reference point describes one complete rigid link.  It is placed at
    # the link's right endpoint, so the last link RP is also the load point.
    link_rp_names = [f"link-{index}-rp" for index in range(LINK_COUNT)]
    assembly.add_reference_point(
        torchfea.ReferencePoint(points[0].tolist()),
        name="wall-rp",
    )
    for link_index, rp_name in enumerate(link_rp_names):
        assembly.add_reference_point(
            torchfea.ReferencePoint(points[link_index + 1].tolist()),
            name=rp_name,
        )

    for link_index, (_part_name, instance_name) in enumerate(link_names):
        assembly.add_constraint(
            torchfea.constraints.Couple(
                instance_name=instance_name,
                set_nodes_name="all",
                rp_name=link_rp_names[link_index],
            ),
            name=f"link-{link_index}-couple",
        )

        assembly.add_load(
            torchfea.loads.BodyForce(
                instance_name=instance_name,
                element_name="C3D4",
                force_density=[0.0, -DENSITY * GRAVITY, 0.0],
            ),
            name=f"link-{link_index}-gravity",
        )

    hinge_pairs = [("wall-rp", link_rp_names[0])]
    hinge_points = [points[0]]
    for hinge_index in range(1, LINK_COUNT):
        hinge_pairs.append((link_rp_names[hinge_index - 1], link_rp_names[hinge_index]))
        hinge_points.append(points[hinge_index])

    for hinge_index, ((master_rp_name, slave_rp_name), point) in enumerate(
        zip(hinge_pairs, hinge_points)
    ):
        assembly.add_constraint(
            torchfea.constraints.Connector(
                master_rp_name=master_rp_name,
                slave_rp_name=slave_rp_name,
                translation_free_directions=[],
                rotation_free_directions=[[0.0, 0.0, 1.0]],
                connection_point=point.tolist(),
            ),
            name=f"hinge-{hinge_index + 1}",
        )

    # Left end: a planar wall hinge.  Ux, Uy, Uz, Rx and Ry are fixed; Rz is
    # the hinge rotation.  Right end: Uy and Uz are fixed, while Ux remains
    # free so the horizontal force can move the tip.
    assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition_RP(
            rp_name="wall-rp",
            indexDoF=[0, 1, 2, 3, 4],
        ),
        name="wall-hinge-boundary",
    )
    # The tip RP is a dependent block because it is the last Joint's slave.
    # A direct Boundary_Condition_RP on it would only overwrite the
    # reconstructed value after the Joint has run, without constraining the
    # active joint coordinates.  Penalize the same Couple-point coordinates
    # instead, so the residual is transferred backward through the Joint
    # chain and the tip is actually held at its initial y/z position.
    for direction in (1, 2):
        assembly.add_load(
            torchfea.loads.Penalty_DoF(
                obj_name=link_rp_names[-1],
                obj_type="rp",
                s=direction,
                target=0.0,
                k=TIP_PENALTY_STIFFNESS,
            ),
            name=f"tip-fixed-{direction}",
        )
    assembly.add_load(
        torchfea.loads.Concentrate_Force(
            rp_name=link_rp_names[-1],
            force=[HORIZONTAL_FORCE, 0.0, 0.0],
        ),
        name="tip-horizontal-force",
    )

    controller.solver = torchfea.solver.StaticImplicitSolver(
        maximum_iteration=300,
        tol_error=1.0e-7,
    )
    return controller, {
        "points": points,
        "link_rp_names": link_rp_names,
        "hinge_pairs": hinge_pairs,
        "link_names": link_names,
    }


def solve_and_report(show_plot: bool = False):
    """Solve the chain and print basic kinematic sanity checks."""
    controller, metadata = build_chain_controller()
    result = controller.solve()
    assembly = controller.assembly
    rgc = assembly._GC2RGC(result.GC)

    hinge_gaps = []
    for hinge_index, ((master_name, slave_name), point) in enumerate(
        zip(metadata["hinge_pairs"], metadata["points"]),
        start=1,
    ):
        master_position = _rp_current_connection_position(
            assembly, rgc, master_name, point
        )
        slave_position = _rp_current_connection_position(
            assembly, rgc, slave_name, point
        )
        hinge_gaps.append(
            float(torch.linalg.vector_norm(master_position - slave_position))
        )

    out_of_plane = []
    for _part_name, instance_name in metadata["link_names"]:
        instance = assembly.get_instance(instance_name)
        out_of_plane.append(float(rgc[instance._RGC_index][:, 2].abs().max()))

    tip = assembly.get_reference_point(metadata["link_rp_names"][-1])
    tip_coordinates = rgc[tip._RGC_index]
    residual = assembly.assemble_force(GC=result.GC)

    print(f"converged: {result.converged}")
    print(f"tip RGC [Ux, Uy, Uz, Rx, Ry, Rz]: {tip_coordinates.tolist()}")
    print(f"max hinge connection gap: {max(hinge_gaps, default=0.0):.3e}")
    print(f"max out-of-plane node displacement: {max(out_of_plane, default=0.0):.3e}")
    print(f"max reduced residual: {float(residual.abs().max()):.3e}")

    if show_plot:
        assembly.show_all(GC=result.GC)
    return controller, result


if __name__ == "__main__":
    solve_and_report(show_plot=True)
