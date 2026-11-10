"""Multi-link hinged chain built with CAD, Couple and penalty SingleDoFJoint.

Each C3D4 link is coupled as a whole to one reference point, imposing rigid
motion on all its nodes. Adjacent reference points are connected by penalty
hinges that release rotation about the initial global z axis. Their anchor
positions may differ by a small amount governed by the penalty stiffness.

The initial chain lies in the x-y plane.  The left end is hinged to a wall,
the right end receives a constant horizontal force, and its y/z translations
are fixed.  Gravity acts in the negative y direction.

Units are SI: m, N, kg, Pa, and rad. Run from the repository root:

    PYTHONPATH=src python examples/joint_chain_test/joint_chain_gravity.py
    PYTHONPATH=src python examples/joint_chain_test/joint_chain_gravity.py --plot

``--joint-stiffness`` controls the shared translation/rotation penalty value;
``--threads`` controls PyTorch CPU operator threads (default 1).
"""

from __future__ import annotations

import argparse
from typing import TypedDict

import numpy as np
import torch

import torchfea
from torchfea.model.elements.materials import LinearElastic
from torchfea.utils import Rotation

LINK_COUNT = 5
LINK_LENGTH = 0.40
LINK_RADIUS = 0.025
MESH_SIZE = 0.04

YOUNG_MODULUS = 2.0e10
POISSON_RATIO = 0.30
DENSITY = 780.0
GRAVITY = 9.81
HORIZONTAL_FORCE = 40.0
JOINT_PENALTY_STIFFNESS = 1.0e7


class ChainMetadata(TypedDict):
    points: np.ndarray
    link_rp_names: list[str]
    hinge_pairs: list[tuple[str, str]]
    hinge_names: list[str]
    link_names: list[tuple[str, str]]


def _link_points(link_count: int, link_length: float) -> np.ndarray:
    """Return the initial hinge locations for a horizontal chain."""
    points = np.zeros((link_count + 1, 3))
    points[:, 0] = np.arange(link_count + 1) * link_length
    return points


def _add_cad_link(
    controller: torchfea.FEAController,
    name: str,
    start: np.ndarray,
    length: float,
    radius: float,
    mesh_size: float,
) -> torchfea.Part:
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
    assembly: torchfea.Assembly,
    rgc: list[torch.Tensor],
    rp_name: str,
    connection_point: np.ndarray | torch.Tensor,
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
    rotation = Rotation.from_rotvec(coordinates[3:]).as_matrix()
    return node + coordinates[:3] + rotation @ (point - node)


def build_chain_controller(
    joint_stiffness: float = JOINT_PENALTY_STIFFNESS,
) -> tuple[torchfea.FEAController, ChainMetadata]:
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

    hinge_pairs = list(zip(["wall-rp", *link_rp_names[:-1]], link_rp_names))
    hinge_names = [f"hinge-{index + 1}" for index in range(LINK_COUNT)]

    for (rp_name1, rp_name2), point, name in zip(hinge_pairs, points[:-1], hinge_names):
        assembly.add_load(
            torchfea.loads.SingleDoFJoint(
                rp_name1=rp_name1,
                rp_name2=rp_name2,
                connection_point=point.tolist(),
                free_direction=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                k=joint_stiffness,
            ),
            name=name,
        )

    # Fix the wall frame completely: the first joint releases the link's Rz.
    # Leaving the wall's own Rz free would introduce a redundant free DoF.
    assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition_RP(
            rp_name="wall-rp",
            indexDoF=[0, 1, 2, 3, 4, 5],
        ),
        name="wall-hinge-boundary",
    )
    # Penalty joints retain independent RP coordinates, so the tip's Uy/Uz
    # can be fixed directly. Ux remains free under the horizontal force.
    assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition_RP(
            rp_name=link_rp_names[-1],
            indexDoF=[1, 2],
        ),
        name="tip-fixed-yz",
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
        "hinge_names": hinge_names,
        "link_names": link_names,
    }


def solve_and_report(
    show_plot: bool = False,
    joint_stiffness: float = JOINT_PENALTY_STIFFNESS,
) -> tuple[torchfea.FEAController, torchfea.solver.StaticResult]:
    """Solve the chain and print basic kinematic sanity checks."""
    controller, metadata = build_chain_controller(joint_stiffness=joint_stiffness)
    assembly = controller.assembly
    controller.initialize()
    initial_rgc = [torch.zeros_like(segment) for segment in assembly._RGC]
    # A straight mechanism has no transverse tangent before tension develops.
    # Seed the small axial gaps corresponding to the applied tensile force.
    for index, rp_name in enumerate(metadata["link_rp_names"], start=1):
        rp = assembly.get_reference_point(rp_name)
        initial_rgc[rp._RGC_index][0] = index * HORIZONTAL_FORCE / joint_stiffness
    result = controller.solve(GC0=assembly._RGC2GC(initial_rgc), if_initialize=False)
    rgc = assembly._GC2RGC(result.GC)

    hinge_gaps = []
    blocked_rotations = []
    for (master_name, slave_name), hinge_name in zip(
        metadata["hinge_pairs"], metadata["hinge_names"]
    ):
        joint = assembly.get_load(hinge_name)
        point = joint.connection_point
        master_position = _rp_current_connection_position(
            assembly, rgc, master_name, point
        )
        slave_position = _rp_current_connection_position(
            assembly, rgc, slave_name, point
        )
        hinge_gaps.append(
            float(torch.linalg.vector_norm(master_position - slave_position))
        )
        master = assembly.get_reference_point(master_name)
        slave = assembly.get_reference_point(slave_name)
        master_rotation = Rotation.from_rotvec(rgc[master._RGC_index][3:])
        slave_rotation = Rotation.from_rotvec(rgc[slave._RGC_index][3:])
        relative_rotation = (
            master_rotation.inverse().compose(slave_rotation).as_rotvec()
        )
        blocked_rotations.append(float(torch.linalg.vector_norm(relative_rotation[:2])))

    out_of_plane = []
    for _part_name, instance_name in metadata["link_names"]:
        instance = assembly.get_instance(instance_name)
        out_of_plane.append(float(rgc[instance._RGC_index][:, 2].abs().max()))

    tip = assembly.get_reference_point(metadata["link_rp_names"][-1])
    tip_coordinates = rgc[tip._RGC_index]
    residual = assembly.assemble_force(GC=result.GC)

    print(f"converged: {result.converged}")
    print(
        f"active DoFs: {result.GC.numel()}, Newton iterations: {controller.solver._iter_now}"
    )
    print(f"joint penalty stiffness: {joint_stiffness:.3e}")
    print(f"tip RGC [Ux, Uy, Uz, Rx, Ry, Rz]: {tip_coordinates.tolist()}")
    print(f"max hinge connection gap: {max(hinge_gaps, default=0.0):.3e}")
    print(
        f"max constrained relative rotation [rad]: {max(blocked_rotations, default=0.0):.3e}"
    )
    print(f"max out-of-plane node displacement: {max(out_of_plane, default=0.0):.3e}")
    print(f"max reduced residual: {float(residual.abs().max()):.3e}")

    if not result.converged or not torch.isfinite(residual).all():
        raise RuntimeError(
            "The joint chain did not converge; inspect the reported residual."
        )
    if show_plot:
        assembly.show_all(GC=result.GC)
    return controller, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show the deformed FE mesh after solving."
    )
    parser.add_argument(
        "--joint-stiffness", type=float, default=JOINT_PENALTY_STIFFNESS
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="PyTorch CPU operator threads."
    )
    options = parser.parse_args()
    if options.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(options.threads)
    solve_and_report(show_plot=True, joint_stiffness=options.joint_stiffness)
