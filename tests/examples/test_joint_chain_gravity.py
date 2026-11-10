from pathlib import Path
import runpy

import torch

from torchfea.model.constraints import Couple
from torchfea.model.loads import SingleDoFJoint
from torchfea.utils import Rotation


def test_joint_chain_uses_penalty_hinges_and_converges():
    path = Path(__file__).resolve().parents[2] / "examples/joint_chain_test/joint_chain_gravity.py"
    example = runpy.run_path(str(path))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        controller, result = example["solve_and_report"](show_plot=False)
        assembly = controller.assembly
        rgc = assembly._GC2RGC(result.GC)
        assert result.converged
        assert result.GC.numel() == 28
        assert assembly.assemble_force(GC=result.GC).abs().max() < 1e-6
        assert len(assembly._constraints) == 5
        assert all(isinstance(constraint, Couple) for constraint in assembly._constraints.values())
        joints = [load for load in assembly._loads.values() if isinstance(load, SingleDoFJoint)]
        assert len(joints) == 5

        for joint in joints:
            rp1 = assembly.get_reference_point(joint.rp_name1)
            rp2 = assembly.get_reference_point(joint.rp_name2)
            q1, q2 = rgc[rp1._RGC_index], rgc[rp2._RGC_index]
            r1, r2 = Rotation.from_rotvec(q1[3:]), Rotation.from_rotvec(q2[3:])
            anchor1 = rp1.node + q1[:3] + r1.apply(joint.connection_point - rp1.node)
            anchor2 = rp2.node + q2[:3] + r2.apply(joint.connection_point - rp2.node)
            assert torch.linalg.vector_norm(anchor2 - anchor1) < 1e-5
            assert torch.linalg.vector_norm(r1.inverse().compose(r2).as_rotvec()[:2]) < 1e-7

        wall = assembly.get_reference_point("wall-rp")
        tip = assembly.get_reference_point("link-4-rp")
        assert torch.equal(rgc[wall._RGC_index], torch.zeros(6))
        assert torch.equal(rgc[tip._RGC_index][1:3], torch.zeros(2))
        assert -0.05 < rgc[tip._RGC_index][0] < 0.0  # shortening as the chain sags
        assert rgc[assembly.get_reference_point("link-1-rp")._RGC_index][1] < -0.01
        meshes = assembly.get_meshes(GC=result.GC)
        assert len(meshes) == 5 and all(mesh.n_points > 0 for mesh in meshes.values())
    finally:
        torch.set_num_threads(previous_threads)
