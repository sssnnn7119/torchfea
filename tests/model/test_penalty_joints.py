import pytest
import torch

from torchfea.model import Assembly, ReferencePoint
from torchfea.model.loads import SingleDoFJoint, SphericalJoint
from torchfea.interfaces import Serializable
from torchfea.utils import Rotation


def make_joint(joint_type, **kwargs):
    assembly = Assembly()
    assembly.add_reference_point(ReferencePoint([0.0, 0.0, 0.0]), "a")
    assembly.add_reference_point(ReferencePoint([1.0, 0.0, 0.0]), "b")
    joint = joint_type("a", "b", connection_point=[0.0, 1.0, 0.0], k=100.0, **kwargs)
    assembly.add_load(joint, "joint")
    assembly.initialize()
    assembly._RGC = [torch.zeros_like(part) for part in assembly._RGC]
    return assembly, joint


def energy(joint, rgc):
    return -joint.get_potential_energy(rgc)


def test_single_dof_hinge_and_offset_anchor():
    assembly, joint = make_joint(SingleDoFJoint, free_direction=[0, 0, 0, 0, 0, 1])
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index2][5] = 0.7
    assert energy(joint, q).item() > 0.0
    q[joint._rp_index2][3] = 0.2
    assert energy(joint, q).item() > 0.0
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index2][5] = 0.7
    # The connection is offset from both RPs: free rotation also requires a
    # translation so the two material points at the anchor stay coincident.
    q[joint._rp_index2][:3] = torch.tensor([0.0, 1.0, 0.0]) - (
        torch.tensor([1.0, 0.0, 0.0])
        + Rotation.from_rotvec(q[joint._rp_index2][3:]).as_matrix() @ torch.tensor([-1.0, 1.0, 0.0])
    )
    assert energy(joint, q).item() == pytest.approx(0.0, abs=1e-9)


def test_slider_spherical_and_common_rigid_motion():
    assembly, slider = make_joint(SingleDoFJoint, free_direction=[1, 0, 0, 0, 0, 0])
    q = [part.clone() for part in assembly._RGC]
    q[slider._rp_index2][0] = 0.3
    assert energy(slider, q).item() == pytest.approx(0.0, abs=1e-10)
    q[slider._rp_index2][1] = 0.2
    assert energy(slider, q).item() == pytest.approx(2.0)

    assembly, ball = make_joint(SphericalJoint)
    q = [part.clone() for part in assembly._RGC]
    q[ball._rp_index2][3:] = torch.tensor([0.1, 0.2, -0.3])
    assert energy(ball, q).item() > 0.0  # offset anchor must move with RP 2
    q[ball._rp_index2][:3] = torch.tensor([0.0, 1.0, 0.0]) - (
        torch.tensor([1.0, 0.0, 0.0])
        + Rotation.from_rotvec(q[ball._rp_index2][3:]).as_matrix() @ torch.tensor([-1.0, 1.0, 0.0])
    )
    assert energy(ball, q).item() == pytest.approx(0.0, abs=1e-9)
    q = [part.clone() for part in assembly._RGC]
    q[ball._rp_index1][:3] = torch.tensor([0.4, -0.2, 0.3])
    q[ball._rp_index1][3:] = q[ball._rp_index2][3:] = torch.tensor([0.1, 0.2, -0.3])
    rotation = Rotation.from_rotvec(q[ball._rp_index1][3:]).as_matrix()
    q[ball._rp_index2][:3] = q[ball._rp_index1][:3] + (
        rotation - torch.eye(3)
    ) @ torch.tensor([1.0, 0.0, 0.0])
    assert energy(ball, q).item() == pytest.approx(0.0, abs=1e-9)
    q[ball._rp_index2][1] += 0.2
    assert energy(ball, q).item() == pytest.approx(2.0)


@pytest.mark.parametrize("joint_type,kwargs", [
    (SingleDoFJoint, {"free_direction": [0, 0, 0, 0, 0, 1]}),
    (SingleDoFJoint, {"free_direction": [1, 0, 0, 0, 0, 0]}),
    (SingleDoFJoint, {"free_direction": [1, 0, 0, 0, 0, 0.5]}),
    (SphericalJoint, {}),
])
@pytest.mark.parametrize("scale", [0.0, 1.0, 6.0, 15.0])
def test_force_and_tangent_match_energy_derivatives(joint_type, kwargs, scale):
    assembly, joint = make_joint(joint_type, **kwargs)
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index1][:] = scale * torch.tensor([0.02, -0.03, 0.04, 0.04, -0.02, 0.03])
    q[joint._rp_index2][:] = scale * torch.tensor([-0.01, 0.05, 0.02, -0.03, 0.06, 0.07])
    indices, force, stiffness_indices, stiffness = joint.get_stiffness(q)
    assert indices.shape == (12,)
    assert indices.dtype == torch.int64
    assert stiffness_indices.shape[0] == 2
    assert stiffness_indices.shape[1] == stiffness.numel()
    assert stiffness_indices.dtype == torch.int64
    assert 0 < stiffness.numel() <= 144
    assert torch.isfinite(stiffness).all()
    matrix = torch.sparse_coo_tensor(
        stiffness_indices, stiffness, (assembly._RGC_list_indexStart[-1],) * 2
    ).coalesce()
    assert matrix.layout == torch.sparse_coo
    assert matrix._nnz() == stiffness.numel()  # no duplicate COO entries
    tangent = matrix.to_dense().index_select(0, indices).index_select(1, indices)
    assert torch.allclose(tangent, tangent.T, atol=1e-7)
    step = 1e-6
    for col in range(12):
        q_plus = [part.clone() for part in q]
        q_minus = [part.clone() for part in q]
        rp_index = joint._rp_index1 if col < 6 else joint._rp_index2
        q_plus[rp_index][col % 6] += step
        q_minus[rp_index][col % 6] -= step
        fd_force = (joint.get_stiffness(q_plus, True)[1] - joint.get_stiffness(q_minus, True)[1]) / (2 * step)
        fd_energy = (energy(joint, q_plus) - energy(joint, q_minus)) / (2 * step)
        assert torch.allclose(fd_force, tangent[:, col], atol=1e-5, rtol=1e-5)
        assert force[col].item() == pytest.approx(-fd_energy.item(), rel=1e-5, abs=1e-5)


def test_invalid_joint_definition():
    with pytest.raises(ValueError, match="different"):
        SphericalJoint("a", "a", [0, 0, 0], 1)
    with pytest.raises(ValueError, match="six-vector"):
        SingleDoFJoint("a", "b", [0, 0, 0], [0] * 6, 1)


def test_half_turn_remains_constrained():
    assembly, joint = make_joint(SingleDoFJoint, free_direction=[0, 0, 0, 0, 0, 1])
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index2][3] = torch.pi
    assert energy(joint, q).item() > 100.0


@pytest.mark.parametrize("joint_type,kwargs", [
    (SingleDoFJoint, {"free_direction": [0, 0, 0, 0, 0, 1]}),
    (SingleDoFJoint, {"free_direction": [1, 0, 0, 0, 0, 0]}),
    (SphericalJoint, {}),
])
def test_joint_assembly_residual_and_sparse_tangent(joint_type, kwargs):
    assembly, joint = make_joint(joint_type, **kwargs)
    assert assembly._RGC_remain_index[joint._rp_index1].all()
    assert assembly._RGC_remain_index[joint._rp_index2].all()
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index1][:] = torch.tensor([0.03, -0.02, 0.04, 0.07, -0.04, 0.09])
    q[joint._rp_index2][:] = torch.tensor([-0.01, 0.11, -0.02, -0.06, 0.08, 0.15])
    residual, indices, values = assembly.assemble_Stiffness_Matrix(RGC=q)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(values).all()
    assert indices.shape[1] == values.numel()
    sparse_tangent = torch.sparse_coo_tensor(indices, values, (12, 12)).coalesce()
    assert sparse_tangent.layout == torch.sparse_coo
    tangent = sparse_tangent.to_dense()
    assert torch.allclose(tangent, tangent.T, atol=1e-7)
    assert torch.allclose(residual, assembly.assemble_force(RGC=q), atol=1e-9)

    step = 1e-6
    for col in range(12):
        q_plus = [part.clone() for part in q]
        q_minus = [part.clone() for part in q]
        rp_index = joint._rp_index1 if col < 6 else joint._rp_index2
        q_plus[rp_index][col % 6] += step
        q_minus[rp_index][col % 6] -= step
        fd_residual = (assembly.assemble_force(RGC=q_plus) - assembly.assemble_force(RGC=q_minus)) / (2 * step)
        fd_energy = (
            assembly._total_Potential_Energy(RGC=q_plus)
            - assembly._total_Potential_Energy(RGC=q_minus)
        ) / (2 * step)
        assert torch.allclose(fd_residual, tangent[:, col], atol=1e-5, rtol=1e-5)
        assert residual[col].item() == pytest.approx(fd_energy.item(), rel=1e-5, abs=1e-5)


def test_centered_spherical_joint_has_only_translation_stiffness_entries():
    assembly = Assembly()
    assembly.add_reference_point(ReferencePoint([0.0, 0.0, 0.0]), "a")
    assembly.add_reference_point(ReferencePoint([0.0, 0.0, 0.0]), "b")
    joint = SphericalJoint("a", "b", [0.0, 0.0, 0.0], k=100.0)
    assembly.add_load(joint, "joint")
    assembly.initialize()
    q = [torch.zeros_like(part) for part in assembly._RGC]
    _, _, indices, values = joint.get_stiffness(q)
    assert values.numel() == 12  # 4 translation blocks, 3 diagonal terms each
    assert not torch.isin(indices[0], torch.tensor([3, 4, 5, 9, 10, 11])).any()
    assert not torch.isin(indices[1], torch.tensor([3, 4, 5, 9, 10, 11])).any()


def test_joint_stiffness_does_not_call_autograd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Joint stiffness must use explicit derivatives")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    assembly, joint = make_joint(SingleDoFJoint, free_direction=[0, 0, 0, 0, 0, 1])
    _, force, _, tangent = joint.get_stiffness(assembly._RGC)
    assert torch.isfinite(force).all()
    assert torch.isfinite(tangent).all()


def test_joint_parameters_include_connection_point_and_direction():
    assembly, joint = make_joint(SingleDoFJoint, free_direction=[0, 0, 0, 0, 0, 1])
    assert joint._parameters.shape == (10,)
    assert torch.equal(joint.connection_point, joint._parameters[1:4])
    assert torch.equal(joint.free_direction, joint._parameters[4:10])
    q = [part.clone() for part in assembly._RGC]
    q[joint._rp_index2][5] = 0.5
    old_energy = energy(joint, q).item()
    old_force = joint.get_stiffness(q, if_onlyforce=True)[1]
    joint.connection_point = [0.0, 0.0, 0.0]
    assert not torch.isclose(energy(joint, q), torch.tensor(old_energy))
    assert not torch.allclose(joint.get_stiffness(q, if_onlyforce=True)[1], old_force)
    joint.free_direction = [1, 0, 0, 0, 0, 0]
    assert torch.equal(joint._parameters[4:10], torch.tensor([1., 0., 0., 0., 0., 0.]))
    with pytest.raises(ValueError, match="connection_point"):
        joint.connection_point = [0.0, 0.0]
    with pytest.raises(ValueError, match="free_direction"):
        joint.free_direction = [0.0] * 6
    restored = Serializable._deserialize(joint._serialize())
    assert torch.equal(restored._parameters, joint._parameters)
    assert torch.equal(restored.connection_point, joint.connection_point)
    assert torch.equal(restored.free_direction, joint.free_direction)


def test_spherical_joint_parameter_layout_and_serialization():
    _, joint = make_joint(SphericalJoint)
    assert joint._parameters.shape == (4,)
    joint._parameters[1:4] = torch.tensor([2.0, 3.0, 4.0])
    assert torch.equal(joint.connection_point, torch.tensor([2.0, 3.0, 4.0]))
    restored = Serializable._deserialize(joint._serialize())
    assert torch.equal(restored._parameters, joint._parameters)
