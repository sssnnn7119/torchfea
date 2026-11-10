import math

import pytest
import torch

from torchfea.utils import Rotation, skew_symmetric


@pytest.mark.parametrize("rotvec", [
    [0.0, 0.0, 0.0],
    [0.2, -0.3, 0.4],
    [math.pi, 0.0, 0.0],
    [0.0, math.pi, 0.0],
    [0.0, 0.0, math.pi],
    [-1.2, 1.4, 0.7],
])
def test_rotation_representations_round_trip(rotvec):
    z = torch.tensor(rotvec)
    rotation = Rotation.from_rotvec(z)
    matrix = rotation.as_matrix()
    assert torch.allclose(matrix, torch.matrix_exp(skew_symmetric(z)), atol=1e-12)
    assert torch.allclose(Rotation.from_quaternion(rotation.as_quaternion()).as_matrix(), matrix, atol=1e-12)
    assert torch.allclose(Rotation.from_matrix(matrix).as_matrix(), matrix, atol=1e-12)
    assert torch.allclose(Rotation.from_euler(rotation.as_euler()).as_matrix(), matrix, atol=1e-12)
    assert torch.allclose(Rotation.from_rotvec(rotation.as_rotvec()).as_matrix(), matrix, atol=1e-12)


def test_euler_xyz_order_and_degrees():
    rotation = Rotation.from_euler([90.0, 0.0, 0.0], degrees=True)
    assert torch.allclose(rotation.apply([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), atol=1e-12)
    assert torch.allclose(rotation.as_euler(degrees=True), torch.tensor([90.0, 0.0, 0.0]), atol=1e-12)
    # For XYZ angles, the active matrix is Rz(yaw) @ Ry(pitch) @ Rx(roll).
    composed = Rotation.from_euler([0.0, 0.0, 90.0], degrees=True).compose(rotation)
    assert torch.allclose(composed.apply([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), atol=1e-12)
    assert torch.allclose(composed.inverse().as_matrix() @ composed.as_matrix(), torch.eye(3), atol=1e-12)


@pytest.mark.parametrize("pitch", [-90.0, 90.0])
def test_euler_gimbal_lock_preserves_rotation(pitch):
    rotation = Rotation.from_euler([20.0, pitch, 35.0], degrees=True)
    recovered = Rotation.from_euler(rotation.as_euler(degrees=True), degrees=True)
    assert torch.allclose(recovered.as_matrix(), rotation.as_matrix(), atol=1e-12)


def test_invalid_rotation_inputs():
    with pytest.raises(ValueError, match="quaternion"):
        Rotation.from_quaternion([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="rotation_vector"):
        Rotation.from_rotvec([1.0, 2.0])
    with pytest.raises(ValueError, match="orthogonal"):
        Rotation.from_matrix(torch.diag(torch.tensor([1.0, 1.0, -1.0])))


