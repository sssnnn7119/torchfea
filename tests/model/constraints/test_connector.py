import numpy as np
import pytest
import torch

from torchfea.model.assembly import Assembly
from torchfea.model.constraints import Connector
from torchfea.model.reference_points import ReferencePoint

TRANSLATION_BASIS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


def test_spherical_joint_keeps_three_relative_rotations():
    joint = Connector(
        "master",
        "slave",
        rotation_free_directions=TRANSLATION_BASIS,
    )

    assert joint._RGC_requirements == (3,)
    assert joint._free_translation_basis_np.shape == (0, 3)
    np.testing.assert_allclose(joint._free_rotation_basis_np, np.eye(3))


def test_hinge_joint_keeps_one_relative_rotation():
    joint = Connector(
        "master",
        "slave",
        rotation_free_directions=[
            [0.0, 0.0, 1.0],
        ],
    )

    assert joint._RGC_requirements == (1,)
    assert joint._free_translation_basis_np.shape == (0, 3)
    assert joint._free_rotation_basis_np.shape == (1, 3)
    np.testing.assert_allclose(joint._free_rotation_basis_np, [[0.0, 0.0, 1.0]])


def test_rigid_joint_has_no_relative_rotation_coordinate():
    joint = Connector(
        "master",
        "slave",
    )

    assert joint._RGC_requirements == (0,)
    assert joint._free_translation_basis_np.shape == (0, 3)
    assert joint._free_rotation_basis_np.shape == (0, 3)


@pytest.mark.parametrize(
    "translation_free_directions, rotation_free_directions",
    [
        ([[1.0, 0.0, 0.0]] * 4, []),
        (
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [],
        ),
        (
            [],
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        ([], [None]),
    ],
)
def test_joint_rejects_invalid_direction_definitions(
    translation_free_directions,
    rotation_free_directions,
):
    with pytest.raises(ValueError):
        Connector(
            "master",
            "slave",
            translation_free_directions,
            rotation_free_directions,
        )


def test_rotation_map_uses_the_master_body_frame():
    joint = Connector(
        "master",
        "slave",
        rotation_free_directions=[
            [0.0, 0.0, 1.0],
        ],
    )
    joint._free_translation_basis = torch.zeros((0, 3))
    joint._free_rotation_basis = torch.tensor([[0.0, 0.0, 1.0]])
    joint._master_connection_offset = torch.tensor([0.0, 0.0, 0.0])
    joint._slave_connection_offset = torch.tensor([1.0, 2.0, 3.0])
    joint._initial_origin_offset = torch.tensor([1.0, 2.0, 3.0])

    active = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # master translation
            0.0,
            0.0,
            0.0,  # master rotation
            0.25,  # free rotation around local z
        ]
    )
    slave = joint._slave_from_active(active)

    expected_translation = (
        joint._initial_origin_offset
        - joint._rotation_matrix(torch.tensor([0.0, 0.0, 0.25]))
        @ joint._slave_connection_offset
    )
    np.testing.assert_allclose(slave[:3].numpy(), expected_translation.numpy())
    np.testing.assert_allclose(
        slave[3:].numpy(),
        np.array([0.0, 0.0, 0.25]),
        atol=1.0e-10,
    )


def test_translation_free_coordinate_uses_master_local_direction():
    joint = Connector(
        "master",
        "slave",
        translation_free_directions=[[1.0, 0.0, 0.0]],
    )
    joint._free_translation_basis = torch.tensor([[1.0, 0.0, 0.0]])
    joint._free_rotation_basis = torch.zeros((0, 3))
    joint._master_connection_offset = torch.zeros(3)
    joint._slave_connection_offset = torch.zeros(3)
    joint._initial_origin_offset = torch.zeros(3)

    active = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # master translation
            0.0,
            0.0,
            np.pi / 2.0,  # master rotation
            2.0,  # local x translation
        ]
    )
    slave = joint._slave_from_active(active)

    np.testing.assert_allclose(
        slave[:3].numpy(),
        np.array([0.0, 2.0, 0.0]),
        atol=1.0e-10,
    )


def test_connection_point_is_localized_in_both_reference_frames():
    assembly = Assembly()
    assembly.add_reference_point(
        ReferencePoint(torch.tensor([0.0, 0.0, 0.0])),
        "master",
    )
    assembly.add_reference_point(
        ReferencePoint(torch.tensor([2.0, 0.0, 0.0])),
        "slave",
    )
    assembly.add_constraint(
        Connector(
            "master",
            "slave",
            connection_point=[0.5, 1.0, 0.0],
        ),
        "joint",
    )

    assembly.initialize()
    joint = assembly.get_constraint("joint")

    np.testing.assert_allclose(
        joint._master_connection_offset.numpy(),
        [0.5, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        joint._slave_connection_offset.numpy(),
        [-1.5, 1.0, 0.0],
    )
