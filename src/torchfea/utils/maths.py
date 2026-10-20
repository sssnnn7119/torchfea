"""Differentiable mathematical helpers used by the mechanics models."""

from __future__ import annotations

import torch


def rotate_vectors(
    rotation_vector: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate one or more 3D vectors with an axis-angle vector.

    The rotation uses Rodrigues' formula directly and never constructs a
    rotation matrix.  ``vectors`` may have shape ``(3,)`` or ``(n, 3)``;
    the returned tensor always has shape ``(n, 3)``.
    """
    rotation_vector = rotation_vector.reshape(3)
    vectors = vectors.reshape(-1, 3)

    theta_squared = torch.dot(rotation_vector, rotation_vector)
    dtype_info = torch.finfo(theta_squared.dtype)
    # The clamp makes sqrt differentiable at the exact zero vector.  The
    # resulting tiny angle is below floating-point resolution in the returned
    # rotation, while the Taylor branch below supplies the correct values.
    theta = torch.sqrt(theta_squared.clamp_min(dtype_info.tiny))
    theta_safe = theta.clamp_min(dtype_info.eps)
    theta_squared_safe = theta_squared.clamp_min(dtype_info.eps)

    # Use Taylor expansions near zero so the operation remains finite and
    # differentiable without a Python-side zero-angle branch.
    small_angle = theta_squared < 1.0e-8
    sinc = torch.where(
        small_angle,
        1.0 - theta_squared / 6.0 + theta_squared**2 / 120.0,
        torch.sin(theta) / theta_safe,
    )
    cosc = torch.where(
        small_angle,
        0.5 - theta_squared / 24.0 + theta_squared**2 / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared_safe,
    )

    rotation = rotation_vector.expand_as(vectors)
    cross = torch.cross(rotation, vectors, dim=1)
    projection = (vectors * rotation_vector).sum(dim=1, keepdim=True)
    return (
        vectors * torch.cos(theta)
        + cross * sinc
        + rotation_vector * projection * cosc
    )


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    return torch.stack(
        [
            torch.stack([torch.zeros_like(vector[0]), -vector[2], vector[1]]),
            torch.stack([vector[2], torch.zeros_like(vector[0]), -vector[0]]),
            torch.stack([-vector[1], vector[0], torch.zeros_like(vector[0])]),
        ]
    )


def rotation_matrix_from_vector(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle rotation vector to a 3x3 rotation matrix."""
    return torch.matrix_exp(skew_symmetric(rotation_vector))


def rotation_vector_from_matrix(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to its axis-angle rotation vector."""
    sine_axis = 0.5 * torch.stack(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]
    )

    sine = torch.linalg.vector_norm(sine_axis)
    cosine = (torch.trace(rotation_matrix) - 1.0) * 0.5
    angle = torch.atan2(sine, cosine)

    # The small-angle branch gives the correct first derivative at the
    # identity while avoiding division by zero.
    regular_factor = angle / sine.clamp_min(1.0e-12)
    small_factor = 1.0 + sine * sine / 6.0
    factor = torch.where(
        sine < 1.0e-7,
        small_factor,
        regular_factor,
    )
    return factor * sine_axis
