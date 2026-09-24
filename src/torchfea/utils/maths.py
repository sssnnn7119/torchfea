"""Differentiable mathematical helpers used by the mechanics models."""

from __future__ import annotations

import torch


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
