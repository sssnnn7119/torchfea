"""One active 3-D rotation with interchangeable representations.

Quaternions use scalar-first [w, x, y, z]. Euler angles use radians and the
XYZ convention R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Rotation vectors use
axis times angle in radians. The canonical quaternion has w >= 0, so the
reported rotation vector is the principal one (angle <= pi).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial

import numpy as np
import torch

RotationInput = Sequence[float] | np.ndarray | torch.Tensor
MatrixInput = Sequence[Sequence[float]] | np.ndarray | torch.Tensor


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 matrix S such that S @ v = vector cross v."""
    zero = torch.zeros_like(vector[0])
    return torch.stack((
        torch.stack((zero, -vector[2], vector[1])),
        torch.stack((vector[2], zero, -vector[0])),
        torch.stack((-vector[1], vector[0], zero)),
    ))


def _series(t: torch.Tensor, coefficients: tuple[float, ...]):
    """Evaluate an even-angle series and its first two derivatives in t=|z|²."""
    coeff = torch.as_tensor(coefficients, device=t.device, dtype=t.dtype)
    n = torch.arange(coeff.numel(), device=t.device, dtype=t.dtype)
    # Separate constants and the linear term: differentiating a vectorized
    # power twice at t=0 can otherwise produce 0 * t**(-1) in autograd.
    powers = torch.cat((torch.ones_like(t).reshape(1), t.reshape(1), t ** n[2:]))
    return (
        coeff @ powers,
        (coeff[1:] * n[1:]) @ powers[:-1],
        (coeff[2:] * n[2:] * (n[2:] - 1)) @ powers[:-2],
    )


_SIN_OVER_ANGLE = tuple((-1.0) ** n / factorial(2 * n + 1) for n in range(24))
_ONE_MINUS_COS = tuple((-1.0) ** n / factorial(2 * n + 2) for n in range(24))


def _quaternion_value(z: torch.Tensor) -> torch.Tensor:
    """Scalar-first unit quaternion for an axis-angle rotation vector."""
    angle_squared = z @ z
    if angle_squared < 1e-10:
        # Smooth at identity, including second derivatives.
        scalar = 1 - angle_squared / 8 + angle_squared**2 / 384
        scale = 0.5 - angle_squared / 48 + angle_squared**2 / 3840
        return torch.cat((scalar.reshape(1), scale * z))
    half_angle = 0.5 * torch.sqrt(angle_squared)
    return torch.cat((
        torch.cos(half_angle).reshape(1),
        0.5 * torch.sinc(half_angle / torch.pi) * z,
    ))


def _quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product, broadcasting over leading dimensions."""
    left_vector, right_vector = torch.broadcast_tensors(left[..., 1:], right[..., 1:])
    return torch.cat((
        left[..., :1] * right[..., :1] - (left_vector * right_vector).sum(dim=-1, keepdim=True),
        left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:]
        + torch.cross(left_vector, right_vector, dim=-1),
    ), dim=-1)


def _quaternion_log_value(q: torch.Tensor) -> torch.Tensor:
    """Principal rotation vector from a scalar-first unit quaternion."""
    w, v = q[0], q[1:]
    s2 = v @ v
    if s2 < 1.0e-10:
        factor = 2 / w - 2 * s2 / (3 * w**3) + 2 * s2**2 / (5 * w**5)
    else:
        s = torch.sqrt(s2)
        factor = 2 * torch.atan2(s, w) / s
    return factor * v


def _vector(value: RotationInput, length: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if tensor.shape != (length,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain {length} finite numbers.")
    return tensor


class Rotation:
    """An active 3-D rotation stored as a normalized scalar-first quaternion.

    Construct with ``from_rotvec``, ``from_quaternion``, ``from_matrix``, or
    ``from_euler``. The corresponding ``as_*`` methods return tensors; angles
    are radians unless ``degrees=True`` is requested for Euler angles.
    """

    __slots__ = ("_quaternion",)

    def __init__(self, quaternion: torch.Tensor) -> None:
        if quaternion.shape != (4,) or not torch.isfinite(quaternion).all():
            raise ValueError("quaternion must contain four finite numbers.")
        norm = torch.linalg.vector_norm(quaternion)
        if norm == 0:
            raise ValueError("quaternion must have a finite nonzero norm.")
        unit = quaternion / norm
        self._quaternion = torch.where(unit[0] < 0, -unit, unit)

    @classmethod
    def from_quaternion(cls, quaternion: RotationInput) -> Rotation:
        """Create from scalar-first [w, x, y, z]."""
        return cls(_vector(quaternion, 4, "quaternion"))

    @classmethod
    def from_rotvec(cls, rotation_vector: RotationInput) -> Rotation:
        """Create from axis times angle, in radians."""
        z = _vector(rotation_vector, 3, "rotation_vector")
        return cls(_quaternion_value(z))

    @classmethod
    def from_matrix(cls, matrix: MatrixInput) -> Rotation:
        """Create from a proper orthogonal 3x3 matrix."""
        r = torch.as_tensor(matrix)
        if not r.is_floating_point():
            r = r.to(torch.get_default_dtype())
        if r.shape != (3, 3) or not torch.isfinite(r).all():
            raise ValueError("matrix must be a finite 3x3 rotation matrix.")
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        if not torch.allclose(r.T @ r, eye, atol=1e-6, rtol=1e-6) or not torch.allclose(
            torch.linalg.det(r), r.new_tensor(1.0), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("matrix must be orthogonal with determinant +1.")

        trace = torch.trace(r)
        if trace > 0:
            scale = 2 * torch.sqrt(1 + trace)
            q = torch.stack((
                scale / 4,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ))
        elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
            scale = 2 * torch.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
            q = torch.stack((
                (r[2, 1] - r[1, 2]) / scale, scale / 4,
                (r[0, 1] + r[1, 0]) / scale,
                (r[0, 2] + r[2, 0]) / scale,
            ))
        elif r[1, 1] >= r[2, 2]:
            scale = 2 * torch.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2])
            q = torch.stack((
                (r[0, 2] - r[2, 0]) / scale,
                (r[0, 1] + r[1, 0]) / scale, scale / 4,
                (r[1, 2] + r[2, 1]) / scale,
            ))
        else:
            scale = 2 * torch.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2])
            q = torch.stack((
                (r[1, 0] - r[0, 1]) / scale,
                (r[0, 2] + r[2, 0]) / scale,
                (r[1, 2] + r[2, 1]) / scale, scale / 4,
            ))
        return cls(q)

    @classmethod
    def from_euler(cls, angles: RotationInput, *, degrees: bool = False) -> Rotation:
        """Create from XYZ roll, pitch, yaw, with R = Rz @ Ry @ Rx."""
        xyz = _vector(angles, 3, "angles")
        if degrees:
            xyz = torch.deg2rad(xyz)
        half = xyz / 2
        sx, sy, sz = torch.sin(half).unbind()
        cx, cy, cz = torch.cos(half).unbind()
        return cls(torch.stack((
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        )))

    def as_quaternion(self) -> torch.Tensor:
        """Return scalar-first [w, x, y, z]."""
        return self._quaternion.clone()

    def as_matrix(self) -> torch.Tensor:
        """Return the active 3x3 rotation matrix."""
        w, v = self._quaternion[0], self._quaternion[1:]
        eye = torch.eye(3, device=v.device, dtype=v.dtype)
        return (w * w - v @ v) * eye + 2 * torch.outer(v, v) + 2 * w * skew_symmetric(v)

    @staticmethod
    def matrix_derivatives(
        rotation_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return R, dR/dz, d²R/dz² for the supplied rotation vector z.

        Shapes are (3,3), (3,3,3), (3,3,3,3), with derivative axes last.
        The vectorized analytic formulas remain finite at zero rotation.
        """
        z = rotation_vector
        t = z @ z
        a, a_t, a_tt = _series(t, _SIN_OVER_ANGLE)
        b, b_t, b_tt = _series(t, _ONE_MINUS_COS)
        s = skew_symmetric(z)
        s2 = s @ s
        eye = torch.eye(3, device=z.device, dtype=z.dtype)
        basis = torch.zeros((3, 3, 3), device=z.device, dtype=z.dtype)
        basis[0, 1, 2], basis[0, 2, 1] = -1, 1
        basis[1, 0, 2], basis[1, 2, 0] = 1, -1
        basis[2, 0, 1], basis[2, 1, 0] = -1, 1
        ds2 = basis @ s + s @ basis
        rotation = eye + a * s + b * s2
        first = (
            2 * a_t * z[:, None, None] * s
            + a * basis
            + 2 * b_t * z[:, None, None] * s2
            + b * ds2
        ).permute(1, 2, 0)
        second = (
            (2 * a_t * eye + 4 * a_tt * torch.outer(z, z))[:, :, None, None] * s
            + 2
            * a_t
            * (
                z[:, None, None, None] * basis[None]
                + z[None, :, None, None] * basis[:, None]
            )
            + (2 * b_t * eye + 4 * b_tt * torch.outer(z, z))[:, :, None, None] * s2
            + 2
            * b_t
            * (z[:, None, None, None] * ds2[None] + z[None, :, None, None] * ds2[:, None])
            + b * (basis[:, None] @ basis[None] + basis[None] @ basis[:, None])
        ).permute(2, 3, 0, 1)
        return rotation, first, second

    def as_rotvec(self) -> torch.Tensor:
        """Return the principal axis-angle rotation vector in radians."""
        return _quaternion_log_value(self._quaternion)

    def as_euler(self, *, degrees: bool = False) -> torch.Tensor:
        """Return XYZ roll, pitch, yaw; chooses roll=0 at gimbal lock."""
        r = self.as_matrix()
        pitch = torch.asin((-r[2, 0]).clamp(-1, 1))
        if torch.abs(torch.cos(pitch)) < 1e-7:
            roll = pitch.new_zeros(())
            yaw = torch.atan2(-r[0, 1], r[1, 1])
        else:
            roll = torch.atan2(r[2, 1], r[2, 2])
            yaw = torch.atan2(r[1, 0], r[0, 0])
        angles = torch.stack((roll, pitch, yaw))
        return torch.rad2deg(angles) if degrees else angles

    def inverse(self) -> Rotation:
        """Return the inverse rotation."""
        return Rotation(self._quaternion * self._quaternion.new_tensor([1, -1, -1, -1]))

    def compose(self, other: Rotation) -> Rotation:
        """Return self @ other: apply other first, then self."""
        if not isinstance(other, Rotation):
            raise TypeError("other must be a Rotation.")
        return Rotation(_quaternion_product(self._quaternion, other._quaternion))

    def apply(self, vector: RotationInput | Sequence[Sequence[float]]) -> torch.Tensor:
        """Rotate a 3-vector or an array whose final dimension is 3."""
        value = torch.as_tensor(vector, device=self._quaternion.device, dtype=self._quaternion.dtype)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError("vector must have a final dimension of length 3.")
        return value @ self.as_matrix().T


