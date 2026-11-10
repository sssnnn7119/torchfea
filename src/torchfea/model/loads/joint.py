"""Penalty joints between two six-DoF reference points."""

from collections.abc import Sequence
from math import factorial

import numpy as np
import torch

from ...utils import Rotation
from ...utils.rotation import _quaternion_product, _series
from .base import BaseLoad

Vector3 = Sequence[float] | np.ndarray | torch.Tensor
Vector6 = Sequence[float] | np.ndarray | torch.Tensor


_COS_HALF = tuple((-1.0) ** n / (2.0 ** (2 * n) * factorial(2 * n)) for n in range(24))
_SIN_HALF_OVER_ANGLE = tuple(
    0.5 * (-1.0) ** n / (2.0 ** (2 * n) * factorial(2 * n + 1)) for n in range(24)
)


def _quaternion_derivatives(z: torch.Tensor):
    """Unit quaternion and its first two rotvec derivatives."""
    t = z @ z
    w, w_t, w_tt = _series(t, _COS_HALF)
    c, c_t, c_tt = _series(t, _SIN_HALF_OVER_ANGLE)
    eye = torch.eye(3, device=z.device, dtype=z.dtype)
    value = torch.cat((w.reshape(1), c * z))
    first = torch.cat(
        (
            (2 * w_t * z)[None],
            c * eye + 2 * c_t * torch.outer(z, z),
        )
    )
    second = torch.cat(
        (
            (2 * w_t * eye + 4 * w_tt * torch.outer(z, z))[None],
            2
            * c_t
            * (
                torch.einsum("ai,j->aij", eye, z)
                + torch.einsum("aj,i->aij", eye, z)
                + torch.einsum("ij,a->aij", eye, z)
            )
            + 4 * c_tt * torch.einsum("a,i,j->aij", z, z, z),
        )
    )
    return value, first, second


def _relative_quaternion_derivatives(z1: torch.Tensor, z2: torch.Tensor):
    q1, dq1, ddq1 = _quaternion_derivatives(z1)
    q2, dq2, ddq2 = _quaternion_derivatives(z2)
    conjugate = torch.tensor([1.0, -1.0, -1.0, -1.0], device=z1.device, dtype=z1.dtype)
    q1, dq1, ddq1 = (
        conjugate * q1,
        conjugate[:, None] * dq1,
        conjugate[:, None, None] * ddq1,
    )
    value = _quaternion_product(q1, q2)
    first = torch.zeros((4, 12), device=z1.device, dtype=z1.dtype)
    second = torch.zeros((4, 12, 12), device=z1.device, dtype=z1.dtype)
    first[:, 3:6] = _quaternion_product(dq1.T, q2).T
    first[:, 9:12] = _quaternion_product(q1, dq2.T).T
    second[:, 3:6, 3:6] = _quaternion_product(ddq1.permute(1, 2, 0), q2).permute(
        2, 0, 1
    )
    second[:, 9:12, 9:12] = _quaternion_product(q1, ddq2.permute(1, 2, 0)).permute(
        2, 0, 1
    )
    cross = _quaternion_product(dq1.T[:, None], dq2.T[None, :]).permute(2, 0, 1)
    second[:, 3:6, 9:12] = cross
    second[:, 9:12, 3:6] = cross.transpose(1, 2)
    # The principal rotation vector uses q and -q interchangeably.  The
    # derivative is valid on either smooth side of the 180-degree branch.
    sign = -1.0 if value[0] < 0 else 1.0
    return sign * value, sign * first, sign * second


def _quaternion_log_derivatives(q: torch.Tensor):
    """Rotation vector and derivatives with respect to quaternion components."""
    w, v = q[0], q[1:]
    s2 = v @ v
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    if s2 < 1.0e-10:
        # Taylor series of 2 atan(|v|/w)/|v|, avoiding 0/0 at identity.
        phi = 2 / w - 2 * s2 / (3 * w**3) + 2 * s2**2 / (5 * w**5)
        phi_w = -2 / w**2 + 2 * s2 / w**4 - 2 * s2**2 / w**6
        phi_ww = 4 / w**3 - 8 * s2 / w**5 + 12 * s2**2 / w**7
        phi_t = -2 / (3 * w**3) + 4 * s2 / (5 * w**5)
        phi_tt = 4 / (5 * w**5)
        phi_wt = 2 / w**4 - 4 * s2 / w**6
        phi_v = 2 * phi_t * v
        phi_vv = 2 * phi_t * eye + 4 * phi_tt * torch.outer(v, v)
        phi_wv = 2 * phi_wt * v
    else:
        s = torch.sqrt(s2)
        denominator = s2 + w * w
        angle = 2 * torch.atan2(s, w)
        angle_s = 2 * w / denominator
        angle_w = -2 * s / denominator
        phi = angle / s
        phi_s = angle_s / s - angle / s**2
        phi_ss = -4 * w / denominator**2 - 2 * angle_s / s**2 + 2 * angle / s**3
        phi_w = angle_w / s
        phi_ww = 4 * w / denominator**2
        phi_sw = 2 * (s2 - w * w) / (denominator**2 * s) - angle_w / s**2
        phi_v = phi_s * v / s
        phi_vv = (phi_s / s) * eye + (phi_ss / s2 - phi_s / s**3) * torch.outer(v, v)
        phi_wv = phi_sw * v / s
    value = phi * v
    first = torch.zeros((3, 4), device=q.device, dtype=q.dtype)
    second = torch.zeros((3, 4, 4), device=q.device, dtype=q.dtype)
    first[:, 0] = phi_w * v
    first[:, 1:] = phi * eye + torch.outer(v, phi_v)
    second[:, 0, 0] = phi_ww * v
    second[:, 0, 1:] = phi_w * eye + torch.outer(v, phi_wv)
    second[:, 1:, 0] = second[:, 0, 1:]
    second[:, 1:, 1:] = (
        torch.einsum("ai,j->aij", eye, phi_v)
        + torch.einsum("aj,i->aij", eye, phi_v)
        + torch.einsum("a,ij->aij", v, phi_vv)
    )
    return value, first, second


class _PenaltyJoint(BaseLoad):
    """Shared geometry and consistent force/tangent for reference-point joints.

    The relative translation and rotation are expressed in RP 1's initial
    frame.  ``connection_point`` is a point in the initial global frame and
    is attached rigidly to both reference points.

    Parameters:
        rp_name1: Name of the first reference point (str).
        rp_name2: Name of the second reference point (str).
        connection_point: Initial global [x, y, z] coordinates (3 floats).
        k: Positive penalty coefficient (float).

    ``_parameters`` layout: [k, connection_x, connection_y, connection_z].
    """

    _serialized_attributes = ["rp_name1", "rp_name2", "_parameters"]
    _serialized_attributes_exclude = [
        "_rp_index1",
        "_rp_index2",
        "_indices_force",
        "_node1",
        "_node2",
    ]

    def __init__(
        self, rp_name1: str, rp_name2: str, connection_point: Vector3, k: float
    ) -> None:
        super().__init__()
        if rp_name1 == rp_name2:
            raise ValueError("A joint needs two different reference points.")
        point = torch.as_tensor(connection_point, dtype=torch.float64)
        if point.shape != (3,) or not torch.isfinite(point).all():
            raise ValueError(
                "connection_point must contain three finite global coordinates."
            )
        if not np.isfinite(k) or k <= 0:
            raise ValueError("k must be finite and positive.")
        self.rp_name1 = rp_name1
        self.rp_name2 = rp_name2
        self._parameters = torch.cat((point.new_tensor([k]), point.clone()))
        self._rp_index1 = None
        self._rp_index2 = None
        self._node1 = None
        self._node2 = None

    @property
    def k(self) -> torch.Tensor:
        return self._parameters[0]

    @k.setter
    def k(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("k must be finite and positive.")
        self._parameters[0] = value

    @property
    def connection_point(self) -> torch.Tensor:
        """Initial global joint position, shape (3,), stored in _parameters."""
        return self._parameters[1:4]

    @connection_point.setter
    def connection_point(self, value: Vector3) -> None:
        point = torch.as_tensor(
            value, device=self._parameters.device, dtype=self._parameters.dtype
        )
        if point.shape != (3,) or not torch.isfinite(point).all():
            raise ValueError(
                "connection_point must contain three finite global coordinates."
            )
        self._parameters[1:4] = point

    def initialize(self, assembly):
        super().initialize(assembly)
        rp1 = assembly.get_reference_point(self.rp_name1)
        rp2 = assembly.get_reference_point(self.rp_name2)
        self._rp_index1, self._rp_index2 = rp1._RGC_index, rp2._RGC_index
        self._node1, self._node2 = rp1.node, rp2.node
        starts = [
            assembly._RGC_list_indexStart[i] for i in (self._rp_index1, self._rp_index2)
        ]
        self._indices_force = torch.cat(
            [
                torch.arange(
                    start, start + 6, device=assembly.device, dtype=torch.int64
                )
                for start in starts
            ]
        )

    def _relative_motion(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        point = self.connection_point.to(device=q1.device, dtype=q1.dtype)
        node1 = self._node1.to(device=q1.device, dtype=q1.dtype)
        node2 = self._node2.to(device=q1.device, dtype=q1.dtype)
        rotation1 = Rotation.from_rotvec(q1[3:])
        rotation2 = Rotation.from_rotvec(q2[3:])
        r1 = rotation1.as_matrix()
        r2 = rotation2.as_matrix()
        x1 = node1 + q1[:3] + r1 @ (point - node1)
        x2 = node2 + q2[:3] + r2 @ (point - node2)
        translation = r1.T @ (x2 - x1)
        rotation = rotation1.inverse().compose(rotation2).as_rotvec()
        return torch.cat((translation, rotation))

    def _motion_derivatives(self, coordinates: torch.Tensor):
        """Relative motion, its 6x12 Jacobian and its 6x12x12 Hessian."""
        q1, q2 = coordinates[:6], coordinates[6:]
        q = coordinates.reshape(2, 6)
        point = self.connection_point.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        nodes = torch.stack((self._node1, self._node2)).to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        rotations, drotations, ddrotations = torch.vmap(Rotation.matrix_derivatives)(
            q[:, 3:]
        )
        eye = torch.eye(3, device=coordinates.device, dtype=coordinates.dtype)
        arm = point - nodes
        x = nodes + q[:, :3] + torch.einsum("nab,nb->na", rotations, arm)
        local_dx = torch.cat(
            (
                eye.expand(2, -1, -1),
                torch.einsum("nabi,nb->nai", drotations, arm),
            ),
            dim=-1,
        )
        dx = torch.block_diag(local_dx[0], local_dx[1]).reshape(2, 3, 12)
        ddx = torch.zeros(
            (2, 3, 12, 12), device=coordinates.device, dtype=coordinates.dtype
        )
        local_ddx = torch.einsum("nabij,nb->naij", ddrotations, arm)
        ddx[0, :, 3:6, 3:6] = local_ddx[0]
        ddx[1, :, 9:12, 9:12] = local_ddx[1]

        delta = x[1] - x[0]
        ddelta = dx[1] - dx[0]
        dddelta = ddx[1] - ddx[0]
        a = rotations[0].T
        da = torch.zeros((3, 3, 12), device=coordinates.device, dtype=coordinates.dtype)
        dda = torch.zeros(
            (3, 3, 12, 12), device=coordinates.device, dtype=coordinates.dtype
        )
        da[:, :, 3:6] = drotations[0].transpose(0, 1)
        dda[:, :, 3:6, 3:6] = ddrotations[0].transpose(0, 1)
        translation = a @ delta
        dtranslation = torch.einsum("abi,b->ai", da, delta) + a @ ddelta
        ddtranslation = (
            torch.einsum("abij,b->aij", dda, delta)
            + torch.einsum("abi,bj->aij", da, ddelta)
            + torch.einsum("abj,bi->aij", da, ddelta)
            + torch.einsum("ab,bij->aij", a, dddelta)
        )

        quaternion, dq, ddq = _relative_quaternion_derivatives(q1[3:], q2[3:])
        rotation, dr_dq, ddr_dq = _quaternion_log_derivatives(quaternion)
        drotation = dr_dq @ dq
        ddrotation = torch.einsum("ab,bij->aij", dr_dq, ddq) + torch.einsum(
            "abc,bi,cj->aij", ddr_dq, dq, dq
        )
        return (
            torch.cat((translation, rotation)),
            torch.cat((dtranslation, drotation)),
            torch.cat((ddtranslation, ddrotation)),
        )

    def _energy(self, coordinates: torch.Tensor) -> torch.Tensor:
        motion = self._relative_motion(coordinates[:6], coordinates[6:])
        constrained = self._constrained_motion(motion)
        k = self.k.to(device=coordinates.device, dtype=coordinates.dtype)
        return 0.5 * k * (constrained @ constrained)

    def _constrained_motion(self, motion: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _coordinates(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (RGC[self._rp_index1].reshape(-1), RGC[self._rp_index2].reshape(-1))
        )

    def _get_stiffness(
        self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs
    ):
        coordinates = self._coordinates(RGC)
        motion, jacobian, motion_hessian = self._motion_derivatives(coordinates)
        constrained, jacobian, motion_hessian = self._constrain_derivatives(
            motion, jacobian, motion_hessian
        )
        k = self.k.to(device=coordinates.device, dtype=coordinates.dtype)
        force = -k * (jacobian.T @ constrained)
        if if_onlyforce:
            return self._indices_force, force
        # Force derivative is the negative penalty-energy Hessian.
        tangent = -k * (
            jacobian.T @ jacobian
            + torch.einsum("a,aij->ij", constrained, motion_hessian)
        )
        indices = self._indices_force
        tangent_values = tangent.reshape(-1)
        nonzero = tangent_values != 0
        stiffness_indices = torch.stack(
            (
                indices.repeat_interleave(12)[nonzero],
                indices.repeat(12)[nonzero],
            )
        )
        return indices, force, stiffness_indices, tangent_values[nonzero]

    def _get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        # Assembly subtracts load potentials from the total energy.
        return -self._energy(self._coordinates(RGC))

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        RGC_remain_index[self._rp_index1][:] = True
        RGC_remain_index[self._rp_index2][:] = True
        return RGC_remain_index

    def _constrain_derivatives(self, motion, jacobian, hessian):
        raise NotImplementedError


class SingleDoFJoint(_PenaltyJoint):
    """Allow motion along one six-vector [tx, ty, tz, rx, ry, rz].

    The direction is specified in the initial global frame, which is RP 1's
    local frame at zero displacement.  Its translational entries have length
    units and rotational entries have radian units.  All five perpendicular
    components are penalized.  Set one of the first three entries for a
    slider, or one of the final three for a hinge.

    Parameters:
        rp_name1: Name of the first reference point (str).
        rp_name2: Name of the second reference point (str).
        connection_point: Initial global [x, y, z] coordinates (3 floats).
        free_direction: Allowed [tx, ty, tz, rx, ry, rz] direction (6 floats).
        k: Positive penalty coefficient (float).

    ``_parameters`` layout: [k, connection_x, connection_y, connection_z,
    direction_tx, direction_ty, direction_tz, direction_rx, direction_ry,
    direction_rz].
    """

    _serialized_attributes = _PenaltyJoint._serialized_attributes

    def __init__(
        self,
        rp_name1: str,
        rp_name2: str,
        connection_point: Vector3,
        free_direction: Vector6,
        k: float,
    ) -> None:
        super().__init__(rp_name1, rp_name2, connection_point, k)
        direction = torch.as_tensor(
            free_direction, device=self._parameters.device, dtype=self._parameters.dtype
        )
        if (
            direction.shape != (6,)
            or not torch.isfinite(direction).all()
            or torch.linalg.vector_norm(direction) == 0
        ):
            raise ValueError("free_direction must be a nonzero finite six-vector.")
        self._parameters = torch.cat((self._parameters, direction.clone()))

    @property
    def free_direction(self) -> torch.Tensor:
        """Allowed six-dimensional relative-motion direction in _parameters."""
        return self._parameters[4:10]

    @free_direction.setter
    def free_direction(self, value: Vector6) -> None:
        direction = torch.as_tensor(
            value, device=self._parameters.device, dtype=self._parameters.dtype
        )
        if (
            direction.shape != (6,)
            or not torch.isfinite(direction).all()
            or torch.linalg.vector_norm(direction) == 0
        ):
            raise ValueError("free_direction must be a nonzero finite six-vector.")
        self._parameters[4:10] = direction

    def _constrained_motion(self, motion: torch.Tensor) -> torch.Tensor:
        direction = self.free_direction.to(device=motion.device, dtype=motion.dtype)
        return motion - direction * ((motion @ direction) / (direction @ direction))

    def _constrain_derivatives(self, motion, jacobian, hessian):
        direction = self.free_direction.to(device=motion.device, dtype=motion.dtype)
        projector = torch.eye(
            6, device=motion.device, dtype=motion.dtype
        ) - torch.outer(direction, direction) / (direction @ direction)
        return (
            projector @ motion,
            projector @ jacobian,
            torch.einsum("ab,bij->aij", projector, hessian),
        )


class SphericalJoint(_PenaltyJoint):
    """Allow all relative rotations while penalizing anchor-point translation.

    Parameters:
        rp_name1: Name of the first reference point (str).
        rp_name2: Name of the second reference point (str).
        connection_point: Initial global [x, y, z] coordinates (3 floats).
        k: Positive penalty coefficient (float).
    """

    def _constrained_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return motion[:3]

    def _constrain_derivatives(self, motion, jacobian, hessian):
        return motion[:3], jacobian[:3], hessian[:3]
