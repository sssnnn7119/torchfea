from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize

from ...utils.bvh import BoundingBoxBVH
from ..elements.dimension3.surfaces import BaseSurface, Q4, Q8, T3, T6
from .linear_tie import LinearTie

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _SurfaceProjection:
    """Closest-point search on an original FE facet, including curved faces."""

    def __init__(self, surface: BaseSurface) -> None:
        if type(surface) not in {T3, T6, Q4, Q8}:
            raise ValueError(f"Unsupported Tie main face type: {type(surface).__name__}.")
        self.triangle: bool = isinstance(surface, (T3, T6))
        self.linear_triangle: bool = type(surface) is T3
        # Differentiate each face family's polynomial once. Projection runs
        # on CPU float64 while runtime interpolation retains the model dtype.
        coefficients = surface.shape_function[0].detach().cpu().to(torch.float64)
        first = torch.stack([
            surface._shape_function_derivative(coefficients, i) for i in range(2)
        ])
        second = torch.stack([
            torch.stack([
                surface._shape_function_derivative(first[i], j) for j in range(2)
            ]) for i in range(2)
        ])
        self.coefficients: FloatArray = coefficients.numpy()
        self.first: FloatArray = first.numpy()
        self.second: FloatArray = second.numpy()
        self.starts: FloatArray
        if self.triangle:
            self.starts = np.array([
                [1 / 3, 1 / 3], [0, 0], [1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
            ], dtype=np.float64)
        else:
            self.starts = np.array([
                [0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1],
                [0, -1], [1, 0], [0, 1], [-1, 0],
            ], dtype=np.float64)

    def evaluate(self, local: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return N [..., n], dN [..., 2, n] and d2N [..., 2, 2, n]."""
        g, h = np.moveaxis(np.asarray(local), -1, 0)
        basis = np.stack([
            np.ones_like(g), g, h, g * h, g * g, h * h, g * g * h, g * h * h,
        ], axis=-1)[..., :self.coefficients.shape[-1]]
        return (
            np.einsum("am,...m->...a", self.coefficients, basis),
            np.einsum("iam,...m->...ia", self.first, basis),
            np.einsum("ijam,...m->...ija", self.second, basis),
        )

    def objective(
        self, local: FloatArray, nodes: FloatArray, point: FloatArray | float,
    ) -> tuple[float, FloatArray, FloatArray]:
        """Squared distance and its exact first and second local derivatives."""
        shape, first, second = self.evaluate(local)
        # The chain rule separates the tangent metric and the curvature-gap
        # term in the Hessian; dropping curvature is incorrect on curved faces.
        relative = nodes - point
        gap = shape @ relative
        tangent = first @ relative
        curvature = np.einsum("ija,ak->ijk", second, relative)
        return (
            float(0.5 * np.dot(gap, gap)),
            tangent @ gap,
            tangent @ tangent.T + np.einsum("ijk,k->ij", curvature, gap),
        )

    def bounds(self, nodes: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Conservative Bernstein bounds; nodal bounds miss curved bulges."""
        if self.triangle:
            # Quadratic triangle Lagrange nodes are converted to Bernstein
            # controls, whose convex hull bounds the entire curved facet.
            controls = nodes[..., :3, :]
            if nodes.shape[-2] == 6:
                edges = np.array([[0, 1], [1, 2], [2, 0]])
                middle = 2 * nodes[..., 3:, :] - 0.5 * nodes[..., edges, :].sum(-2)
                controls = np.concatenate([controls, middle], axis=-2)
        else:
            # Tensor-product degree-two Bernstein controls also cover Q4,
            # and provide conservative Q8 bounds beyond its nodal extrema.
            grid = np.array([(g, h) for g in [-1, 0, 1] for h in [-1, 0, 1]])
            values = np.einsum("qa,...ak->...qk", self.evaluate(grid)[0], nodes)
            values = values.reshape(*nodes.shape[:-2], 3, 3, 3)
            transform = np.array([[1, 0, 0], [-0.5, 2, -0.5], [0, 0, 1]])
            controls = np.einsum("ia,...abk,jb->...ijk", transform, values, transform)
            controls = controls.reshape(*nodes.shape[:-2], 9, 3)
        return controls.min(axis=-2), controls.max(axis=-2)

    def closest_point(
        self, point: FloatArray, nodes: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        """Return local coordinates, FE weights and absolute projection distance."""
        # Remove the global translation before measuring facet size and
        # differentiating the distance; this also avoids cancellation.
        scale = np.linalg.norm(np.ptp(nodes, axis=0))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Degenerate main facet in Tie projection.")
        relative = (nodes - point) / scale
        tangent = self.evaluate(self.starts[0])[1] @ relative
        singular_values = np.linalg.svd(tangent, compute_uv=False)
        if singular_values[-1] <= 1e-12 * singular_values[0]:
            raise ValueError("Degenerate main facet in Tie projection.")

        local: FloatArray | None
        if self.linear_triangle:
            # A planar triangle's minimum is either the orthogonal interior
            # projection or a closest point on one of its three edges.
            edges = np.stack([relative[1] - relative[0], relative[2] - relative[0]], axis=1)
            local = np.linalg.lstsq(edges, -relative[0], rcond=None)[0]
            candidates: list[FloatArray] = []
            if np.all(local >= 0) and local.sum() <= 1:
                candidates.append(local)
            vertices = np.array([[0, 0], [1, 0], [0, 1]])
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                direction = relative[b] - relative[a]
                t = np.clip(-np.dot(relative[a], direction) / np.dot(direction, direction), 0, 1)
                candidates.append((1 - t) * vertices[a] + t * vertices[b])
            local = min(candidates, key=lambda q: self.objective(q, relative, 0)[0])
        else:
            # Bound the natural coordinates and try multiple seeds to locate
            # a minimum on curved facets, retaining edge/corner solutions too.
            bounds = [(0, 1)] * 2 if self.triangle else [(-1, 1)] * 2
            constraints = ({
                "type": "ineq", "fun": lambda q: 1 - q.sum(),
                "jac": lambda q: np.array([-1.0, -1.0]),
            },) if self.triangle else ()
            best_value = np.inf
            local = None
            for start in self.starts:
                value = self.objective(start, relative, 0)[0]
                if value <= 1e-28:
                    local = start.copy()
                    break
                result = minimize(
                    lambda q: self.objective(q, relative, 0)[:2], start,
                    method="SLSQP", jac=True, bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-24, "maxiter": 100},
                )
                if result.success or result.fun <= 1e-26:
                    refined = self._refine_interior(result.x, relative)
                    value = self.objective(refined, relative, 0)[0]
                    if value < best_value:
                        local = refined
                        best_value = value
                if best_value <= 1e-28:
                    break
            if local is None:
                raise ValueError("Main-facet closest-point search did not converge.")
        shape = self.evaluate(local)[0]
        return local, shape, float(np.linalg.norm(shape @ relative) * scale)

    def _refine_interior(self, local: FloatArray, nodes: FloatArray) -> FloatArray:
        """Polish an interior projection using the analytic distance Hessian."""
        def inside(q: FloatArray) -> bool:
            if self.triangle:
                return bool(np.all(q > 1e-12) and q.sum() < 1 - 1e-12)
            return bool(np.all(np.abs(q) < 1 - 1e-12))

        # Newton polishing uses the exact Hessian only inside the domain;
        # backtracking preserves feasibility and decreases squared distance.
        local = local.copy()
        for _ in range(8):
            if not inside(local):
                break
            value, gradient, hessian = self.objective(local, nodes, 0)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) < 2e-15:
                break
            for fraction in [1., 0.5, 0.25, 0.125]:
                candidate = local - fraction * step
                if inside(candidate) and self.objective(candidate, nodes, 0)[0] < value:
                    local = candidate
                    break
            else:
                break
        return local


class NodeToSurfaceTie(LinearTie):
    """Tie secondary surface nodes to fixed points on main FE facets.

    T3, T6, Q4 and Q8 main facets retain their original shape functions.
    ``position_tolerance`` is an absolute initial-distance cutoff. No nodes
    or reference meshes are adjusted. A nonzero initial gap remains a fixed
    global offset, so use coincident surfaces for finite-rotation analyses.
    """

    @property
    def tied_nodes(self) -> torch.Tensor:
        """Secondary instance node IDs participating in this Tie."""
        return self._secondary_nodes

    @property
    def unmatched_nodes(self) -> torch.Tensor:
        """Secondary node IDs outside the initial position tolerance."""
        return self._unmatched_nodes

    @property
    def projection_distances(self) -> torch.Tensor:
        """Initial projection distances in the same order as tied_nodes."""
        return self._projection_distances

    def _build_interpolation_map(
        self,
        main_faces: list[BaseSurface],
        secondary_faces: list[BaseSurface],
        main_coordinates: torch.Tensor,
        secondary_coordinates: torch.Tensor,
    ) -> None:
        # Keep reference-geometry search independent from runtime tensors and
        # preserve original FE connectivity for the interpolation stencils.
        main_coordinates_np = main_coordinates.detach().cpu().to(torch.float64).numpy()
        secondary_coordinates_np = secondary_coordinates.detach().cpu().to(torch.float64).numpy()
        candidates: list[tuple[_SurfaceProjection, IntArray, FloatArray]] = []
        lower_bounds: list[FloatArray] = []
        upper_bounds: list[FloatArray] = []
        for surface in main_faces:
            if not surface._elems.numel():
                continue
            projection = _SurfaceProjection(surface)
            connectivity: IntArray = surface._elems.detach().cpu().to(torch.int64).numpy()
            coordinates = main_coordinates_np[connectivity]
            low, high = projection.bounds(coordinates)
            lower_bounds.extend(low)
            upper_bounds.extend(high)
            candidates.extend((projection, ids, nodes) for ids, nodes in zip(connectivity, coordinates))
        # Build a hierarchy of conservative facet boxes. Each subtree has
        # its own union bounds, so large facets do not inflate all queries.
        lower = np.asarray(lower_bounds)
        upper = np.asarray(upper_bounds)
        tree = BoundingBoxBVH(lower, upper)
        # A zero user tolerance still needs roundoff allowance for points
        # reconstructed from floating-point FE shape functions.
        scale = max(np.abs(lower).max(), np.abs(upper).max())
        roundoff = float(16 * np.finfo(float).eps * scale)
        cutoff = self.position_tolerance + roundoff
        secondary_ids = np.unique(np.concatenate([
            face._elems.detach().cpu().numpy().reshape(-1) for face in secondary_faces
        ]))

        secondary_nodes: list[int] = []
        rows: list[int] = []
        main_nodes: list[int] = []
        weights: list[float] = []
        unmatched: list[int] = []
        distances: list[float] = []
        local_coordinates: list[FloatArray] = []
        main_facets: list[int] = []
        # Visit the smallest geometric lower bound first. Once a projection
        # is found, its actual distance also prunes unvisited tree branches.
        for node_id in secondary_ids:
            point = secondary_coordinates_np[node_id]
            best: tuple[float, IntArray, FloatArray, FloatArray, int] | None = None
            for lower_bound, facet_id in tree.iter_candidates(point, cutoff):
                if best is not None and lower_bound > best[0] + roundoff:
                    break
                projection, ids, nodes = candidates[facet_id]
                local, shape, distance = projection.closest_point(point, nodes)
                # Preserve the original lowest-facet-ID tie break even when
                # the BVH reaches equal-distance projections in another order.
                if (best is None or distance < best[0]
                        or (distance == best[0] and facet_id < best[4])):
                    best = distance, ids, shape, local, facet_id
            if best is None or best[0] > cutoff:
                unmatched.append(int(node_id))
                continue
            distance, ids, shape, local, facet_id = best
            row = len(secondary_nodes)
            secondary_nodes.append(int(node_id))
            active = shape != 0
            rows.extend([row] * np.count_nonzero(active))
            main_nodes.extend(ids[active].tolist())
            weights.extend(shape[active].tolist())
            distances.append(distance)
            local_coordinates.append(local)
            main_facets.append(facet_id)

        if unmatched and not self.allow_unmatched:
            raise ValueError(
                f"Tie '{self.secondary_instance_name}.{self.secondary_surface_name}': "
                f"{len(unmatched)} secondary nodes are outside position_tolerance "
                f"{self.position_tolerance:g}; node IDs: {unmatched[:20]}.")
        # Pack variable-length stencils as COO vectors on the model device.
        # Coordinates and distances are diagnostics, not live search state.
        device = main_coordinates.device
        self._secondary_nodes = torch.tensor(secondary_nodes, dtype=torch.int64, device=device)
        self._mapping_rows = torch.tensor(rows, dtype=torch.int64, device=device)
        self._main_nodes = torch.tensor(main_nodes, dtype=torch.int64, device=device)
        self._weights = main_coordinates.new_tensor(weights)
        self._unmatched_nodes = torch.tensor(unmatched, dtype=torch.int64, device=device)
        self._projection_distances = main_coordinates.new_tensor(distances)
        self._projection_coordinates: torch.Tensor = main_coordinates.new_tensor(
            np.asarray(local_coordinates).reshape(-1, 2))
        self._main_facets: torch.Tensor = torch.tensor(main_facets, dtype=torch.int64, device=device)
