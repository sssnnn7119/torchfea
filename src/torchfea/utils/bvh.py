"""Conservative point-to-box searches for finite element surface projections."""

from __future__ import annotations

from collections.abc import Iterator
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BoundingBoxBVH:
    """Balanced AABB hierarchy with best-first point-distance queries.

    Box IDs preserve the input order. Tree nodes and leaf membership are
    stored in packed arrays; no global maximum facet radius is needed.
    Geometry is copied at construction so queries use a fixed set of bounds.
    """

    def __init__(
        self, lower_bounds: FloatArray, upper_bounds: FloatArray, leaf_size: int = 8,
    ) -> None:
        self._box_lower: FloatArray = np.array(lower_bounds, dtype=np.float64, copy=True)
        self._box_upper: FloatArray = np.array(upper_bounds, dtype=np.float64, copy=True)
        if (self._box_lower.ndim != 2 or self._box_lower.shape[1] != 3
                or self._box_lower.shape != self._box_upper.shape):
            raise ValueError("BVH bounds must be equally sized [n, 3] arrays.")
        if not (np.isfinite(self._box_lower).all() and np.isfinite(self._box_upper).all()):
            raise ValueError("BVH bounds must be finite.")
        if np.any(self._box_lower > self._box_upper):
            raise ValueError("BVH lower bounds must not exceed upper bounds.")
        if leaf_size < 1:
            raise ValueError("BVH leaf_size must be positive.")

        # Partition box IDs, not geometric data. Longest-center-axis median
        # splits keep the tree balanced even when some boxes are much larger.
        self._box_ids: IntArray = np.arange(len(self._box_lower), dtype=np.int64)
        centers = self._box_lower * 0.5 + self._box_upper * 0.5
        node_lower: list[FloatArray] = []
        node_upper: list[FloatArray] = []
        left: list[int] = []
        right: list[int] = []
        starts: list[int] = []
        stops: list[int] = []

        def build(start: int, stop: int) -> int:
            ids = self._box_ids[start:stop]
            node_id = len(node_lower)
            node_lower.append(self._box_lower[ids].min(axis=0))
            node_upper.append(self._box_upper[ids].max(axis=0))
            left.append(-1)
            right.append(-1)
            starts.append(start)
            stops.append(stop)
            if stop - start <= leaf_size:
                return node_id

            # argpartition avoids sorting every subtree. Equal centroids
            # still split by count, so coincident boxes cannot cause recursion
            # without progress; their IDs are resolved at query time.
            axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
            middle = (start + stop) // 2
            order = np.argpartition(centers[ids, axis], middle - start)
            self._box_ids[start:stop] = ids[order]
            left[node_id] = build(start, middle)
            right[node_id] = build(middle, stop)
            return node_id

        self._root: int = build(0, len(self._box_ids)) if self._box_ids.size else -1
        self._lower: FloatArray = np.asarray(node_lower, dtype=np.float64).reshape(-1, 3)
        self._upper: FloatArray = np.asarray(node_upper, dtype=np.float64).reshape(-1, 3)
        self._left: IntArray = np.asarray(left, dtype=np.int64)
        self._right: IntArray = np.asarray(right, dtype=np.int64)
        self._starts: IntArray = np.asarray(starts, dtype=np.int64)
        self._stops: IntArray = np.asarray(stops, dtype=np.int64)

    @staticmethod
    def _box_distances(
        point: FloatArray, lower: FloatArray, upper: FloatArray,
    ) -> FloatArray:
        """Euclidean distances to [n, 3] closed boxes, including planar boxes."""
        gap = np.maximum(np.maximum(lower - point, point - upper), 0)
        return np.linalg.norm(gap, axis=1)

    def iter_candidates(
        self, point: FloatArray, max_distance: float,
    ) -> Iterator[tuple[float, int]]:
        """Yield (box-distance lower bound, box ID) in increasing order.

        Only boxes within max_distance are emitted. A closest-point caller
        can stop when the next lower bound exceeds its best actual distance;
        all remaining subtrees/facets are then safely pruned as well.
        """
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("BVH query point must be a finite 3-vector.")
        if not np.isfinite(max_distance) or max_distance < 0:
            raise ValueError("BVH max_distance must be finite and nonnegative.")
        if self._root < 0:
            return

        # A mixed heap of tree nodes and individual boxes gives globally
        # ordered lower bounds, rather than scanning every box in the radius.
        # False marks tree nodes; True marks facet boxes. At equal distance
        # expand nodes before facets, then emit equal-distance facet IDs in order.
        root_distance = float(self._box_distances(point, self._lower[:1], self._upper[:1])[0])
        if root_distance > max_distance:
            return
        pending: list[tuple[float, bool, int]] = [(root_distance, False, self._root)]
        while pending:
            distance, is_box, identifier = heappop(pending)
            if is_box:
                yield distance, identifier
                continue

            if self._left[identifier] < 0:
                # Test individual Bernstein boxes only after their entire
                # ancestor chain has survived the distance cutoff.
                ids = self._box_ids[self._starts[identifier]:self._stops[identifier]]
                distances = self._box_distances(point, self._box_lower[ids], self._box_upper[ids])
                for box_id, box_distance in zip(ids, distances):
                    if box_distance <= max_distance:
                        heappush(pending, (float(box_distance), True, int(box_id)))
                continue

            # Reject a whole subtree using its union box; a single large
            # facet only affects its own ancestors, not every query radius.
            children = np.array([self._left[identifier], self._right[identifier]], dtype=np.int64)
            distances = self._box_distances(point, self._lower[children], self._upper[children])
            for child, child_distance in zip(children, distances):
                if child_distance <= max_distance:
                    heappush(pending, (float(child_distance), False, int(child)))
