import numpy as np
import pytest
from numpy.typing import NDArray

from torchfea.utils.bvh import BoundingBoxBVH

FloatArray = NDArray[np.float64]


def brute_force_candidates(
    lower: FloatArray, upper: FloatArray, point: FloatArray, cutoff: float,
) -> list[tuple[float, int]]:
    """Independent full-scan reference for candidate membership and order."""
    gap = np.maximum(np.maximum(lower - point, point - upper), 0)
    distances = np.sqrt((gap * gap).sum(axis=1))
    return sorted(
        (float(distance), index)
        for index, distance in enumerate(distances) if distance <= cutoff
    )


@pytest.mark.parametrize("cutoff", [0., 0.1, 0.7, 3.])
def test_bvh_candidates_match_brute_force_distances_and_order(cutoff: float) -> None:
    generator = np.random.default_rng(18)
    lower = generator.uniform(-2, 2, size=(513, 3))
    upper = lower + generator.uniform(0, 1, size=(513, 3))
    # Include zero-thickness surfaces and exact boxes/corners at query points.
    upper[::3, 2] = lower[::3, 2]
    tree = BoundingBoxBVH(lower, upper)
    points = np.vstack([generator.uniform(-2, 2, size=(12, 3)), lower[:3], upper[:3]])
    for point in points:
        actual = list(tree.iter_candidates(point, cutoff))
        expected = brute_force_candidates(lower, upper, point, cutoff)
        assert [box_id for _, box_id in actual] == [box_id for _, box_id in expected]
        np.testing.assert_allclose(
            [distance for distance, _ in actual],
            [distance for distance, _ in expected], rtol=1e-14, atol=1e-14,
        )


def test_equal_centers_and_distances_have_deterministic_box_id_order() -> None:
    lower = np.full((65, 3), -1.)
    upper = np.full((65, 3), 1.)
    tree = BoundingBoxBVH(lower, upper, leaf_size=2)
    assert list(tree.iter_candidates(np.zeros(3), 0.)) == [(0., i) for i in range(65)]


def test_large_box_is_found_even_when_its_center_is_far_from_query() -> None:
    lower = np.array([[-100., -100, 0], [98, 98, 0], [200, 200, 0]])
    upper = np.array([[100., 100, 0], [99, 99, 0], [201, 201, 0]])
    tree = BoundingBoxBVH(lower, upper, leaf_size=1)
    assert list(tree.iter_candidates(np.array([99.5, 99.5, 0.]), 0.)) == [(0., 0)]


def test_large_remote_facet_does_not_force_a_scan_of_all_small_facets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The large facet's old global radius enclosed every small-facet center.
    # Count box tests instead of asserting a machine-dependent wall time.
    count = 4096
    lower = np.zeros((count + 1, 3))
    lower[:count, 0] = np.arange(count) * 2.
    upper = lower + np.array([1., 1, 0])
    lower[-1] = [10000, -5000, -5000]
    upper[-1] = [20000, 5000, 5000]
    tree = BoundingBoxBVH(lower, upper)
    examined_boxes = 0
    original = BoundingBoxBVH._box_distances

    def counted_distances(
        point: FloatArray, bounds_lower: FloatArray, bounds_upper: FloatArray,
    ) -> FloatArray:
        nonlocal examined_boxes
        examined_boxes += len(bounds_lower)
        return original(point, bounds_lower, bounds_upper)

    monkeypatch.setattr(BoundingBoxBVH, "_box_distances", staticmethod(counted_distances))
    assert list(tree.iter_candidates(np.array([0.5, 0.5, 0.]), 1e-8)) == [(0., 0)]
    assert examined_boxes < 128


def test_empty_tree_and_fixed_geometry_snapshot() -> None:
    empty = BoundingBoxBVH(np.empty((0, 3)), np.empty((0, 3)))
    assert list(empty.iter_candidates(np.zeros(3), 1.)) == []
    lower, upper = np.zeros((1, 3)), np.ones((1, 3))
    tree = BoundingBoxBVH(lower, upper)
    lower[:] = 10
    upper[:] = 11
    assert list(tree.iter_candidates(np.zeros(3), 0.)) == [(0., 0)]


@pytest.mark.parametrize("cutoff", [-1., float("nan"), float("inf")])
def test_invalid_query_cutoff_is_rejected(cutoff: float) -> None:
    tree = BoundingBoxBVH(np.zeros((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="max_distance"):
        list(tree.iter_candidates(np.zeros(3), cutoff))


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "reversed", "leaf_size"])
def test_invalid_tree_definition_is_rejected(kind: str) -> None:
    lower, upper = np.zeros((2, 3)), np.ones((2, 3))
    leaf_size = 8
    if kind == "shape":
        upper = np.ones((3, 3))
    elif kind == "nonfinite":
        upper[0, 0] = np.nan
    elif kind == "reversed":
        lower[0, 0] = 2
    else:
        leaf_size = 0
    with pytest.raises(ValueError, match="BVH"):
        BoundingBoxBVH(lower, upper, leaf_size)
