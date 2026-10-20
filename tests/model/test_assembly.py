from collections.abc import Iterable

import numpy as np
import pytest
from numpy.typing import NDArray

from torchfea.model.assembly import Assembly
from torchfea.model.constraints import BaseConstraint


class DofConstraint(BaseConstraint):
    def __init__(self, inputs: Iterable[int] = (), outputs: Iterable[int] = ()) -> None:
        super().__init__()
        self._inputs: NDArray[np.int64] = np.fromiter(inputs, dtype=np.int64)
        self._outputs: NDArray[np.int64] = np.fromiter(outputs, dtype=np.int64)

    def get_input_dofs(self) -> NDArray[np.int64]:
        return self._inputs

    def get_output_dofs(self) -> NDArray[np.int64]:
        return self._outputs


def test_constraint_plan_is_topologically_sorted_and_reversed_for_reduction() -> None:
    assembly = Assembly()
    producer = DofConstraint(inputs={1}, outputs={2})
    consumer = DofConstraint(inputs={2}, outputs={3})

    # Insert and name the consumer first to verify that dependencies take
    # precedence over both insertion and alphabetical order.
    assembly._constraints = {
        "a-consumer": consumer,
        "z-producer": producer,
    }

    assembly._build_constraint_execution_plan()

    assert assembly._constraints_forward == [producer, consumer]
    assert assembly._constraints_backward == [consumer, producer]


def test_constraint_plan_rejects_cycles() -> None:
    assembly = Assembly()
    assembly._constraints = {
        "first": DofConstraint(inputs={2}, outputs={1}),
        "second": DofConstraint(inputs={1}, outputs={2}),
    }

    with pytest.raises(ValueError, match="dependency cycle"):
        assembly._build_constraint_execution_plan()


def test_disjoint_dofs_on_one_instance_remain_independent() -> None:
    assembly = Assembly()
    first = DofConstraint(inputs={0, 1, 2}, outputs={3, 4, 5})
    second = DofConstraint(inputs={6, 7, 8}, outputs={9, 10, 11})
    assembly._constraints = {"first": first, "second": second}
    assembly._build_constraint_execution_plan()
    assert assembly._constraints_forward == [first, second]


def test_repeated_dofs_sparse_indices_and_unowned_inputs() -> None:
    assembly = Assembly()
    # Repeated coefficients describe one writer, while unowned coordinates
    # and gaps in global numbering must not create spurious dependencies.
    producer = DofConstraint(outputs=[10**12, 7, 7])
    consumer = DofConstraint(inputs=[7, 7, 8, 10**12, 10**12 + 1], outputs=[12])
    assembly._constraints = {"a-consumer": consumer, "z-producer": producer}
    assembly._build_constraint_execution_plan()
    assert assembly._constraints_forward == [producer, consumer]


def test_duplicate_writers_are_rejected_despite_repeated_local_indices() -> None:
    assembly = Assembly()
    assembly._constraints = {
        "first": DofConstraint(outputs=[7, 7, 4]),
        "second": DofConstraint(outputs=[8, 7]),
    }
    with pytest.raises(ValueError, match="DoF 7.*'first' and 'second'"):
        assembly._build_constraint_execution_plan()


def test_empty_outputs_and_disabled_constraints_do_not_add_dependencies() -> None:
    assembly = Assembly()
    assembly._build_constraint_execution_plan()
    assert assembly._constraints_forward == []
    disabled = DofConstraint(inputs=[2], outputs=[1])
    disabled.enabled = False
    active = DofConstraint(inputs=[1, 2])
    assembly._constraints = {"disabled": disabled, "active": active}
    assembly._build_constraint_execution_plan()
    assert assembly._constraints_forward == [active]


@pytest.mark.parametrize("seed", range(5))
def test_vectorized_plan_matches_set_based_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = 20
    outputs = rng.permutation(count * 12).reshape(count, 12).astype(np.int64) * 13
    inputs = rng.integers(0, count * 12, size=(count, 30), dtype=np.int64) * 13
    # Retain only sources from earlier producers to form an acyclic graph.
    owner = {int(dof): index for index, dofs in enumerate(outputs) for dof in dofs}
    inputs_by_constraint = [
        dofs[[owner[int(dof)] < index for dof in dofs]]
        for index, dofs in enumerate(inputs)
    ]
    names = [f"constraint-{index:02}" for index in rng.permutation(count)]
    assembly = Assembly()
    assembly._constraints = {
        name: DofConstraint(inputs_by_constraint[int(name[-2:])], outputs[int(name[-2:])])
        for name in names
    }

    # Use scalar set intersections as an independent reference for the
    # deterministic topological order, including branches and shared inputs.
    remaining = set(names)
    expected: list[BaseConstraint] = []
    while remaining:
        ready = sorted(
            name for name in remaining
            if not any(
                set(assembly._constraints[source].get_output_dofs())
                & set(assembly._constraints[name].get_input_dofs())
                for source in remaining if source != name
            )
        )
        assert ready
        name = ready[0]
        expected.append(assembly._constraints[name])
        remaining.remove(name)
    assembly._build_constraint_execution_plan()
    assert assembly._constraints_forward == expected
    assert assembly._constraints_backward == expected[::-1]
