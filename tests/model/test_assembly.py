import pytest

from torchfea.model.assembly import Assembly


class BlockConstraint:
    def __init__(self, inputs=(), outputs=()):
        self.enabled = True
        self._inputs = set(inputs)
        self._outputs = set(outputs)

    def get_input_blocks(self):
        return self._inputs

    def get_output_blocks(self):
        return self._outputs


def test_constraint_plan_is_topologically_sorted_and_reversed_for_reduction():
    assembly = Assembly()
    joint = BlockConstraint(inputs={1}, outputs={2})
    couple = BlockConstraint(inputs={2}, outputs={3})

    # Deliberately insert the consumer first.  The dependency plan must not
    # depend on dictionary insertion order.
    assembly._constraints = {
        "couple": couple,
        "joint": joint,
    }

    assembly._build_constraint_execution_plan()

    assert assembly._constraints_forward == [joint, couple]
    assert assembly._constraints_backward == [couple, joint]


def test_constraint_plan_rejects_cycles():
    assembly = Assembly()
    assembly._constraints = {
        "first": BlockConstraint(inputs={2}, outputs={1}),
        "second": BlockConstraint(inputs={1}, outputs={2}),
    }

    with pytest.raises(ValueError, match="dependency cycle"):
        assembly._build_constraint_execution_plan()
