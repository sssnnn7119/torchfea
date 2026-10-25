from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from torchfea.cad import CADHistory, StepImport, export_step
from torchfea.ui.model import CAEProject
from torchfea.ui.workbench import Workbench


def workbench_state(project: CAEProject) -> tuple[Workbench, list[str]]:
    """Exercise feature edits without starting the viewport or Qt event loop."""
    changes: list[str] = []
    state = SimpleNamespace(
        project=project, _active_tree_data=("geometry", 0), _changed=changes.append,
    )
    return cast(Workbench, state), changes


def test_step_import_appends_features_without_creating_parts_or_instances(tmp_path: Path) -> None:
    source = CADHistory()
    source.add_box(dx=1, dy=2, dz=3)
    step_path = tmp_path / "insert.stp"
    export_step(source, str(step_path))
    project = CAEProject()
    original = project.parts[0].geometry.operations[0]
    state, changes = workbench_state(project)

    # Two imports remain distinct features of the existing Part; both edits
    # must enter the normal history/mesh-invalidation/preview update path.
    Workbench.add_step_feature(state, 0, str(step_path))
    Workbench.add_step_feature(state, 0, str(step_path))
    assert len(project.parts) == len(project.instances) == 1
    operations = project.parts[0].geometry.operations
    assert operations[0] is original
    assert len(operations) == 3
    assert all(isinstance(operation, StepImport) for operation in operations[1:])
    assert operations[1].id != operations[2].id
    assert state._active_tree_data == ("feature", 0, operations[-1].id)
    assert len(changes) == 2


@pytest.mark.parametrize("exists", [False, True])
def test_failed_step_read_leaves_part_history_unchanged(tmp_path: Path, exists: bool) -> None:
    path = tmp_path / "invalid.stp"
    if exists:
        path.touch()
    project = CAEProject()
    before = project.to_dict()
    state, changes = workbench_state(project)
    with pytest.raises(ValueError):
        Workbench.add_step_feature(state, 0, str(path))
    assert project.to_dict() == before
    assert state._active_tree_data == ("geometry", 0)
    assert not changes


def test_discrete_part_does_not_accept_cad_features(tmp_path: Path) -> None:
    project = CAEProject()
    project.parts[0].geometry.source = "discrete"
    before = project.to_dict()
    state, changes = workbench_state(project)
    with pytest.raises(ValueError, match="CAD"):
        Workbench.add_step_feature(state, 0, str(tmp_path / "unused.stp"))
    assert project.to_dict() == before
    assert not changes
