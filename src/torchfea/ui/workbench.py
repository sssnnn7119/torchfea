"""CAE definition workbench: model tree, property editor, and mesh preview."""

from __future__ import annotations

import copy
import numpy as np
from types import SimpleNamespace

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QInputDialog, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSplitter, QSpinBox,
    QTabBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .i18n import T
from .model import (
    BoundaryDefinition, CAEProject, InstanceDefinition, LoadDefinition,
    MaterialDefinition, NamedSet, PartDefinition, ReferencePointDefinition, build_controller,
    auto_node_set_names, parse_indices, surface_node_set_name,
)
from .model import CADOperation
from .gmsh_cad import create_part, preview_brep, primitive_defaults, transform_defaults
from .logging_utils import LOGGER, log_path
from .postprocess import VTKMeshViewer


class Workbench(QWidget):
    """Abaqus-style model tree backed by :class:`CAEProject`."""

    changed = Signal()
    runRequested = Signal(object)
    notify = Signal(str)

    def __init__(self, project: CAEProject | None = None, parent=None):
        super().__init__(parent)
        self.project = project or CAEProject()
        self._preview_controller = None
        self._mesh_parts: dict[str, object] = {}
        self._assembly_mesh_controller = None
        self._brep_part_records: dict[str, list[dict]] = {}
        self._brep_assembly_records: list[dict] = []
        self._preview_highlight: dict | None = None
        self._view_mode = "assembly_geometry"
        self._pick_target: tuple[int, str, int] | None = None
        self._active_tree_data = ("project", None)
        self._history: list[dict] = [self.project.to_dict()]
        self._history_index = 0
        # Mesh generation may be triggered by several related field updates.
        # Coalescing them avoids rebuilding a Gmsh model for every spin-box
        # intermediate value while still keeping the preview live.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._refresh_preview_automatically)
        self._build_ui()
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self._undo_shortcut.activated.connect(self.undo)
        self._redo_shortcut.activated.connect(self.redo)
        self.reload()
        self._preview_timer.start()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self._buttons: list[QPushButton] = []
        for text, slot in [
            ("+ " + T("零件", "Part"), self.add_part),
            ("+ " + T("实例", "Instance"), self.add_instance),
            ("+ " + T("参考点", "Reference Point"), self.add_reference_point),
            ("+ " + T("边界", "Boundary"), self.add_boundary),
            ("+ " + T("载荷", "Load"), self.add_load),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            actions.addWidget(button)
            self._buttons.append(button)
        undo = QPushButton("↶ " + T("撤回", "Undo")); undo.clicked.connect(self.undo)
        redo = QPushButton("↷ " + T("重做", "Redo")); redo.clicked.connect(self.redo)
        actions.addWidget(undo); actions.addWidget(redo)
        merge = QPushButton(T("Part 并集合并", "Union Parts")); merge.clicked.connect(self.merge_parts)
        actions.addWidget(merge)
        actions.addStretch(1)
        self.mesh_size = self._number(self.project.mesh_size, 1e-9, 1e12)
        self.mesh_size.setFixedWidth(90)
        self.mesh_size.editingFinished.connect(self._save_global_mesh_size)
        actions.addWidget(QLabel(T("统一网格尺寸", "Global mesh size")))
        actions.addWidget(self.mesh_size)
        self.mesh_button = QPushButton(T("生成 FE 网格", "Generate FE mesh"))
        self.mesh_button.clicked.connect(self.show_mesh)
        actions.addWidget(self.mesh_button)
        self.preview_button = QPushButton(T("更新预览", "Update preview"))
        self.preview_button.clicked.connect(self.refresh_preview)
        actions.addWidget(self.preview_button)
        outer.addLayout(actions)

        split = QSplitter(Qt.Orientation.Horizontal)
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QHBoxLayout()
        self.tree_filter = QLineEdit()
        self.tree_filter.setPlaceholderText(T("过滤模型树…", "Filter model tree…"))
        self.tree_filter.setClearButtonEnabled(True)
        self.tree_filter.textChanged.connect(self._filter_tree)
        filter_row.addWidget(self.tree_filter)
        tree_layout.addLayout(filter_row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel(T("CAE 模型", "CAE model"))
        self.tree.setMinimumWidth(220)
        self.tree.itemSelectionChanged.connect(self._selected)
        tree_layout.addWidget(self.tree, 1)
        split.addWidget(tree_panel)

        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor = QWidget()
        self.editor_scroll.setWidget(self.editor)
        split.addWidget(self.editor_scroll)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.view_tabs = QTabBar()
        self.view_tabs.addTab(T("Part 几何", "Part Geometry"))
        self.view_tabs.addTab(T("Part 网格", "Part Mesh"))
        self.view_tabs.addTab(T("装配体几何", "Assembly Geometry"))
        self.view_tabs.addTab(T("装配体网格", "Assembly Mesh"))
        self.view_tabs.setExpanding(False)
        self.view_tabs.currentChanged.connect(self._view_tab_changed)
        preview_layout.addWidget(self.view_tabs)
        view_actions = QHBoxLayout()
        view_actions.addWidget(QLabel(T("视图", "View")))
        self.preview_mode = QLabel(T("BREP", "BREP"))
        self.preview_mode.setStyleSheet("color:#86b8ee; font-weight:600")
        view_actions.addWidget(self.preview_mode)
        fit = QPushButton(T("充满视图", "Fit view")); fit.setToolTip(T("将所有可见几何居中并缩放到视图内", "Center and scale all visible geometry to the view"))
        view_actions.addWidget(fit)
        for text, mode in [
            (T("前", "Front"), "front"), (T("后", "Back"), "back"),
            (T("左", "Left"), "left"), (T("右", "Right"), "right"),
            (T("顶", "Top"), "top"), (T("底", "Bottom"), "bottom"),
            (T("等轴", "Iso"), "iso"),
        ]:
            button = QPushButton(text)
            button.setToolTip(T(f"切换到{text}视角", f"Switch to {text} view"))
            button.setFixedWidth(48)
            button.clicked.connect(lambda _checked=False, view=mode: self.preview.set_standard_view(view))
            view_actions.addWidget(button)
        wireframe = QCheckBox(T("线框", "Wireframe"))
        edges = QCheckBox(T("边线", "Edges")); edges.setChecked(True)
        view_actions.addWidget(wireframe); view_actions.addWidget(edges); view_actions.addStretch(1)
        preview_layout.addLayout(view_actions)
        self.preview = VTKMeshViewer()
        fit.clicked.connect(self.preview.reset_camera)
        wireframe.toggled.connect(self.preview.set_wireframe)
        edges.toggled.connect(self.preview.set_edge_visibility)
        self.preview.entityPicked.connect(self._picked_entity)
        preview_layout.addWidget(self.preview, 1)
        # Start with the complete assembly BREP, matching the conventional
        # CAE landing view; the other three products remain one click away.
        self.view_tabs.setCurrentIndex(2)
        split.addWidget(preview_panel)
        split.setSizes([240, 460, 620])
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        outer.addWidget(split, 1)

        foot = QHBoxLayout()
        self.message = QLabel(T("就绪", "Ready"))
        self.message.setStyleSheet("color:#9eabba")
        self.message.setToolTip(T(f"详细日志：{log_path()}", f"Detailed log: {log_path()}"))
        foot.addWidget(self.message, 1)
        log_button = QPushButton(T("日志", "Log"))
        log_button.setToolTip(T("显示详细日志文件位置", "Show detailed log file location"))
        log_button.clicked.connect(self._show_log_path)
        foot.addWidget(log_button)
        self.submit = QPushButton("▶ " + T("提交计算 / 后处理", "Submit for solve / postprocess"))
        self.submit.setStyleSheet("background:#00796b; font-weight:600; padding:7px 18px")
        self.submit.clicked.connect(lambda: self.runRequested.emit(self.project))
        foot.addWidget(self.submit)
        outer.addLayout(foot)
        self._update_view_tabs()

    def set_project(self, project: CAEProject) -> None:
        """Replace the current definition and reset its reversible history."""
        self.project = project
        self._history = [project.to_dict()]
        self._history_index = 0
        self._active_tree_data = ("project", None)
        self._invalidate_mesh_cache()
        self.mesh_size.setValue(project.mesh_size)
        self.reload()
        self._preview_timer.start()

    # ---------------------------------------------------------- tree model
    def reload(self, select_data=None) -> None:
        state = self._tree_state()
        if select_data is None:
            select_data = self._active_tree_data
        self.tree.blockSignals(True)
        self.tree.clear()
        root = self._node(T("模型", "Model"), ("project", None))
        root.setExpanded(True)
        parts = self._node(T("零件", "Parts"), ("parts", None), root)
        parts.setExpanded(True)
        for index, part in enumerate(self.project.parts):
            item = self._node(part.name, ("part", index), parts)
            geometry = self._node(T("几何 / 网格", "Geometry / mesh"), ("geometry", index), item)
            # Keep the OCC history directly under the geometry node.  A
            # second history container added no behavior and only made every
            # feature take one extra click to reach.
            for operation in part.geometry.ensure_operations():
                inputs = f" ← {', '.join(operation.inputs)}" if operation.inputs else ""
                self._node(f"{operation.id} · {self._operation_name(operation.kind)}{inputs}",
                           ("feature", index, operation.id), geometry)
            materials = self._node(T("材料", "Materials"), ("materials", index), item)
            for material_index, material in enumerate(part.materials):
                self._node(
                    f"{material.name} · {self._material_name(material.kind)}",
                    ("material", index, material_index), materials)
            self._sets_branch(
                item, T("节点集", "Node sets"), "node_sets", index, part.node_sets,
                [("all", T("全部节点", "All nodes")),
                 ("surface_all", T("所有表面节点", "All surface nodes")),
                 *[(surface_node_set_name(surface.name),
                    f"{surface.name} · {T('表面节点', 'Surface nodes')}")
                   for surface in part.surface_sets]])
            self._sets_branch(
                item, T("单元集", "Element sets"), "element_sets", index, part.element_sets,
                [("all", T("全部单元", "All elements"))])
            surfaces = self._node(T("表面", "Surfaces"), ("surface_sets", index), item)
            self._node(
                f"{part.exterior_surface} [{T('外表面', 'Exterior')}]",
                ("exterior_surface", index), surfaces)
            for set_index, surface_set in enumerate(part.surface_sets):
                self._node(surface_set.name, ("set", index, "surface_sets", set_index), surfaces)
        assembly = self._node(T("装配体", "Assembly"), ("assembly", None), root)
        assembly.setExpanded(True)
        instances = self._node(T("实例", "Instances"), ("instances", None), assembly)
        for index, item in enumerate(self.project.instances):
            self._node(item.name, ("instance", index), instances)
        rps = self._node(T("参考点", "Reference points"), ("rps", None), assembly)
        for index, item in enumerate(self.project.reference_points):
            self._node(item.name, ("rp", index), rps)
        loads = self._node(T("载荷", "Loads"), ("loads", None), root)
        for index, item in enumerate(self.project.loads):
            self._node(f"{item.name} [{item.kind}]", ("load", index), loads)
        bcs = self._node(T("边界条件", "Boundary conditions"), ("boundaries", None), root)
        for index, item in enumerate(self.project.boundaries):
            self._node(item.name, ("boundary", index), bcs)
        self._node(T("求解器", "Solver"), ("solver", None), root)
        self._restore_tree_state(state)
        self.tree.blockSignals(False)
        # Reapply an active search after rebuilding the tree; otherwise an
        # Apply/undo operation would unexpectedly reveal every node again.
        self._filter_tree(self.tree_filter.text())
        selected = self._find_tree_data(select_data)
        if selected is not None:
            self.tree.setCurrentItem(selected)
            self._selected()
        else:
            self._set_editor(self._project_editor())

    def _tree_state(self) -> dict[tuple, bool]:
        state = {}
        def walk(item):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data is not None: state[tuple(data)] = item.isExpanded()
            for row in range(item.childCount()): walk(item.child(row))
        for row in range(self.tree.topLevelItemCount()): walk(self.tree.topLevelItem(row))
        return state

    def _restore_tree_state(self, state) -> None:
        def walk(item, depth=0):
            data = tuple(item.data(0, Qt.ItemDataRole.UserRole))
            # Keep the working branches open by default so CAD features and
            # Part-owned materials are immediately visible. Other branches
            # retain the user's state.
            default_expanded = depth < 2 or data[0] in {"geometry", "materials"}
            item.setExpanded(state.get(data, default_expanded))
            for row in range(item.childCount()): walk(item.child(row), depth + 1)
        for row in range(self.tree.topLevelItemCount()): walk(self.tree.topLevelItem(row))

    def _find_tree_data(self, target):
        if target is None: return None
        target = tuple(target)
        def walk(item):
            if tuple(item.data(0, Qt.ItemDataRole.UserRole)) == target: return item
            for row in range(item.childCount()):
                found = walk(item.child(row))
                if found is not None: return found
            return None
        for row in range(self.tree.topLevelItemCount()):
            found = walk(self.tree.topLevelItem(row))
            if found is not None: return found
        return None

    def _filter_tree(self, text: str) -> None:
        """Keep matching nodes and their ancestors visible in the model tree."""
        query = text.strip().casefold()
        def visit(item) -> bool:
            own = not query or query in item.text(0).casefold()
            child_match = any(visit(item.child(row)) for row in range(item.childCount()))
            visible = own or child_match
            # Keep the model root visible even when a filter has no matches;
            # this leaves the filter field usable instead of hiding the tree.
            item.setHidden(bool(query) and not visible and item.parent() is not None)
            if query and child_match:
                item.setExpanded(True)
            return visible
        for row in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(row))

    # ------------------------------------------------------------ view modes
    def _selected_part_index(self) -> int | None:
        data = tuple(self._active_tree_data or ())
        if len(data) > 1 and data[0] in {
            "part", "geometry", "feature", "materials", "material", "node_sets", "element_sets", "surface_sets",
            "exterior_surface", "set", "auto_set"
        }:
            return int(data[1])
        return 0 if self.project.parts else None

    def _view_tab_changed(self, index: int) -> None:
        modes = ("part_geometry", "part_mesh", "assembly_geometry", "assembly_mesh")
        if not 0 <= index < len(modes):
            return
        self._view_mode = modes[index]
        LOGGER.info("viewport mode changed: %s", self._view_mode)
        self._update_view_tabs()
        self._render_current_view()

    def _update_view_tabs(self) -> None:
        if not hasattr(self, "view_tabs"):
            return
        part_index = self._selected_part_index()
        part_available = bool(part_index is not None and 0 <= part_index < len(self.project.parts) and
                              self.project.parts[part_index].name in self._mesh_parts)
        all_available = bool(self.project.parts) and bool(self.project.instances) and all(
            part.name in self._mesh_parts for part in self.project.parts)
        self.view_tabs.setTabEnabled(1, part_available)
        self.view_tabs.setTabEnabled(3, all_available)
        current = self.view_tabs.currentIndex()
        if current == 1 and not part_available:
            self.view_tabs.setCurrentIndex(0)
        elif current == 3 and not all_available:
            self.view_tabs.setCurrentIndex(2)

    def _render_current_view(self) -> None:
        """Render exactly the selected one of the four CAE view products."""
        if not self._brep_part_records and self._view_mode in {"part_geometry", "assembly_geometry"}:
            self.preview.clear_scene()
            return
        part_index = self._selected_part_index()
        if part_index is None:
            self.preview.clear_scene()
            return
        if not 0 <= part_index < len(self.project.parts):
            raise IndexError(f"Selected Part index is out of range: {part_index}")
        part = self.project.parts[part_index]
        if self._view_mode == "part_geometry":
            if part.name not in self._brep_part_records:
                raise RuntimeError(f"BREP cache is missing Part: {part.name}")
            self.preview.show_brep(self._brep_part_records[part.name])
            self.preview.highlight_entity(self._preview_highlight)
            self.preview_mode.setText(T("Part 几何", "Part Geometry"))
        elif self._view_mode == "assembly_geometry":
            if not self._brep_assembly_records:
                raise RuntimeError("BREP cache is missing assembly geometry.")
            self.preview.show_brep(
                self._brep_assembly_records,
                reference_points=self._reference_point_records())
            self.preview.highlight_entity(self._preview_highlight)
            self.preview_mode.setText(T("装配体几何", "Assembly Geometry"))
        elif self._view_mode == "part_mesh":
            mesh = self._mesh_parts.get(part.name)
            if mesh is None:
                LOGGER.warning("Part mesh view requested before mesh exists: part=%s", part.name)
                self.preview.clear_scene()
                return
            self.preview.show_part(mesh, part.name, f"{part.name}-mesh")
            self.preview_mode.setText(T("Part 网格", "Part Mesh"))
        elif self._view_mode == "assembly_mesh":
            if not all(part_item.name in self._mesh_parts for part_item in self.project.parts):
                LOGGER.warning("assembly mesh view requested before all Parts are meshed")
                self.preview.clear_scene()
                return
            if self._assembly_mesh_controller is None:
                LOGGER.info("building assembly mesh view from cached Part meshes")
                self._assembly_mesh_controller = self._cached_assembly_view()
            self.preview.show_controller(self._assembly_mesh_controller, brep=False)
            self.preview_mode.setText(T("装配体网格", "Assembly Mesh"))

    def _cached_assembly_view(self):
        """Create a render-only assembly from already generated Part meshes."""
        from torchfea.model import Assembly, Instance, ReferencePoint

        assembly = Assembly()
        for definition in self.project.parts:
            mesh = self._mesh_parts.get(definition.name)
            if mesh is None:
                raise RuntimeError(f"Part mesh is missing: {definition.name}")
            assembly.add_part(mesh, definition.name)
        for definition in self.project.instances:
            assembly.add_instance(Instance(
                definition.part_name, definition.translation, definition.rotation), definition.name)
        for definition in self.project.reference_points:
            assembly.add_reference_point(
                ReferencePoint(definition.point), definition.name)
        return SimpleNamespace(assembly=assembly)

    def _reference_point_records(self):
        return [
            {"name": definition.name, "point": list(definition.point)}
            for definition in self.project.reference_points
        ]

    def _invalidate_mesh_cache(self, clear_brep: bool = True) -> None:
        if self._mesh_parts or self._assembly_mesh_controller is not None:
            LOGGER.info("invalidating mesh view cache")
        self._mesh_parts = {}
        self._preview_controller = None
        self._assembly_mesh_controller = None
        if clear_brep:
            self._brep_part_records = {}
            self._brep_assembly_records = []
            self._preview_highlight = None
        if hasattr(self, "preview"):
            # Never leave a mesh or a pre-edit BREP visible after the model
            # definition has changed.  A fresh preview is scheduled by the
            # caller; until it succeeds the viewport must be empty.
            if clear_brep or self._view_mode in {"part_mesh", "assembly_mesh"}:
                self.preview.clear_scene()
        if hasattr(self, "view_tabs"):
            self._update_view_tabs()

    def _mesh_part(self, part_index: int) -> None:
        definition = self.project.parts[part_index]
        self._mesh_parts.pop(definition.name, None)
        self._preview_controller = None
        self._assembly_mesh_controller = None
        if self._view_mode in {"part_mesh", "assembly_mesh"}:
            self.preview.clear_scene()
        self._update_view_tabs()
        try:
            LOGGER.info("Part mesh requested: part=%s mesh_size=%r", definition.name, self.project.mesh_size)
            self.message.setText(T(f"正在剖分 {definition.name}…", f"Meshing {definition.name}…"))
            self._mesh_parts[definition.name] = create_part(
                definition.geometry.ensure_operations(), self.project.mesh_size)
            self._update_view_tabs()
            self.view_tabs.setCurrentIndex(1)
            self._render_current_view()
            self.message.setText(T(f"{definition.name} 网格已生成", f"Mesh generated for {definition.name}"))
        except Exception as exc:
            LOGGER.exception("Part mesh failed: part=%s", definition.name)
            self.message.setText(T("剖分失败：", "Meshing failed: ") + str(exc))
            QMessageBox.warning(self, T("Part 剖分失败", "Part meshing failed"),
                                T(f"{exc}\n\n详细日志：{log_path()}",
                                  f"{exc}\n\nDetailed log: {log_path()}"))

    def _sets_branch(self, parent, caption, kind, part_index, items, auto_items=()):
        branch = self._node(caption, (kind, part_index), parent)
        for name, label in auto_items:
            auto_item = self._node(
                f"{name} [{T('自动', 'Auto')}]", ("auto_set", part_index, kind, name), branch)
            auto_item.setToolTip(0, label)
        for index, item in enumerate(items):
            self._node(item.name, ("set", part_index, kind, index), branch)

    def _node(self, text, data, parent=None):
        item = QTreeWidgetItem([text])
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    @staticmethod
    def _operation_name(kind: str) -> str:
        names = {
            "fuse": T("并集", "Union"),
            "cut": T("差集", "Difference"),
            "intersect": T("交集", "Intersection"),
            "fragment": T("分割", "Fragment"),
            "translate": T("平移", "Translate"),
            "rotate": T("旋转", "Rotate"),
            "dilate": T("缩放", "Scale"),
            "mirror": T("镜像", "Mirror"),
        }
        return names.get(kind, kind)

    @staticmethod
    def _material_name(kind: str) -> str:
        names = {
            "linear_elastic": T("线弹性", "Linear elastic"),
            "neo_hookean": T("Neo-Hookean", "Neo-Hookean"),
            "neo_hookean_lnj": T("Neo-Hookean lnJ", "Neo-Hookean lnJ"),
        }
        return names.get(kind, kind)

    def _selected(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        self._active_tree_data = tuple(data)
        kind = data[0]
        if kind == "exterior_surface":
            # The Part exterior is an aggregate over all BREP child faces;
            # the viewer will highlight every matching face in red.
            self._preview_highlight = {
                "part_name": self.project.parts[data[1]].name,
                "surface_all": True,
            }
            if self.view_tabs.currentIndex() != 0:
                self.view_tabs.setCurrentIndex(0)
        elif kind == "rp":
            # Reference points are assembly objects; switch to the assembly
            # geometry view so their world coordinates are immediately visible.
            self._preview_highlight = None
            if self.view_tabs.currentIndex() != 2:
                self.view_tabs.setCurrentIndex(2)
        else:
            self._preview_highlight = None
        if kind == "project": editor = self._project_editor()
        elif kind == "part": editor = self._part_editor(data[1])
        elif kind == "geometry": editor = self._geometry_editor(data[1])
        elif kind == "feature":
            editor = self._geometry_editor(data[1])
        elif kind == "materials": editor = self._materials_editor(data[1])
        elif kind == "material": editor = self._material_editor(data[1], data[2])
        elif kind in {"node_sets", "element_sets", "surface_sets"}: editor = self._sets_editor(data[1], kind)
        elif kind == "auto_set": editor = self._auto_set_editor(data[1], data[2], data[3])
        elif kind == "set": editor = self._set_editor_form(*data[1:])
        elif kind == "exterior_surface": editor = self._exterior_surface_editor(data[1])
        elif kind == "instance": editor = self._instance_editor(data[1])
        elif kind == "rp": editor = self._rp_editor(data[1])
        elif kind == "boundary": editor = self._boundary_editor(data[1])
        elif kind == "load": editor = self._load_editor(data[1])
        elif kind == "solver": editor = self._solver_editor()
        else: editor = self._hint_editor(T("选择下方对象进行编辑。", "Select an item below to edit it."))
        self._set_editor(editor)
        self._update_view_tabs()
        if self._brep_part_records:
            self._render_current_view()

    # ---------------------------------------------------------- editor bits
    def _set_editor(self, widget: QWidget) -> None:
        old = self.editor_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.editor = widget
        self.editor_scroll.setWidget(widget)

    @staticmethod
    def _page(title: str, description: str = ""):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(title)
        label.setStyleSheet("font-size:18px; font-weight:600; color:#86b8ee")
        layout.addWidget(label)
        if description:
            hint = QLabel(description)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#aab5c5")
            layout.addWidget(hint)
        form = QFormLayout()
        layout.addLayout(form)
        return page, layout, form

    def _project_editor(self):
        page, layout, form = self._page(T("有限元模型", "Finite element model"))
        label = QLineEdit(self.project.label)
        form.addRow(T("模型名称", "Model label"), label)
        button = QPushButton(T("保存名称", "Save label"))
        button.clicked.connect(lambda: self._save_project_label(label.text()))
        layout.addWidget(button)
        layout.addStretch(1)
        return page

    def _part_editor(self, index: int):
        part = self.project.parts[index]
        page, layout, form = self._page(T("零件定义", "Part definition"))
        name = QLineEdit(part.name)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("材料数", "Materials"), QLabel(str(len(part.materials))))
        save = QPushButton(T("应用零件属性", "Apply part properties"))
        save.clicked.connect(lambda: self._save_part(index, name.text()))
        delete = QPushButton(T("删除零件", "Delete part")); delete.setStyleSheet("background:#9d3434")
        delete.clicked.connect(lambda: self._delete("part", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _materials_editor(self, part_index: int):
        part = self.project.parts[part_index]
        page, layout, form = self._page(T("材料", "Materials"))
        density = self._number(part.density, 1e-15, 1e15, 12)
        form.addRow(T("密度", "Density"), density)
        save_properties = QPushButton(T("应用材料属性", "Apply material properties"))
        save_properties.clicked.connect(
            lambda: self._save_material_properties(part_index, density.value()))
        layout.addWidget(save_properties)
        add = QPushButton("+ " + T("添加材料", "Add material"))
        add.clicked.connect(lambda: self.add_material(part_index))
        layout.addWidget(add)
        for material in part.materials:
            assignment = f" · {material.element_set}"
            layout.addWidget(QLabel(
                f"• {material.name} · {self._material_name(material.kind)}{assignment}"))
        layout.addStretch(1)
        return page

    def _material_editor(self, part_index: int, material_index: int):
        part = self.project.parts[part_index]
        material = part.materials[material_index]
        page, layout, form = self._page(T("材料定义", "Material definition"))
        name = QLineEdit(material.name)
        kind = QComboBox()
        for label, value in [
            (T("线弹性", "Linear elastic"), "linear_elastic"),
            (T("Neo-Hookean", "Neo-Hookean"), "neo_hookean"),
            (T("Neo-Hookean lnJ", "Neo-Hookean lnJ"), "neo_hookean_lnj"),
        ]:
            kind.addItem(label, value)
        kind.setCurrentIndex(max(0, kind.findData(material.kind)))
        element_set = QComboBox()
        element_set.addItem("all")
        for item in part.element_sets:
            if item.name != "all":
                element_set.addItem(item.name)
        element_set.setCurrentText(material.element_set)
        young = self._number(material.young, 1e-12, 1e15)
        poisson = self._number(material.poisson, 0.0, .499)
        mu = self._number(material.mu, 1e-12, 1e15)
        kappa = self._number(material.kappa, 1e-12, 1e15)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("类型", "Type"), kind)
        form.addRow(T("单元集", "Element set"), element_set)
        young_row = form.rowCount()
        form.addRow(T("杨氏模量", "Young's modulus"), young)
        poisson_row = form.rowCount()
        form.addRow(T("泊松比", "Poisson ratio"), poisson)
        mu_row = form.rowCount()
        form.addRow("mu", mu)
        kappa_row = form.rowCount()
        form.addRow("kappa", kappa)

        def update_fields(*_):
            linear = kind.currentData() == "linear_elastic"
            form.setRowVisible(young_row, linear)
            form.setRowVisible(poisson_row, linear)
            form.setRowVisible(mu_row, not linear)
            form.setRowVisible(kappa_row, not linear)

        kind.currentIndexChanged.connect(update_fields)
        update_fields()
        save = QPushButton(T("应用材料", "Apply material"))
        save.clicked.connect(lambda: self._save_material(
            part_index, material_index, name.text(), kind.currentData(),
            element_set.currentText(), young.value(), poisson.value(), mu.value(), kappa.value()))
        delete = QPushButton(T("删除材料", "Delete material"))
        delete.setStyleSheet("background:#9d3434")
        delete.clicked.connect(lambda: self._delete_material(part_index, material_index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _geometry_editor(self, index: int):
        geometry = self.project.parts[index].geometry
        operations = geometry.ensure_operations()
        page, layout, form = self._page(T("几何与剖分", "Geometry and meshing"))
        # Geometry and history nodes are containers.  Their editor only
        # exposes actions that operate on the Part as a whole.  Parameters
        # are shown after selecting an individual feature node, so there is
        # no second, ambiguous "active feature" selector here.
        data = tuple(self._active_tree_data or ())
        context = data[0] if len(data) >= 2 and int(data[1]) == index else "geometry"
        feature_id = data[2] if context == "feature" and len(data) >= 3 else None
        operation = next((op for op in operations if op.id == feature_id), operations[-1])
        fields = {}
        if context == "feature":
            form.addRow(T("特征", "Feature"), QLabel(f"{operation.id} · {self._operation_name(operation.kind)}"))
            if operation.kind in {"fuse", "cut", "intersect", "fragment"}:
                form.addRow(T("输入特征", "Input features"), QLabel(", ".join(operation.inputs)))
            else:
                for key, value in operation.values.items():
                    widget = self._number(value, 1e-10 if "radius" in key else -1e12, 1e12)
                    fields[key] = widget
                    form.addRow(key, widget)

        def update_live_preview(*_):
            """Update the BREP preview without generating a volume mesh."""
            for field_name, field in fields.items():
                operation.values[field_name] = field.value()
            self._preview_timer.start()

        def commit_live_values():
            # ``valueChanged`` above makes the CAD view responsive while a
            # spin box is edited.  Record only when that edit is finished so
            # one typed value is one undo step and the tree/editor stays put.
            self._record_live_edit(T("CAD 属性已记录（Ctrl+Z 可撤回）", "CAD property recorded (Ctrl+Z to undo)"))

        for field in fields.values():
            field.valueChanged.connect(update_live_preview)
            field.editingFinished.connect(commit_live_values)
        if context == "feature":
            save = QPushButton(T("应用 CAD 特征", "Apply CAD feature"))
            save.clicked.connect(lambda: self._save_feature(index, operation.id, fields))
            layout.addWidget(save)
        # Adding a primitive/boolean/transform is a geometry-level action.
        # When a concrete history node such as ``Base-1: box`` is selected,
        # keep the card focused on editing that feature instead of showing
        # actions that belong to the parent Geometry node.
        geometry_context = context == "geometry"
        if geometry_context:
            add_primitive = QPushButton("+ " + T("添加基本体", "Add primitive"))
            add_primitive.clicked.connect(lambda: self._add_primitive(index))
            add_boolean = QPushButton("+ " + T("布尔运算", "Boolean operation"))
            add_boolean.clicked.connect(lambda: self._add_boolean(index))
            add_transform = QPushButton("+ " + T("变换", "Transform"))
            add_transform.clicked.connect(lambda: self._add_transform(index))
            layout.addWidget(add_primitive); layout.addWidget(add_boolean); layout.addWidget(add_transform)
            mesh_part = QPushButton(T("剖分此 Part", "Mesh this Part"))
            mesh_part.setToolTip(T("只生成当前 Part 的 FE 网格", "Generate the FE mesh for this Part only"))
            mesh_part.clicked.connect(lambda: self._mesh_part(index))
            layout.addWidget(mesh_part)
        else:
            delete = QPushButton(T("删除 CAD 特征", "Delete CAD feature")); delete.setStyleSheet("background:#9d3434")
            delete.clicked.connect(lambda: self._delete_feature(index, operation.id))
            if len(operations) > 1:
                layout.addWidget(delete)
        layout.addStretch(1)
        return page

    def _sets_editor(self, part_index: int, kind: str):
        labels = {"node_sets": T("节点集", "Node sets"), "element_sets": T("单元集", "Element sets"), "surface_sets": T("表面集", "Surface sets")}
        tips = {"node_sets": T("使用 0 基节点编号：0, 3:8, 12。", "Use zero-based node IDs: 0, 3:8, 12."),
                "element_sets": T("使用 0 基单元编号：0, 3:8, 12。", "Use zero-based element IDs: 0, 3:8, 12."),
                "surface_sets": T("格式为 单元号:面号，例如 0:1, 2:3；面号从 1 开始。", "Format element:side, e.g. 0:1, 2:3; sides begin at 1.")}
        page, layout, _ = self._page(labels[kind], tips[kind])
        add = QPushButton("+ " + T("新建集合", "New set"))
        add.clicked.connect(lambda: self._add_set(part_index, kind))
        layout.addWidget(add)
        for item in getattr(self.project.parts[part_index], kind):
            layout.addWidget(QLabel(f"• {item.name}: {item.selection}"))
        layout.addStretch(1)
        return page

    def _auto_set_editor(self, part_index: int, kind: str, name: str):
        """Read-only card for a system-generated collection."""
        part = self.project.parts[part_index]
        if kind == "element_sets":
            title = T("全部单元", "All elements")
            scope = T("该 Part 的全部 C3D4 单元。", "Every C3D4 element in this Part.")
        elif name == "all":
            title = T("全部节点", "All nodes")
            scope = T("该 Part 的全部节点。", "Every node in this Part.")
        elif name == "surface_all":
            title = T("所有表面节点", "All surface nodes")
            scope = T("该 Part 所有外表面上的节点。", "Every node on the Part boundary surfaces.")
        else:
            title = name
            scope = T("对应表面的全部节点。", "Every node belonging to the corresponding surface.")
        page, layout, form = self._page(
            title,
            T("系统集合，自动生成，不可编辑或删除。", "System collection; generated automatically and read-only."))
        form.addRow(T("名称", "Name"), QLabel(name))
        form.addRow(T("范围", "Scope"), QLabel(scope))
        form.addRow(T("所属 Part", "Part"), QLabel(part.name))
        layout.addStretch(1)
        return page

    def _exterior_surface_editor(self, part_index: int):
        part = self.project.parts[part_index]
        page, layout, form = self._page(
            T("Part 外表面", "Part exterior surface"),
            T("自动包含该 Part 的全部外部面；只能改名，不能删除。",
              "Automatically contains every exterior face; it can be renamed but not deleted."))
        name = QLineEdit(part.exterior_surface)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("范围", "Scope"), QLabel(T("全部外部面", "All exterior faces")))
        save = QPushButton(T("应用外表面", "Apply exterior surface"))
        save.clicked.connect(lambda: self._save_exterior_surface(part_index, name.text()))
        layout.addWidget(save)
        layout.addStretch(1)
        return page

    def _set_editor_form(self, part_index: int, kind: str, item_index: int):
        item = getattr(self.project.parts[part_index], kind)[item_index]
        pick_hint = {
            "node_sets": T("从预览选择节点。", "Pick nodes in the view."),
            "element_sets": T("从预览选择体。", "Pick a body in the view."),
            "surface_sets": T("从预览选择面。", "Pick a face in the view."),
        }[kind]
        page, layout, form = self._page(T("集合定义", "Set definition"), pick_hint)
        name = QLineEdit(item.name); selection = QLineEdit(item.selection)
        form.addRow(T("名称", "Name"), name); form.addRow(T("选择", "Selection"), selection)
        pick_label = {
            "node_sets": T("从预览选节点", "Pick node from view"),
            "element_sets": T("从预览选体", "Pick body from view"),
            "surface_sets": T("从预览选面", "Pick face from view"),
        }[kind]
        pick = QPushButton(pick_label)
        pick.clicked.connect(lambda: self._begin_preview_pick(part_index, kind, item_index))
        save = QPushButton(T("应用集合", "Apply set")); save.clicked.connect(lambda: self._save_set(part_index, kind, item_index, name.text(), selection.text()))
        delete = QPushButton(T("删除集合", "Delete set")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete_set(part_index, kind, item_index))
        layout.addWidget(pick); layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _instance_editor(self, index: int):
        item = self.project.instances[index]
        page, layout, form = self._page(T("装配实例", "Assembly instance"))
        name = QLineEdit(item.name); part = QComboBox(); part.addItems([p.name for p in self.project.parts]); part.setCurrentText(item.part_name)
        translation = [self._number(v) for v in item.translation]; rotation = [self._number(v) for v in item.rotation]
        form.addRow(T("名称", "Name"), name); form.addRow(T("零件", "Part"), part)
        for key, widgets in [("T", translation), ("R (rad)", rotation)]:
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            for w in widgets: h.addWidget(w)
            form.addRow(key + " x/y/z", row)

        def update_live_preview(*_):
            current = self.project.instances[index]
            current.part_name = part.currentText()
            current.translation = [w.value() for w in translation]
            current.rotation = [w.value() for w in rotation]
            self._preview_timer.start()

        def commit_live_values():
            self._record_live_edit(T("实例属性已记录（Ctrl+Z 可撤回）", "Instance property recorded (Ctrl+Z to undo)"))

        part.currentIndexChanged.connect(update_live_preview)
        part.currentIndexChanged.connect(commit_live_values)
        for field in [*translation, *rotation]:
            field.valueChanged.connect(update_live_preview)
            field.editingFinished.connect(commit_live_values)
        save = QPushButton(T("应用实例", "Apply instance")); save.clicked.connect(lambda: self._save_instance(index, name.text(), part.currentText(), translation, rotation))
        delete = QPushButton(T("删除实例", "Delete instance")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete("instance", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _rp_editor(self, index: int):
        item = self.project.reference_points[index]
        page, layout, form = self._page(T("参考点", "Reference point"))
        name = QLineEdit(item.name); xyz = [self._number(v) for v in item.point]
        form.addRow(T("名称", "Name"), name); form.addRow("x / y / z", self._row(xyz))
        save = QPushButton(T("应用参考点", "Apply reference point")); save.clicked.connect(lambda: self._save_rp(index, name.text(), xyz))
        delete = QPushButton(T("删除参考点", "Delete reference point")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete("rp", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _boundary_editor(self, index: int):
        item = self.project.boundaries[index]
        page, layout, form = self._page(T("位移边界条件", "Displacement boundary"))
        name = QLineEdit(item.name); instance = self._instance_combo(item.instance); nodes = QLineEdit(item.node_set)
        dofs = [QComboBox() for _ in range(3)]
        for i, box in enumerate(dofs):
            box.addItem(f"U{i + 1}", i)
            box.addItem(T("自由", "Free"), -1)
            box.setCurrentIndex(0 if i in item.dofs else 1)
        form.addRow(T("名称", "Name"), name); form.addRow(T("实例", "Instance"), instance); form.addRow(T("节点集", "Node set"), nodes); form.addRow(T("约束自由度", "Fixed DOFs"), self._row(dofs))
        save = QPushButton(T("应用边界", "Apply boundary")); save.clicked.connect(lambda: self._save_boundary(index, name.text(), instance.currentText(), nodes.text(), dofs))
        delete = QPushButton(T("删除边界", "Delete boundary")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete("boundary", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _load_editor(self, index: int):
        item = self.project.loads[index]
        page, layout, form = self._page(T("载荷定义", "Load definition"))
        name = QLineEdit(item.name); kind = QComboBox()
        for text, value in [(T("压力", "Pressure"), "pressure"), (T("体力", "Body force"), "body_force"), (T("集中力", "Concentrated force"), "concentrated_force")]: kind.addItem(text, value)
        kind.setCurrentIndex(max(0, kind.findData(item.kind)))
        instance = self._instance_combo(item.instance); target = QLineEdit(item.target); values = [self._number(v) for v in item.values]
        form.addRow(T("名称", "Name"), name); form.addRow(T("类型", "Type"), kind); form.addRow(T("实例", "Instance"), instance); form.addRow(T("目标", "Target"), target); form.addRow(T("数值 (P 或 Fx/Fy/Fz)", "Values (P or Fx/Fy/Fz)"), self._row(values))
        save = QPushButton(T("应用载荷", "Apply load")); save.clicked.connect(lambda: self._save_load(index, name.text(), kind.currentData(), instance.currentText(), target.text(), values))
        delete = QPushButton(T("删除载荷", "Delete load")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete("load", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _solver_editor(self):
        page, layout, form = self._page(T("静力隐式求解器", "Static implicit solver"))
        maximum = self._integer(self.project.solver_maximum_iteration, 1, 100000); tolerance = self._number(self.project.solver_tolerance, 1e-12, 1.0, 8)
        form.addRow(T("最大迭代", "Maximum iterations"), maximum); form.addRow(T("收敛误差", "Tolerance"), tolerance)
        save = QPushButton(T("应用求解器设置", "Apply solver settings")); save.clicked.connect(lambda: self._save_solver(maximum.value(), tolerance.value()))
        layout.addWidget(save); layout.addStretch(1)
        return page

    def _hint_editor(self, message):
        page, layout, _ = self._page(message); layout.addStretch(1); return page

    # --------------------------------------------------------------- actions
    def add_part(self):
        n = len(self.project.parts) + 1
        name = f"Part-{n}"
        self.project.parts.append(PartDefinition(name=name))
        # A CAE part becomes visible and analysable only after it has an
        # assembly instance.  Create the usual independent instance together
        # with a newly added part so a new cylinder appears immediately.
        self.project.instances.append(InstanceDefinition(name=f"{name}-1", part_name=name))
        self._changed(T("已添加零件和装配实例", "Part and assembly instance added"))
    def add_instance(self):
        if not self.project.parts:
            QMessageBox.information(self, T("需要零件", "Part required"),
                                    T("请先建立一个零件。", "Create a part first."))
            return
        n = len(self.project.instances) + 1; self.project.instances.append(InstanceDefinition(name=f"Instance-{n}", part_name=self.project.parts[0].name)); self._changed(T("已添加实例", "Instance added"))
    def add_reference_point(self):
        n = len(self.project.reference_points) + 1
        self.project.reference_points.append(ReferencePointDefinition(name=f"RP-{n}"))
        self._active_tree_data = ("rp", n - 1)
        self._changed(T("已添加参考点", "Reference point added"))
    def add_boundary(self):
        if not self.project.instances:
            QMessageBox.information(self, T("需要实例", "Instance required"),
                                    T("请先建立一个装配实例。", "Create an assembly instance first."))
            return
        n = len(self.project.boundaries) + 1; self.project.boundaries.append(BoundaryDefinition(name=f"BC-{n}", instance=self.project.instances[0].name)); self._changed(T("已添加边界条件", "Boundary added"))
    def add_load(self):
        if not self.project.instances:
            QMessageBox.information(self, T("需要实例", "Instance required"),
                                    T("请先建立一个装配实例。", "Create an assembly instance first."))
            return
        n = len(self.project.loads) + 1; self.project.loads.append(LoadDefinition(name=f"Load-{n}", instance=self.project.instances[0].name)); self._changed(T("已添加载荷", "Load added"))
    def add_material(self, part_index):
        part = self.project.parts[part_index]
        n = len(part.materials) + 1
        part.materials.append(MaterialDefinition(name=f"Material-{n}", element_set="all"))
        self._active_tree_data = ("material", part_index, n - 1)
        self._changed(T("已添加材料", "Material added"))
    def _add_set(self, part_index, kind):
        n = len(getattr(self.project.parts[part_index], kind)) + 1
        prefix = {"node_sets": "Nset", "element_sets": "Elset", "surface_sets": "Surface"}[kind]
        getattr(self.project.parts[part_index], kind).append(NamedSet(f"{prefix}-{n}", "0" if kind != "surface_sets" else "0:1")); self._changed(T("已添加集合", "Set added"))

    def merge_parts(self):
        """Create one OCC *union* Part from two untransformed source Parts."""
        if len(self.project.parts) < 2:
            QMessageBox.information(self, T("需要两个零件", "Two parts required"),
                                    T("请先建立至少两个 Part。", "Create at least two Parts first."))
            return
        names = [part.name for part in self.project.parts]
        object_name, accepted = QInputDialog.getItem(self, T("Part 并集合并", "Union Parts"),
                                                      T("对象 Part", "Object Part"), names, 0, False)
        if not accepted: return
        tool_name, accepted = QInputDialog.getItem(self, T("Part 并集合并", "Union Parts"),
                                                    T("工具 Part", "Tool Part"), [name for name in names if name != object_name], 0, False)
        if not accepted: return
        # This toolbar action is deliberately a union.  Difference,
        # intersection, and fragment remain available in the in-Part CAD
        # history, where the object/tool features are explicit.
        kind = "fuse"
        source_a = next(part for part in self.project.parts if part.name == object_name)
        source_b = next(part for part in self.project.parts if part.name == tool_name)
        instances_a = [item for item in self.project.instances if item.part_name == object_name]
        instances_b = [item for item in self.project.instances if item.part_name == tool_name]
        if any(np.linalg.norm(item.translation) > 1e-12 or np.linalg.norm(item.rotation) > 1e-12
               for item in [*instances_a, *instances_b]):
            QMessageBox.warning(self, T("需要零件坐标", "Part coordinates required"), T(
                "跨 Part 布尔运算目前要求两个 Part 的 instance 没有平移或转角。请先把几何参数放到同一 Part 坐标系。",
                "Cross-Part booleans currently require untransformed instances. Put both geometries in one Part coordinate system first."))
            return
        source_instance_names = {item.name for item in [*instances_a, *instances_b]}
        if any(item.instance in source_instance_names for item in [*self.project.boundaries, *self.project.loads]):
            QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                "源 Part 的 instance 已被边界或载荷使用；请先删除或迁移它们再合并。",
                "A source Part instance is used by a boundary or load; remove or migrate those first."))
            return
        if any([source_a.node_sets, source_a.element_sets, source_a.surface_sets,
                source_b.node_sets, source_b.element_sets, source_b.surface_sets]):
            QMessageBox.warning(self, T("存在集合", "Existing sets"), T(
                "跨 Part 布尔会改变 OCC 实体编号，当前不能安全迁移源 Part 的节点集、单元集或表面集。请先删除或在合并后重新选择它们。",
                "A cross-Part boolean changes OCC entity tags, so existing node, element, and surface sets cannot be migrated safely. Remove and recreate them after merging."))
            return
        material_signature = lambda part: [
            (item.name, item.kind, item.young, item.poisson, item.mu, item.kappa, item.element_set)
            for item in part.materials]
        if material_signature(source_a) != material_signature(source_b):
            QMessageBox.warning(self, T("材料不一致", "Material mismatch"), T(
                "跨 Part 并集不会自动迁移不一致的材料定义，请先统一两个 Part 的材料。",
                "Cross-Part union cannot migrate different material definitions automatically; unify the Part materials first."))
            return
        if not np.isclose(float(source_a.density), float(source_b.density)):
            QMessageBox.warning(self, T("密度不一致", "Density mismatch"), T(
                "跨 Part 并集要求两个 Part 的材料密度一致。",
                "Cross-Part union requires the two Parts to use the same material density."))
            return
        operations, finals = [], []
        for prefix, source in (("A", source_a), ("B", source_b)):
            original = source.geometry.ensure_operations()
            mapping = {operation.id: f"{prefix}_{operation.id}" for operation in original}
            for operation in original:
                operations.append(CADOperation(mapping[operation.id], operation.kind,
                                               copy.deepcopy(operation.values),
                                               [mapping[input_id] for input_id in operation.inputs]))
            # A source Part can legitimately contain several disconnected
            # leaf features.  Fold them to one feature before the final
            # cross-Part boolean so none is silently dropped.
            leaves = [mapping[feature_id] for feature_id in self._live_feature_ids(original)]
            if not leaves:
                QMessageBox.warning(
                    self, T("几何历史无有效末端", "Invalid CAD history"),
                    T(f"Part {source.name} 没有可合并的末端特征。",
                      f"Part {source.name} has no live CAD feature to merge."))
                return
            final = leaves[0]
            for leaf_index, leaf in enumerate(leaves[1:], start=2):
                joined = f"{prefix}_Fuse-leaf-{leaf_index}"
                operations.append(CADOperation(joined, "fuse", {}, [final, leaf]))
                final = joined
            finals.append(final)
        merged_name = f"{object_name}_Union_{tool_name}"
        operations.append(CADOperation(f"{kind.title()}-Merged", kind, {}, finals))
        merged = PartDefinition(name=merged_name,
                                geometry=type(source_a.geometry)(operations=operations),
                                young=source_a.young, poisson=source_a.poisson,
                                materials=copy.deepcopy(source_a.materials),
                                density=float(source_a.density))
        self.project.parts = [part for part in self.project.parts if part.name not in {object_name, tool_name}]
        self.project.parts.append(merged)
        self.project.instances = [item for item in self.project.instances if item.part_name != tool_name]
        for item in self.project.instances:
            if item.part_name == object_name: item.part_name = merged_name
        self._active_tree_data = ("part", len(self.project.parts) - 1)
        LOGGER.info("cross-Part OCC union created: %s + %s -> %s", object_name, tool_name, merged_name)
        self._changed(T("已创建 Part 并集", "Part union created"))

    def _save_project_label(self, label): self.project.label = label or "untitled"; self._changed(T("模型名称已保存", "Model label saved"))
    def _save_part(self, i, name):
        name = name or f"Part-{i + 1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.parts)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("Part 名称必须唯一。", "Part names must be unique.")); return
        part = self.project.parts[i]; previous = part.name; part.name = name
        for instance in self.project.instances:
            if instance.part_name == previous: instance.part_name = part.name
        self._changed(T("零件属性已保存", "Part properties saved"))

    def _save_material(self, part_index, material_index, name, kind, element_set,
                       young, poisson, mu, kappa):
        part = self.project.parts[part_index]
        name = name.strip() or f"Material-{material_index + 1}"
        if any(index != material_index and item.name == name
               for index, item in enumerate(part.materials)):
            QMessageBox.warning(
                self, T("名称重复", "Duplicate name"),
                T("材料名称在同一 Part 内必须唯一。", "Material names must be unique within a Part."))
            return
        if kind == "linear_elastic" and (young <= 0 or not 0 <= poisson < .5):
            QMessageBox.warning(
                self, T("材料参数无效", "Invalid material parameters"),
                T("线弹性材料需要 E>0 且 0≤ν<0.5。", "Linear elastic material needs E>0 and 0≤nu<0.5."))
            return
        if kind != "linear_elastic" and (mu <= 0 or kappa <= 0):
            QMessageBox.warning(
                self, T("材料参数无效", "Invalid material parameters"),
                T("Neo-Hookean 材料需要 mu、kappa 大于 0。", "Neo-Hookean material needs positive mu and kappa."))
            return
        part.materials[material_index] = MaterialDefinition(
            name=name, kind=kind, young=young, poisson=poisson,
            mu=mu, kappa=kappa, element_set=element_set)
        self._changed(T("材料已保存", "Material saved"))

    def _save_material_properties(self, part_index: int, density: float):
        if not np.isfinite(float(density)) or float(density) <= 0:
            QMessageBox.warning(
                self, T("材料参数无效", "Invalid material properties"),
                T("密度必须是有限正数。", "Density must be a finite positive number."))
            return
        self.project.parts[part_index].density = float(density)
        self._changed(T("材料属性已保存", "Material properties saved"))

    def _delete_material(self, part_index, material_index):
        part = self.project.parts[part_index]
        if len(part.materials) <= 1:
            QMessageBox.information(
                self, T("无法删除", "Cannot delete"),
                T("Part 至少需要一个材料。", "A Part needs at least one material."))
            return
        del part.materials[material_index]
        self._active_tree_data = ("materials", part_index)
        self._changed(T("材料已删除", "Material deleted"))
    def _save_feature(self, part_index, feature_id, fields):
        operation = next(op for op in self.project.parts[part_index].geometry.ensure_operations()
                         if op.id == feature_id)
        for key, field in fields.items(): operation.values[key] = field.value()
        self._changed(T("CAD 特征已保存", "CAD feature saved"))

    def _add_primitive(self, part_index):
        names = ["box", "wedge", "cylinder", "sphere", "cone", "torus"]
        kind, accepted = QInputDialog.getItem(self, T("添加基本体", "Add primitive"),
                                               T("Gmsh OCC 类型", "Gmsh OCC type"), names, 0, False)
        if not accepted: return
        operations = self.project.parts[part_index].geometry.ensure_operations()
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(CADOperation(identifier, kind, primitive_defaults(kind)))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 基本体", "OCC primitive added"))

    def _add_boolean(self, part_index):
        operations = self.project.parts[part_index].geometry.ensure_operations()
        sources = self._live_feature_ids(operations)
        if len(sources) < 2:
            QMessageBox.information(self, T("需要两个特征", "Two features required"),
                                    T("请先添加至少两个基本体。", "Add at least two primitives first."))
            return
        choices = [
            (T("并集", "Union"), "fuse"),
            (T("差集", "Difference"), "cut"),
            (T("交集", "Intersection"), "intersect"),
            (T("分割", "Fragment"), "fragment"),
        ]
        label, accepted = QInputDialog.getItem(self, T("布尔运算", "Boolean operation"),
                                                T("类型", "Type"), [item[0] for item in choices], 0, False)
        if not accepted: return
        kind = dict(choices)[label]
        first, accepted = QInputDialog.getItem(self, T("布尔运算", "Boolean operation"),
                                                T("对象", "Object"), sources, 0, False)
        if not accepted: return
        second, accepted = QInputDialog.getItem(self, T("布尔运算", "Boolean operation"),
                                                 T("工具", "Tool"), [item for item in sources if item != first], 0, False)
        if not accepted: return
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(CADOperation(identifier, kind, {}, [first, second]))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 布尔特征", "OCC boolean feature added"))

    def _add_transform(self, part_index):
        operations = self.project.parts[part_index].geometry.ensure_operations()
        sources = self._live_feature_ids(operations)
        if not sources:
            QMessageBox.information(self, T("需要特征", "A feature is required"),
                                    T("请先添加一个基本体或布尔特征。", "Add a primitive or boolean feature first."))
            return
        kind, accepted = QInputDialog.getItem(self, T("添加 OCC 变换", "Add OCC transform"),
                                               T("类型", "Type"), ["translate", "rotate", "dilate", "mirror"], 0, False)
        if not accepted: return
        source, accepted = QInputDialog.getItem(self, T("添加 OCC 变换", "Add OCC transform"),
                                                 T("输入特征", "Input feature"), sources, 0, False)
        if not accepted: return
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(CADOperation(identifier, kind, transform_defaults(kind), [source]))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 变换特征", "OCC transform feature added"))

    @staticmethod
    def _live_feature_ids(operations) -> list[str]:
        """Return history leaves that currently own live OCC entities."""
        consumed = {input_id for operation in operations for input_id in operation.inputs}
        return [operation.id for operation in operations if operation.id not in consumed]

    def _delete_feature(self, part_index, feature_id):
        operations = self.project.parts[part_index].geometry.ensure_operations()
        if len(operations) <= 1:
            QMessageBox.information(
                self, T("无法删除", "Cannot delete"),
                T("Part 至少需要一个 CAD 特征。", "A Part needs at least one CAD feature."))
            return
        if any(feature_id in op.inputs for op in operations):
            QMessageBox.warning(self, T("无法删除", "Cannot delete"),
                                T("该特征被布尔运算引用，请先删除依赖特征。", "This feature is used by a boolean operation; delete its dependent feature first."))
            return
        operations[:] = [op for op in operations if op.id != feature_id]
        self._active_tree_data = ("feature", part_index, operations[0].id)
        self._changed(T("CAD 特征已删除", "CAD feature deleted"))
    def _save_set(self, p, kind, i, name, text):
        name = name or "Set"
        if any(index != i and item.name == name for index, item in enumerate(getattr(self.project.parts[p], kind))):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("集合名称在同一 Part 内必须唯一。", "Set names must be unique within a Part.")); return
        if ((kind == "node_sets" and name in auto_node_set_names(self.project.parts[p]))
                or (kind == "element_sets" and name == "all")):
            QMessageBox.warning(
                self, T("名称保留", "Reserved name"),
                T("该名称由系统集合使用。", "This name is reserved for a system collection.")); return
        if kind == "surface_sets" and name == self.project.parts[p].exterior_surface:
            QMessageBox.warning(
                self, T("名称重复", "Duplicate name"),
                T("表面集不能使用 Part 外表面的名称。", "A surface set cannot use the Part exterior-surface name.")); return
        part = self.project.parts[p]
        collection = getattr(part, kind)
        previous = collection[i].name
        collection[i] = NamedSet(name, text)
        if kind == "element_sets" and previous != name:
            # Keep material regions attached to the renamed set rather than
            # leaving a deferred missing-reference error at mesh time.
            for material in part.materials:
                if material.element_set == previous:
                    material.element_set = name
        self._changed(T("集合已保存", "Set saved"))
    def _save_exterior_surface(self, part_index, name):
        name = name.strip()
        part = self.project.parts[part_index]
        if not name:
            QMessageBox.warning(
                self, T("名称无效", "Invalid name"),
                T("外表面名称不能为空。", "The exterior-surface name cannot be empty.")); return
        if name.startswith("gmsh_surface_"):
            QMessageBox.warning(
                self, T("名称保留", "Reserved name"),
                T("外表面不能使用 gmsh_surface_ 开头的名称。",
                  "The exterior surface cannot use a name beginning with gmsh_surface_.")); return
        if any(item.name == name for item in part.surface_sets):
            QMessageBox.warning(
                self, T("名称重复", "Duplicate name"),
                T("外表面名称不能与其他表面集重复。", "The exterior-surface name must be unique from other surface sets.")); return
        if name != part.exterior_surface:
            old_name = part.exterior_surface
            part.exterior_surface = name
            instance_parts = {item.name: item.part_name for item in self.project.instances}
            retargeted = 0
            for load in self.project.loads:
                if (load.kind == "pressure" and load.target == old_name
                        and instance_parts.get(load.instance) == part.name):
                    load.target = name
                    retargeted += 1
            LOGGER.info("Part exterior surface renamed: part=%s old=%s new=%s pressure_loads_retargeted=%d",
                        part.name, old_name, name, retargeted)
            self._changed(T("Part 外表面已重命名", "Part exterior surface renamed"))

    def _save_instance(self, i, name, part, translation, rotation):
        name = name or f"Instance-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.instances)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("实例名称必须唯一。", "Instance names must be unique.")); return
        self.project.instances[i] = InstanceDefinition(name, part, [w.value() for w in translation], [w.value() for w in rotation]); self._changed(T("实例已保存", "Instance saved"))
    def _save_rp(self, i, name, xyz):
        name = name or f"RP-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.reference_points)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("参考点名称必须唯一。", "Reference point names must be unique.")); return
        self.project.reference_points[i] = ReferencePointDefinition(name, [w.value() for w in xyz]); self._changed(T("参考点已保存", "Reference point saved"))
    def _save_boundary(self, i, name, ins, nodes, dofs):
        name = name or f"BC-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.boundaries)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("边界条件名称必须唯一。", "Boundary names must be unique.")); return
        self.project.boundaries[i] = BoundaryDefinition(name, ins, nodes, [d.currentData() for d in dofs if d.currentData() >= 0]); self._changed(T("边界已保存", "Boundary saved"))
    def _save_load(self, i, name, kind, ins, target, values):
        name = name or f"Load-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.loads)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("载荷名称必须唯一。", "Load names must be unique.")); return
        self.project.loads[i] = LoadDefinition(name, kind, ins, target, [w.value() for w in values]); self._changed(T("载荷已保存", "Load saved"))
    def _save_solver(self, maximum, tolerance): self.project.solver_maximum_iteration = maximum; self.project.solver_tolerance = tolerance; self._changed(T("求解器设置已保存", "Solver settings saved"))
    def _delete_set(self, p, kind, i):
        part = self.project.parts[p]
        collection = getattr(part, kind)
        if kind == "element_sets":
            name = collection[i].name
            if any(material.element_set == name for material in part.materials):
                QMessageBox.warning(
                    self, T("存在材料引用", "Material reference exists"),
                    T("该单元集仍被材料使用，请先修改材料分配。",
                      "This element set is assigned to a material; change the material assignment first."))
                return
        del collection[i]
        self._changed(T("集合已删除", "Set deleted"))
    def _delete(self, kind, index):
        table = {"part": self.project.parts, "instance": self.project.instances, "rp": self.project.reference_points, "boundary": self.project.boundaries, "load": self.project.loads}[kind]
        if kind == "part":
            if len(table) == 1:
                QMessageBox.information(self, T("无法删除", "Cannot delete"), T("模型至少需要一个零件。", "A model needs at least one part.")); return
            part_name = table[index].name
            instance_names = {item.name for item in self.project.instances if item.part_name == part_name}
            if any(item.instance in instance_names for item in [*self.project.boundaries, *self.project.loads]):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该零件的实例仍被边界或载荷使用，请先删除或迁移这些定义。",
                    "An instance of this part is still used by a boundary or load. Remove or migrate those definitions first."))
                return
            del table[index]
            self.project.instances[:] = [item for item in self.project.instances if item.part_name != part_name]
            self._active_tree_data = ("project", None)
            self._changed(T("零件已删除", "Part deleted"))
            return
        if kind == "instance":
            if len(table) == 1:
                QMessageBox.information(
                    self, T("无法删除", "Cannot delete"),
                    T("模型至少需要一个装配实例。", "A model needs at least one assembly instance."))
                return
            instance_name = table[index].name
            if any(item.instance == instance_name for item in [*self.project.boundaries, *self.project.loads]):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该实例仍被边界或载荷使用，请先删除或迁移这些定义。",
                    "This instance is still used by a boundary or load. Remove or migrate those definitions first."))
                return
            self._active_tree_data = ("project", None)
        if kind == "rp":
            rp_name = table[index].name
            if any(item.kind == "concentrated_force" and item.target == rp_name for item in self.project.loads):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该参考点仍被集中力使用，请先删除或迁移该载荷。",
                    "This reference point is still used by a concentrated force. Remove or migrate that load first."))
                return
        del table[index]; self._changed(T("对象已删除", "Object deleted"))
    def _changed(self, message):
        self._invalidate_mesh_cache()
        self._push_history()
        self.message.setText(message)
        self.reload()
        self.changed.emit()
        self.notify.emit(message)
        self._preview_timer.start()

    def _record_live_edit(self, message):
        """Commit an already-previewed field change without rebuilding Qt UI."""
        previous_index = self._history_index
        self._push_history()
        if self._history_index != previous_index:
            self._invalidate_mesh_cache()
            self.message.setText(message)
            self.changed.emit()
            self.notify.emit(message)

    def _refresh_preview_automatically(self):
        self.refresh_preview(silent=True)

    def _show_log_path(self):
        QMessageBox.information(self, T("CAE 日志", "CAE log"),
                                T(f"详细日志文件：\n{log_path()}", f"Detailed log file:\n{log_path()}"))

    def refresh_preview(self, silent=False):
        try:
            LOGGER.info("UI BREP refresh start: parts=%d instances=%d silent=%s",
                        len(self.project.parts), len(self.project.instances), silent)
            self.message.setText(T("更新 BREP…", "Updating BREP…"))
            if not self.project.instances:
                raise ValueError("CAE project needs at least one assembly instance.")
            part_names = {part.name for part in self.project.parts}
            missing_instances = [item.name for item in self.project.instances
                                 if item.part_name not in part_names]
            if missing_instances:
                raise ValueError(
                    f"Instances reference missing Parts: {missing_instances}")
            records = []
            part_records = {}
            for part in self.project.parts:
                base = preview_brep(part.geometry.ensure_operations())
                # Part Geometry is also a selectable view.  Keep ownership
                # metadata on its records, just as on assembly instances;
                # otherwise a correctly picked face is rejected because its
                # ``part_name`` is missing.
                part_records[part.name] = [
                    {**record, "part_name": part.name, "instance_name": None}
                    for record in base
                ]
                instances = [item for item in self.project.instances if item.part_name == part.name]
                for instance in instances:
                    for record in base:
                        record = dict(record)
                        record["points"] = self._transform_points(record["points"], instance.translation, instance.rotation)
                        record["vertex_points"] = self._transform_points(record["vertex_points"], instance.translation, instance.rotation)
                        record["part_name"] = part.name
                        record["instance_name"] = instance.name
                        records.append(record)
            # Publish only a complete refresh.  A failure in Part N must not
            # leave Part 1's new records mixed with an older assembly view.
            self._brep_part_records = part_records
            self._brep_assembly_records = records
            self._update_view_tabs()
            self._render_current_view()
            self.message.setText(T("BREP 已更新", "BREP updated"))
            LOGGER.info("UI BREP refresh complete: records=%d", len(records))
        except Exception as exc:
            LOGGER.exception("UI BREP refresh failed")
            self._brep_part_records = {}
            self._brep_assembly_records = []
            self._preview_highlight = None
            self.preview.clear_scene()
            self._update_view_tabs()
            self.message.setText(T("预览失败：", "Preview failed: ") + str(exc))
            if not silent:
                QMessageBox.warning(self, T("预览失败", "Preview failed"), str(exc))

    def show_mesh(self):
        # Rebuilding the complete model invalidates every old Part mesh.  Keep
        # the BREP cache so the user can return to geometry view if this build
        # succeeds, but do not keep an old analysis mesh visible on failure.
        self._invalidate_mesh_cache(clear_brep=False)
        try:
            LOGGER.info("UI FE mesh requested: mesh_size=%r (%s)", self.project.mesh_size,
                        type(self.project.mesh_size).__name__)
            self.message.setText(T("生成 FE 网格…", "Generating FE mesh…"))
            self._preview_controller = build_controller(self.project)
            self._assembly_mesh_controller = self._preview_controller
            self._mesh_parts = dict(self._preview_controller.assembly._parts)
            self._update_view_tabs()
            self.view_tabs.setCurrentIndex(3)
            self._render_current_view()
            self.message.setText(T("FE 网格已生成", "FE mesh generated"))
            LOGGER.info("UI FE mesh complete")
        except Exception as exc:
            LOGGER.exception("UI FE mesh failed")
            if self._view_mode in {"part_mesh", "assembly_mesh"}:
                self.preview.clear_scene()
            else:
                self._render_current_view()
            self._update_view_tabs()
            self.message.setText(T("剖分失败：", "Meshing failed: ") + str(exc))
            detail = T(f"{exc}\n\n详细日志：{log_path()}", f"{exc}\n\nDetailed log: {log_path()}")
            QMessageBox.warning(self, T("剖分失败", "Meshing failed"), detail)

    @staticmethod
    def _transform_points(points, translation, rotation):
        coordinates = np.asarray(points, dtype=float)
        rotation_vector = np.asarray(rotation, dtype=float)
        angle = np.linalg.norm(rotation_vector)
        if angle > 1e-12:
            axis = rotation_vector / angle
            cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
            rotation_matrix = (
                np.eye(3) * np.cos(angle)
                + (1 - np.cos(angle)) * np.outer(axis, axis)
                + np.sin(angle) * cross)
            coordinates = coordinates @ rotation_matrix.T
        return coordinates + np.asarray(translation, dtype=float)

    def _save_global_mesh_size(self):
        value = self.mesh_size.value()
        if value != self.project.mesh_size:
            self.project.mesh_size = value
            self._changed(T("统一网格尺寸已保存", "Global mesh size saved"))

    def _push_history(self) -> None:
        project_state = self.project.to_dict()
        if project_state == self._history[self._history_index]:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(project_state)
        self._history_index += 1

    def undo(self) -> None:
        if self._history_index == 0:
            return
        self._history_index -= 1
        self.project = CAEProject.from_dict(self._history[self._history_index])
        self._invalidate_mesh_cache()
        self.mesh_size.setValue(self.project.mesh_size)
        self.message.setText(T("已撤回 CAD/模型操作", "CAD/model operation undone"))
        self.reload()
        self._preview_timer.start()

    def redo(self) -> None:
        if self._history_index >= len(self._history) - 1:
            return
        self._history_index += 1
        self.project = CAEProject.from_dict(self._history[self._history_index])
        self._invalidate_mesh_cache()
        self.mesh_size.setValue(self.project.mesh_size)
        self.message.setText(T("已重做 CAD/模型操作", "CAD/model operation redone"))
        self.reload()
        self._preview_timer.start()

    # --------------------------------------------------------- view picking
    def _begin_preview_pick(self, part_index: int, kind: str, item_index: int) -> None:
        self._pick_target = (part_index, kind, item_index)
        mode = {"node_sets": "node", "element_sets": "body", "surface_sets": "face"}[kind]
        # Set definitions belong to one Part.  Use its isolated BREP view so
        # another instance or Part cannot intercept the click.
        if self.view_tabs.currentIndex() != 0:
            self.view_tabs.setCurrentIndex(0)
        self.preview.set_selection_mode(mode)
        self.message.setText(T("选择模式：点击预览实体", "Selection mode: pick an entity"))

    def _picked_entity(self, entity: dict) -> None:
        LOGGER.debug("viewport entity picked: %s", entity)
        if self._pick_target is None:
            return
        part_index, kind, item_index = self._pick_target
        definition = self.project.parts[part_index]
        if entity.get("part_name") != definition.name:
            self.message.setText(T("请选择当前集合所属零件上的实体。", "Pick an entity on the part that owns this set."))
            return
        item = getattr(definition, kind)[item_index]
        try:
            if "surface_tag" in entity:
                tag = int(entity["surface_tag"])
                if kind == "element_sets":
                    volumes = entity.get("volume_tags", [])
                    if not volumes:
                        raise ValueError("The selected BREP face has no owning volume.")
                    selection = f"gmsh:volume:{int(volumes[0])}"
                elif kind == "node_sets" and "vertex_tag" in entity:
                    selection = f"gmsh:vertex:{int(entity['vertex_tag'])}"
                else:
                    # A BREP face maps to its complete analysis-surface node
                    # set or its complete element-face surface after meshing.
                    selection = f"gmsh:surface:{tag}"
            elif kind == "node_sets":
                selection = self._format_indices(np.append(parse_indices(item.selection), entity["node_id"]))
            elif kind == "element_sets":
                selection = self._format_indices(entity["body_element_ids"])
            else:
                selection = self._format_surface(entity["surface"])
        except Exception as exc:
            self.message.setText(T("选择写入失败：", "Could not apply selection: ") + str(exc))
            return
        highlight = None
        if entity.get("surface_tag") is not None:
            highlight = {
                "part_name": entity.get("part_name"),
                "instance_name": entity.get("instance_name"),
                "surface_tag": int(entity["surface_tag"]),
            }
        getattr(definition, kind)[item_index] = NamedSet(item.name, selection)
        self._pick_target = None
        self.preview.set_selection_mode("navigate")
        self._changed(T("已从预览写入集合选择", "Set selection written from preview"))
        # _changed() invalidates and rebuilds the BREP cache.  Restore the
        # picked face after that invalidation so the next refresh paints it
        # red instead of losing the visual selection.
        self._preview_highlight = highlight

    @staticmethod
    def _format_indices(values) -> str:
        values = sorted({int(value) for value in values})
        ranges: list[str] = []
        start = previous = None
        for value in values:
            if start is None: start = previous = value
            elif value == previous + 1: previous = value
            else:
                ranges.append(str(start) if start == previous else f"{start}:{previous}")
                start = previous = value
        if start is not None:
            ranges.append(str(start) if start == previous else f"{start}:{previous}")
        return ", ".join(ranges)

    @staticmethod
    def _format_surface(surface) -> str:
        # The existing surface text grammar is element:side.  Keeping it
        # explicit makes the result editable and compatible with parse_faces.
        surface_tokens = []
        for element_ids, side in surface:
            surface_tokens.extend(
                f"{int(element)}:{int(side) + 1}"
                for element in np.asarray(element_ids, dtype=int).ravel())
        return ", ".join(surface_tokens)

    def _instance_combo(self, current: str) -> QComboBox:
        combo_box = QComboBox()
        combo_box.addItems([instance.name for instance in self.project.instances])
        combo_box.setCurrentText(current)
        return combo_box

    @staticmethod
    def _number(value, low=-1e12, high=1e12, decimals=5) -> QDoubleSpinBox:
        spin_box = QDoubleSpinBox()
        spin_box.setRange(low, high)
        spin_box.setDecimals(decimals)
        numeric_value = float(value)
        spin_box.setValue(numeric_value)
        spin_box.setSingleStep(max(10 ** -decimals, abs(numeric_value) / 10 or .1))
        return spin_box

    @staticmethod
    def _integer(value, low=1, high=100000) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setRange(low, high)
        spin_box.setValue(int(value))
        return spin_box

    @staticmethod
    def _row(widgets) -> QWidget:
        row_widget = QWidget()
        layout = QHBoxLayout(row_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets: layout.addWidget(widget)
        return row_widget
