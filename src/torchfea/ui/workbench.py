"""CAE definition workbench: model tree, property editor, and mesh preview."""

from __future__ import annotations

import copy
import os
import numpy as np
from types import SimpleNamespace

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QInputDialog, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSplitter, QSpinBox,
    QSlider, QTabBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QMenu,
)

from .i18n import T
from .model import (
    BoundaryDefinition, CAEProject, ConstraintDefinition, GeometryDefinition,
    InstanceDefinition, LoadDefinition, MaterialDefinition, NamedSet, PartDefinition,
    ReferencePointDefinition, build_controller,
    auto_node_set_names, cad_selector, parse_indices, surface_node_set_name,
)
from torchfea.cad import (
    BooleanOperation,
    Box,
    CADHistory,
    Cone,
    Cut,
    Cylinder,
    Dilate,
    Fragment,
    Fuse,
    Intersect,
    Mirror,
    Rotate,
    Sphere,
    StepImport,
    Torus,
    Translate,
    Wedge,
    create_part,
    preview_brep,
    preview_discrete_part,
)
from .postprocess import VTKMeshViewer


def _brep_face_points(record) -> np.ndarray:
    """Return face-owned vertices from a shared BREP tessellation array."""
    points = np.asarray(record.get("points", []), dtype=float).reshape(-1, 3)
    triangles = np.asarray(record.get("triangles", []), dtype=int).reshape(-1, 3)
    if len(triangles):
        points = points[np.unique(triangles.reshape(-1))]
    return points


_PRIMITIVE_OPERATION_TYPES = {
    operation_type.kind: operation_type
    for operation_type in (Box, Wedge, Cylinder, Sphere, Cone, Torus)
}
_BOOLEAN_OPERATION_TYPES = {
    operation_type.kind: operation_type
    for operation_type in (Fuse, Cut, Intersect, Fragment)
}
_TRANSFORM_OPERATION_TYPES = {
    operation_type.kind: operation_type
    for operation_type in (Translate, Rotate, Dilate, Mirror)
}


class Workbench(QWidget):
    """Abaqus-style model tree backed by :class:`CAEProject`."""

    #: Hand the current definition to the postprocess page.  Entering that
    #: page does not start a solve; the solve is an explicit second action
    #: there, matching the definition/observer flow used by MorphOpt.
    submitRequested = Signal(object, object)
    notify = Signal(str)
    newRequested = Signal()
    openRequested = Signal()
    saveRequested = Signal()
    exportInpRequested = Signal()
    exportPartStepRequested = Signal(int)
    importPartStepRequested = Signal()

    def __init__(self, project: CAEProject | None = None, parent=None):
        super().__init__(parent)
        self.project = project or CAEProject()
        self._mesh_parts: dict[str, object] = {}
        self._assembly_mesh_controller = None
        self._assembly_mesh_view = None
        self._brep_part_records: dict[str, list[dict]] = {}
        self._brep_assembly_records: list[dict] = []
        self._preview_highlight: dict | None = None
        self._view_mode = "assembly_geometry"
        self._pick_target: tuple[int, str, int] | None = None
        self._active_tree_data = ("project", None)
        # The editor is deliberately modeless: users can type several fields
        # before pressing Apply.  Keep that state separate from the project
        # history so changing the tree cannot silently discard those edits.
        self._editor_dirty = False
        # Monotonic UI-definition revision.  MainWindow uses this to know
        # whether a controller loaded from disk is still the exact model
        # represented by the editor, without hashing on every tree click.
        self._definition_revision = 0
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
        self._seed_discrete_mesh_cache()
        self._seed_loaded_cad_mesh_cache()
        self._update_view_tabs()
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
        actions.setSpacing(6)
        # Keep the workbench header focused on document-level actions.  Model
        # creation is intentionally available from the tree context menu,
        # matching the definition workbench in MorphOpt.
        for text, tooltip, signal in [
            (T("新建", "New"), T("新建 CAE 模型", "Create a new CAE model"), self.newRequested),
            (T("打开", "Open"), T("打开模型", "Open model"), self.openRequested),
            (T("保存", "Save"), T("保存当前模型", "Save the current model"), self.saveRequested),
            (T("导出 INP", "Export INP"),
             T("导出 Abaqus 输入文件", "Export an Abaqus input file"), self.exportInpRequested),
        ]:
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.setStyleSheet("padding:4px 14px;")
            button.clicked.connect(lambda _checked=False, action_signal=signal: action_signal.emit())
            actions.addWidget(button)
        undo = QPushButton("↶ " + T("撤回", "Undo")); undo.clicked.connect(self.undo)
        redo = QPushButton("↷ " + T("重做", "Redo")); redo.clicked.connect(self.redo)
        actions.addWidget(undo); actions.addWidget(redo)
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
        self.tree.setMinimumWidth(230)
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._open_tree_menu)
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
        section_actions = QHBoxLayout()
        section_actions.addWidget(QLabel(T("剖视", "Section")))
        self.section_enabled = QCheckBox(T("启用", "On"))
        self.section_axis = QComboBox()
        self.section_axis.addItem("X", "x")
        self.section_axis.addItem("Y", "y")
        self.section_axis.addItem("Z", "z")
        self.section_axis.setCurrentIndex(2)
        self.section_position = QSlider(Qt.Orientation.Horizontal)
        self.section_position.setRange(0, 100)
        self.section_position.setValue(50)
        self.section_position.setFixedWidth(130)
        self.section_position.setToolTip(T(
            "拖动剖切面位置", "Move the section plane"))
        section_actions.addWidget(self.section_enabled)
        section_actions.addWidget(self.section_axis)
        section_actions.addWidget(self.section_position)
        section_actions.addStretch(1)
        preview_layout.addLayout(section_actions)
        self.preview = VTKMeshViewer()
        fit.clicked.connect(self.preview.reset_camera)
        wireframe.toggled.connect(self.preview.set_wireframe)
        edges.toggled.connect(self.preview.set_edge_visibility)
        self.section_enabled.toggled.connect(self._update_section_view)
        self.section_axis.currentIndexChanged.connect(self._update_section_view)
        self.section_position.valueChanged.connect(self._update_section_view)
        self.preview.entityPicked.connect(self._picked_entity)
        preview_layout.addWidget(self.preview, 1)
        # Start with the complete assembly BREP, matching the conventional
        # CAE landing view; the other three products remain one click away.
        self.view_tabs.setCurrentIndex(2)
        split.addWidget(preview_panel)
        split.setSizes([230, 560, 460])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        outer.addWidget(split, 1)

        foot = QHBoxLayout()
        self.message = QLabel(T("就绪", "Ready"))
        self.message.setStyleSheet("color:#9eabba")
        foot.addWidget(self.message, 1)
        self.submit = QPushButton("▶ " + T("进入后处理", "Open postprocess"))
        self.submit.setToolTip(T(
            "提交已生成网格的模型并进入后处理",
            "Submit the meshed model and open postprocess"))
        self.submit.setStyleSheet("background:#00796b; font-weight:600; padding:7px 18px")
        self.submit.clicked.connect(self._submit_project)
        foot.addWidget(self.submit)
        outer.addLayout(foot)
        self._update_view_tabs()

    def _update_section_view(self, *_args) -> None:
        """Apply the shared clipping-plane controls to the CAD/mesh view."""
        self.preview.set_section(
            self.section_enabled.isChecked(),
            str(self.section_axis.currentData() or "z"),
            self.section_position.value() / 100.0,
        )

    def _submit_project(self) -> None:
        """Submit a definition only after every used Part has a mesh."""
        if self._editor_dirty and not self._confirm_submit_with_pending_edit():
            return
        if not self._all_instance_parts_meshed():
            QMessageBox.warning(
                self,
                T("尚未剖分", "Mesh not generated"),
                T("请先生成所有实例零件的 FE 网格。",
                  "Generate FE meshes for all instance Parts first."))
            return
        try:
            if self._assembly_mesh_controller is None:
                self._assembly_mesh_controller = build_controller(
                    self.project, meshed_parts=self._mesh_parts)
        except Exception as exc:
            QMessageBox.warning(
                self, T("模型准备失败", "Model preparation failed"), str(exc))
            return
        self.submitRequested.emit(self.project, self._assembly_mesh_controller)

    def _all_instance_parts_meshed(self) -> bool:
        """Return whether every Part used by an Instance has a native mesh."""
        instance_part_names = {item.part_name for item in self.project.instances}
        return bool(instance_part_names) and all(
            name in self._mesh_parts for name in instance_part_names)

    def set_project(self, project: CAEProject) -> None:
        """Replace the current definition and reset its reversible history."""
        self.project = project
        self._history = [project.to_dict()]
        self._history_index = 0
        self._active_tree_data = ("project", None)
        self._editor_dirty = False
        self._definition_revision = 0
        self._invalidate_mesh_cache()
        self._seed_loaded_cad_mesh_cache()
        self._update_view_tabs()
        self.mesh_size.setValue(project.mesh_size)
        self.reload()
        self._preview_timer.start()

    def close_cleanup(self) -> None:
        """Stop deferred preview work and release the embedded VTK view."""
        self._preview_timer.stop()
        self.preview.shutdown()

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
            discrete = part.geometry.source.casefold() == "discrete"
            geometry_label = T("离散几何 / 网格", "Discrete geometry / mesh") if discrete \
                else T("几何 / 网格", "Geometry / mesh")
            geometry = self._node(geometry_label, ("geometry", index), item)
            # Keep the OCC history directly under the geometry node.  A
            # second history container added no behavior and only made every
            # feature take one extra click to reach.
            for operation in part.geometry.operations:
                inputs = (
                    f" ← {', '.join(operation.input_ids)}"
                    if operation.input_ids
                    else ""
                )
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
            self._node(f"{item.name} · {self._load_name(item.kind)}", ("load", index), loads)
        bcs = self._node(T("边界条件", "Boundary conditions"), ("boundaries", None), root)
        for index, item in enumerate(self.project.boundaries):
            self._node(item.name, ("boundary", index), bcs)
        constraints = self._node(T("约束", "Constraints"), ("constraints", None), root)
        for index, item in enumerate(self.project.constraints):
            self._node(f"{item.name} [{item.kind}]", ("constraint", index), constraints)
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

    # --------------------------------------------------------- tree actions
    def _open_tree_menu(self, position) -> None:
        """Open the context menu for the model-tree item under the cursor."""
        item = self.tree.itemAt(position)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        data = tuple(item.data(0, Qt.ItemDataRole.UserRole) or ())
        if not data:
            return
        # setCurrentItem() may have been rejected by the unsaved-edit guard.
        # Do not open a menu for the item the user was prevented from entering.
        if tuple(self._active_tree_data) != data:
            return
        kind = data[0]
        # Keep the menu owned by the workbench window.  A QTreeWidget is an
        # ordinary child widget, not a top-level native window; using it as a
        # menu parent can produce ``QWidgetWindow ... must be a top level
        # window`` warnings with some Qt platform plugins.
        menu = QMenu(self.window())

        def add_action(label, callback):
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, action_callback=callback: action_callback())
            return action

        def add_section(_title):
            # QMenu.addSection() is rendered as an empty row by some Qt
            # styles used with PySide6.  The menu already has concise action
            # names, so separators are sufficient and avoid a blank first
            # item in the tree context menu.
            if menu.actions():
                menu.addSeparator()

        def add_geometry_actions(part_index):
            add_section(T("几何", "Geometry"))
            primitives = [
                (T("Box", "Box"), "box"),
                (T("Wedge", "Wedge"), "wedge"),
                (T("Cylinder", "Cylinder"), "cylinder"),
                (T("Sphere", "Sphere"), "sphere"),
                (T("Cone", "Cone"), "cone"),
                (T("Torus", "Torus"), "torus"),
            ]
            for label, primitive_kind in primitives:
                add_action(label, lambda p=part_index, k=primitive_kind: self._add_primitive(p, k))
            add_section(T("布尔", "Boolean"))
            for label, boolean_kind in [
                (T("并集", "Union"), "fuse"),
                (T("差集", "Difference"), "cut"),
                (T("交集", "Intersection"), "intersect"),
                (T("分割", "Fragment"), "fragment"),
            ]:
                add_action(label, lambda p=part_index, k=boolean_kind: self._add_boolean(p, k))
            add_section(T("变换", "Transform"))
            for label, transform_kind in [
                (T("平移", "Translate"), "translate"),
                (T("旋转", "Rotate"), "rotate"),
                (T("缩放", "Dilate"), "dilate"),
                (T("镜像", "Mirror"), "mirror"),
            ]:
                add_action(label, lambda p=part_index, k=transform_kind: self._add_transform(p, k))

        if kind in {"project", "parts"}:
            add_section(T("模型", "Model"))
            add_action(T("零件", "Part"), self.add_part)
            add_action(T("导入 STEP", "Import STEP"), self.importPartStepRequested.emit)
            union_action = menu.addAction(T("Part 并集合并", "Union Parts"))
            union_action.setEnabled(all(
                part.geometry.source.casefold() != "discrete"
                for part in self.project.parts))
            union_action.triggered.connect(lambda _checked=False: self.merge_parts())
        if kind in {"project", "assembly", "instances"}:
            add_section(T("装配", "Assembly"))
            add_action(T("实例", "Instance"), self.add_instance)
        if kind in {"project", "assembly", "rps"}:
            add_action(T("参考点", "Reference point"), self.add_reference_point)
        if kind in {"project", "loads"}:
            add_section(T("载荷", "Loads"))
            for label, load_kind in [
                (T("压力", "Pressure"), "pressure"),
                (T("体力", "Body force"), "body_force"),
                (T("集中力", "Concentrated force"), "concentrated_force"),
                (T("力矩", "Moment"), "moment"),
                (T("参考点-参考点弹簧", "RP-to-RP spring"), "spring_rp_rp"),
                (T("参考点-固定点弹簧", "RP-to-point spring"), "spring_rp_point"),
                (T("自由度罚函数", "DoF penalty"), "penalty_dof"),
                (T("自接触", "Self contact"), "contact_self"),
                (T("接触", "Contact"), "contact"),
            ]:
                add_action(label, lambda k=load_kind: self.add_load(k))
        if kind in {"project", "boundaries"}:
            add_section(T("边界", "Boundaries"))
            add_action(T("节点边界", "Node boundary"), lambda: self.add_boundary("nodes"))
            add_action(T("参考点边界", "Reference-point boundary"),
                       lambda: self.add_boundary("reference_point"))
        if kind in {"project", "constraints"}:
            add_section(T("约束", "Constraints"))
            add_action(T("耦合", "Coupling"), lambda: self.add_constraint("couple"))
        if kind in {"part", "materials"}:
            part_index = int(data[1]) if kind == "part" else int(data[1])
            add_section(T("材料", "Materials"))
            for label, material_kind in [
                (T("线弹性", "Linear elastic"), "linear_elastic"),
                (T("Neo-Hookean", "Neo-Hookean"), "neo_hookean"),
                (T("Neo-Hookean lnJ", "Neo-Hookean lnJ"), "neo_hookean_lnj"),
            ]:
                add_action(label, lambda p=part_index, k=material_kind: self.add_material(p, k))
        if kind in {"part", "geometry"} and self.project.parts[int(data[1])].geometry.source.casefold() != "discrete":
            part_index = int(data[1])
            add_geometry_actions(part_index)
        if kind in {"part", "node_sets", "element_sets", "surface_sets"}:
            part_index = int(data[1])
            if kind == "part":
                add_section(T("集合", "Sets"))
                for set_kind, label in (
                    ("node_sets", T("节点集", "Node set")),
                    ("element_sets", T("单元集", "Element set")),
                    ("surface_sets", T("表面集", "Surface set")),
                ):
                    add_action(label, lambda p=part_index, k=set_kind: self._add_set(p, k))
            else:
                set_kind = kind
                add_action(
                    T({"node_sets": "节点集", "element_sets": "单元集", "surface_sets": "表面集"}[kind],
                      {"node_sets": "Node set", "element_sets": "Element set", "surface_sets": "Surface set"}[kind]),
                    lambda p=part_index, k=set_kind: self._add_set(p, k))
        if kind in {"geometry", "part"}:
            part_index = int(data[1])
            mesh_action = menu.addAction(T("剖分此 Part", "Mesh this Part"))
            mesh_action.triggered.connect(lambda _checked=False, p=part_index: self._mesh_part(p))
        if kind == "part" and self.project.parts[int(data[1])].geometry.source.casefold() != "discrete":
            step_action = menu.addAction(T("导出 STEP", "Export STEP"))
            step_action.triggered.connect(
                lambda _checked=False, p=int(data[1]): self.exportPartStepRequested.emit(p))
        if kind in {"part", "feature", "material", "set", "instance", "rp", "boundary", "load", "constraint"}:
            menu.addSeparator()
            delete_action = menu.addAction(T("删除", "Delete"))
            if kind == "feature":
                delete_action.triggered.connect(
                    lambda _checked=False, p=int(data[1]), feature=data[2]: self._delete_feature(p, feature))
            elif kind == "material":
                delete_action.triggered.connect(
                    lambda _checked=False, p=int(data[1]), m=int(data[2]): self._delete_material(p, m))
            elif kind == "set":
                delete_action.triggered.connect(
                    lambda _checked=False, p=int(data[1]), k=data[2], i=int(data[3]): self._delete_set(p, k, i))
            else:
                delete_action.triggered.connect(
                    lambda _checked=False, object_kind=kind, object_index=int(data[1]):
                    self._delete(object_kind, object_index))
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(position))

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
        self._update_view_tabs()
        self._render_current_view()

    def _update_view_tabs(self) -> None:
        if not hasattr(self, "view_tabs"):
            return
        part_index = self._selected_part_index()
        part_available = bool(part_index is not None and 0 <= part_index < len(self.project.parts) and
                              self.project.parts[part_index].name in self._mesh_parts)
        # The assembly mesh only needs Parts that are actually instantiated.
        # An uninstantiated CAD Part must not block a mesh-only assembly view.
        instance_part_names = {item.part_name for item in self.project.instances}
        all_available = bool(instance_part_names) and all(
            name in self._mesh_parts for name in instance_part_names)
        self.view_tabs.setTabEnabled(1, part_available)
        self.view_tabs.setTabEnabled(3, all_available)
        if hasattr(self, "submit"):
            self.submit.setEnabled(self._all_instance_parts_meshed())
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
            # Part views are reserved for selecting that Part's own CAD
            # entities.  Assembly reference points and load glyphs belong to
            # the assembly views and must not intercept or obscure picking.
            self.preview.show_brep(self._brep_part_records[part.name])
            self.preview.highlight_entity(self._preview_highlight)
            if part.geometry.source.casefold() == "discrete":
                self.preview_mode.setText(T("Part 离散几何", "Part Discrete Geometry"))
            else:
                self.preview_mode.setText(T("Part 几何", "Part Geometry"))
        elif self._view_mode == "assembly_geometry":
            if not self._brep_assembly_records:
                raise RuntimeError("BREP cache is missing assembly geometry.")
            boundary_records, couple_records = self._constraint_visual_records(
                visible_records=self._brep_assembly_records)
            self.preview.show_brep(
                self._brep_assembly_records,
                reference_points=self._reference_point_records(),
                load_records=self._load_visual_records(
                    visible_records=self._brep_assembly_records),
                boundary_records=boundary_records,
                couple_records=couple_records)
            self.preview.highlight_entity(self._preview_highlight)
            self.preview_mode.setText(T("装配体几何", "Assembly Geometry"))
        elif self._view_mode == "part_mesh":
            mesh = self._mesh_parts.get(part.name)
            if mesh is None:
                self.preview.clear_scene()
                return
            self.preview.show_part(mesh, part.name, f"{part.name}-mesh")
            self.preview_mode.setText(T("Part 网格", "Part Mesh"))
        elif self._view_mode == "assembly_mesh":
            if not all(part_item.name in self._mesh_parts for part_item in self.project.parts):
                self.preview.clear_scene()
                return
            if self._assembly_mesh_view is None:
                self._assembly_mesh_view = self._cached_assembly_view()
            boundary_records, couple_records = self._constraint_visual_records()
            self.preview.show_controller(
                self._assembly_mesh_view, brep=False,
                reference_points=self._reference_point_records(),
                load_records=self._load_visual_records(),
                boundary_records=boundary_records,
                couple_records=couple_records)
            self.preview_mode.setText(T("装配体网格", "Assembly Mesh"))

    def _cached_assembly_view(self):
        """Create a render-only assembly from already generated Part meshes."""
        from torchfea.model import Assembly, Instance, ReferencePoint

        assembly = Assembly()
        instance_part_names = {item.part_name for item in self.project.instances}
        for definition in self.project.parts:
            if definition.name not in instance_part_names:
                continue
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

    def _constraint_visual_records(self, visible_records=None):
        """Build display-only markers for boundaries and node couplings."""
        instances = {item.name: item for item in self.project.instances}
        parts = {item.name: item for item in self.project.parts}
        reference_points = {
            item.name: np.asarray(item.point, dtype=float).reshape(3)
            for item in self.project.reference_points
        }

        def limited(values, maximum=48):
            values = np.asarray(values, dtype=float).reshape(-1, 3)
            if len(values):
                values = np.unique(values, axis=0)
            if len(values) <= maximum:
                return values
            return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]

        def records_for(instance_name, part_name_value):
            if visible_records is None:
                return []
            part_definition = parts[part_name_value]
            return [record for record in visible_records
                    if record.get("part_name") == part_definition.name
                    and (record.get("instance_name") in (None, instance_name))]

        def set_selection(part_definition, set_name):
            if set_name == "all":
                return "all"
            if set_name == "surface_all":
                return "surface_all"
            if set_name.endswith("_nodes") and set_name.startswith("surface_"):
                surface_name = set_name[len("surface_"):-len("_nodes")]
                match = next((item for item in part_definition.surface_sets
                              if item.name == surface_name), None)
                return match.selection if match is not None else "surface_all"
            match = next((item for item in part_definition.node_sets
                          if item.name == set_name), None)
            return match.selection if match is not None else "all"

        def target_points(instance_name, set_name):
            instance = instances.get(instance_name)
            if instance is None:
                return np.empty((0, 3))
            part_definition = parts.get(instance.part_name)
            if part_definition is None:
                return np.empty((0, 3))
            native_part = self._mesh_parts.get(part_definition.name)
            if native_part is not None and visible_records is None:
                selection = set_selection(part_definition, set_name)
                if selection == "all":
                    indices = np.arange(len(native_part.nodes), dtype=int)
                elif set_name in native_part.set_nodes:
                    indices = np.asarray(native_part.set_nodes[set_name], dtype=int)
                elif selection == "surface_all" and "surface_all" in native_part.set_nodes:
                    indices = np.asarray(native_part.set_nodes["surface_all"], dtype=int)
                else:
                    try:
                        indices = parse_indices(selection)
                    except (TypeError, ValueError):
                        indices = np.empty(0, dtype=int)
                indices = indices[(indices >= 0) & (indices < len(native_part.nodes))]
                points = native_part.nodes.detach().cpu().numpy()[indices]
                points = self._transform_points(points, instance.translation, instance.rotation)
                return limited(points)

            selection = set_selection(part_definition, set_name)
            records = records_for(instance_name, part_definition.name)
            if selection in {"all", "surface_all", "cad:all"}:
                points = [_brep_face_points(record) for record in records]
                return limited(np.concatenate(points) if points else [])
            if selection.startswith(("cad:vertex:", "gmsh:vertex:")):
                tag = int(selection.rsplit(":", 1)[1])
                points = [point for record in records
                          for vertex_tag, point in zip(
                              record.get("vertex_tags", []), record.get("vertex_points", []))
                          if int(vertex_tag) == tag]
                return limited(points)
            if selection.startswith(("cad:surface:", "gmsh:surface:")):
                tag = int(selection.rsplit(":", 1)[1])
                points = [_brep_face_points(record) for record in records
                          if int(record.get("surface_tag", -1)) == tag]
                return limited(np.concatenate(points) if points else [])
            # A BREP has no analysis nodes before meshing.  For selections
            # without a CAD topology tag, face centers are the stable
            # fallback; explicit CAD faces/vertices were handled above.
            centers = [np.mean(_brep_face_points(record), axis=0)
                       for record in records if len(_brep_face_points(record))]
            return limited(centers)

        def applies_to_part(instance_name):
            return instance_name in instances

        boundary_records = []
        for boundary in self.project.boundaries:
            if not boundary.enabled:
                continue
            if boundary.kind == "reference_point":
                if boundary.reference_point in reference_points:
                    boundary_records.append({
                        "name": boundary.name,
                        "point": reference_points[boundary.reference_point],
                        "dofs": boundary.dofs,
                    })
                continue
            if not applies_to_part(boundary.instance):
                continue
            for point in target_points(boundary.instance, boundary.node_set):
                boundary_records.append({
                    "name": boundary.name, "point": point, "dofs": boundary.dofs,
                })

        couple_records = []
        for constraint in self.project.constraints:
            if not constraint.enabled or not applies_to_part(constraint.instance):
                continue
            target = reference_points.get(constraint.reference_point)
            if target is None:
                continue
            points = target_points(constraint.instance, constraint.node_set)
            if len(points):
                couple_records.append({
                    "name": constraint.name,
                    "points": points,
                    "reference_point": target,
                })
        return boundary_records, couple_records

    def _load_visual_records(self, part_name=None, visible_records=None, mesh_part=None):
        """Build display-only arrows and links for the current viewport.

        The solver-facing load definitions remain unchanged.  This adapter
        turns their targets into a small number of world-space glyph records,
        allowing the same viewer to show CAD, Part mesh, and assembly mesh
        products without introducing a second load model.
        """
        instance_by_name = {item.name: item for item in self.project.instances}
        reference_points = {}
        for item in self.project.reference_points:
            point = np.asarray(item.point, dtype=float).reshape(-1)
            if point.size != 3 or not np.all(np.isfinite(point)):
                raise ValueError(f"Reference point '{item.name}' must contain three finite coordinates.")
            reference_points[item.name] = point
        centers = {}
        face_records = []

        def add_group(key, points):
            values = np.asarray(points, dtype=float).reshape(-1, 3)
            if len(values):
                centers[key] = np.mean(values, axis=0)

        if visible_records:
            grouped = {}
            for record in visible_records:
                points = _brep_face_points(record)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                    continue
                key = record.get("instance_name") or record.get("part_name") or part_name
                grouped.setdefault(key, []).append(points)
                face_records.append((key, record))
            for key, groups in grouped.items():
                add_group(key, np.concatenate(groups, axis=0))
        else:
            # FE mesh previews do not carry BREP faces.  Their node clouds
            # still give stable load origins in both Part and assembly views.
            if mesh_part is not None:
                add_group(part_name, mesh_part.nodes.detach().cpu().numpy())
            else:
                for instance in self.project.instances:
                    if part_name is not None and instance.part_name != part_name:
                        continue
                    native_part = self._mesh_parts.get(instance.part_name)
                    if native_part is None:
                        continue
                    points = native_part.nodes.detach().cpu().numpy()
                    points = self._transform_points(points, instance.translation, instance.rotation)
                    add_group(instance.name, points)

        def target_center(instance_name: str, target_part: str | None = None):
            if instance_name in centers:
                return centers[instance_name]
            instance = instance_by_name.get(instance_name)
            if instance is not None:
                if instance.name in centers:
                    return centers[instance.name]
                if instance.part_name in centers:
                    return centers[instance.part_name]
                target_part = instance.part_name
            if target_part in centers:
                return centers[target_part]
            if centers:
                return next(iter(centers.values()))
            return None

        def load_is_visible(load):
            if not load.enabled:
                return False
            if part_name is None:
                return True
            instance = instance_by_name.get(load.instance)
            instance2 = instance_by_name.get(load.instance2)
            return ((instance is not None and instance.part_name == part_name)
                    or (instance2 is not None and instance2.part_name == part_name)
                    or not load.instance)

        def load_records_for_faces(load):
            instance = instance_by_name.get(load.instance)
            load_part = instance.part_name if instance is not None else part_name
            candidates = [
                record for key, record in face_records
                if (load.instance and key == load.instance)
                or (load_part and key == load_part)
                or (not load.instance and key == part_name)
            ]
            # A picked CAD surface stores its OCC tag in the normal FE
            # selection string.  Restrict pressure arrows to that face.
            target_tag = None
            target_part = next(
                (item for item in self.project.parts if item.name == load_part), None)
            if target_part is None:
                raise ValueError(f"Part '{load_part}' is missing for load '{load.name}'.")
            surface = next((item for item in target_part.surface_sets
                            if item.name == load.target), None)
            if surface is None and load.target != target_part.exterior_surface:
                raise ValueError(
                    f"Surface set '{load.target}' is missing on Part '{target_part.name}'.")
            selection = surface.selection if surface is not None else ""
            if selection.startswith(("cad:surface:", "gmsh:surface:")):
                target_tag = int(selection.rsplit(":", 1)[1])
            if target_tag is not None:
                filtered = [record for record in candidates
                            if int(record.get("surface_tag", -1)) == target_tag]
                if filtered:
                    return filtered
            return candidates

        records = []
        for load in self.project.loads:
            if not load_is_visible(load):
                continue
            kind = str(load.kind)
            values = np.asarray(load.values, dtype=float).reshape(-1)
            colors = {
                "pressure": (1.0, .25, .12),
                "body_force": (.25, .75, 1.0),
                "concentrated_force": (1.0, .30, .12),
                "moment": (.90, .30, 1.0),
                "spring_rp_rp": (.95, .78, .15),
                "spring_rp_point": (.95, .78, .15),
                "contact": (.30, 1.0, .50),
                "contact_self": (.30, 1.0, .50),
                "penalty_dof": (.95, .50, .15),
            }
            color = colors[kind]
            if kind == "pressure":
                pressure = float(values[0])
                candidates = load_records_for_faces(load)
                if candidates:
                    for face in candidates[:16]:
                        points = np.asarray(face.get("points", []), dtype=float)
                        triangles = np.asarray(face.get("triangles", []), dtype=int)
                        if len(points) == 0 or len(triangles) == 0:
                            continue
                        triangle = triangles[min(len(triangles) - 1, len(triangles) // 2)]
                        if len(triangle) < 3:
                            continue
                        a, b, c = points[triangle[:3]]
                        normal = np.cross(b - a, c - a)
                        norm = np.linalg.norm(normal)
                        if norm <= 1e-12:
                            continue
                        normal /= norm
                        face_center = np.mean(points[triangle[:3]], axis=0)
                        body_center = target_center(load.instance)
                        if body_center is not None and np.dot(normal, face_center - body_center) < 0:
                            normal *= -1.0
                        records.append({"kind": kind, "name": load.name,
                                        "origin": face_center,
                                        "vector": normal * (1.0 if pressure >= 0 else -1.0),
                                        "color": color})
                else:
                    origin = target_center(load.instance)
                    if origin is not None:
                        records.append({"kind": kind, "name": load.name, "origin": origin,
                                        "vector": np.array([0., 0., 1. if pressure >= 0 else -1.]),
                                        "color": color})
            elif kind == "body_force":
                vector = values[:3]
                origin = target_center(load.instance)
                if origin is not None and np.linalg.norm(vector) > 1e-12:
                    records.append({"kind": kind, "name": load.name, "origin": origin,
                                    "vector": vector, "color": color})
            elif kind in {"concentrated_force", "moment"}:
                origin = reference_points[load.target]
                vector = values[:3]
                if origin is not None and np.linalg.norm(vector) > 1e-12:
                    records.append({"kind": kind, "name": load.name, "origin": origin,
                                    "vector": vector, "color": color})
            elif kind == "spring_rp_rp":
                start = reference_points.get(load.target)
                end = reference_points.get(load.target2)
                if start is not None and end is not None:
                    records.append({"kind": kind, "name": load.name,
                                    "start": start, "end": end, "color": color})
            elif kind == "spring_rp_point":
                start = reference_points.get(load.target)
                end = values[:3]
                if start is not None:
                    records.append({"kind": kind, "name": load.name,
                                    "start": start, "end": end, "color": color})
            elif kind == "contact":
                start = target_center(load.instance)
                end = target_center(load.instance2, part_name)
                if start is not None and end is not None and np.linalg.norm(start - end) > 1e-12:
                    records.append({"kind": kind, "name": load.name,
                                    "start": start, "end": end, "color": color})
            elif kind == "contact_self":
                origin = target_center(load.instance)
                if origin is not None:
                    records.extend([
                        {"kind": kind, "name": load.name, "origin": origin,
                         "vector": np.array([0., 0., 1.]), "color": color},
                        {"kind": kind, "name": load.name, "origin": origin,
                         "vector": np.array([0., 0., -1.]), "color": color},
                    ])
            elif kind == "penalty_dof":
                # Penalty_DoF is a solver coupling rather than a physical
                # force.  A small axis glyph still makes its target and DoF
                # visible while editing the model.
                origin = reference_points.get(load.target, target_center(load.target))
                if origin is not None:
                    dof = int(round(values[0]))
                    axis = np.zeros(3)
                    axis[dof % 3] = 1.0
                    records.append({"kind": kind, "name": load.name,
                                    "origin": origin, "vector": axis,
                                    "color": color})
        return records

    def _invalidate_mesh_cache(self, clear_brep: bool = True) -> None:
        self._mesh_parts = {}
        self._seed_discrete_mesh_cache()
        self._assembly_mesh_controller = None
        self._assembly_mesh_view = None
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

    def _seed_discrete_mesh_cache(self) -> None:
        """Expose imported native meshes without invoking Gmsh again."""
        for definition in self.project.parts:
            geometry = definition.geometry
            if geometry.source.casefold() != "discrete":
                continue
            native_part = geometry.runtime_part
            if native_part is None and geometry.discrete_data is not None:
                # A project restored from a UI/history snapshot may retain
                # only the native serialized Part.  Rehydrate it directly;
                # a discrete Part must never go through Gmsh meshing again.
                from torchfea.interfaces import Serializable
                native_part = Serializable._deserialize(geometry.discrete_data)
                geometry.runtime_part = native_part
            if native_part is not None:
                self._mesh_parts[definition.name] = native_part

    def _seed_loaded_cad_mesh_cache(self) -> None:
        """Expose a CAD Part mesh restored from a saved native controller."""
        for definition in self.project.parts:
            if definition.geometry.source.casefold() == "cad":
                native_part = definition.geometry.runtime_part
                if native_part is not None:
                    self._mesh_parts[definition.name] = native_part

    def _mesh_part(self, part_index: int) -> None:
        definition = self.project.parts[part_index]
        self._mesh_parts.pop(definition.name, None)
        self._assembly_mesh_controller = None
        self._assembly_mesh_view = None
        if self._view_mode in {"part_mesh", "assembly_mesh"}:
            self.preview.clear_scene()
        self._update_view_tabs()
        try:
            self.message.setText(T(f"正在剖分 {definition.name}…", f"Meshing {definition.name}…"))
            if definition.geometry.source.casefold() == "discrete":
                native_part = definition.geometry.runtime_part
                if native_part is None:
                    raise ValueError(f"Discrete Part '{definition.name}' has no native mesh.")
                self._mesh_parts[definition.name] = native_part
                status = T(f"{definition.name} 已有离散网格", f"{definition.name} already has a discrete mesh")
            else:
                history = CADHistory(
                    definition.geometry.operations,
                    mesh_size=self.project.mesh_size,
                    exterior_surface=definition.exterior_surface,
                )
                for item in definition.node_sets:
                    history.add_node_set(item.name, cad_selector(item.selection))
                for item in definition.element_sets:
                    history.add_element_set(item.name, cad_selector(item.selection))
                for item in definition.surface_sets:
                    history.add_surface_set(item.name, cad_selector(item.selection))
                self._mesh_parts[definition.name] = create_part(
                    history, self.project.mesh_size)
                status = T(f"{definition.name} 网格已生成", f"Mesh generated for {definition.name}")
            self._update_view_tabs()
            self.view_tabs.setCurrentIndex(1)
            self._render_current_view()
            self.message.setText(status)
        except Exception as exc:
            self.message.setText(T("剖分失败：", "Meshing failed: ") + str(exc))
            QMessageBox.warning(self, T("Part 剖分失败", "Part meshing failed"),
                                str(exc))

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
            "step_import": T("导入 STEP", "STEP import"),
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

    @staticmethod
    def _load_name(kind: str) -> str:
        names = {
            "pressure": T("压力", "Pressure"),
            "body_force": T("体力", "Body force"),
            "concentrated_force": T("集中力", "Concentrated force"),
            "moment": T("力矩", "Moment"),
            "spring_rp_rp": T("参考点弹簧", "RP spring"),
            "spring_rp_point": T("固定点弹簧", "Point spring"),
            "penalty_dof": T("自由度罚函数", "DoF penalty"),
            "contact_self": T("自接触", "Self contact"),
            "contact": T("接触", "Contact"),
        }
        return names.get(kind, kind)

    def _selected(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        data = tuple(data)
        if data != tuple(self._active_tree_data) and not self._confirm_editor_switch():
            return
        self._active_tree_data = tuple(data)
        self._editor_dirty = False
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
        elif kind == "set" and data[2] == "surface_sets":
            part = self.project.parts[data[1]]
            surface = part.surface_sets[data[3]]
            selection = str(surface.selection).strip()
            if selection.startswith(("cad:surface:", "gmsh:surface:")):
                self._preview_highlight = {
                    "part_name": part.name,
                    "surface_tag": int(selection.rsplit(":", 1)[1]),
                }
                if self.view_tabs.currentIndex() != 0:
                    self.view_tabs.setCurrentIndex(0)
            else:
                self._preview_highlight = None
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
        elif kind == "constraint": editor = self._constraint_editor(data[1])
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
        self._editor_dirty = False
        self._watch_editor(widget)

    def _watch_editor(self, widget: QWidget) -> None:
        """Mark the current editor dirty when a user changes any field."""
        for field in widget.findChildren(QLineEdit):
            field.textChanged.connect(self._mark_editor_dirty)
        for field in widget.findChildren(QComboBox):
            field.currentIndexChanged.connect(self._mark_editor_dirty)
        for field in widget.findChildren(QDoubleSpinBox):
            field.valueChanged.connect(self._mark_editor_dirty)
        for field in widget.findChildren(QSpinBox):
            field.valueChanged.connect(self._mark_editor_dirty)
        for field in widget.findChildren(QCheckBox):
            field.stateChanged.connect(self._mark_editor_dirty)

    def _mark_editor_dirty(self, *_args) -> None:
        self._editor_dirty = True

    def _confirm_editor_switch(self) -> bool:
        """Ask before replacing an editor that has not been applied."""
        if not self._editor_dirty:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(T("未应用修改", "Unapplied changes"))
        box.setText(T(
            "当前节点有未应用的修改，是否放弃并切换？",
            "The current node has unapplied changes. Discard them and switch?"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.button(QMessageBox.StandardButton.Yes).setText(T("放弃并切换", "Discard and switch"))
        box.button(QMessageBox.StandardButton.No).setText(T("继续编辑", "Keep editing"))
        if box.exec() == QMessageBox.StandardButton.Yes:
            return True
        self._restore_active_tree_item()
        return False

    def _confirm_submit_with_pending_edit(self) -> bool:
        """Ask before leaving an editor with fields that were not applied."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(T("未应用修改", "Unapplied changes"))
        box.setText(T(
            "当前节点有未应用的修改，是否放弃并进入后处理？",
            "The current node has unapplied changes. Discard them and open postprocess?"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.button(QMessageBox.StandardButton.Yes).setText(T("放弃并提交", "Discard and submit"))
        box.button(QMessageBox.StandardButton.No).setText(T("继续编辑", "Keep editing"))
        return box.exec() == QMessageBox.StandardButton.Yes

    def _restore_active_tree_item(self) -> None:
        """Restore the previous tree selection after a canceled switch."""
        item = self._find_tree_data(self._active_tree_data)
        if item is None:
            return
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(item)
        item.setSelected(True)
        self.tree.blockSignals(False)

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
        form.addRow(T("零件", "Parts"), QLabel(str(len(self.project.parts))))
        form.addRow(T("实例", "Instances"), QLabel(str(len(self.project.instances))))
        layout.addStretch(1)
        return page

    def _part_editor(self, index: int):
        part = self.project.parts[index]
        page, layout, form = self._page(T("零件定义", "Part definition"))
        name = QLineEdit(part.name)
        form.addRow(T("名称", "Name"), name)
        source = part.geometry.source.casefold()
        form.addRow(T("几何来源", "Geometry source"), QLabel(
            T("离散网格（无 CAD 历史）", "Discrete mesh (no CAD history)")
            if source == "discrete" else T("CAD 特征历史", "CAD feature history")))
        form.addRow(T("材料数", "Materials"), QLabel(str(len(part.materials))))
        save = QPushButton(T("应用零件属性", "Apply part properties"))
        save.clicked.connect(lambda: self._save_part(index, name.text()))
        delete = QPushButton(T("删除零件", "Delete part")); delete.setStyleSheet("background:#9d3434")
        delete.clicked.connect(lambda: self._delete("part", index))
        layout.addWidget(save)
        if source != "discrete":
            export_step = QPushButton(T("导出 STEP", "Export STEP"))
            export_step.clicked.connect(lambda: self.exportPartStepRequested.emit(index))
            layout.addWidget(export_step)
        layout.addWidget(delete); layout.addStretch(1)
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
        operations = geometry.operations
        page, layout, form = self._page(T("几何与剖分", "Geometry and meshing"))
        if geometry.source.casefold() == "discrete":
            form.addRow(T("来源", "Source"), QLabel(
                T("INP/原生有限元网格", "INP/native finite-element mesh")))
            form.addRow(T("CAD 历史", "CAD history"), QLabel(
                T("无（离散几何）", "None (discrete geometry)")))
            form.addRow(T("节点数", "Nodes"), QLabel(
                str(len(geometry.runtime_part.nodes))))
            layout.addStretch(1)
            return page
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
            if isinstance(operation, StepImport):
                form.addRow(T("文件", "File"), QLabel(operation.filename))
                form.addRow(T("状态", "Status"), QLabel(
                    T("文件内容已嵌入模型", "File content is embedded in the model")))
            elif isinstance(operation, BooleanOperation):
                form.addRow(
                    T("对象特征", "Object feature"),
                    QLabel(operation.object_id),
                )
                form.addRow(
                    T("工具特征", "Tool feature"),
                    QLabel(operation.tool_id),
                )
            else:
                for key in operation.parameter_names:
                    value = getattr(operation, key)
                    widget = self._number(value, 1e-10 if "radius" in key else -1e12, 1e12)
                    fields[key] = widget
                    form.addRow(key, widget)

        def update_live_preview(*_):
            """Update the BREP preview without generating a volume mesh."""
            for field_name, field in fields.items():
                setattr(operation, field_name, field.value())
            self._preview_timer.start()

        def commit_live_values():
            # ``valueChanged`` above makes the CAD view responsive while a
            # spin box is edited.  Record only when that edit is finished so
            # one typed value is one undo step and the tree/editor stays put.
            self._record_live_edit(T("CAD 属性已记录（Ctrl+Z 可撤回）", "CAD property recorded (Ctrl+Z to undo)"))

        for field in fields.values():
            field.valueChanged.connect(update_live_preview)
            field.editingFinished.connect(commit_live_values)
        if context == "feature" and fields:
            save = QPushButton(T("应用 CAD 特征", "Apply CAD feature"))
            save.clicked.connect(lambda: self._save_feature(index, operation.id, fields))
            layout.addWidget(save)
        # Adding a primitive/boolean/transform is a geometry-level action.
        # When a concrete history node such as ``Box-1`` is selected,
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
        page, layout, form = self._page(T("边界条件", "Boundary condition"))
        name = QLineEdit(item.name)
        kind = QComboBox()
        kind.addItem(T("节点边界", "Node boundary"), "nodes")
        kind.addItem(T("参考点边界", "Reference-point boundary"), "reference_point")
        kind.setCurrentIndex(max(0, kind.findData(item.kind)))
        instance = self._instance_combo(item.instance)
        nodes = self._node_set_combo(item.instance, item.node_set)
        reference_point = self._reference_point_combo(item.reference_point)
        dofs = [QComboBox() for _ in range(6)]
        for i, box in enumerate(dofs):
            box.addItem(f"U{i + 1}" if i < 3 else f"R{i - 2}", i)
            box.addItem(T("自由", "Free"), -1)
            box.setCurrentIndex(0 if i in item.dofs else 1)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("类型", "Type"), kind)
        instance_row = form.rowCount(); form.addRow(T("实例", "Instance"), instance)
        node_row = form.rowCount(); form.addRow(T("节点集", "Node set"), nodes)
        rp_row = form.rowCount(); form.addRow(T("参考点", "Reference point"), reference_point)
        dof_row = form.rowCount(); form.addRow(T("约束自由度", "Fixed DOFs"), self._row(dofs))

        def update_fields(*_):
            reference = kind.currentData() == "reference_point"
            form.setRowVisible(instance_row, not reference)
            form.setRowVisible(node_row, not reference)
            form.setRowVisible(rp_row, reference)
            form.setRowVisible(dof_row, True)
            for i, box in enumerate(dofs):
                box.setVisible(reference or i < 3)

        kind.currentIndexChanged.connect(update_fields)
        instance.currentTextChanged.connect(
            lambda value: self._replace_combo_items(
                nodes, self._node_set_names(value), nodes.currentText()))
        update_fields()
        save = QPushButton(T("应用边界", "Apply boundary")); save.clicked.connect(
            lambda: self._save_boundary(index, name.text(), kind.currentData(),
                                        instance.currentText(), nodes.currentText(),
                                        reference_point.currentText(), dofs))
        delete = QPushButton(T("删除边界", "Delete boundary")); delete.setStyleSheet("background:#9d3434"); delete.clicked.connect(lambda: self._delete("boundary", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _constraint_editor(self, index: int):
        item = self.project.constraints[index]
        page, layout, form = self._page(T("约束定义", "Constraint definition"))
        name = QLineEdit(item.name)
        kind = QComboBox()
        kind.addItem(T("耦合", "Coupling"), "couple")
        kind.setCurrentIndex(max(0, kind.findData(item.kind)))
        instance = self._instance_combo(item.instance)
        nodes = self._node_set_combo(item.instance, item.node_set)
        reference_point = self._reference_point_combo(item.reference_point)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("类型", "Type"), kind)
        form.addRow(T("实例", "Instance"), instance)
        form.addRow(T("节点集", "Node set"), nodes)
        form.addRow(T("参考点", "Reference point"), reference_point)
        instance.currentTextChanged.connect(
            lambda value: self._replace_combo_items(
                nodes, self._node_set_names(value), nodes.currentText()))
        save = QPushButton(T("应用约束", "Apply constraint"))
        save.clicked.connect(lambda: self._save_constraint(
            index, name.text(), kind.currentData(), instance.currentText(),
            nodes.currentText(), reference_point.currentText()))
        delete = QPushButton(T("删除约束", "Delete constraint"))
        delete.setStyleSheet("background:#9d3434")
        delete.clicked.connect(lambda: self._delete("constraint", index))
        layout.addWidget(save); layout.addWidget(delete); layout.addStretch(1)
        return page

    def _load_editor(self, index: int):
        item = self.project.loads[index]
        page, layout, form = self._page(T("载荷定义", "Load definition"))
        name = QLineEdit(item.name); kind = QComboBox()
        for text, value in [
            (T("压力", "Pressure"), "pressure"),
            (T("体力", "Body force"), "body_force"),
            (T("集中力", "Concentrated force"), "concentrated_force"),
            (T("力矩", "Moment"), "moment"),
            (T("参考点-参考点弹簧", "RP-to-RP spring"), "spring_rp_rp"),
            (T("参考点-固定点弹簧", "RP-to-point spring"), "spring_rp_point"),
            (T("自由度罚函数", "DoF penalty"), "penalty_dof"),
            (T("自接触", "Self contact"), "contact_self"),
            (T("接触", "Contact"), "contact"),
        ]:
            kind.addItem(text, value)
        kind.setCurrentIndex(max(0, kind.findData(item.kind)))
        instance = self._instance_combo(item.instance)
        instance2 = self._instance_combo(item.instance2 or item.instance)
        target = QComboBox()
        target2 = QComboBox()
        values = [self._number(v, -1e15, 1e15, 8) for v in self._load_value_defaults(item)]
        value_labels = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
        value_row = self._value_grid(values)
        form.addRow(T("名称", "Name"), name)
        form.addRow(T("类型", "Type"), kind)
        instance_row = form.rowCount(); form.addRow(T("实例", "Instance"), instance)
        instance2_row = form.rowCount(); form.addRow(T("第二实例", "Second instance"), instance2)
        target_row = form.rowCount(); form.addRow(T("目标", "Target"), target)
        target2_row = form.rowCount(); form.addRow(T("第二目标", "Second target"), target2)
        value_row_index = form.rowCount(); form.addRow(T("数值", "Values"), value_row)

        value_counts = {
            "pressure": 1, "body_force": 3, "concentrated_force": 3,
            "moment": 3, "spring_rp_rp": 2, "spring_rp_point": 5,
            "penalty_dof": 3, "contact_self": 10, "contact": 7,
        }
        target2_kinds = {"spring_rp_rp", "penalty_dof", "contact"}
        instance_kinds = {"pressure", "body_force", "contact_self", "contact"}

        def update_fields(*_):
            load_kind = kind.currentData()
            self._replace_combo_items(
                target2,
                self._load_target_names(load_kind, instance.currentText(), instance2.currentText(), True),
                target2.currentText())
            self._replace_combo_items(
                target,
                self._load_target_names(load_kind, instance.currentText(), instance2.currentText(), False,
                                        target2.currentText()),
                target.currentText())
            form.setRowVisible(instance_row, load_kind in instance_kinds)
            form.setRowVisible(instance2_row, load_kind == "contact")
            form.setRowVisible(target_row, True)
            form.setRowVisible(target2_row, load_kind in target2_kinds)
            form.setRowVisible(value_row_index, value_counts.get(load_kind, 0) > 0)
            count = value_counts.get(load_kind, 0)
            for value_index, (widget, label) in enumerate(zip(values, value_labels)):
                widget.setVisible(value_index < count)
                widget.setToolTip(label)

        kind.currentIndexChanged.connect(update_fields)
        instance.currentTextChanged.connect(update_fields)
        instance2.currentTextChanged.connect(update_fields)
        target2.currentTextChanged.connect(update_fields)
        update_fields()
        save = QPushButton(T("应用载荷", "Apply load")); save.clicked.connect(
            lambda: self._save_load(index, name.text(), kind.currentData(),
                                    instance.currentText(), instance2.currentText(),
                                    target.currentText(), target2.currentText(), values))
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

    def add_step_part(self, path: str) -> None:
        """Create a Part whose first CAD feature is an embedded STEP import."""
        history = CADHistory()
        history.import_step(path)
        stem = os.path.splitext(os.path.basename(os.fspath(path)))[0] or "ImportedPart"
        name = stem
        suffix = 1
        existing_names = {part.name for part in self.project.parts}
        while name in existing_names:
            suffix += 1
            name = f"{stem}-{suffix}"
        self.project.parts.append(PartDefinition(
            name=name, geometry=GeometryDefinition(operations=history.operations)))
        self.project.instances.append(InstanceDefinition(
            name=f"{name}-1", part_name=name))
        self._active_tree_data = ("part", len(self.project.parts) - 1)
        self._changed(T(f"已导入 Part：{name}", f"Imported Part: {name}"))

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
    def add_boundary(self, kind="nodes"):
        if not self.project.instances:
            QMessageBox.information(self, T("需要实例", "Instance required"),
                                    T("请先建立一个装配实例。", "Create an assembly instance first."))
            return
        if kind == "reference_point" and not self.project.reference_points:
            QMessageBox.information(
                self, T("需要参考点", "Reference point required"),
                T("请先建立一个参考点。", "Create a reference point first."))
            return
        n = len(self.project.boundaries) + 1
        instance_name = self.project.instances[0].name
        reference_point = self.project.reference_points[0].name if self.project.reference_points else ""
        self.project.boundaries.append(BoundaryDefinition(
            name=f"BC-{n}", instance=instance_name, node_set="all", kind=kind,
            reference_point=reference_point))
        self._changed(T("已添加边界条件", "Boundary added"))

    def add_constraint(self, kind="couple"):
        if not self.project.instances or not self.project.reference_points:
            QMessageBox.information(
                self, T("需要实例和参考点", "Instance and reference point required"),
                T("耦合约束需要先建立实例和参考点。", "A coupling needs an instance and a reference point first."))
            return
        n = len(self.project.constraints) + 1
        self.project.constraints.append(ConstraintDefinition(
            name=f"Constraint-{n}", kind=kind,
            instance=self.project.instances[0].name,
            reference_point=self.project.reference_points[0].name,
        ))
        self._active_tree_data = ("constraint", n - 1)
        self._changed(T("已添加约束", "Constraint added"))

    def add_load(self, kind="pressure"):
        if not self.project.instances:
            QMessageBox.information(self, T("需要实例", "Instance required"),
                                    T("请先建立一个装配实例。", "Create an assembly instance first."))
            return
        reference_point_kinds = {
            "concentrated_force", "moment", "spring_rp_rp", "spring_rp_point",
        }
        if kind in reference_point_kinds and not self.project.reference_points:
            QMessageBox.information(
                self, T("需要参考点", "Reference point required"),
                T("该载荷需要先建立一个参考点。", "This load needs a reference point first."))
            return
        if kind == "spring_rp_rp" and len(self.project.reference_points) < 2:
            QMessageBox.information(
                self, T("需要两个参考点", "Two reference points required"),
                T("参考点-参考点弹簧需要两个参考点。",
                  "An RP-to-RP spring needs two reference points."))
            return
        n = len(self.project.loads) + 1
        instance = self.project.instances[0]
        part = next((part for part in self.project.parts if part.name == instance.part_name),
                    self.project.parts[0])
        reference_points = self.project.reference_points
        target = part.exterior_surface
        target2 = ""
        values = [1.0]
        if kind == "body_force":
            target, values = "C3D4", [0.0, 0.0, -9.81e-6]
        elif kind in {"concentrated_force", "moment"}:
            target = reference_points[0].name
            values = [0.0, 0.0, 0.0]
        elif kind == "spring_rp_rp":
            target = reference_points[0].name
            target2 = reference_points[1].name
            values = [1.0, -1.0]
        elif kind == "spring_rp_point":
            target = reference_points[0].name
            values = [0.0, 0.0, 0.0, 1.0, -1.0]
        elif kind == "penalty_dof":
            target, values = instance.name, [0.0, 0.0, 1.0e4]
        elif kind == "contact_self":
            values = [0.5, 1.5, 1.5, 1e-5, 40.0, -0.8, -0.85, 1.5, 0.9, 1.0]
        elif kind == "contact":
            target2 = part.exterior_surface
            values = [1e-5, 40.0, -0.4, -0.85, 1.5, 0.9, 1.0]
        self.project.loads.append(LoadDefinition(
            name=f"Load-{n}", kind=kind, instance=instance.name,
            instance2=instance.name, target=target, target2=target2, values=values))
        self._active_tree_data = ("load", n - 1)
        self._changed(T("已添加载荷", "Load added"))
    def add_material(self, part_index, kind="linear_elastic"):
        part = self.project.parts[part_index]
        n = len(part.materials) + 1
        part.materials.append(MaterialDefinition(name=f"Material-{n}", kind=kind, element_set="all"))
        self._active_tree_data = ("material", part_index, n - 1)
        self._changed(T("已添加材料", "Material added"))
    def _add_set(self, part_index, kind):
        part = self.project.parts[part_index]
        n = len(getattr(part, kind)) + 1
        prefix = {"node_sets": "Nset", "element_sets": "Elset", "surface_sets": "Surface"}[kind]
        selector = "cad:all" if part.geometry.source.casefold() == "cad" else (
            "0" if kind != "surface_sets" else "0:1")
        getattr(part, kind).append(NamedSet(f"{prefix}-{n}", selector))
        self._changed(T("已添加集合", "Set added"))

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
        if any(item.instance in source_instance_names for item in [*self.project.boundaries,
                                                                   *self.project.loads,
                                                                   *self.project.constraints]):
            QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                "源 Part 的 instance 已被边界、载荷或约束使用；请先删除或迁移它们再合并。",
                "A source Part instance is used by a boundary, load, or constraint; remove or migrate those first."))
            return
        if any([source_a.node_sets, source_a.element_sets, source_a.surface_sets,
                source_b.node_sets, source_b.element_sets, source_b.surface_sets]):
            QMessageBox.warning(self, T("存在集合", "Existing sets"), T(
                "跨 Part 布尔会改变 OCC 实体编号，当前不能安全迁移源 Part 的节点集、单元集或表面集。请先删除或在合并后重新选择它们。",
                "A cross-Part boolean changes OCC entity tags, so existing node, element, and surface sets cannot be migrated safely. Remove and recreate them after merging."))
            return
        def material_signature(part):
            return [(item.name, item.kind, item.young, item.poisson,
                     item.mu, item.kappa, item.element_set)
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
            original = source.geometry.operations
            mapping = {operation.id: f"{prefix}_{operation.id}" for operation in original}
            for operation in original:
                operations.append(operation.copy_with_mapped_ids(
                    mapping[operation.id], mapping))
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
                operations.append(Fuse(joined, final, leaf))
                final = joined
            finals.append(final)
        merged_name = f"{object_name}_Union_{tool_name}"
        operations.append(Fuse(f"{kind.title()}-Merged", finals[0], finals[1]))
        merged = PartDefinition(name=merged_name,
                                geometry=type(source_a.geometry)(operations=operations),
                                materials=copy.deepcopy(source_a.materials),
                                density=float(source_a.density))
        self.project.parts = [part for part in self.project.parts if part.name not in {object_name, tool_name}]
        self.project.parts.append(merged)
        self.project.instances = [item for item in self.project.instances if item.part_name != tool_name]
        for item in self.project.instances:
            if item.part_name == object_name: item.part_name = merged_name
        self._active_tree_data = ("part", len(self.project.parts) - 1)
        self._changed(T("已创建 Part 并集", "Part union created"))

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
        operation = next(op for op in self.project.parts[part_index].geometry.operations
                         if op.id == feature_id)
        for key, field in fields.items():
            setattr(operation, key, field.value())
        self._changed(T("CAD 特征已保存", "CAD feature saved"))

    def _add_primitive(self, part_index, kind=None):
        names = ["box", "wedge", "cylinder", "sphere", "cone", "torus"]
        if kind is None:
            kind, accepted = QInputDialog.getItem(self, T("添加基本体", "Add primitive"),
                                                   T("Gmsh OCC 类型", "Gmsh OCC type"), names, 0, False)
            if not accepted:
                return
        if kind not in names:
            raise ValueError(f"Unknown primitive kind: {kind}")
        operations = self.project.parts[part_index].geometry.operations
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(_PRIMITIVE_OPERATION_TYPES[kind](identifier))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 基本体", "OCC primitive added"))

    def _add_boolean(self, part_index, kind=None):
        operations = self.project.parts[part_index].geometry.operations
        sources = self._live_feature_ids(operations)
        if len(sources) < 2:
            QMessageBox.information(self, T("需要两个特征", "Two features required"),
                                    T("请先添加至少两个基本体。", "Add at least two primitives first."))
            return
        choices = {
            "fuse": T("并集", "Union"),
            "cut": T("差集", "Difference"),
            "intersect": T("交集", "Intersection"),
            "fragment": T("分割", "Fragment"),
        }
        if kind is None:
            label, accepted = QInputDialog.getItem(
                self, T("布尔运算", "Boolean operation"), T("类型", "Type"),
                list(choices.values()), 0, False)
            if not accepted:
                return
            kind = next(value for value, label_value in choices.items() if label_value == label)
        if kind not in choices:
            raise ValueError(f"Unknown boolean kind: {kind}")
        first, accepted = QInputDialog.getItem(self, T("布尔运算", "Boolean operation"),
                                                T("对象", "Object"), sources, 0, False)
        if not accepted: return
        second, accepted = QInputDialog.getItem(self, T("布尔运算", "Boolean operation"),
                                                 T("工具", "Tool"), [item for item in sources if item != first], 0, False)
        if not accepted: return
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(_BOOLEAN_OPERATION_TYPES[kind](identifier, first, second))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 布尔特征", "OCC boolean feature added"))

    def _add_transform(self, part_index, kind=None):
        operations = self.project.parts[part_index].geometry.operations
        sources = self._live_feature_ids(operations)
        if not sources:
            QMessageBox.information(self, T("需要特征", "A feature is required"),
                                    T("请先添加一个基本体或布尔特征。", "Add a primitive or boolean feature first."))
            return
        transform_kinds = ["translate", "rotate", "dilate", "mirror"]
        if kind is None:
            kind, accepted = QInputDialog.getItem(
                self, T("添加 OCC 变换", "Add OCC transform"), T("类型", "Type"),
                transform_kinds, 0, False)
            if not accepted:
                return
        if kind not in transform_kinds:
            raise ValueError(f"Unknown transform kind: {kind}")
        source, accepted = QInputDialog.getItem(self, T("添加 OCC 变换", "Add OCC transform"),
                                                 T("输入特征", "Input feature"), sources, 0, False)
        if not accepted: return
        identifier = f"{kind.title()}-{sum(op.kind == kind for op in operations) + 1}"
        operations.append(_TRANSFORM_OPERATION_TYPES[kind](identifier, source))
        self._active_tree_data = ("feature", part_index, identifier)
        self._changed(T("已添加 OCC 变换特征", "OCC transform feature added"))

    @staticmethod
    def _live_feature_ids(operations) -> list[str]:
        """Return history leaves that currently own live OCC entities."""
        consumed = {
            input_id
            for operation in operations
            for input_id in operation.input_ids
        }
        return [operation.id for operation in operations if operation.id not in consumed]

    def _delete_feature(self, part_index, feature_id):
        operations = self.project.parts[part_index].geometry.operations
        if len(operations) <= 1:
            QMessageBox.information(
                self, T("无法删除", "Cannot delete"),
                T("Part 至少需要一个 CAD 特征。", "A Part needs at least one CAD feature."))
            return
        if any(feature_id in op.input_ids for op in operations):
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
        if part.geometry.source.casefold() == "cad":
            try:
                cad_selector(text)
            except ValueError as exc:
                QMessageBox.warning(
                    self, T("CAD 选择无效", "Invalid CAD selection"), str(exc)); return
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
            for load in self.project.loads:
                if (load.kind == "pressure" and load.target == old_name
                        and instance_parts.get(load.instance) == part.name):
                    load.target = name
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
    def _save_boundary(self, i, name, kind, ins, nodes, reference_point, dofs):
        name = name or f"BC-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.boundaries)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("边界条件名称必须唯一。", "Boundary names must be unique.")); return
        selected_dofs = [d.currentData() for d in dofs if d.currentData() >= 0]
        if kind == "reference_point":
            if not reference_point:
                QMessageBox.warning(self, T("缺少参考点", "Reference point required"),
                                     T("参考点边界必须指定参考点。", "Select a reference point.")); return
            self.project.boundaries[i] = BoundaryDefinition(
                name=name, kind=kind, instance="", reference_point=reference_point,
                dofs=selected_dofs)
        else:
            if not ins or not nodes:
                QMessageBox.warning(self, T("边界目标无效", "Invalid boundary target"),
                                     T("节点边界必须指定实例和节点集。", "Select an instance and node set.")); return
            self.project.boundaries[i] = BoundaryDefinition(
                name=name, kind=kind, instance=ins, node_set=nodes, dofs=selected_dofs)
        self._changed(T("边界已保存", "Boundary saved"))

    def _save_constraint(self, i, name, kind, instance, nodes, reference_point):
        name = name or f"Constraint-{i + 1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.constraints)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"),
                                 T("约束名称必须唯一。", "Constraint names must be unique.")); return
        if kind != "couple":
            QMessageBox.warning(self, T("约束类型无效", "Invalid constraint type"),
                                 T("当前只支持耦合约束。", "Only coupling is currently supported.")); return
        if not instance or not nodes or not reference_point:
            QMessageBox.warning(
                self, T("耦合目标无效", "Invalid coupling target"),
                T("请选择已有的实例、节点集和参考点。",
                  "Select an existing instance, node set, and reference point.")); return
        self.project.constraints[i] = ConstraintDefinition(
            name=name, kind=kind, instance=instance, node_set=nodes,
            reference_point=reference_point)
        self._changed(T("约束已保存", "Constraint saved"))
    @staticmethod
    def _load_value_defaults(item: LoadDefinition) -> list[float]:
        defaults = {
            "pressure": [1.0],
            "body_force": [0.0, 0.0, -9.81e-6],
            "concentrated_force": [0.0, 0.0, 0.0],
            "moment": [0.0, 0.0, 0.0],
            "spring_rp_rp": [1.0, -1.0],
            "spring_rp_point": [0.0, 0.0, 0.0, 1.0, -1.0],
            "penalty_dof": [0.0, 0.0, 1.0e4],
            "contact_self": [0.5, 1.5, 1.5, 1e-5, 40.0, -0.8, -0.85, 1.5, 0.9, 1.0],
            "contact": [1e-5, 40.0, -0.4, -0.85, 1.5, 0.9, 1.0],
        }
        default_values = defaults[item.kind]
        values = list(item.values)
        if not values:
            values = default_values
        elif len(values) != len(default_values):
            raise ValueError(
                f"Load '{item.name}' has {len(values)} values; "
                f"expected {len(default_values)}.")
        return values + [0.0] * (10 - len(values))

    def _save_load(self, i, name, kind, ins, ins2, target, target2, values):
        name = name or f"Load-{i+1}"
        if any(index != i and item.name == name for index, item in enumerate(self.project.loads)):
            QMessageBox.warning(self, T("名称重复", "Duplicate name"), T("载荷名称必须唯一。", "Load names must be unique.")); return
        count = {
            "pressure": 1, "body_force": 3, "concentrated_force": 3,
            "moment": 3, "spring_rp_rp": 2, "spring_rp_point": 5,
            "penalty_dof": 3, "contact_self": 10, "contact": 7,
        }[kind]
        load_values = [w.value() for w in values[:count]]
        self.project.loads[i] = LoadDefinition(
            name=name, kind=kind, instance=ins, instance2=ins2,
            target=target.strip(), target2=target2.strip(), values=load_values)
        self._changed(T("载荷已保存", "Load saved"))
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
        table = {"part": self.project.parts, "instance": self.project.instances,
                 "rp": self.project.reference_points, "boundary": self.project.boundaries,
                 "load": self.project.loads, "constraint": self.project.constraints}[kind]
        if kind == "part":
            if len(table) == 1:
                QMessageBox.information(self, T("无法删除", "Cannot delete"), T("模型至少需要一个零件。", "A model needs at least one part.")); return
            part_name = table[index].name
            instance_names = {item.name for item in self.project.instances if item.part_name == part_name}
            if any(item.instance in instance_names for item in [*self.project.boundaries,
                                                               *self.project.loads,
                                                               *self.project.constraints]):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该零件的实例仍被边界、载荷或约束使用，请先删除或迁移这些定义。",
                    "An instance of this part is still used by a boundary, load, or constraint. Remove or migrate those definitions first."))
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
            if any(item.instance == instance_name for item in [*self.project.boundaries,
                                                               *self.project.loads,
                                                               *self.project.constraints]):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该实例仍被边界、载荷或约束使用，请先删除或迁移这些定义。",
                    "This instance is still used by a boundary, load, or constraint. Remove or migrate those definitions first."))
                return
            self._active_tree_data = ("project", None)
        if kind == "rp":
            rp_name = table[index].name
            if any((item.kind in {"concentrated_force", "moment", "spring_rp_point", "spring_rp_rp"}
                    and (item.target == rp_name or item.target2 == rp_name))
                   for item in self.project.loads):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该参考点仍被载荷使用，请先删除或迁移相关载荷。",
                    "This reference point is still used by a load. Remove or migrate that load first."))
                return
            if any(item.reference_point == rp_name for item in [*self.project.boundaries,
                                                                  *self.project.constraints]):
                QMessageBox.warning(self, T("存在依赖", "Existing dependencies"), T(
                    "该参考点仍被边界或约束使用，请先删除或迁移它们。",
                    "This reference point is still used by a boundary or constraint.")); return
        del table[index]; self._changed(T("对象已删除", "Object deleted"))
    def _changed(self, message):
        # Every successful Apply/add/delete operation is now the new clean
        # state of the editor.  A failed validation leaves the flag intact so
        # the user is still warned before navigating away.
        self._editor_dirty = False
        if self._push_history():
            self._definition_revision += 1
        self._invalidate_mesh_cache()
        self.message.setText(message)
        self.reload()
        self.notify.emit(message)
        self._preview_timer.start()

    def _record_live_edit(self, message):
        """Commit an already-previewed field change without rebuilding Qt UI."""
        if self._push_history():
            self._definition_revision += 1
            self._invalidate_mesh_cache()
            self.message.setText(message)
            self.notify.emit(message)

    def _refresh_preview_automatically(self):
        self.refresh_preview(silent=True)

    def refresh_preview(self, silent=False):
        try:
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
                if part.geometry.source.casefold() == "discrete":
                    native_part = part.geometry.runtime_part
                    if native_part is None and part.geometry.discrete_data is not None:
                        from torchfea.interfaces import Serializable
                        native_part = Serializable._deserialize(part.geometry.discrete_data)
                        part.geometry.runtime_part = native_part
                    if native_part is None:
                        raise ValueError(f"Discrete Part '{part.name}' has no native mesh.")
                    base = preview_discrete_part(native_part)
                else:
                    base = preview_brep(part.geometry.operations)
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
            # Publish only a complete refresh so failed previews never leave
            # stale geometry in the viewport.
            self._brep_part_records = part_records
            self._brep_assembly_records = records
            self._update_view_tabs()
            self._render_current_view()
            self.message.setText(T("BREP 已更新", "BREP updated"))
        except Exception as exc:
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
            self.message.setText(T("生成 FE 网格…", "Generating FE mesh…"))
            controller = build_controller(self.project)
            self._assembly_mesh_controller = controller
            self._assembly_mesh_view = None
            self._mesh_parts = dict(controller.assembly._parts)
            self._update_view_tabs()
            self.view_tabs.setCurrentIndex(3)
            self._render_current_view()
            self.message.setText(T("FE 网格已生成", "FE mesh generated"))
        except Exception as exc:
            if self._view_mode in {"part_mesh", "assembly_mesh"}:
                self.preview.clear_scene()
            else:
                self._render_current_view()
            self._update_view_tabs()
            self.message.setText(T("剖分失败：", "Meshing failed: ") + str(exc))
            QMessageBox.warning(self, T("剖分失败", "Meshing failed"), str(exc))

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

    def _push_history(self) -> bool:
        project_state = self.project.to_dict()
        if project_state == self._history[self._history_index]:
            return False
        self._history = self._history[:self._history_index + 1]
        self._history.append(project_state)
        self._history_index += 1
        return True

    def undo(self) -> None:
        if self._history_index == 0:
            return
        self._history_index -= 1
        self.project = CAEProject.from_dict(self._history[self._history_index])
        self._editor_dirty = False
        self._definition_revision += 1
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
        self._editor_dirty = False
        self._definition_revision += 1
        self._invalidate_mesh_cache()
        self.mesh_size.setValue(self.project.mesh_size)
        self.message.setText(T("已重做 CAD/模型操作", "CAD/model operation redone"))
        self.reload()
        self._preview_timer.start()

    # --------------------------------------------------------- view picking
    def _begin_preview_pick(self, part_index: int, kind: str, item_index: int) -> None:
        if self.project.parts[part_index].geometry.source.casefold() == "discrete":
            self._pick_target = None
            self.message.setText(T(
                "离散几何不能进行 CAD 拾取，请直接编辑集合。",
                "Discrete geometry has no CAD picking; edit the set directly."))
            return
        self._pick_target = (part_index, kind, item_index)
        mode = {"node_sets": "node", "element_sets": "body", "surface_sets": "face"}[kind]
        # Set definitions belong to one Part.  Use its isolated BREP view so
        # another instance or Part cannot intercept the click.
        if self.view_tabs.currentIndex() != 0:
            self.view_tabs.setCurrentIndex(0)
        self.preview.set_selection_mode(mode)
        self.message.setText(T("选择模式：点击预览实体", "Selection mode: pick an entity"))

    def _picked_entity(self, entity: dict) -> None:
        if self._pick_target is None:
            return
        part_index, kind, item_index = self._pick_target
        definition = self.project.parts[part_index]
        if definition.geometry.source.casefold() == "discrete":
            self.message.setText(T(
                "离散几何不能进行 CAD 拾取。",
                "Discrete geometry cannot be picked as CAD."))
            self._pick_target = None
            return
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
                    selection = f"cad:volume:{int(volumes[0])}"
                elif kind == "node_sets" and "vertex_tag" in entity:
                    selection = f"cad:vertex:{int(entity['vertex_tag'])}"
                else:
                    # A BREP face maps to its complete analysis-surface node
                    # set or its complete element-face surface after meshing.
                    selection = f"cad:surface:{tag}"
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
        # Keep the element:side grammar explicit for editing.
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

    def _part_for_instance(self, instance_name: str) -> PartDefinition | None:
        instance = next((item for item in self.project.instances
                         if item.name == instance_name), None)
        if instance is None:
            return None
        return next((part for part in self.project.parts
                     if part.name == instance.part_name), None)

    def _node_set_names(self, instance_name: str) -> list[str]:
        """Return every node set available through an assembly instance."""
        part = self._part_for_instance(instance_name)
        if part is None:
            return []
        names = ["all", "surface_all"]
        names.extend(surface_node_set_name(item.name) for item in part.surface_sets)
        names.extend(item.name for item in part.node_sets)
        return list(dict.fromkeys(names))

    def _element_set_names(self, instance_name: str) -> list[str]:
        part = self._part_for_instance(instance_name)
        if part is None:
            return []
        return list(dict.fromkeys(["all", *(item.name for item in part.element_sets)]))

    def _surface_names(self, instance_name: str) -> list[str]:
        part = self._part_for_instance(instance_name)
        if part is None:
            return []
        return list(dict.fromkeys([
            str(part.exterior_surface),
            *(item.name for item in part.surface_sets),
        ]))

    def _node_set_combo(self, instance_name: str, current: str) -> QComboBox:
        combo_box = QComboBox()
        self._replace_combo_items(combo_box, self._node_set_names(instance_name), current)
        return combo_box

    @staticmethod
    def _replace_combo_items(combo_box: QComboBox, values: list[str], current: str = "") -> None:
        """Replace a reference combo without allowing arbitrary text."""
        current = str(current or "")
        combo_box.blockSignals(True)
        combo_box.clear()
        combo_box.addItems([str(value) for value in values])
        if current in values:
            combo_box.setCurrentText(current)
        elif combo_box.count():
            combo_box.setCurrentIndex(0)
        combo_box.blockSignals(False)

    def _penalty_target_names(self, object_type: str) -> list[str]:
        maps = {
            "instance": [item.name for item in self.project.instances],
            "rp": [item.name for item in self.project.reference_points],
            "load": [item.name for item in self.project.loads],
            "constraint": [item.name for item in self.project.constraints],
            "boundary": [item.name for item in self.project.boundaries],
        }
        if object_type == "auto":
            values = [value for names in maps.values() for value in names]
        else:
            values = maps.get(object_type, [])
        return list(dict.fromkeys(values))

    def _load_target_names(self, kind: str, instance_name: str,
                           instance2_name: str, second: bool,
                           object_type: str = "auto") -> list[str]:
        """Return valid target names for one load type and target position."""
        if kind in {"pressure", "contact_self"}:
            return self._surface_names(instance_name)
        if kind == "body_force":
            return ["C3D4"]
        if kind in {"concentrated_force", "moment", "spring_rp_point"}:
            return [point.name for point in self.project.reference_points]
        if kind == "spring_rp_rp":
            return [point.name for point in self.project.reference_points]
        if kind == "penalty_dof":
            if second:
                return ["auto", "instance", "rp", "load", "constraint", "boundary"]
            return self._penalty_target_names(object_type)
        if kind == "contact":
            return self._surface_names(instance2_name if second else instance_name)
        return []

    def _reference_point_combo(self, current: str) -> QComboBox:
        combo_box = QComboBox()
        combo_box.addItems([point.name for point in self.project.reference_points])
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

    @staticmethod
    def _value_grid(widgets) -> QWidget:
        """Compact two-row numeric editor used by multi-parameter loads."""
        row_widget = QWidget()
        layout = QGridLayout(row_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, index // 5, index % 5)
        return row_widget
