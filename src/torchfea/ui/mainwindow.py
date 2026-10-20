"""Top-level shell for the CAE definition and postprocess workspaces."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStackedWidget, QWidget

from torchfea import load_model
from torchfea.cad import export_step
from torchfea.solver.static import StaticResult
from .i18n import LanguageSelector, T
from .model import CAEProject, build_controller
from .postprocess import PostprocessPanel
from .workbench import Workbench


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = CAEProject()
        # Keep the native controller loaded from the last model file while
        # its corresponding UI definition remains untouched.  Rebuilding a
        # discrete/native model through CAEProject can normalize derived
        # values, so result validation should use the original controller
        # whenever the user has not edited the model.
        self._loaded_controller = None
        self._loaded_revision: int | None = None
        self._loaded_model_hash: str | None = None
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.workbench = Workbench(self.project)
        self.postprocess = PostprocessPanel()
        self.stack.addWidget(self.workbench)
        self.stack.addWidget(self.postprocess)
        self._connect_workbench(self.workbench)
        self._connect_postprocess(self.postprocess)
        self.statusBar().addPermanentWidget(LanguageSelector(self._apply_language))
        self._make_menu()
        self.resize(1500, 880)
        self._set_title()

    def _make_menu(self) -> None:
        # Rebuilding the menu is intentional: all labels are generated through
        # T() at creation time, so a language switch must not leave a mixture
        # of Chinese and English actions behind.
        self.menuBar().clear()
        file_menu = self.menuBar().addMenu(T("文件", "File"))
        new_action = file_menu.addAction(T("新建", "New"))
        new_action.triggered.connect(self.new_project)
        open_action = file_menu.addAction(T("打开模型…", "Open model…"))
        open_action.triggered.connect(self.open_project)
        save_action = file_menu.addAction(T("保存模型…", "Save model…"))
        save_action.triggered.connect(self.save_project)
        file_menu.addSeparator()
        export_inp_action = file_menu.addAction(
            T("导出 Abaqus Part (.inp)…", "Export Abaqus Part (.inp)…"))
        export_inp_action.triggered.connect(self.export_inp)
        export_result_action = file_menu.addAction(
            T("导出计算结果…", "Export calculation result…"))
        export_result_action.triggered.connect(self.export_result)
        import_result_action = file_menu.addAction(
            T("导入结果查看…", "Import result for viewing…"))
        import_result_action.triggered.connect(self.import_result)
        file_menu.addSeparator()
        quit_action = file_menu.addAction(T("退出", "Quit"))
        quit_action.triggered.connect(self.close)

    def _connect_workbench(self, workbench: Workbench) -> None:
        workbench.submitRequested.connect(self.submit_project)
        workbench.notify.connect(self.statusBar().showMessage)
        workbench.newRequested.connect(self.new_project)
        workbench.openRequested.connect(self.open_project)
        workbench.saveRequested.connect(self.save_project)
        workbench.exportInpRequested.connect(self.export_inp)
        workbench.exportPartStepRequested.connect(self.export_part_step)
        workbench.importPartStepRequested.connect(self.import_part_step)

    def _connect_postprocess(self, postprocess: PostprocessPanel) -> None:
        postprocess.backRequested.connect(self.show_workbench)
        postprocess.importResultRequested.connect(self.import_result)
        postprocess.exportResultRequested.connect(self.export_result)
        postprocess.notify.connect(self.statusBar().showMessage)

    def _set_title(self) -> None:
        suffix = (
            T("后处理", "Postprocess")
            if self.stack.currentWidget() is self.postprocess
            else T("前处理", "Preprocessor"))
        self.setWindowTitle(f"torchfea CAE — {suffix}")

    def _show_page(self, page: QWidget, message: str | None = None,
                   timeout: int = 3000) -> None:
        """Switch pages through one path so title and status stay in sync."""
        self.stack.setCurrentWidget(page)
        self._set_title()
        if message:
            self.statusBar().showMessage(message, timeout)

    def _replace_workbench(self) -> QWidget:
        """Recreate the preprocessor while retaining the current project."""
        old_workbench = self.workbench
        definition_revision = old_workbench._definition_revision
        old_workbench.close_cleanup()
        self.stack.removeWidget(old_workbench)
        old_workbench.deleteLater()
        self.workbench = Workbench(self.project)
        self.workbench._definition_revision = definition_revision
        self._connect_workbench(self.workbench)
        self.stack.insertWidget(0, self.workbench)
        return old_workbench

    def _apply_language(self) -> None:
        current_widget = self.stack.currentWidget()
        self._make_menu()

        # Recreate the preprocessor with the same project object.  This
        # updates all static labels while retaining every definition already
        # entered by the user.  It also avoids trying to mutate dozens of
        # editor labels one by one.
        old_workbench = self._replace_workbench()
        # Keep the VTK page alive and update its compact toolbar in place.
        # Recreating an embedded OpenGL widget during a language switch can
        # invalidate the native context and used to make the two pages behave
        # differently depending on which one was visible.
        self.postprocess.apply_language()
        current_page = self.workbench if current_widget is old_workbench else self.postprocess
        self.stack.setCurrentWidget(current_page)
        self._set_title()

    def new_project(self) -> None:
        if self.postprocess._thread is not None:
            QMessageBox.warning(
                self, T("计算进行中", "Solve in progress"),
                T("请等待当前计算结束后再新建模型。",
                  "Wait for the current solve to finish before creating a new model."))
            return
        self.project = CAEProject()
        self._loaded_controller = None
        self._loaded_revision = None
        self._loaded_model_hash = None
        self._replace_workbench()
        self.postprocess.set_project(self.project)
        self._show_page(self.workbench, T("已新建模型。", "New model created."))

    def open_project(self) -> None:
        if self.postprocess._thread is not None:
            QMessageBox.warning(
                self, T("计算进行中", "Solve in progress"),
                T("请等待当前计算结束后再打开模型。",
                  "Wait for the current solve to finish before opening a model."))
            return
        path, _ = QFileDialog.getOpenFileName(
            self, T("打开模型", "Open model"), "", "torchfea model (*.npz);;All (*)")
        if not path:
            return
        try:
            controller = load_model(path)
            # Capture the native identity before projecting the model into
            # editable UI definitions.  The projection is deliberately
            # isolated, but retaining this value also protects the file
            # identity if future UI defaults are added.
            loaded_model_hash = controller.get_model_hash()
            self.project = CAEProject.from_controller(controller)
            self.project.label = Path(path).stem
            self.workbench.set_project(self.project)
            self._loaded_controller = controller
            self._loaded_revision = self.workbench._definition_revision
            self._loaded_model_hash = loaded_model_hash
            self.postprocess.set_project(self.project, controller=controller)
            self._show_page(self.workbench)
            self.statusBar().showMessage(T("已打开：", "Opened: ") + path, 4000)
        except Exception as exc:
            QMessageBox.warning(self, T("打开失败", "Open failed"), str(exc))

    def save_project(self) -> None:
        self.project = self.workbench.project
        path, _ = QFileDialog.getSaveFileName(
            self, T("保存模型", "Save model"), f"{self.project.label}.npz",
            "torchfea model (*.npz)")
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        try:
            # A load/save round trip must preserve the native model exactly.
            # The editable CAE projection is intentionally not a byte-for-byte
            # copy of every native runtime/derived field, so rebuilding it
            # here would change the model check code even when the user did
            # not edit anything.  Rebuild only after an applied UI change.
            if (self._loaded_controller is not None
                    and self._loaded_revision == self.workbench._definition_revision
                    and not self.workbench._editor_dirty):
                controller = self._loaded_controller
            elif self.workbench._assembly_mesh_controller is not None:
                controller = self.workbench._assembly_mesh_controller
            elif self.workbench._all_instance_parts_meshed():
                controller = build_controller(
                    self.project,
                    meshed_parts=self.workbench._mesh_parts)
            else:
                controller = build_controller(self.project)
            controller.save_model(path)
            # Saving a native model also saves its already generated Part
            # meshes.  Keep the same controller in the workbench so the UI
            # does not regress to an "unmeshed" state after saving.
            self.workbench._assembly_mesh_controller = controller
            self.workbench._mesh_parts = dict(controller.assembly._parts)
            self.workbench._update_view_tabs()
            # The saved file and this exact editable definition now share the
            # same native controller identity.  Future result imports may use
            # it directly until the next applied editor change.
            self._loaded_controller = controller
            self._loaded_revision = self.workbench._definition_revision
            self._loaded_model_hash = controller.get_model_hash()
            self.statusBar().showMessage(T("已保存：", "Saved: ") + path, 4000)
        except Exception as exc:
            QMessageBox.warning(self, T("保存失败", "Save failed"), str(exc))

    def export_inp(self) -> None:
        self.project = self.workbench.project
        path, _ = QFileDialog.getSaveFileName(self, T("导出 Abaqus Part", "Export Abaqus Part"), f"{self.project.label}.inp", "Abaqus input (*.inp)")
        if not path:
            return
        try:
            controller = build_controller(self.project)
            parts = controller.assembly._parts
            output = Path(path)
            if len(parts) == 1:
                part_name, part = next(iter(parts.items()))
                part.export_inp(str(output), part_name)
                outputs = [str(output)]
            else:
                outputs = []
                for part_name, part in parts.items():
                    target = output.with_name(f"{output.stem}_{part_name}{output.suffix}")
                    part.export_inp(str(target), part_name)
                    outputs.append(str(target))
            self.statusBar().showMessage(T("已导出 INP：", "Exported INP: ") + ", ".join(outputs), 6000)
        except Exception as exc:
            QMessageBox.warning(self, T("导出失败", "Export failed"), str(exc))

    def export_result(self) -> None:
        """Choose a native result or UI table export for the active solve."""
        if self.postprocess.result is None:
            QMessageBox.warning(
                self, T("没有计算结果", "No result"),
                T("请先完成一次计算。", "Run a solve before exporting results."))
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, T("导出计算结果", "Export calculation result"),
            f"{self.project.label}_result.npz",
            "torchfea result (*.npz);;CSV result table (*.csv)")
        if not path:
            return
        output = Path(path)
        if output.suffix.casefold() not in {".npz", ".csv"}:
            output = output.with_suffix(".csv" if "CSV" in selected_filter else ".npz")
        try:
            self.postprocess.export_result(str(output))
            self.statusBar().showMessage(
                T("已导出计算结果：", "Calculation result exported: ") + str(output), 6000)
        except Exception as exc:
            QMessageBox.warning(self, T("导出失败", "Export failed"), str(exc))

    def import_result(self) -> None:
        """Load a result only when its model identity matches the editor."""
        path, _ = QFileDialog.getOpenFileName(
            self, T("导入计算结果", "Import calculation result"), "",
            "torchfea result (*.npz);;All (*)")
        if not path:
            return
        try:
            # The result class owns its file format.  The UI only validates
            # the model identity and derives display fields when needed.
            result = StaticResult.load(path)
            result_hash = str(result.model_hash)
            if not result_hash:
                raise ValueError(
                    "结果文件没有模型哈希码，无法确认它属于当前模型；请重新导出结果。")

            # If the editor has not changed since opening/saving, validate
            # against the native controller itself.  This is essential for
            # INP/discrete models whose UI projection intentionally omits
            # runtime-derived details.  For an edited definition, rebuild it
            # and validate the new native model instead.
            controller = None
            expected_hash = None
            result_compatibility_hash = getattr(result, "compatibility_hash", None)
            if (self._loaded_controller is not None
                    and self._loaded_revision == self.workbench._definition_revision
                    and not self.workbench._editor_dirty):
                loaded_hash = self._loaded_model_hash
                if loaded_hash is None:
                    raise RuntimeError("Loaded model identity is unavailable.")
                loaded_compatibility_hash = (
                    self._loaded_controller.get_result_compatibility_hash())
                if (result_hash == loaded_hash
                        or (result_compatibility_hash
                            and result_compatibility_hash == loaded_compatibility_hash)):
                    controller = self._loaded_controller
                    expected_hash = loaded_hash

            if controller is None:
                controller = build_controller(self.project)
                expected_hash = controller.get_model_hash()
                expected_compatibility_hash = (
                    controller.get_result_compatibility_hash())
                compatible = (
                    result_compatibility_hash
                    and result_compatibility_hash == expected_compatibility_hash
                )
                if result_hash != expected_hash and not compatible:
                    raise ValueError(
                        "结果与当前模型不匹配：\n"
                        f"当前模型：{expected_hash}\n结果文件：{result_hash}")

            if not getattr(controller.assembly, "_RGC", None):
                controller.initialize()
            try:
                controller.GC2RGC(result.GC)
            except Exception as exc:
                raise ValueError("结果自由度与当前模型不一致，无法显示。") from exc
            if not result_compatibility_hash:
                # Legacy result archives have only the strict hash. They are
                # accepted only on an exact match, so it is safe to add the
                # current compatibility identity for future exports.
                result.compatibility_hash = (
                    controller.get_result_compatibility_hash())
            self.postprocess._solved(controller, result, None)
            self._show_page(
                self.postprocess,
                T("结果已导入。", "Result imported."), 4000)
        except Exception as exc:
            QMessageBox.warning(
                self, T("导入失败", "Import failed"),
                T(f"结果导入失败：{exc}", f"Result import failed: {exc}"))

    def export_part_step(self, part_index: int) -> None:
        """Export one Part's final OCC BREP without generating an FE mesh."""
        self.project = self.workbench.project
        if not 0 <= int(part_index) < len(self.project.parts):
            QMessageBox.warning(self, T("导出失败", "Export failed"),
                                T("指定的 Part 不存在。", "The selected Part does not exist."))
            return
        part = self.project.parts[int(part_index)]
        path, _ = QFileDialog.getSaveFileName(
            self, T("导出 STEP", "Export STEP"), f"{part.name}.stp",
            "STEP geometry (*.stp *.step)")
        if not path:
            return
        output = Path(path)
        if output.suffix.casefold() not in {".stp", ".step"}:
            output = output.with_suffix(".stp")
        try:
            export_step(part.geometry.operations, str(output))
            self.statusBar().showMessage(
                T("已导出 STEP：", "Exported STEP: ") + str(output), 6000)
        except Exception as exc:
            QMessageBox.warning(self, T("导出失败", "Export failed"), str(exc))

    def import_part_step(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, T("导入 STEP", "Import STEP"), "",
            "STEP geometry (*.stp *.step);;All (*)")
        if not path:
            return
        try:
            self.workbench.add_step_part(path)
            self.project = self.workbench.project
            self.statusBar().showMessage(
                T("已导入 STEP：", "Imported STEP: ") + path, 6000)
        except Exception as exc:
            QMessageBox.warning(self, T("导入失败", "Import failed"), str(exc))

    def submit_project(self, project, controller) -> None:
        """Move the meshed definition and its native controller to postprocess."""
        self.project = project
        self.postprocess.set_project(project, controller=controller)
        self._show_page(
            self.postprocess,
            T("模型已提交：点击“1 · 开始计算”。",
              "Model submitted: click “1 · Start solve”."),
            4000)

    def closeEvent(self, event) -> None:
        """Finalize VTK children while Qt and the OpenGL context are alive."""
        self.workbench.close_cleanup()
        self.postprocess.close_cleanup()
        super().closeEvent(event)

    def show_workbench(self) -> None:
        self._show_page(
            self.workbench,
            T("前处理：可继续编辑模型。", "Preprocessor: continue editing the model."),
            3000)
