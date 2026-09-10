"""Top-level shell for the CAE definition and postprocess workspaces."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStackedWidget, QWidget

from .i18n import LanguageSelector, T
from .model import CAEProject, build_controller
from .postprocess import PostprocessPanel
from .workbench import Workbench


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = CAEProject()
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
        self.resize(1520, 900)
        self._set_title()

    def _make_menu(self) -> None:
        # Rebuilding the menu is intentional: all labels are generated through
        # T() at creation time, so a language switch must not leave a mixture
        # of Chinese and English actions behind.
        self.menuBar().clear()
        file_menu = self.menuBar().addMenu(T("文件", "File"))
        new_action = file_menu.addAction(T("新建", "New"))
        new_action.triggered.connect(self.new_project)
        open_action = file_menu.addAction(T("打开 CAE 定义…", "Open CAE definition…"))
        open_action.triggered.connect(self.open_project)
        save_action = file_menu.addAction(T("保存 CAE 定义…", "Save CAE definition…"))
        save_action.triggered.connect(self.save_project)
        file_menu.addSeparator()
        export_model_action = file_menu.addAction(
            T("导出 torchfea 模型 (.npz)…", "Export torchfea model (.npz)…"))
        export_model_action.triggered.connect(self.export_model)
        export_inp_action = file_menu.addAction(
            T("导出 Abaqus Part (.inp)…", "Export Abaqus Part (.inp)…"))
        export_inp_action.triggered.connect(self.export_inp)
        file_menu.addSeparator()
        quit_action = file_menu.addAction(T("退出", "Quit"))
        quit_action.triggered.connect(self.close)

    def _connect_workbench(self, workbench: Workbench) -> None:
        workbench.runRequested.connect(self.goto_postprocess)
        workbench.notify.connect(self.statusBar().showMessage)

    def _connect_postprocess(self, postprocess: PostprocessPanel) -> None:
        postprocess.backRequested.connect(self.show_workbench)
        postprocess.notify.connect(self.statusBar().showMessage)

    def _set_title(self) -> None:
        suffix = (
            T("后处理", "Postprocess")
            if self.stack.currentWidget() is self.postprocess
            else T("前处理", "Preprocessor"))
        self.setWindowTitle(f"torchfea CAE — {suffix}")

    def _replace_workbench(self) -> QWidget:
        """Recreate the preprocessor while retaining the current project."""
        old_workbench = self.workbench
        self.stack.removeWidget(old_workbench)
        old_workbench.deleteLater()
        self.workbench = Workbench(self.project)
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

        # Preserve an existing result view when it is safe to rebuild it.  A
        # running solver owns a worker thread, so leave that live page intact
        # until the calculation has finished.
        if current_widget is self.postprocess and self.postprocess._thread is None:
            old_postprocess = self.postprocess
            self.stack.removeWidget(old_postprocess)
            old_postprocess.deleteLater()
            self.postprocess = PostprocessPanel()
            self._connect_postprocess(self.postprocess)
            self.stack.addWidget(self.postprocess)
            self.postprocess.set_project(self.project)
            self.postprocess.controller = old_postprocess.controller
            self.postprocess.result = old_postprocess.result
            self.postprocess.refresh_view()
        current_page = self.workbench if current_widget is old_workbench else self.postprocess
        self.stack.setCurrentWidget(current_page)
        self._set_title()

    def new_project(self) -> None:
        self.project = CAEProject()
        self._replace_workbench()
        self.stack.setCurrentWidget(self.workbench)
        self._set_title()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, T("打开 CAE 定义", "Open CAE definition"), "", "torchfea CAE (*.torchfea.json *.json);;All (*)")
        if not path:
            return
        try:
            self.project = CAEProject.load(path)
            self.workbench.set_project(self.project)
            self.statusBar().showMessage(T("已打开：", "Opened: ") + path, 4000)
        except Exception as exc:
            QMessageBox.warning(self, T("打开失败", "Open failed"), str(exc))

    def save_project(self) -> None:
        self.project = self.workbench.project
        path, _ = QFileDialog.getSaveFileName(self, T("保存 CAE 定义", "Save CAE definition"), f"{self.project.label}.torchfea.json", "torchfea CAE (*.torchfea.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".torchfea.json"
        self.project.save(path)
        self.statusBar().showMessage(T("已保存：", "Saved: ") + path, 4000)

    def export_model(self) -> None:
        self.project = self.workbench.project
        path, _ = QFileDialog.getSaveFileName(self, T("导出 torchfea 模型", "Export torchfea model"), f"{self.project.label}.npz", "torchfea model (*.npz)")
        if not path:
            return
        try:
            controller = build_controller(self.project)
            controller.save_model(path)
            self.statusBar().showMessage(T("已导出原生 torchfea 模型：", "Exported native torchfea model: ") + path, 5000)
        except Exception as exc:
            QMessageBox.warning(self, T("导出失败", "Export failed"), str(exc))

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

    def goto_postprocess(self, project) -> None:
        self.project = project
        self.postprocess.set_project(project)
        self.stack.setCurrentWidget(self.postprocess)
        self._set_title()

    def show_workbench(self) -> None:
        self.stack.setCurrentWidget(self.workbench)
        self._set_title()
