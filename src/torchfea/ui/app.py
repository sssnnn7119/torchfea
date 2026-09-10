"""Application entry point and shared dark theme for torchfea CAE."""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

_DARK_QSS = """
QMainWindow, QDialog, QWidget { background:#12161d; color:#e4e8ef; }
QMenuBar, QToolBar, QMenu, QStatusBar { background:#191f29; color:#d9e0ea; }
QTreeWidget, QTableWidget, QPlainTextEdit, QTextEdit { background:#1c2330; border:1px solid #334055; }
QHeaderView::section { background:#283346; border:0; padding:5px; }
QTabWidget::pane { border:1px solid #334055; }
QTabBar::tab { background:#1c2330; padding:7px 12px; color:#aab5c5; }
QTabBar::tab:selected { background:#1559a6; color:white; }
QPushButton { background:#1559a6; border:0; border-radius:3px; padding:6px 12px; color:white; }
QPushButton:hover { background:#2472c4; }
QPushButton:disabled { background:#3a4351; color:#8993a2; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background:#252e3d; border:1px solid #3b4960; padding:4px; }
QGroupBox { border:1px solid #334055; margin-top:9px; padding-top:7px; }
QGroupBox::title { left:8px; padding:0 4px; color:#86b8ee; }
QSplitter::handle { background:#334055; }
"""


def create_application(argv: list[str] | None = None):
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv or sys.argv)
    app.setStyleSheet(_DARK_QSS)
    return app


def main() -> int:
    app = create_application()
    from .mainwindow import MainWindow
    window = MainWindow()
    window.show()
    return app.exec()
