"""Small Chinese/English helper shared by the CAE pages."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox


def _settings() -> QSettings:
    # Share the setting namespace with MorphOpt so the two UIs keep one
    # language preference when they are hosted in the same application.
    return QSettings("MorphOptUI", "ui")


def is_en() -> bool:
    return _settings().value("language", "zh", str) == "en"


def T(zh: str, en: str) -> str:
    return en if is_en() else zh


class LanguageSelector(QComboBox):
    def __init__(self, callback=None, parent=None):
        super().__init__(parent)
        self.addItem("中文", "zh")
        self.addItem("English", "en")
        self.setFixedWidth(110)
        current = _settings().value("language", "zh", str)
        self.setCurrentIndex(max(0, self.findData(current)))
        self.currentIndexChanged.connect(lambda: self._changed(callback))

    def _changed(self, callback) -> None:
        settings = _settings()
        settings.setValue("language", self.currentData())
        settings.sync()
        if callback:
            callback()
