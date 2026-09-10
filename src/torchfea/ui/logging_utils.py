"""File logging shared by the CAE UI and its native Gmsh adapter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from logging.handlers import RotatingFileHandler


LOGGER_NAME = "torchfea.cae"
_LOG_PATH_WARNING: str | None = None


def _log_path() -> Path:
    global _LOG_PATH_WARNING
    requested = os.environ.get("TORCHFEA_LOG_FILE", "torchfea_cae.log")
    path = Path(requested).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Check that the target is writable without truncating an existing log.
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "torchfea_cae.log"
        _LOG_PATH_WARNING = f"requested log file {path} is unavailable ({exc}); using {fallback}"
        return fallback


LOG_PATH = _log_path()
LOGGER = logging.getLogger(LOGGER_NAME)


def configure_logging() -> logging.Logger:
    """Install one rotating debug file handler and return the CAE logger."""
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    if not any(getattr(handler, "_torchfea_cae", False) for handler in LOGGER.handlers):
        handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler._torchfea_cae = True
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
        LOGGER.addHandler(handler)
    global _LOG_PATH_WARNING
    if _LOG_PATH_WARNING:
        LOGGER.warning(_LOG_PATH_WARNING)
        _LOG_PATH_WARNING = None
    return LOGGER


def log_path() -> str:
    return str(LOG_PATH)


configure_logging()
