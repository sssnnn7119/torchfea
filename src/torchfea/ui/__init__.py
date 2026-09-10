"""PySide6 CAE workbench for defining and reviewing torchfea models."""

from __future__ import annotations


def run_app() -> int:
    """Create and run the torchfea CAE application."""
    from .app import main
    return main()


__all__ = ["run_app"]
