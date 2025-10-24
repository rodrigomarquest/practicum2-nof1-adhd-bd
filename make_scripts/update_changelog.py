#!/usr/bin/env python3
"""Top-level shim for make_scripts.release.update_changelog.

Preserves the historical top-level import/CLI path while implementation
lives in `make_scripts.release.update_changelog`.
"""
from __future__ import annotations
from importlib import import_module
from types import ModuleType


_mod: ModuleType = import_module("make_scripts.release.update_changelog")

# Re-export public names from the implementation module
for _n, _v in vars(_mod).items():
    if not _n.startswith("_"):
        globals()[_n] = _v  # type: ignore

__all__ = [n for n in dir(_mod) if not n.startswith("_")]


if __name__ == "__main__":
    if hasattr(_mod, 'main'):
        raise SystemExit(_mod.main())
    raise SystemExit("No main() in make_scripts.release.update_changelog")
