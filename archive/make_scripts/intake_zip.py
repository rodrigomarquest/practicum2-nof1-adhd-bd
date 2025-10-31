#!/usr/bin/env python3
"""Top-level shim for make_scripts.intake.intake_zip.

This preserves the historical import/CLI path while the implementation
lives in `make_scripts.intake.intake_zip`.
"""
from __future__ import annotations
from importlib import import_module
from types import ModuleType


_mod: ModuleType = import_module("make_scripts.intake.intake_zip")

# Re-export public names (non-underscore) from the implementation
for _n, _v in vars(_mod).items():
    if not _n.startswith("_"):
        globals()[_n] = _v  # type: ignore

__all__ = [n for n in dir(_mod) if not n.startswith("_")]


if __name__ == "__main__":
    if hasattr(_mod, "main"):
        _mod.main()
    else:
        raise SystemExit("No main() available in make_scripts.intake.intake_zip")
