#!/usr/bin/env python3
"""Compatibility shim for make_scripts.xsource.xsource_fuse.

This file preserves the old top-level import path while the authoritative
implementation lives in `make_scripts.xsource.xsource_fuse`.
"""
from __future__ import annotations
from importlib import import_module
from types import ModuleType


_mod: ModuleType = import_module("make_scripts.xsource.xsource_fuse")

# Re-export public names from the implementation module
for _name, _obj in vars(_mod).items():
    if not _name.startswith("_"):
        globals()[_name] = _obj  # type: ignore

__all__ = [n for n in dir(_mod) if not n.startswith("_")]


if __name__ == "__main__":
    # forward CLI to implementation.main if present
    if hasattr(_mod, "main"):
        _mod.main()
    else:
        raise SystemExit("No main() available in make_scripts.xsource.xsource_fuse")
