#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.run_a6_apple

This file preserves the historical import path `make_scripts.run_a6_apple`
but delegates implementation to `make_scripts.apple.run_a6_apple`.
"""
from __future__ import annotations
from importlib import import_module
from types import ModuleType

_mod: ModuleType = import_module("make_scripts.apple.run_a6_apple")

# Re-export public names
for _n, _v in vars(_mod).items():
    if not _n.startswith("_"):
        globals()[_n] = _v  # type: ignore

__all__ = [n for n in dir(_mod) if not n.startswith("_")]

if __name__ == "__main__":
    if hasattr(_mod, 'main'):
        raise SystemExit(_mod.main())
    raise SystemExit("No main() in make_scripts.apple.run_a6_apple")

