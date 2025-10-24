#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.run_a6_apple

This file preserves the historical import path `make_scripts.run_a6_apple`
but delegates implementation to `make_scripts.apple.run_a6_apple`.
"""
from importlib import import_module

_m = import_module('make_scripts.apple.run_a6_apple')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)

