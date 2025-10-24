#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.utils.px_drift

Preserves the historical import path `make_scripts.px_drift` and delegates
implementation to `make_scripts.utils.px_drift`.
"""
from importlib import import_module

_m = import_module('make_scripts.utils.px_drift')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
