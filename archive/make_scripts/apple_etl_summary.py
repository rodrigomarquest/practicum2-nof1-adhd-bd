#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.apple_etl_summary"""
from importlib import import_module
_m = import_module('make_scripts.apple.apple_etl_summary')
__all__ = getattr(_m, '__all__', [])
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
