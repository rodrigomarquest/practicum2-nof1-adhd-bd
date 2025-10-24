#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.ios.normalize_ios_usage"""
from importlib import import_module
_m = import_module('make_scripts.ios.normalize_ios_usage')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
