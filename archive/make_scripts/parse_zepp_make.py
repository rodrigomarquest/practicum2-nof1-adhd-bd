#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.zepp.parse_zepp_make"""
from importlib import import_module
_m = import_module('make_scripts.zepp.parse_zepp_make')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
