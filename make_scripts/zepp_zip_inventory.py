#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.zepp.zepp_zip_inventory

Preserves the historical import path `make_scripts.zepp_zip_inventory` and
delegates implementation to `make_scripts.zepp.zepp_zip_inventory`.
"""
from importlib import import_module
_m = import_module('make_scripts.zepp.zepp_zip_inventory')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
