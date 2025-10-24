#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.zepp.create_cardio_zepp_loader"""
from importlib import import_module
_m = import_module('make_scripts.zepp.create_cardio_zepp_loader')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
#!/usr/bin/env python3
