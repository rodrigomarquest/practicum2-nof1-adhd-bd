"""Compatibility shim: import implementation from make_scripts.utils.io_utils

This module preserves the historical top-level import path `make_scripts.io_utils`.
It dynamically imports the authoritative implementation at
`make_scripts.utils.io_utils` and re-exports its public names.
"""
from importlib import import_module

_m = import_module("make_scripts.utils.io_utils")
__all__ = [k for k in dir(_m) if not k.startswith("_")]

"""Compatibility shim: import implementation from make_scripts.utils.io_utils

This module preserves the historical top-level import path `make_scripts.io_utils`.
It dynamically imports the authoritative implementation at
`make_scripts.utils.io_utils` and re-exports its public names.
"""
from importlib import import_module

_m = import_module("make_scripts.utils.io_utils")
__all__ = [k for k in dir(_m) if not k.startswith("_")]

for _k in __all__:
    globals()[_k] = getattr(_m, _k)
