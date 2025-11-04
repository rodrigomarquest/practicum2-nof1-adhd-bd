"""Thin backwards-compatible shim: etl_modules.apple_raw_to_per_metric

This module delegates to the canonical implementation under
`src.domains.apple_raw_to_per_metric` so we keep a single source of truth
for Apple per-metric extraction while preserving legacy import paths.
"""
from importlib import import_module as _imp
_mod = _imp("src.domains.apple_raw_to_per_metric")
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")})
del _imp, _mod
__all__ = [k for k in globals().keys() if not k.startswith("__")]
