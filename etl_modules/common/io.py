"""Compatibility shim: expose a small, stable subset of IO helpers.

This module preserves legacy import paths under ``etl_modules.common.io``
while the canonical implementations live in ``src.lib``. We also re-export
the older ``src.domains.common.io`` namespace so other legacy names keep
working.
"""
from importlib import import_module as _imp

# Pull in the domain-backed utilities for backward compatibility
_mod = _imp("src.domains.common.io")
globals().update(_mod.__dict__)

# Prefer the canonical lib implementations for CSV/write helpers
try:
	from src.lib.io_guards import write_csv, atomic_backup_write  # type: ignore
except Exception:
	try:
		from lib.io_guards import write_csv, atomic_backup_write  # type: ignore
	except Exception:
		write_csv = None
		atomic_backup_write = None

try:
	from src.lib.df_utils import zscore, rolling_cv, safe_merge_on_date, ensure_columns  # type: ignore
except Exception:
	try:
		from lib.df_utils import zscore, rolling_cv, safe_merge_on_date, ensure_columns  # type: ignore
	except Exception:
		zscore = None
		rolling_cv = None
		safe_merge_on_date = None
		ensure_columns = None

del _imp, _mod

__all__ = [k for k in globals().keys() if not k.startswith("__")]
