# Thin re-export for legacy import path: etl_modules.io_utils
# Ensures private helper `_open_zip_any` is also available for legacy callers.
from importlib import import_module as _imp
_mod = _imp('src.domains.io_utils')
globals().update(_mod.__dict__)

# Explicitly expose the private helper if present
try:
    _open_zip_any = _mod._open_zip_any  # noqa: F401
except Exception:
    pass

# Clean up
del _imp, _mod
__all__ = [k for k in globals().keys() if not k.startswith('__')]
