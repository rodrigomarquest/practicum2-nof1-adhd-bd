"""Wrapper module exposing the repository's top-level `etl_pipeline.py` under `src`.

This wrapper preserves backwards compatibility while keeping the original file in-place
so history is preserved. It dynamically loads the top-level module and re-exports symbols.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'etl_pipeline.py'
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location('etl_pipeline_orig', str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules['etl_pipeline_orig'] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith('_')]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError('Top-level etl_pipeline.py not found; please ensure repository root contains it.')
