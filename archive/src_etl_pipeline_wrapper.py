"""Preserved dynamic wrapper for `src/etl_pipeline.py` moved during v4 refactor.

This file is an archival copy of the original wrapper that re-exported the top-level
`etl_pipeline.py`. It's kept under `archive/` so history and prior helper remain available.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / 'src' / 'etl_pipeline.py'
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location('etl_pipeline_orig', str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules['etl_pipeline_orig'] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith('_')]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError('src/etl_pipeline.py not found; archival wrapper preserved original behavior.')
