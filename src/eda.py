"""Wrapper exposing `notebooks/NB-01_EDA-ANY-DAILY-overview.py` as `src.eda`.

This keeps the original notebook-based script in place while providing a stable import path
for v4 notebooks and scripts.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'notebooks' / 'NB-01_EDA-ANY-DAILY-overview.py'
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location('nb1_eda_orig', str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules['nb1_eda_orig'] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith('_')]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError('notebooks/NB-01_EDA-ANY-DAILY-overview.py not found')
