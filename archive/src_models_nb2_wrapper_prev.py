"""Wrapper exposing `notebooks/NB2_Baseline_and_LSTM.py` as `src.models_nb2`.

The wrapper dynamically loads the script so downstream code can `from src.models_nb2 import ...`.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'notebooks' / 'NB2_Baseline_and_LSTM.py'
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location('nb2_orig', str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules['nb2_orig'] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith('_')]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError('notebooks/NB2_Baseline_and_LSTM.py not found')
