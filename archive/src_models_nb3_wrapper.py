"""Archival copy of the `src.models_nb3` dynamic wrapper.

Kept for history after moving or replacing the original module.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / 'src' / 'models_nb3.py'
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location('nb3_orig', str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules['nb3_orig'] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith('_')]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError('src/models_nb3.py not found; archival wrapper preserved original behavior.')
