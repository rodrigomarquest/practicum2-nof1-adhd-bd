"""Wrapper exposing `notebooks/NB-01_EDA-ANY-DAILY-overview.py` as `src.eda`.

This keeps the original notebook-based script in place while providing a stable import path
for v4 notebooks and scripts.
"""
from pathlib import Path
import importlib.util
import sys

# Safe default ENV guard for notebooks expecting ENV
try:
    ENV  # noqa: F821
except NameError:
    ENV = {}

def _load_notebook_module():
    """Load the original notebook-based EDA module and expose its names.

    This is done lazily to avoid import-time side-effects. Callers that need the
    notebook's functions should call this helper (or use main()).
    """
    ROOT = Path(__file__).resolve().parents[1]
    SRC_PATH = ROOT / 'notebooks' / 'NB-01_EDA-ANY-DAILY-overview.py'
    if SRC_PATH.exists():
        spec = importlib.util.spec_from_file_location('nb1_eda_orig', str(SRC_PATH))
        _mod = importlib.util.module_from_spec(spec)
        sys.modules['nb1_eda_orig'] = _mod
        # Inject a safe ENV into the notebook module namespace so imports that
        # reference ENV won't raise NameError during exec.
        try:
            setattr(_mod, 'ENV', ENV)
        except Exception:
            pass
        # Execute module in its own namespace; notebook may reference ENV from here
        spec.loader.exec_module(_mod)
        names = [n for n in dir(_mod) if not n.startswith('_')]
        for _name in names:
            globals()[_name] = getattr(_mod, _name)
        return _mod
    raise ImportError('notebooks/NB-01_EDA-ANY-DAILY-overview.py not found')


def main():
    """Entrypoint for `python -m src.eda`.

    Loads the original notebook module and forwards to its main() if present.
    """
    mod = _load_notebook_module()
    if hasattr(mod, 'main'):
        return mod.main()
    raise SystemExit('eda entrypoint: no main() found in loaded module')


if __name__ == '__main__':
    sys.exit(main())
