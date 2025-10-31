"""Wrapper exposing `notebooks/NB3_DeepLearning.py` as `src.models_nb3`."""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "notebooks" / "NB3_DeepLearning.py"
if SRC_PATH.exists():
    spec = importlib.util.spec_from_file_location("nb3_orig", str(SRC_PATH))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules["nb3_orig"] = _mod
    spec.loader.exec_module(_mod)
    __all__ = [n for n in dir(_mod) if not n.startswith("_")]
    for _name in __all__:
        globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError("notebooks/NB3_DeepLearning.py not found")

# When executed as a script, forward execution to the loaded notebook module's main()
if __name__ == "__main__":
    # prefer the notebook's main() if available
    if "main" in globals():
        try:
            sys.exit(globals()["main"]())
        except SystemExit:
            raise
        except Exception:
            raise
    else:
        raise SystemExit("nb3 entrypoint: no main() found in loaded module")
