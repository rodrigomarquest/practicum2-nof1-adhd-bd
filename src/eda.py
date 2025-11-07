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
    SRC_PATH = ROOT / "notebooks" / "NB-01_EDA-ANY-DAILY-overview.py"
    if SRC_PATH.exists():
        spec = importlib.util.spec_from_file_location("nb1_eda_orig", str(SRC_PATH))
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["nb1_eda_orig"] = _mod
        # Ensure the notebook module has safe defaults to avoid NameError
        # when the notebook references NB_MODE, ENV or common names at top-level.
        try:
            if not hasattr(_mod, "NB_MODE"):
                setattr(_mod, "NB_MODE", "local")
            # Prefer an existing ENV if present; otherwise use the guard ENV
            if not hasattr(_mod, "ENV"):
                setattr(_mod, "ENV", ENV)
            # Inject commonly-used modules/names so notebooks that expect them don't fail
            import os as _os

            setattr(_mod, "os", _os)
            setattr(_mod, "sys", sys)
            setattr(_mod, "Path", Path)
            # Provide a sensible INPUT_CSV fallback if notebook expects it
            if not hasattr(_mod, "INPUT_CSV"):
                # find any joined/features_daily.csv under data/etl and pick the first
                candidates = list(
                    Path("data").glob("etl/*/*/joined/features_daily.csv")
                )
                if candidates:
                    setattr(_mod, "INPUT_CSV", str(candidates[0]))
                    try:
                        setattr(_mod, "INPUT_SHA256", None)
                    except Exception:
                        pass
            # Provide a minimal dataframe and commonly-used DataFrame helpers so
            # notebooks that assume `df`, `DATE_COL`, and `numeric_cols` exist can run
            try:
                if not hasattr(_mod, "df") and hasattr(_mod, "INPUT_CSV"):
                    import pandas as _pd
                    import numpy as _np

                    p = Path(getattr(_mod, "INPUT_CSV"))
                    if p.exists():
                        try:
                            _df = _pd.read_csv(p)
                        except Exception:
                            # fallback to an empty DataFrame
                            _df = _pd.DataFrame()
                    else:
                        _df = _pd.DataFrame()
                    setattr(_mod, "df", _df)
                    # date column heuristics
                    if not hasattr(_mod, "DATE_COL"):
                        if "date" in _df.columns:
                            setattr(_mod, "DATE_COL", "date")
                        elif "date_utc" in _df.columns:
                            setattr(_mod, "DATE_COL", "date_utc")
                        else:
                            setattr(
                                _mod,
                                "DATE_COL",
                                _df.columns[0] if len(_df.columns) else None,
                            )
                    # numeric columns list
                    if not hasattr(_mod, "numeric_cols"):
                        try:
                            ncols = list(
                                _df.select_dtypes(include=[_np.number]).columns
                            )
                        except Exception:
                            ncols = []
                        setattr(_mod, "numeric_cols", ncols)
                    # segment detection
                    if not hasattr(_mod, "has_segment"):
                        setattr(_mod, "has_segment", "segment_id" in _df.columns)
                    if not hasattr(_mod, "segment_counts") and getattr(
                        _mod, "has_segment"
                    ):
                        try:
                            setattr(
                                _mod,
                                "segment_counts",
                                _df["segment_id"].value_counts().to_dict(),
                            )
                        except Exception:
                            setattr(_mod, "segment_counts", {})
            except Exception:
                # best-effort only; do not block loading the notebook
                pass
        except Exception:
            pass
        # Execute module in its own namespace; notebook may reference NB_MODE/ENV
        spec.loader.exec_module(_mod)
        names = [n for n in dir(_mod) if not n.startswith("_")]
        for _name in names:
            globals()[_name] = getattr(_mod, _name)
        return _mod
    raise ImportError("notebooks/NB-01_EDA-ANY-DAILY-overview.py not found")


def main():
    """Entrypoint for `python -m src.eda`.

    Loads the original notebook module and forwards to its main() if present.
    """
    mod = _load_notebook_module()
    if hasattr(mod, "main"):
        return mod.main()
    # Notebook module does not provide a main() entrypoint. We already executed
    # the notebook during loading (preview mode). Treat this as a successful
    # run for the QC/make flow.
    return 0


if __name__ == "__main__":
    sys.exit(main())
