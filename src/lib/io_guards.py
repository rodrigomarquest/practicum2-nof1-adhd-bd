"""IO guards and atomic-write helpers.

Purpose
-------
Provide conservative, canonical CSV/atomic-write helpers to avoid
accidental writes into sensitive `data_ai/` paths. Callers should
use `write_csv()` or `atomic_backup_write()`; they should not import
or call the low-level guard directly.

Configuration (environment variables)
-------------------------------------
- ETL_STRICT_WRITES=1  -> keep warnings (no blocking; retained for
    compatibility with previous strict modes).
- ETL_ALLOW_DATA_AI=1  -> disables warnings and allows writes into
    `data_ai/` paths without messages.

Important
---------
The guard intentionally never raises. It only prints a warning when
writing into a `data_ai/` path (unless disabled by
`ETL_ALLOW_DATA_AI=1`). The guard is invoked only inside
`write_csv()`; other callers should not call it.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import shutil
import tempfile
import pandas as pd
from typing import Optional
import warnings
import pkgutil
import re
import importlib.util



def _should_warn_data_ai() -> bool:
    """Return True if writes into data_ai/ should emit a warning.

If ETL_ALLOW_DATA_AI=1 is set, warnings are disabled. ETL_STRICT_WRITES
is accepted for compatibility but does not change behavior (guard never
raises).
"""
    if os.getenv("ETL_ALLOW_DATA_AI") == "1":
        return False
    # reserved for future behaviors; currently warnings are the strictest action
    return True


def _guard_no_data_ai(p: Path) -> None:
    """Non-raising guard: prints a warning when `p` is inside `data_ai/`.

    This helper is internal and should only be used by `write_csv()`.
    It deliberately does not raise so callers are not forced to handle
    exceptions; the project uses write_csv() to centralize write safety.
    """
    try:
        s = str(p)
    except Exception:
        return
    if "data_ai" in s:
        if _should_warn_data_ai():
            print(f"[warn] write to data_ai/ path detected: {p} (set ETL_ALLOW_DATA_AI=1 to silence)")


def atomic_backup_write(
    df: pd.DataFrame,
    path: Path,
    backup_name: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Atomically write `df` to `path`, backing up existing file first.

    If `path` exists, copy it to <path>_<backup_name or 'prev'>.csv before
    writing. Ensures parent dirs exist. In dry_run mode only prints actions.
    """
    p = Path(path)
    if dry_run:
        print(f"DRY RUN: would ensure dir {p.parent}")
        if p.exists():
            bname = backup_name or "prev"
            # compute backup path according to backup_name rules
            backup = _compute_backup_path(p, backup_name)
            print(f"DRY RUN: would backup existing {p} -> {backup}")
        print(f"DRY RUN: would write DataFrame -> {p}")
        return

    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        backup = _compute_backup_path(p, backup_name)
        shutil.copy2(p, backup)

    # write to temporary file and move into place for atomicity
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), prefix=p.name + ".tmp.") as tf:
        tmp = Path(tf.name)
        df.to_csv(tmp, index=False)
    tmp.replace(p)


def _compute_backup_path(p: Path, backup_name: Optional[str]) -> Path:
    """Compute the backup path for a target path `p` given `backup_name`.

    Rules:
    - If backup_name is None -> use default: <stem>_prev<suffix>
    - If backup_name has a suffix (Path(backup_name).suffix != '') -> treat
      backup_name as an exact filename and return p.with_name(backup_name)
    - Else -> treat backup_name as a token and produce: <stem>_<backup_name><suffix>
    """
    if backup_name is None:
        return p.with_name(p.stem + "_prev" + p.suffix)
    bn = str(backup_name)
    from pathlib import Path as _P

    if _P(bn).suffix:
        # backup_name looks like a filename with extension -> use it exactly
        return p.with_name(bn)
    # otherwise treat as suffix token
    return p.with_name(p.stem + "_" + bn + p.suffix)


def write_csv(df: pd.DataFrame, path: Path, *, dry_run: bool = False, backup_name: Optional[str] = None) -> None:
    """Convenience wrapper that writes CSV with atomic backup semantics.

    Respects dry_run and uses `atomic_backup_write` internally.
    """
    # perform non-raising guard check (prints warning if writing into data_ai/)
    try:
        _guard_no_data_ai(Path(path))
    except Exception:
        # guard is best-effort and must not block writes
        pass
    atomic_backup_write(df=df, path=Path(path), backup_name=backup_name, dry_run=dry_run)


def ensure_joined_snapshot(snapshot_dir: Path | str) -> Path:
    """Resolve and (if needed) migrate the joined snapshot CSV path.

    Behaviour:
    - Preferred filename: <snapshot_dir>/joined/joined_features_daily.csv
    - Legacy filename:  <snapshot_dir>/joined/features_daily.csv
    If the legacy file exists and the preferred file does not, this helper
    will warn and attempt to migrate the legacy file to the preferred name
    (first trying an atomic rename, then a copy on failure).

    Returns the Path to the preferred joined file (may not exist on disk).
    """
    sd = Path(snapshot_dir)
    jdir = sd / "joined"
    old = jdir / "features_daily.csv"
    new = jdir / "joined_features_daily.csv"

    migrated = False
    # If new already exists, return it immediately
    if new.exists():
        print(f"INFO: ensure_joined_snapshot: resolved={new} migrated={migrated}")
        return new

    # If only old exists, attempt migration and warn
    if old.exists():
        warnings.warn(
            f"Found legacy joined/features_daily.csv at {old}; migrating to {new}",
            stacklevel=2,
        )
        try:
            jdir.mkdir(parents=True, exist_ok=True)
            old.replace(new)
            migrated = True
            print(f"INFO: ensure_joined_snapshot: resolved={new} migrated={migrated}")
            return new
        except Exception:
            try:
                shutil.copy2(old, new)
                migrated = True
                print(f"INFO: ensure_joined_snapshot: resolved={new} migrated={migrated}")
                return new
            except Exception:
                # Migration failed; fallthrough and return the expected new path
                print(f"INFO: ensure_joined_snapshot: resolved={new} migrated={migrated}")
                return new

    # Neither file exists; return the expected new path for callers to create
    print(f"INFO: ensure_joined_snapshot: resolved={new} migrated={migrated}")
    return new


def write_joined_features(df: pd.DataFrame, snapshot_dir: Path | str, *, dry_run: bool = False) -> Path:
    """Write the joined features DataFrame to the canonical joined path.

    Resolves the canonical path via `ensure_joined_snapshot(snapshot_dir)` and
    atomically writes `df` to that path. When the file exists, always create a
    backup using the name `joined_features_daily_prev.csv`.

    Returns the resolved joined file Path.
    """
    joined_path = ensure_joined_snapshot(snapshot_dir)
    # atomic_backup_write will create the parent dir if needed
    # Use the standardized backup name as requested.
    atomic_backup_write(df=df, path=joined_path, backup_name="joined_features_daily_prev.csv", dry_run=dry_run)
    return joined_path


if __name__ == "__main__":
    # small self-check: ensure that exact-file backup_name is respected
    t = Path("data/etl/P000001/snapshots/2025-09-29/joined/joined_features_daily.csv")
    b = _compute_backup_path(t, "joined_features_daily_prev.csv")
    # The backup filename must be exactly the provided backup_name
    assert b.name == "joined_features_daily_prev.csv", f"backup name mismatch: {b.name}"
    print("io_guards: backup-name self-check OK ->", b)
