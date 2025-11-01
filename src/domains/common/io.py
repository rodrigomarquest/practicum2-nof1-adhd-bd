from __future__ import annotations
from pathlib import Path
import shutil
from typing import Tuple


def etl_snapshot_root(pid: str, snapshot: str) -> Path:
    """Return canonical snapshot root under data/etl/<PID>/<SNAPSHOT>/.

    Historically the layout used an intermediate 'snapshots' directory
    (data/etl/<pid>/snapshots/<snapshot>/). The canonical v4 layout places
    snapshots directly under the participant directory: data/etl/<pid>/<snapshot>/.
    This helper centralizes that decision so callers don't need to hardcode the
    intermediate component.
    """
    pid = str(pid)
    snap = str(snapshot)
    return Path("data") / "etl" / pid / snap


def per_metric_dir(pid: str, snapshot: str) -> Path:
    p = etl_snapshot_root(pid, snapshot) / "per-metric"
    p.mkdir(parents=True, exist_ok=True)
    return p


def joined_dir(pid: str, snapshot: str) -> Path:
    p = etl_snapshot_root(pid, snapshot) / "joined"
    p.mkdir(parents=True, exist_ok=True)
    return p


def extracted_dir(pid: str, snapshot: str, device: str = "apple") -> Path:
    p = etl_snapshot_root(pid, snapshot) / "extracted" / device
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(pid: str, snapshot: str) -> Path:
    p = etl_snapshot_root(pid, snapshot) / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def migrate_from_data_ai_if_present(pid: str, snapshot: str) -> Tuple[bool, str]:
    """If same-snapshot files exist under data_ai/<pid>/snapshots/<snapshot>/, move them
    into data/etl/<pid>/snapshots/<snapshot>/ preserving structure and return (True, message).
    This is a safe, on-demand migration done only when data/etl missing the joined files.
    """
    src = Path("data_ai") / pid / "snapshots" / snapshot
    dst = etl_snapshot_root(pid, snapshot)
    moved_any = False
    msgs = []
    if not src.exists():
        return False, "no data_ai snapshot"
    # consider only files directly under src and its subdirs
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            dstp = dst / rel
            if dstp.exists():
                continue
            dstp.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(dstp))
                moved_any = True
                msgs.append(f"moved {p} -> {dstp}")
            except Exception as e:
                msgs.append(f"failed {p}: {e}")
    if moved_any:
        return True, "; ".join(msgs)
    return False, "no files moved"


"""Lightweight I/O utilities (stubs)."""
import os
import warnings
import pandas as pd


def read_csv_if_exists(path: str, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs) if os.path.exists(path) else None
    except Exception as e:
        warnings.warn(f"read_csv_if_exists falhou: {e}")
        return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_local_dt(series, tz: str):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(tz)
    except Exception:
        return s


def date_col(series):
    return pd.to_datetime(series).dt.date
