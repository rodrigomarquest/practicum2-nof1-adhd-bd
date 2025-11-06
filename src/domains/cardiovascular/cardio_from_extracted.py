"""Simple Zepp cardio (heart-rate) per-day aggregator.

This is intentionally minimal: reads HEARTRATE / HEARTRATE_AUTO CSVs and
aggregates mean/max/count per day, returning Zepp-prefixed columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd
import argparse
import os
from ..parse_zepp_export import discover_zepp_tables
from lib.io_guards import write_csv

try:
    import src.etl_pipeline as etl
except Exception:
    try:
        import etl_pipeline as etl  # type: ignore
    except Exception:
        etl = None
from ..common.io import etl_snapshot_root

logger = logging.getLogger("etl.cardio")


def _read(paths: List[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df


def load_zepp_cardio_daily(tables: Dict[str, List[Path]], home_tz: str) -> pd.DataFrame:
    # read both HEARTRATE and HEARTRATE_AUTO
    paths = []
    if "HEARTRATE" in tables:
        paths.extend(tables["HEARTRATE"])
    if "HEARTRATE_AUTO" in tables:
        paths.extend(tables["HEARTRATE_AUTO"])
    if not paths:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    df = _read(paths)
    if df.empty:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    # try to find timestamp and hr value
    cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for cand in ("timestamp", "time", "ts", "date"):
        if cand in cols:
            ts_col = cols[cand]
            break
    hr_col = None
    for cand in ("value", "heartrate", "hr", "bpm"):
        if cand in cols:
            hr_col = cols[cand]
            break

    if ts_col is None or hr_col is None:
        logger.info("zepp cardio insufficient columns: rows=0")
        return pd.DataFrame()

    try:
        ser = pd.to_datetime(df[ts_col], errors="coerce")
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        ser = ser.dt.tz_convert(home_tz)
        df["date"] = ser.dt.date.astype(str)
    except Exception:
        df["date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.date.astype(str)

    df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
    grouped = df.groupby("date")[hr_col].agg(["mean", "max", "count"]).reset_index()
    grouped = grouped.rename(columns={
        "mean": "zepp_hr_mean",
        "max": "zepp_hr_max",
        "count": "zepp_n_hr",
    })
    logger.info("zepp cardio rows=%d", len(grouped))
    return grouped


def _write_qc(snap_dir: Path, domain: str, df: pd.DataFrame, dry_run: bool = False) -> None:
    qc_dir = snap_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / f"{domain}_seed_qc.csv"
    if df is None or getattr(df, "empty", True):
        meta = {"date_min": "", "date_max": "", "n_days": 0, "n_rows": 0}
    else:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        date_min = dates.min().date().isoformat() if not dates.empty else ""
        date_max = dates.max().date().isoformat() if not dates.empty else ""
        n_days = df["date"].nunique() if "date" in df.columns else 0
        meta = {"date_min": date_min, "date_max": date_max, "n_days": int(n_days), "n_rows": int(len(df))}
    meta_df = pd.DataFrame([meta])
    write_csv(meta_df, qc_path, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(prog="cardio_from_extracted")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dry-run", dest="dry_run", type=int, default=1)
    args = ap.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)

    # resolve snapshot if needed
    if snap == "auto":
        if etl is None:
            print("[error] cannot resolve 'auto' snapshot; etl resolver not available")
            return 3
        snap = etl.resolve_snapshot(snap, pid)

    snap_dir = etl_snapshot_root(pid, snap)
    zepp_root = snap_dir / "extracted" / "zepp" / "cloud"
    tables = discover_zepp_tables(zepp_root) if zepp_root.exists() else {}

    df = load_zepp_cardio_daily(tables, "UTC") if tables else pd.DataFrame()

    out = snap_dir / "features" / "cardio" / "features_daily.csv"

    # Dry-run: report counts but do not write
    if dry_run:
        print(f"[dry-run] cardio seed — OUT={out} rows={0 if df is None else len(df)}")
        _write_qc(snap_dir, "cardio", df, dry_run=True)
        return 0

    # Real run
    if df is None or getattr(df, "empty", True):
        print("[info] cardio seed: no rows")
        _write_qc(snap_dir, "cardio", pd.DataFrame(), dry_run=False)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, out, dry_run=False)
    _write_qc(snap_dir, "cardio", df, dry_run=False)
    print(f"[ok] wrote cardio features -> {out} rows={len(df)}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
