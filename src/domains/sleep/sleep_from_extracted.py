"""Minimal Zepp sleep daily aggregator.

Reads SLEEP/*.csv and aggregates total/deep/light/rem hours per day when
available. Columns in result are prefixed with `zepp_slp_`.
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

logger = logging.getLogger("etl.sleep")


def _read(paths: List[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def load_zepp_sleep_daily(tables: Dict[str, List[Path]], home_tz: str) -> pd.DataFrame:
    if "SLEEP" not in tables:
        logger.info("zepp sleep rows=0")
        return pd.DataFrame()
    df = _read(tables["SLEEP"])
    if df.empty:
        logger.info("zepp sleep rows=0")
        return pd.DataFrame()

    # normalize date column
    if "date" not in df.columns:
        for c in ("start_time", "start_date", "day", "timestamp"):
            if c in df.columns:
                try:
                    df["date"] = pd.to_datetime(df[c], errors="coerce").dt.date.astype(str)
                    break
                except Exception:
                    continue
    # common columns: total_hours, deep_hours, light_hours, rem_hours
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    out["zepp_slp_total_h"] = pd.to_numeric(df.get("total_hours", df.get("duration_h", df.get("duration", 0))), errors="coerce").fillna(0)
    out["zepp_slp_deep_h"] = pd.to_numeric(df.get("deep_hours", 0), errors="coerce").fillna(0)
    out["zepp_slp_light_h"] = pd.to_numeric(df.get("light_hours", 0), errors="coerce").fillna(0)
    out["zepp_slp_rem_h"] = pd.to_numeric(df.get("rem_hours", 0), errors="coerce").fillna(0)

    agg = {k: "sum" for k in out.columns if k != "date"}
    out = out.groupby("date").agg(agg).reset_index()
    logger.info("zepp sleep rows=%d", len(out))
    return out


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

    ap = argparse.ArgumentParser(prog="sleep_from_extracted")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dry-run", dest="dry_run", type=int, default=1)
    args = ap.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)

    if snap == "auto":
        if etl is None:
            print("[error] cannot resolve 'auto' snapshot; etl resolver not available")
            return 3
        snap = etl.resolve_snapshot(snap, pid)

    snap_dir = etl_snapshot_root(pid, snap)
    zepp_root = snap_dir / "extracted" / "zepp" / "cloud"
    tables = discover_zepp_tables(zepp_root) if zepp_root.exists() else {}

    df = load_zepp_sleep_daily(tables, "UTC") if tables else pd.DataFrame()

    out = snap_dir / "features" / "sleep" / "features_daily.csv"

    if dry_run:
        print(f"[dry-run] sleep seed — OUT={out} rows={0 if df is None else len(df)}")
        _write_qc(snap_dir, "sleep", df, dry_run=True)
        return 0

    if df is None or getattr(df, "empty", True):
        print("[info] sleep seed: no rows")
        _write_qc(snap_dir, "sleep", pd.DataFrame(), dry_run=False)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, out, dry_run=False)
    _write_qc(snap_dir, "sleep", df, dry_run=False)
    print(f"[ok] wrote sleep features -> {out} rows={len(df)}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
