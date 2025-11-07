"""Minimal aggregator for joined features.

Reads data/etl/<PID>/<SNAPSHOT>/joined/joined_features_daily.csv,
validates and writes a canonical aggregate to
reports/aggregates/<PID>/<SNAPSHOT>/joined_aggregate.csv.

This is intentionally minimal and idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import traceback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aggregate_joined")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dry-run", type=int, default=1)
    args = ap.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)

    in_path = Path("data") / "etl" / pid / snap / "joined" / "joined_features_daily.csv"
    out_dir = Path("reports") / "aggregates" / pid / snap
    out_path = out_dir / "joined_aggregate.csv"

    if not in_path.exists():
        print("ERROR: joined not found", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(in_path, parse_dates=["date"], dtype=dict())
    except Exception:
        print("ERROR: failed to read joined file", file=sys.stderr)
        traceback.print_exc()
        return 1

    # coerce date to YYYY-MM-DD and sort ascending
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    except Exception:
        # best-effort: convert to string
        df["date"] = df["date"].astype(str)

    df = df.sort_values("date", na_position="last").reset_index(drop=True)

    if dry_run:
        print(f"DRY RUN: IN={in_path} OUT={out_path} rows={len(df)}")
        return 0

    # ensure out dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # use canonical writer if available
        try:
            from lib.io_guards import write_csv

            write_csv(df, out_path, dry_run=False)
        except Exception:
            # fallback
            df.to_csv(out_path, index=False)
        print(f"wrote {out_path} rows={len(df)}")
        return 0
    except Exception:
        print("ERROR: failed to write aggregate", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
