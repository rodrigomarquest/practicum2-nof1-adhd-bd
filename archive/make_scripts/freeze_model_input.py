#!/usr/bin/env python3
"""Freeze model input (extracted from Makefile heredoc).

Writes a model_input/features_daily.csv and meta.json/lock.ok in the snapshot dir.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from make_scripts.common import parse_pid_snap


def main(argv=None):
    pid, snap, policy = parse_pid_snap(argv)
    base = Path(f"data_ai/{pid}/snapshots/{snap}")
    join = base / "hybrid_join" / policy / "join_hybrid_daily.csv"
    feat = base / "features_daily.csv"
    outd = base / "model_input" / policy
    if not join.exists():
        raise SystemExit("join_hybrid_daily.csv missing for this POLICY")
    if not feat.exists():
        raise SystemExit("features_daily.csv missing; run ETL Apple first")
    outd.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(join, dtype={"policy": "string"}, parse_dates=["date"]) 
    f = pd.read_csv(feat, parse_dates=["date"]) 
    cols = [c for c in f.columns if c != "sleep_minutes_hybrid"]
    df = f[cols].merge(j[["date", "sleep_minutes_hybrid", "sleep_source", "policy"]], on="date", how="left")
    df.to_csv(outd / "features_daily.csv", index=False)
    meta = {
        "pid": pid,
        "snapshot": snap,
        "policy": policy,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    (outd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (outd / "lock.ok").write_text("frozen\n")
    print(f"✅ model_input frozen → {outd}")


if __name__ == "__main__":
    main()
