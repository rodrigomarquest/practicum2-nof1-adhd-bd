#!/usr/bin/env python3
"""XZ1: Produce xsource_metric_map.json mapping Apple <-> Zepp daily columns, units and fusion policy.

Writes deterministic JSON (sorted keys) to the provided joined_dir.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any


DEFAULT_MAP = {
    "hr": {
        "apple_cols": ["hr_mean", "hr_median", "hr_std"],
        "zepp_cols": ["zepp_hr_mean", "zepp_hr_median", "zepp_hr_std"],
        "unit": "bpm",
        "fusion": "quality_weighted_mean"
    },
    "hrv_sdnn": {
        "apple_cols": ["hrv_sdnn_ms"],
        "zepp_cols": ["zepp_hrv_sdnn"],
        "unit": "ms",
        "fusion": "prefer_apple_if_present_else_zepp"
    },
    "hrv_rmssd": {
        "apple_cols": [],
        "zepp_cols": ["zepp_hrv_rmssd"],
        "unit": "ms",
        "fusion": "prefer_zepp"
    },
    "sleep_total_minutes": {
        "apple_cols": ["sleep_minutes","sleep_total_minutes"],
        "zepp_cols": ["zepp_sleep_minutes"],
        "unit": "min",
        "fusion": "max_confidence"
    }
}


def atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--joined-dir", required=True, help="Output joined dir")
    p.add_argument("--force", action="store_true", help="Overwrite existing map")
    args = p.parse_args(argv)

    outdir = Path(args.joined_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "xsource_metric_map.json"

    # If exists and not force, skip (idempotent)
    if outpath.exists() and not args.force:
        print(f"SKIP: {outpath} already exists (use --force to overwrite)")
        return 0

    # Deterministic JSON: sort keys and use stable ordering
    text = json.dumps(DEFAULT_MAP, sort_keys=True, indent=2)
    atomic_write(outpath, text)
    print("WROTE:", outpath)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
