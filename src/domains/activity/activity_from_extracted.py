"""Seed activity features directly from extracted/ (Apple + optional Zepp).

Produces: data/etl/<PID>/<SNAPSHOT>/features/activity/features_daily.csv
and QC summary at qc/activity_seed_qc.csv.

Minimal, idempotent, dry-run aware. Uses canonical write helper
`lib.io_guards.write_csv` to avoid accidental writes to data_ai/.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

try:
    # helpers for snapshot resolution
    import src.etl_pipeline as etl
except Exception:
    try:
        import etl_pipeline as etl  # type: ignore
    except Exception:
        etl = None

from lib.io_guards import write_csv
import json
from collections import defaultdict
from dateutil import parser as dt_parser
from dateutil import tz as dateutil_tz

# optional tqdm for progress bars (safe fallback if not installed)
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def _tqdm(it, *a, **k):
        return it


logger = logging.getLogger("etl.activity")


def discover_apple(snapshot_dir: Path) -> List[Path]:
    """Discover Apple sources under extracted/apple/*/apple_health_export/.

    Prefer export.xml files. If none, look for daily CSVs matching '*daily*.csv'
    under each apple_health_export folder. Returns list of Paths (exports first).
    """
    base = snapshot_dir / "extracted" / "apple"
    if not base.exists():
        return []

    export_xmls: List[Path] = []
    daily_csvs: List[Path] = []

    # look under each variant folder
    for variant_dir in [p for p in base.iterdir() if p.is_dir()]:
        ah_dir = variant_dir / "apple_health_export"
        if not ah_dir.exists():
            # also accept exports directly under variant_dir
            ah_dir = variant_dir
        # prefer explicit export.xml
        ex = ah_dir / "export.xml"
        if ex.exists():
            export_xmls.append(ex)
            continue
        # otherwise collect any export.xml anywhere under this variant
        for p in ah_dir.rglob("export.xml"):
            export_xmls.append(p)
        # fallback: any daily CSVs
        for p in ah_dir.rglob("*daily*.csv"):
            daily_csvs.append(p)

    if export_xmls:
        return export_xmls
    return daily_csvs


def discover_zepp_csvs(snapshot_dir: Path) -> List[Path]:
    base = snapshot_dir / "extracted" / "zepp"
    if not base.exists():
        return []
    candidates = list(base.rglob("*.csv"))
    return candidates


def load_apple_daily(paths: List[Path], home_tz: str) -> pd.DataFrame:
    """Load daily apple activity from export.xml paths or fallback daily CSVs.

    If export.xml paths are provided, parse them (streaming) and aggregate
    per-day using home_tz for localization. If only CSVs are provided, read
    and normalize them to the target schema.
    """
    if not paths:
        return pd.DataFrame()

    # detect whether these are export.xml files (check first path)
    if any(p.name.lower() == "export.xml" for p in paths) or any(p.suffix.lower() == ".xml" for p in paths):
        # parse export.xml streaming to collect ActivitySummary and Record nodes
        daily = defaultdict(lambda: defaultdict(float))
        # also track presence of goal/close fields (use last/max)
        extras = defaultdict(dict)

        for xml in paths:
            try:
                disable = not (os.getenv("ETL_TQDM") == "1" and not os.getenv("CI"))
                it = _tqdm(ET.iterparse(str(xml), events=("end",)), desc=f"Apple export.xml ({xml.name})", disable=disable)
                for event, elem in it:
                    tag = elem.tag
                    if tag.endswith("ActivitySummary"):
                        # ActivitySummary often contains attributes like date, activeEnergyBurned, appleExerciseTime, appleStandHours, and goals
                        attrs = elem.attrib
                        date_str = attrs.get("date") or attrs.get("startDate")
                        if date_str:
                            # assume ISO date or YYYY-MM-DD
                            try:
                                d = str(pd.to_datetime(date_str).date())
                            except Exception:
                                d = date_str
                            # map known attributes
                            if "activeEnergyBurned" in attrs:
                                daily[d]["apple_active_kcal"] = float(attrs.get("activeEnergyBurned", 0))
                            if "appleExerciseTime" in attrs:
                                daily[d]["apple_exercise_min"] = float(attrs.get("appleExerciseTime", 0))
                            if "appleStandHours" in attrs:
                                daily[d]["apple_stand_hours"] = float(attrs.get("appleStandHours", 0))
                            # goals
                            if "activeEnergyBurnedGoal" in attrs:
                                extras[d]["apple_move_goal_kcal"] = float(attrs.get("activeEnergyBurnedGoal", 0))
                            if "appleExerciseTimeGoal" in attrs:
                                extras[d]["apple_exercise_goal_min"] = float(attrs.get("appleExerciseTimeGoal", 0))
                            if "appleStandHoursGoal" in attrs:
                                extras[d]["apple_stand_goal_hours"] = float(attrs.get("appleStandHoursGoal", 0))
                            # rings close heuristics
                            if "move" in attrs:
                                extras[d]["apple_rings_close_move"] = int(attrs.get("move") in ("1", "true", "True"))
                            if "exercise" in attrs:
                                extras[d]["apple_rings_close_exercise"] = int(attrs.get("exercise") in ("1", "true", "True"))
                            if "stand" in attrs:
                                extras[d]["apple_rings_close_stand"] = int(attrs.get("stand") in ("1", "true", "True"))
                    elif tag.endswith("Record") or tag.endswith("Workout"):
                        attrs = elem.attrib
                        t = attrs.get("type", "")
                        v = attrs.get("value")
                        sdt = attrs.get("startDate") or attrs.get("creationDate")
                        if not (t and v and sdt):
                            elem.clear(); continue
                        try:
                            dt = dt_parser.parse(sdt)
                            # localize to home_tz then take date
                            if dt.tzinfo is None:
                                # assume UTC if no tz
                                dt = dt.replace(tzinfo=dateutil_tz.gettz("UTC"))
                            local_dt = dt.astimezone(dateutil_tz.gettz(home_tz or "UTC"))
                            d = str(local_dt.date())
                        except Exception:
                            try:
                                d = str(pd.to_datetime(sdt).date())
                            except Exception:
                                elem.clear(); continue

                        # map type substrings to columns
                        sval = None
                        try:
                            sval = float(v)
                        except Exception:
                            try:
                                sval = int(v)
                            except Exception:
                                sval = None

                        if sval is None:
                            elem.clear(); continue

                        if "StepCount" in t or "stepCount" in t or "Step" in t:
                            daily[d]["apple_steps"] += int(sval)
                        elif "Distance" in t:
                            # preserve as meters if possible
                            daily[d]["apple_distance_m"] += float(sval)
                        elif "ActiveEnergy" in t:
                            daily[d]["apple_active_kcal"] += float(sval)
                        elif "ExerciseTime" in t or "AppleExerciseTime" in t:
                            daily[d]["apple_exercise_min"] += float(sval)
                        elif "StandHours" in t or "StandHour" in t or "Stand" in t:
                            daily[d]["apple_stand_hours"] += float(sval)

                    # free memory
                    elem.clear()
            except Exception:
                logger.warning(f"failed to parse export.xml: {xml}")

        # build DataFrame from daily dicts
        rows = []
        for d, metrics in sorted(daily.items()):
            row = {"date": d}
            row.update(metrics)
            # merge extras
            if d in extras:
                row.update(extras[d])
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        # ensure canonical columns
        return out

    # fallback: CSVs
    parts = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # normalize date column
            if "date" not in df.columns:
                for c in ("day", "local_date", "start_date"):
                    if c in df.columns:
                        df = df.rename(columns={c: "date"})
                        break
            parts.append(df)
        except Exception:
            logger.warning(f"failed to read apple CSV: {p}")
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    # coerce date format
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        except Exception:
            df["date"] = df["date"].astype(str)
    return df


def aggregate_zepp(paths: List[Path]) -> pd.DataFrame:
    parts = []
    if paths:
        disable = not (os.getenv("ETL_TQDM") == "1" and not os.getenv("CI"))
        for p in _tqdm(paths, desc="Zepp daily CSVs", disable=disable):
            try:
                df = pd.read_csv(p)
                parts.append(df)
            except Exception:
                logger.info(f"zepp: failed to read {p}; skipping")
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    # ensure date column
    if "date" not in df.columns:
        for c in ("day", "local_date", "start_date"):
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    return df


def build_activity_seed(apple_df: pd.DataFrame, zepp_df: pd.DataFrame) -> pd.DataFrame:
    # Merge/aggregate per-date; simplest approach: use apple as principal
    out_cols = [
        "date",
        "apple_steps",
        "apple_distance_m",
        "apple_active_kcal",
        "apple_exercise_min",
        "apple_stand_hours",
        "apple_move_goal_kcal",
        "apple_exercise_goal_min",
        "apple_stand_goal_hours",
        "apple_rings_close_move",
        "apple_rings_close_exercise",
        "apple_rings_close_stand",
    ]

    if apple_df is None or apple_df.empty:
        seed = pd.DataFrame(columns=out_cols)
        return seed

    # Ensure date is formatted YYYY-MM-DD
    df = apple_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    # Aggregate by date taking sum for steps/kcal and max for goals
    agg = {
        "apple_steps": "sum",
        "apple_distance_m": "sum",
        "apple_active_kcal": "sum",
        "apple_exercise_min": "sum",
        "apple_stand_hours": "sum",
        "apple_move_goal_kcal": "max",
        "apple_exercise_goal_min": "max",
        "apple_stand_goal_hours": "max",
        "apple_rings_close_move": "max",
        "apple_rings_close_exercise": "max",
        "apple_rings_close_stand": "max",
    }
    present = {k: v for k, v in agg.items() if k in df.columns}
    grouped = df.groupby("date").agg(present).reset_index()

    # Derived fields
    if "apple_exercise_min" in grouped.columns:
        grouped["act_active_min"] = grouped["apple_exercise_min"].astype(float)
    else:
        grouped["act_active_min"] = np.nan

    if "apple_steps" in grouped.columns:
        grouped["act_steps"] = grouped["apple_steps"].astype(float)
    else:
        grouped["act_steps"] = np.nan

    # Reorder and ensure columns
    final_cols = ["date"] + out_cols[1:] + ["act_steps", "act_active_min"]
    for c in final_cols:
        if c not in grouped.columns:
            grouped[c] = pd.NA

    return grouped[final_cols]


def write_seed(df: pd.DataFrame, snapshot_dir: Path, dry_run: bool) -> Path:
    out = snapshot_dir / "features" / "activity" / "features_daily.csv"
    if dry_run:
        print(f"DRY RUN: would write activity seed -> {out} (rows={len(df)})")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use canonical helper
    write_csv(df.fillna(""), out, dry_run=False)
    return out


def write_qc(summary: dict, snapshot_dir: Path, dry_run: bool) -> None:
    qc_dir = snapshot_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / "activity_seed_qc.csv"
    if dry_run:
        print(f"DRY RUN: would write QC -> {qc_path} : {summary}")
        return
    pd.DataFrame([summary]).to_csv(qc_path, index=False)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="activity_from_extracted")
    parser.add_argument("--pid", required=False)
    parser.add_argument("--participant", dest="pid", required=False)
    parser.add_argument("--snapshot", default="auto")
    parser.add_argument("--dry-run", type=int, default=1)
    args = parser.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)

    if etl is not None:
        if snap == "auto":
            snap = etl.resolve_snapshot(snap, pid)
        snapshot_dir = etl.etl_snapshot_root(pid, snap)
    else:
        # fallback to path composition
        snapshot_dir = Path("data") / "etl" / pid / snap

    print(f"INFO: activity_from_extracted start snapshot={snapshot_dir} dry_run={dry_run}")
    try:
        if os.getenv("ETL_TQDM") == "1" and not os.getenv("CI"):
            print("INFO: progress bars enabled (ETL_TQDM=1)")
    except Exception:
        pass

    # Discover Apple exports (prefer export.xml under variant/apple_health_export/)
    apple_paths = discover_apple(snapshot_dir)
    print(f"INFO: apple sources found: {len(apple_paths)}")

    # determine home_tz from participant profile if present
    home_tz = "UTC"
    try:
        profile_path = Path("data") / "config" / f"{pid}_profile.json"
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as fh:
                prof = json.load(fh)
                home_tz = prof.get("home_tz", home_tz)
    except Exception:
        logger.info("could not read participant profile for home_tz; defaulting to UTC")

    apple_df = load_apple_daily(apple_paths, home_tz) if apple_paths else pd.DataFrame()

    # Zepp: only consider if password set and files exist
    zepp_paths = discover_zepp_csvs(snapshot_dir)
    zepp_available = bool(os.getenv("ZEPP_ZIP_PASSWORD")) and bool(zepp_paths)
    if zepp_paths:
        print(f"INFO: zepp sources found: {len(zepp_paths)} (will use if ZEPP_ZIP_PASSWORD set)")
    else:
        print("INFO: zepp sources found: 0")

    zepp_df = aggregate_zepp(zepp_paths) if zepp_available else pd.DataFrame()
    if not zepp_available and zepp_paths:
        print("INFO: ZEPP_ZIP_PASSWORD not set; zepp data will be ignored")

    seed = build_activity_seed(apple_df, zepp_df)

    # QC summary
    if not seed.empty:
        date_min = seed["date"].min()
        date_max = seed["date"].max()
        n_days = seed.shape[0]
        mean_steps = float(seed["act_steps"].replace({pd.NA: None}).dropna().mean()) if "act_steps" in seed.columns else 0.0
        p95_kcal = float(seed["apple_active_kcal"].dropna().quantile(0.95)) if "apple_active_kcal" in seed.columns else 0.0
    else:
        date_min = ""
        date_max = ""
        n_days = 0
        mean_steps = 0.0
        p95_kcal = 0.0

    source_summary = []
    source_summary.append("apple:OK" if not apple_df.empty else "apple:SKIP")
    if zepp_paths:
        source_summary.append("zepp:OK" if zepp_available and not zepp_df.empty else "zepp:SKIP")
    summary = {
        "date_min": str(date_min),
        "date_max": str(date_max),
        "n_days": int(n_days),
        "mean_steps": float(mean_steps) if not np.isnan(mean_steps) else 0.0,
        "p95_active_kcal": float(p95_kcal) if not np.isnan(p95_kcal) else 0.0,
        "source_summary": ";".join(source_summary),
    }

    out_path = snapshot_dir / "features" / "activity" / "features_daily.csv"
    print(f"INFO: activity seed rows={len(seed)} out={out_path}")

    if dry_run:
        # dry-run: treat 'no data' as success but clearly log it
        if len(seed) == 0:
            print("[dry-run] nothing to do: activity seed")
        else:
            print(f"DRY RUN: would write activity seed -> {out_path} (rows={len(seed)})")
        print("DRY RUN: seed not written")
        return 0

    # real run: write outputs if any rows, QC always written when writing
    if len(seed) == 0:
        # no data in a real run -> indicate 'nothing to do' with exit code 2
        print("INFO: activity_from_extracted: no seed rows found")
        return 2

    write_seed(seed, snapshot_dir, dry_run=False)
    write_qc(summary, snapshot_dir, dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
