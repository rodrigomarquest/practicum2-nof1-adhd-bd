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
import time
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
try:
    # new centralized zepp discovery and loaders
    from src.domains.parse_zepp_export import discover_zepp_tables
    from src.domains.activity.zepp_activity import load_zepp_activity_daily
except Exception:
    # best-effort import for different execution contexts
    try:
        from domains.parse_zepp_export import discover_zepp_tables  # type: ignore
        from domains.activity.zepp_activity import load_zepp_activity_daily  # type: ignore
    except Exception:
        discover_zepp_tables = None  # type: ignore
        load_zepp_activity_daily = None  # type: ignore

# optional tqdm for progress bars (safe fallback if not installed)
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def _tqdm(it, *a, **k):
        return it


logger = logging.getLogger("etl.activity")


def discover_apple(snapshot_dir: Path) -> list[tuple[Path, str, str]]:
    """Discover Apple sources under extracted/apple/*/apple_health_export/.

    Prefer export.xml files. If none, look for daily CSVs matching '*daily*.csv'
    under each apple_health_export folder.
    
    Returns: List of (Path, vendor, variant) tuples
    """
    base = snapshot_dir / "extracted" / "apple"
    if not base.exists():
        return []

    export_xmls: list[tuple[Path, str, str]] = []
    daily_csvs: list[tuple[Path, str, str]] = []

    # look under each variant folder
    for variant_dir in [p for p in base.iterdir() if p.is_dir()]:
        variant = variant_dir.name  # e.g., "inapp"
        ah_dir = variant_dir / "apple_health_export"
        if not ah_dir.exists():
            # also accept exports directly under variant_dir
            ah_dir = variant_dir
        # prefer explicit export.xml
        ex = ah_dir / "export.xml"
        if ex.exists():
            export_xmls.append((ex, "apple", variant))
            continue
        # otherwise collect any export.xml anywhere under this variant
        for p in ah_dir.rglob("export.xml"):
            export_xmls.append((p, "apple", variant))
        # fallback: any daily CSVs
        for p in ah_dir.rglob("*daily*.csv"):
            daily_csvs.append((p, "apple", variant))

    if export_xmls:
        return export_xmls
    return daily_csvs


def discover_zepp_csvs(snapshot_dir: Path) -> List[Path]:
    base = snapshot_dir / "extracted" / "zepp"
    if not base.exists():
        return []
    candidates = list(base.rglob("*.csv"))
    return candidates


def load_apple_daily(paths: List[Path], home_tz: str, max_records: int | None = None) -> pd.DataFrame:
    """Load daily apple activity from export.xml paths or fallback daily CSVs.

    If export.xml paths are provided, parse them (streaming) and aggregate
    per-day using home_tz for localization. If only CSVs are provided, read
    and normalize them to the target schema.
    
    Args:
        paths: List of Path objects (export.xml or CSV files)
        home_tz: User's home timezone
        max_records: Limit parsing to max_records (for testing)
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
                records_processed = 0
                should_stop = False
                for event, elem in it:
                    if should_stop:
                        break
                    tag = elem.tag
                    if tag.endswith("ActivitySummary"):
                        # ActivitySummary often contains attributes like dateComponents, date, activeEnergyBurned, appleExerciseTime, appleStandHours, and goals
                        attrs = elem.attrib
                        date_str = (
                            attrs.get("dateComponents") or  # ActivitySummary uses this (YYYY-MM-DD format)
                            attrs.get("date") or            # fallback for other nodes
                            attrs.get("startDate")          # Record nodes use this
                        )
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
                        
                        # Early filter: only process activity-relevant types
                        is_activity_type = (
                            "StepCount" in t or "stepCount" in t or "Step" in t or
                            "Distance" in t or
                            "ActiveEnergy" in t or
                            "ExerciseTime" in t or "AppleExerciseTime" in t or
                            "StandHours" in t or "StandHour" in t or "Stand" in t
                        )
                        if not is_activity_type:
                            elem.clear(); continue
                        
                        v = attrs.get("value")
                        sdt = attrs.get("startDate") or attrs.get("creationDate")
                        if not (v and sdt):
                            elem.clear(); continue
                        try:
                            dt = dt_parser.parse(sdt)
                            # UTC → home_tz conversion: ensure UTC tz is set, then convert to home_tz
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=dateutil_tz.gettz("UTC"))
                            elif dt.tzinfo != dateutil_tz.gettz("UTC"):
                                # if already has tz, convert to UTC first
                                dt = dt.astimezone(dateutil_tz.gettz("UTC"))
                            # now convert UTC to home_tz
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
                            records_processed += 1  # Count only activity-relevant records
                        elif "Distance" in t:
                            # preserve as meters if possible
                            daily[d]["apple_distance_m"] += float(sval)
                            records_processed += 1  # Count only activity-relevant records
                        elif "ActiveEnergy" in t:
                            daily[d]["apple_active_kcal"] += float(sval)
                            records_processed += 1  # Count only activity-relevant records
                        elif "ExerciseTime" in t or "AppleExerciseTime" in t:
                            daily[d]["apple_exercise_min"] += float(sval)
                            records_processed += 1  # Count only activity-relevant records
                        elif "StandHours" in t or "StandHour" in t or "Stand" in t:
                            daily[d]["apple_stand_hours"] += float(sval)
                            records_processed += 1  # Count only activity-relevant records

                        # Stop if max_records reached (for activity-relevant records only)
                        if max_records is not None and records_processed >= max_records:
                            should_stop = True

                    # free memory
                    elem.clear()
            except (KeyboardInterrupt, SystemExit):
                raise
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
        # coerce date -> datetime64 and normalize to midnight (keep dtype)
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        except Exception:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
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
    # coerce date format robustly (try ISO then dayfirst)
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        # if many NaT, try dayfirst parsing
        if dt.isna().sum() > (len(dt) // 10):
            try:
                dt2 = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                # prefer dt2 where dt is NaT
                dt = dt.fillna(dt2)
            except Exception:
                pass
        # normalize to midnight and drop tz
        try:
            dt = dt.dt.tz_localize(None).dt.normalize()
        except Exception:
            dt = dt.dt.normalize()
        df["date"] = dt

    # map possible column name variants into canonical apple_* fields
    # heuristics: step columns contain 'step'; distance contains 'distance'; energy contains 'active' or 'energy'
    mapping_candidates = {
        "apple_steps": [c for c in df.columns if "step" in c.lower()],
        "apple_distance_m": [c for c in df.columns if "distance" in c.lower()],
        "apple_active_kcal": [c for c in df.columns if "active" in c.lower() and ("kcal" in c.lower() or "energy" in c.lower() or "calor" in c.lower())],
        "apple_exercise_min": [c for c in df.columns if "exercise" in c.lower() or "appleexercisetime" in c.lower()],
        "apple_stand_hours": [c for c in df.columns if "stand" in c.lower() and "hour" in c.lower()],
        # goals
        "apple_move_goal_kcal": [c for c in df.columns if "goal" in c.lower() and ("energy" in c.lower() or "move" in c.lower())],
        "apple_exercise_goal_min": [c for c in df.columns if "goal" in c.lower() and "exercise" in c.lower()],
        "apple_stand_goal_hours": [c for c in df.columns if "goal" in c.lower() and "stand" in c.lower()],
        # rings close
        "apple_rings_close_move": [c for c in df.columns if c.lower() in ("move", "move_close", "move_closed")],
        "apple_rings_close_exercise": [c for c in df.columns if c.lower() in ("exercise", "exercise_close", "exercise_closed")],
        "apple_rings_close_stand": [c for c in df.columns if c.lower() in ("stand", "stand_close", "stand_closed")],
    }

    # build a reduced dataframe with canonical columns where possible
    reduced = pd.DataFrame()
    if "date" in df.columns:
        reduced["date"] = df["date"]
    for canon, candidates in mapping_candidates.items():
        if candidates:
            # pick first candidate column
            reduced[canon] = df[candidates[0]]
        else:
            # absent -> full of NaN
            reduced[canon] = pd.NA

    # If distance values look like kilometers (typical daily values < 50), convert to meters
    if reduced["apple_distance_m"].notna().any():
        # coerce numeric
        reduced["apple_distance_m"] = pd.to_numeric(reduced["apple_distance_m"], errors="coerce")
        sample = reduced["apple_distance_m"].dropna()
        if not sample.empty:
            median = sample.median()
            if median < 50:
                logger.warning("apple: converting distance values from km->m (heuristic median<50)")
                reduced["apple_distance_m"] = reduced["apple_distance_m"] * 1000.0

    # ensure numeric types for numeric fields
    for c in ["apple_steps", "apple_distance_m", "apple_active_kcal", "apple_exercise_min", "apple_stand_hours",
              "apple_move_goal_kcal", "apple_exercise_goal_min", "apple_stand_goal_hours",
              "apple_rings_close_move", "apple_rings_close_exercise", "apple_rings_close_stand"]:
        if c in reduced.columns:
            reduced[c] = pd.to_numeric(reduced[c], errors="coerce")

    # group by date and aggregate
    if "date" in reduced.columns:
        agg_map = {
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
        grouped = reduced.groupby("date").agg(agg_map).reset_index()
    else:
        grouped = pd.DataFrame()

    # ensure date is datetime64 and normalized
    if "date" in grouped.columns:
        grouped["date"] = pd.to_datetime(grouped["date"]).dt.normalize()

    return grouped


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

    # If apple_df empty -> return empty DataFrame with canonical columns
    final_cols = [
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
        "act_steps",
        "act_active_min",
    ]

    if apple_df is None or apple_df.empty:
        seed = pd.DataFrame(columns=final_cols)
        return seed

    df = apple_df.copy()
    # Expect 'date' to be datetime64; coerce if necessary
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

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
    if "date" in df.columns and present:
        grouped = df.groupby("date").agg(present).reset_index()
    else:
        # ensure we still produce rows grouping by date if present but with no fields
        if "date" in df.columns:
            grouped = df[["date"]].drop_duplicates().reset_index(drop=True)
        else:
            grouped = pd.DataFrame()

    # Derived fields mapping
    if "apple_exercise_min" in grouped.columns:
        grouped["act_active_min"] = grouped["apple_exercise_min"].astype(float)
    else:
        grouped["act_active_min"] = pd.NA

    if "apple_steps" in grouped.columns:
        grouped["act_steps"] = grouped["apple_steps"].astype(float)
    else:
        grouped["act_steps"] = pd.NA

    # Ensure all final columns exist and have appropriate numeric dtypes where possible
    for c in final_cols:
        if c not in grouped.columns:
            grouped[c] = pd.NA

    # Remove negative values (clip at 0)
    num_cols = [c for c in final_cols if c != "date" and c in grouped.columns]
    for c in num_cols:
        try:
            grouped[c] = pd.to_numeric(grouped[c], errors="coerce")
            grouped[c] = grouped[c].clip(lower=0)
        except Exception:
            # leave as-is if conversion fails
            pass

    # reorder
    grouped = grouped[final_cols]
    return grouped


def write_seed(df: pd.DataFrame, snapshot_dir: Path, dry_run: bool, vendor: str | None = None, variant: str | None = None) -> Path:
    """Write activity seed features to appropriate directory structure.
    
    If vendor and variant are provided, writes to:
      features/<domain>/<vendor>/<variant>/features_daily.csv
    Otherwise writes to:
      features/<domain>/features_daily.csv (legacy)
    """
    if vendor and variant:
        out = snapshot_dir / "features" / "activity" / vendor / variant / "features_daily.csv"
    else:
        out = snapshot_dir / "features" / "activity" / "features_daily.csv"
    
    if dry_run:
        print(f"DRY RUN: would write activity seed -> {out} (rows={len(df)})")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use canonical helper
    # Keep 'date' as datetime until write; serialize to YYYY-MM-DD strings now
    df_out = df.copy()
    if "date" in df_out.columns and pd.api.types.is_datetime64_any_dtype(df_out["date"]):
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    # Do not coerce NaN -> empty; allow missing values to remain
    write_csv(df_out, out, dry_run=False)
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
    parser.add_argument("--max-records", type=int, default=None, help="Limit parsing to max_records (for testing)")
    parser.add_argument("--allow-empty", type=int, default=1,
                        help="When 0: exit with code 2 if no rows were produced. When 1: write empty features file and exit 0.")
    args = parser.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)
    allow_empty = bool(getattr(args, "allow_empty", 1))
    max_records = getattr(args, "max_records", None)

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
    apple_paths_with_meta = discover_apple(snapshot_dir)
    print(f"INFO: apple sources found: {len(apple_paths_with_meta)}")
    
    # Reorganize for processing: group by vendor/variant
    apple_by_variant = {}  # {(vendor, variant): [paths]}
    for path, vendor, variant in apple_paths_with_meta:
        key = (vendor, variant)
        if key not in apple_by_variant:
            apple_by_variant[key] = []
        apple_by_variant[key].append(path)

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

    # Process each vendor/variant combination separately
    apple_dfs = {}  # {(vendor, variant): dataframe}
    for (vendor, variant), paths in apple_by_variant.items():
        df = load_apple_daily(paths, home_tz, max_records=max_records) if paths else pd.DataFrame()
        apple_dfs[(vendor, variant)] = df
        print(f"INFO: {vendor}/{variant}: {len(df)} rows")
    
    # Combine all apple sources into single dataframe
    apple_df = pd.concat(
        [df for df in apple_dfs.values() if not df.empty],
        ignore_index=True,
        sort=False
    ) if apple_dfs else pd.DataFrame()

    # Warn if ActivitySummary file not present (many older snapshots are per-metric only)
    try:
        has_activitysummary = any("activitysummary" in path.name.lower() for path, _, _ in apple_paths_with_meta)
    except Exception:
        has_activitysummary = False
    if not has_activitysummary:
        logger.warning("ActivitySummary not found in apple exports; building seed from per-metric files only")

    # Zepp: discover per-domain tables under extracted/zepp/cloud
    zepp_root = snapshot_dir / "extracted" / "zepp" / "cloud"
    tables = {}
    if discover_zepp_tables is not None:
        try:
            tables = discover_zepp_tables(zepp_root)
        except Exception:
            tables = {}
    else:
        # fallback: simple rglob of CSVs (legacy behavior)
        zepp_paths = list((snapshot_dir / "extracted" / "zepp").rglob("*.csv")) if (snapshot_dir / "extracted" / "zepp").exists() else []
        if zepp_paths:
            tables = {"ACTIVITY": zepp_paths}

    zepp_has_files = bool(tables)
    zepp_available = bool(os.getenv("ZEPP_ZIP_PASSWORD")) and zepp_has_files
    if zepp_has_files:
        print(f"INFO: zepp tables discovered: {', '.join([f'{k}={len(v)}' for k,v in tables.items()])}")
    else:
        print("INFO: zepp tables discovered: 0")

    if load_zepp_activity_daily is not None and zepp_available:
        try:
            zepp_df = load_zepp_activity_daily(tables, home_tz, max_records=max_records)
        except Exception:
            zepp_df = pd.DataFrame()
    else:
        if zepp_has_files and not os.getenv("ZEPP_ZIP_PASSWORD"):
            print("INFO: ZEPP_ZIP_PASSWORD not set; zepp data will be ignored")
        zepp_df = pd.DataFrame()

    seed = build_activity_seed(apple_df, zepp_df)

    # QC summary with enhanced metadata
    start_time = time.time()
    if not seed.empty:
        date_min = seed["date"].min()
        date_max = seed["date"].max()
        n_days = seed.shape[0]
        mean_steps = float(seed["act_steps"].replace({pd.NA: None}).dropna().mean()) if "act_steps" in seed.columns else 0.0
        p95_kcal = float(seed["apple_active_kcal"].dropna().quantile(0.95)) if "apple_active_kcal" in seed.columns else 0.0
        # Coverage: days with at least one non-NaN metric
        n_days_with_data = 0
        for _, row in seed.iterrows():
            row_dict = row.to_dict()
            if any(pd.notna(v) for k, v in row_dict.items() if k != "date"):
                n_days_with_data += 1
        expected_days = (pd.to_datetime(date_max) - pd.to_datetime(date_min)).days + 1 if not seed.empty else 0
        coverage_pct = round((n_days_with_data / expected_days * 100), 2) if expected_days > 0 else 0.0
    else:
        date_min = ""
        date_max = ""
        n_days = 0
        mean_steps = 0.0
        p95_kcal = 0.0
        n_days_with_data = 0
        expected_days = 0
        coverage_pct = 0.0

    source_summary = []
    source_summary.append("apple:OK" if not apple_df.empty else "apple:SKIP")
    if zepp_has_files:
        source_summary.append("zepp:OK" if zepp_available and not zepp_df.empty else "zepp:SKIP")
    
    # Collect source file names
    input_paths = [p for p, _, _ in apple_paths_with_meta] if apple_paths_with_meta else []
    if zepp_has_files and zepp_available:
        for paths_list in tables.values():
            input_paths.extend(paths_list)
    source_files_str = ";".join([p.name for p in input_paths]) if input_paths else "none"
    
    summary = {
        "date_min": str(date_min),
        "date_max": str(date_max),
        "n_days": int(n_days),
        "n_days_with_data": int(n_days_with_data),
        "coverage_pct": float(coverage_pct),
        "mean_steps": float(mean_steps) if not np.isnan(mean_steps) else 0.0,
        "p95_active_kcal": float(p95_kcal) if not np.isnan(p95_kcal) else 0.0,
        "source_summary": ";".join(source_summary),
        "source_files": source_files_str,
        "processing_time_sec": round(time.time() - start_time, 2),
        "processed_at": pd.Timestamp.now(tz="UTC").isoformat(),
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

    # real run: always write features file (headers if empty) and QC
    # Log summary at info level
    logger.info(f"activity seed: rows={len(seed)} date_min={summary.get('date_min')} date_max={summary.get('date_max')}")

    # Write features per vendor/variant combination
    for (vendor, variant), df in apple_dfs.items():
        if not df.empty:
            write_seed(df, snapshot_dir, dry_run=False, vendor=vendor, variant=variant)
    
    # Write Zepp features if available
    if not zepp_df.empty:
        write_seed(zepp_df, snapshot_dir, dry_run=False, vendor="zepp", variant="cloud")
    
    # Also write combined to legacy location for backward compatibility
    write_seed(seed, snapshot_dir, dry_run=False)
    write_qc(summary, snapshot_dir, dry_run=False)

    if len(seed) == 0 and not allow_empty:
        print("INFO: activity_from_extracted: no seed rows found and --allow-empty=0")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
