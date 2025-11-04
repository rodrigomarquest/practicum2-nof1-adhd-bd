from __future__ import annotations

import argparse
from pathlib import Path
import sys
import shutil
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# canonical helpers
from lib.io_guards import write_csv, atomic_backup_write, ensure_joined_snapshot, write_joined_features
from lib.df_utils import zscore, rolling_cv, safe_merge_on_date, ensure_columns


def find_latest_snapshot(repo_root: Path, pid: str) -> Optional[str]:
    base = repo_root / "data" / "etl" / pid
    if not base.exists():
        return None
    # folders that look like YYYY-MM-DD
    snaps = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("20")]
    if not snaps:
        return None
    snaps.sort()
    return snaps[-1]


# local helpers use canonical ones from lib.df_utils


def compute_activity_features(df: pd.DataFrame, steps_hourly: Optional[pd.DataFrame]) -> pd.DataFrame:
    # expect df has date column (datetime or str) and apple_* columns
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # base derived
    df["act_steps_tmp"] = df.get("apple_steps")
    df["act_steps"] = df["act_steps_tmp"]
    df["act_active_min"] = df.get("apple_exercise_min")

    # act_vigorous_min: use rolling median 28d of apple_active_kcal
    kcal = df.get("apple_active_kcal")
    if kcal is None:
        df["act_vigorous_min"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        med28 = kcal.rolling(window=28, min_periods=1).median()
        vig = (kcal.fillna(0) - med28).clip(lower=0) / 10.0
        vig = vig.round().replace({np.nan: pd.NA}).astype("Int64")
        df["act_vigorous_min"] = vig

    # act_sedentary_min: (24*60 - slp_hours*60) - act_active_min - (apple_stand_hours*10)
    slp = df.get("slp_hours")
    stand = df.get("apple_stand_hours")
    active_min = df.get("act_active_min")
    if slp is None:
        slp = pd.Series(0.0, index=df.index)
    if stand is None:
        stand = pd.Series(0.0, index=df.index)
    if active_min is None:
        active_min = pd.Series(pd.NA, index=df.index)

    total_min = 24 * 60
    sed = (total_min - slp.fillna(0) * 60) - active_min.fillna(0) - (stand.fillna(0) * 10)
    sed = sed.clip(lower=0)
    df["act_sedentary_min"] = sed.replace({np.nan: pd.NA}).astype("Int64")

    # act_steps_cv7: coef var over 7-day window of act_steps
    steps_series = df["act_steps"].astype("float")
    roll_std = steps_series.rolling(window=7, min_periods=1).std()
    roll_mean = steps_series.rolling(window=7, min_periods=1).mean()
    cv7 = (roll_std / roll_mean) * 100
    df["act_steps_cv7"] = cv7.replace({np.nan: pd.NA})

    # QX missing
    req_cols = ["apple_steps", "apple_active_kcal", "apple_exercise_min", "apple_stand_hours"]
    df["qx_act_missing"] = df[req_cols].isna().any(axis=1).astype(int)

    # CMPs: need slp_hours_cv7, cv_hr_mean, cv_hrv_mean if present
    # slp_hours_cv7 may be present in df already; if not, compute from slp_hours
    if "slp_hours_cv7" not in df.columns and "slp_hours" in df.columns:
        df["slp_hours_cv7"] = df["slp_hours"].rolling(window=7, min_periods=1).std() / df["slp_hours"].rolling(window=7, min_periods=1).mean() * 100

    # cmp_activation = z(-slp_hours) + z(act_active_min) + z(cv_hr_mean) - z(cv_hrv_mean)
    cmp_activation = (
        zscore(-df.get("slp_hours", pd.Series(dtype=float)))
        + zscore(df.get("act_active_min", pd.Series(dtype=float)))
        + zscore(df.get("cv_hr_mean", pd.Series(dtype=float)))
        - zscore(df.get("cv_hrv_mean", pd.Series(dtype=float)))
    )
    df["cmp_activation"] = cmp_activation.replace({np.nan: pd.NA})

    # cmp_stability = 1 - mean([slp_hours_cv7, act_steps_cv7])
    a = df.get("slp_hours_cv7")
    b = df.get("act_steps_cv7")
    mean_ab = pd.concat([a, b], axis=1).mean(axis=1)
    df["cmp_stability"] = (1 - mean_ab).replace({np.nan: pd.NA})

    # cmp_fatigue = z(slp_debt=max(0,7-slp_hours)) + z(act_sedentary_min) - z(cv_hrv_mean)
    slp_hours = df.get("slp_hours", pd.Series(dtype=float)).fillna(0)
    slp_debt = (7 - slp_hours).clip(lower=0)
    cmp_fatigue = zscore(slp_debt) + zscore(df.get("act_sedentary_min", pd.Series(dtype=float))) - zscore(df.get("cv_hrv_mean", pd.Series(dtype=float)))
    df["cmp_fatigue"] = cmp_fatigue.replace({np.nan: pd.NA})

    # Steps-hourly derived metrics if provided
    df["act_IS"] = pd.Series([pd.NA] * len(df))
    df["act_IV"] = pd.Series([pd.NA] * len(df))
    df["act_first_move_min"] = pd.Series([pd.NA] * len(df))
    df["act_last_move_min"] = pd.Series([pd.NA] * len(df))

    if steps_hourly is not None and not steps_hourly.empty:
        # pivot to days x 24 hours
        sh = steps_hourly.copy()
        sh["date"] = pd.to_datetime(sh["date"]).dt.date
        pivot = sh.pivot_table(index="date", columns="hour", values="steps", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(columns=list(range(24)), fill_value=0)

        # for each date compute IS and IV using a 7-day rolling window ending at that date
        dates = df["date"].tolist()
        pivot = pivot.sort_index()
        for i, d in enumerate(dates):
            # select window: last 7 days up to d
            window = pivot.loc[:d].tail(7)
            if window.empty:
                continue
            # flatten to series x_i
            x = window.values.flatten()
            if np.nanstd(x) == 0:
                isv = np.nan
                iv = np.nan
            else:
                m = np.nanmean(x)
                # IS: variance of hourly means across day-hours
                hourly_means = np.nanmean(window.values, axis=0)
                is_num = np.sum((hourly_means - m) ** 2) / 24.0
                is_den = np.nansum((x - m) ** 2) / x.size
                isv = is_num / is_den if is_den != 0 else np.nan
                # IV: sum((x_{i+1}-x_i)^2)/( (N-1)*sum((x_i - m)^2)/N )
                diffs = np.diff(x)
                iv_num = np.nansum(diffs ** 2) / (x.size - 1) if x.size > 1 else np.nan
                iv_den = np.nansum((x - m) ** 2) / x.size
                iv = iv_num / iv_den if iv_den != 0 else np.nan

            df.loc[df["date"] == d, "act_IS"] = isv
            df.loc[df["date"] == d, "act_IV"] = iv

            # first/last minute of movement for that day using pivot (that day's row)
            if d in pivot.index:
                day_row = pivot.loc[d]
                nonzero = np.where(day_row.values > 0)[0]
                if nonzero.size:
                    first_h = int(nonzero[0])
                    last_h = int(nonzero[-1])
                    df.loc[df["date"] == d, "act_first_move_min"] = int(first_h * 60)
                    df.loc[df["date"] == d, "act_last_move_min"] = int(last_h * 60 + 59)

    # cleanup tmp column
    if "act_steps_tmp" in df.columns:
        df = df.drop(columns=["act_steps_tmp"])

    return df


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--pid", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dry-run", type=int, default=1)
    args = p.parse_args(argv)

    repo = Path(args.repo_root)
    pid = args.pid
    snap = args.snapshot

    if snap == "auto":
        snap = find_latest_snapshot(repo, pid)
        if snap is None:
            print("ERROR: could not auto-discover latest snapshot for", pid)
            return 1

    # paths
    snap_root = repo / "data" / "etl" / pid / snap
    # note: write safeguards (warnings for data_ai/) are handled by `write_csv()`;
    # do not perform standalone guard calls here to avoid duplicate behavior.

    # Resolve (and migrate if needed) the canonical joined features CSV
    joined_path = ensure_joined_snapshot(snap_root)
    if not joined_path.exists():
        print("ERROR: input file missing:", joined_path)
        return 1

    per_metric_steps = snap_root / "per-metric" / "steps_hourly.csv"
    steps_df = None
    if per_metric_steps.exists():
        steps_df = pd.read_csv(per_metric_steps)

    df = pd.read_csv(joined_path)

    derived = compute_activity_features(df, steps_df)

    # idempotent merge by date: only add columns that did not exist in original
    new_cols = [c for c in derived.columns if c not in df.columns and c != "date"]
    merged = df.copy()
    if new_cols:
        for c in new_cols:
            merged[c] = derived[c]

    # ensure sorted by date
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.sort_values("date")

    # backup + write (atomic) — use centralized helper to standardize backup name
    f_activity = snap_root / "joined" / "features_activity.csv"
    if args.dry_run:
        print("DRY RUN: would write columns:", new_cols)
        print(f"DRY RUN: would write per-domain activity CSV -> {f_activity}")
        write_joined_features(merged, snap_root, dry_run=True)
    else:
        # write per-domain activity CSV to interim joined/ to assist join_run
        try:
            write_csv(derived, f_activity, dry_run=False, backup_name=None)
            print(f"WROTE per-domain activity CSV: {f_activity}")
        except Exception:
            # non-fatal: continue and still write canonical joined
            pass
        write_joined_features(merged, snap_root, dry_run=False)
        print("WROTE:", joined_path)

    # QC
    qc_path = snap_root / "qc"
    qc_path.mkdir(parents=True, exist_ok=True)
    qc_file = qc_path / "activity_qc.csv"
    qc = pd.DataFrame()
    qc["date"] = merged["date"]
    qc["act_steps"] = merged.get("act_steps")
    qc["act_active_min"] = merged.get("act_active_min")
    qc["act_IS"] = merged.get("act_IS")
    qc["act_IV"] = merged.get("act_IV")
    qc["act_steps_cv7"] = merged.get("act_steps_cv7")
    qc["qx_act_missing"] = merged.get("qx_act_missing")

    write_csv(qc, qc_file, dry_run=bool(args.dry_run), backup_name=None)
    if args.dry_run:
        print("DRY RUN: would write QC ->", qc_file)
    else:
        print("WROTE QC:", qc_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
