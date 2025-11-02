"""Top-level enrichment orchestrator.

This module implements a small, idempotent enrichment step over the
canonical joined features CSV located at
`<snapshot_dir>/joined/joined_features_daily.csv`.

Public API
----------
- enrich_run(snapshot_dir, *, dry_run=False) -> int

Exit codes
----------
- 0 : success (or dry-run with valid input)
- 2 : joined file missing or missing required 'date' column
- 1 : IO / unexpected error

The implementation is intentionally conservative: it does not overwrite
existing enrichment columns and only creates the requested derived columns
when called. It also writes a small QC file under `qc/enriched_qc.csv`.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import traceback
from typing import Optional

import pandas as pd
import numpy as np

from lib.io_guards import ensure_joined_snapshot, write_joined_features
from lib.io_guards import write_csv
from lib.df_utils import zscore, ensure_columns


def enrich_run(snapshot_dir: Path | str, *, dry_run: bool = False) -> int:
    """Run canonical global enrichments for a snapshot.

    Behaviour implemented per P9 requirements (cmp_/qx_ fields + QC).
    The function is idempotent and will not overwrite existing columns.
    """
    sd = Path(snapshot_dir)
    joined_path = ensure_joined_snapshot(sd)

    print(f"INFO: enrich_run start snapshot_dir={sd} dry_run={dry_run}")

    # Verify joined exists and is readable
    if not joined_path.exists():
        if dry_run:
            print("[dry-run] joined missing — enrich skipped")
            print(f"INFO: enrich_run skipped (joined missing at {joined_path})")
            return 0
        print(f"[error] joined file not found at {joined_path}")
        return 2

    try:
        df = pd.read_csv(joined_path)
    except Exception:
        print(f"[error] failed to read joined CSV at {joined_path}")
        traceback.print_exc()
        return 1

    if "date" not in df.columns:
        if dry_run:
            print("[dry-run] joined missing — enrich skipped")
            print("INFO: enrich_run skipped (joined missing required 'date' column)")
            return 0
        print("[error] joined CSV missing required 'date' column")
        return 2

    # normalize date column (keep as string but sortable)
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    except Exception:
        df["date"] = df["date"].astype(str)

    n_rows_before = len(df)

    # Track how many columns we create
    cmp_created = 0
    qx_created = 0

    # Prepare safe series for inputs (use pandas Series of NaN if missing)
    def _s(name):
        return df[name] if name in df.columns else pd.Series([np.nan] * len(df))

    slp_hours = _s("slp_hours").astype(float)
    act_active_min = _s("act_active_min").astype(float)
    cv_hr_mean = _s("cv_hr_mean").astype(float)
    cv_hrv_mean = _s("cv_hrv_mean").astype(float)
    slp_hours_cv7 = _s("slp_hours_cv7").astype(float)
    act_steps_cv7 = _s("act_steps_cv7").astype(float)
    act_sedentary_min = _s("act_sedentary_min").astype(float)

    # cmp_activation = z(-slp_hours) + z(act_active_min) + z(cv_hr_mean) - z(cv_hrv_mean)
    try:
        z_slp = zscore(-slp_hours)
        z_act_active = zscore(act_active_min)
        z_cv_hr = zscore(cv_hr_mean)
        z_cv_hrv = zscore(cv_hrv_mean)
        cmp_activation = z_slp + z_act_active + z_cv_hr - z_cv_hrv
    except Exception:
        cmp_activation = pd.Series([np.nan] * len(df))

    # cmp_stability = 1 - mean([slp_hours_cv7, act_steps_cv7])
    try:
        # compute row-wise mean ignoring NaN
        cmp_stability = 1 - pd.concat([slp_hours_cv7, act_steps_cv7], axis=1).mean(axis=1, skipna=True)
    except Exception:
        cmp_stability = pd.Series([np.nan] * len(df))

    # cmp_fatigue = z(max(0, 7 - slp_hours)) + z(act_sedentary_min) - z(cv_hrv_mean)
    try:
        short_sleep = (7.0 - slp_hours).clip(lower=0.0)
        z_short_sleep = zscore(short_sleep)
        z_sedentary = zscore(act_sedentary_min)
        z_cv_hrv = zscore(cv_hrv_mean)
        cmp_fatigue = z_short_sleep + z_sedentary - z_cv_hrv
    except Exception:
        cmp_fatigue = pd.Series([np.nan] * len(df))

    # QC: qx_act_missing = 1 if any of [apple_steps, apple_active_kcal, apple_exercise_min, apple_stand_hours] is NaN
    act_qc_cols = ["apple_steps", "apple_active_kcal", "apple_exercise_min", "apple_stand_hours"]
    try:
        any_missing = df[act_qc_cols].isna().any(axis=1) if all(c in df.columns for c in act_qc_cols) else pd.Series([True] * len(df))
        qx_act_missing = any_missing.astype(int)
    except Exception:
        qx_act_missing = pd.Series([pd.NA] * len(df))

    # Idempotent writes: do not overwrite existing columns
    if "cmp_activation" not in df.columns:
        df["cmp_activation"] = cmp_activation
        cmp_created += 1
    if "cmp_stability" not in df.columns:
        df["cmp_stability"] = cmp_stability
        cmp_created += 1
    if "cmp_fatigue" not in df.columns:
        df["cmp_fatigue"] = cmp_fatigue
        cmp_created += 1
    if "qx_act_missing" not in df.columns:
        df["qx_act_missing"] = qx_act_missing
        qx_created += 1

    # Final ordering by date
    try:
        df = df.sort_values("date").reset_index(drop=True)
    except Exception:
        # if sorting fails, proceed but warn
        print("[warn] failed to sort by date; leaving original order")

    print(f"INFO: enrich_run computed: rows_in={n_rows_before} cmp_created={cmp_created} qx_created={qx_created}")

    # Write joined features back using canonical writer
    try:
        out_path = write_joined_features(df, sd, dry_run=dry_run)
    except Exception:
        print("[error] failed to write joined features")
        traceback.print_exc()
        return 1

    # Write QC file: summary row + preview (10 head + 10 tail)
    try:
        qc_dir = sd / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / "enriched_qc.csv"

        date_min = df["date"].min() if len(df) else ""
        date_max = df["date"].max() if len(df) else ""
        summary = {
            "date_min": str(date_min),
            "date_max": str(date_max),
            "n_rows": int(len(df)),
            "n_cmp_created": int(cmp_created),
            "n_qx_created": int(qx_created),
        }

        preview_cols = ["date", "cmp_activation", "cmp_stability", "cmp_fatigue", "qx_act_missing"]
        preview_df = df[[c for c in preview_cols if c in df.columns]].copy()
        head_tail = pd.concat([preview_df.head(10), preview_df.tail(10)])

        # Build CSV content: one-line summary, blank, then preview table
        from io import StringIO

        buf = StringIO()
        pd.DataFrame([summary]).to_csv(buf, index=False)
        buf.write("\n")
        if not head_tail.empty:
            head_tail.to_csv(buf, index=False)
        content = buf.getvalue()

        # atomic write of text content
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(qc_dir), prefix=qc_path.name + ".tmp.") as tf:
            tf.write(content)
            tmp = Path(tf.name)
        tmp.replace(qc_path)
    except Exception:
        print("[error] failed to write QC file")
        traceback.print_exc()
        return 1

    print(f"INFO: enrich_run done -> wrote joined: {out_path} qc: {qc_path}")
    return 0


__all__ = ["enrich_run"]
