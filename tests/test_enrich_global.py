import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pandas.testing as pdt


# Ensure src is importable during tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domains.enriched.enrich_global import enrich_run
from lib.io_guards import write_joined_features, ensure_joined_snapshot


def make_minimal_joined(df: pd.DataFrame, snapshot_dir: Path) -> Path:
    """Write the provided DataFrame as the canonical joined CSV for snapshot."""
    # write_joined_features will create parent dirs as needed
    write_joined_features(df, snapshot_dir, dry_run=False)
    return ensure_joined_snapshot(snapshot_dir)


def test_enrich_missing_joined(tmp_path, monkeypatch):
    # silence TZ guard during tests
    monkeypatch.setenv("ETL_TZ_GUARD", "0")

    snap = tmp_path / "data" / "etl" / "PX" / "2025-11-01"
    snap.mkdir(parents=True, exist_ok=True)

    rc = enrich_run(snap, dry_run=True)
    assert int(rc) == 2


def test_enrich_happy_path_and_idempotence(tmp_path, monkeypatch):
    monkeypatch.setenv("ETL_TZ_GUARD", "0")

    snap = tmp_path / "data" / "etl" / "PX" / "2025-11-01"
    snap.mkdir(parents=True, exist_ok=True)

    # create minimal joined with required columns (3 rows)
    df = pd.DataFrame(
        {
            "date": ["2025-10-29", "2025-10-30", "2025-10-31"],
            "slp_hours": [7.5, 5.0, np.nan],
            "slp_hours_cv7": [5.0, 10.0, np.nan],
            "act_active_min": [30.0, 5.0, 0.0],
            "act_steps": [1000, 200, np.nan],
            "act_sedentary_min": [300.0, 200.0, 0.0],
            "act_steps_cv7": [12.0, 25.0, np.nan],
            "cv_hr_mean": [60.0, 70.0, 65.0],
            "cv_hrv_mean": [50.0, 40.0, np.nan],
            "apple_steps": [1000, np.nan, 0],
            "apple_active_kcal": [200.0, 10.0, np.nan],
            "apple_exercise_min": [30.0, 0.0, np.nan],
            "apple_stand_hours": [12.0, 0.0, np.nan],
        }
    )

    joined_path = make_minimal_joined(df, snap)

    # Run enrichment first time (should create columns and backup prev file)
    rc1 = enrich_run(snap, dry_run=False)
    assert int(rc1) == 0

    # QC file exists
    qc = snap / "qc" / "enriched_qc.csv"
    assert qc.exists()

    # joined prev backup should exist (since enrichment writes over existing)
    prev = joined_path.parent / "joined_features_daily_prev.csv"
    assert prev.exists()

    df_after_1 = pd.read_csv(joined_path)
    # check expected columns present
    for c in ["cmp_activation", "cmp_stability", "cmp_fatigue", "qx_act_missing"]:
        assert c in df_after_1.columns
        # not all NaN (at least one valid value expected)
        assert not df_after_1[c].isna().all()

    # Run enrichment second time (idempotence)
    rc2 = enrich_run(snap, dry_run=False)
    assert int(rc2) == 0

    df_after_2 = pd.read_csv(joined_path)

    # compare the computed columns for equality (idempotent)
    for c in ["cmp_activation", "cmp_stability", "cmp_fatigue", "qx_act_missing"]:
        # use pandas testing which treats NaNs in same locations as equal
        pdt.assert_series_equal(df_after_1[c], df_after_2[c], check_names=False)
