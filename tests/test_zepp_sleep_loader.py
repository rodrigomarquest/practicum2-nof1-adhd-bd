import pandas as pd
from pathlib import Path


def test_discover_and_load_zepp_sleep(tmp_path):
    cloud = tmp_path / "extracted" / "zepp" / "cloud"
    sl_dir = cloud / "SLEEP"
    sl_dir.mkdir(parents=True)

    # simple sleep CSV: start_time, total_hours, deep_hours, light_hours, rem_hours
    df = pd.DataFrame({
        "start_time": ["2025-11-01T22:00:00Z", "2025-11-02T22:00:00Z"],
        "total_hours": [7.5, 6.0],
        "deep_hours": [2.0, 1.5],
        "light_hours": [4.5, 3.5],
        "rem_hours": [1.0, 1.0],
    })
    f = sl_dir / "sleep.csv"
    df.to_csv(f, index=False)

    from src.domains.parse_zepp_export import discover_zepp_tables
    from src.domains.sleep.sleep_from_extracted import load_zepp_sleep_daily

    tables = discover_zepp_tables(cloud)
    assert "SLEEP" in tables

    out = load_zepp_sleep_daily(tables, "UTC")
    # expected columns (these must exist in output)
    required = [
        "date",
        "zepp_slp_total_h",
    ]
    for c in required:
        assert c in out.columns

    # should have rows and numeric totals
    assert len(out) > 0
    assert out["zepp_slp_total_h"].sum() > 0
