import pandas as pd
from pathlib import Path


def test_discover_and_load_zepp_cardio(tmp_path):
    # build temp extracted layout
    cloud = tmp_path / "extracted" / "zepp" / "cloud"
    hr_dir = cloud / "HEARTRATE"
    hr_dir.mkdir(parents=True)

    # create toy heartrate CSV with timestamps and hr values
    df = pd.DataFrame({
        "timestamp": ["2025-11-01T01:00:00Z", "2025-11-01T02:00:00Z", "2025-11-02T03:00:00Z"],
        "heartrate": [60, 70, 65],
    })
    f = hr_dir / "hr.csv"
    df.to_csv(f, index=False)

    from src.domains.parse_zepp_export import discover_zepp_tables
    from src.domains.cardiovascular.cardio_from_extracted import load_zepp_cardio_daily

    tables = discover_zepp_tables(cloud)
    assert "HEARTRATE" in tables

    out = load_zepp_cardio_daily(tables, "UTC")
    # expected columns
    for c in ("date", "zepp_hr_mean", "zepp_hr_max", "zepp_n_hr"):
        assert c in out.columns

    # should have >0 rows and normalized dates
    assert len(out) > 0
    assert all(len(d) == 10 and d.count("-") == 2 for d in out["date"].astype(str))
