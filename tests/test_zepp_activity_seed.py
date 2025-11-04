import pandas as pd
from pathlib import Path
import os


def test_discover_and_load_zepp_activity(tmp_path, monkeypatch):
    # create sample zepp extracted layout
    cloud = tmp_path / "extracted" / "zepp" / "cloud"
    act_dir = cloud / "ACTIVITY"
    hd_dir = cloud / "HEALTH_DATA"
    act_dir.mkdir(parents=True)
    hd_dir.mkdir(parents=True)

    # ACTIVITY daily csv
    df_act = pd.DataFrame({"date": ["2025-11-01", "2025-11-02"], "steps": [1000, 2000], "distance_m": [800.0, 1600.0]})
    act_file = act_dir / "activity_daily.csv"
    df_act.to_csv(act_file, index=False)

    # HEALTH_DATA alternate file (should be ignored if ACTIVITY present)
    df_hd = pd.DataFrame({"date": ["2025-11-01"], "total_steps": [900], "distance": [700.0]})
    hd_file = hd_dir / "health_data.csv"
    df_hd.to_csv(hd_file, index=False)

    # import the discovery helper and loader
    from src.domains.parse_zepp_export import discover_zepp_tables
    from src.domains.activity.zepp_activity import load_zepp_activity_daily

    tables = discover_zepp_tables(cloud)
    assert "ACTIVITY" in tables
    # check counts
    assert len(tables["ACTIVITY"]) == 1

    df = load_zepp_activity_daily(tables, home_tz="UTC")
    assert "zepp_steps" in df.columns
    assert df.loc[df["date"] == "2025-11-01", "zepp_steps"].iloc[0] == 1000
    assert df.loc[df["date"] == "2025-11-02", "zepp_steps"].iloc[0] == 2000
