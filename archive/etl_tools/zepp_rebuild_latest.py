# etl_tools/zepp_rebuild_latest.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def rebuild_one(base: Path, fname: str) -> bool:
    dfs = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and sub.name != "_latest":
            f = sub / fname
            if f.exists():
                d = pd.read_csv(f, dtype={"date": "string"})
                d["export_id"] = sub.name
                dfs.append(d)
    if not dfs:
        return False
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = (
        all_df.sort_values(["date", "export_id"])
              .drop_duplicates(subset=["date"], keep="last")
              .drop(columns=["export_id"])
    )
    safe_write_csv(all_df, base / "_latest" / fname)
    return True

def main():
    ap = argparse.ArgumentParser("zepp-rebuild-latest")
    ap.add_argument("--root", required=True, help="data_etl/<PID>/zepp_processed")
    args = ap.parse_args()
    base = Path(args.root)
    base.joinpath("_latest").mkdir(parents=True, exist_ok=True)

    any_done = False
    for name in [
        "zepp_sleep_daily.csv",
        "zepp_hr_daily.csv",
        "zepp_health_daily.csv",
        "zepp_body_daily.csv",
        "zepp_activity_daily.csv",
    ]:
        ok = rebuild_one(base, name)
        if ok:
            print(f"✅ rebuilt: {name}")
            any_done = True
    if not any_done:
        print("ℹ️ nothing to rebuild (no versioned files found).")
    else:
        print(f"↪️  _latest under: {base/'_latest'}")

if __name__ == "__main__":
    main()
