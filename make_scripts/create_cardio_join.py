#!/usr/bin/env python3
"""Write etl_modules/cardiovascular/_join/join.py with canonical content.

This generator is intentionally minimal and side-effect free on import.
"""
from pathlib import Path
from typing import List, Optional


CONTENT = r'''"""
Cross-platform join rules for cardio domain (functional minimal).

HR:
  - If Apple exists -> average Apple + others by timestamp (keeps alignment)
  - Else -> average across providers
HRV:
  - Prefer Apple SDNN when present
  - Else first available metric (e.g., Zepp RMSSD/SDNN)
"""
from typing import List, Optional
import pandas as pd


def _valid(dfs: List[Optional[pd.DataFrame]]) -> List[pd.DataFrame]:
    return [d for d in dfs if d is not None and not d.empty]


def join_hr(hr_sources: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    valid = _valid(hr_sources)
    if not valid:
        return None
    if len(valid) == 1:
        df = valid[0].copy()
        return df.sort_values("timestamp")

    def is_apple(df: pd.DataFrame) -> bool:
        return "source" in df.columns and (df["source"] == "apple").any()

    has_apple = any(is_apple(df) for df in valid)

    merged = pd.concat([df[["timestamp", "bpm"]] for df in valid], ignore_index=True)
    agg = merged.groupby("timestamp", as_index=False).agg({"bpm": "mean"})
    agg = agg.sort_values("timestamp")
    agg["date"] = pd.to_datetime(agg["timestamp"]).dt.date
    agg["source"] = "apple+mix" if has_apple else "mix"
    return agg


def select_hrv(hrv_sources: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    valid = _valid(hrv_sources)
    if not valid:
        return None

    # prefer Apple SDNN
    for df in valid:
        if "metric" in df.columns and (df["metric"] == "sdnn_ms").any():
            return df.sort_values("timestamp")

    # fallback: first available
    return valid[0].sort_values("timestamp")
'''


def main():
    target = Path('etl_modules/cardiovascular/_join/join.py')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding='utf-8') == CONTENT:
        print('SKIP:', target)
        return 0
    target.write_text(CONTENT, encoding='utf-8')
    print('WROTE:', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
