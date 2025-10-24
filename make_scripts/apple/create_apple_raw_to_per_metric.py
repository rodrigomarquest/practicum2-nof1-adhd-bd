#!/usr/bin/env python3
"""Create etl_modules/apple_raw_to_per_metric.py from extracted heredoc.

This writes the canonical module text if it differs from what's on disk.
"""
from pathlib import Path
import sys

CONTENT = r'''"""
Extrai per-metric Apple (HR minuto-a-minuto, HRV SDNN) a partir do export.xml,
reutilizando funções do pipeline legado: iter_health_records, make_tz_selector.
Saídas no snapshot:
  per-metric/apple_heart_rate.csv   (timestamp,bpm)
  per-metric/apple_hrv_sdnn.csv     (timestamp,sdnn_ms)
  per-metric/apple_sleep_intervals.csv (opcional)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict
import importlib

def export_per_metric(export_xml: Path, out_snapshot_dir: Path,
                      cutover: str, tz_before: str, tz_after: str) -> Dict[str,str]:
    legacy = importlib.import_module("etl_pipeline_legacy")
    out_pm = out_snapshot_dir / "per-metric"
    out_pm.mkdir(parents=True, exist_ok=True)

    # TZ selector
    from datetime import date
    y, m, d = map(int, cutover.split("-"))
    tz_selector = legacy.make_tz_selector(date(y, m, d), tz_before, tz_after)

    # Stream records
    recs = legacy.iter_health_records(export_xml, tz_selector)

    hr_rows, hrv_rows, sleep_rows = [], [], []
    HR_ID   = "HKQuantityTypeIdentifierHeartRate"
    HRV_ID  = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    SLEEP_ID= "HKCategoryTypeIdentifierSleepAnalysis"

    for (type_id, value, s_local, e_local, unit, device, tz_name, src_ver, src_name) in recs:
        if type_id == HR_ID and value is not None:
            hr_rows.append({"timestamp": s_local.isoformat(), "bpm": float(value)})
        elif type_id == HRV_ID and value is not None:
            hrv_rows.append({"timestamp": s_local.isoformat(), "sdnn_ms": float(value)})
        elif type_id == SLEEP_ID and e_local is not None:
            sleep_rows.append({"start": s_local.isoformat(), "end": e_local.isoformat(), "raw_value": value})

    written = {}
    if hr_rows:
        p = out_pm / "apple_heart_rate.csv"
        pd.DataFrame(hr_rows).sort_values("timestamp").to_csv(p, index=False)
        written["apple_heart_rate"] = str(p)
    if hrv_rows:
        p = out_pm / "apple_hrv_sdnn.csv"
        pd.DataFrame(hrv_rows).sort_values("timestamp").to_csv(p, index=False)
        written["apple_hrv_sdnn"] = str(p)
    if sleep_rows:
        p = out_pm / "apple_sleep_intervals.csv"
        pd.DataFrame(sleep_rows).sort_values("start").to_csv(p, index=False)
        written["apple_sleep_intervals"] = str(p)
    return written
'''

def main():
    target = Path('etl_modules') / 'apple_raw_to_per_metric.py'
    if target.exists():
        cur = target.read_text(encoding='utf-8')
        if cur == CONTENT:
            print('SKIP: apple_raw_to_per_metric up-to-date')
            return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONTENT, encoding='utf-8')
    print('WROTE:', target)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
