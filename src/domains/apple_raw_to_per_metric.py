from __future__ import annotations

"""
Extrai per-metric Apple (HR minuto-a-minuto, HRV SDNN) a partir do export.xml,
reutilizando funções do pipeline legado: iter_health_records, make_tz_selector.
Saídas no snapshot:
  per-metric/apple_heart_rate.csv
  per-metric/apple_hrv_sdnn.csv
  per-metric/apple_sleep_intervals.csv (opcional)
"""
from pathlib import Path
import pandas as pd
from typing import Dict
import importlib
import os

# Optional tqdm for progress bars with safe fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, *a, **k):
        return it

try:
    from lib.io_guards import write_csv  # type: ignore
except Exception:
    def write_csv(df, path, dry_run=False, backup_name=None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(p)


def export_per_metric(
    export_xml: Path,
    out_snapshot_dir: Path,
    cutover: str,
    tz_before: str,
    tz_after: str,
    max_records: int = None,
) -> Dict[str, str]:
    legacy = importlib.import_module("etl_pipeline")
    out_pm = out_snapshot_dir / "per-metric"
    out_pm.mkdir(parents=True, exist_ok=True)

    # TZ selector
    from datetime import date

    y, m, d = map(int, cutover.split("-"))
    tz_selector = legacy.make_tz_selector(date(y, m, d), tz_before, tz_after)

    # Stream records with optional max_records limit
    recs = legacy.iter_health_records(export_xml, tz_selector, max_records=max_records)
    # progress control (disabled by default unless ETL_TQDM=1 and not CI)
    disable = bool(os.getenv("ETL_TQDM", "0") == "0" or os.getenv("CI"))
    recs_iter = tqdm(recs, desc=f"Parsing {export_xml.name}", disable=disable)

    hr_rows, hrv_rows, sleep_rows = [], [], []
    HR_ID = "HKQuantityTypeIdentifierHeartRate"
    HRV_ID = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    SLEEP_ID = "HKCategoryTypeIdentifierSleepAnalysis"

    for (
        type_id,
        value,
        s_local,
        e_local,
        unit,
        device,
        tz_name,
        src_ver,
        src_name,
    ) in recs_iter:
        if type_id == HR_ID and value is not None:
            hr_rows.append({"timestamp": s_local.isoformat(), "bpm": float(value)})
        elif type_id == HRV_ID and value is not None:
            hrv_rows.append({"timestamp": s_local.isoformat(), "sdnn_ms": float(value)})
        elif type_id == SLEEP_ID and e_local is not None:
            sleep_rows.append(
                {
                    "start": s_local.isoformat(),
                    "end": e_local.isoformat(),
                    "raw_value": value,
                }
            )

    written = {}
    if hr_rows:
        p = out_pm / "apple_heart_rate.csv"
        write_csv(pd.DataFrame(hr_rows).sort_values("timestamp"), p)
        written["apple_heart_rate"] = str(p)
    if hrv_rows:
        p = out_pm / "apple_hrv_sdnn.csv"
        write_csv(pd.DataFrame(hrv_rows).sort_values("timestamp"), p)
        written["apple_hrv_sdnn"] = str(p)
    if sleep_rows:
        p = out_pm / "apple_sleep_intervals.csv"
        write_csv(pd.DataFrame(sleep_rows).sort_values("start"), p)
        written["apple_sleep_intervals"] = str(p)
    return written
