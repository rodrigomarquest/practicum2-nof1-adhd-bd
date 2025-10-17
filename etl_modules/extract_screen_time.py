from __future__ import annotations
from pathlib import Path
from xml.etree.ElementTree import iterparse
from collections import defaultdict
from dateutil import tz
import pandas as pd
import numpy as np

def _normalize_dt_with_cutover(dt_str: str | None, cutover_str: str, tz_before: str, tz_after: str):
    if not dt_str:
        return None
    ts_utc = pd.to_datetime(dt_str, utc=True, errors="coerce")
    if ts_utc is None or pd.isna(ts_utc):
        return None
    cutover_date = pd.to_datetime(cutover_str).date()
    target_tz = tz.gettz(tz_before) if ts_utc.date() < cutover_date else tz.gettz(tz_after)
    return ts_utc.tz_convert(target_tz)

def extract_apple_screen_time(export_xml_path: Path, output_csv_path: Path,
                              cutover_str: str, tz_before: str, tz_after: str) -> pd.DataFrame:
    if not export_xml_path.exists():
        raise FileNotFoundError(f"Apple Health export.xml not found: {export_xml_path}")
    agg_seconds = defaultdict(float)
    ctx = iterparse(str(export_xml_path), events=("end",))
    for _, elem in ctx:
        if elem.tag != "Record":
            elem.clear(); continue
        rtype = (elem.attrib.get("type") or "").lower()
        if "screentime" not in rtype:
            elem.clear(); continue
        start_dt = _normalize_dt_with_cutover(elem.attrib.get("startDate"), cutover_str, tz_before, tz_after)
        end_dt   = _normalize_dt_with_cutover(elem.attrib.get("endDate"),   cutover_str, tz_before, tz_after)
        seconds = None
        v = elem.attrib.get("value")
        if v is not None:
            try:
                seconds = float(v)
                if seconds > 36*3600:  # sometimes ms in odd exports
                    seconds /= 1000.0
            except Exception:
                seconds = None
        if seconds is None and (start_dt is not None and end_dt is not None):
            seconds = max(0.0, (end_dt - start_dt).total_seconds())
        if seconds is not None and start_dt is not None:
            agg_seconds[start_dt.date()] += seconds
        elem.clear()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not agg_seconds:
        df_empty = pd.DataFrame(columns=["date", "screen_time_min", "source"])
        df_empty.to_csv(output_csv_path, index=False)
        return df_empty

    df_usage = pd.DataFrame({
        "date": pd.to_datetime(list(agg_seconds.keys())),
        "screen_time_min": np.array(list(agg_seconds.values()), dtype="float64")/60.0
    }).sort_values("date")
    df_usage["source"] = "AppleHealth"
    df_usage.to_csv(output_csv_path, index=False)
    return df_usage
