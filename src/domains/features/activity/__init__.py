"""Activity feature engineering package

Default input: snapshot dir (data/etl/<PID>/<SNAPSHOT>/). Reads the canonical
joined CSV and per-metric artifacts (e.g. per-metric/steps_hourly.csv) when
available.

Default output: writes merged columns back into the snapshot's `joined/`
CSV and writes QC outputs under `qc/`.
"""

__all__ = ["compute_activity_features"]
