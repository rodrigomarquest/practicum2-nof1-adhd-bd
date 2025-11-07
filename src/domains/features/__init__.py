"""features package

Per-domain feature engineering modules.

Default input: snapshot directory (data/etl/<PID>/<SNAPSHOT>/)
Default output: per-domain outputs are written under the snapshot's `joined/`
directory (e.g. joined/joined_features_daily.csv and per-domain CSVs).

Each subpackage (cardio, activity, sleep, screen) should expose a
small public API like `compute_*_features(snapshot_dir, ...)` or
`write_*_outputs(snapshot_dir, ...)`.
"""

__all__ = ["cardio", "activity", "sleep", "screen"]
