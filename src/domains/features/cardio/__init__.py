"""Cardio feature engineering package

Input (default): snapshot dir (data/etl/<PID>/<SNAPSHOT>/). Modules in this
package may read per-metric artifacts (e.g. HR, HRV) and the snapshot's
`joined/` files as inputs.

Output (default): write per-domain outputs into the snapshot's `joined/`
directory, for example `features_cardiovascular.csv` and
`joined_features_daily.csv` (merged).

Keep public functions small and focused; implementations should be import-safe
and suitable for being called from a CLI runner.
"""

__all__ = ["build_cardio_features", "write_cardio_outputs"]
