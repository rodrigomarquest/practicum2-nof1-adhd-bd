"""Sleep feature engineering package

Default input: snapshot dir (data/etl/<PID>/<SNAPSHOT>/). May read sleep
artifacts and the canonical joined CSV.

Default output: sleep-derived per-day metrics written into the snapshot's
`joined/` directory and/or processed/ depending on implementation details.
"""

__all__ = []
