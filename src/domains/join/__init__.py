"""join package

Contains modules that merge per-domain artifacts into the snapshot's joined
dataset. Typical usage: joiners consume per-domain outputs (e.g. zepp processed
artifacts, apple per-metric files) and merge columns into the joined CSV by
`date`.

Default input: per-domain artifact directories and the snapshot `joined/` CSV.
Default output: updated `joined/joined_features_daily.csv`.
"""

__all__ = ["cardio", "activity", "global"]
