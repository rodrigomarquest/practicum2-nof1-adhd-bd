"""Global join helpers

Top-level joins that coordinate merging across domains. These modules should
call per-domain joiners and produce the canonical joined dataset under
`joined/joined_features_daily.csv`.

Default input: snapshot dir and all per-domain artifact directories.
Default output: canonical joined CSV and optional manifests.
"""

__all__ = []
