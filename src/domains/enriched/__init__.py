"""enriched package

Contains enrichment layers that post-process the joined dataset into enriched
representations used by EDA, modeling, or downstream analysis.

Structure:
 - pre/: enrich steps that run before global joins (domain-local enrichments)
 - post/: enrich steps that run after global join is produced
 - enrich_global.py: top-level enrichment orchestrator for the joined dataset

Default input: snapshot `joined/joined_features_daily.csv`.
Default output: enriched artifacts under `joined/` or `enriched/` depending on
implementation.
"""

__all__ = ["pre", "post", "enrich_global"]
