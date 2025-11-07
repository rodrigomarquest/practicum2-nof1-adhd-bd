"""Post-join enrichment package

Steps that operate after the global joined CSV is created. Examples: feature
normalization, additional derived columns, or dataset versioning helpers.
"""

from .postjoin_enricher import enrich_postjoin_run

__all__ = ["enrich_postjoin_run"]
