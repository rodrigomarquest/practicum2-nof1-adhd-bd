"""Pre-join enrichment package

Place domain-local enrichment steps here. These modules should be small,
idempotent, and import-safe. They typically read domain artifacts and may
write intermediate per-domain enriched outputs consumed by joiners.
"""

from .prejoin_enricher import enrich_prejoin_run

__all__ = ["enrich_prejoin_run"]
