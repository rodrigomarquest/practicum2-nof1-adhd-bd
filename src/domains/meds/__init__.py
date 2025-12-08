"""Medication domain module for Apple Health medication records.

This domain aggregates medication-related events from Apple Health exports
(export.xml) into daily summaries. It follows the same architectural patterns
as the cardio, sleep, and activity domains.

Apple Health medication-related record types:
- HKClinicalTypeIdentifierMedicationRecord (iOS 15+ clinical records)
- HKCategoryTypeIdentifierMindfulSession (some meditation/mindfulness apps)
- Any third-party app that logs medication to Apple Health

Default snapshot: 2025-12-07 (latest production snapshot)
Raw data source: data/raw/<PID>/apple/export/apple_health_export_YYYYMMDD.zip

Output: daily_meds.csv with columns:
- date (YYYY-MM-DD)
- med_any (0/1)
- med_event_count (int)
- med_names (comma-separated list of unique medication names, if available)
- med_sources (comma-separated list of unique source apps)

Usage:
    # From CLI:
    python -m src.domains.meds.meds_from_extracted --pid P000001 --snapshot 2025-12-07 --dry-run 0
    
    # From Python:
    from src.domains.meds import run_meds_aggregation
    outputs = run_meds_aggregation(participant="P000001", snapshot="2025-12-07")
"""

from .meds_from_extracted import (
    load_apple_meds_daily,
    run_meds_aggregation,
    MedsAggregator,
    discover_apple_meds,
    find_latest_apple_zip,
    extract_apple_zip_for_meds,
    main,
)

__all__ = [
    "load_apple_meds_daily",
    "run_meds_aggregation", 
    "MedsAggregator",
    "discover_apple_meds",
    "find_latest_apple_zip",
    "extract_apple_zip_for_meds",
    "main",
]
