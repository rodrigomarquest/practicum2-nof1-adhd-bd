"""CLI entrypoint package for executable scripts moved from /scripts.

This package contains thin wrappers that call into `src.*` modules and
are intended to be run as `python -m cli.etl_runner` so Makefile and other
invokers don't rely on a scripts/ directory.

Location: src/cli/

Modules:
    extract_biomarkers: Extract clinical biomarkers from wearable data
    prepare_zepp_data: Prepare Zepp backup data for biomarkers extraction
    etl_runner: ETL pipeline orchestrator (legacy)
    run_etl_with_timer: ETL with timing instrumentation (legacy)
"""

__all__ = ["etl_runner", "run_etl_with_timer", "extract_biomarkers"]
