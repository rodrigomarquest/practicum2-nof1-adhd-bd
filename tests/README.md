# Tests Directory — Deprecated (v4.2+)

## Status: ⚠️ Legacy pytest tests removed

This directory previously contained pytest-based unit tests created during exploratory model implementation (v4.0–4.1).

## Current QA Strategy (v4.2+)

The project now uses a **domain-specific ETL audit system** instead of pytest:

### ETL Audit Module

- **File:** `src/etl/etl_audit.py` (~1,874 lines)
- **Purpose:** PhD-level regression testing for data integrity
- **Domains:** cardio, activity, sleep, meds, som, unified_ext, labels

### Usage

```bash
# Individual domain audits
make qc-cardio PID=P000001 SNAPSHOT=2025-11-07
make qc-activity PID=P000001 SNAPSHOT=2025-11-07
make qc-sleep PID=P000001 SNAPSHOT=2025-11-07

# Run all audits
make qc-all PID=P000001 SNAPSHOT=2025-11-07
```

### Outputs

- **CSV:** `data/etl/<PID>/<SNAPSHOT>/qc/<domain>_audit.csv`
- **JSON:** `data/etl/<PID>/<SNAPSHOT>/qc/<domain>_audit.json`
- **Markdown:** `docs/reports/qc/QC_<domain>_<PID>_<SNAPSHOT>.md`

### Exit Codes

- `0` = PASS (all checks passed)
- `1` = FAIL (critical issues found)

### Why Not pytest?

1. **Domain-specific:** ETL audits are tailored to physiological data QC
2. **Integrated:** Audits run as part of pipeline workflow
3. **Provenance:** Generates scientific-grade QC reports automatically
4. **Reproducibility:** PhD-level traceability for academic rigor

## Historical Context

Deleted tests (v4.3.2 cleanup):

- `test_canonical_pbsi_integration.py` → Superseded by `make qc-labels`
- `test_cardio_sleep_cli.py` → Superseded by `make qc-cardio`, `make qc-sleep`
- `test_enrich_global.py` → Superseded by `make qc-unified-ext`
- `test_zepp_*_loader.py` → Integration tested in production
- `idempotence_a6_a7_a8.py` → Replaced by provenance SHA256 tracking

See `docs/copilot/TESTS_CLEANUP_AUDIT.md` for full analysis.
