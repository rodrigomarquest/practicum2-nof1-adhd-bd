# ETL Audit System Implementation Plan

**Date**: 2025-11-19  
**Status**: Ready for implementation  
**Objective**: Create a repeatable regression test suite for ETL pipeline integrity across HR, Steps, and Sleep domains

---

## Overview

Transform `etl_audit.py` into a domain-specific auditing system that can:

1. Run automated regression tests for HR, Steps, and Sleep features
2. Output clear PASS/FAIL results with critical anomaly detection
3. Generate CSV files with per-day audit results for QC tracking
4. Be invoked via Makefile for CI/CD integration

---

## Step 1: Enhanced `etl_audit.py` with Domain Support

### Current State

- Basic ETL auditor class exists
- Only does general extraction/unification checks
- No domain-specific logic
- No PASS/FAIL exit codes

### Required Changes

**1.1 Add CLI with argparse**:

```python
parser = argparse.ArgumentParser(
    description="ETL Feature Integrity Audit - Domain-specific regression testing"
)
parser.add_argument("--participant", "--pid", required=True, help="Participant ID (e.g., P000001)")
parser.add_argument("--snapshot", required=True, help="Snapshot date (e.g., 2025-11-07)")
parser.add_argument("--domain", required=True, choices=["hr", "steps", "sleep"],
                   help="Domain to audit: hr, steps, or sleep")
```

**1.2 Add `run_hr_audit()` method** to `ETLAuditor` class:

**Checks to implement**:

- Event-level Parquet exists (`export_apple_hr_events.parquet`)
- Daily cache Parquet exists with ALL 5 metrics (`apple_hr_mean`, `apple_hr_min`, `apple_hr_max`, `apple_hr_std`, `apple_n_hr`)
- **CRITICAL**: No fabricated relationships (hr_min == hr_mean for all rows)
  - Threshold: < 10% fabrication allowed (legitimate single-sample days)
  - If > 10%, mark as FAIL
- **CRITICAL**: hr_std is non-zero for multi-sample days
- Consistency check: cache days == daily_cardio.csv days
- Consistency check: daily_cardio.csv → features_daily_unified.csv → features_daily_labeled.csv
- **WARNING**: No column duplication (hr_mean vs apple_hr_mean)

**Statistics to compute**:

- Mean HR (daily avg): mean ± std
- Min/Max HR range
- Std HR (daily variability)
- Samples/day distribution
- Fabrication rates (hr_min == hr_mean, hr_std == 0)

**Per-day CSV output** (`data/etl/{PID}/{SNAPSHOT}/qc/hr_feature_audit.csv`):

```csv
date,hr_mean,hr_min,hr_max,hr_std,hr_samples,is_fabricated_min,is_fabricated_std,is_single_sample
2021-05-14,70.15,62.0,124.0,8.5,779,false,false,false
2021-05-15,65.80,65.80,120.0,0.0,1,true,true,true
...
```

**1.3 Add `run_steps_audit()` method**:

**Checks to implement**:

- `daily_activity.csv` exists with steps column
- Steps present in unified CSV
- No all-NaN columns
- Vendor prioritization visible (Apple vs Zepp)
- Consistency across extraction → unification → labeling

**Statistics to compute**:

- Mean steps/day ± std
- Median steps/day
- Range (min to max)
- Days with data (steps > 0)

**Per-day CSV output** (`data/etl/{PID}/{SNAPSHOT}/qc/steps_feature_audit.csv`):

```csv
date,steps,has_data
2017-12-04,8523,true
2017-12-05,0,false
...
```

**1.4 Add `run_sleep_audit()` method**:

**Checks to implement**:

- `daily_sleep.csv` exists with sleep_hours
- Sleep hours in plausible range (0-24)
- Sleep efficiency in range [0, 1] (if present)
- Consistency across extraction → unification → labeling
- No invalid values

**Statistics to compute**:

- Mean sleep hours ± std
- Median sleep hours
- Range (min to max)
- Days with sleep > 0
- Mean sleep efficiency ± std (if present)

**Per-day CSV output** (`data/etl/{PID}/{SNAPSHOT}/qc/sleep_feature_audit.csv`):

```csv
date,sleep_hours,sleep_efficiency,has_sleep,is_valid_range
2017-12-04,7.5,0.89,true,true
2017-12-05,0.0,,false,true
...
```

**1.5 Add final PASS/FAIL summary**:

```python
def print_final_summary(self):
    """Print final PASS/FAIL summary."""
    n_critical = len(self.audit_results["issues"])
    n_warnings = len(self.audit_results["warnings"])

    if self.audit_results["pass"]:
        logger.info("✅ AUDIT STATUS: PASS")
        logger.info(f"   Domain: {self.domain.upper()}")
        logger.info(f"   Participant: {self.participant}")
        logger.info(f"   Snapshot: {self.snapshot}")
    else:
        logger.info("❌ AUDIT STATUS: FAIL")
        logger.info(f"\n🔴 {n_critical} CRITICAL issues found:")
        for issue in self.audit_results["issues"]:
            logger.info(f"   - [{issue['category']}] {issue['description']}")

    return 0 if self.audit_results["pass"] else 1
```

**1.6 Main CLI routing**:

```python
def main():
    args = parser.parse_args()
    auditor = ETLAuditor(args.participant, args.snapshot, args.domain)

    if args.domain == "hr":
        passed = auditor.run_hr_audit()
    elif args.domain == "steps":
        passed = auditor.run_steps_audit()
    elif args.domain == "sleep":
        passed = auditor.run_sleep_audit()

    auditor.save_results()  # Save CSV + JSON
    auditor.print_final_summary()

    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## Step 2: Feature Integrity Audit Documentation

Create comprehensive markdown audits following the **exact pattern** of `HR_FEATURE_INTEGRITY_AUDIT.md`:

### 2.1 `STEPS_FEATURE_INTEGRITY_AUDIT.md`

**Structure** (mirror HR audit):

```markdown
# Steps Feature Integrity Audit Report

**Project**: practicum2-nof1-adhd-bd  
**Date**: 2025-11-19  
**Auditor**: PhD-level Data Engineer (AI-assisted)  
**Scope**: Complete Steps data lifecycle from Raw → Daily CSV → Unified → PBSI → Modeling  
**Status**: [🔴 CRITICAL | ⚠️ WARNING | ✅ PASS]

## Executive Summary

[Key findings - critical issues and verified correct behaviors]

## Section 1: Raw Steps Data Sources

[Trace Apple and Zepp raw steps data]

## Section 2: Daily Steps Aggregation

[Document daily_activity.csv schema and aggregation logic]

## Section 3: Steps in Unified Data

[Trace through unify_daily.py - vendor prioritization, column mapping]

## Section 4: Steps in Labeled Features

[Check features_daily_labeled.csv for steps columns]

## Section 5: Steps in PBSI

[Document how steps contribute to activity_sub]

## Section 6: Steps in ML6/ML7

[Verify steps used in modeling]

## Section 7: Detected Inconsistencies

[Document any issues found]

## Section 8: Fix Recommendations

[Actionable fixes, similar to HR audit]
```

**Key Questions to Answer**:

1. What are the column names in raw data? (Apple: `total_steps`, Zepp: `steps`)
2. How is vendor prioritization handled? (Apple > Zepp)
3. Is there forward-fill or backfill applied?
4. Are there duplicate columns? (e.g., `steps` vs `apple_steps` vs `total_steps`)
5. What is the missing data pattern? (% NaN per vendor, per time period)
6. Are Zepp steps merged correctly post-2024-06 (when Apple Watch stopped)?

### 2.2 `SLEEP_FEATURE_INTEGRITY_AUDIT.md`

**Structure** (mirror HR audit):

```markdown
# Sleep Feature Integrity Audit Report

**Project**: practicum2-nof1-adhd-bd  
**Date**: 2025-11-19  
**Auditor**: PhD-level Data Engineer (AI-assisted)  
**Scope**: Complete Sleep data lifecycle from Raw → Daily CSV → Unified → PBSI → Modeling  
**Status**: [🔴 CRITICAL | ⚠️ WARNING | ✅ PASS]

## Executive Summary

[Key findings]

## Section 1: Raw Sleep Data Sources

[Apple HKCategoryTypeIdentifierSleepAnalysis, Zepp SLEEP CSVs]

## Section 2: Sleep Aggregation Logic

[How overlapping sleep intervals are handled]

## Section 3: Timezone Handling

[Midnight boundary issues, DST transitions]

## Section 4: Sleep in Unified Data

[Vendor prioritization, column mapping]

## Section 5: Sleep in PBSI

[sleep_sub computation: -0.6*z_sleep_dur + 0.4*z_sleep_eff]

## Section 6: Sleep Stages

[Deep, REM, Light - are they tracked? Used?]

## Section 7: Detected Inconsistencies

[Document any issues]

## Section 8: Fix Recommendations

[Actionable fixes]
```

**Key Questions to Answer**:

1. How are overlapping sleep sessions handled? (Merged? Deduplicated?)
2. Is main sleep vs naps distinguished?
3. What happens at midnight boundaries? (Timezone naive date truncation?)
4. Is sleep_efficiency computed consistently? (In-bed time vs asleep time)
5. Are there stage-specific metrics? (deep_sleep_h, rem_h, light_h)
6. How does Apple → Zepp transition affect sleep tracking? (Different sensors)

---

## Step 3: Makefile Integration

Add to `Makefile`:

```makefile
# QC Targets
.PHONY: qc-etl qc-hr qc-steps qc-sleep qc-all

# Default participant/snapshot (can override with: make qc-hr PID=P000002 SNAPSHOT=2025-10-22)
PID ?= P000001
SNAPSHOT ?= 2025-11-07

qc-hr:
\t@echo "Running HR feature integrity audit..."
\tpython -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT) --domain hr

qc-steps:
\t@echo "Running Steps feature integrity audit..."
\tpython -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT) --domain steps

qc-sleep:
\t@echo "Running Sleep feature integrity audit..."
\tpython -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT) --domain sleep

qc-all: qc-hr qc-steps qc-sleep
\t@echo "All domain audits complete"

# Alias for backward compatibility
qc-etl: qc-hr
```

**Usage Examples**:

```bash
# Run HR audit for P000001/2025-11-07
make qc-hr

# Run all audits for P000001/2025-11-07
make qc-all

# Run HR audit for different participant
make qc-hr PID=P000002 SNAPSHOT=2025-10-22

# Run steps audit
make qc-steps
```

---

## Step 4: Testing and Validation

### 4.1 Test HR Audit

```bash
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain hr
```

**Expected Output**:

```
================================================================================
HR FEATURE INTEGRITY AUDIT: P000001/2025-11-07
================================================================================

✓ Event-level Parquet: 4,677,088 HR records
✓ Daily cache Parquet: 1315 days
✓ Cache schema: All 5 metrics present

Fabrication Check:
  hr_min == hr_mean: 7/1315 (0.5%)
  hr_std == 0.0: 7/1315 (0.5%)

✓ daily_cardio.csv: 1315 days
✓ features_daily_unified.csv: 2828 days, HR columns: ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples']
  Days with HR data: 1315/2828 (46.5%)
✓ features_daily_labeled.csv: 2828 days, HR columns: ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples']

Computing per-day statistics...

HR Summary Statistics:
  Mean HR (daily avg): 83.15 ± 10.23 bpm
  Min HR range: 55.0 to 120.0 bpm
  Max HR range: 100.0 to 208.0 bpm
  Std HR (daily): 14.23 ± 5.67 bpm
  Samples/day: 3556 (median), range 1-30653

✓ Saved per-day results: data/etl/P000001/2025-11-07/qc/hr_feature_audit.csv
✓ Saved audit summary: data/etl/P000001/2025-11-07/qc/hr_audit_summary.json

================================================================================
AUDIT SUMMARY
================================================================================

✅ AUDIT STATUS: PASS
   Domain: HR
   Participant: P000001
   Snapshot: 2025-11-07

================================================================================
```

**Exit code**: 0 (PASS)

### 4.2 Test Steps Audit

```bash
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain steps
```

**Expected Output**: Similar format, domain-specific checks

### 4.3 Test Sleep Audit

```bash
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep
```

**Expected Output**: Similar format, domain-specific checks

---

## Acceptance Criteria

✅ **CLI Interface**:

- [ ] Can run: `python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain hr`
- [ ] Can run: `python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain steps`
- [ ] Can run: `python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep`
- [ ] Returns exit code 0 (PASS) or 1 (FAIL)

✅ **Output Files**:

- [ ] Generates CSV: `data/etl/{PID}/{SNAPSHOT}/qc/hr_feature_audit.csv`
- [ ] Generates CSV: `data/etl/{PID}/{SNAPSHOT}/qc/steps_feature_audit.csv`
- [ ] Generates CSV: `data/etl/{PID}/{SNAPSHOT}/qc/sleep_feature_audit.csv`
- [ ] Generates JSON summaries for each domain

✅ **Console Output**:

- [ ] Clear PASS/FAIL status at end
- [ ] Lists critical issues if FAIL
- [ ] Lists warnings (non-blocking)
- [ ] Shows summary statistics

✅ **Makefile Integration**:

- [ ] `make qc-hr` runs HR audit
- [ ] `make qc-steps` runs Steps audit
- [ ] `make qc-sleep` runs Sleep audit
- [ ] `make qc-all` runs all three

✅ **Documentation**:

- [ ] `STEPS_FEATURE_INTEGRITY_AUDIT.md` created (mirrors HR audit structure)
- [ ] `SLEEP_FEATURE_INTEGRITY_AUDIT.md` created (mirrors HR audit structure)
- [ ] Both documents have "Next Steps" section with fix recommendations

✅ **Code Quality**:

- [ ] No circular dependencies
- [ ] Docstrings for all methods
- [ ] Consistent code style with rest of repository
- [ ] No import errors

---

## Implementation Notes

1. **Use existing HR audit as template** - the pattern is proven and comprehensive
2. **Reuse common patterns** - fabrication checks, consistency checks, NaN analysis
3. **Keep it surgical** - focus on data integrity, not feature engineering
4. **Document everything** - PhD-level traceability required
5. **Exit codes matter** - CI/CD needs 0 (PASS) or 1 (FAIL)

---

## Next Steps After Implementation

1. Run all three audits on P000001/2025-11-07
2. Generate audit markdown documents based on findings
3. Fix any critical issues discovered
4. Integrate into CI/CD pipeline (GitHub Actions)
5. Extend to activity domain (exercise_min, move_kcal, stand_hours)

---

**Status**: Ready for implementation
**Estimated Time**: 2-3 hours for full implementation + testing
**Priority**: HIGH (regression testing critical for scientific reproducibility)
