# Thesis to Configuration Manual Migration Audit

**Date:** 2026-01-24  
**Objective:** Identify content suitable for migration from thesis body to Configuration Manual (Appendix F)  
**Context:** MSc examiner feedback: "Move implementation details to configuration manual (no page limit)"

---

## Executive Summary

**Estimated page reduction: 8–12 pages** (from current ~60 pages)

**Migration strategy:**

- **Type B (Verbatim move):** 15 blocks → ~6 pages saved
- **Type C (Summarize + reference):** 8 blocks → ~4 pages saved
- **Type A (Must remain):** Core methodology, evaluation logic, research design

**Key principle:** Preserve all conceptual/methodological content while relocating implementation specifics to Configuration Manual.

---

## Classification Legend

- **A** — Must remain in thesis body (core methodology, experimental design, evaluation)
- **B** — Candidate for verbatim move to Configuration Manual
- **C** — Candidate for summarization in thesis + detailed version in Configuration Manual

---

## SECTION 1: Introduction

**Current location:** `text/introduction.tex` (lines 1–80)

### Classification: **A (Must remain)**

**Rationale:** Core research context, motivation, contributions, structure outline. Essential for examiner understanding.

**Action:** No migration.

---

## SECTION 2: Related Work

**Current location:** `text/relatedwork.tex` (lines 1–150)

### Classification: **A (Must remain)**

**Rationale:** Academic literature review, gap analysis, positioning of contribution. Required for scholarly assessment.

**Action:** No migration.

---

## SECTION 3: Methodology

### BLOCK 3.1: Data Aggregation Pipeline Details

**Location:** `methodology.tex`, lines 108–166  
**Section:** §3.4 Data Aggregation Pipeline  
**Content type:** Technical implementation (Apple XML parsing, Zepp CSV processing)

**Classification:** **B (Verbatim move)**

**Rationale:**  
Implementation-specific parsing logic, file format details, and schema mappings are necessary for reproducibility but not required for understanding the research methodology conceptually. Examiners need to know _what_ was aggregated and _why_, but not the exact XML structure or CSV column names.

**Suggested location in Config Manual:** §F.9 "Data Source Parsing Specifications"

**Retention in thesis (1 paragraph summary):**

```
Daily physiological features were extracted from heterogeneous sources (Apple Health XML
and Zepp CSV) through deterministic parsing routines. Heart rate, sleep, and activity
metrics were aggregated into canonical daily summaries. Full parsing specifications,
schema mappings, and file format details are provided in Configuration Manual §F.9.
```

**Estimated page savings:** ~1.5 pages

---

### BLOCK 3.2: Daily Unification Technical Details

**Location:** `methodology.tex`, lines 167–215  
**Section:** §3.4.3 Daily Unification  
**Content type:** Data merging logic, priority rules, source flags

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _concept_ of source priority and unification is methodological; the _exact rules_ (e.g., "Apple steps preferred over Zepp unless Apple = NaN") are implementation. Conceptual logic must remain; detailed priority tables can move.

**Suggested location in Config Manual:** §F.10 "Unification Rules and Source Priority Tables"

**Retention in thesis (reduced to 1 paragraph):**

```
Apple Health data was prioritized for activity metrics when available; Zepp data served
as fallback for missing Apple records. Cardiovascular features originated exclusively
from Amazfit devices. Binary missingness flags and source identifiers were retained for
quality control. Detailed unification rules and priority tables are documented in
Configuration Manual §F.10.
```

**Estimated page savings:** ~1 page

---

### BLOCK 3.3: HR Feature Engineering Implementation

**Location:** `methodology.tex`, lines 216–280  
**Section:** §3.5 Heart-Rate Feature Engineering  
**Content type:** Canonical schema, caching mechanism, outlier thresholds

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The research contribution is segment-wise normalization and anti-leak design. The specific Parquet caching structure, vendor naming conventions, and exact outlier thresholds (30–220 bpm) are implementation details.

**Suggested location in Config Manual:** §F.11 "Heart Rate Feature Schema and Caching"

**Retention in thesis (reduced to 1 paragraph):**

```
Five daily HR metrics (mean, min, max, std, sample count) were computed from raw PPG
samples, cached in Parquet format, and filtered for biologically plausible ranges.
Caching ensures reproducibility across pipeline executions. Full schema, caching logic,
and outlier removal thresholds are specified in Configuration Manual §F.11.
```

**Estimated page savings:** ~1.5 pages

---

### BLOCK 3.4: PBSI Formula and Subscores (Mathematical Details)

**Location:** `methodology.tex`, lines 306–340  
**Section:** §3.6.2 Domain-Level Subscores + §3.6.3 Composite PBSI Score

**Classification:** **SPLIT (Conceptual A + Technical B)**

**Rationale:**  
The _existence_ of PBSI, its three domains (sleep/cardio/activity), and percentile-based thresholding are methodological. The _exact weights_ (e.g., 0.6 × sleep_hours + 0.4 × sleep_efficiency) and percentile values (P25 = -0.370, P75 = +0.321) are technical specifications.

**Action:**

- **Keep in thesis:** Conceptual explanation of PBSI domains, sign convention, percentile approach
- **Move to Config Manual (§F.12):** Exact formulas, weight coefficients, threshold values, RMSSD substitution details

**Retention in thesis (reduced formulas):**

```
PBSI combines three domain subscores:
- Sleep: weighted combination of duration and efficiency
- Cardiovascular: mean HR (inverted), HRV proxy (RMSSD), max HR (inverted)
- Activity: steps and exercise proxy

The composite score uses domain-specific weights (sleep: 0.40, cardio: 0.35, activity: 0.25)
and percentile-based thresholds (P25/P75) to produce three-class labels. Complete formulas,
weight rationale, and threshold computation are detailed in Configuration Manual §F.12.
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.5: SoM Labels Technical Processing

**Location:** `methodology.tex`, lines 367–420  
**Section:** §3.7 State of Mind (SoM) Labels

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _role_ of SoM as primary supervised target is methodological. The ETL module path (`src/domains/som/som_from_autoexport.py`), AutoExport data type (`HKStateOfMind`), and percentile-mapping logic are implementation.

**Suggested location in Config Manual:** §F.13 "SoM Label Extraction and Mapping"

**Retention in thesis (reduced to 1 paragraph):**

```
State of Mind (SoM) labels were extracted from Apple Health AutoExport and aggregated into
a three-class daily target (Negative/Unstable, Neutral, Positive/Stable). Coverage is sparse
(77 days) due to voluntary self-reporting. Deterministic extraction and class-mapping logic
are documented in Configuration Manual §F.13.
```

**Estimated page savings:** ~1 page

---

### BLOCK 3.6: Behavioural Segmentation Rules

**Location:** `methodology.tex`, lines 425–455  
**Section:** §3.8 Behavioural Segmentation

**Classification:** **A (Must remain)**

**Rationale:**  
Segmentation is a core anti-leak safeguard and methodological innovation. The two rules (calendar boundaries + gaps >1 day) must remain in thesis for academic assessment.

**Action:** No migration. (This is conceptually essential.)

---

### BLOCK 3.7: Missing Data Hardware Context Table

**Location:** `methodology.tex`, lines 463–490  
**Section:** §3.9.1 Missing Data Patterns and Hardware Context

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _justification_ for temporal filtering (MAR vs MNAR) is methodological. The exact device timeline (P1-P2 iPhone-only, P3 Amazfit GTR, P4-P6 GTR2/GTR4/Helio) is historical context useful for reproducibility but not required for understanding the MAR assumption.

**Suggested location in Config Manual:** §F.14 "Device Timeline and Missingness Patterns"

**Retention in thesis (1 sentence summary):**

```
Cardiovascular data became sustainably available from 2021-05-11 onwards (Amazfit GTR 2
deployment). Prior periods lacked wearable HR sensors, rendering MAR assumptions implausible.
Device timeline and coverage statistics are detailed in Configuration Manual §F.14.
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.8: MICE Configuration Parameters

**Location:** `methodology.tex`, lines 505–540  
**Section:** §3.9.2 Stage 2: MICE Imputation

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _segment-aware MICE strategy_ is methodological. The exact sklearn parameters (`max_iter=10`, `random_state=42`, `sample_posterior=True`) are technical configuration.

**Suggested location in Config Manual:** §F.15 "MICE Imputation Configuration"

**Retention in thesis (reduced):**

```
Segment-aware multiple imputation by chained equations (MICE) addressed residual missingness
(11.9% after temporal filtering). Imputation was performed separately within each behavioural
segment to prevent cross-contamination. Full MICE configuration (iteration limits, seeds,
posterior sampling) is specified in Configuration Manual §F.15.
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.9: Feature Sets (FS-A through FS-D) Tables

**Location:** `methodology.tex`, lines 623–645  
**Section:** §3.11.1 Experiment Suite and Feature Sets

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _existence_ of four feature sets for ablation is methodological. The exact feature lists (e.g., FS-A: `sleep_hours`, `sleep_quality_score`, `hr_mean/min/max/std`, ...) are technical inventory.

**Suggested location in Config Manual:** §F.16 "Feature Set Specifications (FS-A to FS-D)"

**Retention in thesis (reduced to summary table):**

```
Four feature sets were evaluated:
- FS-A (10 features): baseline sleep, cardio, activity
- FS-B (15 features): FS-A + HRV metrics
- FS-C (18 features): FS-B + medication features
- FS-D (19 features): FS-C + PBSI auxiliary feature

Complete feature inventories are provided in Configuration Manual §F.16.
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.10: ML6/ML7 Algorithm Lists

**Location:** `methodology.tex`, lines 647–678  
**Section:** §3.11.2 ML6-Extended + §3.11.3 ML7-Extended

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _types_ of models (classical tabular, deep sequence) and evaluation protocol are methodological. The exact enumeration of 10 algorithms and 4 architectures with hyperparameters can move.

**Suggested location in Config Manual:** §F.17 "Model Architecture Specifications"

**Retention in thesis (reduced):**

```
ML6-Extended evaluated 10 classical algorithms (logistic regression, Random Forest, gradient
boosting, XGBoost, LightGBM, SVM, Naive Bayes, KNN). ML7-Extended evaluated 4 deep learning
architectures (LSTM, GRU, Conv1D, CNN-LSTM). All models used 6-fold temporal cross-validation
and balanced class weights. Complete model configurations and hyperparameters are documented
in Configuration Manual §F.17.
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.11: Drift Analysis Procedures

**Location:** `methodology.tex`, lines 679–740  
**Section:** §3.12 Drift and Explainability Analysis

**Classification:** **A (Must remain)**

**Rationale:**  
ADWIN, KS tests, and SHAP-based drift monitoring are core methodological contributions. The _interpretation_ of drift results (6 ADWIN drifts, 93 KS shifts) is evaluation content.

**Action:** No migration. (Essential methodology.)

---

### BLOCK 3.12: Deterministic Execution Command

**Location:** `methodology.tex`, lines 755–765  
**Section:** §3.13.1 Deterministic Execution

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _principle_ of determinism (fixed seeds, Makefile workflow) is methodological. The exact command syntax (`python -m scripts.run_full_pipeline --participant P000001 ...`) is operational detail.

**Suggested location in Config Manual:** Already exists in Appendix F (§F.5.1)

**Retention in thesis (1 sentence):**

```
The entire pipeline is deterministic and re-runnable via a Makefile-driven workflow with
fixed seeds (see Configuration Manual §F.5).
```

**Estimated page savings:** ~0.3 pages

---

### BLOCK 3.13: Provenance Directory Structure

**Location:** `methodology.tex`, lines 767–795  
**Section:** §3.13.2 Provenance Tracking

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _existence_ of provenance tracking is methodological. The exact directory tree (`data/etl/{PID}/{SNAPSHOT}/`) is operational detail.

**Suggested location in Config Manual:** Already exists in Appendix F (§F.4)

**Retention in thesis (1 sentence):**

```
Each pipeline execution generates a timestamped snapshot directory containing all intermediate
artifacts (see Configuration Manual §F.4 for structure).
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.14: QC Makefile Targets

**Location:** `methodology.tex`, lines 797–830  
**Section:** §3.13.3 Automated Quality-Control Framework

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _existence_ of automated QC is methodological validation. The exact Makefile targets (`make qc-hr`, `make qc-steps`, etc.) and exit code conventions (0=PASS) are operational.

**Suggested location in Config Manual:** Already exists in Appendix F (§F.7)

**Retention in thesis (reduced to 1 paragraph):**

```
Automated quality-control routines validated completeness, biological plausibility, and
consistency across heart rate, steps, and sleep domains. All domains passed QC in the final
snapshot (see Configuration Manual §F.7 for audit specifications).
```

**Estimated page savings:** ~0.5 pages

---

### BLOCK 3.15: QC Validation Results Table

**Location:** `methodology.tex`, lines 850–869  
**Section:** §3.14.3 Validation Results

**Classification:** **B (Verbatim move)**

**Rationale:**  
The _fact_ that QC passed is methodological validation. The exact numbers (1,315 validated days, 0.5% fabrication rate, etc.) are technical audit results.

**Suggested location in Config Manual:** §F.18 "Quality Control Audit Results"

**Retention in thesis (1 sentence):**

```
All physiological domains passed automated QC with no range violations (detailed audit
results in Configuration Manual §F.18).
```

**Estimated page savings:** ~0.3 pages

---

## SECTION 4: Results (Implementation.tex)

### BLOCK 4.1: Dataset Overview Table

**Location:** `implementation.tex`, lines 28–49  
**Section:** §4.1 Dataset Overview

**Classification:** **C (Summarize + reference)**

**Rationale:**  
High-level dataset characteristics (time span, SoM coverage) are essential results. Granular counts (5,986 Apple HR records parsed, 960 Zepp records) are technical inventory.

**Suggested location in Config Manual:** §F.19 "Dataset Statistics and Record Counts"

**Retention in thesis (reduced table to 5 rows):**

```
Property                   | Value
---------------------------|---------------------------
Time span (full)           | 2017-12-04 to 2025-12-07
Time span (SoM ML)         | 2023-11-19 to 2025-12-07
Total days                 | 2,868
SoM-labelled days          | 77
Unified features (FS-B)    | 15

Detailed record counts in Configuration Manual §F.19.
```

**Estimated page savings:** ~0.3 pages

---

### BLOCK 4.2: PBSI Label Distribution Table

**Location:** `implementation.tex`, lines 66–78  
**Section:** §4.2 PBSI Label Distribution

**Classification:** **A (Must remain)**

**Rationale:**  
Label distribution is a core evaluation result showing near-balanced classes (34.9%/33.0%/32.0%). Essential for assessing modelling context.

**Action:** No migration.

---

### BLOCK 4.3: Segmentation Statistics

**Location:** `implementation.tex`, lines 123–135  
**Section:** §4.3 Behavioural Segmentation

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _fact_ of 122 segments and their anti-leak role is methodological. The exact breakdown (93 time_boundary, 28 gap, 1 initial) is technical enumeration.

**Suggested location in Config Manual:** §F.20 "Segmentation Trigger Statistics"

**Retention in thesis (1 sentence):**

```
The segmentation procedure detected 122 behavioural segments across the full timeline,
with 57 segments retained after the ML temporal filter (breakdown in Configuration Manual §F.20).
```

**Estimated page savings:** ~0.2 pages

---

### BLOCK 4.4: ML6/ML7 Performance Tables

**Location:** `implementation.tex`, lines 147–180  
**Section:** §4.4 ML6 Baseline + §4.5 ML7 LSTM

**Classification:** **A (Must remain)**

**Rationale:**  
Performance metrics (F1, balanced accuracy) are core evaluation results. Required for research assessment.

**Action:** No migration.

---

## SECTION 5: Evaluation

### BLOCK 5.1: Extended ML6/ML7 Tables

**Location:** `evaluation.tex`, lines 89–142 + 154–175  
**Sections:** Extended ML6 Models (Table 5.2), Extended ML7 Models (Table 5.3)

**Classification:** **C (Summarize + reference)**

**Rationale:**  
The _comparison_ showing XGBoost (F1=0.631) and CNN-LSTM (F1=0.526) as best models is an evaluation result. The full 10×4 performance matrix with all algorithms and metrics is technical detail.

**Suggested location in Config Manual:** §F.21 "Extended Model Performance Matrix"

**Retention in thesis (reduced tables to top 3 models only):**

```
Top 3 ML6 Models           | F1-macro       | Balanced Acc.
---------------------------|----------------|---------------
Random Forest              | 0.517 ± 0.133  | 0.542
KNN-5                      | 0.503 ± 0.090  | 0.528
Gradient Boosting          | 0.499 ± 0.094  | 0.510

Top 3 ML7 Models           | F1-macro       | Balanced Acc.
---------------------------|----------------|---------------
CNN_LSTM                   | 0.526 ± 0.155  | 0.633
LSTM                       | 0.521 ± 0.234  | 0.648
Conv1D                     | 0.437 ± 0.035  | 0.541

Full leaderboards in Configuration Manual §F.21.
```

**Estimated page savings:** ~0.5 pages

---

## SECTION 6: Discussion

**Current location:** `text/discussion.tex`

### Classification: **A (Must remain)**

**Rationale:** Interpretation, limitations, ethical considerations, clinical context. Core scholarly content.

**Action:** No migration.

---

## SECTION 7: Conclusion

**Current location:** `text/conclusion.tex`

### Classification: **A (Must remain)**

**Rationale:** Summary of contributions, future work. Essential thesis component.

**Action:** No migration.

---

## SECTION 8: Design (Clinical Background)

**Current location:** `text/design.tex`

### Classification: **A (Must remain)**

**Rationale:** Clinical context for N-of-1 study. Provides essential background for interpretation.

**Action:** No migration.

---

## Migration Summary Table

| Block ID | Current Location           | Section Title             | Type  | Page Savings | Config Manual § |
| -------- | -------------------------- | ------------------------- | ----- | ------------ | --------------- |
| 3.1      | methodology.tex:108–166    | Data Aggregation Pipeline | B     | 1.5          | F.9             |
| 3.2      | methodology.tex:167–215    | Daily Unification         | C     | 1.0          | F.10            |
| 3.3      | methodology.tex:216–280    | HR Feature Engineering    | C     | 1.5          | F.11            |
| 3.4      | methodology.tex:306–340    | PBSI Formulas             | SPLIT | 0.5          | F.12            |
| 3.5      | methodology.tex:367–420    | SoM Label Processing      | C     | 1.0          | F.13            |
| 3.7      | methodology.tex:463–490    | Hardware Context Table    | B     | 0.5          | F.14            |
| 3.8      | methodology.tex:505–540    | MICE Configuration        | C     | 0.5          | F.15            |
| 3.9      | methodology.tex:623–645    | Feature Set Tables        | B     | 0.5          | F.16            |
| 3.10     | methodology.tex:647–678    | ML Algorithm Lists        | C     | 0.5          | F.17            |
| 3.12     | methodology.tex:755–765    | Deterministic Execution   | B     | 0.3          | F.5 (exists)    |
| 3.13     | methodology.tex:767–795    | Provenance Structure      | B     | 0.5          | F.4 (exists)    |
| 3.14     | methodology.tex:797–830    | QC Makefile Targets       | B     | 0.5          | F.7 (exists)    |
| 3.15     | methodology.tex:850–869    | QC Results Table          | B     | 0.3          | F.18            |
| 4.1      | implementation.tex:28–49   | Dataset Overview Table    | C     | 0.3          | F.19            |
| 4.3      | implementation.tex:123–135 | Segmentation Stats        | C     | 0.2          | F.20            |
| 5.1      | evaluation.tex:89–175      | Extended Model Tables     | C     | 0.5          | F.21            |

**Total estimated page reduction:** **10.1 pages**

---

## Implementation Plan

### Phase 1: Create New Configuration Manual Sections (§F.9–F.21)

**New sections to add to Appendix F:**

1. **§F.9 Data Source Parsing Specifications** ← Block 3.1
2. **§F.10 Unification Rules and Source Priority Tables** ← Block 3.2
3. **§F.11 Heart Rate Feature Schema and Caching** ← Block 3.3
4. **§F.12 PBSI Formula Specifications** ← Block 3.4 (technical part)
5. **§F.13 SoM Label Extraction and Mapping** ← Block 3.5
6. **§F.14 Device Timeline and Missingness Patterns** ← Block 3.7
7. **§F.15 MICE Imputation Configuration** ← Block 3.8
8. **§F.16 Feature Set Specifications (FS-A to FS-D)** ← Block 3.9
9. **§F.17 Model Architecture Specifications** ← Block 3.10
10. **§F.18 Quality Control Audit Results** ← Block 3.15
11. **§F.19 Dataset Statistics and Record Counts** ← Block 4.1
12. **§F.20 Segmentation Trigger Statistics** ← Block 4.3
13. **§F.21 Extended Model Performance Matrix** ← Block 5.1

### Phase 2: Edit Thesis Body Files

For each migrated block:

1. Replace detailed content with 1–2 sentence summary
2. Add cross-reference: "(see Configuration Manual §F.X)"
3. Preserve conceptual/methodological explanation
4. Verify citations are retained

### Phase 3: Validation

- [ ] Check all cross-references are correct (§F.X numbers)
- [ ] Verify no scientific content lost
- [ ] Ensure citations preserved
- [ ] Recompile thesis (verify page count reduction)
- [ ] Check Appendix F is complete and self-contained

---

## Risk Assessment

### Low Risk

- Moving technical tables (feature lists, QC results, device timelines)
- Moving verbatim command syntax (already duplicated in Appendix F)
- Moving implementation parameters (MICE config, outlier thresholds)

### Medium Risk

- Summarizing PBSI formulas (must retain conceptual explanation of domains)
- Condensing algorithm lists (must retain evaluation protocol description)

### No Risk (Must NOT migrate)

- Research design rationale
- Anti-leak safeguards conceptual explanation
- Segmentation methodology
- Performance metrics and evaluation results
- Drift detection methodology
- Discussion and interpretation

---

## Expected Outcome

**Before migration:** ~60 pages  
**After migration:** ~50 pages (16.7% reduction)  
**Configuration Manual growth:** +8 pages (from current ~6 pages to ~14 pages)

**Quality preservation:**

- ✅ All technical detail preserved (in Config Manual)
- ✅ All scientific content intact
- ✅ Improved readability (thesis body focuses on methodology/results)
- ✅ Enhanced reproducibility (Config Manual becomes comprehensive technical reference)
- ✅ Examiner-friendly (separates conceptual from implementation)

---

## Next Steps

1. **Obtain user approval** for migration plan
2. **Execute Phase 1** (create §F.9–F.21 in appendix_f.tex)
3. **Execute Phase 2** (edit methodology.tex, implementation.tex, evaluation.tex)
4. **Execute Phase 3** (validation and compilation)
5. **Generate before/after page count comparison**

---

**END OF AUDIT REPORT**
