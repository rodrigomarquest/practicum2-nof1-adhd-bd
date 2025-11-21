# Ì∫Ä v4.1.8 ‚Äì ML6/ML7 Pipeline Finalization & Public Reproducibility

**Release date:** 2025-11-21  
**Branch:** `main`  
**Pipeline version:** v4.1.7 (scientific), v4.1.8 (public snapshot)  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business ‚Äì Practicum Part 2 (N-of-1 ADHD + BD)

---

## Ì¥ß Summary

This release finalizes the deterministic ML6/ML7 pipeline with temporal filtering, segment-wise normalization, and public ETL snapshot for external reproducibility. All modeling results align with dissertation Tables 3-4 and include comprehensive drift analysis (ADWIN, KS tests, SHAP).

**Key improvements:**
- ML7 temporal filter (>= 2021-05-11) applied for consistency with ML6
- Segment-aware MICE imputation (0 missing values in ML dataset)
- Public ETL snapshot (669 days) enables stages 3-9 replication
- Automated QC framework passes all domains (HR, sleep, activity)

---

## Ì≥ä Performance Results

### ML6 Logistic Regression Baseline

**6-fold calendar-based cross-validation:**
- **Macro-F1:** 0.69 ¬± 0.16
- **Best fold:** 0.87 (Fold 2, Sept-Nov 2022)
- **Worst fold:** 0.39 (Fold 5, Mar-May 2024)
- **Balanced accuracy range:** 0.54-0.87

**Top predictors (SHAP):**
1. Sleep efficiency
2. HRV RMSSD (autonomic variability)
3. Normalized heart rate mean

### ML7 LSTM Sequence Model

**14-day sliding window:**
- **Macro-F1:** 0.25
- **Weighted F1:** 0.42
- **Balanced accuracy:** 0.39
- **AUROC (One-vs-Rest):** 0.58

**Key finding:** ML6 baseline outperforms ML7 sequence model (0.69 vs 0.25), consistent with literature showing simple models excel under high non-stationarity and weak labels.

---

## ÔøΩÔøΩ Dataset Configuration

- **Total days (full timeline):** 2,828 (2017-12-04 to 2025-10-21)
- **ML-filtered dataset:** 1,625 days (>= 2021-05-11, Amazfit-only era)
- **ML7 sequences:** 1,612 (14-day windows)
- **Label distribution:** 322/596/707 (19.8% / 36.7% / 43.5%)
- **Behavioral segments:** 119 (full), 48 (ML-filtered)
- **Missing data:** 0 (post-MICE imputation)

---

## Ì¥ç Drift Detection Results

- **ADWIN long-term drifts:** 6 events
- **KS distribution shifts:** 93 significant (p < 0.05)
- **SHAP temporal patterns:** Early folds prioritize cardiovascular; recent folds emphasize sleep

---

## Ìºê Public ETL Snapshot

**Location:** `data/etl/P000001/2025-11-07/`  
**Coverage:** 669 days (2023-12-01 to 2025-09-29)  
**Access:** OneDrive for NCI evaluators

**Enables stages 3-9 replication without raw health data**

---

## ‚öôÔ∏è Quick Start

```bash
# Clone repository
git clone https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd.git
cd practicum2-nof1-adhd-bd

# Setup environment (Python 3.11+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements/local.txt

# Run pipeline from ETL snapshot (stages 3-9)
python scripts/run_full_pipeline.py \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-stage 3 \
  --end-stage 9
```

---

## Ì≥Ñ Documentation

- `REPRODUCING_WITH_ETL_SNAPSHOT.md` ‚Äì External reproducibility guide
- `ETL_PUBLIC_SHARING_IMPLEMENTATION.md` ‚Äì Data sharing strategy
- `ML7_TEMPORAL_FILTER_UPDATE_2025-11-21.md` ‚Äì Pipeline technical details

---

## Ì∑æ Citation

Teixeira, R. M. (2025). _A Deterministic N-of-1 Pipeline for Multimodal Digital Phenotyping in ADHD and Bipolar Disorder._  
MSc Dissertation, National College of Ireland.  
GitHub: [https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

‚öñÔ∏è **License:** CC BY-NC-SA 4.0  
**Supervisor:** Dr. Agatha Mattos  
**Student ID:** 24130664
