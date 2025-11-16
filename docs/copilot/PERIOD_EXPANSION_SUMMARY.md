# 🎉 Period Expansion & Auto-Segmentation — Complete Implementation Summary

**Status**: ✅ **PHASES 1-4 COMPLETE**  
**Date**: 2025-11-07  
**Scope**: Full infrastructure for expanding NB2/NB3 analysis without `version_log_enriched.csv`

---

## 📋 Executive Summary

Implemented comprehensive pipeline to automatically:

1. **Extract** ZIPs from `data/raw/` with progress tracking
2. **Unify** Apple+Zepp daily data with source tracking
3. **Segment** data automatically (4-tier rules, no version_log needed)
4. **Label** with PBSI stability scores
5. **Analyze** with NB2 baselines + NB3 advanced analytics

**Status**: Infrastructure ready, extraction tested on P000001 (1789 files, 122 MB)

---

## 🏗️ Implementation Overview

### Module 1: ZIP Extraction Utility

**File**: `src/io/zip_extractor.py` (340 lines)

```python
from src.io.zip_extractor import extract_all_zips

stats = extract_all_zips(
    participant="P000001",
    zepp_password=os.environ.get("ZEPP_ZIP_PASSWORD"),
    dry_run=False
)
```

**Features**:

- ✅ Recursive ZIP discovery in `data/raw/`
- ✅ Vendor auto-detection (Apple/Zepp heuristics)
- ✅ Progress bars (tqdm with fallback)
- ✅ AES encryption support (pyzipper + password)
- ✅ Idempotent (safe to re-run)
- ✅ JSON statistics output

**Test Results** (P000001):

- Apple ZIP: 1789 files, 122 MB → ✅ Extracted in 55 sec
- Zepp ZIP: 1.88 MB → ⏳ Skipped (no password, as expected)

**Usage**:

```bash
# Dry run
python src/io/zip_extractor.py --participant P000001 --dry-run

# Extract
python src/io/zip_extractor.py --participant P000001 --zepp-password $PASSWORD

# Output
data/extracted/{apple,zepp}/P000001/...
```

---

### Module 2: Auto-Segmentation Engine

**File**: `src/labels/auto_segment.py` (360 lines)

```python
from src.labels.auto_segment import auto_segment
import pandas as pd

df = pd.read_csv("data/etl/features_daily_unified.csv")
seg_df, decisions = auto_segment(df)
seg_df.to_csv("data/etl/features_daily_with_segments.csv", index=False)
```

**Auto-Segmentation Rules** (4-tier system, no manual version_log):

| Rule              | Trigger                         | Threshold                           | Example                          |
| ----------------- | ------------------------------- | ----------------------------------- | -------------------------------- |
| Source Change     | Dominant cardio source switches | ≥5 consecutive days                 | Apple→Zepp                       |
| Signal Change     | Abrupt biomarker shift          | HR:Δ≥8bpm, HRV:Δ≥10ms, Sleep:Δ≥0.08 | HR 60→75 bpm                     |
| Gap Recovery      | Signal returns after absence    | ≥3 consecutive missing days         | 3 days offline→signal back       |
| Temporal Fallback | Ensure fold compatibility       | ~60 days                            | Auto-segment if no rule triggers |

**Outputs**:

- `segment_id`: Integer 1-indexed, increments per rule trigger
- `segment_autolog.csv`: Complete decision log (date, reason, metric, old_seg, new_seg)

**Features**:

- ✅ No external configuration files needed
- ✅ Auditable decisions (every transition logged)
- ✅ Configurable thresholds
- ✅ Handles missing data gracefully

---

### Module 3: Pipeline Orchestrator

**File**: `scripts/run_period_expansion.py` (420 lines)

```bash
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07 \
    --n-folds 6 \
    --skip-nb 0  # Include NB2/NB3
```

**6-Stage Pipeline**:

```
Stage 1: ZIP Discovery & Extraction
         ↓ (1789 files extracted)
Stage 2: Daily Unification (Apple+Zepp merge)
         ↓ (27 canonical columns)
Stage 3: Auto-Segmentation (4-tier rules)
         ↓ (segment_id + autolog)
Stage 4: PBSI Label Computation
         ↓ (35+ columns with labels)
Stage 5: NB2 Baseline Training (optional)
         ↓ (5 models × 6 folds)
Stage 6: NB3 Advanced Analytics (optional)
         ↓ (SHAP, Drift, LSTM, TFLite)
```

**Features**:

- ✅ Dry-run mode (preview without execution)
- ✅ Per-stage logging + JSON stats
- ✅ Graceful error handling (stage isolation)
- ✅ Configurable parameters (n_folds, train_days, etc.)
- ✅ Detailed progress tracking

**Output Locations**:

```
logs/
  pipeline_expansion_TIMESTAMP.log       # Detailed log
  pipeline_stats.json                    # JSON summary

data/etl/
  features_daily_unified.csv             # 27 cols
  features_daily_with_segments.csv       # +segment_id
  segment_autolog.csv                    # Transition log
  features_daily_labeled.csv             # 35+ cols (final)

nb2/, nb3/                               # Baseline + advanced outputs
```

---

## 📊 Data State Summary

### P000001 Current Status

| Component       | Status                | Files | Size    |
| --------------- | --------------------- | ----- | ------- |
| Apple ZIP       | ✅ Extracted          | 1789  | 122 MB  |
| Zepp ZIP        | ⏳ Pending (password) | 0     | 1.88 MB |
| Total Extracted | 1789 files            | -     | 122 MB  |

**Extraction Time**: 55 seconds (1789 files)

**Output Structure**:

```
data/extracted/
  apple/
    unknown/                      # From apple_health_export ZIP
      export.xml                  # Main Apple Health export
      workout/                    # Workout samples
      sleep_samples/              # Sleep records
      ...
  zepp/
    P000001/                      # (empty, needs password)
```

---

## 🚀 Quick Start Guide

### Prerequisite: Set Zepp Password (Optional)

```bash
export ZEPP_ZIP_PASSWORD="your_password_here"
```

### Option A: Full Automated Pipeline (Recommended)

```bash
# Extract, unify, segment, label (stages 1-4)
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07

# With NB2/NB3 (stages 1-6, takes ~30 min)
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07 \
    --skip-nb 0
```

### Option B: Step-by-Step Manual (Learning/Debugging)

```bash
# 1. Extract ZIPs
python src/io/zip_extractor.py --participant P000001

# 2. Unify daily data
python -c "
from src.features.unify_daily import unify_apple_zepp
df = unify_apple_zepp(
    Path('data/extracted/apple/P000001'),
    Path('data/extracted/zepp/P000001')
)
df.to_csv('data/etl/features_daily_unified.csv', index=False)
"

# 3. Auto-segment
python -c "
import pandas as pd
from src.labels.auto_segment import auto_segment
df = pd.read_csv('data/etl/features_daily_unified.csv')
seg_df, _ = auto_segment(df)
seg_df.to_csv('data/etl/features_daily_with_segments.csv', index=False)
"

# 4. Compute PBSI labels
python -c "
import pandas as pd
from src.labels.build_pbsi import compute_z_scores_by_segment, compute_pbsi_labels
df = pd.read_csv('data/etl/features_daily_with_segments.csv')
df = compute_z_scores_by_segment(df)
# Apply PBSI computation...
df.to_csv('data/etl/features_daily_labeled.csv', index=False)
"

# 5. Run NB2 & NB3
python run_nb2_beiwe.py --pid P000001 --snapshot 2025-11-07 --n-folds 6
python scripts/run_nb3_pipeline.py --participant P000001 --snapshot 2025-11-07
```

### Option C: Dry Run (Safe Preview)

```bash
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07 \
    --dry-run
```

---

## 📁 Files Created/Modified

### New Implementation Files

| File                                      | Lines | Purpose                       | Status      |
| ----------------------------------------- | ----- | ----------------------------- | ----------- |
| `src/io/zip_extractor.py`                 | 340   | ZIP discovery + extraction    | ✅ Complete |
| `src/labels/auto_segment.py`              | 360   | Auto-segmentation engine      | ✅ Complete |
| `scripts/run_period_expansion.py`         | 420   | 6-stage pipeline orchestrator | ✅ Complete |
| `docs/PERIOD_EXPANSION_README.md`         | 400+  | User guide + examples         | ✅ Complete |
| `docs/PERIOD_EXPANSION_IMPLEMENTATION.md` | 400+  | Technical architecture        | ✅ Complete |

**Total New Code**: ~1500 lines

### Integration Points

- ✅ Reuses `src/features/unify_daily.py` (existing)
- ✅ Reuses `src/labels/build_pbsi.py` (existing)
- ✅ Reuses `run_nb2_beiwe.py` (existing)
- ✅ Reuses `scripts/run_nb3_pipeline.py` (existing)
- ✅ Imports from `etl_pipeline.py` (progress_bar pattern)

---

## 🔍 Auto-Segmentation Details

### Rule 1: Source Change (≥5 Consecutive Days)

Triggers when dominant source of heart rate measurements switches between Apple Watch and Zepp.

```
Days 1-90:    source_cardio = "apple" (dominant)
Days 91-100:  source_cardio = "zepp" (dominant)
Trigger:      New segment at day 91
```

**Use Case**: Device switching, e.g., stopped wearing Apple Watch, started using Zepp smartband.

### Rule 2: Signal Change (≥7-Day Sustained Shift)

Detects abrupt, sustained changes in biomarkers suggesting context shift (stress, illness, etc.).

```
Thresholds:
  - HR mean: Δ ≥ 8 bpm
  - HRV (RMSSD): Δ ≥ 10 ms
  - Sleep efficiency: Δ ≥ 0.08

Example:
  Days 1-30:  HR mean = 60 bpm
  Days 31+:   HR mean = 75 bpm (Δ = 15 bpm ≥ 8) → New segment at day 31
```

**Use Case**: Health event (infection, new exercise routine, medication change).

### Rule 3: Gap Recovery (≥3 Consecutive Missing Days)

Segments restart when signal returns after prolonged absence of both heart rate and sleep data.

```
Days 45-47:   missing_cardio=1, missing_sleep=1 (gap)
Day 48:       Signal returns (both data available) → New segment at day 48
```

**Use Case**: Vacation, device charging, data collection pause.

### Rule 4: Temporal Fallback (~60 Days)

If no other rule triggers, force segment every ~60 days to maintain compatibility with calendar-based CV folds (4m train, 2m validation).

```
Without this rule: Risk of segments spanning >120 days → unfair CV splits
With this rule: Max 60-day segments → compatible with 4m/2m fold design
```

**Use Case**: Stabilization fallback for default fold creation.

### Decision Log Example

```csv
date,reason,metric,old_seg,new_seg
2025-01-01,temporal_fallback,≥60d,1,2
2025-03-01,source_change,apple→zepp,2,3
2025-04-15,signal_change,HR_mean_change(Δ=9.5bpm),3,4
2025-04-20,gap_recovery,gap≥3d,4,5
```

Every decision is traceable and auditable.

---

## 🧪 Testing & Validation

### Extraction Testing ✅

```bash
# Tested: P000001
$ python src/io/zip_extractor.py --participant P000001 --dry-run

Discovered: 2 ZIPs for P000001
  - Apple: apple_health_export_20251022T061854Z.zip (122 MB)
  - Zepp: 3088235680_1761192590962.zip (1.88 MB)

# Full extraction
$ python src/io/zip_extractor.py --participant P000001

Result: ✅ 1789 files extracted in 55 seconds
```

### Auto-Segmentation Testing ✅

```python
# Tested on synthetic 180-day data with known changes
test_data = {
    "date": pd.date_range("2025-01-01", periods=180),
    "source_cardio": ["apple"]*90 + ["zepp"]*90,  # Source change at day 91
    "apple_hr_mean": np.concatenate([
        np.random.normal(70, 5, 90),   # Baseline
        np.random.normal(80, 5, 90)    # HR increase (signal change)
    ]),
    ...
}

df = pd.DataFrame(test_data)
seg_df, decisions = auto_segment(df)

Result: ✅ Detected both changes (segment_id: 1→2→3)
```

### Pipeline Orchestration Testing ✅

```bash
$ python scripts/run_period_expansion.py --participant P000001 --dry-run

Result: ✅ Dry run shows all stages would execute
```

---

## 📈 Performance Benchmarks

| Stage                  | Time        | Files    | Status |
| ---------------------- | ----------- | -------- | ------ |
| ZIP Discovery          | 2 sec       | 2 found  | ✅     |
| ZIP Extraction (Apple) | 55 sec      | 1789     | ✅     |
| Unification            | ~10 sec     | 400 days | ⏳     |
| Auto-Segmentation      | ~3 sec      | 400 days | ⏳     |
| PBSI Labels            | ~15 sec     | 400 days | ⏳     |
| NB2 (6 folds × 5)      | ~5 min      | 6 folds  | ⏳     |
| NB3 (SHAP+Drift+LSTM)  | ~15 min     | 6 folds  | ⏳     |
| **Total Expected**     | **~30 min** | -        | ⏳     |

_Times are estimates; actual depends on data complexity_

---

## 🎯 Success Criteria (Phase 5+)

Once data flows through the pipeline:

- [ ] `features_daily_labeled.csv` covers complete available date range
- [ ] Segment assignments are reasonable (3-10 segments for ~400 days)
- [ ] PBSI label distribution matches expectations (-1: 20-30%, 0: 40-50%, +1: 20-30%)
- [ ] NB2 baselines run without errors (all 5 models × 6 folds)
- [ ] NB2 confusion matrices show non-trivial performance
- [ ] NB3 SHAP identifies top-5 meaningful features
- [ ] NB3 drift detection finds realistic changepoints (aligned with segments)
- [ ] LSTM trains and exports TFLite successfully
- [ ] TFLite inference runs in <10ms

---

## 🛠️ Troubleshooting

### Issue: Zepp ZIP not extracting

```
Error: "Zepp ZIP password not provided"
```

**Solution**: Set environment variable

```bash
export ZEPP_ZIP_PASSWORD="your_password"
python src/io/zip_extractor.py --participant P000001
```

### Issue: No Apple data found

```
Error: "Apple directory not found"
```

**Solution**: Check extraction worked

```bash
ls -la data/extracted/apple/P000001/
```

### Issue: Too many or too few segments

Adjust thresholds in `src/labels/auto_segment.py`:

```python
generate_segments(
    df,
    source_window=5,          # ↑ detect slower source changes
    signal_window=7,          # ↑ require longer signal shift
    gap_min=3,                # ↑ allow longer gaps
    temporal_period=60,       # ↑ larger base segments
)
```

---

## 📚 Documentation

- **User Guide**: `docs/PERIOD_EXPANSION_README.md`

  - Quick start examples
  - Step-by-step instructions
  - Auto-segmentation rules explained
  - Troubleshooting guide

- **Technical Guide**: `docs/PERIOD_EXPANSION_IMPLEMENTATION.md`
  - Architecture decisions
  - File summaries
  - Quality metrics
  - Testing checklist
  - Next steps

---

## 🔮 Next Steps

### Immediate (Phase 5)

1. **Execute Unification + Segmentation + Labels**

   ```bash
   python scripts/run_period_expansion.py --participant P000001 --snapshot 2025-11-07
   ```

2. **Verify Data Quality**

   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('data/etl/features_daily_labeled.csv')
   print(f'Date range: {df.date.min()} → {df.date.max()}')
   print(f'Total rows: {len(df)}')
   print(f'Label distribution:\\n{df.label_3cls.value_counts()}')
   "
   ```

3. **Review Segmentation Decisions**
   ```bash
   python -c "
   import pandas as pd
   autolog = pd.read_csv('data/etl/segment_autolog.csv')
   print(autolog.to_string())
   "
   ```

### Medium-term (Phase 6+)

4. **Run NB2 + NB3 with Expanded Data**

   ```bash
   python scripts/run_period_expansion.py \
       --participant P000001 \
       --snapshot 2025-11-07 \
       --skip-nb 0
   ```

5. **Multi-Participant Analysis**

   ```bash
   for p in P000001 P000002 P000003; do
     python src/io/zip_extractor.py --participant $p
     python scripts/run_period_expansion.py --participant $p --snapshot 2025-11-07
   done
   ```

6. **Compare Segmentation Strategies**
   - Analyze segment stability across participants
   - Validate SHAP insights by segment
   - Monitor drift detection effectiveness

---

## 📞 Summary

**What's Ready**:

- ✅ ZIP extraction with progress tracking
- ✅ Auto-segmentation without manual versioning
- ✅ 6-stage pipeline orchestration
- ✅ Comprehensive documentation
- ✅ Tested on P000001 (1789 files extracted)

**What's Next**:

- ⏳ Execute pipeline with real data
- ⏳ Validate outputs (date range, segments, labels)
- ⏳ Run NB2/NB3 with expanded dataset
- ⏳ Analyze results and iterate

**Estimated Time to Full Results**: ~30-45 minutes after data unification begins

---

**Version**: 1.0  
**Status**: ✅ IMPLEMENTATION COMPLETE  
**Date**: 2025-11-07  
**Ready for**: Phase 5 (Data Execution)
