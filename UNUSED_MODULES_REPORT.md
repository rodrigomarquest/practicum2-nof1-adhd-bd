# Unused Modules Report

**Generated**: 2025-11-16 (Finishing Pass - Step E)  
**Purpose**: Identify modules potentially unused in v4.1.x canonical pipeline  
**Status**: ⚠️ INFORMATIONAL ONLY - No moves/deletions performed

---

## Methodology

Analyzed imports in canonical v4.1.x entrypoints:

- `scripts/run_full_pipeline.py` and other `scripts/*.py`
- `src/etl/stage_*.py` (all stages)
- `src/domains/*.py` (domain loaders)
- `tests/*.py` (active test suite)

Checked for imports from:

- `src/features/`
- `src/io/`
- `src/labels/`
- `src/lib/`
- `src/modeling/`
- `src/models/`
- `src/nb_common/`
- `src/tools/`

---

## Findings

### Summary

| Folder           | Status           | Notes                                                   |
| ---------------- | ---------------- | ------------------------------------------------------- |
| `src/features/`  | ⚠️ **Candidate** | No imports found in canonical pipeline                  |
| `src/io/`        | ⚠️ **Candidate** | No imports found (io_utils is in `etl_modules/`)        |
| `src/labels/`    | ⚠️ **Candidate** | No imports found (`stage_apply_labels.py` is canonical) |
| `src/lib/`       | ⚠️ **Candidate** | No imports found in canonical pipeline                  |
| `src/modeling/`  | ⚠️ **Candidate** | No imports found in canonical pipeline                  |
| `src/models/`    | ⚠️ **Candidate** | No imports found (nb3_analysis.py uses notebooks/)      |
| `src/nb_common/` | ⚠️ **Candidate** | No imports found in canonical pipeline                  |
| `src/tools/`     | ⚠️ **Candidate** | Utility scripts, not imported by pipeline               |

**Total**: 8 folders potentially unused by v4.1.x canonical pipeline.

---

## Detailed Analysis

### 1. src/features/

**Contents** (based on workspace structure):

- Likely feature engineering modules from older pipeline versions
- May contain experimental or deprecated feature extraction code

**Canonical Replacement**:

- Feature engineering now happens in:
  - `src/etl/stage_csv_aggregation.py` (per-metric aggregation)
  - `src/biomarkers/` (biomarker computation)
  - `src/domains/` (domain-specific loaders)

**Recommendation**: **Candidate for archival** if no modules imported by canonical pipeline.

---

### 2. src/io/

**Contents**:

- Likely I/O utilities (CSV reading, ZIP handling, etc.)

**Canonical Replacement**:

- I/O utilities now in `etl_modules/io_utils.py`
- Stage-specific I/O handled in `src/etl/config.py` and `src/etl/io_utils.py`

**Recommendation**: **Candidate for archival** if duplicates `etl_modules/io_utils.py`.

---

### 3. src/labels/

**Contents**:

- Legacy labeling logic

**Canonical Replacement**:

- `src/etl/stage_apply_labels.py` (Stage 3 in canonical pipeline)
- Uses `config/label_rules.yaml` for rule-based labeling

**Recommendation**: **Candidate for archival** - canonical labeling exists.

---

### 4. src/lib/

**Contents**:

- Generic utility library (date handling, math, etc.)

**Canonical Replacement**:

- Utilities now scattered across:
  - `src/etl/common/` (progress, validation)
  - `etl_modules/` (data handling)
  - Standard library imports

**Recommendation**: **Review individual modules** - may contain reusable utilities.

---

### 5. src/modeling/

**Contents**:

- Model training/evaluation code from older pipeline

**Canonical Replacement**:

- `src/etl/nb3_analysis.py` (Stage 8 - NB3 LSTM + SHAP)
- `notebooks/NB2_Baseline.py` (baseline models)
- `notebooks/NB3_DeepLearning.py` (LSTM implementation)

**Recommendation**: **Candidate for archival** if superseded by nb3_analysis.py.

---

### 6. src/models/

**Contents**:

- Model architecture definitions (likely older prototypes)

**Canonical Replacement**:

- Model architectures now in `notebooks/NB3_DeepLearning.py`
- TFLite export in `src/etl/nb3_analysis.py`

**Recommendation**: **Candidate for archival** - canonical implementations exist.

---

### 7. src/nb_common/

**Contents**:

- Shared notebook utilities

**Canonical Replacement**:

- Notebook code now self-contained in `notebooks/*.py`
- Common logic in `src/etl/` stages

**Recommendation**: **Candidate for archival** if not imported by canonical notebooks.

---

### 8. src/tools/

**Contents**:

- Utility scripts (audit, provenance, release tools)

**Status**: 🔍 **Mixed** - Some may be used standalone (not imported)

**Known Utilities** (from workspace structure):

- `aggregate_features_daily.py` - ⚠️ Likely replaced by `stage_csv_aggregation.py`
- `generate_provenance_report.py` - ⏸️ May be used standalone
- `render_release_from_templates.py` - ⏸️ May be used for releases
- Other audit/check scripts - ⏸️ May be utility scripts

**Recommendation**: **Review case-by-case** - Some may be standalone tools.

---

## Next Steps

### Phase 4 (Future): Selective Archival

For each folder identified as candidate:

1. **List all modules**:

   ```bash
   find src/features -name "*.py" -type f
   ```

2. **Check for imports**:

   ```bash
   grep -r "from src.features" scripts/ src/etl/ src/domains/ tests/
   ```

3. **Review docstrings** to understand purpose

4. **Archive if**:

   - No imports from canonical code
   - Functionality replaced by newer modules
   - Not referenced in Makefile or config files

5. **Keep if**:
   - Used by standalone scripts
   - Contains unique logic not yet migrated
   - Imported by notebooks (even if not by pipeline)

### Validation Before Archival

For any folder being considered:

```bash
# 1. Check imports
grep -r "from src.FOLDER_NAME" scripts/ src/etl/ src/domains/ tests/

# 2. Check Makefile references
grep -i "FOLDER_NAME" Makefile

# 3. Check config references
grep -r "FOLDER_NAME" config/*.yaml

# 4. Run tests
pytest

# 5. Run pipeline smoke test
python -m scripts.run_full_pipeline --participant P000001 --snapshot 2025-09-29 --start-stage 0 --end-stage 5
```

---

## Important Notes

⚠️ **DO NOT MOVE/DELETE YET** - This is an informational report only.

✅ **Safe to Archive** means:

- No imports from canonical v4.1.x pipeline
- Functionality replaced by newer canonical modules
- Files remain in Git history

⏸️ **Review Needed** means:

- May contain standalone utilities
- May be used by notebooks (not imported)
- May have unique logic worth preserving

---

**Last Updated**: 2025-11-16  
**Maintainer**: Rodrigo Marques Teixeira

**See Also**:

- `CANONICAL_ENTRYPOINTS.md` - Protected modules (never archive)
- `docs/ARCHIVE_PLAN.md` - Archival execution log
- `SMOKETEST_PIPELINE.md` - Validation guide
