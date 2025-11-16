# 🚀 v4.1.2 – Strict Path Normalization & Fail-Fast Password Validation

**Release date:** 2025-11-07  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

This release implements **strict canonical path normalization** in the ETL pipeline, removing intermediate PID subdirectories from extracted data paths, and adds **fail-fast Zepp password validation** to prevent silent extraction failures. These changes ensure cleaner directory structures, predictable file locations, and immediate error detection for encrypted Zepp data.

All changes maintain backward compatibility while enforcing cleaner, more predictable file organization across the pipeline.

---

## 🧩 Highlights

### 🔒 Security & Validation

- **Fail-Fast Zepp Password Check**
  - Pipeline aborts with `exit 2` if Zepp ZIP detected but no password provided
  - Checks both `--zepp-password` argument and `ZEP_ZIP_PASSWORD` / `ZEPP_ZIP_PASSWORD` environment variables
  - Validation occurs in:
    - Makefile `env` target (pre-flight check)
    - `run_full_pipeline.py` Stage 0 (before extraction)
  - Clear error message: `[FATAL] Zepp ZIP found but no password provided. Use --zepp-password or set ZEP_ZIP_PASSWORD.`

### 🗂️ Path Normalization

- **Canonical Paths Without Intermediate PID**
  - **Before (incorrect):** `data/etl/P000001/2025-11-07/extracted/apple/P000001/daily_sleep.csv`
  - **After (correct):** `data/etl/P000001/2025-11-07/extracted/apple/daily_sleep.csv`
  - Applies to both Apple and Zepp data
  - Modified stages:
    - `src/etl/stage_csv_aggregation.py` (write to canonical paths)
    - `src/etl/stage_unify_daily.py` (read from canonical paths only)

### 🔄 File Management

- **Automatic Renaming for Existing Files**
  - If `daily_sleep.csv` exists → automatically renamed to `daily_sleep.prev.csv`
  - Prevents accidental overwrites while preserving previous runs
  - Logged with `[RENAME]` prefix for audit trails

### 🚫 No Fallback Reading

- **Strict Canonical Path Enforcement**
  - `stage_unify_daily.py` only reads from canonical locations
  - No alternative/legacy directory searches
  - Missing files logged as: `[WARN] Missing {source} daily_{metric} at canonical path: {path}`
  - Empty DataFrame returned if file not found (no silent failures)

---

## 🧪 Testing & Validation

### Modified Files

1. **Makefile**

   - Added `ZPWD` variable for Zepp password
   - `env` target: fail-fast check for missing password
   - All pipeline targets: pass `--zepp-password` when `ZPWD` set

2. **scripts/run_full_pipeline.py**

   - Stage 0: fail-fast validation before extraction
   - Unified `zpwd` variable (accepts both env var names)
   - Exit code 2 for missing password

3. **src/etl/stage_csv_aggregation.py**

   - Removed `/ participant` from output paths
   - Added automatic `.prev.csv` renaming
   - Clear logging for renamed files

4. **src/etl/stage_unify_daily.py**
   - Removed `/ participant` from input paths
   - No fallback path logic
   - Warning-only for missing canonical files

### Expected Behavior

```bash
# ❌ Without password (should abort with exit 2):
make pipeline PID=P000001 SNAPSHOT=auto
# ERR: Zepp ZIP detected but no password provided (set ZEP_ZIP_PASSWORD or pass ZPWD=...)

# ✅ With password (should work):
make pipeline PID=P000001 SNAPSHOT=auto ZPWD="password123"
# [OK] Pipeline completes with canonical paths
```

---

## 🧠 Next Steps

- Validate pipeline with real Zepp encrypted ZIPs
- Verify no legacy path references remain in codebase
- Test `RUN_REPORT.md` consistency with new paths
- Add integration tests for fail-fast behavior
- Document canonical path structure in README

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---
