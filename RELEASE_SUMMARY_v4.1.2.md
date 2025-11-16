# 📦 Release v4.1.2 - Deployment Summary

**Release Date:** 2025-11-07  
**Tag:** v4.1.2  
**Commit:** 067f7d3 (pushed to main)  
**GitHub Release:** https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.2

---

## ✅ Deployment Checklist

- [x] Code changes committed
- [x] CHANGELOG.md updated
- [x] Release notes created (docs/release_notes/release_notes_v4.1.2.md)
- [x] Provenance files generated (dist/assets/v4.1.2/)
- [x] Git tag v4.1.2 created and pushed
- [x] GitHub Release published with assets
- [x] Main branch pushed to origin

---

## 📊 Release Metrics

- **Files Modified:** 4 core files

  - Makefile
  - scripts/run_full_pipeline.py
  - src/etl/stage_csv_aggregation.py
  - src/etl/stage_unify_daily.py

- **Lines Changed:** ~85 lines
- **Files Added:** 2

  - docs/release_notes/release_notes_v4.1.2.md
  - RUN_REPORT.md

- **Breaking Changes:** 2
  - Legacy paths with intermediate PID no longer supported
  - Missing Zepp password causes exit 2 (not silent failure)

---

## 🎯 Key Features

### 1. Fail-Fast Password Validation

- **Exit Code 2** when Zepp ZIP detected without password
- Checks both `--zepp-password` and `ZEP_ZIP_PASSWORD` / `ZEPP_ZIP_PASSWORD`
- Validation at 2 levels: Makefile `env` target + Stage 0

### 2. Canonical Path Normalization

**Before:**

```
data/etl/P000001/2025-11-07/extracted/apple/P000001/daily_sleep.csv
                                           ^^^^^^^^ (unwanted)
```

**After:**

```
data/etl/P000001/2025-11-07/extracted/apple/daily_sleep.csv
                                           (clean!)
```

### 3. File Management

- Automatic `.prev.csv` renaming for existing files
- No fallback path reading (strict enforcement)
- Clear warning logs for missing canonical files

---

## 🔗 Assets Published

**GitHub Release Assets:**

1. `provenance.csv` - Version metadata and commit hash
2. `manifest.json` - Detailed change manifest
3. `release_notes_v4.1.2.md` - Full release documentation

**Access:** https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.2

---

## 📝 Migration Guide

### For Existing Installations

1. **Update from v4.1.1 to v4.1.2:**

   ```bash
   git pull origin main
   git checkout v4.1.2
   ```

2. **Set Zepp Password (if using encrypted ZIPs):**

   ```bash
   export ZEP_ZIP_PASSWORD="your_password"
   # OR
   make pipeline PID=P000001 SNAPSHOT=auto ZPWD="your_password"
   ```

3. **Verify Canonical Paths:**
   - Old data at `extracted/apple/P000001/` will NOT be read
   - Run pipeline once to generate new canonical paths
   - Previous `.csv` files auto-preserved as `.prev.csv`

### For New Installations

1. **Clone repository:**

   ```bash
   git clone https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd.git
   cd practicum2-nof1-adhd-bd
   git checkout v4.1.2
   ```

2. **Install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements/base.txt
   ```

3. **Run pipeline:**
   ```bash
   make pipeline PID=P000001 SNAPSHOT=auto ZPWD="password_if_needed"
   ```

---

## 🧪 Testing Performed

### Manual Testing

- ✅ Pipeline with Zepp password (successful)
- ✅ Pipeline without Zepp password (exit 2 as expected)
- ✅ Canonical paths verified (no intermediate PID)
- ✅ `.prev.csv` renaming confirmed
- ✅ RUN_REPORT.md consistency maintained

### Edge Cases Validated

- ✅ Missing password with Zepp ZIP present
- ✅ Existing `.csv` files (renamed to `.prev.csv`)
- ✅ Missing canonical files (warning logged, empty DataFrame)
- ✅ Both env var names (`ZEP_ZIP_PASSWORD` and `ZEPP_ZIP_PASSWORD`)

---

## 🔄 Next Steps

### Immediate (v4.1.3 candidates)

- [ ] Add integration tests for fail-fast behavior
- [ ] Verify no legacy path references in other modules
- [ ] Add canonical path structure to README.md
- [ ] Test with real encrypted Zepp ZIPs

### Future Enhancements

- [ ] Add pre-commit hook for path validation
- [ ] CI/CD pipeline for automated testing
- [ ] Performance benchmarks for Stage 0-9
- [ ] Multi-participant batch processing

---

## 📞 Support

**Issues:** https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/issues  
**Discussions:** https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/discussions  
**Maintainer:** Rodrigo Marques Teixeira (24130664@student.ncirl.ie)

---

## 📚 Documentation

- **Release Notes:** docs/release_notes/release_notes_v4.1.2.md
- **Changelog:** CHANGELOG.md (v4.1.2 entry)
- **Provenance:** dist/assets/v4.1.2/
- **GitHub Release:** https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.2

---

**🎉 Release v4.1.2 Successfully Deployed!**

_Generated: 2025-11-07 16:50:00 UTC_
