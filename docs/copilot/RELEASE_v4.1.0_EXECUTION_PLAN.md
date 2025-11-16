# Release v4.1.0 – Execution Plan & Status

**Date:** 2025-11-07  
**Status:** Ready for PR Generation  
**Branch:** `release/v4.1.0`  
**Target:** Merge to `main`

---

## 📋 Release Metadata

| Field                    | Value                                                 |
| ------------------------ | ----------------------------------------------------- |
| **Version**              | 4.1.0                                                 |
| **Tag**                  | v4.1.0                                                |
| **Title**                | Fase 3 ETL Consolidation & Production-Ready Analytics |
| **Release Date**         | 2025-11-07T00:00:00Z                                  |
| **Commits Since v4.0.4** | 30                                                    |
| **Branch**               | `release/v4.1.0` → `main`                             |

---

## ✅ Completed Setup

### 1. Release Notes ✓

- **File:** `docs/release_notes/release_notes_v4.1.0.md`
- **Status:** Updated with comprehensive highlights, testing results, and academic citation
- **Content:**
  - Fase 3 analytics pipeline completion
  - 4 critical data quality issue resolutions
  - CLI entrypoints documentation
  - Reproducibility & audit trail details

### 2. CHANGELOG ✓

- **File:** `CHANGELOG.md`
- **Status:** Added `[v4.1.0]` section with full summary
- **Sections:**
  - Summary & key achievements
  - Added features
  - Bug fixes
  - Infrastructure changes
  - Issues resolved (with #8, #9, #10, #13)

### 3. Provenance Artifacts ✓

- `provenance/pip_freeze_2025-11-07.txt` — ✅ Generated
- `provenance/etl_provenance_report.csv` — ✅ Exists
- `provenance/data_audit_summary.md` — ✅ Exists

### 4. Git Status ✓

- Working tree committed
- Release artifacts staged and committed: `a64e31e`
- GitHub CLI authenticated: ✅

---

## 🎯 Issues to Close Automatically on PR Merge

When the PR is merged, the following issues will be closed via `Closes #X` annotations:

| Issue | Title                                              | Status      |
| ----- | -------------------------------------------------- | ----------- |
| #8    | Remove Deprecated Cardio ETL step                  | ✅ RESOLVED |
| #9    | Incorrect data_etl participant snapshots directory | ✅ RESOLVED |
| #10   | ETL: cardio outputs written to wrong path          | ✅ RESOLVED |
| #13   | Snapshot date incoherence across sources           | ✅ RESOLVED |

**Notes:**

- Issue #7 (Mandatory Release Publish Title): Partially addressed; RELEASE_TITLE enforced.
- Issue #11 (make clean-all): Pending environment verification.
- Issue #14 (Legacy shim cleanup): Scheduled for v4.2.0.
- Issue #15 (Community suggestion): Awaiting maintainer decision.

---

## 🚀 Next Steps: Automated Release via Makefile

### Command to Generate PR

```bash
make release-pr \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"
```

**Expected behavior:**

1. Creates/switches to branch `release/v4.1.0` (already exists)
2. Commits release artifacts (if any changes)
3. Pushes branch to origin
4. Creates GitHub PR with:
   - Title: `Release v4.1.0 – Fase 3 ETL Consolidation & Production-Ready Analytics`
   - Body: Release notes + `Closes #8`, `Closes #9`, `Closes #10`, `Closes #13`
   - Base: `main`
   - Head: `release/v4.1.0`

---

## 📝 PR Body Template (Auto-Generated)

The `release-pr` target will generate a PR body containing:

```
[Release notes content from docs/release_notes/release_notes_v4.1.0.md]

Closes #8
Closes #9
Closes #10
Closes #13
```

---

## 🔄 Post-Merge Steps (Automated via GitHub Actions or Manual)

After PR is merged to `main`:

```bash
# 1. Tag the release
make release-tag RELEASE_VERSION=4.1.0

# 2. Publish to GitHub Releases
make release-publish \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"
```

---

## 📊 Metrics

| Metric                            | Value                                         |
| --------------------------------- | --------------------------------------------- |
| Release commits (v4.0.4 → v4.1.0) | 30                                            |
| Main contributors                 | Rodrigo Teixeira (27), Rodrigo M Teixeira (3) |
| Issues resolved                   | 4 (#8, #9, #10, #13)                          |
| Domains implemented               | 3 (activity, cardio, sleep)                   |
| Notebooks refactored              | 1 (NB1_EDA_daily)                             |
| ETL code lines                    | ~2,500 across domains modules                 |

---

## 🔐 Verification Checklist

Before running the release:

- [x] `CHANGELOG.md` updated with [v4.1.0] section
- [x] `docs/release_notes/release_notes_v4.1.0.md` complete and accurate
- [x] All provenance files present
- [x] Git working tree clean (artifacts committed)
- [x] GitHub CLI authenticated
- [x] Issues identified and ready to close
- [x] Release title defined and validated

---

## 🆘 Rollback Plan (if needed)

If issues arise:

```bash
# Revert the commit
git reset --soft HEAD~1
git reset docs/release_notes/release_notes_v4.1.0.md CHANGELOG.md

# Delete the remote branch
git push origin --delete release/v4.1.0

# Fix issues locally and retry
```

---

## 📞 Contact & Support

- **Maintainer:** Rodrigo Marques Teixeira
- **Supervisor:** Dr. Agatha Mattos
- **Institution:** National College of Ireland, MSc AI for Business
- **Repository:** [https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

**Status:** ✅ **READY FOR RELEASE PR GENERATION**
