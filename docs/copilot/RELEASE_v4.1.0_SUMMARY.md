# Release v4.1.0 – Preparation Summary & Execution Guide

**Document Date:** 2025-11-07  
**Status:** ✅ **READY FOR AUTOMATED RELEASE PR GENERATION**

---

## 📊 Release Overview

### v4.1.0: Fase 3 ETL Consolidation & Production-Ready Analytics

| Aspect              | Details                         |
| ------------------- | ------------------------------- |
| **Version**         | 4.1.0                           |
| **Git Tag**         | v4.1.0                          |
| **Base Branch**     | `release/v4.1.0` (pre-existing) |
| **Target Branch**   | `main`                          |
| **Commits**         | 30 since v4.0.4                 |
| **Issues to Close** | #8, #9, #10, #13 (4 total)      |
| **Release Date**    | 2025-11-07                      |
| **Status**          | ✅ All artifacts prepared       |

---

## ✅ Artifacts Prepared

### 1. Release Notes ✓

**File:** `docs/release_notes/release_notes_v4.1.0.md`

Complete academic-style release notes including:

- Executive summary with Fase 3 milestone
- Highlights of 4 resolved issues
- Unified extract orchestration details
- Production-ready analytics pipeline
- Testing & validation results
- Reproducibility guarantees
- Academic citation & license

**Key sections:**

- 🧱 ETL & Data Quality (with issue closes)
- 🧪 Fase 3 Analytics Pipeline
- ⚙️ Infrastructure & Developer Experience
- 📊 Data Snapshot Date Resolution
- 📚 Documentation & Migration
- 🧪 Testing & Validation
- 📈 Statistics (30 commits, 4 issues, 3 domains)

### 2. CHANGELOG ✓

**File:** `CHANGELOG.md`

New section `[v4.1.0] – 2025-11-07` with:

- Summary of Fase 3 consolidation
- Key achievements
- Added features (unified extract, snapshot paths, CLI entrypoints, pipeline, NB migration, provenance, tqdm)
- Fixed issues (cardio PID, activity import, exit codes, shell compatibility, deprecated steps)
- Changed behavior (ETL namespace, line endings, timezone support, platform compatibility)
- Infrastructure improvements
- Documentation updates
- Issues resolved explicitly listed: Closes #8, #9, #10, #13

### 3. Provenance Artifacts ✓

All required files present in `provenance/`:

- ✅ `pip_freeze_2025-11-07.txt` — Generated today via `python -m pip freeze`
- ✅ `etl_provenance_report.csv` — Exists (524 bytes, last updated Nov 1)
- ✅ `data_audit_summary.md` — Exists (593 bytes, last updated Nov 1)

### 4. PR Body ✓

**File:** `dist/release_pr_body_4.1.0.md`

Ready-to-use PR body with:

- Release title and summary
- Key features grouped by category
- Statistics (30 commits, 4 issues, 3 domains)
- Testing & validation results
- Explicit issue closes: `Closes #8`, `Closes #9`, `Closes #10`, `Closes #13`

### 5. Git Status ✓

- Working tree clean
- Release artifacts committed (`a64e31e`)
- Branch: `release/v4.1.0`
- GitHub CLI authenticated with repo access

---

## 🎯 Issues to Close on PR Merge

When this PR is merged to `main`, these 4 issues will be automatically closed:

| #   | Title                                              | Resolution                                                                             |
| --- | -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| #8  | Remove Deprecated Cardio ETL step                  | Standalone cardio orchestrator removed, integrated into full ETL                       |
| #9  | Incorrect data_etl participant snapshots directory | Snapshot path normalized to `data/etl/<PID>/<YYYY-MM-DD>/`                             |
| #10 | ETL: cardio outputs written to wrong path          | PID extraction from snapshot_dir corrected; outputs to `data/etl/<PID>/<SNAP>/joined/` |
| #13 | Snapshot date incoherence across sources           | Manifest-based provenance with snapshot date validation and warnings implemented       |

---

## 🚀 Execution Steps

### Step 1: Generate PR via Automation (Makefile)

Execute this command from the repository root:

```bash
make release-pr \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"
```

**What this does:**

1. Ensures `release/v4.1.0` branch exists (it does)
2. Adds release artifacts to git (already done)
3. Commits with message: `"chore(release): ..."`
4. Pushes to `origin/release/v4.1.0`
5. Creates GitHub PR via `gh CLI` with:
   - **Title:** `Release v4.1.0 – Fase 3 ETL Consolidation & Production-Ready Analytics`
   - **Body:** Release notes content + `Closes #8`, `Closes #9`, `Closes #10`, `Closes #13`
   - **Base:** `main`
   - **Head:** `release/v4.1.0`

### Step 2: Review PR on GitHub

Visit: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/pulls

Verify:

- [ ] PR title includes version and descriptive title
- [ ] PR body contains release highlights
- [ ] PR body includes `Closes #X` for each issue
- [ ] All commits are present (30 from v4.0.4)
- [ ] No conflicts with `main`

### Step 3: Merge PR

After review, merge PR to `main`:

```bash
# Option A: Via GitHub UI (Squash or regular merge)
# Recommended: Regular merge to preserve commit history

# Option B: Via gh CLI
gh pr merge <PR_NUMBER> --merge
```

**Important:** GitHub will automatically close issues #8, #9, #10, #13 when merge happens.

### Step 4: Publish Release (Post-Merge)

After PR is merged:

```bash
# Switch to main and pull latest
git switch main
git pull origin main

# Create annotated tag
git tag -a v4.1.0 -m "Release v4.1.0: Fase 3 ETL Consolidation & Production-Ready Analytics"
git push origin v4.1.0

# Publish to GitHub Releases
make release-publish \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"
```

This creates a GitHub Release with:

- Tag: v4.1.0
- Title: Release v4.1.0 – Fase 3 ETL Consolidation & Production-Ready Analytics
- Body: Full release notes
- Attachments: Changelog, pip freeze snapshot, provenance files

---

## 📋 Pre-Release Validation Checklist

Before executing `make release-pr`, verify:

- [x] Branch is `release/v4.1.0`
- [x] CHANGELOG.md includes [v4.1.0] section
- [x] Release notes file exists and is complete
- [x] All provenance artifacts present
- [x] GitHub CLI authenticated (`gh auth status` shows logged in)
- [x] Git working tree is clean (artifacts committed)
- [x] Release title and version are correct
- [x] Issues are correctly identified (#8, #9, #10, #13)

---

## 🔍 Issue Resolution Details

### Issue #8: Remove Deprecated Cardio ETL step

- **Commit:** `6a7768a`
- **Change:** Removed standalone cardio orchestrator; functionality integrated into full ETL pipeline
- **Status:** ✅ RESOLVED

### Issue #9: Incorrect data_etl participant snapshots directory

- **Commits:** `65ba18d`, `c103bfb`
- **Change:** Canonical path changed from `data/etl/<PID>/snapshots/<DATE>/` to `data/etl/<PID>/<DATE>/`
- **Additional:** Migration helper script provided for legacy layouts
- **Status:** ✅ RESOLVED

### Issue #10: ETL: cardio outputs written to wrong path

- **Commit:** `6a7768a`
- **Change:** Fixed PID extraction from snapshot_dir; outputs now write to correct `data/etl/<PID>/<SNAP>/joined/`
- **Root cause:** Used incorrect array indexing when computing PID from path parts
- **Status:** ✅ RESOLVED

### Issue #13: Snapshot date incoherence across sources

- **Commit:** `5dc0192`
- **Change:** Unified extract orchestration with snapshot resolution, discovery, QC, manifest logging
- **Additional:** Manifest records source file, size, modification timestamp; validates against --snapshot parameter
- **Status:** ✅ RESOLVED

---

## 📈 Release Statistics

| Metric                    | Value                                         |
| ------------------------- | --------------------------------------------- |
| Commits since v4.0.4      | 30                                            |
| Main contributors         | Rodrigo Teixeira (27), Rodrigo M Teixeira (3) |
| Issues closed             | 4                                             |
| Domains implemented       | 3 (activity, cardio, sleep stubs)             |
| Notebooks refactored      | 1 (NB1_EDA_daily)                             |
| ETL code added            | ~2,500 lines across domain modules            |
| Documentation pages added | 1 (NB1_EDA_MIGRATION.md)                      |

---

## 🔄 Related Issues (Not Closing in v4.1.0)

These issues remain open and will be addressed in future releases:

| #   | Title                                     | Scheduled | Action                                              |
| --- | ----------------------------------------- | --------- | --------------------------------------------------- |
| #7  | Mandatory Release Publish Title           | v4.1.0    | ⚠️ Partially addressed; RELEASE_TITLE enforced      |
| #11 | make clean-all does not remove data/etl   | TBD       | ⚠️ Pending Windows path handling verification       |
| #14 | Cleanup legacy shims in etl_modules       | v4.2.0    | 📋 Backward compatibility maintained for now        |
| #15 | Nice initiative! Maybe Sklong could be... | TBD       | 📋 Community suggestion; awaiting maintainer review |

---

## 🛠️ Makefile Release Targets Reference

For future reference, the available Makefile release targets are:

```makefile
# Verify preconditions
make release-verify RELEASE_VERSION=4.1.0

# Generate release summary
make release-summary

# Create release notes
make release-draft RELEASE_VERSION=4.1.0

# Freeze dependencies
make release-freeze RELEASE_VERSION=4.1.0

# Create local tag
make release-tag RELEASE_VERSION=4.1.0

# Create PR (automated)
make release-pr RELEASE_VERSION=4.1.0 RELEASE_TITLE="..." ISSUES="8 9 10 13"

# Publish to GitHub Releases
make release-publish RELEASE_VERSION=4.1.0 RELEASE_TITLE="..." ISSUES="8 9 10 13"

# Full local workflow (without push/publish)
make release-final RELEASE_VERSION=4.1.0
```

---

## ✨ Key Achievements of v4.1.0

1. **Production-Ready Fase 3 Pipeline** — Full ETL→Analytics workflow with atomic writes and QC validation
2. **Data Quality Resolution** — 4 critical issues fixed, manifest-based provenance implemented
3. **Developer Experience** — Unified CLI, cross-platform Makefile, optional progress bars
4. **Reproducibility** — End-to-end audit trail from raw exports to processed features
5. **Documentation** — Academic-style release notes, migration guides, citation ready

---

## 📞 Support Information

- **Repository:** [https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)
- **Maintainer:** Rodrigo Marques Teixeira
- **Supervisor:** Dr. Agatha Mattos
- **Institution:** National College of Ireland, MSc AI for Business
- **License:** CC BY-NC-SA 4.0
- **Student ID:** 24130664

---

## 🚀 Ready to Release!

All preparation is complete. Execute:

```bash
make release-pr \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"
```

Then follow the GitHub PR review and merge workflow as outlined in **Step 2–4** above.

**Status:** ✅ **READY FOR RELEASE PR GENERATION**
