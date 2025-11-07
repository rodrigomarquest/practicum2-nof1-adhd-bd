#!/bin/bash
# Release v4.1.0 – Automated PR Generation Command
# Prepared: 2025-11-07
# Branch: release/v4.1.0
# Status: ✅ READY TO EXECUTE

# ============================================
# STEP 1: Generate Release PR (Automated)
# ============================================
# Execute this from repository root:

make release-pr \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"

# Expected output:
# - PR created at: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/pull/N
# - Title: Release v4.1.0 – Fase 3 ETL Consolidation & Production-Ready Analytics
# - Body includes: Release notes + Closes #8, #9, #10, #13

# ============================================
# STEP 2: Review & Merge PR (Manual)
# ============================================
# 1. Visit: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/pulls
# 2. Find the v4.1.0 PR
# 3. Verify all content is correct
# 4. Merge to main (regular merge or squash)
# 5. Issues #8, #9, #10, #13 auto-close on merge

# ============================================
# STEP 3: Publish Release (Automated)
# ============================================
# After merge, execute:

git switch main
git pull origin main

make release-publish \
  RELEASE_VERSION=4.1.0 \
  RELEASE_TITLE="Fase 3 ETL Consolidation & Production-Ready Analytics" \
  ISSUES="8 9 10 13"

# Expected output:
# - GitHub Release created: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.0
# - Release includes: Tag, title, notes, attachments (changelog, pip freeze, provenance)

# ============================================
# VERIFICATION CHECKLIST
# ============================================
# Before running step 1:
# ✅ Branch is release/v4.1.0 (confirmed)
# ✅ GitHub CLI authenticated (confirmed)
# ✅ CHANGELOG.md updated (confirmed)
# ✅ Release notes prepared (confirmed)
# ✅ Provenance artifacts present (confirmed)
# ✅ Git working tree clean (confirmed)

# ============================================
# KEY FILES PREPARED
# ============================================
# - docs/release_notes/release_notes_v4.1.0.md (comprehensive notes)
# - CHANGELOG.md (v4.1.0 section added)
# - provenance/pip_freeze_2025-11-07.txt (dependencies frozen)
# - provenance/etl_provenance_report.csv (audit trail)
# - provenance/data_audit_summary.md (QC metrics)
# - dist/release_pr_body_4.1.0.md (ready-to-use PR body)

# ============================================
# ISSUES TO CLOSE
# ============================================
# When PR merges, these auto-close:
# - #8: Remove Deprecated Cardio ETL step
# - #9: Incorrect data_etl participant snapshots directory
# - #10: ETL: cardio outputs written to wrong path
# - #13: Snapshot date incoherence across sources

# ============================================
# RELEASE STATISTICS
# ============================================
# - Version: v4.1.0
# - Commits since v4.0.4: 30
# - Issues closed: 4
# - Main contributors: Rodrigo Teixeira (27), Rodrigo M Teixeira (3)
# - Domains implemented: 3 (activity, cardio, sleep)
# - ETL code: ~2,500 lines
# - Key feature: Fase 3 production-ready analytics

# ============================================
# DOCUMENTATION
# ============================================
# See docs/RELEASE_v4.1.0_SUMMARY.md for:
# - Detailed preparation status
# - Issue resolution details
# - Full execution guide
# - Verification checklist

# ============================================
# STATUS: ✅ READY FOR RELEASE
# ============================================
