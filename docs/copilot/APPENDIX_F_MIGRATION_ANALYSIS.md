# Appendix F Migration Analysis (ANALYSIS ONLY)

**Date:** 2026-01-23  
**Scope:** Configuration Manual → Appendix F merger  
**Status:** 🔍 Analysis Complete — NO EDITS PERFORMED YET

---

## 1. Located Paths (Confirmed)

| File       | Path                                              | Lines | Status                 |
| ---------- | ------------------------------------------------- | ----- | ---------------------- |
| **Source** | `docs/latex/latest/configuration_manual_full.tex` | 309   | ✅ Exists (standalone) |
| **Target** | `docs/latex/latest/appendix_f.tex`                | 238   | ✅ Exists (appendix)   |

Both files confirmed present and readable.

---

## 2. Structural Comparison

### 2.1 Section Inventory

#### Configuration Manual Full (`configuration_manual_full.tex`)

**Preamble (Lines 1–48):**

- Standalone document class (`\documentclass[12pt,a4paper]{article}`)
- Full package imports (inputenc, fontenc, geometry, hyperref, etc.)
- Title, author, date metadata
- `\maketitle`, `\tableofcontents`, `\newpage`

**Content Sections (Lines 51–309):**

1. **Section 1: Introduction** (Line 51)

   - Brief overview of manual purpose
   - References "A Deterministic N-of-1 Pipeline..." paper title

2. **Section 2: System Architecture** (Line 57)

   - Components description
   - **Figure 1:** `system-architecture-paper-v3.png` (Line 62)

3. **Section 3: Environment Setup** (Line 70)

   - Subsection 3.1: Requirements
   - Python 3.11+ dependency reference

4. **Section 4: Data Management and Preprocessing** (Line 77)

   - Detailed prose on data sources (Apple Health, Amazfit GTR2/GTR4, SoM)
   - **Figure 2:** `etl-pipeline-paper.png` (Line 86)

5. **Section 5: Ethics and Governance** (Line 95)

   - Privacy, autonomy, data governance discussion
   - REC approval context (self-tracking exception)
   - Future multi-participant extension requirements

6. **Section 7: Reproducibility and Version Control** (Line 109)

   - **Subsection 7.1:** Repository and Code Access
   - **Subsection 7.2:** Data Availability and Privacy
   - **Subsection 7.3:** Snapshot Parameters
   - **Subsection 7.4:** Directory Structure
   - **Subsection 7.5:** Canonical Entry Points (F.5.1, F.5.2, F.5.3)
   - **Subsection 7.6:** Determinism and Reproducibility
   - **Subsection 7.7:** Automated QC
   - **Subsection 7.8:** Minimal Reproduction Guide (8.1, 8.2)

7. **Section 8: Future Work and Extensions** (Line 297)
   - EMA instruments, HRV metrics, cluster-based segmentation
   - Multi-participant generalizability

**End:** `\end{document}` (Line 309)

---

#### Appendix F (`appendix_f.tex`)

**Header (Lines 1–6):**

- `\section*{Appendix F --- Configuration Manual (Pipeline v4.3.1)}`
- `\addcontentsline{toc}{section}{...}`
- Introductory paragraph

**Content Sections (Lines 8–237):**

- **F.1:** Repository and Code Access
- **F.2:** Data Availability and Privacy
- **F.3:** Snapshot Parameters
- **F.4:** Directory Structure
- **F.5:** Canonical Entry Points (F.5.1, F.5.2, F.5.3)
- **F.6:** Determinism and Reproducibility
- **F.7:** Automated QC (Optional but Recommended)
- **F.8:** Minimal Reproduction Guide (F.8.1, F.8.2)

**End:** No `\end{document}` (appendix fragment)

---

### 2.2 Content Mapping

| Configuration Manual Section | Appendix F Section | Status               |
| ---------------------------- | ------------------ | -------------------- |
| **1. Introduction**          | _(missing)_        | ❌ Not in Appendix F |
| **2. System Architecture**   | _(missing)_        | ❌ Not in Appendix F |
| **3. Environment Setup**     | _(missing)_        | ❌ Not in Appendix F |
| **4. Data Management**       | _(missing)_        | ❌ Not in Appendix F |
| **5. Ethics and Governance** | _(missing)_        | ❌ Not in Appendix F |
| **7.1 Repository Access**    | **F.1**            | ✅ Equivalent        |
| **7.2 Data Availability**    | **F.2**            | ✅ Equivalent        |
| **7.3 Snapshot Parameters**  | **F.3**            | ✅ Equivalent        |
| **7.4 Directory Structure**  | **F.4**            | ✅ Equivalent        |
| **7.5 Entry Points**         | **F.5**            | ✅ Equivalent        |
| **7.6 Determinism**          | **F.6**            | ✅ Equivalent        |
| **7.7 QC**                   | **F.7**            | ✅ Equivalent        |
| **7.8 Reproduction Guide**   | **F.8**            | ✅ Equivalent        |
| **8. Future Work**           | _(missing)_        | ❌ Not in Appendix F |

---

## 3. Verdict: Content Equivalence

### 3.1 Core Reproducibility Content (F.1–F.8)

**Status:** ✅ **EQUIVALENT**

**Evidence:**

- Sections 7.1–7.8 in `configuration_manual_full.tex` are **semantically identical** to F.1–F.8 in `appendix_f.tex`
- Diff analysis shows only formatting differences:
  - `\subsection{1. ...}` → `\subsection*{F.1 ...}` (numbering style)
  - `%` comments → `#` comments in verbatim blocks
  - `2,868` → `2{,}868` (LaTeX number formatting)
  - Line wrapping variations (no semantic change)

**Critical Parameters Match:**

- Participant: `P000001`
- Snapshot: `2025-12-09`
- Seed: `42`
- Pipeline: `v4.3.1`
- CLI commands: **Identical**
- File paths: **Identical**

### 3.2 Missing Content Analysis

**Missing from Appendix F (5 sections):**

1. **Introduction** (Section 1)

   - ~3 lines
   - Generic manual purpose statement
   - Paper title reference

2. **System Architecture** (Section 2)

   - ~15 lines prose + 1 figure
   - Components: Apple Health, Amazfit GTR2/GTR4, SoM
   - **Figure:** `system-architecture-paper-v3.png`

3. **Environment Setup** (Section 3)

   - ~3 lines
   - Python 3.11+ requirement
   - Reference to requirements.txt

4. **Data Management and Preprocessing** (Section 4)

   - ~18 lines prose + 1 figure
   - Data sources narrative
   - ETL pipeline overview
   - **Figure:** `etl-pipeline-paper.png`

5. **Ethics and Governance** (Section 5)

   - ~17 lines
   - Privacy, autonomy, data governance
   - REC approval discussion
   - Multi-participant extension requirements

6. **Future Work and Extensions** (Section 8)
   - ~11 lines
   - EMA instruments, HRV metrics
   - Cluster segmentation, causal models
   - Multi-participant generalizability

**Total Missing:** ~67 lines + 2 figures

---

### 3.3 Missing from Configuration Manual

**Status:** ✅ None

All content in Appendix F is present (and equivalent) in Configuration Manual Section 7.

---

## 4. Detailed Diff Summary

### 4.1 Formatting Differences (Not Semantic)

| Configuration Manual            | Appendix F                        | Impact                    |
| ------------------------------- | --------------------------------- | ------------------------- |
| `\subsection{1. Repository...}` | `\subsection*{F.1 Repository...}` | Numbering only            |
| `% Daily CSVs`                  | `# Daily CSVs`                    | Comment style in verbatim |
| `2,868`                         | `2{,}868`                         | LaTeX comma formatting    |
| Line breaks                     | Wrapped differently               | Whitespace only           |

**Verdict:** No semantic differences — content identical.

### 4.2 Content Conflicts

**None detected.**

All CLI commands, paths, snapshot IDs, and configuration values match exactly.

---

## 5. Merge Plan (Step-by-Step)

### 5.1 Strategy

**Goal:** Migrate all unique content from `configuration_manual_full.tex` into `appendix_f.tex` while preserving F.1–F.8 numbering.

**Approach:** Insert missing sections **before F.1** as introductory material (F.0.x subsections).

### 5.2 Proposed Structure

```latex
\section*{Appendix F --- Configuration Manual (Pipeline v4.3.1)}
\addcontentsline{toc}{section}{Appendix F --- Configuration Manual (Pipeline v4.3.1)}

[Existing intro paragraph]

% ======================================================
% NEW: Introductory Material from configuration_manual_full.tex
% ======================================================

\subsection*{F.0.1 Overview}
[Content from Section 1: Introduction]

\subsection*{F.0.2 System Architecture}
[Content from Section 2: System Architecture]
- Figure: system-architecture-paper-v3.png

\subsection*{F.0.3 Environment Requirements}
[Content from Section 3: Environment Setup]

\subsection*{F.0.4 Data Sources and ETL Overview}
[Content from Section 4: Data Management and Preprocessing]
- Figure: etl-pipeline-paper.png

\subsection*{F.0.5 Ethics and Governance}
[Content from Section 5: Ethics and Governance]

% ======================================================
% EXISTING: F.1–F.8 (unchanged)
% ======================================================

\subsection*{F.1 Repository and Code Access}
[... existing content ...]

\subsection*{F.2 Data Availability and Privacy}
[... existing content ...]

[... F.3–F.7 unchanged ...]

\subsection*{F.8 Minimal Reproduction Guide}
[... existing content ...]

% ======================================================
% NEW: Future Work (optional — see Risk Assessment)
% ======================================================

\subsection*{F.9 Future Work and Extensions}
[Content from Section 8: Future Work]
```

### 5.3 Insertion Points (Exact Line Numbers)

**In `appendix_f.tex`:**

| New Section                   | Insert After Line    | Content Source (config_manual) |
| ----------------------------- | -------------------- | ------------------------------ |
| **F.0.1 Overview**            | Line 6 (after intro) | Lines 51–53                    |
| **F.0.2 System Architecture** | After F.0.1          | Lines 57–67 + Figure (Line 62) |
| **F.0.3 Environment**         | After F.0.2          | Lines 70–73                    |
| **F.0.4 Data Sources**        | After F.0.3          | Lines 77–91 + Figure (Line 86) |
| **F.0.5 Ethics**              | After F.0.4          | Lines 95–106                   |
| **F.9 Future Work**           | After F.8 (Line 237) | Lines 297–309                  |

### 5.4 Figure Path Adjustments

**Required Changes:**

1. **system-architecture-paper-v3.png**

   - Original path: `docs/figures/system-architecture-paper-v3.png`
   - No change needed (path is correct for Overleaf)

2. **etl-pipeline-paper.png**
   - Original path: `docs/figures/etl-pipeline-paper.png`
   - No change needed (path is correct for Overleaf)

**Action:** Verify both figures exist in Overleaf at specified paths.

---

## 6. Risk & Size Assessment

### 6.1 Length Impact

**Current Appendix F:** 238 lines  
**Missing Content:** ~67 lines + 2 figures (~10 lines)  
**Future Work:** ~11 lines  
**Total after merge:** ~316 lines

**Comparison:**

- Configuration Manual (full): 309 lines
- Appendix F (merged): ~316 lines (includes appendix header)

**Page Count Estimate:**

- Current Appendix F: ~5 pages
- Merged Appendix F: ~7–8 pages

### 6.2 Research Paper Length Constraints

**Typical NCI MSc Thesis Constraints:**

- Main body: 10,000–15,000 words
- Appendices: No strict limit (but should be reasonable)

**Assessment:**

- Adding 2 pages to Appendix F is **acceptable**
- Appendices are supplementary — reviewers expect detailed configuration
- Alternative (if space critical): Move F.9 (Future Work) to main paper Section 6 (Discussion)

### 6.3 Compression Strategy (If Needed)

**Option 1: Minimal Compression (Recommended)**

- Keep all F.0.x sections as-is
- No content loss
- Preserve scientific fidelity

**Option 2: Aggressive Compression (Only if required)**

- **F.0.1 Overview:** Reduce to 1 sentence (merge into intro)
- **F.0.3 Environment:** Merge into F.1 (Repository section)
- **F.0.5 Ethics:** Move to main paper Section 6 (Discussion)
- **F.9 Future Work:** Move to main paper Section 6 (Future Research)

**Recommended:** Option 1 (no compression) — thesis format allows detailed appendices.

---

## 7. Conflicts & Risks

### 7.1 Identified Conflicts

**None.**

All paths, commands, snapshot IDs, and parameters are consistent between both files.

### 7.2 Potential Risks

| Risk                             | Severity  | Mitigation                                     |
| -------------------------------- | --------- | ---------------------------------------------- |
| **Figures missing in Overleaf**  | 🟡 Medium | Verify `docs/figures/*.png` exist before merge |
| **LaTeX compilation errors**     | 🟡 Medium | Test compile after each section insertion      |
| **Appendix too long (8+ pages)** | 🟢 Low    | Acceptable for MSc thesis format               |
| **Numbering confusion (F.0.x)**  | 🟢 Low    | F.0.x clearly indicates "introductory"         |
| **Duplicate content**            | 🟢 Low    | No overlap detected (sections are disjoint)    |

### 7.3 Resolution Plan

**Before Merge:**

1. ✅ Verify figure files exist: `system-architecture-paper-v3.png`, `etl-pipeline-paper.png`
2. ✅ Confirm Overleaf project has `appendix_f.tex` write access
3. ✅ Create backup: `cp appendix_f.tex appendix_f_backup.tex`

**During Merge:**

1. Insert F.0.1–F.0.5 sections (test compile after each)
2. Verify figures render correctly
3. Check cross-references (e.g., "Section 7" → "Section F.5")
4. Add F.9 (Future Work) at end
5. Full compile + PDF review

**After Merge:**

1. Update `main.tex` to remove `configuration_manual_full.tex` reference (if present)
2. Archive `configuration_manual_full.tex` (move to `docs/latex/archive/`)
3. Update README to document canonical source: `appendix_f.tex`

---

## 8. Final Recommendations

### 8.1 Merge Execution Plan

**Phase 1: Preparation (5 minutes)**

- Verify figures exist in Overleaf
- Backup `appendix_f.tex`

**Phase 2: Content Insertion (20 minutes)**

- Add F.0.1 (Overview) after intro paragraph
- Add F.0.2 (System Architecture) with figure
- Add F.0.3 (Environment)
- Add F.0.4 (Data Sources) with figure
- Add F.0.5 (Ethics)
- Test compile (should succeed)

**Phase 3: Future Work (5 minutes)**

- Add F.9 (Future Work) after F.8.2
- Test compile

**Phase 4: Validation (10 minutes)**

- Full PDF review
- Check figure rendering
- Verify cross-references
- Compare against configuration_manual_full.tex (ensure zero content loss)

**Phase 5: Cleanup (5 minutes)**

- Archive `configuration_manual_full.tex`
- Update main paper references (if needed)
- Commit changes with clear message

**Total Time:** ~45 minutes

### 8.2 Content Fidelity Assurance

**Zero-Loss Guarantee:**

- All 309 lines from `configuration_manual_full.tex` will be present in merged `appendix_f.tex`
- All CLI commands preserved verbatim
- All file paths unchanged
- All snapshot parameters identical
- Both figures included

**Verification Checklist:**

- [ ] Introduction content → F.0.1
- [ ] System Architecture + Figure 1 → F.0.2
- [ ] Environment Setup → F.0.3
- [ ] Data Management + Figure 2 → F.0.4
- [ ] Ethics & Governance → F.0.5
- [ ] F.1–F.8 unchanged
- [ ] Future Work → F.9
- [ ] All CLI commands exact match
- [ ] All paths exact match
- [ ] Figures render correctly

### 8.3 Post-Merge Actions

**Required:**

1. ✅ Remove or deprecate `configuration_manual_full.tex`
2. ✅ Update `main.tex` (remove references to standalone config manual)
3. ✅ Update README: "Appendix F is the canonical configuration source"

**Recommended:**

1. ✅ Archive `configuration_manual_full.tex` to `docs/latex/archive/`
2. ✅ Add comment in archived file: "Merged into appendix_f.tex on 2026-01-23"
3. ✅ Update CHANGELOG.md: "Consolidated configuration manual into Appendix F"

---

## 9. Summary

### 9.1 Structural Verdict

**Configuration Manual vs Appendix F:**

- **Sections 7.1–7.8** (Config Manual) ≡ **F.1–F.8** (Appendix F) → ✅ **EQUIVALENT**
- **Sections 1–5, 8** (Config Manual) → ❌ **MISSING** from Appendix F
- **Total Missing:** 6 sections (~77 lines + 2 figures)

### 9.2 Merge Plan Verdict

**Recommendation:** ✅ **PROCEED WITH MERGE**

**Proposed Structure:**

```
Appendix F
├── Intro (existing)
├── F.0.1 Overview (new)
├── F.0.2 System Architecture + Figure (new)
├── F.0.3 Environment (new)
├── F.0.4 Data Sources + Figure (new)
├── F.0.5 Ethics & Governance (new)
├── F.1–F.8 (existing, unchanged)
└── F.9 Future Work (new)
```

**Outcome:**

- ✅ Zero content loss
- ✅ All CLI commands preserved
- ✅ All paths/parameters unchanged
- ✅ F.1–F.8 numbering preserved
- ✅ Figures included
- ✅ Reasonable length (~8 pages)

### 9.3 Risk Assessment

**Overall Risk:** 🟢 **LOW**

- No content conflicts detected
- No parameter mismatches
- Merge is purely additive (no deletions or rewrites)
- Appendix format allows detailed content

### 9.4 Next Steps

**Immediate:**

1. Get user approval to proceed with merge
2. Execute Phase 1 (Preparation)
3. Begin Phase 2 (Content Insertion)

**Post-Merge:**

1. Archive standalone configuration manual
2. Update main paper references
3. Document changes in CHANGELOG.md

---

**Analysis Completed:** 2026-01-23  
**Analyst:** GitHub Copilot (gpt-4)  
**Status:** ✅ Ready for execution upon approval  
**Confidence:** HIGH (structural analysis complete, no conflicts detected)
