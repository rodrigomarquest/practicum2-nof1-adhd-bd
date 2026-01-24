# NCI Template Migration Analysis

**Status:** ANALYSIS ONLY — NO FILES MODIFIED  
**Date:** 2025-01-23  
**Objective:** Analyze the structural requirements for migrating from the consolidated `main.tex` (latest/) to the NCI multi-file template (templ/)

---

## Executive Summary

This document provides a complete structural analysis of the NCI template migration **WITHOUT executing any changes**. The goal is to map the current single-file paper structure (`docs/latex/latest/main.tex`, 1,874 lines) to the NCI multi-file template pattern (`docs/latex/templ/`), ensuring zero content loss and maintaining compilation integrity.

### Key Findings

1. **Template Structure:** NCI template uses `researchProject.tex` as main entry point with `\input{text/*.tex}` for modular sections
2. **Current Structure:** Single `main.tex` contains all content (~1,874 lines) + 6 appendix files
3. **Section Mapping:** 8 main sections + abstract + declaration map to 9 template files in `text/`
4. **Missing Template File:** Template has `text/appendix.tex` in `\includeonly` but file doesn't exist
5. **Figure Migration:** 15 PNG figures (+ 14 PDF versions) need copying from `latest/figures/` to `templ/figures/`
6. **Bibliography:** Both use `refs.bib`, minimal changes needed

---

## 1. Template Structure Discovery

### 1.1 NCI Template Files (docs/latex/templ/)

**Main Entry Point:**

- `researchProject.tex` (141 lines) — document root, preamble, package inclusions

**Section Files (text/ subdirectory):**

```
text/abstract.tex         — abstract environment
text/introduction.tex     — Section 1: Introduction
text/relatedwork.tex      — Section 2: Literature Review
text/methodology.tex      — Section 3: Methodology
text/design.tex           — Section 4: Design (or Clinical Background)
text/implementation.tex   — Section 5: Implementation (or Results Part 1)
text/evaluation.tex       — Section 6: Evaluation (or Results Part 2)
text/conclusion.tex       — Section 7: Conclusion
text/declaration.tex      — Declaration of authorship
```

**Missing File:**

- `text/appendix.tex` is listed in `\includeonly` on line 100 but **does not exist** in `templ/text/`

**Inclusion Pattern (researchProject.tex, lines 107-119):**

```latex
\include{titlepage}
\include{text/declaration}

\title{\mytitle}%
\author{\myname \\ \studentID}%
\date{}
\maketitle

\input{text/abstract}
\input{text/introduction}
\input{text/relatedwork}
\input{text/methodology}
\input{text/design}
\input{text/implementation}
\input{text/evaluation}
\input{text/conclusion}

\bibliographystyle{dcu}
\bibliography{refs}
```

### 1.2 Current Paper Structure (docs/latex/latest/)

**Main File:**

- `main.tex` (1,874 lines) — all content in single file

**Appendix Files:**

```
appendix_a.tex  — Project Log and Dataset Versioning
appendix_b.tex  — Drift Detection Results
appendix_c.tex  — Sensor Mapping
appendix_d.tex  — Feature Glossary
appendix_e.tex  — Use of AI Tools
appendix_f.tex  — Configuration Manual (285 lines, recently merged)
```

**Section Structure (from grep analysis):**

```
Line   70: \section{Introduction}
Line  110: \section{Literature Review}
Line  167: \section{Methodology}
Line 1012: \section{Clinical Background of the Participant}
Line 1057: \section{Results}
Line 1289: \section{Extended Modelling Experiments}
Line 1378: \section{Discussion}
Line 1657: \section{Conclusion}
Line 1708: \section*{Code Availability}
Line 1722: \section*{Acknowledgments}
Line 1726: \appendix
Line 1729: \input{appendix_a.tex}
Line 1730: \input{appendix_b.tex}
Line 1731: \input{appendix_c.tex}
Line 1732: \input{appendix_d.tex}
Line 1733: \input{appendix_e.tex}
Line 1734: \input{appendix_f.tex}
```

---

## 2. Section-to-Template Mapping

| Current Section (main.tex)                       | Line Range | Template File                                     | Action Required                  |
| ------------------------------------------------ | ---------- | ------------------------------------------------- | -------------------------------- |
| **Preamble** (documentclass, packages, metadata) | 1-69       | `researchProject.tex`                             | Update metadata only             |
| **Abstract**                                     | 30-68      | `text/abstract.tex`                               | Copy abstract content            |
| **Declaration**                                  | N/A        | `text/declaration.tex`                            | Already exists (template only)   |
| **Introduction**                                 | 70-109     | `text/introduction.tex`                           | Copy section content             |
| **Literature Review**                            | 110-166    | `text/relatedwork.tex`                            | Copy section content             |
| **Methodology**                                  | 167-1011   | `text/methodology.tex`                            | Copy section content (845 lines) |
| **Clinical Background of the Participant**       | 1012-1056  | `text/design.tex`                                 | Copy section content (rename)    |
| **Results** (sections 5 + 6)                     | 1057-1377  | `text/implementation.tex` + `text/evaluation.tex` | Split into two files             |
| **Discussion**                                   | 1378-1656  | _(missing template file)_                         | **Needs new file decision**      |
| **Conclusion**                                   | 1657-1707  | `text/conclusion.tex`                             | Copy section content             |
| **Code Availability** (unnumbered)               | 1708-1721  | _(append to conclusion?)_                         | **Needs placement decision**     |
| **Acknowledgments** (unnumbered)                 | 1722-1725  | _(append to conclusion?)_                         | **Needs placement decision**     |
| **Appendix** (A-F, 6 separate files)             | 1726-1734  | `text/appendix.tex` (?)                           | **Missing template file**        |

### 2.1 Mapping Issues Identified

**Issue 1: Discussion Section**

- Current paper has a dedicated `\section{Discussion}` (~279 lines, lines 1378-1656)
- NCI template has no `text/discussion.tex` file
- **Options:**
  - Create new `text/discussion.tex` and add to inclusion chain
  - Merge Discussion into `text/evaluation.tex` (non-standard)
  - Merge Discussion into `text/conclusion.tex` (may be too long)

**Issue 2: Unnumbered Sections (Code Availability, Acknowledgments)**

- `\section*{Code Availability}` and `\section*{Acknowledgments}` are not standard in template
- **Options:**
  - Append both to `text/conclusion.tex` (total ~65 lines)
  - Create custom `text/backmatter.tex` file

**Issue 3: Results Section Split**

- Current paper has single `\section{Results}` (lines 1057-1377, ~321 lines)
- Template expects two files: `text/implementation.tex` + `text/evaluation.tex`
- **Options:**
  - Split Results by subsection boundaries (need subsection analysis)
  - Rename `implementation.tex → results_part1.tex`, `evaluation.tex → results_part2.tex`
  - Keep all Results in `implementation.tex`, leave `evaluation.tex` for Extended Experiments

**Issue 4: Appendix Files**

- Template lists `text/appendix.tex` in `\includeonly` but file doesn't exist
- Current paper uses 6 separate appendix files (`appendix_a.tex` through `appendix_f.tex`)
- **Options:**
  - Create single `text/appendix.tex` that includes all 6 files
  - Keep appendices as separate files, update template to `\input{../latest/appendix_*.tex}`
  - Copy all 6 appendix files to `templ/` directory root

---

## 3. Figure Migration Plan

### 3.1 Current Figures (docs/latex/latest/figures/)

**Total Files:** 31 (15 PNG + 15 PDF + 1 PNG added recently)

**Figure Inventory:**

```
etl-pipeline-paper.png                    (144 KB, added 2025-01-23)
system-architecture-paper-v3.png          (??? KB, added 2025-01-23)
fig01_pbsi_timeline.pdf / .png            (52 KB / 497 KB)
fig02_cardio_distributions.pdf / .png     (35 KB / 238 KB)
fig03_sleep_activity_boxplots.pdf / .png  (29 KB / 189 KB)
fig04_segmentwise_normalization.pdf / .png (40 KB / 427 KB)
fig05_missing_data_pattern.pdf / .png     (57 KB / 152 KB)
fig06_label_distribution_timeline.pdf / .png (26 KB / 396 KB)
fig07_correlation_heatmap.pdf / .png      (28 KB / 382 KB)
fig08_temporal_drift.pdf / .png           (30 KB / 277 KB)
... (truncated for brevity)
```

### 3.2 Template Figures (docs/latex/templ/figures/)

**Current Content:**

- `google.png` (placeholder/example only)

**Required Action:**

- Copy **all 29 figures** (15 PNG + 14 PDF, excluding google.png) from `latest/figures/` to `templ/figures/`
- Verify `\graphicspath{{./figures/}}` in `researchProject.tex` (line 33) — **already correct**

### 3.3 Figure Path Validation

**Current main.tex:**

- Uses `\includegraphics{...}` commands (grep found 0 matches — likely uses different command or environment)
- Need to search for `\begin{figure}` and `\caption{}` to identify all figure references

**Template researchProject.tex:**

- Line 32: `\DeclareGraphicsExtensions{.pdf,.png,.jpg}`
- Line 33: `\graphicspath{{./figures/}}`
- **No changes needed** if figures are copied to `templ/figures/`

---

## 4. Bibliography and References

### 4.1 Current Setup (main.tex)

**Not explicitly shown in lines 1-150**, but appendix section shows:

```latex
% (Expected at end of main.tex)
\bibliographystyle{...}
\bibliography{refs}
```

### 4.2 Template Setup (researchProject.tex, lines 133-135)

```latex
\bibliographystyle{dcu}
\bibliography{refs}
```

**File Location:**

- Current: `docs/latex/latest/refs.bib`
- Template: `docs/latex/templ/refs.bib`

**Required Action:**

- **Option 1:** Copy `latest/refs.bib` to `templ/refs.bib` (recommended)
- **Option 2:** Update `researchProject.tex` to `\bibliography{../latest/refs}` (fragile)

---

## 5. Package and Preamble Comparison

### 5.1 Template Packages (researchProject.tex, lines 19-31)

```latex
\usepackage[british,UKenglish]{babel}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage[export]{adjustbox}
\usepackage[absolute,overlay]{textpos}
\usepackage{tikz}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{lastpage}
\usepackage[bookmarks=true, colorlinks=false]{hyperref}
```

### 5.2 Current Paper Packages (main.tex, lines 14-20)

```latex
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{cite}
```

### 5.3 Package Compatibility Analysis

| Package      | Template | Current | Status            | Action                                           |
| ------------ | -------- | ------- | ----------------- | ------------------------------------------------ |
| `geometry`   | ✅       | ✅      | ✅ Match          | No change                                        |
| `graphicx`   | ✅       | ✅      | ✅ Match          | No change                                        |
| `amsmath`    | ✅       | ✅      | ✅ Match          | No change                                        |
| `hyperref`   | ✅       | ✅      | ⚠️ Options differ | Template uses `bookmarks=true, colorlinks=false` |
| `natbib`     | ✅       | ❌      | ⚠️ Missing        | Template uses natbib, current uses `cite`        |
| `booktabs`   | ❌       | ✅      | ⚠️ Extra          | Add to template preamble                         |
| `float`      | ❌       | ✅      | ⚠️ Extra          | Add to template preamble                         |
| `babel`      | ✅       | ❌      | ℹ️ Optional       | Language support                                 |
| `adjustbox`  | ✅       | ❌      | ℹ️ Optional       | Template-specific                                |
| `textpos`    | ✅       | ❌      | ℹ️ Optional       | Template-specific (titlepage)                    |
| `tikz`       | ✅       | ❌      | ℹ️ Optional       | Template-specific                                |
| `algorithm*` | ✅       | ❌      | ℹ️ Optional       | Template-specific                                |
| `lastpage`   | ✅       | ❌      | ℹ️ Optional       | Template-specific                                |

**Required Preamble Updates:**

1. Add `\usepackage{booktabs}` to template (for table formatting)
2. Add `\usepackage{float}` to template (for `[H]` placement)
3. **Decision:** Keep `natbib` (template) or switch to `cite` (current)?
   - Template uses `\bibliographystyle{dcu}` with `natbib`
   - Current uses `\cite{}` commands (compatible with both)
   - **Recommendation:** Keep `natbib` and `dcu` style (NCI standard)

---

## 6. Content Splitting Strategy

### 6.1 Methodology Section (845 lines — Largest Section)

**Current Structure (lines 167-1011):**

```
\section{Methodology}
  \subsection{Research Design}
  \subsection{Data Sources}
  \subsection{... (multiple subsections)}
```

**Template Expectation:**

- Single file: `text/methodology.tex`
- **No splitting required** — copy entire section as-is

### 6.2 Results Section (321 lines — Requires Split Decision)

**Current Structure (lines 1057-1377):**

```
\section{Results}
  \subsection{Baseline Models (ML6)}
  \subsection{Sequence Models (ML7)}
  \subsection{SHAP Explanations}
  \subsection{Drift Analysis}
  ... (additional subsections)
```

**Template Files:**

- `text/implementation.tex` (expected: implementation details)
- `text/evaluation.tex` (expected: results and evaluation)

**Recommended Split:**

- **Option A (By Content Type):**
  - `implementation.tex` ← ML6/ML7 model descriptions
  - `evaluation.tex` ← Results, SHAP, Drift
- **Option B (By Model Type):**
  - `implementation.tex` ← Baseline Models (ML6)
  - `evaluation.tex` ← Sequence Models (ML7) + SHAP + Drift
- **Option C (Sequential):**
  - `implementation.tex` ← First half of Results
  - `evaluation.tex` ← Second half of Results

**Recommendation:** Analyze subsection boundaries before choosing split point

### 6.3 Extended Modelling Section (89 lines)

**Current:** `\section{Extended Modelling Experiments}` (lines 1289-1377)

**Template Mapping:**

- No dedicated template file
- **Options:**
  - Append to `text/evaluation.tex`
  - Create custom `text/extended.tex`
  - Move to appendix

---

## 7. Migration Risk Assessment

### 7.1 High-Risk Areas

| Risk                    | Description                              | Mitigation                                  |
| ----------------------- | ---------------------------------------- | ------------------------------------------- |
| **Content Loss**        | Sections don't map 1:1 to template files | Pre-migration content audit                 |
| **Figure Paths**        | Broken `\includegraphics` references     | Validate all figure commands before compile |
| **Bibliography**        | `natbib` vs `cite` incompatibility       | Test compilation with `dcu` style           |
| **Appendix Handling**   | Template missing appendix infrastructure | Create `text/appendix.tex` wrapper          |
| **Compilation Failure** | Missing packages (`booktabs`, `float`)   | Update template preamble first              |

### 7.2 Medium-Risk Areas

| Risk                   | Description                         | Mitigation                               |
| ---------------------- | ----------------------------------- | ---------------------------------------- |
| **Discussion Section** | No template file for Discussion     | Create `text/discussion.tex`             |
| **Results Split**      | Arbitrary split may break narrative | Analyze subsection structure             |
| **Metadata Update**    | Student ID, dates, supervisor names | Update `researchProject.tex` lines 40-53 |

### 7.3 Low-Risk Areas

| Risk             | Description                                       | Mitigation                                    |
| ---------------- | ------------------------------------------------- | --------------------------------------------- |
| **Abstract**     | Direct copy                                       | Copy lines 30-68 to `text/abstract.tex`       |
| **Introduction** | Direct copy                                       | Copy lines 70-109 to `text/introduction.tex`  |
| **Conclusion**   | Direct copy (may need Code Availability appended) | Copy lines 1657-1707 to `text/conclusion.tex` |

---

## 8. Recommended Migration Sequence

### Phase 1: Pre-Migration Preparation (ANALYSIS COMPLETE)

✅ **Step 1.1:** Analyze template structure (researchProject.tex)  
✅ **Step 1.2:** Map current sections to template files  
✅ **Step 1.3:** Identify missing template files  
✅ **Step 1.4:** Assess package compatibility  
✅ **Step 1.5:** Create migration plan document (this file)

### Phase 2: Template Infrastructure Setup

⏳ **Step 2.1:** Update `researchProject.tex` metadata (lines 40-53)

- Student ID: 24130664
- Title: "A Deterministic N-of-1 Pipeline..."
- Supervisor: Dr. Agatha Mattos
- Year: 2026
- Due date: 28/01/2026

⏳ **Step 2.2:** Add missing packages to `researchProject.tex` preamble

- After line 31, add:
  ```latex
  \usepackage{booktabs}  % For professional tables
  \usepackage{float}     % For [H] placement
  ```

⏳ **Step 2.3:** Create missing template files

- `text/discussion.tex` (new file)
- `text/appendix.tex` (wrapper for 6 appendix files)

⏳ **Step 2.4:** Update `\includeonly` list (line 94-104)

- Add `text/discussion` after `text/evaluation`
- Keep `text/appendix` (will create wrapper)

⏳ **Step 2.5:** Update document body inclusion chain (after line 119)

- After `\input{text/evaluation}`, add:
  ```latex
  \input{text/discussion}
  ```

### Phase 3: Figure and Bibliography Migration

⏳ **Step 3.1:** Copy all figures

```bash
cp docs/latex/latest/figures/*.png docs/latex/templ/figures/
cp docs/latex/latest/figures/*.pdf docs/latex/templ/figures/
# Exclude google.png if desired
```

⏳ **Step 3.2:** Copy bibliography

```bash
cp docs/latex/latest/refs.bib docs/latex/templ/refs.bib
```

⏳ **Step 3.3:** Verify figure paths in copied content

- Search for `\includegraphics{figures/...}` patterns
- Update to `\includegraphics{...}` (graphicspath already set)

### Phase 4: Content Migration (Section by Section)

⏳ **Step 4.1:** Migrate Abstract

- Source: `main.tex` lines 30-68
- Target: `text/abstract.tex`
- Replace template placeholder with actual content

⏳ **Step 4.2:** Migrate Introduction

- Source: `main.tex` lines 70-109
- Target: `text/introduction.tex`
- **Preserve section command:** `\section{Introduction}` + `\label{sec:Introduction}`

⏳ **Step 4.3:** Migrate Literature Review

- Source: `main.tex` lines 110-166
- Target: `text/relatedwork.tex`
- **Section title:** Keep as "Literature Review" or rename to "Related Work"?

⏳ **Step 4.4:** Migrate Methodology

- Source: `main.tex` lines 167-1011 (845 lines)
- Target: `text/methodology.tex`
- **Verify subsection structure preserved**

⏳ **Step 4.5:** Migrate Clinical Background

- Source: `main.tex` lines 1012-1056
- Target: `text/design.tex`
- **Decision:** Rename file to `text/clinical_background.tex` or keep `design.tex`?

⏳ **Step 4.6:** Migrate Results (Part 1)

- Source: `main.tex` lines 1057-??? (split point TBD)
- Target: `text/implementation.tex`
- **Requires subsection analysis to determine split**

⏳ **Step 4.7:** Migrate Results (Part 2)

- Source: `main.tex` lines ???-1288
- Target: `text/evaluation.tex`
- **Includes Extended Modelling Experiments or separate?**

⏳ **Step 4.8:** Migrate Discussion

- Source: `main.tex` lines 1378-1656
- Target: `text/discussion.tex` (NEW FILE)

⏳ **Step 4.9:** Migrate Conclusion + Backmatter

- Source: `main.tex` lines 1657-1725
- Target: `text/conclusion.tex`
- **Includes:** Conclusion + Code Availability + Acknowledgments

### Phase 5: Appendix Migration

⏳ **Step 5.1:** Create appendix wrapper file

- File: `text/appendix.tex`
- Content:

  ```latex
  %% appendix.tex - Wrapper for all appendix sections

  \appendix

  % Include all appendix files from latest/
  \input{../latest/appendix_a.tex}
  \input{../latest/appendix_b.tex}
  \input{../latest/appendix_c.tex}
  \input{../latest/appendix_d.tex}
  \input{../latest/appendix_e.tex}
  \input{../latest/appendix_f.tex}
  ```

⏳ **Step 5.2:** OR copy all appendix files to templ/ root

```bash
cp docs/latex/latest/appendix_*.tex docs/latex/templ/
```

- Update `text/appendix.tex` to use relative paths:
  ```latex
  \input{../appendix_a.tex}
  \input{../appendix_b.tex}
  ...
  ```

### Phase 6: Compilation Testing

⏳ **Step 6.1:** Backup current template

```bash
cp -r docs/latex/templ docs/latex/templ.backup.2025-01-23
```

⏳ **Step 6.2:** Test compile in Overleaf (or local pdflatex)

```bash
cd docs/latex/templ
pdflatex researchProject.tex
bibtex researchProject
pdflatex researchProject.tex
pdflatex researchProject.tex
```

⏳ **Step 6.3:** Validate output

- Check all sections appear in correct order
- Verify all figures render
- Confirm bibliography compiles
- Validate appendices appear after main text

⏳ **Step 6.4:** Fix compilation errors

- Common issues:
  - Missing `\section{}` commands in template files
  - Incorrect figure paths
  - Missing package imports
  - `natbib` citation format mismatches

### Phase 7: Final Verification

⏳ **Step 7.1:** Content audit

- Compare page count: `latest/main.pdf` vs `templ/researchProject.pdf`
- Verify all sections present
- Check figure numbering consistency

⏳ **Step 7.2:** Cross-reference validation

- Test `\ref{}` commands (section, figure, table references)
- Verify `\cite{}` commands resolve correctly

⏳ **Step 7.3:** Create migration report

- Document any content changes
- List structural decisions made
- Note any deviations from template

---

## 9. Open Questions for User Decision

### 9.1 Structural Decisions

**Q1: Discussion Section Placement**

- Template has no `text/discussion.tex` file
- Options:
  - [A] Create new `text/discussion.tex` (recommended)
  - [B] Merge Discussion into `text/evaluation.tex`
  - [C] Merge Discussion into `text/conclusion.tex`
- **User decision required:** Which approach?

**Q2: Results Section Split**

- Current single `\section{Results}` must split into `implementation.tex` + `evaluation.tex`
- Options:
  - [A] Split by model type (ML6 vs ML7)
  - [B] Split by content (Methods vs Results)
  - [C] Analyze subsections first, then decide
- **User decision required:** Need subsection analysis?

**Q3: Extended Modelling Experiments**

- Current: Separate `\section{Extended Modelling Experiments}` (89 lines)
- Options:
  - [A] Append to `text/evaluation.tex`
  - [B] Create new `text/extended.tex`
  - [C] Move to appendix
- **User decision required:** Preferred placement?

**Q4: Clinical Background Section**

- Maps to template file `text/design.tex`
- Options:
  - [A] Keep filename `design.tex`, update content
  - [B] Rename to `text/clinical_background.tex`, update inclusions
- **User decision required:** Filename preference?

**Q5: Appendix Strategy**

- Template lists `text/appendix.tex` but file doesn't exist
- Current: 6 separate appendix files
- Options:
  - [A] Create wrapper `text/appendix.tex` that includes files from `../latest/`
  - [B] Copy all 6 appendix files to `templ/` root
  - [C] Consolidate all appendices into single `text/appendix.tex`
- **User decision required:** Which approach?

**Q6: Unnumbered Sections (Code Availability, Acknowledgments)**

- Not standard in template
- Options:
  - [A] Append both to `text/conclusion.tex`
  - [B] Create custom `text/backmatter.tex`
  - [C] Move to appendix
- **User decision required:** Preferred placement?

### 9.2 Technical Decisions

**Q7: Bibliography Style**

- Template uses `natbib` + `dcu` style
- Current uses `cite` package + unknown style
- **User decision required:** Switch to `natbib`? (Recommended: Yes)

**Q8: Figure Format Preference**

- Current has both PDF and PNG versions of figures
- Template `\DeclareGraphicsExtensions{.pdf,.png,.jpg}` prefers PDF
- **User decision required:** Copy both formats or PDF only?

**Q9: Placeholder Removal**

- Template `text/` files contain instructional placeholder text
- **User decision required:** Remove all placeholders or keep as comments?

---

## 10. Next Steps (Post-Analysis)

Once user provides decisions on open questions:

1. **Create detailed subsection-level mapping** for Results section split
2. **Generate exact line-number ranges** for each content block
3. **Create execution checklist** with copy/paste commands
4. **Prepare backup strategy** for rollback if needed
5. **Execute migration** in phases (infrastructure → content → validation)

---

## 11. File Manifest

### Files to Create

| File Path                   | Purpose            | Source Content                  |
| --------------------------- | ------------------ | ------------------------------- |
| `templ/text/discussion.tex` | Discussion section | `main.tex` lines 1378-1656      |
| `templ/text/appendix.tex`   | Appendix wrapper   | New file with `\input` commands |

### Files to Modify

| File Path                       | Modification                       | Lines Affected |
| ------------------------------- | ---------------------------------- | -------------- |
| `templ/researchProject.tex`     | Update metadata                    | 40-53          |
| `templ/researchProject.tex`     | Add packages (`booktabs`, `float`) | After line 31  |
| `templ/researchProject.tex`     | Update `\includeonly` list         | 94-104         |
| `templ/researchProject.tex`     | Add `\input{text/discussion}`      | After line 119 |
| `templ/text/abstract.tex`       | Replace placeholder                | All content    |
| `templ/text/introduction.tex`   | Replace placeholder                | All content    |
| `templ/text/relatedwork.tex`    | Replace placeholder                | All content    |
| `templ/text/methodology.tex`    | Replace placeholder                | All content    |
| `templ/text/design.tex`         | Replace placeholder                | All content    |
| `templ/text/implementation.tex` | Replace placeholder                | All content    |
| `templ/text/evaluation.tex`     | Replace placeholder                | All content    |
| `templ/text/conclusion.tex`     | Replace placeholder                | All content    |

### Files to Copy

| Source                  | Destination                 | Count                     |
| ----------------------- | --------------------------- | ------------------------- |
| `latest/figures/*.png`  | `templ/figures/`            | 15 files                  |
| `latest/figures/*.pdf`  | `templ/figures/`            | 14 files                  |
| `latest/refs.bib`       | `templ/refs.bib`            | 1 file                    |
| `latest/appendix_*.tex` | `templ/` or keep in latest/ | 6 files (decision needed) |

---

## 12. Estimated Migration Effort

| Phase                       | Tasks                           | Estimated Time | Risk Level |
| --------------------------- | ------------------------------- | -------------- | ---------- |
| **Phase 1: Analysis**       | ✅ Complete                     | 2 hours        | ✅ Low     |
| **Phase 2: Infrastructure** | Template updates, file creation | 1 hour         | 🟡 Medium  |
| **Phase 3: Figures/Bib**    | File copying, path validation   | 30 minutes     | 🟢 Low     |
| **Phase 4: Content**        | Section-by-section migration    | 3-4 hours      | 🔴 High    |
| **Phase 5: Appendix**       | Wrapper creation or file copy   | 30 minutes     | 🟡 Medium  |
| **Phase 6: Testing**        | Compilation, error fixing       | 1-2 hours      | 🔴 High    |
| **Phase 7: Verification**   | Audit, cross-refs, report       | 1 hour         | 🟡 Medium  |
| **Total**                   |                                 | **9-11 hours** |            |

---

## 13. Success Criteria

Migration is considered successful when:

1. ✅ `researchProject.tex` compiles without errors in Overleaf
2. ✅ All sections from `main.tex` appear in output PDF
3. ✅ All 15 figures render correctly
4. ✅ Bibliography compiles with all citations resolved
5. ✅ All 6 appendices appear after main text
6. ✅ Page count matches original `main.pdf` (±5% tolerance)
7. ✅ Cross-references (`\ref`, `\cite`) work correctly
8. ✅ Document structure follows NCI template conventions

---

## Appendix: Command Reference

### Useful Grep Patterns

```bash
# Find all section commands
grep -n '\\section' docs/latex/latest/main.tex

# Find all subsection commands
grep -n '\\subsection' docs/latex/latest/main.tex

# Find all figure environments
grep -n '\\begin{figure}' docs/latex/latest/main.tex

# Find all table environments
grep -n '\\begin{table}' docs/latex/latest/main.tex

# Find all citations
grep -n '\\cite' docs/latex/latest/main.tex

# Find all labels
grep -n '\\label' docs/latex/latest/main.tex

# Find all cross-references
grep -n '\\ref' docs/latex/latest/main.tex
```

### File Operations (When Executing Migration)

```bash
# Backup template before migration
cp -r docs/latex/templ docs/latex/templ.backup.$(date +%Y%m%d)

# Copy figures (PNG only)
cp docs/latex/latest/figures/*.png docs/latex/templ/figures/

# Copy figures (both formats)
cp docs/latex/latest/figures/{*.png,*.pdf} docs/latex/templ/figures/

# Copy bibliography
cp docs/latex/latest/refs.bib docs/latex/templ/

# Count lines in section (example: Methodology)
sed -n '167,1011p' docs/latex/latest/main.tex | wc -l
```

---

**END OF ANALYSIS DOCUMENT**
