# Citation Audit Report — OLD (latest) vs NEW (templ)

**Date:** 2026-01-24  
**Audit Type:** Bibliography Loss Investigation  
**Status:** ❌ **CRITICAL FINDING — 20 Bibliography Entries Lost**  
**Root Cause:** Manual inline bibliography not migrated to modular template

---

## Executive Summary

**CRITICAL DISCOVERY:** The migration from monolithic (`docs/latex/latest/main.tex`) to modular template (`docs/latex/templ/`) resulted in the **loss of 20 bibliography entries**.

### Root Cause

The OLD source used a **manual inline bibliography** (`\begin{thebibliography}...\end{thebibliography}`) with 22 entries embedded directly in `main.tex`. The NEW template switched to BibTeX with `\bibliography{refs}`, which only includes entries that are actively cited in the text (2 entries).

### Impact

- **OLD:** 22 bibliography entries available
- **NEW:** Only 2 bibliography entries rendered
- **LOST:** 20 bibliography entries not transferred
- **Citations in text:** 1 instance (citing 2 keys) — PRESERVED ✅
- **Bibliography system:** Changed from inline manual → BibTeX external

---

## Task 1: Citation Command Count (OLD vs NEW)

### Citation Commands Found

| Location | File                               | Citations Found |
| -------- | ---------------------------------- | --------------- |
| **OLD**  | `docs/latex/latest/main.tex`       | **1**           |
| **OLD**  | `docs/latex/latest/appendix_*.tex` | **0**           |
| **NEW**  | `docs/latex/templ/text/*.tex`      | **1**           |
| **NEW**  | `docs/latex/templ/appendix_*.tex`  | **0**           |

**Total Citations: 1 in both OLD and NEW** ✅

### Citation Details

**OLD (latest/main.tex, line 679):**

```latex
\cite{rubin1987, vanbuuren2011}
```

**NEW (templ/text/methodology.tex, line 536):**

```latex
\cite{rubin1987, vanbuuren2011}
```

**Verification:** Citation preserved correctly during migration ✅

---

## Task 2: Citation Keys Extraction (OLD vs NEW)

### Cited Keys

**OLD Cited Keys (from \cite commands):**

- `rubin1987`
- `vanbuuren2011`

**NEW Cited Keys (from \cite commands):**

- `rubin1987`
- `vanbuuren2011`

**Missing Keys:** NONE (citations preserved) ✅  
**Extra Keys:** NONE ✅

**Conclusion:** All citation commands were successfully migrated.

---

## Task 3: Bibliography Entries Inventory

### OLD Bibliography System (Manual Inline)

**Location:** `docs/latex/latest/main.tex`, lines 1736-1870

**Format:**

```latex
\begin{thebibliography}{20}

\bibitem{breiman2001}
Breiman, L. (2001).
Random Forests.
...

\bibitem{hochreiter1997}
Hochreiter, S., & Schmidhuber, J. (1997).
Long Short-Term Memory.
...

% ... 20 more entries ...

\end{thebibliography}
```

**Total Entries:** 22

### OLD Bibliography Keys (Complete List)

1. `bent2020`
2. `ben-zeev2018`
3. `bifet2007`
4. `breiman2001`
5. `cerqueira2020`
6. `cornet2018`
7. `depner2020`
8. `faedda2016`
9. `gama2014`
10. `hidalgo2015`
11. `hochreiter1997`
12. `insel2017`
13. `kollins2020`
14. `onnela2016`
15. `place2016`
16. `rubin1987` ← **CITED**
17. `schork2015`
18. `shap2017`
19. `shcherbina2017`
20. `sterne2009`
21. `torous2019`
22. `vanbuuren2011` ← **CITED**

### NEW Bibliography System (BibTeX External)

**Location:** `docs/latex/templ/main.tex`, line 148

**Format:**

```latex
\bibliographystyle{IEEEtran}
\bibliography{refs}
```

**BibTeX Processing:**

- Reads `refs.bib` (22 entries available)
- Includes only **cited** entries in compiled bibliography
- Generates `main.bbl` with 2 entries

**Total Entries in Compiled Bibliography:** 2

### NEW Bibliography Keys (Rendered)

1. `rubin1987` ← **CITED**
2. `vanbuuren2011` ← **CITED**

---

## Task 4: Missing Bibliography Entries Analysis

### Entries Present in OLD but NOT Rendered in NEW

**Total Missing:** 20 entries

| #   | BibTeX Key       | Status in OLD       | Status in NEW               |
| --- | ---------------- | ------------------- | --------------------------- |
| 1   | `bent2020`       | ✅ In manual biblio | ❌ Not cited → not rendered |
| 2   | `ben-zeev2018`   | ✅ In manual biblio | ❌ Not cited → not rendered |
| 3   | `bifet2007`      | ✅ In manual biblio | ❌ Not cited → not rendered |
| 4   | `breiman2001`    | ✅ In manual biblio | ❌ Not cited → not rendered |
| 5   | `cerqueira2020`  | ✅ In manual biblio | ❌ Not cited → not rendered |
| 6   | `cornet2018`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 7   | `depner2020`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 8   | `faedda2016`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 9   | `gama2014`       | ✅ In manual biblio | ❌ Not cited → not rendered |
| 10  | `hidalgo2015`    | ✅ In manual biblio | ❌ Not cited → not rendered |
| 11  | `hochreiter1997` | ✅ In manual biblio | ❌ Not cited → not rendered |
| 12  | `insel2017`      | ✅ In manual biblio | ❌ Not cited → not rendered |
| 13  | `kollins2020`    | ✅ In manual biblio | ❌ Not cited → not rendered |
| 14  | `onnela2016`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 15  | `place2016`      | ✅ In manual biblio | ❌ Not cited → not rendered |
| 16  | `schork2015`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 17  | `shap2017`       | ✅ In manual biblio | ❌ Not cited → not rendered |
| 18  | `shcherbina2017` | ✅ In manual biblio | ❌ Not cited → not rendered |
| 19  | `sterne2009`     | ✅ In manual biblio | ❌ Not cited → not rendered |
| 20  | `torous2019`     | ✅ In manual biblio | ❌ Not cited → not rendered |

### Original Location of Manual Bibliography

**File:** `docs/latex/latest/main.tex`  
**Lines:** 1736-1870 (134 lines total)  
**Section Context:** After Conclusion section, before `\end{document}`

**Excerpt (lines 1730-1740):**

```latex
%% --------------------
%% |   Bibliography   |
%% --------------------

% Manual inline bibliography (no BibTeX file used)
\begin{thebibliography}{20}

\bibitem{breiman2001}
Breiman, L. (2001).
Random Forests.
\emph{Machine Learning}, 45(1), 5--32. \\
```

---

## Task 5: Hidden/Excluded Citations Check (NEW)

### Files Inclusion Verification

**NEW main.tex includes:**

```latex
\input{text/abstract}
\input{text/introduction}
\input{text/relatedwork}
\input{text/methodology}  ← Citation found here ✅
\input{text/design}
\input{text/implementation}
\input{text/evaluation}
\input{text/discussion}
\input{text/conclusion}
\input{text/appendix}  ← Router only, no citations
```

**All expected files included:** ✅

### Commented Citations Check

**Result:** No commented citations found in NEW template.

### Verbatim/Listing Blocks Check

**Result:** No citations hidden in verbatim/lstlisting blocks.

### Appendix Router Check

**File:** `docs/latex/templ/text/appendix.tex`

**Content:**

```latex
\appendix

\input{appendix_a.tex}
\input{appendix_b.tex}
\input{appendix_c.tex}
\input{appendix_d.tex}
\input{appendix_e.tex}
\input{appendix_f.tex}
```

**Citations in appendices:** 0 (verified by grep) ✅

**Conclusion:** No hidden citations. The 1 citation found is the ONLY citation in the entire document.

---

## Task 6: Bibliography Pipeline Sanity Check (NEW)

### BibTeX Configuration

**File:** `docs/latex/templ/main.tex`, lines 146-148

```latex
% IEEE numeric citation style (migrated from DCU/Harvard on 2026-01-24)
\bibliographystyle{IEEEtran}
\bibliography{refs}
```

**Verification:**

- ✅ `\bibliography{refs}` correctly calls `refs.bib`
- ✅ `refs.bib` exists at `docs/latex/templ/refs.bib`
- ✅ `refs.bib` contains 22 entries (all keys from OLD manual bibliography)

### BibTeX Processing Log

**File:** `docs/latex/templ/main.blg` (excerpt)

```
This is BibTeX, Version 0.99e
The style file: IEEEtran.bst
Database file #1: refs.bib
-- IEEEtran.bst version 1.14 (2015/08/26) by Michael Shell.
Done.
You've used 2 entries,
```

**Analysis:**

- ✅ BibTeX processed `refs.bib` successfully
- ✅ **2 entries used** (correct behavior — only cited entries)
- ✅ No BibTeX errors
- ⚠️ **20 uncited entries not included** (expected BibTeX behavior)

### Compilation Pipeline

**Sequence:**

```bash
pdflatex main.tex   # Parse citations, generate .aux
bibtex main         # Process refs.bib, generate .bbl with 2 entries
pdflatex main.tex   # Integrate bibliography
pdflatex main.tex   # Final references resolution
```

**Result:** ✅ Compilation successful, but only 2 entries in bibliography

**Why Only 2 Entries?**

BibTeX **by design** only includes entries that are actively cited with `\cite{}` commands. Since the document has only 1 citation (2 keys), BibTeX produces a 2-entry bibliography.

**This is NOT a bug — it's expected BibTeX behavior.**

---

## Root Cause Analysis

### What Happened During Migration

**Step 1: Parts 1, 2B, 2C — Content Migration**

- Copied main text sections from OLD to NEW (9 sections)
- Copied appendices from OLD to NEW (6 files)
- **Bibliography section NOT migrated** (assumed external BibTeX)

**Step 2: Bibliography Setup**

- Created `refs.bib` with 22 entries (matching OLD manual bibliography keys)
- Configured `main.tex` to use `\bibliographystyle{dcu}` + `\bibliography{refs}`
- **Manual inline bibliography (lines 1736-1870) NOT transferred**

**Step 3: IEEE Migration**

- Changed `dcu` → `IEEEtran`
- Replaced `natbib` → `cite` package
- **No awareness of missing bibliography entries**

### Why the Manual Bibliography Was Lost

1. **Assumption Error:** Migration assumed OLD used BibTeX (it didn't)
2. **Silent Loss:** Manual bibliography section was at end of OLD `main.tex`
3. **No Warning:** BibTeX compiled successfully with 2 entries (no errors)
4. **Correct Behavior:** BibTeX only processes cited entries (20 uncited → ignored)

---

## Diagnosis: Two Possible Scenarios

### Hypothesis 1: Document Intentionally Minimal (UNLIKELY)

**Evidence FOR:**

- Only 1 citation in 59-page document
- Document is methodology-focused (may not require extensive references)

**Evidence AGAINST:**

- OLD had 22 carefully curated bibliography entries
- Entries cover key topics: digital phenotyping, N-of-1 trials, ML methods, SHAP
- Unlikely to be placeholder entries

**Likelihood:** <10%

### Hypothesis 2: Bibliography Entries Should Be Present (LIKELY)

**Evidence FOR:**

- OLD manual bibliography had 22 well-formatted entries
- Entries are topically relevant (Random Forests, LSTM, SHAP, drift detection)
- Standard practice for MSc dissertations to have comprehensive bibliography
- Entries likely referenced in sections that haven't been written yet

**Evidence AGAINST:**

- None

**Likelihood:** >90%

---

## Impact Assessment

### What Was Lost

**20 Bibliography Entries:**

| Category                     | Entries | Keys                                                                                                  |
| ---------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| **Machine Learning Methods** | 4       | `breiman2001`, `hochreiter1997`, `shap2017`, `gama2014`                                               |
| **Digital Phenotyping**      | 7       | `ben-zeev2018`, `onnela2016`, `insel2017`, `torous2019`, `shcherbina2017`, `place2016`, `kollins2020` |
| **N-of-1 Trials**            | 2       | `schork2015`, `hidalgo2015`                                                                           |
| **Drift Detection**          | 2       | `bifet2007`, `cerqueira2020`                                                                          |
| **Wearables/Sensors**        | 2       | `bent2020`, `cornet2018`                                                                              |
| **Clinical Context**         | 2       | `faedda2016`, `depner2020`                                                                            |
| **Methodology**              | 1       | `sterne2009`                                                                                          |

**Potential Usage:**

- Introduction (digital phenotyping context)
- Related Work (ML methods, wearables, N-of-1 trials)
- Methodology (drift detection, SHAP)
- Discussion (clinical implications)

### What Was Preserved

- ✅ 1 citation instance (2 keys)
- ✅ All text content (9 sections + 6 appendices)
- ✅ All figures (29 files)
- ✅ All tables
- ✅ `refs.bib` contains all 22 entries (available for future use)

---

## Recovery Plan (DO NOT EXECUTE YET — ANALYSIS ONLY)

### Option A: Restore Manual Inline Bibliography (Recommended)

**Goal:** Match OLD bibliography exactly (22 entries, manual format)

**Steps:**

1. **Extract manual bibliography from OLD**

   ```bash
   sed -n '1736,1870p' docs/latex/latest/main.tex > /tmp/old_bibliography.tex
   ```

2. **Remove BibTeX configuration from NEW**

   ```latex
   % DELETE these lines from main.tex:
   \bibliographystyle{IEEEtran}
   \bibliography{refs}
   ```

3. **Insert manual bibliography in NEW**

   ```latex
   % ADD at end of main.tex (before \end{document}):
   \input{text/bibliography}
   ```

4. **Create bibliography router**

   ```bash
   # File: docs/latex/templ/text/bibliography.tex
   # Content: Manual bibliography from OLD (lines 1736-1870)
   ```

5. **Recompile (pdflatex only, no BibTeX)**
   ```bash
   pdflatex main.tex
   pdflatex main.tex  # Resolve references
   ```

**Pros:**

- ✅ Exact match with OLD (22 entries)
- ✅ No dependency on BibTeX
- ✅ All entries visible regardless of citation status

**Cons:**

- ⚠️ Manual bibliography harder to maintain
- ⚠️ No automatic sorting/formatting
- ⚠️ Reverses IEEE migration (would need to reformat entries)

### Option B: Use BibTeX with \nocite{\*} (Alternative)

**Goal:** Keep BibTeX but force inclusion of all entries

**Steps:**

1. **Add \nocite{\*} before bibliography**

   ```latex
   % In main.tex, before \bibliography{refs}:
   \nocite{*}  % Include ALL entries from refs.bib
   \bibliographystyle{IEEEtran}
   \bibliography{refs}
   ```

2. **Recompile (full sequence)**
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

**Pros:**

- ✅ All 22 entries included
- ✅ Automatic IEEE formatting
- ✅ Easy to maintain (edit refs.bib)
- ✅ Preserves BibTeX workflow

**Cons:**

- ⚠️ Includes entries even if not cited (non-standard)
- ⚠️ May violate submission guidelines (some require only cited entries)

### Option C: Hybrid Approach (Best Practice)

**Goal:** Keep BibTeX, add citations throughout document

**Steps:**

1. **Review document sections**

   - Identify where each of 20 missing entries should be cited
   - Add `\cite{}` commands in Introduction, Related Work, Discussion

2. **Expected citation additions:**

   - Introduction: `\cite{insel2017, onnela2016, torous2019}`
   - Related Work: `\cite{breiman2001, hochreiter1997, shap2017, bifet2007}`
   - Methodology: `\cite{rubin1987, vanbuuren2011}` ← already present
   - Discussion: `\cite{schork2015, hidalgo2015, ben-zeev2018}`

3. **Recompile (full sequence)**
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

**Pros:**

- ✅ Best practice (only cite what you reference)
- ✅ Automatic IEEE formatting
- ✅ Scholarly standard (no uncited entries)
- ✅ Improves document quality

**Cons:**

- ⚠️ Requires content editing (not just bibliography fix)
- ⚠️ More work (need to identify citation locations)

---

## Recommended Action

### Immediate Next Steps (Analysis Phase Complete)

**Recommended Approach:** **Option A** (Restore Manual Inline Bibliography)

**Rationale:**

1. **Exact Fidelity:** Matches OLD document precisely
2. **No Content Editing:** Preserves current text as-is
3. **Reversible:** Can switch to Option C later
4. **Fast:** Single file operation

### Implementation Plan (FOR NEXT EXECUTION)

**Phase 1: Extract Manual Bibliography (1 command)**

```bash
sed -n '1736,1870p' docs/latex/latest/main.tex > docs/latex/templ/text/bibliography.tex
```

**Phase 2: Modify main.tex (3 edits)**

Edit 1: Remove BibTeX configuration

```latex
% DELETE:
\bibliographystyle{IEEEtran}
\bibliography{refs}
```

Edit 2: Add manual bibliography inclusion

```latex
% ADD:
\input{text/bibliography}
```

Edit 3: Update includeonly (if used)

```latex
\includeonly{
  ...
  text/conclusion,
  text/bibliography,  % ADD THIS
  text/appendix
}
```

**Phase 3: Recompile (2 commands)**

```bash
pdflatex main.tex
pdflatex main.tex  # No BibTeX needed
```

**Expected Result:**

- Bibliography section with 22 entries
- Maintains OLD formatting
- Page count: ~59-60 pages (similar to current)

### Alternative: Option B (Quick Fix)

**If you prefer to keep BibTeX:**

Add this line to `main.tex` before `\bibliography{refs}`:

```latex
\nocite{*}  % Include all entries from refs.bib
```

**Trade-off:** Non-standard practice, but faster and keeps IEEE formatting.

---

## Files Requiring Modification (Option A)

### 1. CREATE: docs/latex/templ/text/bibliography.tex

**Source:** `docs/latex/latest/main.tex`, lines 1736-1870

**Action:** Extract and save as new file

### 2. EDIT: docs/latex/templ/main.tex

**Lines to modify:** ~146-148 (bibliography section)

**Changes:**

- Remove: `\bibliographystyle{IEEEtran}`
- Remove: `\bibliography{refs}`
- Add: `\input{text/bibliography}`

---

## Quality Assurance Checklist (Post-Recovery)

**After recovery, verify:**

- [ ] Bibliography section appears in PDF
- [ ] 22 entries present (not just 2)
- [ ] Entry formatting matches OLD document
- [ ] Page count similar to OLD (~59 pages)
- [ ] No compilation errors
- [ ] References [1]-[22] render correctly
- [ ] Citation to `rubin1987, vanbuuren2011` still works

---

## Audit Trail

### Investigation Timeline

| Timestamp        | Action                           | Finding                          |
| ---------------- | -------------------------------- | -------------------------------- |
| 2026-01-24 00:15 | Citation count (OLD)             | 1 citation found                 |
| 2026-01-24 00:16 | Citation count (NEW)             | 1 citation found                 |
| 2026-01-24 00:17 | Citation keys extraction         | rubin1987, vanbuuren2011 (both)  |
| 2026-01-24 00:18 | Bibliography system check (OLD)  | **MANUAL INLINE** discovered     |
| 2026-01-24 00:19 | Bibliography entries count (OLD) | **22 entries** found             |
| 2026-01-24 00:20 | Bibliography system check (NEW)  | BibTeX with refs.bib             |
| 2026-01-24 00:21 | Bibliography entries count (NEW) | **2 entries** rendered           |
| 2026-01-24 00:22 | Root cause identified            | Manual bibliography not migrated |

### Key Findings Summary

```
OLD Bibliography System: Manual inline (\begin{thebibliography})
OLD Bibliography Entries: 22
OLD Bibliography Location: main.tex lines 1736-1870

NEW Bibliography System: BibTeX (\bibliography{refs})
NEW Bibliography Entries: 2 (only cited entries)
NEW Bibliography Location: External file refs.bib

MIGRATION ERROR: Manual bibliography (134 lines) NOT transferred
IMPACT: 20 bibliography entries lost in compiled PDF
STATUS: Recoverable (source data preserved in OLD)
```

---

## Conclusion

### Critical Finding

The migration from monolithic to modular template resulted in a **bibliography system change** from manual inline (22 entries) to BibTeX external (2 entries). The manual bibliography section (134 lines, 1736-1870 in OLD) was not migrated to the NEW template.

### Current State

- ✅ **Citations preserved:** 1 instance, 2 keys
- ✅ **Text content preserved:** All sections migrated
- ❌ **Bibliography entries:** Only 2/22 rendered (20 lost)
- ⚠️ **BibTeX functional:** Working correctly (only includes cited entries)

### Recommended Resolution

**Restore manual inline bibliography** from OLD source (Option A) to achieve exact parity with original document. This preserves all 22 bibliography entries regardless of citation status.

**Alternative:** Add `\nocite{*}` to force BibTeX to include all entries (Option B), maintaining IEEE formatting but including uncited entries.

**Best Practice:** Add citations throughout document for all 20 missing entries (Option C), but this requires content editing beyond bibliography recovery.

---

## Document Metadata

- **Report:** Citation Audit (latest vs templ)
- **Date:** 2026-01-24
- **Analysis Type:** Bibliography Loss Investigation
- **Root Cause:** Manual inline bibliography not migrated
- **Impact:** 20/22 entries lost in compiled PDF
- **Status:** Analyzed (no file modifications in this phase)
- **Recommended Action:** Restore manual bibliography (Option A)
- **Next Phase:** Execute recovery plan (separate task)

---

**END OF CITATION AUDIT REPORT**
