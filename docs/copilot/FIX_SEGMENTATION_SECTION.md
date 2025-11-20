# Corrected Segmentation Section for main.tex

## 🔴 CRITICAL FIX REQUIRED

**Location**: `docs/main.tex`, lines 440-493  
**Section**: 3.4 Behavioural Segmentation

---

## Current Text (INCORRECT — Contains Non-Existent Rules)

The current PDF describes **3 segmentation rules**, but the code only implements **2**.

**Problems**:

1. Rule 2.2 states "gaps longer than **three days**" but code uses `if delta > 1` (gap > **1 day**)
2. Rule 2.3 "Abrupt Behavioural Shifts" with threshold detection **does not exist in the code**
3. This describes the unused `auto_segment.py` module, not the actual pipeline

---

## CORRECTED TEXT (Replace lines 440-493 with this)

```latex
\subsection{Behavioural Segmentation}
\label{sec:segmentation}

Behavioural segmentation partitions the eight-year timeline into internally homogeneous
periods that reflect stable routines, physiological baselines and device usage patterns.
Segments are used exclusively for normalisation and anti-leak protection; they do not
participate directly in modelling or label computation. The final deterministic method
used in this study combines two chronological rules applied in fixed order, matching
the implementation in \texttt{stage\_apply\_labels.py} for the reproducible snapshot
dated 2025--11--07.

\subsubsection{Calendar-Based Boundaries}

A new segment is created at every month-to-month and year-to-year transition. These
boundaries capture natural structural changes in daily routines, sleep timing and activity
patterns, and provide a coarse but reliable temporal backbone. This rule ensures that no
segment spans periods with different seasonal or routine characteristics.

\subsubsection{Gaps Longer Than One Day}

Any break of more than one consecutive day in the unified daily table triggers a
segment boundary. Such gaps typically arise from device non-wear, travel, illness or
unusual schedules. Treating each gap as a deterministic boundary prevents spurious
continuity and ensures that segment statistics are based on contiguous periods only.

\subsubsection{Deterministic Assignment and Anti-Leak Role}

Each day receives exactly one segment identifier assigned in chronological order based
exclusively on the two rules above. All features used in modelling (and all PBSI components)
are z-scored \emph{within} their segment, preventing global statistics from leaking future
information into past days. This design eliminates temporal shortcuts and ensures that
ML6/ML7 models learn only within-segment structure rather than long-term trends.

\subsubsection{Final Segment Structure}

Applying these rules to the 2017--2025 timeline yields a total of 119 behavioural segments,
with durations ranging from 7 to 91 days. These segments provide stable, interpretable
contexts for subsequent feature scaling, PBSI computation and drift analysis.
```

---

## What Changed

| Element                    | Old (Wrong)                                        | New (Correct)                        |
| -------------------------- | -------------------------------------------------- | ------------------------------------ |
| **Number of rules**        | 3 rules                                            | 2 rules                              |
| **Rule 2.2 gap threshold** | "gaps longer than **three days**"                  | "more than **one consecutive day**"  |
| **Rule 2.3**               | "Abrupt Behavioural Shifts" with threshold formula | **DELETED** (does not exist in code) |
| **Subsection 2.4**         | "Deterministic Assignment and Anti-Leak Role"      | Simplified and clarified             |
| **Total changes**          | ~55 lines                                          | ~30 lines (simpler, accurate)        |

---

## Why This Matters

The incorrect description suggests a sophisticated multi-rule segmentation system that **does not exist in the actual pipeline**.

The confusion arose because:

- A sophisticated module `src/labels/auto_segment.py` exists with 4 detection rules
- BUT it is **not imported or used anywhere** in the codebase
- The actual pipeline uses only 2 simple rules in `stage_apply_labels.py`

A PhD examiner or journal reviewer would immediately notice this discrepancy when trying to reproduce the results.

---

## Verification

After making this change, verify:

1. ✅ Line 807 in Results section still says "119 behavioural segments" (already correct)
2. ✅ No other parts of the paper reference Rule 2.3 or "abrupt shifts"
3. ✅ Recompile LaTeX without errors
4. ✅ Run `make qc-all` to confirm pipeline still produces 119 segments

---

## Optional Enhancement (Future Work Section)

Consider adding this paragraph to the Discussion or Future Work section:

```latex
\textbf{Alternative Segmentation Approaches.}
While this study employed a deterministic two-rule segmentation strategy (calendar
boundaries and gaps), a more sophisticated multi-rule approach was prototyped in
\texttt{src/labels/auto\_segment.py}. This module detects data source transitions
(Apple$\leftrightarrow$Zepp), physiological signal changes (HR, HRV, sleep efficiency
shifts exceeding threshold $\tau$), and temporal fallbacks. However, the simpler
calendar-based approach was selected for the final pipeline due to superior
reproducibility and interpretability trade-offs. Future work with multiple participants
could explore adaptive segmentation methods to capture inter-individual variability.
```

This acknowledges the unused module and positions it as future work rather than a mistake.

---

## Estimated Time to Fix

- **Replace text in main.tex**: 5 minutes
- **Recompile and check**: 5 minutes
- **Add optional Future Work paragraph**: 10 minutes

**Total**: ~20 minutes

---

**Priority**: 🔴 **CRITICAL — Fix before any submission, defense, or publication**
