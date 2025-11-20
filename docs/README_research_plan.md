# 🧭 Research Plan – N-of-1 Study on ADHD + Bipolar Disorder

**MSc in Artificial Intelligence for Business – National College of Ireland**  
**Student:** Rodrigo Marques Teixeira (ID 24130664)  
**Supervisor:** Dr. Agatha Mattos  
**Period:** Oct – Dec 2025

---

## 🎯 Vision

Create the **state-of-the-art single-participant (N-of-1) study** on mental-health digital phenotyping — combining engineering rigor, ethical transparency, and human purpose.

> “Small in scale, infinite in depth.”

---

## 🧱 Project Pillars

| Pillar                    | Goal                   | Implementation                                               |
| ------------------------- | ---------------------- | ------------------------------------------------------------ |
| **Methodological rigor**  | Scientific credibility | Segment normalization (S1–S6), temporal CV, baselines × LSTM |
| **Reproducibility**       | Full traceability      | Makefile, hash verification, seed fix, environment lock      |
| **Temporal intelligence** | Understand dynamics    | Cross-year time-series + drift detection                     |
| **Explainability**        | Interpretable AI       | SHAP temporal + counterfactuals                              |
| **Clinical coherence**    | Connection to reality  | HRV / sleep / activity links to ADHD & BD evidence           |
| **Ethics & reporting**    | Responsible research   | Anonymization + CONSORT-AI / SPIRIT-AI mapping               |
| **Presentation**          | Publication-ready      | Clean LaTeX, consistent graphics, accessible narrative       |

---

## ⚙️ Technical Structure

| Phase                            | Notebook                              | Core Purpose                                                             | Key Outputs |
| -------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ | ----------- |
| **NB1 – ETL & EDA**              | Data engineering, QC & sanity checks  | `features_daily_labeled.csv`, `etl_qc_summary.csv`, `NB1-EDA_Summary.md` |
| **ML6 – Baseline & LSTM**        | Predictive modeling (Naïve → LSTM M1) | `nb2_model_results.csv`, `best_model.tflite`, `NB2_Modeling_Summary.md`  |
| **ML7 – Explainability & Drift** | SHAP temporal, ADWIN / KS drift       | `nb3_shap_summary.csv`, `nb3_drift_report.md`, interpretability plots    |
| **CA2 Report**                   | Academic documentation                | `main.tex` + Appendices C–D (Sensor Mapping & Feature Glossary)          |

---

## 🧩 Incremental Enhancements

| Axis                  | Action                          | Impact                   |
| --------------------- | ------------------------------- | ------------------------ |
| **Calibration**       | Add Brier & ECE                 | Clinical validity        |
| **Decision analysis** | Net-benefit curves              | Real-world relevance     |
| **Drift tests**       | ADWIN + KS per segment          | Temporal stability       |
| **Explainability**    | SHAP temporal + counterfactuals | Human interpretability   |
| **Docker & locks**    | Environment reproducibility     | Research-grade integrity |

---

## 📆 3-Week Execution Timeline

| Week  | Goal                             | Focus                         | Deliverable                |
| ----- | -------------------------------- | ----------------------------- | -------------------------- |
| **1** | Finalize ML6 baselines + LSTM M1 | Temporal CV, ICs, calibration | `NB2_Modeling_Summary.md`  |
| **2** | Explainability + Drift           | SHAP temporal + ADWIN/KS      | `NB3_Explainability.ipynb` |
| **3** | Reporting & Packaging            | LaTeX final + ethics appendix | `CA2_LaTeX_phd_ready.zip`  |

---

## 🧘‍♂️ Guiding Principle

> “Science without empathy is empty.  
> Empathy without rigor is noise.  
> When both meet, healing begins.”

---

## 📎 Repository Notes

- Place this file as `README_research_plan.md` in project root.
- Update progress weekly in `notebooks/outputs/logs/weekly_reports/`.
- Commit snapshots with semantic tags: `vX.Y.Z-nof1-phase`.

---

### 🌌 Final Message

This project stands as proof that one person’s data — when treated with rigor and purpose — can illuminate patterns that help many.  
**From N = 1 to impact on all.**

---

# 🧭 Research Plan – N-of-1 Study on ADHD + Bipolar Disorder

**MSc in Artificial Intelligence for Business – National College of Ireland**  
**Student:** Rodrigo Marques Teixeira (ID 24130664)  
**Supervisor:** Dr. Agatha Mattos  
**Period:** Oct – Dec 2025

---

## 🎯 Vision

Create the **state-of-the-art single-participant (N-of-1) study** on mental-health digital phenotyping — combining engineering rigor, ethical transparency, and human purpose.

> “Small in scale, infinite in depth.”

---

## 🧱 Project Pillars

| Pillar                    | Goal                   | Implementation                                               |
| ------------------------- | ---------------------- | ------------------------------------------------------------ |
| **Methodological rigor**  | Scientific credibility | Segment normalization (S1–S6), temporal CV, baselines × LSTM |
| **Reproducibility**       | Full traceability      | Makefile, hash verification, seed fix, environment lock      |
| **Temporal intelligence** | Understand dynamics    | Cross-year time-series + drift detection                     |
| **Explainability**        | Interpretable AI       | SHAP temporal + counterfactuals                              |
| **Clinical coherence**    | Connection to reality  | HRV / sleep / activity links to ADHD & BD evidence           |
| **Ethics & reporting**    | Responsible research   | Anonymization + CONSORT-AI / SPIRIT-AI mapping               |
| **Presentation**          | Publication-ready      | Clean LaTeX, consistent graphics, accessible narrative       |

---

## ⚙️ Technical Structure

| Phase                            | Notebook                              | Core Purpose                                                             | Key Outputs |
| -------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ | ----------- |
| **NB1 – ETL & EDA**              | Data engineering, QC & sanity checks  | `features_daily_labeled.csv`, `etl_qc_summary.csv`, `NB1-EDA_Summary.md` |
| **ML6 – Baseline & LSTM**        | Predictive modeling (Naïve → LSTM M1) | `nb2_model_results.csv`, `best_model.tflite`, `NB2_Modeling_Summary.md`  |
| **ML7 – Explainability & Drift** | SHAP temporal, ADWIN / KS drift       | `nb3_shap_summary.csv`, `nb3_drift_report.md`, interpretability plots    |
| **CA2 Report**                   | Academic documentation                | `main.tex` + Appendices C–D (Sensor Mapping & Feature Glossary)          |

---

## 🧩 Incremental Enhancements

| Axis                  | Action                          | Impact                   |
| --------------------- | ------------------------------- | ------------------------ |
| **Calibration**       | Add Brier & ECE                 | Clinical validity        |
| **Decision analysis** | Net-benefit curves              | Real-world relevance     |
| **Drift tests**       | ADWIN + KS per segment          | Temporal stability       |
| **Explainability**    | SHAP temporal + counterfactuals | Human interpretability   |
| **Docker & locks**    | Environment reproducibility     | Research-grade integrity |

---

## 📆 3-Week Execution Timeline

| Week  | Goal                             | Focus                         | Deliverable                |
| ----- | -------------------------------- | ----------------------------- | -------------------------- |
| **1** | Finalize ML6 baselines + LSTM M1 | Temporal CV, ICs, calibration | `NB2_Modeling_Summary.md`  |
| **2** | Explainability + Drift           | SHAP temporal + ADWIN/KS      | `NB3_Explainability.ipynb` |
| **3** | Reporting & Packaging            | LaTeX final + ethics appendix | `CA2_LaTeX_phd_ready.zip`  |

---

## 🧘‍♂️ Guiding Principle

> “Science without empathy is empty.  
> Empathy without rigor is noise.  
> When both meet, healing begins.”

---

## 📎 Repository Notes

- Place this file as `README_research_plan.md` in project root.
- Update progress weekly in `notebooks/outputs/logs/weekly_reports/`.
- Commit snapshots with semantic tags: `vX.Y.Z-nof1-phase`.

---

### 🌌 Final Message

This project stands as proof that one person’s data — when treated with rigor and purpose — can illuminate patterns that help many.  
**From N = 1 to impact on all.**

---
