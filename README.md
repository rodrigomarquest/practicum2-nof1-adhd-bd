# N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-lightgrey.svg)](https://www.kaggle.com/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-October%202025-brightgreen.svg)]()

**Author:** Rodrigo Marques Teixeira
**Supervisor:** Dr. Agatha Mattos
**Course:** MSc Artificial Intelligence for Business – National College of Ireland
**Student ID:** 24130664
**Period:** Jan 2023 – Jan 2026

---

## 📘 About the Project

This repository consolidates the full technical and ethical framework developed during **Practicum Part 2** of the MSc in Artificial Intelligence for Business at the National College of Ireland.
It integrates **data preprocessing (ETL)**, **feature engineering**, **model training (baselines and LSTM)**, **SHAP explainability**, and **ethical documentation** compliant with GDPR and NCI Ethics requirements.

The project extends the previous Practicum (CA2) phase and focuses on reprocessing and modelling data from an **N-of-1 longitudinal study on comorbidity ADHD + Bipolar Disorder**, collected from wearable sensors (Apple Health, Amazfit GTR4, Helio Ring) and EMA-based self-reports.

---

## 📊 Repository Structure

```
etl/            → ETL scripts and data quality control
notebooks/      → Feature engineering and model training notebooks (Kaggle-compatible)
models/         → Trained models and explainability results
docs/           → Ethics, consent, configuration manual, and data governance
data/           → Synthetic or anonymised sample data for reproducibility
```

---

## ⚙️ Reproducibility and Environment

### 1. Requirements

```bash
cd etl
pip install -r requirements.txt
python etl_pipeline.py
```

### 2. Output

* Generates `features_daily_sample.csv` and `etl_qc_summary.csv`
* Normalises physiological and behavioural signals per segment (S1–S6)
* Handles missingness, z-score scaling, and fallback replacements

---

## 🔷 Modeling and Explainability

The modeling stage extends the ETL outputs to time-series classification notebooks executed in Kaggle (GPU T4, Python 3.10):

1. **01_feature_engineering.ipynb** – Daily feature extraction and aggregation
2. **02_model_training.ipynb** – Baselines (Naïve, Moving Average, Logistic Regression) and LSTM models (M1–M3)
3. **03_shap_analysis.ipynb** – SHAP explainability for drift detection and top feature attribution
4. **04_rule_based_baseline.ipynb** – Clinical heuristic baseline (rule-based model)

**Cross-validation:** 6 temporal folds (4 months training / 2 months validation)
**Metrics:** F1-macro, AUROC-OvR, Balanced Accuracy, Cohen’s κ, and McNemar p-test
**Output:** `models/best_model.tflite`, `metrics_summary.csv`, and `shap_top5_features.csv`

---

## 🧠 Ethics and Data Protection

* Repository contains **no identifiable data**.
* All sensitive raw data are stored locally and encrypted.
* Consent forms, participant information, and data management plan are available in `/docs`.
* Data collection follows GDPR, NCI Ethics Committee procedures, and anonymisation guidelines.

### Ethical Scope

* **Phase 1:** Self-data (personal wearable and app-based metrics)
* **Phase 2:** Expansion to family/friends/classmates with consent and anonymisation

---

## 🖊️ Version Control and Tagging

| Tag                      | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| `v2.0-pre-ethics`        | ETL + documentation prior to ethics approval                     |
| `v2.1-ethics-approved`   | Documents reviewed and approved by supervisor / ethics committee |
| `v2.2-modeling-complete` | Modeling and SHAP results finalised                              |
| `v2.3-final-report`      | CA3 / Dissertation submission version                            |

---

## 🌐 Citation

If referencing this repository in academic work:

> Teixeira, R. M. (2025). *N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)*. National College of Ireland. GitHub repository: [https://github.com/rodrigomarques/practicum2-nof1-adhd-bd](https://github.com/rodrigomarques/practicum2-nof1-adhd-bd)

---

## 🔒 License

This work is licensed under a **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.
It may be reused for educational or academic purposes with proper credit and under the same license terms.

---

## 📞 Contact

* **Rodrigo Marques Teixeira** – MSc AI for Business, National College of Ireland
* **Supervisor:** Dr. Agatha Mattos
* Email: [insert academic or contact email here]

---

**Version:** v2.0-pre-ethics
**Last updated:** October 2025
