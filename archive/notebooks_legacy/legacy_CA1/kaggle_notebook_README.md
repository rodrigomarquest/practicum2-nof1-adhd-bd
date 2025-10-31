# Personalised Passive Multisensor Monitoring for ADHD and Bipolar Disorder (N-of-1)

This notebook is part of a master's research practicum exploring the feasibility of personalised digital phenotyping using passive sensor data collected over 942 days from a single adult with comorbid ADHD and Bipolar Disorder (BD).

---

## 🧠 Project Summary

- **Study Type:** N-of-1, longitudinal
- **Duration:** Jan 2023 – Aug 2025
- **Data Sources:** Apple Health, Zepp OS (GTR4 + Helio Ring), iPhone (ScreenTime), manual EMA
- **Labels:** Simulated labels (real EMA integration planned)
- **Features:** 24 engineered digital biomarkers
- **Goal:** Lightweight, on-device neural network to detect mood/attention states

---

## ⚙️ Notebooks Overview

- `feature_engineering.ipynb`: Creates 24 daily features from raw health/screen/sleep signals
- `model_training.ipynb`: Baselines, LSTM, SHAP explanations, concept drift, TFLite export

---

## 🧪 Best Performing Model

| Metric       | Value                 |
|--------------|------------------------|
| F1-macro     | 0.928 (simulated labels) |
| Model        | Dense Neural Network    |
| Model Size   | 11.7 KB (.tflite)       |
| Latency      | ~96.1 ms/sample         |

---

## 📉 SHAP – Top Features

- `depressive_pattern`
- `manic_pattern`
- `energy_proxy`

---

## 🔁 Next Steps

- Real EMA collection (bi-weekly)
- External testing with Apple HealthKit donors (opt-in)
- Drift detection on live deployment
- Expansion to Android-based passive sensing

---

## 🧩 Ethics and Transparency

AI tools (OpenAI ChatGPT + Claude) were used for code structuring, debugging, and explanation. Full disclosure is available in the associated LaTeX appendix.

---

## 📄 License and Usage

Data is anonymised and author-owned. Please do not redistribute raw CSVs without permission.

---

© Rodrigo Marques Teixeira – MSc in Artificial Intelligence – NCI (2025)
