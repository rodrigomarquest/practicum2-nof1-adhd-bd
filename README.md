N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)

![Last Updated](https://img.shields.io/badge/Last%20Updated-October 2025-brightgreen.svg)

Author: Rodrigo Marques Teixeira
Supervisor: Dr. Agatha Mattos
Course: MSc Artificial Intelligence for Business – National College of Ireland
Student ID: 24130664
Period: Jan 2023 – Jan 2026

📘 About the Project

This repository consolidates the full technical and ethical framework developed during Practicum Part 2 of the MSc in Artificial Intelligence for Business at NCI.
It integrates data pre-processing (ETL), feature engineering, time-series modeling (baselines + LSTM), SHAP explainability, and complete GDPR/ethics documentation.

The project extends the previous Practicum (Part 1) phase and focuses on reprocessing and modelling data from an N-of-1 longitudinal study on comorbidity ADHD + Bipolar Disorder, collected from wearable sensors (Apple Health, Amazfit GTR 4, Helio Ring) and self-reports (EMA / State of Mind).

📊 Repository Structure
etl/ → ETL scripts and data quality control
ios_extract/ → iPhone Screen Time and KnowledgeC extraction tools
notebooks/ → Feature engineering and model training (Kaggle-compatible)
models/ → Trained models, metrics and SHAP explainability
docs/ → Ethics, consent, configuration manual, governance
data/ → Synthetic or anonymised samples for reproducibility

⚙️ Reproducibility and Environment
cd etl
pip install -r requirements.txt
python etl_pipeline.py

Outputs

features_daily_sample.csv

etl_qc_summary.csv

Includes z-score scaling, missing-value handling, and segment normalization (S1–S6).

🧩 iOS Screen Time / Usage Extraction

A dedicated sub-module (ios_extract/) automates the pipeline from encrypted local iTunes backup → decrypted Manifest → Screen Time/KnowledgeC data.

ios_extract/
├─ decrypt_manifest.py # decrypts Manifest.db and validates SQLite
├─ quick_post_backup_probe.py # probes blobs present (flags=1)
├─ extract_plist_screentime.py # extracts DeviceActivity & ScreenTimeAgent plists
├─ smart_extract_plists.py # adaptive extractor (multi-strategy)
├─ plist_to_usage_csv.py # parses plists → usage_daily_from_plists.csv
├─ extract_knowledgec.py # extracts CoreDuet/KnowledgeC.db when present
└─ parse_knowledgec_usage.py # parses KnowledgeC.db → usage_daily_from_knowledgec.csv

Typical workflow

python decrypt_manifest.py
python quick_post_backup_probe.py
python smart_extract_plists.py
python plist_to_usage_csv.py

Expected outputs

decrypted_output/
├─ Manifest_decrypted.db
├─ screentime_plists/
│ ├─ DeviceActivity.plist
│ ├─ ScreenTimeAgent.plist
│ └─ usage_daily_from_plists.csv
└─ knowledgec/
└─ KnowledgeC.db → usage_daily_from_knowledgec.csv

If only plists exist, the CSV may be empty (settings-only snapshot).
Once KnowledgeC.db appears, the parser aggregates daily app-level usage.

Dependencies

python -m pip install iphone-backup-decrypt==0.9.0 pycryptodome

🔷 Modeling and Explainability
Notebook Focus Output
01_feature_engineering.ipynb Daily feature aggregation 97 features (27 engineered)
02_model_training.ipynb Baselines (Naïve, MA, LogReg) + LSTM M1–M3 best_model.tflite + metrics
03_shap_analysis.ipynb SHAP drift + top-5 features shap_top5_features.csv
04_rule_based_baseline.ipynb Clinical heuristic baseline Rule-based comparison

Temporal CV: 6 folds (4 m train / 2 m val)
Metrics: F1-macro/weighted, AUROC-OvR, Balanced ACC, Cohen κ, McNemar p.

🧠 Ethics and Data Protection

No identifiable data committed.

Raw exports encrypted locally.

/docs contains consent forms and data management plan.

GDPR + NCI Ethics compliance with anonymisation.

Phases

Self-data (Apple Health, Amazfit, Helio Ring).

Future opt-in collection from family/friends with consent.

🧮 Version Control and Tagging
Tag Description
v2.0-pre-ethics ETL + docs before ethics approval
v2.1-ethics-approved Ethics approved revision
v2.2-modeling-complete Final models and SHAP results
v2.3-final-report CA3 / dissertation submission
🧭 Execution Timeline (Oct → Jan)
Week Milestone
W1 – Oct 2025 KnowledgeC integration + ETL update
W2 – Nov 2025 Model retraining + LSTM drift analysis
W3 – Dec 2025 Final LaTeX report + appendices
Jan 2026 Viva defense and repository archival
🌐 Citation

Teixeira, R. M. (2025). N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2).
National College of Ireland. GitHub repository: https://github.com/rodrigomarques/practicum2-nof1-adhd-bd

🔒 License

Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).
Reuse allowed for academic purposes with proper credit.

📞 Contact

Rodrigo Marques Teixeira – MSc AI for Business (NCI)
Supervisor: Dr. Agatha Mattos
Email: x24130664@student.ncirl.ie

Version: v2.0-pre-ethics  Last updated: October 2025
