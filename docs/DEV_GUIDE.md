# 🧭 Practicum2 – Developer Guide (v4-series)

**Project:** N-of-1 ADHD + Bipolar Disorder  
**Version:** v4.0.2 (2025-11)  
**Maintainer:** Rodrigo M. Teixeira  
**Repository layout & operational conventions for the MSc Practicum research project.**

---

## 1. 📁 Repository Structure (v4 Architecture)

practicum2-nof1-adhd-bd/
├── src/ # Canonical Python source (modularized for Kaggle & local)
│ ├── etl_pipeline.py # ETL entrypoint (local data ingestion)
│ ├── make_labels.py # Pseudo-label generation based on heuristic rules
│ ├── eda.py # Exploratory Data Analysis & QC pipeline
│ ├── models_nb2.py # Baseline models (rule-based, logistic, etc.)
│ ├── models_nb3.py # Deep learning (LSTM/CNN) models
│ ├── utils.py # Shared helpers
│ ├── nb_common/ # Unified helpers for Kaggle notebooks
│ │ ├── portable.py
│ │ ├── features.py
│ │ ├── io.py
│ │ └── tf_models.py
│ ├── domains/ # Domain-specific ETL modules
│ │ ├── cardiovascular/
│ │ ├── common/
│ │ ├── iphone_backup/
│ │ ├── zepp/
│ │ └── ...
│ └── tools/ # Internal research / audit utilities
│
├── notebooks/ # Research notebooks (EDA, ML, etc.)
│ ├── NB1_EDA_daily.ipynb
│ ├── NB2_Baselines_LSTM.ipynb
│ ├── NB3_DeepLearning.py
│ └── notebooks/outputs/
│
├── config/ # Configuration & metadata
│ ├── settings.yaml # Project defaults (PID, snapshot, paths)
│ ├── label_rules.yaml # Heuristic pseudo-label definitions
│ └── participants.yaml # Registered participants
│
├── requirements/ # Environment dependency sets
│ ├── base.txt
│ ├── dev.txt
│ ├── kaggle.txt
│ └── local.txt
│
├── provenance/ # Data provenance, version logs, integrity checks
│ ├── etl_provenance_report.csv
│ └── zepp_zip_filelist.tsv
│
├── docs/ # Academic deliverables and guides
│ ├── build/
│ ├── release_notes/
│ ├── ETL_EDA_MODELING_PLAN.md
│ ├── configuration_manual_full.tex
│ └── DEV_GUIDE.md
│
├── archive/ # Legacy scripts and historical backups (frozen)
│
├── dist/assets/ # Auto-generated artifacts (e.g., Kaggle zips)
│
├── Makefile # Clean, documented v4 Makefile
├── .gitignore # Ignore non-versioned data
├── .gitattributes # Control EOL, export rules for releases
└── LICENSE, README.md, CHANGELOG.md

yaml
Copiar código

---

## 2. ⚙️ Makefile Overview

The `Makefile` is simplified and grouped by purpose.

| Target                                                          | Description                                                            |
| --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **install-base / install-dev / install-kaggle / install-local** | Install dependencies for each environment.                             |
| **etl**                                                         | Run the unified ETL pipeline (`src.etl_pipeline`).                     |
| **labels**                                                      | Generate heuristic pseudo-labels (`src.make_labels`).                  |
| **qc**                                                          | Perform quick EDA/QC validation (`src.eda`).                           |
| **pack-kaggle**                                                 | Package a cleaned dataset snapshot for Kaggle.                         |
| **clean / clean-data / clean-provenance / clean-all**           | Progressive cleanup of temporary data, logs, and provenance artifacts. |
| **help**                                                        | Display the list of available targets.                                 |

> 💡 Run `make help` anytime for a quick overview.  
> The Makefile autodetects Windows vs Linux environments (`.venv/Scripts/python.exe` vs `.venv/bin/python`).

---

## 3. 🧩 Environment Management

- Always create or refresh your virtual environment with:
  ```bash
  python -m venv .venv
  make install-dev
  Use requirements/base.txt for reproducibility across machines.
  ```

The Kaggle runtime uses requirements/kaggle.txt — keep it lightweight (no system-level deps).

The Makefile installs with platform-specific Python automatically.

4. 🧱 Versioning & Branch Policy
   Branch Purpose
   main Stable, publication-ready deliverables (LaTeX, notebooks, ETL).
   v4-main Current active development branch for Practicum Part 2.
   archive/\* Historical branches from Practicum Part 1 (read-only).

Tagging convention:

v3.9.9-pristine → last pre-v4 snapshot.

v4.0.0 → initial refactor (clean src, config, Makefile).

v4.0.1 → layout hygiene and provenance cleanup.

v4.0.2 → environment centralization + developer documentation.

5. 🧰 Git Hygiene
   5.1 .gitignore – What not to version
   Used to keep local data and transient files out of the repository.

Category Example Rationale
Raw data data/etl/, data/ai/, decrypted_output/ Contains personal or high-volume data.
Cache & logs logs/, processed/, **pycache**/ Reproducible artifacts.
Models _.h5, _.tflite, \*.pkl Large binary outputs, not source.
IDE / OS noise .vscode/, .DS_Store, .venv/ Environment-specific.

“If it can be regenerated, it doesn’t belong in Git.”

5.2 .gitattributes – How versioned files behave
Controls EOL normalization, export rules, and diff behavior.

Feature Example Purpose
EOL normalization _.py text eol=lf Force LF on scripts for cross-platform use.
Binary diff control _.ipynb -text Avoid messy diffs for Jupyter notebooks.
Export hygiene data/etl/ export-ignore Exclude heavy data from git archive.
Provenance trim provenance/pip*freeze*\*.txt export-ignore Skip transient logs from release zips.

“.gitignore controls what enters the repo.
.gitattributes controls how it behaves once inside.”

5.3 Combined workflow
Stage .gitignore .gitattributes
Local dev Protects privacy, avoids noise. Normalizes EOL, diff, and merge.
CI / Kaggle Ensures lean workspace. Keeps builds clean and reproducible.
Release (zip/tar) No effect. Excludes local data, logs, and caches.

6. 🧪 Quality & Reproducibility
   Each dataset version (S1–S6) is isolated in data/etl/PXXXXXX/snapshots/DATE/.

The ETL normalizes, imputes, and z-scores within segment boundaries.

Reproducibility is tracked through provenance/etl_provenance_report.csv and pip_freeze_TIMESTAMP.txt.

Pseudo-labels are defined in config/label_rules.yaml, with a scientific basis for ADHD/BD signal proxies.

7. 🧾 Release Checklist (for tagging v4.x)
   Ensure make clean-all runs without error.

Run local ETL → make etl.

Generate pseudo-labels → make labels.

Package snapshot → make pack-kaggle.

Commit & tag:

bash
Copiar código
git add -A
git commit -m "release: v4.0.2 clean Makefile, centralized requirements, full docs"
git tag -a v4.0.2 -m "Stable foundation for Practicum Part 2 – N-of-1 modeling" 8. 🧘 Ethical & Compliance Notes
No identifiable participant data is versioned.

All physiological metrics are anonymized and aggregated.

Pseudo-labels follow published heuristics; any clinical inference must remain illustrative.

Ensure compliance with GDPR and NCI ethics guidelines before data sharing or publication.

9. 🪶 Credits
   Author: Rodrigo Marques Teixeira
   Supervisor: Dr. Agatha Mattos
   Institution: National College of Ireland – MSc in Artificial Intelligence for Business
   Collaborators: Claude (GitHub Copilot), ChatGPT (GPT-5), Kaggle environment

“Train models you can explain. Collect data you can defend.
Code as if your future self were reviewing your thesis.” — Mestre Yoda
