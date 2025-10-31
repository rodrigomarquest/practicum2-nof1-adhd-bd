# Kaggle – 02_model_training.py (N-of-1 ADHD+BD)
# Author: Rodrigo Marques Teixeira • Co-author: GPT-5 Thinking
# Env: Python 3.10 • TensorFlow 2.15+ • scikit-learn • xgboost • shap • numpy • pandas • scipy • matplotlib
# Goal: Baselines + LSTM (M1/M2/M3) with 6-fold temporal CV (4m train / 2m val), SHAP/Drift, TFLite export.
# -------------------------------------------------------------------------------------
# Usage (Kaggle):
# 1) Upload/attach the dataset folder containing: features_daily_updated.csv, version_log_enriched.csv
# 2) Open a new Kaggle Notebook (GPU T4), add this file as a script, and run the cells (or copy/paste blocks).
# 3) Configure the paths below in CONFIG.
# -------------------------------------------------------------------------------------

# =========================
# HOW TO ENABLE GPU (KAGGLE)
# =========================
# • Click on the ⚙️ icon in the right sidebar (Settings)
# • Under 'Accelerator', choose 'GPU (T4)'
# • Save and rerun your notebook
# If you do not see the setting, click 'Save Version → Advanced → Accelerator: GPU'
# -------------------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

import shap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================
# CONFIGURATION
# ============================
@dataclass
class CONFIG:
    DATA_DIR: str = "/kaggle/input/nof1-data-p000001"  # dataset path
    FEATURES_FILE: str = "features_daily_updated.csv"  # <- updated filename
    VERSION_LOG_FILE: str = "version_log_enriched.csv"
    DATE_COL: str = "date"
    LABEL_COL: str = "label"
    SEGMENT_COL: str = "segment_id"
    WINDOW: int = 7
    N_FOLDS: int = 6
    TRAIN_MONTHS: int = 4
    VAL_MONTHS: int = 2
    RESULTS_DIR: str = "./artifacts"

CFG = CONFIG()
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

print("📦 Kaggle Run – N-of-1 ADHD+BD Predictive Modelling")
print("🎯 Objective: Baselines + LSTM with temporal CV, SHAP & Drift • Export TFLite")

# ===================================
# DATA LOADING
# ===================================
def load_data(cfg: CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fpath = os.path.join(cfg.DATA_DIR, cfg.FEATURES_FILE)
    vpath = os.path.join(cfg.DATA_DIR, cfg.VERSION_LOG_FILE)
    print(f"➡️ Loading data from: {fpath}")
    df = pd.read_csv(fpath)
    ver = pd.read_csv(vpath)
    if cfg.DATE_COL in df.columns:
        df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])
        df = df.sort_values(cfg.DATE_COL).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df, ver

# ===================================
# MAIN ENTRY
# ===================================
def main():
    df, ver = load_data(CFG)
    print(df.head())
    print("✅ Data loaded successfully.")

if __name__ == "__main__":
    main()
