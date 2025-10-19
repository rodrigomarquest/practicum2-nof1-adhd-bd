import json, os
from pathlib import Path
from uuid import uuid4

NB_PATH = Path("notebooks/03_eda_cardio_plus.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)

def code(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.rstrip("\n") + "\n",
        "id": uuid4().hex[:8],
    }

def md(src: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.rstrip("\n") + "\n",
        "id": uuid4().hex[:8],
    }

cells = []

# 0) Título
cells.append(md("# 03 — EDA Cardiovascular (Apple + Zepp)\n\n"
                "Notebook com path robusto, gráficos e salvamento em `eda_outputs/`."))

# 1) Imports e detecção de REPO_ROOT
cells.append(code(r'''
import os, sys, re, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Detecta REPO_ROOT (sobe até achar 'data_ai' ou '.git') ----
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    tried = 0
    while cur and tried < 10:
        if (cur / "data_ai").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
        tried += 1
    # fallback: cwd
    return start.resolve()

CWD = Path.cwd()
REPO_ROOT = find_repo_root(CWD)
print("REPO_ROOT =", REPO_ROOT)

# Parametrização por env
PID  = os.getenv("PID", "P000001")
SNAP = os.getenv("SNAP", "2025-09-29")

SNAPDIR = REPO_ROOT / "data_ai" / PID / "snapshots" / SNAP
OUTDIR  = SNAPDIR / "eda_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("SNAPDIR =", SNAPDIR)
print("OUTDIR  =", OUTDIR)
'''))

# 2) Checagem de arquivos esperados
cells.append(code(r'''
files = [
    SNAPDIR/"features_cardiovascular.csv",
    SNAPDIR/"features_daily_updated.csv",
    SNAPDIR/"per-metric"/"apple_heart_rate.csv",
    SNAPDIR/"per-metric"/"apple_hrv_sdnn.csv",
    SNAPDIR/"per-metric"/"apple_sleep_intervals.csv",
]
print("== Arquivos esperados ==")
for f in files:
    print(f"- {f.relative_to(REPO_ROOT)}  exists=", f.exists(), " size=", (f.stat().st_size if f.exists() else 0))
'''))

# 3) Carregamento robusto do features_cardiovascular
cells.append(code(r'''
fc_path = SNAPDIR/"features_cardiovascular.csv"
if not fc_path.exists():
    print("⚠️  features_cardiovascular.csv não encontrado em", fc_path)
    print("Sugestão: rode o stage cardio novamente, ex.:")
    print("  python etl_pipeline.py cardio --participant", PID, "--snapshot", SNAP, "--zepp_dir <opcional>")
    fc = None
else:
    fc = pd.read_csv(fc_path, parse_dates=["date"])
    # normaliza 'date' para date puro
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    # se veio timezone-aware, remova tz
    if hasattr(fc["date"].dt, "tz_localize"):
        try:
            fc["date"] = fc["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
    fc["date"] = fc["date"].dt.date
    print("features_cardiovascular shape:", fc.shape)
    display(fc.head(3))
'''))

# 4) Gráfico HR mean diário
cells.append(code(r'''
if fc is not None and not fc.empty and "hr_mean" in fc.columns:
    df = fc.dropna(subset=["date"]).copy()
    df = df.sort_values("date")
    s_date = pd.to_datetime(df["date"])
    hr = pd.to_numeric(df["hr_mean"], errors="coerce")

    # Tendência 7d (média móvel, valores suficientes)
    trend = pd.Series(hr).rolling(window=7, min_periods=3, center=False).mean()

    plt.figure(figsize=(12,4))
    plt.plot(s_date, hr, label="HR mean (daily)")
    plt.plot(s_date, trend, label="7d moving avg")
    plt.title("Daily Heart Rate (mean) — Apple/Zepp")
    plt.xlabel("Date")
    plt.ylabel("bpm")
    plt.legend()
    plt.tight_layout()

    out_png = OUTDIR/"hr_mean_daily.png"
    plt.savefig(out_png)
    print("Saved:", out_png)
    plt.show()
else:
    print("ℹ️  Sem coluna 'hr_mean' — gráfico HR mean diário não gerado.")
'''))

# 5) Gráfico HRV mean diário
cells.append(code(r'''
if fc is not None and not fc.empty and "hrv_ms_mean" in fc.columns:
    df = fc.dropna(subset=["date"]).copy()
    df = df.sort_values("date")
    s_date = pd.to_datetime(df["date"])
    hrv = pd.to_numeric(df["hrv_ms_mean"], errors="coerce")

    trend = pd.Series(hrv).rolling(window=7, min_periods=3, center=False).mean()

    plt.figure(figsize=(12,4))
    plt.plot(s_date, hrv, label="HRV (SDNN) mean (daily)")
    plt.plot(s_date, trend, label="7d moving avg")
    plt.title("Daily HRV SDNN (mean) — Apple/Zepp")
    plt.xlabel("Date")
    plt.ylabel("ms")
    plt.legend()
    plt.tight_layout()

    out_png = OUTDIR/"hrv_mean_daily.png"
    plt.savefig(out_png)
    print("Saved:", out_png)
    plt.show()
else:
    print("ℹ️  Sem coluna 'hrv_ms_mean' — gráfico HRV mean diário não gerado.")
'''))

# 6) Histogramas (HR e HRV)
cells.append(code(r'''
if fc is not None and not fc.empty:
    if "hr_mean" in fc.columns:
        v = pd.to_numeric(fc["hr_mean"], errors="coerce").dropna()
        if not v.empty:
            plt.figure(figsize=(7,4))
            plt.hist(v, bins=40)
            plt.title("Histogram — HR daily mean")
            plt.xlabel("bpm"); plt.ylabel("count"); plt.tight_layout()
            out_png = OUTDIR/"hist_hr_mean.png"
            plt.savefig(out_png); print("Saved:", out_png)
            plt.show()
    if "hrv_ms_mean" in fc.columns:
        v = pd.to_numeric(fc["hrv_ms_mean"], errors="coerce").dropna()
        if not v.empty:
            plt.figure(figsize=(7,4))
            plt.hist(v, bins=40)
            plt.title("Histogram — HRV SDNN daily mean")
            plt.xlabel("ms"); plt.ylabel("count"); plt.tight_layout()
            out_png = OUTDIR/"hist_hrv_mean.png"
            plt.savefig(out_png); print("Saved:", out_png)
            plt.show()
'''))

# 7) Conclusão
cells.append(md("## Conclusões rápidas\n"
                "- Gráficos salvos em `eda_outputs/`.\n"
                "- Se não apareceram gráficos, verifique se `features_cardiovascular.csv` foi gerado."))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("CREATED/UPDATED:", NB_PATH)
