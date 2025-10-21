#!/usr/bin/env bash
# ======================================================
# install_cardio_min.sh
# Instala implementação funcional mínima do ETL Cardiovascular
# (Apple + Zepp loaders, _join e features). Faz backup se o
# arquivo já existir: cria *.bak.YYYYmmddHHMMSS
# ======================================================
set -euo pipefail

ts="$(date +%Y%m%d%H%M%S)"

backup_and_write() {
  local file="$1"; shift
  local dir; dir="$(dirname "$file")"
  mkdir -p "$dir"
  if [ -f "$file" ]; then
    cp -f "$file" "$file.bak.$ts"
    echo "BACKUP: $file -> $file.bak.$ts"
  fi
  cat > "$file" <<'PYCODE'
$CONTENT$
PYCODE
  echo "WROTE:  $file"
}

# -----------------------------
# apple/loader.py
# -----------------------------
python make_scripts/create_cardio_apple_loader.py

# -----------------------------
# zepp/loader.py
# -----------------------------
python make_scripts/create_cardio_zepp_loader.py

# -----------------------------
# _join/join.py
# -----------------------------
python make_scripts/create_cardio_join.py

# -----------------------------
# cardio_features.py
# -----------------------------
python make_scripts/create_cardio_features.py

    df = hr.copy()
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["date","segment_id",
                                     "cv_hr_cosinor_mesor","cv_hr_cosinor_amp","cv_hr_cosinor_acrophase"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.dropna(subset=["timestamp","bpm"]).sort_values("timestamp").set_index("timestamp")

    rows = []
    for d, g in df.groupby("date"):
        cov = g["bpm"].notna().mean() * 100.0
        if cov < 20:  # mínimo p/ cosinor
            rows.append({"date": d, "cv_hr_cosinor_mesor": np.nan,
                         "cv_hr_cosinor_amp": np.nan, "cv_hr_cosinor_acrophase": np.nan})
            continue
        idx = g.index.tz_localize(None)
        minutes = (idx.hour * 60 + idx.minute).astype(float).values
        omega = 2.0 * math.pi / (24.0 * 60.0)
        X = np.column_stack([np.ones_like(minutes), np.cos(omega * minutes), np.sin(omega * minutes)])
        y = g["bpm"].astype(float).values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            mesor = float(beta[0])
            amp = float(math.sqrt(beta[1]**2 + beta[2]**2))
            acroph = float(math.atan2(-beta[2], beta[1]))
        except Exception:
            mesor = amp = acroph = np.nan
        rows.append({"date": d, "cv_hr_cosinor_mesor": mesor,
                     "cv_hr_cosinor_amp": amp, "cv_hr_cosinor_acrophase": acroph})
    out = pd.DataFrame(rows)

    if segments is not None and not segments.empty:
        seg = segments.copy()
        seg["date"] = pd.to_datetime(seg["date"]).dt.date
        out = out.merge(seg[["date","segment_id"]], on="date", how="left")
    else:
        out["segment_id"] = pd.NA

    return out

def merge_cardio_outputs(hr_feats: pd.DataFrame,
                         hrv_feats: pd.DataFrame,
                         circ_feats: pd.DataFrame) -> pd.DataFrame:
    def _safe(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["date","segment_id"])
        return df
    out = _safe(hr_feats)
    for df in (_safe(hrv_feats), _safe(circ_feats)):
        out = out.merge(df, on=["date","segment_id"], how="outer")
    out = out.sort_values("date")
    return out

def update_features_daily(features_cardio: pd.DataFrame, features_daily_path: str) -> str:
    """Merge incremental com features_daily.csv (se existir).
    Chaves: date (+ segment_id, se existir em ambos).
    """
    # caminho de saída = mesmo diretório
    out_dir = os.path.dirname(features_daily_path)
    out_path = os.path.join(out_dir, "features_daily_updated.csv")

    base = read_csv_if_exists(features_daily_path)
    if base is None or base.empty:
        features_cardio.to_csv(out_path, index=False)
        return out_path

    base = base.copy()
    if "date" in base.columns:
        base["date"] = pd.to_datetime(base["date"]).dt.date

    keys = ["date"]
    if "segment_id" in base.columns and "segment_id" in features_cardio.columns:
        keys.append("segment_id")

    merged = base.merge(features_cardio, on=keys, how="left")
    merged.to_csv(out_path, index=False)
    return out_path
PY
)
backup_and_write "etl_modules/cardiovascular/cardio_features.py" "$CONTENT"

echo "✅ Cardiovascular minimal implementation installed."
echo "→ Próximo: rode o stage cardio no seu snapshot:"
echo "   python etl_pipeline.py --stage cardio --input data_ai/P000001/snapshots/2025-09-29 --out data_ai/P000001/snapshots/2025-09-29"
