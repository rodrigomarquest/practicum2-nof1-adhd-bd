from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Iterable

def _read_csv_safe(p: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Lê CSV garantindo dtype consistente para 'date' e filtrando colunas quando útil."""
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"date": "string"})
    if usecols:
        keep = [c for c in usecols if c in df.columns]
        if keep:
            df = df[keep]
    return df

def _merge_on_date(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge progressivo por 'date' (outer), ordena e de-duplica."""
    if not dfs:
        return pd.DataFrame(columns=["date"])
    base = dfs[0]
    for d in dfs[1:]:
        if d is None or d.empty:
            continue
        base = base.merge(d, on="date", how="outer")
    base = base.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return base
def load_daily(outdir: Path) -> pd.DataFrame:
    outdir = Path(outdir)  # passe .../zepp_processed/_latest
    dfs = []
    for name, cols in [
        ("zepp_emotion_daily.csv", ["date","emotion_score"]),
        ("zepp_stress_daily.csv",  ["date","stress_score"]),
        ("zepp_temp_daily.csv",    ["date","skin_temp_c"]),
        ("zepp_sleep_daily.csv",   ["date","zepp_sleep_minutes"]),
        ("zepp_hr_daily.csv",      ["date","zepp_hr_mean","zepp_hr_median","zepp_hr_p95"]),
        ("zepp_health_daily.csv",  ["date","zepp_spo2_mean","zepp_temp_mean","zepp_stress_mean"]),
        ("zepp_body_daily.csv",    ["date","zepp_weight_kg"]),
        ("zepp_activity_daily.csv",["date","zepp_steps","zepp_calories"]),
    ]:
        p = outdir / name
        if p.exists():
            df = pd.read_csv(p, dtype={"date":"string"})
            keep = [c for c in cols if c in df.columns]
            dfs.append(df[keep])
    if not dfs:
        return pd.DataFrame(columns=["date"])
    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df, on="date", how="outer")
    base = base.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return base

def join_into_features(
    features_daily_path: str | Path,
    zepp_daily_dir: str | Path,
    version_log_path: str | Path | None = None,
) -> bool:
    """
    Faz left-join das colunas Zepp em features_daily.csv por 'date'.
    - features_daily_path: caminho do CSV principal do snapshot.
    - zepp_daily_dir: diretório com os artefatos do parse_zepp_export.py.
    - version_log_path: (opcional) version_log_enriched.csv para anotar colunas adicionadas.
    """
    fpath = Path(features_daily_path)
    zdir = Path(zepp_daily_dir)

    if not fpath.exists():
        return False

    # carrega Zepp
    z = load_daily(zdir)
    if z is None or z.empty or "date" not in z.columns:
        return False

    # carrega features do snapshot
    f = pd.read_csv(fpath, dtype={"date": "string"})
    if "date" not in f.columns:
        return False

    # evita colisão de nomes: se já houver colunas iguais, mantemos as novas como sufixo "_z"
    common_cols = [c for c in z.columns if c != "date" and c in f.columns]
    if common_cols:
        z = z.rename(columns={c: f"{c}_z" for c in common_cols})

    # merge
    out = f.merge(z, on="date", how="left")
    out.to_csv(fpath, index=False)

    # version log opcional
    if version_log_path:
        vpath = Path(version_log_path)
        try:
            if vpath.exists():
                vl = pd.read_csv(vpath)
                if not vl.empty:
                    last = vl.index.max()
                    added_cols = [c for c in out.columns if c not in f.columns]
                    vl.loc[last, "zepp_joined"] = True
                    vl.loc[last, "zepp_cols_added"] = ",".join(added_cols)
                    vl.to_csv(vpath, index=False)
        except Exception:
            # não bloquear o pipeline por causa do version log
            pass

    return True
