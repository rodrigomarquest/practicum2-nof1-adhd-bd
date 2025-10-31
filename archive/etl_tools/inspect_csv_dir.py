# etl_tools/inspect_csv_dir.py
from __future__ import annotations
import argparse, io, csv, codecs, re
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

DATE_CAND_SUBSTR = [
    "timestamp", "time", "date", "datetime", "starttime", "endtime",
    "measuretime", "start_time", "end_time", "date_time"
]
ISO_TRY = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
    "%Y/%m/%d",
]

def sniff_sep(sample: str) -> Optional[str]:
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return None

def read_csv_robust_text(text: str) -> pd.DataFrame:
    sep = sniff_sep(text.splitlines()[0] if text else ",")
    # 1) tenta engine C (rápido)
    try:
        return pd.read_csv(io.StringIO(text), sep=sep if sep else None,
                           engine="c", on_bad_lines="error", low_memory=False)
    except Exception:
        pass
    # 2) fallback: engine python (tolerante, sem low_memory)
    try:
        return pd.read_csv(io.StringIO(text), sep=sep if sep else None,
                           engine="c", on_bad_lines="error", low_memory=False)
    except Exception:
        # 3) último recurso: tenta como TSV
        return pd.read_csv(io.StringIO(text), sep="\t",
                           engine="python", on_bad_lines="skip")


def read_csv_robust(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    raw = path.read_bytes()
    text = codecs.decode(raw, "utf-8", errors="replace")
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
    sep = sniff_sep(text.splitlines()[0] if text else ",")
    df = read_csv_robust_text(text)
    return df, sep

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Tenta converter série (string/numérica) para datetime UTC naive (sem tz)."""
    # números? (epoch)
    if pd.api.types.is_numeric_dtype(s):
        m = pd.Series(s, copy=False).astype("float64")
        if np.isfinite(m).any():
            if (m > 1e12).mean() > 0.5:
                return pd.to_datetime(m, unit="ms", errors="coerce", utc=True).dt.tz_convert(None)
            if (m > 1e9).mean() > 0.5:
                return pd.to_datetime(m, unit="s", errors="coerce", utc=True).dt.tz_convert(None)
    # strings? formatos comuns primeiro
    if pd.api.types.is_string_dtype(s):
        for fmt in ISO_TRY:
            try:
                return pd.to_datetime(s, format=fmt, errors="raise", utc=True).dt.tz_convert(None)
            except Exception:
                pass
    # fallback flexível (silencia o warning do pandas)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format*", category=UserWarning)
        return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    cols = list(df.columns)
    # 1) prioridade por nome
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in DATE_CAND_SUBSTR):
            parsed = parse_datetime_series(df[c])
            ok = parsed.notna().mean()
            if ok >= 0.2:  # pelo menos 20% parsável
                return c
    # 2) fallback: varre primeiras N colunas e mede parsabilidade
    best_c, best_ok = None, 0.0
    for c in cols[: min(8, len(cols))]:
        parsed = parse_datetime_series(df[c])
        ok = parsed.notna().mean()
        if ok > best_ok:
            best_ok, best_c = ok, c
    return best_c if best_ok >= 0.2 else None

def summarize_csv(path: Path, base_dir: Path) -> dict:
    rel = path.relative_to(base_dir).as_posix()
    try:
        df, sep = read_csv_robust(path)
        rows, cols = df.shape
        date_col = detect_date_column(df)
        dmin = dmax = None
        if date_col is not None:
            dt = parse_datetime_series(df[date_col])
            if dt.notna().any():
                dmin = str(dt.min().date())
                dmax = str(dt.max().date())
        return {
            "file": rel,
            "rows": rows,
            "cols": cols,
            "separator": sep or "",
            "date_col": date_col or "",
            "date_min": dmin or "",
            "date_max": dmax or "",
            "columns": ",".join(df.columns.astype(str).tolist())
        }, df
    except Exception as e:
        return {"file": rel, "rows": "", "cols": "", "separator": "",
                "date_col": "", "date_min": "", "date_max": "",
                "columns": "", "error": repr(e)}, pd.DataFrame()

def main():
    ap = argparse.ArgumentParser("inspect-csv-dir")
    ap.add_argument("--dir", required=True, help="Diretório base para varrer CSVs")
    ap.add_argument("--out", required=True, help="Caminho do Excel de saída (.xlsx)")
    ap.add_argument("--pattern", default="**/*.csv", help="Glob relativo (default: **/*.csv)")
    ap.add_argument("--sample", type=int, default=5, help="Linhas de amostra por arquivo (default: 5)")
    args = ap.parse_args()

    base = Path(args.dir)
    out_xlsx = Path(args.out)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # 1) montar lista de arquivos
    files = sorted(base.glob(args.pattern))

    # 2) FILTRO — ignora nossos próprios relatórios e quaisquer CSV que comecem com "_"
    #    (evita loops e lixo no summary)
    summary_name = out_xlsx.with_suffix(".summary.csv").name        # e.g. "_csv_report.summary.csv"
    report_stem  = out_xlsx.stem.lower()                            # e.g. "_csv_report"
    filtered = []
    for p in files:
        name = p.name
        if name.startswith("_"):            # ignora CSVs auxiliares (_*.csv)
            if name == summary_name:        # nosso summary especificamente
                continue
            if report_stem in name.lower(): # qualquer arquivo contendo "csv_report"
                continue
        filtered.append(p)
    files = filtered

    summaries = []
    columns_rows = []
    samples = []

    for p in files:
        s, df = summarize_csv(p, base)
        summaries.append(s)
        if not df.empty:
            # inventário de colunas
            for c in df.columns:
                colname = "file_original" if c == "file" else c
                columns_rows.append({
                    "file": s["file"],
                    "column": colname,
                    "non_null": int(pd.Series(df[c]).notna().sum())
                })
            # amostras
            head = df.head(args.sample).copy()
            if "file" in head.columns:
                head = head.rename(columns={"file": "file_original"})
            head.insert(0, "file", s["file"])
            samples.append(head)

    # DataFrames de saída
    df_sum  = pd.DataFrame(summaries)
    df_cols = pd.DataFrame(columns_rows) if columns_rows else pd.DataFrame(columns=["file","column","non_null"])
    df_samp = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    # salva CSV de resumo lado a lado
    df_sum.to_csv(out_xlsx.with_suffix(".summary.csv"), index=False, encoding="utf-8")

    # salva Excel com abas
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        df_sum.to_excel(xl, sheet_name="summary", index=False)
        df_cols.to_excel(xl, sheet_name="columns", index=False)
        if not df_samp.empty:
            df_samp.to_excel(xl, sheet_name="samples", index=False)

    print(f"✅ CSV inspection written → {out_xlsx}")
    print(f"   Summary rows: {len(df_sum)}  | Columns rows: {len(df_cols)}  | Samples: {len(df_samp)}")

if __name__ == "__main__":
    main()
