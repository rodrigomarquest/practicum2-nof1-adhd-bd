#!/usr/bin/env python
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

def _read_if(p: Path):
    try:
        if p.exists():
            df = pd.read_csv(p)
            if "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
    except Exception:
        pass
    return None

def _merge_daily(snapshot: Path) -> pd.DataFrame:
    names = [
        "sleep_daily.csv",
        "cardio_daily.csv",
        "activity_daily.csv",
        "health_usage_daily.csv",     # screentime
        "features_cardiovascular.csv" # se existir
    ]
    dfs = []
    for n in names:
        df = _read_if(snapshot / n)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["date"])
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="date", how="outer", suffixes=("", "_dup"))
        dup = [c for c in out.columns if c.endswith("_dup")]
        if dup:
            out.drop(columns=dup, inplace=True, errors="ignore")
    return out.sort_values("date").reset_index(drop=True)

def _build_segments(snapshot: Path) -> pd.DataFrame:
    vre = snapshot / "version_log_enriched.csv"
    vrw = snapshot / "version_raw.csv"
    if vre.exists():
        seg = pd.read_csv(vre)
        if {"start","end","segment_id"} <= set(seg.columns):
            seg["start"] = pd.to_datetime(seg["start"])
            seg["end"]   = pd.to_datetime(seg["end"])
            rows = []
            for _, r in seg.iterrows():
                if pd.isna(r["start"]) or pd.isna(r["end"]):
                    continue
                for d in pd.date_range(r["start"], r["end"], freq="D"):
                    rows.append({"date": d.date(), "segment_id": int(r["segment_id"])})
            return pd.DataFrame(rows)
    if vrw.exists():
        v = pd.read_csv(vrw)
        if "date" in v.columns:
            v = v.copy()
            v["date"] = pd.to_datetime(v["date"]).dt.date
            keys = [c for c in ["ios_version","watch_model","watch_fw","tz_name","source_name"] if c in v.columns]
            sig = v[keys].astype(str).agg(" | ".join, axis=1) if keys else pd.Series(["__X__"]*len(v))
            change = sig.ne(sig.shift(fill_value="__START__")).astype(int).cumsum()
            return pd.DataFrame({"date": v["date"], "segment_id": change})
    return pd.DataFrame(columns=["date","segment_id"])

def _zscore_by_segment(df: pd.DataFrame, seg: pd.DataFrame):
    if df.empty:
        return df.assign(segment_id=pd.NA), []
    w = df.copy()
    w = w.merge(seg, on="date", how="left") if not seg.empty else w.assign(segment_id=pd.NA)
    num = [c for c in w.select_dtypes(include=[np.number]).columns if c != "segment_id"]
    zcols = []
    def _z(x):
        mu = x.mean(); sd = x.std(ddof=0)
        if sd is None or sd == 0 or np.isnan(sd): return pd.Series([np.nan]*len(x), index=x.index)
        return (x - mu) / sd
    if "segment_id" in w.columns:
        for c in num:
            zc = f"{c}__z"
            w[zc] = w.groupby("segment_id", dropna=False)[c].transform(_z)
            zcols.append(zc)
    return w, zcols

def _write_csv(df: pd.DataFrame, p: Path):
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)

def _sha256(path: Path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def run(snapshot_dir: Path):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base = _merge_daily(snapshot_dir)
    seg  = _build_segments(snapshot_dir)
    feat, zcols = _zscore_by_segment(base, seg)

    out_csv = snapshot_dir / "features_daily_updated.csv"
    _write_csv(feat, out_csv)

    manifest = {
        "type":"features",
        "snapshot_dir": str(snapshot_dir),
        "inputs": {
            "sources": [n for n in ["sleep_daily.csv","cardio_daily.csv","activity_daily.csv","health_usage_daily.csv","features_cardiovascular.csv"] if (snapshot_dir/n).exists()],
            "segments": "version_log_enriched.csv" if (snapshot_dir/"version_log_enriched.csv").exists() else "version_raw.csv" if (snapshot_dir/"version_raw.csv").exists() else None
        },
        "outputs": {"features_daily_updated.csv": _sha256(out_csv), "zscore_columns": zcols}
    }
    with open(snapshot_dir/"features_manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # QC
    dfu = feat
    n_rows = len(dfu)
    date_min = str(min(dfu["date"])) if n_rows else None
    date_max = str(max(dfu["date"])) if n_rows else None
    seg_counts = (dfu["segment_id"].value_counts(dropna=False)
                  .rename_axis("segment_id").reset_index(name="rows")) if "segment_id" in dfu.columns else pd.DataFrame(columns=["segment_id","rows"])

    def _nan_ratio(s): return float(s.isna().mean()*100.0) if len(s) else 0.0
    nan_pct = pd.DataFrame({"column": dfu.columns, "nan_pct": [_nan_ratio(dfu[c]) for c in dfu.columns]}).sort_values("nan_pct", ascending=False)

    zcols = [c for c in dfu.columns if c.endswith("__z")]
    def _out_ratio(s): 
        try: return float((s.abs()>3).mean()*100.0)
        except: return 0.0
    outlier_pct = pd.DataFrame({"column": zcols, "outlier_abs_gt3_pct": [_out_ratio(dfu[c]) for c in zcols]}).sort_values("outlier_abs_gt3_pct", ascending=False) if zcols else pd.DataFrame(columns=["column","outlier_abs_gt3_pct"])

    parts = []
    parts.append(pd.DataFrame([{"section":"basic","n_rows":n_rows,"date_min":date_min,"date_max":date_max}]))
    if not seg_counts.empty:
        sc = seg_counts.copy(); sc.insert(0,"section","segment_rows"); parts.append(sc)
    if not nan_pct.empty:
        nn = nan_pct.copy(); nn.insert(0,"section","nan_pct"); parts.append(nn)
    if not outlier_pct.empty:
        oo = outlier_pct.copy(); oo.insert(0,"section","z_outliers"); parts.append(oo)
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _write_csv(summary, snapshot_dir/"etl_qc_summary.csv")

    lines = []
    lines.append(f"# ETL Report — {snapshot_dir.name}\n")
    lines.append(f"- Rows: **{n_rows}**, Date range: **{date_min} → {date_max}**")
    if not seg_counts.empty:
        top_seg = seg_counts.sort_values("rows", ascending=False).head(5)
        seg_str = ", ".join([f"S{int(r.segment_id)}: {int(r.rows)}" if pd.notna(r.segment_id) else f"NA: {int(r.rows)}" for _, r in top_seg.iterrows()])
        lines.append(f"- Top segments by rows: {seg_str}")
    if not nan_pct.empty:
        lines.append("\n## Highest NaN columns (top 10)\n")
        for _, r in nan_pct.head(10).iterrows():
            lines.append(f"- `{r.column}` — {r.nan_pct:.2f}% NaN")
    if not outlier_pct.empty:
        lines.append("\n## Z-score outliers |z|>3 (top 10)\n")
        for _, r in outlier_pct.head(10).iterrows():
            lines.append(f"- `{r.column}` — {r.outlier_abs_gt3_pct:.2f}% rows")
    with open(snapshot_dir/"etl_report.md","w",encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    snapdir = Path(f"data_ai/{args.participant}/snapshots/{args.snapshot}")
    run(snapdir)
    print("✅ features_daily_updated.csv + etl_qc_summary.csv + etl_report.md gerados em:", snapdir)
