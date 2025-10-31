"""
etl_pipeline.py — Apple Health ETL (streaming) → daily CSVs + features_daily.csv

Converte Apple Health export.xml em tabelas diárias para:
- HR (Heart Rate), HRV SDNN, Sleep (horas dormidas), Temperature.
Gera version log por device/firmware, define segment_id, e (opcional) faz z-score
por segmento. Stream-parse via ElementTree.iterparse (memória constante).

USO (PowerShell/bash):
  python etl_pipeline.py PATH/TO/export.xml OUT_DIR ^
    --cutover-date 2024-03-11 --tz-before America/Sao_Paulo --tz-after Europe/Dublin

Saídas (OUT_DIR):
  health_hr_daily.csv
  health_hrv_sdnn_daily.csv
  health_sleep_daily.csv
  health_temp_daily.csv
  version_raw.csv
  version_log_enriched.csv     # segment_id, start/end, ios/watch/fw, tz_name, source_name
  features_daily.csv           # wide (date, segment_id, *_mean, *_std, n_*)
  etl_qc_summary.csv           # resumo básico

Notas:
- Screen Time NÃO vem no export.xml → pipeline separado.
- Emotion EXCLUÍDO.
- Sleep = horas dormidas (somando intervalos marcados como “Asleep”).
"""
import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import xml.etree.ElementTree as ET

from etl_modules.extract_screen_time import extract_apple_screen_time
from etl_modules.iphone_backup.iphone_backup import EncryptedBackup  # mantido por compat

import numpy as np
import pandas as pd
import json

# --- Multi-participant / multi-snapshot discovery ---
RAW_ROOT = Path("data_etl")
AI_ROOT  = Path("data_ai")
MANIFEST = Path("data_ai/DATA_MANIFEST.json")  # opcional; se existir, é priorizado

# Canonical snapshot id: YYYY-MM-DD. Accept legacy YYYYMMDD and convert.
_SNAP_ISO  = re.compile(r"^20\d{2}-[01]\d-[0-3]\d$")
_SNAP_8DIG = re.compile(r"^20\d{6}$")  # YYYYMMDD

def _canon_snap_id(name: str) -> str:
    """
    Return snapshot id as YYYY-MM-DD.
    Accepts 'YYYY-MM-DD' or legacy 'YYYYMMDD'; rejects ddmmyyyy to avoid ambiguity.
    """
    name = name.strip()
    if _SNAP_ISO.match(name):
        return name
    if _SNAP_8DIG.match(name):
        # YYYYMMDD -> YYYY-MM-DD
        return f"{name[0:4]}-{name[4:6]}-{name[6:8]}"
    raise ValueError(f"Invalid snapshot folder '{name}'. Expected YYYY-MM-DD (or YYYYMMDD legacy).")

def discover_etl_jobs():
    """
    Discover (participant_id, snapshot_id, xml_file).

    Accepts both:
      data_etl/<PID>/snapshots/<YYYY-MM-DD>/export.xml
      data_etl/<PID>/<YYYY-MM-DD>/export.xml

    Also accepts legacy <YYYYMMDD> and converts to ISO.
    """
    jobs = []
    for pdir in sorted(RAW_ROOT.glob("P*/")):
        pid = pdir.name

        # We probe two bases: with and without 'snapshots'
        for base in (pdir / "snapshots", pdir):
            if not base.exists():
                continue
            for sdir in sorted([d for d in base.iterdir() if d.is_dir()]):
                try:
                    snap_iso = _canon_snap_id(sdir.name)
                except ValueError:
                    continue  # skip non-date folders

                xml = sdir / "export.xml"
                if xml.exists():
                    jobs.append({"pid": pid, "snap": snap_iso, "xml": xml})
    return jobs

def ensure_ai_outdir(pid: str, snap: str) -> Path:
    snap_iso = _canon_snap_id(snap)
    outdir = AI_ROOT / pid / "snapshots" / snap_iso
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ---------- Timezone (zoneinfo, fallback pytz) ----------
try:
    from zoneinfo import ZoneInfo  # 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None
    try:
        import pytz  # type: ignore
    except Exception:
        pytz = None

def get_tz(tz_name: str):
    if ZoneInfo is not None:
        return ZoneInfo(tz_name)
    if 'pytz' in sys.modules or 'pytz' in globals():
        return pytz.timezone(tz_name)  # type: ignore
    raise RuntimeError("No timezone support. Use Python 3.9+ or install pytz.")

# ---------- HealthKit types ----------
TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRate": ("hr", "count/min"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("hrv_sdnn", "ms"),
    "HKCategoryTypeIdentifierSleepAnalysis": ("sleep", "category"),
    "HKQuantityTypeIdentifierBodyTemperature": ("temp", "degC"),
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": ("temp", "degC"),
    "HKQuantityTypeIdentifierBasalBodyTemperature": ("temp", "degC"),
}

# Sleep categories (string moderna e int legado) — contamos só "asleep"
ASLEEP_STR = {
    "HKCategoryValueSleepAnalysisAsleep",              # genérico
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}
ASLEEP_INT = {1, 3, 4, 5}

DEVICE_KEY_MAP = {"name":"name", "manufacturer":"manufacturer",
                  "model":"watch_model", "hardware":"watch_fw", "software":"ios_version"}

def fix_iso_tz(s: str) -> str:
    """Normaliza ISO da Apple p/ 'YYYY-MM-DD HH:MM:SS±HH:MM' ou +00:00."""
    if s.endswith("Z"):
        return s[:-1] + "+00:00"
    m = re.search(r"([+-]\d{2})(\d{2})$", s)  # '+0100' → '+01:00'
    if m:
        s = s[:m.start()] + f"{m.group(1)}:{m.group(2)}"
    return s

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(fix_iso_tz(s))

def parse_device_string(device_str: str):
    info = {}
    if not device_str:
        return info
    parts = [p.strip() for p in device_str.strip("<>").split(",")]
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            info[k.strip().lower()] = v.strip()
    # mapear chaves
    mapped = {}
    for k, v in info.items():
        if k in DEVICE_KEY_MAP:
            mapped[DEVICE_KEY_MAP[k]] = v
    return mapped

def make_tz_selector(cutover: date, tz_before: str, tz_after: str):
    tz_b = get_tz(tz_before)
    tz_a = get_tz(tz_after)
    def selector(dt_aware: datetime):
        # escolhe TZ pela data civil em UTC (determinístico)
        day_utc = dt_aware.astimezone(get_tz("UTC")).date()
        return tz_a if day_utc >= cutover else tz_b
    return selector

def iter_health_records(xml_path: Path, tz_selector):
    """Yield: (type_id, value, start_local, end_local, unit, device_dict, tz_name, source_version, source_name)."""
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if elem.tag != "Record":
            continue
        type_id = elem.attrib.get("type")
        if type_id not in TYPE_MAP:
            elem.clear(); continue

        s = elem.attrib.get("startDate")
        e = elem.attrib.get("endDate")
        if not s:
            elem.clear(); continue

        sdt = parse_dt(s)
        edt = parse_dt(e) if e else None

        tz = tz_selector(sdt)
        s_local = sdt.astimezone(tz)
        e_local = edt.astimezone(tz) if edt else None

        unit = elem.attrib.get("unit")
        val_raw = elem.attrib.get("value")

        if type_id == "HKCategoryTypeIdentifierSleepAnalysis":
            value = val_raw  # string/int; tratamos depois
        else:
            try:
                value = float(val_raw) if val_raw is not None else None
            except Exception:
                value = None

        device = parse_device_string(elem.attrib.get("device", ""))
        tz_name = getattr(tz, 'key', getattr(tz, 'zone', str(tz)))
        source_version = elem.attrib.get("sourceVersion", "")
        source_name    = elem.attrib.get("sourceName", "")

        yield type_id, value, s_local, e_local, unit, device, tz_name, source_version, source_name
        elem.clear()

def aggregate_daily(records):
    """Agraga métricas/dia e constroi version log bruto."""
    daily = defaultdict(list)
    version_rows = []

    for (type_id, value, s_local, e_local, unit,
         device, tz_name, src_ver, src_name) in records:

        metric, _ = TYPE_MAP[type_id]
        day = s_local.date()

        if type_id == "HKCategoryTypeIdentifierSleepAnalysis":
            # soma horas dormidas (apenas categorias Asleep)
            cat = None
            if isinstance(value, str):
                cat = value.strip()
            else:
                try:
                    cat = int(value) if value is not None else None
                except Exception:
                    cat = None
            if e_local is None:
                continue
            dur_h = (e_local - s_local).total_seconds() / 3600.0
            is_asleep = (isinstance(cat, str) and cat in ASLEEP_STR) or (isinstance(cat, int) and cat in ASLEEP_INT)
            if is_asleep:
                daily[(metric, day)].append(dur_h)
        else:
            if value is not None:
                daily[(metric, day)].append(float(value))

        ios_ver = device.get("ios_version", "") or src_ver
        if ios_ver or device:
            version_rows.append({
                "date": day,
                "ios_version": ios_ver,
                "watch_model": device.get("watch_model", ""),
                "watch_fw": device.get("watch_fw", ""),
                "tz_name": tz_name,
                "source_name": src_name
            })

    metric_dfs = {}
    for (metric, day), values in daily.items():
        arr = np.array(values, dtype=float)
        n = int(arr.size)
        if metric == "sleep":
            total_h = float(np.nansum(arr)) if n else 0.0  # total dormido no dia
            mean = float(np.nanmean(arr)) if n else np.nan
            std  = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
            row = {
                "date": day,
                "sleep_sum_h": total_h,
                "sleep_mean": mean,
                "sleep_std":  std,
                "n_sleep": n
            }
        else:
            mean = float(np.nanmean(arr)) if n else np.nan
            std  = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
            row = {
                "date": day,
                f"{metric}_mean": mean,
                f"{metric}_std":  std,
                f"n_{metric}": n
            }
        metric_dfs.setdefault(metric, []).append(row)

    metric_dfs = {m: pd.DataFrame(rows).sort_values("date") for m, rows in metric_dfs.items()}

    vdf = pd.DataFrame(version_rows).drop_duplicates().sort_values("date") if version_rows else pd.DataFrame()
    return metric_dfs, vdf

def build_segments_from_versions(vdf: pd.DataFrame) -> pd.DataFrame:
    """Cria segments (segment_id, start/end) ao mudar ios/watch/fw/tz."""
    if vdf.empty:
        return pd.DataFrame(columns=["segment_id","start","end","ios_version","watch_model","watch_fw","tz_name","source_name"])

    df = vdf.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    keys = ["ios_version","watch_model","watch_fw","tz_name","source_name"]
    sig = df[keys].astype(str).agg(" | ".join, axis=1)
    change = sig.ne(sig.shift(fill_value="__START__"))

    segments = []
    seg_id = 0
    seg_start = df["date"].iloc[0]
    last_vals = df.loc[df.index[0], keys].to_dict()
    for i in range(1, len(df)):
        if change.iloc[i]:
            seg_id += 1
            segments.append({
                "segment_id": seg_id,
                "start": seg_start.date(),
                "end": df["date"].iloc[i-1].date(),
                **last_vals
            })
            seg_start = df["date"].iloc[i]
            last_vals = df.loc[df.index[i], keys].to_dict()
    seg_id += 1
    segments.append({
        "segment_id": seg_id,
        "start": seg_start.date(),
        "end": df["date"].iloc[-1].date(),
        **last_vals
    })
    return pd.DataFrame(segments)

def zscore_by_segment(feat: pd.DataFrame, seg_col="segment_id"):
    cont_cols = [c for c in feat.columns if c.endswith(("_mean","_std")) and not c.startswith("n_")]

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for c in cont_cols:
            mu = g[c].mean()
            sd = g[c].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            g[c] = g[c].astype(float)
            g[c + "_z"] = (g[c] - mu) / sd
        return g

    # pandas ≥ 2.2 tem include_groups; anteriores não
    try:
        return (
            feat.groupby(seg_col, dropna=False, as_index=False)
                .apply(_z, include_groups=False)
                .reset_index(drop=True)
        )
    except TypeError:
        return (
            feat.groupby(seg_col, dropna=False, as_index=False)
                .apply(_z)
                .reset_index(drop=True)
        )

def etl(xml_file: Path, outdir: Path, cutover_str: str, tz_before: str, tz_after: str,
        do_zscore: bool = True, labels_csv: Path | None = None):
    outdir.mkdir(parents=True, exist_ok=True)

    y, m, d = map(int, cutover_str.split("-"))
    tz_selector = make_tz_selector(date(y, m, d), tz_before, tz_after)

    records = iter_health_records(xml_file, tz_selector)
    metric_dfs, vdf = aggregate_daily(records)

    # save per-metric CSVs
    for metric, df in metric_dfs.items():
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        df.to_csv(outdir / f"health_{metric}_daily.csv", index=False)

    # version logs
    if not vdf.empty:
        vdf.to_csv(outdir / "version_raw.csv", index=False)
        seg_df = build_segments_from_versions(vdf)
        seg_df.to_csv(outdir / "version_log_enriched.csv", index=False)
    else:
        seg_df = pd.DataFrame(columns=["segment_id","start","end","ios_version","watch_model","watch_fw","tz_name","source_name"])

    # merge wide
    dfs = []
    for m in ("hr","hrv_sdnn","sleep","temp"):
        f = outdir / f"health_{m}_daily.csv"
        if f.exists():
            dfi = pd.read_csv(f, parse_dates=["date"])
            dfs.append(dfi)
    if dfs:
        feat = dfs[0]
        for dfi in dfs[1:]:
            feat = feat.merge(dfi, on="date", how="outer")
        feat = feat.sort_values("date").reset_index(drop=True)
    else:
        feat = pd.DataFrame(columns=["date"])

    # segment_id by date
    if not seg_df.empty and not feat.empty:
        s2 = seg_df.copy()
        s2["start"] = pd.to_datetime(s2["start"])
        s2["end"] = pd.to_datetime(s2["end"])
        feat["segment_id"] = np.nan
        for _, r in s2.iterrows():
            mask = (feat["date"] >= r["start"]) & (feat["date"] <= r["end"])
            feat.loc[mask, "segment_id"] = int(r["segment_id"])
    else:
        if "segment_id" not in feat.columns:
            feat["segment_id"] = np.nan

    # optional z-score by segment
    if do_zscore and not feat.empty:
        feat = zscore_by_segment(feat, seg_col="segment_id")

    # optional labels merge
    if labels_csv is not None and Path(labels_csv).exists() and not feat.empty:
        lab = pd.read_csv(labels_csv, parse_dates=["date"])
        feat = feat.merge(lab, on="date", how="left")

    # write features
    feat.to_csv(outdir / "features_daily.csv", index=False)

    # ----- Zepp join (opcional, resiliente) -----
    try:
        from etl_modules.zepp_join import join_into_features

        # inferir PID a partir de outdir: data_ai/<PID>/snapshots/<SNAP>
        pid = "P000001"
        try:
            parts = outdir.resolve().parts
            if "data_ai" in parts:
                pid = parts[parts.index("data_ai") + 1]
        except Exception:
            pass

        zepp_dir = Path("data_etl") / pid / "zepp_processed" / "_latest"

        join_ok = join_into_features(
            features_daily_path=outdir / "features_daily.csv",
            zepp_daily_dir=zepp_dir,
            version_log_path=outdir / "version_log_enriched.csv"
        )
        print(f"Zepp join ({pid}): {'OK' if join_ok else 'no-data'}")
    except Exception as e:
        print(f"⚠️ Zepp join skipped: {e}")

    # ---------- QC accumulation ----------
    qc = []
    for m in ("hr","hrv_sdnn","sleep","temp"):
        m_cols = [c for c in feat.columns if c.startswith(m) and c.endswith("_mean")]
        if not m_cols:
            continue
        present = int(pd.Series(feat[m_cols[0]]).notna().sum())
        qc.append({"metric": m, "days_present": present})
    if "segment_id" in feat.columns:
        vc = pd.Series(feat["segment_id"]).value_counts(dropna=True)
        for seg, cnt in vc.items():
            qc.append({"metric": f"segment_{int(seg)}", "days_present": int(cnt)})

    # --- Apple Screen Time (with BR→IE cutover) ---
    try:
        out_usage = outdir / "health_usage_daily.csv"
        df_usage = extract_apple_screen_time(
            export_xml_path=xml_file,
            output_csv_path=out_usage,
            cutover_str=cutover_str,
            tz_before=tz_before,
            tz_after=tz_after,
        )
        if df_usage.empty or int(df_usage.shape[0]) == 0:
            print("ℹ️ Apple Health export.xml does not contain Screen Time records. "
                  "If you need real Screen Time, provide either an iOS encrypted backup "
                  "SQLite (ScreenTime.sqlite) or a daily CSV from Shortcuts under "
                  f"{(outdir.parent.parent / '.../screen_time/') }.")
        mean_min = float(df_usage["screen_time_min"].mean()) if not df_usage.empty else 0.0
        print(f"📱 Screen Time → {out_usage} | days={len(df_usage)} | mean={mean_min:.1f} min/day")

        # Append QC for screen time (even if 0 days; helps downstream)
        qc.extend([
            {"metric": "screen_time_days", "value": int(len(df_usage))},
            {"metric": "screen_time_mean_min", "value": mean_min},
            {"metric": "screen_time_median_min",
             "value": float(df_usage["screen_time_min"].median()) if not df_usage.empty else 0.0},
            {"metric": "screen_time_min_min",
             "value": float(df_usage["screen_time_min"].min()) if not df_usage.empty else 0.0},
            {"metric": "screen_time_max_min",
             "value": float(df_usage["screen_time_min"].max()) if not df_usage.empty else 0.0},
        ])
    except Exception as e:
        print(f"⚠️ Screen Time extraction skipped: {e}")

    # write QC at the end (includes screen time metrics if extracted)
    pd.DataFrame(qc).to_csv(outdir / "etl_qc_summary.csv", index=False)

    print(f"Done. Wrote: {outdir} (features_daily.csv, version_log_enriched.csv, per-metric CSVs).")

def main():
    parser = argparse.ArgumentParser(description="ETL Apple Health → data_ai/")
    parser.add_argument("--participant", help="Participant ID, e.g., P000001")
    parser.add_argument("--snapshot", help="Snapshot id (YYYY-MM-DD or YYYYMMDD)")
    parser.add_argument("--cutover", required=True, help="Cutover date YYYY-MM-DD (BR→IE)")
    parser.add_argument("--tz_before", required=True, help="IANA TZ before cutover, e.g., America/Sao_Paulo")
    parser.add_argument("--tz_after", required=True, help="IANA TZ after cutover, e.g., Europe/Dublin")
    args = parser.parse_args()

    # SINGLE RUN (when both provided)
    if args.participant and args.snapshot:
        pid = args.participant
        snap_iso = _canon_snap_id(args.snapshot)

        # Try with and without the 'snapshots' middle folder on input
        candidates = [
            RAW_ROOT / pid / "snapshots" / snap_iso / "export.xml",
            RAW_ROOT / pid / "snapshots" / snap_iso.replace("-", "") / "export.xml",
            RAW_ROOT / pid / snap_iso / "export.xml",
            RAW_ROOT / pid / snap_iso.replace("-", "") / "export.xml",
        ]
        xml = next((p for p in candidates if p.exists()), None)
        if xml is None:
            print(f"⚠️ export.xml not found. Tried:\n  - " + "\n  - ".join(str(p) for p in candidates))
            sys.exit(1)

        outdir = ensure_ai_outdir(pid, snap_iso)
        print(f"🚀 ETL single: {pid} / {snap_iso}")
        etl(xml_file=xml,
            outdir=outdir,
            cutover_str=args.cutover,
            tz_before=args.tz_before,
            tz_after=args.tz_after)
        return

    # Batch-run (auto discovery)
    jobs = discover_etl_jobs()
    if not jobs:
        print("⚠️ Nothing to process. Expected: data_etl/<PID>/snapshots/<YYYY-MM-DD>/export.xml (legacy YYYYMMDD also supported)")
        sys.exit(0)

    print(f"🚀 ETL batch: {len(jobs)} snapshot(s) found")
    for j in jobs:
        pid, snap, xml = j["pid"], j["snap"], j["xml"]
        outdir = ensure_ai_outdir(pid, snap)
        print(f"→ {pid} / {snap}")
        etl(xml_file=xml, outdir=outdir,
            cutover_str=args.cutover, tz_before=args.tz_before, tz_after=args.tz_after)

if __name__ == "__main__":
    main()
