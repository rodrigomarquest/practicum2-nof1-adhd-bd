#!/usr/bin/env bash
set -euo pipefail

# 1) Backup e rename do pipeline atual → etl_pipeline_legacy.py (se ainda não fizemos)
if [ -f "etl_pipeline.py" ] && [ ! -f "etl_pipeline_legacy.py" ]; then
  cp -f etl_pipeline.py "etl_pipeline_legacy.py"
  echo "BACKUP: etl_pipeline.py -> etl_pipeline_legacy.py"
fi

# 2) Garantir utilitário Apple raw→per-metric (usa funções do legacy)
mkdir -p etl_modules
cat > etl_modules/apple_raw_to_per_metric.py <<'PY'
"""
Extrai per-metric Apple (HR minuto-a-minuto, HRV SDNN) a partir do export.xml,
reutilizando funções do pipeline legado: iter_health_records, make_tz_selector.
Saídas no snapshot:
  per-metric/apple_heart_rate.csv   (timestamp,bpm)
  per-metric/apple_hrv_sdnn.csv     (timestamp,sdnn_ms)
  per-metric/apple_sleep_intervals.csv (opcional)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict
import importlib

def export_per_metric(export_xml: Path, out_snapshot_dir: Path,
                      cutover: str, tz_before: str, tz_after: str) -> Dict[str,str]:
    legacy = importlib.import_module("etl_pipeline_legacy")
    out_pm = out_snapshot_dir / "per-metric"
    out_pm.mkdir(parents=True, exist_ok=True)

    # TZ selector
    from datetime import date
    y, m, d = map(int, cutover.split("-"))
    tz_selector = legacy.make_tz_selector(date(y, m, d), tz_before, tz_after)

    # Stream records
    recs = legacy.iter_health_records(export_xml, tz_selector)

    hr_rows, hrv_rows, sleep_rows = [], [], []
    HR_ID   = "HKQuantityTypeIdentifierHeartRate"
    HRV_ID  = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    SLEEP_ID= "HKCategoryTypeIdentifierSleepAnalysis"

    for (type_id, value, s_local, e_local, unit, device, tz_name, src_ver, src_name) in recs:
        if type_id == HR_ID and value is not None:
            hr_rows.append({"timestamp": s_local.isoformat(), "bpm": float(value)})
        elif type_id == HRV_ID and value is not None:
            hrv_rows.append({"timestamp": s_local.isoformat(), "sdnn_ms": float(value)})
        elif type_id == SLEEP_ID and e_local is not None:
            sleep_rows.append({"start": s_local.isoformat(), "end": e_local.isoformat(), "raw_value": value})

    written = {}
    if hr_rows:
        p = out_pm / "apple_heart_rate.csv"
        pd.DataFrame(hr_rows).sort_values("timestamp").to_csv(p, index=False)
        written["apple_heart_rate"] = str(p)
    if hrv_rows:
        p = out_pm / "apple_hrv_sdnn.csv"
        pd.DataFrame(hrv_rows).sort_values("timestamp").to_csv(p, index=False)
        written["apple_hrv_sdnn"] = str(p)
    if sleep_rows:
        p = out_pm / "apple_sleep_intervals.csv"
        pd.DataFrame(sleep_rows).sort_values("start").to_csv(p, index=False)
        written["apple_sleep_intervals"] = str(p)
    return written
PY

# 3) Instalar novo etl_pipeline.py (CLI por subcomandos)
cat > etl_pipeline.py <<'PY'
from __future__ import annotations
"""
etl_pipeline.py — Orquestrador modular (v2)

Subcomandos:
  - apple-per-metric : gera per-metric/ a partir do export.xml (sem fallback)
  - cardio           : executa o estágio cardiovascular (Apple/Zepp + join + features)
  - legacy-apple     : executa o pipeline legado (compatibilidade com flags antigas)

Exemplos:
  python etl_pipeline.py apple-per-metric --participant P000001 --snapshot 2025-09-29 \
      --cutover 2024-03-11 --tz-before America/Sao_Paulo --tz-after Europe/Dublin

  python etl_pipeline.py cardio --participant P000001 --snapshot 2025-09-29
  # ou apontando o diretório diretamente:
  python etl_pipeline.py cardio --snapshot-dir data_ai/P000001/snapshots/2025-09-29

  python etl_pipeline.py legacy-apple --participant P000001 --snapshot 2025-09-29 \
      --cutover 2024-03-11 --tz-before America/Sao_Paulo --tz-after Europe/Dublin
"""
import argparse
import sys
from pathlib import Path
import importlib
from typing import Optional

# Utilidades simples (auto-contidas)
RAW_ROOT = Path("data_etl")
AI_ROOT  = Path("data_ai")

def _canon_snap_id(name: str) -> str:
    name = name.strip()
    if len(name) == 10 and name[4] == "-" and name[7] == "-":
        return name
    if len(name) == 8 and name.isdigit():
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    raise ValueError(f"Invalid snapshot '{name}'. Expected YYYY-MM-DD or YYYYMMDD.")

def _find_export_xml(pid: str, snap_iso: str) -> Optional[Path]:
    candidates = [
        RAW_ROOT / pid / "snapshots" / snap_iso / "export.xml",
        RAW_ROOT / pid / "snapshots" / snap_iso.replace("-", "") / "export.xml",
        RAW_ROOT / pid / snap_iso / "export.xml",
        RAW_ROOT / pid / snap_iso.replace("-", "") / "export.xml",
    ]
    return next((p for p in candidates if p.exists()), None)

def _ensure_ai_outdir(pid: str, snap_iso: str) -> Path:
    outdir = AI_ROOT / pid / "snapshots" / snap_iso
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

# -------------------- Subcomandos --------------------
def cmd_apple_per_metric(args: argparse.Namespace) -> int:
    snap_iso = _canon_snap_id(args.snapshot)
    xml = _find_export_xml(args.participant, snap_iso)
    if xml is None:
        print("⚠️ export.xml not found. Tried:")
        print("   -", RAW_ROOT / args.participant / "snapshots" / snap_iso / "export.xml")
        print("   -", RAW_ROOT / args.participant / "snapshots" / snap_iso.replace("-", "") / "export.xml")
        print("   -", RAW_ROOT / args.participant / snap_iso / "export.xml")
        print("   -", RAW_ROOT / args.participant / snap_iso.replace("-", "") / "export.xml")
        return 1
    outdir = _ensure_ai_outdir(args.participant, snap_iso)
    apple_pm = importlib.import_module("etl_modules.apple_raw_to_per_metric")
    written = apple_pm.export_per_metric(xml, outdir, args.cutover, args.tz_before, args.tz_after)
    if not written:
        print("No per-metric files produced.")
        return 2
    print("Per-metric written:")
    for k,v in written.items():
        print(" -", k, "->", v)
    return 0

def cmd_cardio(args: argparse.Namespace) -> int:
    if args.snapshot_dir:
        snapdir = Path(args.snapshot_dir)
    else:
        snap_iso = _canon_snap_id(args.snapshot)
        snapdir = AI_ROOT / args.participant / "snapshots" / snap_iso
    if not snapdir.exists():
        print(f"⚠️ Snapshot directory not found: {snapdir}")
        return 1

    from etl_modules.config import CardioCfg
    from etl_modules.cardiovascular.cardio_etl import run_stage_cardio
    res = run_stage_cardio(str(snapdir), str(snapdir), CardioCfg())
    print(res)
    return 0

def cmd_legacy_apple(args: argparse.Namespace) -> int:
    # Roda o pipeline antigo com as mesmas flags (para compatibilidade)
    legacy = importlib.import_module("etl_pipeline_legacy")
    snap_iso = _canon_snap_id(args.snapshot)
    outdir = _ensure_ai_outdir(args.participant, snap_iso)

    xml = _find_export_xml(args.participant, snap_iso)
    if xml is None:
        print("⚠️ export.xml not found (legacy).")
        return 1

    print(f"🚀 Legacy Apple ETL: {args.participant} / {snap_iso}")
    legacy.etl(xml_file=xml,
               outdir=outdir,
               cutover_str=args.cutover,
               tz_before=args.tz_before,
               tz_after=args.tz_after)
    return 0

def main():
    ap = argparse.ArgumentParser(description="ETL Orchestrator (modular v2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("apple-per-metric", help="Extract per-metric data (Apple) from export.xml")
    p1.add_argument("--participant", required=True)
    p1.add_argument("--snapshot", required=True)
    p1.add_argument("--cutover", required=True)
    p1.add_argument("--tz_before", required=True)
    p1.add_argument("--tz_after", required=True)
    p1.set_defaults(func=cmd_apple_per_metric)

    p2 = sub.add_parser("cardio", help="Run cardiovascular stage on a snapshot directory")
    g = p2.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot-dir", help="Path to data_ai/<PID>/snapshots/<YYYY-MM-DD>")
    g.add_argument("--snapshot", help="Snapshot id; use with --participant")
    p2.add_argument("--participant", help="Participant ID (required if --snapshot)")
    p2.set_defaults(func=cmd_cardio)

    p3 = sub.add_parser("legacy-apple", help="Run the legacy Apple ETL (backward compatible)")
    p3.add_argument("--participant", required=True)
    p3.add_argument("--snapshot", required=True)
    p3.add_argument("--cutover", required=True)
    p3.add_argument("--tz_before", required=True)
    p3.add_argument("--tz_after", required=True)
    p3.set_defaults(func=cmd_legacy_apple)

    args = ap.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
PY

echo "✅ etl_pipeline.py (v2) instalado com subcomandos."
echo "   - apple-per-metric  |  - cardio  |  - legacy-apple"
