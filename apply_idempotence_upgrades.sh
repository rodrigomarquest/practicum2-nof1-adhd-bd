#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d%H%M%S)"

backup() {
  local f="$1"
  if [ -f "$f" ]; then cp -f "$f" "$f.bak.$TS" && echo "BACKUP: $f -> $f.bak.$TS"; fi
}

# 1) Patch etl_pipeline.py (atomic write + manifests + --zepp_dir)
backup etl_pipeline.py
python - <<'PY'
from pathlib import Path
import re, json, hashlib

p = Path("etl_pipeline.py")
src = p.read_text(encoding="utf-8")

def ensure_helpers(s: str) -> str:
    if "def _sha256_file(" in s and "def _write_atomic_csv(" in s and "def _write_atomic_json(" in s:
        return s
    helper = r'''
import os, json, tempfile, hashlib

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_atomic_csv(df, out_path: str):
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def _write_atomic_json(obj, out_path: str):
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".json")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
'''
    # inserir helpers após imports principais
    idx = s.find("\nimport argparse")
    if idx == -1:
        idx = s.find("import argparse")
    if idx != -1:
        # inserir helpers depois da primeira sequência de imports
        # encontra fim do bloco de imports
        m = re.search(r"(?:^|\n)(?:import|from)[^\n]+\n(?:[ \t]*from[^\n]+\n|[ \t]*import[^\n]+\n)*", s)
        insert_at = m.end() if m else idx
        return s[:insert_at] + helper + s[insert_at:]
    return helper + s

def patch_extract_manifest(s: str) -> str:
    # substituir to_csv por _write_atomic_csv e criar manifest
    if "extract_apple_per_metric(" not in s:
        return s
    s = re.sub(r'\.to_csv\(([^)]+)\)', r'__CSV__\(\1\)', s)  # placeholder para não conflitar com outras partes
    s = s.replace("__CSV__(", "_write_atomic_csv(")
    # adicionar geração de manifest ao final da função extract_apple_per_metric
    s = re.sub(
        r"(return written\s*\n)",
        r"""
    # ---- manifest com hashes do export.xml e saídas ----
    manifest = {
        "type": "extract",
        "export_xml": str(xml_file),
        "params": {"cutover": cutover_str, "tz_before": tz_before, "tz_after": tz_after},
        "outputs": {},
    }
    try:
        import os
        if os.path.exists(xml_file):
            manifest["export_sha256"] = _sha256_file(str(xml_file))
    except Exception:
        pass
    for k, v in list(written.items()):
        try:
            manifest["outputs"][k] = {"path": v, "sha256": _sha256_file(v)}
        except Exception:
            manifest["outputs"][k] = {"path": v, "sha256": None}
    _write_atomic_json(manifest, str(outdir / "extract_manifest.json"))
\g<1>""",
        s,
        count=1
    )
    return s

def patch_cardio_cmd(s: str) -> str:
    # adicionar --zepp_dir ao parser do subcomando cardio
    s = s.replace('p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")',
                  'p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")\n    p_car.add_argument("--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)")')
    # setar env var antes de chamar run_stage_cardio e escrever cardio_manifest.json
    s = s.replace(
        'res = run_stage_cardio(snapdir)\n        print("✅ cardio concluído")\n        print(res)\n        return 0',
        r'''import os, glob
        if args.zepp_dir:
            os.environ["ZEPPOVERRIDE_DIR"] = args.zepp_dir
        res = run_stage_cardio(snapdir)

        # manifest do cardio (inputs e outputs com hashes)
        inputs = {}
        # per-metric
        for pm in ["apple_heart_rate.csv","apple_hrv_sdnn.csv","apple_sleep_intervals.csv"]:
            pth = snapdir / "per-metric" / pm
            if pth.exists():
                try:
                    inputs[pm] = {"path": str(pth), "sha256": _sha256_file(str(pth))}
                except Exception:
                    inputs[pm] = {"path": str(pth), "sha256": None}
        # segments
        for name in ["version_log_enriched.csv","version_raw.csv"]:
            pth = snapdir / name
            if pth.exists():
                try:
                    inputs[name] = {"path": str(pth), "sha256": _sha256_file(str(pth))}
                except Exception:
                    inputs[name] = {"path": str(pth), "sha256": None}
        # Zepp
        zdir = os.environ.get("ZEPPOVERRIDE_DIR","")
        if zdir:
            zepp_files = glob.glob(os.path.join(zdir, "*.csv"))
        else:
            zepp_files = []
            # manter compat: se não passar zepp_dir, o loader decidirá o caminho
        zepp = {}
        for z in zepp_files:
            try: zepp[z] = {"path": z, "sha256": _sha256_file(z)}
            except Exception: zepp[z] = {"path": z, "sha256": None}

        outputs = {}
        for k,v in res.items():
            try:
                outputs[k] = {"path": v, "sha256": _sha256_file(v)}
            except Exception:
                outputs[k] = {"path": v, "sha256": None}

        manifest = {
            "type": "cardio",
            "snapshot_dir": str(snapdir),
            "zepp_dir": zdir or None,
            "inputs": {"per_metric": inputs, "zepp": zepp},
            "outputs": outputs
        }
        _write_atomic_json(manifest, str(snapdir / "cardio_manifest.json"))

        print("✅ cardio concluído")
        print(res)
        return 0'''
    )
    return s

src = ensure_helpers(src)
src = patch_extract_manifest(src)
src = patch_cardio_cmd(src)
p.write_text(src, encoding="utf-8")
print("WROTE: etl_pipeline.py (helpers + manifests + --zepp_dir + atomic writes)")
PY

# 2) Patch cardio_features.py (atomic writes)
backup etl_modules/cardiovascular/cardio_features.py
python - <<'PY'
from pathlib import Path
import re
p = Path("etl_modules/cardiovascular/cardio_features.py")
s = p.read_text(encoding="utf-8")

if "_write_atomic_csv" not in s:
    s = s.replace('import math, os', 'import math, os, tempfile, json, hashlib')

    s = s.replace('from etl_modules.common.io import read_csv_if_exists',
                  'from etl_modules.common.io import read_csv_if_exists')

    s = s.replace('def update_features_daily(', 'def _sha256_file(path: str):\n    h=hashlib.sha256()\n    with open(path,"rb") as f:\n        for ch in iter(lambda: f.read(1024*1024), b""):\n            h.update(ch)\n    return h.hexdigest()\n\ndef _write_atomic_csv(df, out_path: str):\n    d = os.path.dirname(out_path) or "."\n    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")\n    os.close(fd)\n    try:\n        df.to_csv(tmp, index=False)\n        os.replace(tmp, out_path)\n    finally:\n        if os.path.exists(tmp):\n            try: os.remove(tmp)\n            except: pass\n\n\ndef update_features_daily(')

    # substituir .to_csv por _write_atomic_csv (duas ocorrências)
    s = re.sub(r'\.to_csv\(([^)]+)\)', r'__CSV__\(\1\)', s)
    s = s.replace('__CSV__(', '_write_atomic_csv(')

    p.write_text(s, encoding="utf-8")
    print("WROTE: cardio_features.py (atomic CSV)")
else:
    print("SKIP: cardio_features.py already has atomic helpers")
PY

# 3) Patch Zepp loader (env var ZEPPOVERRIDE_DIR para caminho fixo)
backup etl_modules/cardiovascular/zepp/loader.py
python - <<'PY'
from pathlib import Path
import re, os
p = Path("etl_modules/cardiovascular/zepp/loader.py")
s = p.read_text(encoding="utf-8")

if "ZEPPOVERRIDE_DIR" not in s:
    s = s.replace(
        'def _first_existing(paths: List[str]) -> Optional[str]:',
        'import os\n\ndef _first_existing(paths: List[str]) -> Optional[str]:'
    )
    # inserir checagem de env var nos candidatos HR e HRV
    s = s.replace(
        'candidates = [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_hr_daily.csv"),',
        'envdir = os.environ.get("ZEPPOVERRIDE_DIR")\n        candidates = ([os.path.join(envdir, "zepp_hr_daily.csv")] if envdir else []) + [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_hr_daily.csv"),'
    )
    s = s.replace(
        'candidates = [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_daily_features.csv"),',
        'envdir = os.environ.get("ZEPPOVERRIDE_DIR")\n        candidates = ([os.path.join(envdir, "zepp_daily_features.csv")] if envdir else []) + [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_daily_features.csv"),'
    )
    p.write_text(s, encoding="utf-8")
    print("WROTE: zepp/loader.py (env ZEPPOVERRIDE_DIR support)")
else:
    print("SKIP: zepp/loader.py already supports ZEPPOVERRIDE_DIR")
PY

echo "✅ All patches applied."
