#!/usr/bin/env python3
"""Apply helper and manifest patches to etl_pipeline.py (extracted from heredoc).

This script inserts atomic write helpers and modifies extract/cardio behavior to
write manifests and accept --zepp_dir. Run from project root.
"""
from pathlib import Path
import re


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
    # insert after first argparse import if present
    idx = s.find("\nimport argparse")
    if idx == -1:
        idx = s.find("import argparse")
    if idx != -1:
        m = re.search(r"(?:^|\n)(?:import|from)[^\n]+\n(?:[ \t]*from[^\n]+\n|[ \t]*import[^\n]+\n)*", s)
        insert_at = m.end() if m else idx
        return s[:insert_at] + helper + s[insert_at:]
    return helper + s


def patch_extract_manifest(s: str) -> str:
    if "extract_apple_per_metric(" not in s:
        return s
    s = re.sub(r'\.to_csv\(([^)]+)\)', r'__CSV__(\1)', s)
    s = s.replace("__CSV__(", "_write_atomic_csv(")
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
        count=1,
    )
    return s


def patch_cardio_cmd(s: str) -> str:
    s = s.replace('p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")',
                  'p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")\n    p_car.add_argument("--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)")')
    s = s.replace(
        'res = run_stage_cardio(snapdir)\n        print("✅ cardio concluído")\n        print(res)\n        return 0',
        r'''import os, glob
        if args.zepp_dir:
            os.environ["ZEPPOVERRIDE_DIR"] = args.zepp_dir
        res = run_stage_cardio(snapdir)

        # manifest do cardio (inputs e outputs com hashes)
        inputs = {}
        for pm in ["apple_heart_rate.csv","apple_hrv_sdnn.csv","apple_sleep_intervals.csv"]:
            pth = snapdir / "per-metric" / pm
            if pth.exists():
                try:
                    inputs[pm] = {"path": str(pth), "sha256": _sha256_file(str(pth))}
                except Exception:
                    inputs[pm] = {"path": str(pth), "sha256": None}
        for name in ["version_log_enriched.csv","version_raw.csv"]:
            pth = snapdir / name
            if pth.exists():
                try:
                    inputs[name] = {"path": str(pth), "sha256": _sha256_file(str(pth))}
                except Exception:
                    inputs[name] = {"path": str(pth), "sha256": None}
        zdir = os.environ.get("ZEPPOVERRIDE_DIR","")
        if zdir:
            zepp_files = glob.glob(os.path.join(zdir, "*.csv"))
        else:
            zepp_files = []
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
        return 0''')
    return s


def main():
    p = Path("etl_pipeline.py")
    if not p.exists():
        print("etl_pipeline.py not found")
        raise SystemExit(1)
    src = p.read_text(encoding="utf-8")
    src = ensure_helpers(src)
    src = patch_extract_manifest(src)
    src = patch_cardio_cmd(src)
    p.write_text(src, encoding="utf-8")
    print("WROTE: etl_pipeline.py (helpers + manifests + --zepp_dir + atomic writes)")


if __name__ == '__main__':
    main()
