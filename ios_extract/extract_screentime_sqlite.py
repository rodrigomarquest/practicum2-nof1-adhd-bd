#!/usr/bin/env python3
import os, sys, getpass, sqlite3, traceback, datetime
from pathlib import Path
from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR = os.environ.get("BACKUP_DIR", r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E")
OUT_DIR    = os.path.abspath(os.environ.get("OUT_DIR", "decrypted_output"))
MANIFEST   = os.path.join(OUT_DIR, "Manifest_decrypted.db")
CANDS_TSV  = os.path.join(OUT_DIR, "_work", "probe_screentime_sqlite.tsv")
DST_DIR    = os.path.join(OUT_DIR, "screentime_sqlite")
LOG_PATH   = os.path.join(DST_DIR, "extract_screentime_sqlite.log")

Path(DST_DIR).mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"[{ts}] {msg}\n")

def main():
    if not os.path.isfile(MANIFEST):
        log(f"❌ Manifest not found at {MANIFEST}. Run decrypt first.")
        sys.exit(1)
    if not os.path.isfile(CANDS_TSV):
        log(f"❌ Candidates TSV not found at {CANDS_TSV}. Run probe-sqlite first.")
        sys.exit(1)

    pw = os.environ.get("BACKUP_PASSWORD")
    if pw:
        log("🔐 Using BACKUP_PASSWORD from environment.")
    else:
        log("⚠️ BACKUP_PASSWORD not found; asking interactively…")
        pw = getpass.getpass("Enter iTunes / backup passphrase: ")

    try:
        enc = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
        log("✅ EncryptedBackup opened successfully.")
    except Exception:
        log(traceback.format_exc()); log("❌ Failed to open encrypted backup.")
        sys.exit(1)

    # Ler candidatos
    rows = []
    with open(CANDS_TSV, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            parts += [""] * (4 - len(parts))
            fid, dom, rel, flags = [p.strip() for p in parts[:4]]
            # só pega caminhos que parecem arquivo .db/.sqlite
            if rel and (rel.lower().endswith(".db") or ".sqlite" in rel.lower()):
                rows.append((fid, dom, rel, flags))

    if not rows:
        log("ℹ️  No SQLite candidates after filtering. (Likely not present in this backup.)")
        return

    ok = fail = 0
    for fid, dom, rel, flags in rows:
        safe_d = (dom or "NoDomain").replace("/", "_")
        fname  = os.path.basename(rel)
        out_p  = os.path.join(DST_DIR, safe_d, fname)
        Path(os.path.dirname(out_p)).mkdir(parents=True, exist_ok=True)
        try:
            enc.extract_file(relative_path=rel, output_filename=out_p)
            log(f"✅ {rel} → {out_p}")
            ok += 1
        except Exception as e:
            log(f"❌ FAIL {rel} ({fid}): {e!r}")
            fail += 1
            
    log(f"✔ Done. OK={ok} FAIL={fail}")

    # listar tabelas/integridade dos SQLites extraídos
    if ok:
        import glob
        for p in glob.glob(os.path.join(DST_DIR, "**", "*.db"), recursive=True) + \
                 glob.glob(os.path.join(DST_DIR, "**", "*.sqlite*"), recursive=True):
            try:
                con = sqlite3.connect(p); cur = con.cursor()
                cur.execute("PRAGMA integrity_check;"); ic = cur.fetchone()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tbls = [r[0] for r in cur.fetchall()]
                con.close()
                log(f"🧪 {p} | integrity={ic} | tables={tbls[:20]}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
