#!/usr/bin/env python3
import os
import sqlite3
import getpass
import sys
import traceback
import datetime
from iphone_backup_decrypt import EncryptedBackup

# === Config ==========================================================
BACKUP_DIR = os.environ.get(
    "BACKUP_DIR",
    r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
)
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", "decrypted_output"))
MANIFEST_DB = os.path.join(OUT_DIR, "Manifest_decrypted.db")
DST_DIR = os.path.join(OUT_DIR, "knowledgec")
LOG_PATH = os.path.join(DST_DIR, "extract_knowledgec.log")

SQL = """
SELECT fileID, domain, relativePath, flags
FROM Files
WHERE flags=1 AND (
       relativePath LIKE 'Library/CoreDuet/Knowledge/KnowledgeC.db'
    OR relativePath LIKE 'Library/CoreDuet/Knowledge/%'
    OR relativePath LIKE '%KnowledgeC.db%'
    OR relativePath LIKE 'Library/Knowledge/%'
    OR relativePath LIKE 'Library/Application Support/Knowledge/%'
)
ORDER BY relativePath;
"""


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"[{ts}] {msg}\n")


def blob_path(base, file_id):
    return os.path.join(base, file_id[:2], file_id)


def main():
    if not os.path.isfile(MANIFEST_DB):
        log("❌ Manifest_decrypted.db não encontrado. Rode antes o decrypt_manifest.py.")
        sys.exit(1)

    os.makedirs(DST_DIR, exist_ok=True)

    # 1️⃣ senha do ambiente (com fallback)
    pw = os.environ.get("BACKUP_PASSWORD")
    if pw:
        log("🔐 Using BACKUP_PASSWORD from environment.")
    else:
        log("⚠️ BACKUP_PASSWORD not found. Asking interactively...")
        pw = getpass.getpass("Enter iTunes / backup passphrase: ")

    # 2️⃣ abrir backup criptografado
    try:
        log(f"📁 Opening encrypted backup: {BACKUP_DIR}")
        enc = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
        log("✅ EncryptedBackup opened successfully.")
    except Exception:
        log(traceback.format_exc())
        log("❌ Failed to open encrypted backup (wrong password or invalid backup).")
        sys.exit(1)

    # 3️⃣ consultar manifest
    con = sqlite3.connect(MANIFEST_DB)
    cur = con.cursor()
    rows = cur.execute(SQL).fetchall()
    con.close()

    if not rows:
        log("⚠️ Nenhum registro KnowledgeC encontrado no Manifest.")
        sys.exit(0)

    log(f"🔎 Candidatos KnowledgeC no Manifest: {len(rows)}")
    ok = fail = 0

    for fid, dom, rel, flags in rows:
        src = blob_path(BACKUP_DIR, fid)
        if flags != 1 or not os.path.isfile(src) or os.path.getsize(src) == 0:
            continue

        safe_d = (dom or "NoDomain").replace("/", "_")
        safe_r = (rel or "no_relativePath").replace(":", "_").replace("\\", "/")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        out_path = os.path.join(DST_DIR, safe_d, f"{timestamp}_{os.path.basename(safe_r)}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            enc.extract_file(file_id=fid, output_filename=out_path)
            ok += 1
            log(f"✅ Extracted {rel} → {out_path}")
        except Exception as e:
            fail += 1
            log(f"❌ FAIL {fid} {rel} → {e}")
            with open(os.path.join(DST_DIR, "_extract_errors.log"), "a", encoding="utf-8") as f:
                f.write(f"FAIL {fid} {dom} {rel}\n")

    log(f"✅ Extraction completed: OK={ok} | FAIL={fail}")
    log(f"📂 Output: {DST_DIR}")

    # 4️⃣ Validação dos .db extraídos
    if ok > 0:
        import glob
        cands = glob.glob(os.path.join(DST_DIR, "**", "*.db"), recursive=True)
        log("\n🧪 PRAGMA integrity_check para candidatos .db:")
        for p in cands:
            try:
                c = sqlite3.connect(p)
                k = c.cursor()
                k.execute("PRAGMA integrity_check;")
                ic = k.fetchone()
                k.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tbls = [r[0] for r in k.fetchall()]
                c.close()
                log(f" → {os.path.basename(p)} integrity={ic} tables={tbls[:10]}")
            except Exception:
                log(f"⚠️  Could not validate {p}")

    log("🏁 Done.")


if __name__ == "__main__":
    main()
