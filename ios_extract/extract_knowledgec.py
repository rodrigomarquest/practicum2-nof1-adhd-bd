#!/usr/bin/env python3
import os, sqlite3, getpass, sys, traceback
from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR  = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
OUT_DIR     = os.path.abspath("decrypted_output")
MANIFEST_DB = os.path.join(OUT_DIR, "Manifest_decrypted.db")
DST_DIR     = os.path.join(OUT_DIR, "knowledgec")

SQL = """
SELECT fileID, domain, relativePath, flags
FROM Files
WHERE flags=1 AND (
       relativePath LIKE 'Library/CoreDuet/Knowledge/KnowledgeC.db'
    OR relativePath LIKE 'Library/CoreDuet/Knowledge/%'
    OR relativePath LIKE '%KnowledgeC.db%'
    OR relativePath LIKE 'Library/Knowledge/%'               -- variações antigas
    OR relativePath LIKE 'Library/Application Support/Knowledge/%'
)
ORDER BY relativePath
"""

def blob_path(base, file_id):
    return os.path.join(base, file_id[:2], file_id)

def main():
    if not os.path.isfile(MANIFEST_DB):
        print("❌ Manifest_decrypted.db não encontrado. Rode antes o decrypt_manifest.py.")
        sys.exit(1)

    os.makedirs(DST_DIR, exist_ok=True)

    # abrir backup criptografado
    pw = getpass.getpass("Enter iTunes / backup passphrase: ")
    try:
        enc = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
    except Exception:
        print(traceback.format_exc())
        print("❌ Não foi possível abrir o backup (senha errada ou backup inválido).")
        sys.exit(1)

    con = sqlite3.connect(MANIFEST_DB)
    cur = con.cursor()
    rows = cur.execute(SQL).fetchall()
    con.close()

    if not rows:
        print("⚠️  Nada com 'KnowledgeC' encontrado no Manifest.")
        sys.exit(0)

    print(f"🔎 Candidatos KnowledgeC no Manifest: {len(rows)}")
    ok = fail = 0
    for fid, dom, rel, flags in rows:
        src = blob_path(BACKUP_DIR, fid)
        # só extrai se blob existir localmente
        if flags != 1 or not os.path.isfile(src) or os.path.getsize(src) == 0:
            continue
        safe_d = (dom or "NoDomain").replace("/", "_")
        safe_r = (rel or "no_relativePath").replace(":", "_").replace("\\", "/")
        out_path = os.path.join(DST_DIR, safe_d, safe_r)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            enc.extract_file(file_id=fid, output_filename=out_path)
            ok += 1
        except Exception:
            fail += 1
            with open(os.path.join(DST_DIR, "_extract_errors.log"), "a", encoding="utf-8") as f:
                f.write(f"FAIL {fid} {dom} {rel}\n")

    print(f"✅ Extração: OK={ok} | FAIL={fail}")
    print(f"📂 Saída: {DST_DIR}")

    # Validação rápida de .db extraídos
    if ok > 0:
        import glob
        cands = glob.glob(os.path.join(DST_DIR, "**", "*.db"), recursive=True)
        print("\n🧪 PRAGMA integrity_check para candidatos .db:")
        for p in cands:
            try:
                c = sqlite3.connect(p)
                k = c.cursor()
                k.execute("PRAGMA integrity_check;")
                ic = k.fetchone()
                k.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tbls = [r[0] for r in k.fetchall()]
                c.close()
                print(f" → {p}\n    integrity_check={ic} tables={tbls[:12]}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
