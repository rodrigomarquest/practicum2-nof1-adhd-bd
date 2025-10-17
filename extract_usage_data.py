#!/usr/bin/env python3
import os, sqlite3, getpass, traceback, sys
from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR  = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
OUT_DIR     = os.path.abspath("decrypted_output")
DST_DIR     = os.path.join(OUT_DIR, "usage_extract")
MANIFEST_DB = os.path.join(OUT_DIR, "Manifest_decrypted.db")

QUERIES = {
    # DeviceActivity (quando existir no backup)
    "deviceactivity_paths": """
      SELECT fileID, domain, relativePath, flags FROM Files
      WHERE flags=1 AND (
            relativePath LIKE 'Library/DeviceActivity/%'
         OR domain LIKE 'AppDomainGroup-group.com.apple.deviceactivity%'
         OR relativePath LIKE '%DeviceActivity%'
      )
    """,
    # Preferências ScreenTime (úteis para limites/config)
    "screentime_prefs": """
      SELECT fileID, domain, relativePath, flags FROM Files
      WHERE flags=1 AND (
            relativePath LIKE 'Library/Preferences/%ScreenTime%'
         OR domain LIKE '%ScreenTime%'
      )
    """,
    # KnowledgeC (CoreDuet) — base para uso de apps / eventos
    "knowledgec": """
      SELECT fileID, domain, relativePath, flags FROM Files
      WHERE flags=1 AND (
            relativePath LIKE 'Library/CoreDuet/Knowledge/KnowledgeC.db'
         OR relativePath LIKE 'Library/CoreDuet/Knowledge/%'
         OR relativePath LIKE '%KnowledgeC.db%'
      )
    """,
}

def blob_path(backup_dir, file_id):
    return os.path.join(backup_dir, file_id[:2], file_id)

def extract_set(b, name, rows):
    ok = fail = 0
    for fid, dom, rel, flags in rows:
        src = blob_path(BACKUP_DIR, fid)
        if not (flags == 1 and os.path.isfile(src) and os.path.getsize(src) > 0):
            continue  # só tenta quando há blob
        safe_domain = (dom or "NoDomain").replace("/", "_")
        safe_rel = (rel or "no_relativePath").replace(":", "_").replace("\\", "/")
        out_path = os.path.join(DST_DIR, name, safe_domain, safe_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            b.extract_file(file_id=fid, output_filename=out_path)
            ok += 1
        except Exception:
            fail += 1
            with open(os.path.join(DST_DIR, "_extract_errors.log"), "a", encoding="utf-8") as f:
                f.write(f"[{name}] FAIL {fid} {dom} {rel}\n")
    return ok, fail

def main():
    if not os.path.isfile(MANIFEST_DB):
        print("❌ Manifest_decrypted.db não encontrado.")
        sys.exit(1)

    os.makedirs(DST_DIR, exist_ok=True)
    pw = getpass.getpass("Enter iTunes / backup passphrase: ")
    try:
        b = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
    except Exception:
        print(traceback.format_exc())
        print("❌ Não foi possível abrir o backup (senha errada ou backup inválido).")
        sys.exit(1)

    con = sqlite3.connect(MANIFEST_DB)
    cur = con.cursor()

    total_ok = total_fail = 0
    for name, sql in QUERIES.items():
        try:
            rows = cur.execute(sql).fetchall()
        except Exception:
            rows = []
        print(f"🔎 {name}: {len(rows)} candidatos no manifest")
        ok, fail = extract_set(b, name, rows)
        print(f"   → extraídos: {ok} | falhas: {fail}")
        total_ok += ok; total_fail += fail

    con.close()
    print(f"\n✅ Extração finalizada: OK={total_ok} | FAIL={total_fail}")
    print(f"📂 Saída em: {DST_DIR}")
    print("   • Se houver 'KnowledgeC.db', é o caminho mais sólido p/ Screen Time diário.")
    print("   • Prefs do Screen Time (plist) podem estar em 'screentime_prefs'.")

if __name__ == "__main__":
    main()
