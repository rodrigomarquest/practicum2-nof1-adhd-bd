#!/usr/bin/env python3
import os, sqlite3

BACKUP_DIR  = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
MANIFEST_DB = r"C:\dev\practicum2-nof1-adhd-bd\decrypted_output\Manifest_decrypted.db"

SQL = """
SELECT fileID, domain, relativePath, flags
FROM Files
WHERE
  relativePath LIKE 'Library/DeviceActivity/%'
  OR domain LIKE 'AppDomainGroup-group.com.apple.deviceactivity%'
  OR relativePath LIKE 'Library/Preferences/%ScreenTime%'
  OR domain LIKE '%ScreenTime%'
  OR relativePath LIKE '%DeviceActivity%'
  OR relativePath LIKE '%ScreenTime%'
"""

def blob_path(backup_dir, file_id):
    return os.path.join(backup_dir, file_id[:2], file_id)

con = sqlite3.connect(MANIFEST_DB)
cur = con.cursor()
rows = cur.execute(SQL).fetchall()
con.close()

print(f"Encontrados {len(rows)} candidatos (manifest). Amostragem dos primeiros 30:")
missing, present, dirs = 0, 0, 0
for i, (fid, dom, rel, flags) in enumerate(rows[:30], 1):
    path = blob_path(BACKUP_DIR, fid)
    is_file = (flags == 1)  # 1 = arquivo; 2 = diretório
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    tag = ("DIR" if not is_file else ("OK" if exists and size>0 else "MISS"))
    if not is_file:
        dirs += 1
    elif exists and size>0:
        present += 1
    else:
        missing += 1
    print(f"{i:02d}) {tag} flags={flags} fid={fid} | {dom} | {rel} | blob_exist={exists} size={size} -> {path}")

print(f"\nResumo blobs → PRESENTE={present} | AUSENTE={missing} | DIRETÓRIOS={dirs}")
