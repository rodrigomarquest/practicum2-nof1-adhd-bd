# quick_post_backup_probe.py
import os, sqlite3
BACKUP_DIR  = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
MANIFEST_DB = r"C:\dev\practicum2-nof1-adhd-bd\decrypted_output\Manifest_decrypted.db"

SQL = """
SELECT fileID, domain, relativePath, flags
FROM Files
WHERE flags=1 AND (
     relativePath LIKE 'Library/CoreDuet/Knowledge/KnowledgeC.db'
  OR relativePath LIKE 'Library/CoreDuet/Knowledge/%'
  OR relativePath LIKE '%KnowledgeC.db%'
  OR relativePath LIKE '%CoreDuet%Knowledge%'
  OR relativePath LIKE 'Library/DeviceActivity/%'
  OR domain LIKE 'AppDomainGroup-group.com.apple.deviceactivity%'
  OR relativePath LIKE '%DeviceActivity%'
  OR relativePath LIKE 'Library/Preferences/%ScreenTime%'
  OR domain LIKE '%ScreenTime%'
)
ORDER BY relativePath
"""

def blob_path(base, fid): return os.path.join(base, fid[:2], fid)
con = sqlite3.connect(MANIFEST_DB); cur = con.cursor()
rows = cur.execute(SQL).fetchall(); con.close()
print("Candidatos flags=1:", len(rows))
present = []
for fid, dom, rel, flags in rows:
    src = blob_path(BACKUP_DIR, fid)
    if os.path.isfile(src) and os.path.getsize(src) > 0:
        present.append((fid, dom, rel, src))
print("Com blob presente:", len(present))
for i, (_, dom, rel, src) in enumerate(present[:12], 1):
    print(f"{i:02d}) {dom} | {rel} -> {src}")
