#!/usr/bin/env python3
import os, sqlite3, sys
from pathlib import Path

OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", "decrypted_output"))
MANIFEST = os.path.join(OUT_DIR, "Manifest_decrypted.db")
OUT_TSV  = os.path.join(OUT_DIR, "_work", "probe_screentime_sqlite.tsv")
Path(os.path.dirname(OUT_TSV)).mkdir(parents=True, exist_ok=True)

if not os.path.isfile(MANIFEST):
    print(f"❌ Manifest not found at {MANIFEST}. Run decrypt first.")
    sys.exit(1)

con = sqlite3.connect(MANIFEST)
cur = con.cursor()

Q = r"""
SELECT fileID, domain, relativePath, flags
FROM Files
WHERE flags=1
  AND relativePath IS NOT NULL
  AND (
        lower(relativePath) LIKE '%/screentime/%.db'
     OR lower(relativePath) LIKE '%/screentime/%.sqlite%'
     OR lower(relativePath) LIKE '%screentime%.db'
     OR lower(relativePath) LIKE '%screentime%.sqlite%'
     OR lower(relativePath) LIKE '%rmadminstore%.db'
     OR lower(relativePath) LIKE '%rmadminstore%.sqlite%'
  )
ORDER BY relativePath;
"""
rows = cur.execute(Q).fetchall()
con.close()

with open(OUT_TSV, "w", encoding="utf-8") as f:
    f.write("fileID\tdomain\trelativePath\tflags\n")
    for r in rows:
        f.write("\t".join(str(x or "") for x in r) + "\n")

print(f"✔ Wrote: {OUT_TSV}  (candidates={len(rows)})")
