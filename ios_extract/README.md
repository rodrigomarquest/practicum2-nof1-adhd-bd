iOS Screen Time & KnowledgeC Extraction (Runbook)

Encrypted local backup → decrypted Manifest → Screen Time plists → KnowledgeC (when present) → CSV for ETL.

Status (Oct 2025): Manifest_decrypted.db validated; DeviceActivity.plist & ScreenTimeAgent.plist extracted; waiting for KnowledgeC.db after Screen Time reactivation and a few minutes of real app usage.

0. Requirements

Python 3.10+ (Windows, VS Code / Git Bash)

Virtual env: .venv activated

python -m venv .venv
source .venv/Scripts/activate

Packages

python -m pip install --upgrade pip setuptools wheel
python -m pip install iphone-backup-decrypt==0.9.0 pycryptodome

Local encrypted backup (Finder/iTunes → This Computer + Encrypt local backup)
Default path used here:

C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E

fastpbkdf2 is optional. If it fails to build, ignore it.

1. Files & Layout
   ios_extract/
   ├─ decrypt_manifest.py # Decrypt Manifest.db and validate (SQLite)
   ├─ quick_post_backup_probe.py # Probe candidates w/ flags=1 + on-disk blobs
   ├─ smart_extract_plists.py # Adaptive extractor for DeviceActivity & ScreenTimeAgent
   ├─ plist_to_usage_csv.py # Heuristic parser → usage_daily_from_plists.csv
   ├─ extract_knowledgec.py # Extract CoreDuet/KnowledgeC.db (when present)
   ├─ parse_knowledgec_usage.py # (to be added) Parse KnowledgeC → usage_daily_from_knowledgec.csv
   └─ README.md # This file

Outputs land in the repo-level decrypted_output/:

decrypted_output/
├─ Manifest_decrypted.db
├─ screentime_plists/
│ ├─ DeviceActivity.plist
│ ├─ ScreenTimeAgent.plist
│ └─ usage_daily_from_plists.csv # may be empty (settings-only snapshot)
└─ knowledgec/
└─ KnowledgeC.db # appears after normal app usage

2. Quick Start (happy path)

Run from repo root with .venv active.

2.1 Decrypt and validate Manifest
python ios_extract/decrypt_manifest.py

# Expect: PRAGMA integrity_check -> ('ok',) and tables ['Files','Properties']

2.2 Probe for usable files (flags=1 + blob present)
python ios_extract/quick_post_backup_probe.py

# Look for 'Library/CoreDuet/Knowledge/KnowledgeC.db' or the two Screen Time plists

2.3 Extract Screen Time plists (if KnowledgeC not present yet)
python ios_extract/smart_extract_plists.py
python ios_extract/plist_to_usage_csv.py

# Output: decrypted_output/screentime_plists/usage_daily_from_plists.csv (may be empty)

2.4 Extract KnowledgeC when it appears

After re-enabling Screen Time on iPhone and using a few apps for ~10–15 min:

python ios_extract/extract_knowledgec.py

# Then (once schema confirmed)

python ios_extract/parse_knowledgec_usage.py

# Target: decrypted_output/usage_daily_from_knowledgec.csv

You can also use the Makefile shortcuts:

make decrypt probe extract-plists plist-csv
make extract-knowledgec parse-knowledgec

3. How to force KnowledgeC to appear

On iPhone: Settings → Screen Time → Turn Off → reboot → Turn On.

Use a few apps normally (Safari, Messages, WhatsApp, YouTube) for ~10–15 min; lock/unlock a few times.

Connect by cable → do encrypted local backup again → rerun probe:

python ios_extract/decrypt_manifest.py
python ios_extract/quick_post_backup_probe.py

Expect a hit such as HomeDomain | Library/CoreDuet/Knowledge/KnowledgeC.db.

4. ETL Integration

Once you have either usage_daily_from_plists.csv or usage_daily_from_knowledgec.csv, run ETL:

python etl/etl_pipeline.py \
 --cutover 2023-04-10 \
 --tz_before America/Sao_Paulo \
 --tz_after Europe/Dublin

Outputs (example): features_daily.csv, etl_qc_summary.csv.
Downstream notebooks (Kaggle): 01_feature_engineering.ipynb, 02_model_training.ipynb, 03_shap_analysis.ipynb, 04_rule_based_baseline.ipynb.

5. Troubleshooting (v0.9.0 library quirks)
   Symptom Cause Fix
   ModuleNotFoundError: iphone_backup_decrypt venv not active or pkg not installed Activate .venv, then python -m pip install iphone-backup-decrypt==0.9.0
   PBKDF2 very slow on first open Normal on encrypted backup Be patient (optionally try fastpbkdf2)
   extract_file(...) / extract_file_as_bytes(...) signature errors API differs across builds Use smart_extract_plists.py (multi-strategy)
   Only directories / 0 blobs extracted “Only in iCloud” or Screen Time not yet repopulated Do encrypted local backup; enable Screen Time; use apps 10–15 min; back up again
   Plist CSV is empty Snapshot contains only settings Wait for KnowledgeC and parse it (per-app daily usage)
6. Privacy & Security

Never commit decrypted outputs (decrypted_output/, .plist, .db, .sqlite\*).

Keep backup passphrase out of scripts and version control.

ETL uses anonymised/aggregated features; raw personal data remains local only.

.gitignore already blocks PII/binaries; use make deepclean to remove local decrypted files.

7. Minimal FAQs

Q: Do I need to sign into iTunes/Apple ID?
A: No. For encrypted local backups you don’t need to be signed in.

Q: Will toggling Dark Mode or using the phone break the backup?
A: No. You can change settings while the backup runs; just keep the device connected and unlocked.

Q: The probe still doesn’t show KnowledgeC.db.
A: Use apps for longer, ensure Screen Time is on, and repeat an encrypted local backup.

8. Useful one-liners

List first 12 tables of any SQLite you extracted:

python - << 'PY'
import sqlite3, glob
for p in glob.glob(r"decrypted_output/\*_/_.db", recursive=True):
con=sqlite3.connect(p); c=con.cursor()
c.execute("PRAGMA integrity_check;"); print(p, "->", c.fetchone())
c.execute("SELECT name FROM sqlite_master WHERE type='table'");
print("tables:", [r[0] for r in c.fetchall()][:12]); con.close()
PY

Show the two Screen Time plists as JSON previews:

python - << 'PY'
import os, plistlib, json, glob
for p in glob.glob(r"decrypted_output/screentime_plists/\*.plist"):
with open(p,'rb') as f: d=plistlib.load(f)
print("\n==", os.path.basename(p));
print(json.dumps({k:d[k] for k in list(d)[:10]}, indent=2)[:1200])
PY

9. Change log anchor

v2.0-pre-ethics (2025-10-17) — Added iOS extraction module, adaptive plist extraction, plist CSV heuristic, Makefile targets, hardened .gitignore.
