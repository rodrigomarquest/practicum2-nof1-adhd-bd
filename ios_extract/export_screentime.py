#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_screentime.py
Author: Rodrigo's helper

What it does:
1) Optionally decrypts an iOS backup using iphone-backup-decrypt (CLI).
2) Scans decrypted backup for Screen Time–related SQLite DBs:
   - RMAdminStore-*.sqlite, knowledgeC.db, CoreDuet, DeviceActivity, etc.
3) Exports ALL tables from those DBs to CSV (UTF-8), with an index file.
4) (Optional) Attempts a light "summary" if recognizable columns exist.

Usage examples (PowerShell):
    $env:IOS_BACKUP_PASSWORD = "YOUR_PASSWORD"
    python export_screentime.py --do-decrypt
    python export_screentime.py --decrypted-dir "C:\\path\\to\\_decrypted"

Requirements:
- Python 3.x
- sqlite3 (stdlib)
- iphone-backup-decrypt installed (pip) if using --do-decrypt
"""

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from etl_modules.iphone_backup.iphone_backup import EncryptedBackup

# --- USER-SPECIFIC DEFAULTS (as requested) ---
UUID = "00008120-000E18E21144A01E"
BACKUP_DIR = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
# Decrypted default target inside the backup directory:
DEFAULT_DECRYPTED_DIR = os.path.join(BACKUP_DIR, "_decrypted")

# --- Candidate file name patterns for Screen Time / related ---
NAME_HINTS = (
    "rmadmin",          # RMAdminStore-Local.sqlite / RMAdminStore-Cloud.sqlite
    "screentime",       # rare, but check
    "deviceactivity",   # iOS 16+ device/activity frameworks
    "knowledgec",       # knowledgeC.db (CoreDuet / Knowledge)
    "coreduet",         # assorted CoreDuet DBs
    "usage",            # generic usage
)

# Tables to prioritize for a light daily summary, if present:
TABLE_HINTS_FOR_SUMMARY = (
    "usage", "application", "app_usage", "device_activity", "screen", "activity"
)

def is_probable_sqlite(path: Path) -> bool:
    """Quick check: does the file start with SQLite header?"""
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
        return header.startswith(b"SQLite format 3\000")
    except Exception:
        return False

def export_sqlite_tables(db_path: Path, out_dir: Path, max_rows: int = 100000):
    """Export all tables from a SQLite DB to CSV."""
    exports = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            # Sanitize filename
            safe_t = re.sub(r"[^A-Za-z0-9_\-]+", "_", t)
            out_csv = out_dir / f"{db_path.name}__{safe_t}.csv"
            try:
                cur.execute(f"SELECT * FROM '{t}' LIMIT {max_rows};")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
                with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    if cols:
                        w.writerow(cols)
                    w.writerows(rows)
                exports.append((t, str(out_csv), len(rows)))
            except Exception as e:
                # Log failure but continue
                err_csv = out_dir / f"{db_path.name}__{safe_t}__ERROR.txt"
                err_csv.write_text(f"Failed exporting table '{t}' from {db_path}:\n{e}", encoding="utf-8")
    except Exception as e:
        # Entire DB failed
        err_txt = out_dir / f"{db_path.name}__ERROR.txt"
        err_txt.write_text(f"Failed opening DB {db_path}:\n{e}", encoding="utf-8")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return exports

def find_candidates(decrypted_root: Path):
    """Find candidate DB files by name hints."""
    hits = []
    for root, _, files in os.walk(decrypted_root):
        for f in files:
            lower = f.lower()
            if any(h in lower for h in NAME_HINTS):
                p = Path(root) / f
                hits.append(p)
    return hits

def light_daily_summary(db_path: Path, out_dir: Path):
    """
    Try a light daily summary from 'likely' tables if they exist and have
    recognizable columns. This is best-effort and schema-agnostic.
    """
    summary_rows = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]

        for t in tables:
            if not any(h in t.lower() for h in TABLE_HINTS_FOR_SUMMARY):
                continue
            # Probe table columns
            cur.execute(f"PRAGMA table_info('{t}')")
            cols = [r[1].lower() for r in cur.fetchall()]
            if not cols:
                continue

            # Heuristic: try to pick timestamp-like and duration-like columns
            ts_cols = [c for c in cols if "time" in c or "date" in c or c in ("ts","timestamp")]
            dur_cols = [c for c in cols if "duration" in c or "seconds" in c or c.endswith("_sec")]
            bundle_cols = [c for c in cols if "bundle" in c or "app" in c or "process" in c or "domain" in c]

            # Build a generic query
            sel_cols = []
            # prefer up to 1 ts, 1 duration, 1 bundle
            if ts_cols:
                sel_cols.append(ts_cols[0])
            if dur_cols:
                sel_cols.append(dur_cols[0])
            if bundle_cols:
                sel_cols.append(bundle_cols[0])
            if not sel_cols:
                continue

            try:
                cur.execute(f"SELECT {', '.join([f'\"{c}\"' for c in sel_cols])} FROM '{t}' LIMIT 50000;")
                for row in cur.fetchall():
                    # Normalize:
                    ts_val = row[0] if ts_cols else None
                    dur_val = row[1] if dur_cols and len(row) > 1 else None
                    bundle_val = row[2] if bundle_cols and len(row) > 2 else None
                    summary_rows.append({
                        "db": db_path.name,
                        "table": t,
                        "timestamp_raw": ts_val,
                        "duration_raw": dur_val,
                        "bundle_like": bundle_val
                    })
            except Exception:
                # ignore if fails
                pass
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if summary_rows:
        out_csv = out_dir / f"{db_path.name}__LIGHT_SUMMARY.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["db","table","timestamp_raw","duration_raw","bundle_like"])
            w.writeheader()
            w.writerows(summary_rows)
        return str(out_csv), len(summary_rows)
    return None, 0

def do_decrypt(backup_dir: Path, decrypted_dir: Path, password: str):
    """
    Calls iphone-backup-decrypt CLI to create a readable copy.
    Tries both `iphone_backup_decrypt` entrypoint and `python -m iphone_backup_decrypt`.
    """
    decrypted_dir.mkdir(parents=True, exist_ok=True)
    # Common CLIs:
    cmds = [
        ["iphone_backup_decrypt", "--backup-dir", str(backup_dir), "--outdir", str(decrypted_dir), "--password", password],
        [sys.executable, "-m", "iphone_backup_decrypt", "--backup-dir", str(backup_dir), "--outdir", str(decrypted_dir), "--password", password],
        # Some forks use a 'decrypt.py' script; last resort if present in PATH/cwd.
    ]
    last_err = None
    for cmd in cmds:
        try:
            print(f"[decrypt] Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            print("[decrypt] Done.")
            return True
        except FileNotFoundError as e:
            last_err = e
            continue
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    print("[decrypt] Failed to run iphone-backup-decrypt. Is it installed and in PATH?")
    if last_err:
        print(f"[decrypt] Last error: {last_err}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Export iOS Screen Time related DBs from decrypted backup.")
    parser.add_argument("--backup-dir", default=BACKUP_DIR, help="Path to the iOS backup (encrypted/original).")
    parser.add_argument("--decrypted-dir", default=None, help="Path to decrypted backup. Default: <backup>\\_decrypted")
    parser.add_argument("--out-base", default=None, help="Base output dir. Default: <backup>\\exports_screentime\\YYYYMMDD_HHMMSS")
    parser.add_argument("--do-decrypt", action="store_true", help="Attempt to decrypt the backup before exporting.")
    parser.add_argument("--password", default=os.environ.get("IOS_BACKUP_PASSWORD", ""), help="Backup password (or set IOS_BACKUP_PASSWORD env var).")
    parser.add_argument("--max-rows", type=int, default=100000, help="Max rows per table export.")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir).expanduser()
    if not backup_dir.exists():
        print(f"[err] Backup dir not found: {backup_dir}")
        sys.exit(1)

    decrypted_dir = Path(args.decrypted_dir) if args.decrypted_dir else Path(DEFAULT_DECRYPTED_DIR)
    out_base = Path(args.out_base) if args.out_base else (backup_dir / "exports_screentime" / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_base.mkdir(parents=True, exist_ok=True)

    # Step 1: decrypt if requested or if decrypted dir missing
    if args.do_decrypt or not decrypted_dir.exists():
        if not args.password:
            print("[warn] No password provided. Set --password or IOS_BACKUP_PASSWORD to decrypt.")
        else:
            ok = do_decrypt(backup_dir, decrypted_dir, args.password)
            if not ok:
                print("[warn] Decrypt step failed or skipped. If you already have decrypted files, pass --decrypted-dir.")
    print(f"[info] Using decrypted dir: {decrypted_dir}")
    if not decrypted_dir.exists():
        print(f"[err] Decrypted dir does not exist: {decrypted_dir}\n"
              f"      Provide --decrypted-dir or run with --do-decrypt and a valid password.")
        sys.exit(2)

    # Step 2: find candidate DBs
    candidates = find_candidates(decrypted_dir)
    # Filter to likely sqlite files by header (best-effort)
    sqlite_candidates = [p for p in candidates if is_probable_sqlite(p)]
    # It's possible some DBs don't match header due to WAL/SHM; include *.sqlite explicitly:
    sqlite_candidates = list({*sqlite_candidates, *[p for p in candidates if p.suffix.lower() in (".sqlite", ".db")]} )

    # Make subdir for DB exports
    db_out_dir = out_base / "db_exports"
    db_out_dir.mkdir(parents=True, exist_ok=True)

    # Index file
    index_path = out_base / "_export_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as idxfh:
        w = csv.writer(idxfh)
        w.writerow(["db_path", "table", "csv_path", "rowcount", "light_summary_csv", "light_summary_rows"])

        if not sqlite_candidates:
            print("[warn] No candidate DBs found. Try searching manually or verify that decryption worked.")
        for db in sorted(set(sqlite_candidates)):
            print(f"[scan] {db}")
            # Export all tables
            exports = export_sqlite_tables(db, db_out_dir, max_rows=args.max_rows)
            # Light summary
            summary_csv, summary_rows = light_daily_summary(db, db_out_dir)

            if exports:
                for t, csv_path, nrows in exports:
                    w.writerow([str(db), t, csv_path, nrows, summary_csv or "", summary_rows])
            else:
                w.writerow([str(db), "", "", 0, summary_csv or "", summary_rows])

    print("\n[done] Export finished.")
    print(f"[done] Output folder: {out_base}")
    print(f"[done] Index file: {index_path}")
    print("Tip: Import the *_LIGHT_SUMMARY.csv to inspect quick signals; full tables are in db_exports/.")

if __name__ == "__main__":
    main()
