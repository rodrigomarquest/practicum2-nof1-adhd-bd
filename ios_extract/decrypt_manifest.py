#!/usr/bin/env python3
import os, sqlite3, sys, getpass, traceback
from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
OUT_DIR = os.path.abspath("decrypted_output")
MANIFEST_OUT = os.path.join(OUT_DIR, "Manifest_decrypted.db")

def fail(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)

def main():
    print("📁 Backup dir:", BACKUP_DIR)
    if not os.path.isdir(BACKUP_DIR):
        fail("Backup directory not found.")

    os.makedirs(OUT_DIR, exist_ok=True)

    pw = getpass.getpass("Enter iTunes / backup passphrase: ")

    try:
        print("🔐 Opening encrypted backup (PBKDF2 step may take a while)...")
        b = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
    except Exception as e:
        print("\nTraceback:\n" + traceback.format_exc())
        fail("Could not open backup. Check the passphrase and that this backup is encrypted.")

    try:
        print("💾 Saving decrypted manifest to:", MANIFEST_OUT)
        b.save_manifest_file(output_filename=MANIFEST_OUT)
    except Exception:
        print("\nTraceback:\n" + traceback.format_exc())
        fail("Failed to save decrypted Manifest. Is the backup complete and encrypted?")

    if not os.path.isfile(MANIFEST_OUT) or os.path.getsize(MANIFEST_OUT) == 0:
        fail("Manifest_decrypted.db not created or empty.")

    try:
        print("🧪 Validating SQLite integrity...")
        conn = sqlite3.connect(MANIFEST_OUT)
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        ic = cur.fetchone()
        print("→ integrity_check:", ic)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        print("→ tables:", tables)
        conn.close()
    except Exception:
        print("\nTraceback:\n" + traceback.format_exc())
        fail("Manifest is not a valid SQLite file.")

    print("\n✅ Done. Manifest OK at:", MANIFEST_OUT)

if __name__ == "__main__":
    main()
