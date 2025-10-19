#!/usr/bin/env python3
import os
import getpass
import datetime
import traceback
from iphone_backup_decrypt import EncryptedBackup

# === Config defaults ==========================================================
BACKUP_DIR = os.environ.get(
    "BACKUP_DIR",
    r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
)
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", r"decrypted_output\screentime_plists"))
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "extract_screentime.log")

TARGETS = [
    ("Library/Preferences/com.apple.DeviceActivity.plist",  "DeviceActivity.plist"),
    ("Library/Preferences/com.apple.ScreenTimeAgent.plist", "ScreenTimeAgent.plist"),
]


def log(msg: str):
    """Append a message to the local log file with timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"[{ts}] {msg}\n")


def main():
    # 1️⃣ senha do ambiente, com fallback para getpass
    pw = os.environ.get("BACKUP_PASSWORD")
    if pw:
        log("🔐 Using BACKUP_PASSWORD from environment.")
    else:
        log("⚠️ BACKUP_PASSWORD not found in environment. Asking interactively...")
        pw = getpass.getpass("Enter iTunes / backup passphrase: ")

    # 2️⃣ tenta abrir o backup
    try:
        log(f"📁 Opening encrypted backup: {BACKUP_DIR}")
        b = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
        log("✅ EncryptedBackup opened successfully.")
    except Exception as e:
        log(f"❌ Failed to open backup: {e}")
        log(traceback.format_exc())
        return

    # 3️⃣ extrai os plists
    for rel, base_name in TARGETS:
        try:
            blob = b.extract_file_as_bytes(rel)
            if not blob:
                log(f"⚠️ Empty blob for {rel}")
                continue

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            out_name = f"{os.path.splitext(base_name)[0]}_{timestamp}.plist"
            out_path = os.path.join(OUT_DIR, out_name)

            with open(out_path, "wb") as f:
                f.write(blob)

            # também grava uma cópia "canônica" sem timestamp (última extração)
            canonical = os.path.join(OUT_DIR, base_name)
            with open(canonical, "wb") as f:
                f.write(blob)

            log(f"✅ Extracted {base_name} → {out_path}")
        except Exception as e:
            log(f"❌ Failed {base_name}: {e}")
            log(traceback.format_exc())

    log("🏁 Extraction completed.")


if __name__ == "__main__":
    main()
