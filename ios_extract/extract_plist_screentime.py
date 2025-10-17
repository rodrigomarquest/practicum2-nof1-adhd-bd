#!/usr/bin/env python3
import os, getpass

from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
OUT_DIR    = os.path.abspath(r"decrypted_output\screentime_plists")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    ("Library/Preferences/com.apple.DeviceActivity.plist",  "DeviceActivity.plist"),
    ("Library/Preferences/com.apple.ScreenTimeAgent.plist", "ScreenTimeAgent.plist"),
]

def main():
    pw = getpass.getpass("Enter iTunes / backup passphrase: ")
    b  = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)

    for rel, out_name in TARGETS:
        out_path = os.path.join(OUT_DIR, out_name)
        try:
            # ✅ tua versão aceita só o relativePath (string)
            blob = b.extract_file_as_bytes(rel)
            if not blob:
                print(f"❌ Empty blob: {rel}")
                continue
            with open(out_path, "wb") as f:
                f.write(blob)
            print(f"✅ Extracted {out_name} → {out_path}")
        except Exception as e:
            print(f"❌ Failed {out_name}: {e}")

if __name__ == "__main__":
    main()
