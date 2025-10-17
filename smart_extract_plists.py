#!/usr/bin/env python3
import os, getpass, sqlite3, inspect, traceback

from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR   = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
MANIFEST_DB  = r"C:\dev\practicum2-nof1-adhd-bd\decrypted_output\Manifest_decrypted.db"
OUT_DIR      = os.path.abspath(r"decrypted_output\screentime_plists")
os.makedirs(OUT_DIR, exist_ok=True)

# targets
RELATIVE_PATHS = [
    ("HomeDomain", "Library/Preferences/com.apple.DeviceActivity.plist", "DeviceActivity.plist"),
    ("HomeDomain", "Library/Preferences/com.apple.ScreenTimeAgent.plist", "ScreenTimeAgent.plist"),
]
FILEIDS = {
    "DeviceActivity.plist":  "abb3e415ff2e3c5bb095c3a3be8ad729e2743033",
    "ScreenTimeAgent.plist": "8b11cec6faa5c449bfeba31d7610f57d5769a4b5",
}

def blob_path(base, fid): 
    return os.path.join(base, fid[:2], fid)

def row_by_fileid(fid):
    con = sqlite3.connect(MANIFEST_DB); con.row_factory = sqlite3.Row
    cur = con.cursor()
    r = cur.execute("SELECT fileID, domain, relativePath, flags, file FROM Files WHERE fileID=? AND flags=1", (fid,)).fetchone()
    con.close()
    return r

def main():
    if not os.path.isfile(MANIFEST_DB):
        print("❌ Manifest_decrypted.db não encontrado — rode decrypt_manifest.py primeiro.")
        return

    pw = getpass.getpass("Enter iTunes / backup passphrase: ")
    b  = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)

    # introspecção
    names = dir(b)
    has_extract_files = hasattr(b, "extract_files")
    has_extract_file_as_bytes = hasattr(b, "extract_file_as_bytes")
    has_decrypt_inner = hasattr(b, "_decrypt_inner_file")
    has_decrypt_to_disk = hasattr(b, "_decrypt_file_to_disk")

    print("🔎 Métodos:", [n for n in names if any(k in n.lower() for k in ("file", "extract", "save"))])

    for domain, rel, out_name in RELATIVE_PATHS:
        out_path = os.path.join(OUT_DIR, out_name)
        fid = FILEIDS[out_name]
        print(f"\n→ Extraindo {out_name} | {domain} | {rel} | fid={fid}")

        # Tenta Strategy 1: extract_files(file_ids=[...])
        if has_extract_files:
            try:
                sig = str(inspect.signature(b.extract_files))
                # tenta via file_ids
                if "file_ids" in sig:
                    b.extract_files(output_folder=OUT_DIR, file_ids=[fid])
                    if os.path.isfile(out_path):
                        print(f"✅ Strategy 1 (file_ids) OK: {out_path}")
                        continue
                # tenta via domain + relative_paths
                if "relative_paths" in sig and "domain" in sig:
                    b.extract_files(output_folder=OUT_DIR, domain=domain, relative_paths=[rel])
                    if os.path.isfile(out_path):
                        print(f"✅ Strategy 1 (domain+relative_paths) OK: {out_path}")
                        continue
            except Exception as e:
                print(f"… Strategy 1 falhou: {e}")

        # Tenta Strategy 2: extract_file_as_bytes(relative_path, domain_like)
        if has_extract_file_as_bytes:
            try:
                sig2 = str(inspect.signature(b.extract_file_as_bytes))
                if "(self, relative_path" in sig2 and "domain_like" in sig2:
                    blob = b.extract_file_as_bytes(rel, domain)
                    if blob:
                        with open(out_path, "wb") as f:
                            f.write(blob)
                        print(f"✅ Strategy 2 OK: {out_path}")
                        continue
            except Exception as e:
                print(f"… Strategy 2 falhou: {e}")

        # Tenta Strategy 3: “baixo nível” com file_bplist do Manifest
        try:
            r = row_by_fileid(fid)
            if not r:
                print("… Manifest não retornou row para esse fileID/flags=1")
                raise RuntimeError("manifest_row_missing")

            src = blob_path(BACKUP_DIR, fid)
            if not os.path.isfile(src):
                print("… Blob fisicamente ausente:", src)
                raise RuntimeError("blob_missing")

            with open(src, "rb") as f:
                raw = f.read()
            if not raw:
                print("… Blob vazio:", src)
                raise RuntimeError("blob_empty")

            # Preferir decrypt_to_disk se aceitar (raw, out_path, file_bplist)
            used = False
            if has_decrypt_to_disk:
                try:
                    sig3 = str(inspect.signature(b._decrypt_file_to_disk))
                    if "file_bplist" in sig3 and "output_filename" in sig3:
                        # algumas builds usam kwargs
                        b._decrypt_file_to_disk(file_bplist=r["file"], output_filename=out_path, filedata=raw)  # tenta com kwargs comuns
                        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                            print(f"✅ Strategy 3 (_decrypt_file_to_disk) OK: {out_path}")
                            used = True
                    else:
                        # fallback: tentar posicional comum (filedata, output_filename, file_bplist)
                        try:
                            b._decrypt_file_to_disk(raw, out_path, r["file"])
                            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                                print(f"✅ Strategy 3 (positional) OK: {out_path}")
                                used = True
                        except Exception:
                            pass
                except Exception as e:
                    print(f"… _decrypt_file_to_disk falhou: {e}")

            if not used and has_decrypt_inner:
                try:
                    sig4 = str(inspect.signature(b._decrypt_inner_file))
                    # tenta posicional (filedata, file_bplist)
                    dec = None
                    try:
                        dec = b._decrypt_inner_file(raw, r["file"])
                    except TypeError:
                        # tenta kwargs invertidos
                        try:
                            dec = b._decrypt_inner_file(filedata=raw, file_bplist=r["file"])
                        except Exception:
                            pass
                    if dec:
                        with open(out_path, "wb") as f:
                            f.write(dec)
                        print(f"✅ Strategy 3 (_decrypt_inner_file) OK: {out_path}")
                        continue
                except Exception as e:
                    print(f"… _decrypt_inner_file falhou: {e}")

            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                continue
            else:
                raise RuntimeError("all_strategies_failed")

        except Exception as e:
            print("❌ Todas as estratégias falharam para", out_name)
            traceback.print_exc()

    print("\n📂 Verifique:", OUT_DIR)
    print("Se algum .plist foi gerado, me diga quais; eu já gero o parser para CSV diário.")

if __name__ == "__main__":
    main()
