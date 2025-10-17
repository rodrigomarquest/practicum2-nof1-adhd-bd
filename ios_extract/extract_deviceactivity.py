#!/usr/bin/env python3
import os, sqlite3, getpass, traceback, sys
from iphone_backup_decrypt import EncryptedBackup

BACKUP_DIR = r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E"
OUT_DIR    = os.path.abspath("decrypted_output")
DST_DIR    = os.path.join(OUT_DIR, "device_activity")
MANIFEST_DB= os.path.join(OUT_DIR, "Manifest_decrypted.db")

QUERIES = [
    # Caminhos típicos do framework DeviceActivity
    "SELECT fileID, domain, relativePath FROM Files WHERE relativePath LIKE 'Library/DeviceActivity/%'",
    # Alguns iOS gravam preferências do Screen Time
    "SELECT fileID, domain, relativePath FROM Files WHERE relativePath LIKE 'Library/Preferences/%ScreenTime%'",
    # Varre qualquer ocorrência (fallback)
    "SELECT fileID, domain, relativePath FROM Files WHERE relativePath LIKE '%DeviceActivity%'",
    "SELECT fileID, domain, relativePath FROM Files WHERE domain LIKE '%ScreenTime%'",
]

def fail(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)

def main():
    if not os.path.isfile(MANIFEST_DB):
        fail("Manifest_decrypted.db não encontrado. Rode antes o decrypt_manifest.py com sucesso.")
    os.makedirs(DST_DIR, exist_ok=True)

    # Abrir backup encriptado (para extrair os ficheiros brutos por fileID)
    pw = getpass.getpass("Enter iTunes / backup passphrase: ")
    try:
        b = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
    except Exception:
        print(traceback.format_exc())
        fail("Não foi possível abrir o backup (passphrase errada ou backup inválido).")

    # Procurar candidatos no Manifest
    conn = sqlite3.connect(MANIFEST_DB)
    cur  = conn.cursor()
    seen = {}
    for q in QUERIES:
        try:
            for fileID, domain, rel in cur.execute(q):
                key = (fileID, domain, rel)
                if key in seen: 
                    continue
                seen[key] = True
        except Exception:
            # Query pode falhar em versões antigas — ignora e segue
            continue
    conn.close()

    if not seen:
        fail("Nenhum candidato a DeviceActivity/ScreenTime encontrado no Manifest.")

    print(f"🗂️  Encontrados {len(seen)} candidatos. Extraindo para: {DST_DIR}")

    ok, fail_count = 0, 0
    for (fileID, domain, rel) in seen:
        # Monta um caminho de saída legível
        safe_rel = rel.replace(":", "_").replace("\\", "/")
        out_path = os.path.join(DST_DIR, domain.replace("/", "_"), safe_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            b.extract_file(file_id=fileID, output_filename=out_path)
            ok += 1
        except Exception:
            fail_count += 1
            # Salvamos um logzinho mínimo para depurar
            with open(os.path.join(DST_DIR, "_extract_errors.log"), "a", encoding="utf-8") as f:
                f.write(f"FAIL {fileID} {domain} {rel}\n")
            continue

    print(f"✅ Extração concluída. Sucesso: {ok} | Falhas: {fail_count}")
    print(f"📂 Verifique: {DST_DIR}")
    print("   Procure por SQLite/Plist dentro de 'Library/DeviceActivity' ou 'Library/Preferences/com.apple.ScreenTime*.plist'.")

if __name__ == "__main__":
    main()
