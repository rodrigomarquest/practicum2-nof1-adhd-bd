#!/usr/bin/env python3
import os, sqlite3, sys, getpass, traceback, shutil
from datetime import datetime
from pathlib import Path
from iphone_backup_decrypt import EncryptedBackup

# --- Config por env (Makefile chama com OUT_DIR/BACKUP_DIR/BACKUP_PASSWORD) ---
BACKUP_DIR       = os.environ.get("BACKUP_DIR", r"C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E")
OUT_DIR          = os.path.abspath(os.environ.get("OUT_DIR", "decrypted_output"))
BACKUP_PASSWORD  = os.environ.get("BACKUP_PASSWORD", "")  # se vazio, cai no getpass

# nomes padrão
CANON_NAME       = "Manifest_decrypted.db"
STAMP            = datetime.now().strftime("%Y%m%d_%H%M")
STAMPED_NAME     = f"Manifest_decrypted_{STAMP}.db"

OUT_PATH         = Path(OUT_DIR)
CANON_PATH       = OUT_PATH / CANON_NAME
STAMPED_PATH     = OUT_PATH / STAMPED_NAME
MARK_LAST_MAN    = OUT_PATH / "last_decrypted_manifest.txt"
MARK_LAST_SRC    = OUT_PATH / "last_backup_dir.txt"

def fail(msg: str, tb: bool = False):
    if tb:
        print("\nTraceback:\n" + traceback.format_exc())
    print(f"\n❌ {msg}")
    sys.exit(1)

def info(msg: str):
    print(msg, flush=True)

def validate_sqlite(db_path: Path):
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        ic = cur.fetchone()
        print("🧪 integrity_check:", ic)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        print("📋 tables:", tables)
        # presença de Files nem sempre é garantida dependendo da versão, mas ajuda:
        if "Files" not in tables and "files" not in [t.lower() for t in tables]:
            print("⚠️  Nota: tabela 'Files' não encontrada — possível variação de schema; prossiga com probe que faz auto-detecção.")
        conn.close()
    except Exception:
        fail("Manifest não é um SQLite válido.", tb=True)

def main():
    print("📁 Backup dir:", BACKUP_DIR)
    print("📦 OUT_DIR:", OUT_PATH)

    if not os.path.isdir(BACKUP_DIR):
        fail("Backup directory não encontrado.")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # senha: env var ou prompt
    pw = BACKUP_PASSWORD or getpass.getpass("Enter iTunes / backup passphrase: ")

    # abrir backup criptografado
    try:
        print("🔐 Abrindo backup criptografado (PBKDF2 pode levar um tempo)...")
        b = EncryptedBackup(backup_directory=BACKUP_DIR, passphrase=pw)
    except Exception:
        fail("Não consegui abrir o backup. Verifique a senha e se o backup está criptografado.", tb=True)

    # salvar manifest em nome DATADO (não sobrescreve o anterior)
    try:
        print(f"💾 Salvando manifest decriptado (datado): {STAMPED_PATH}")
        b.save_manifest_file(output_filename=str(STAMPED_PATH))
    except Exception:
        fail("Falha ao salvar Manifest decriptado. O backup está completo e criptografado?", tb=True)

    if not STAMPED_PATH.exists() or STAMPED_PATH.stat().st_size == 0:
        fail("Arquivo datado do Manifest não foi criado ou está vazio.")

    # validar SQLite
    validate_sqlite(STAMPED_PATH)

    # atualizar canônico com backup do anterior (sem perder histórico)
    try:
        if CANON_PATH.exists() and CANON_PATH.stat().st_size > 0:
            backup_old = OUT_PATH / f"{CANON_PATH.stem}.prev_{STAMP}.db"
            shutil.move(str(CANON_PATH), str(backup_old))
            print(f"↪️  Backup do canônico anterior: {backup_old}")
        shutil.copy2(str(STAMPED_PATH), str(CANON_PATH))
        print(f"✅ Canônico atualizado: {CANON_PATH}")
    except Exception:
        # não é fatal, pois o datado já está salvo/validado
        print("⚠️  Não consegui atualizar o canônico; use o datado. (Detalhes acima)")

    # gravar marcadores para os próximos passos
    try:
        MARK_LAST_MAN.write_text(str(STAMPED_PATH), encoding="utf-8")
        MARK_LAST_SRC.write_text(str(BACKUP_DIR), encoding="utf-8")
        print(f"📝 last_decrypted_manifest.txt → {MARK_LAST_MAN}")
        print(f"📝 last_backup_dir.txt        → {MARK_LAST_SRC}")
    except Exception:
        print("⚠️  Não consegui escrever marcadores (last_*). Continuando...")

    print("\n✅ Done. Manifest OK em:")
    print("   • Datado :", STAMPED_PATH)
    print("   • Canônico:", CANON_PATH if CANON_PATH.exists() else "(não atualizado)")

if __name__ == "__main__":
    main()
