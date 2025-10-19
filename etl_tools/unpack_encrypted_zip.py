# etl_tools/unpack_encrypted_zip.py
from __future__ import annotations
import argparse
import os
from pathlib import Path, PurePosixPath
import pyzipper, zipfile

def _safe_join(base: Path, member: str) -> Path:
    # evita path traversal, normaliza separadores e remove barras iniciais
    rel = PurePosixPath(member).as_posix().lstrip("/")
    return base.joinpath(*Path(rel).parts)

def _open_zip(zip_path: Path, password: str | None):
    """
    Abre o ZIP com suporte a AES (pyzipper) e ZipCrypto (zipfile) como fallback.
    Retorna (zf, mode) onde mode é 'pyzipper' ou 'zipfile'.
    """
    # tente pyzipper (AES)
    try:
        zf = pyzipper.AESZipFile(zip_path, "r")
        if password:
            zf.pwd = password.encode("utf-8")
        _ = zf.namelist()
        return zf, "pyzipper"
    except Exception:
        # fallback zipfile (ZipCrypto ou sem senha)
        zf2 = zipfile.ZipFile(zip_path, "r")
        _ = zf2.namelist()
        return zf2, "zipfile"

def main():
    ap = argparse.ArgumentParser("unpack-encrypted-zip")
    ap.add_argument("--zip", required=True, help="Caminho do arquivo ZIP")
    ap.add_argument("--out", required=True, help="Diretório de saída")
    ap.add_argument("--password", help="Senha do ZIP (ou use ZEPP_ZIP_PASSWORD)")

    # ✅ use dest com underscore
    ap.add_argument("--only-csv", dest="only_csv", action="store_true",
                    help="Extrair apenas arquivos .csv")
    ap.add_argument("--include", help="Lista de pastas top-level a incluir (ex.: 'HEARTRATE,SLEEP')")
    ap.add_argument("--no-overwrite", dest="no_overwrite", action="store_true",
                    help="Não sobrescrever arquivos já existentes")
    ap.add_argument("--list-only", dest="list_only", action="store_true",
                    help="Somente gerar manifest e sair")
    args = ap.parse_args()


    zip_path = Path(args.zip)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    password = args.password or os.getenv("ZEPP_ZIP_PASSWORD") or None
    include_set = None
    if args.include:
        include_set = {s.strip().strip("/").upper() for s in args.include.split(",") if s.strip()}

    zf, mode = _open_zip(zip_path, password)

    # manifest
    names = zf.namelist()
    (out_dir / "manifest.txt").write_text("\n".join(names), encoding="utf-8")

    if args.list_only:
        print(f"{len(names)} entradas listadas em {out_dir/'manifest.txt'}")
        zf.close()
        return

    n_files, n_dirs, n_skipped = 0, 0, 0

    # use .infolist() para detectar diretório com segurança
    for info in zf.infolist():
        name = info.filename

        # filtro por pasta top-level (se fornecido)
        top = PurePosixPath(name).parts[0].upper() if PurePosixPath(name).parts else ""
        if include_set is not None and top not in include_set:
            n_skipped += 1
            continue

        # pular diretórios
        is_dir = False
        try:
            is_dir = info.is_dir()
        except AttributeError:
            is_dir = name.endswith("/")
        if is_dir:
            (out_dir / PurePosixPath(name)).mkdir(parents=True, exist_ok=True)
            n_dirs += 1
            continue

        # filtro por extensão
        if args.only_csv and not name.lower().endswith(".csv"):
            n_skipped += 1
            continue

        target = _safe_join(out_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)

        if args.no_overwrite and target.exists():
            n_skipped += 1
            continue
        
        if mode == "pyzipper":
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())
        else:  # zipfile
            # zipfile usa pwd= no open() se precisar de senha (raro aqui)
            kwargs = {}
            if password:
                kwargs["pwd"] = password.encode("utf-8")
            with zf.open(info, "r", **kwargs) as src, open(target, "wb") as dst:
                dst.write(src.read())
        n_files += 1

    zf.close()
    print(f"✅ Unpacked → {out_dir}")
    print(f"   files: {n_files} | dirs: {n_dirs} | skipped: {n_skipped}")
    print(f"   manifest: {out_dir/'manifest.txt'}")

if __name__ == "__main__":
    main()
