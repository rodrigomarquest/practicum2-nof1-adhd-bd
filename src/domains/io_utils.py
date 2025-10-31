from __future__ import annotations

import io
import zipfile
import json
import csv
import fnmatch
from pathlib import Path, PurePosixPath
import pandas as pd
import pyzipper


# ---------------- Exceptions ----------------
class ZipOpenError(RuntimeError): ...


class CsvReadError(RuntimeError): ...


# ---------------- Helpers ----------------
def _open_zip_any(zip_path: Path, password: str | None):
    """Abre ZIP com suporte a ZipCrypto (zipfile) e AES (pyzipper)."""
    zp = Path(zip_path)
    try:
        zf = pyzipper.AESZipFile(zp, mode="r")
        if password:
            zf.pwd = password.encode("utf-8")
        _ = zf.namelist()
        return zf
    except Exception as e_py:
        try:
            zf2 = zipfile.ZipFile(zp, mode="r")
            _ = zf2.namelist()
            return zf2
        except Exception as e_zip:
            raise ZipOpenError(f"Failed to open ZIP '{zp}': {e_py!r} / {e_zip!r}")


def _norm_posix_lower(name: str) -> str:
    """Normaliza separador e caixa para matching robusto."""
    return str(PurePosixPath(name)).lower()


def _find_member_any(names: list[str], patterns: list[str]) -> str | None:
    """Retorna o primeiro membro que casar com qualquer padrão (case-insensitive)."""
    names_n = [_norm_posix_lower(n) for n in names]
    pats_n = [_norm_posix_lower(p) for p in patterns]
    for pat in pats_n:
        for n, nn in zip(names, names_n):
            if fnmatch.fnmatch(nn, pat):
                return n
    return None


def read_tabular_text(text: str, **read_csv_kwargs) -> pd.DataFrame:
    """Lê CSV/TSV a partir de texto, tentando sniffer de delimitador."""
    if "sep" not in read_csv_kwargs:
        try:
            first = text.splitlines()[0] if text else ","
            dialect = csv.Sniffer().sniff(first)
            read_csv_kwargs["sep"] = dialect.delimiter
        except Exception:
            pass
    return pd.read_csv(io.StringIO(text), **read_csv_kwargs)


def zip_list_members(container: Path, password: str | None):
    """Lista membros de um ZIP (útil para depuração)."""
    zf = _open_zip_any(container, password)
    try:
        return zf.namelist()
    finally:
        zf.close()


# ---------------- Main API ----------------
def read_csv_sniff(
    container: str | Path,  # caminho ZIP/diretório/arquivo
    member_glob: str | list[str],  # padrão(ões) para buscar dentro do ZIP/dir
    password: str | None = None,  # senha do ZIP, se houver
    *,
    encoding: str | None = "utf-8",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Lê tabela (CSV/TSV/JSON) de:
      • ZIP (AES/ZipCrypto), diretório (rglob) ou arquivo direto;
      • 'member_glob' pode ser str ou lista de padrões; case-insensitive; cobre subdirs;
      • se a extensão alvo for .json, usa json_normalize.
    """
    # Guard-rails contra inversão acidental de argumentos
    if isinstance(container, (list, tuple)):
        raise TypeError(
            "read_csv_sniff(container=..., member_glob=...) recebeu uma LISTA em 'container'. "
            "Use argumentos nomeados corretamente."
        )
    if isinstance(member_glob, (str, Path)):
        patterns = [str(member_glob)]
    elif isinstance(member_glob, (list, tuple)):
        patterns = [str(p) for p in member_glob]
    else:
        raise TypeError("member_glob deve ser str ou lista de str.")

    container = Path(container)

    # ---- ZIP ----
    if container.is_file() and container.suffix.lower() == ".zip":
        zf = _open_zip_any(container, password)
        try:
            names = zf.namelist()
            target = _find_member_any(names, patterns)
            if not target:
                # Tenta curingas genéricos por subdiretórios
                wild = []
                for p in patterns:
                    p = p.replace("\\", "/")
                    base = p.split("/")[-1]
                    wild += [f"**/*{base}*", f"*{base}*"]
                target = _find_member_any(names, wild) if wild else None
            if not target:
                raise CsvReadError(f"No members matching {patterns} in {container}")

            # lê bytes (pyzipper e zipfile têm assinaturas diferentes para senha)
            try:
                with zf.open(target, "r") as fh:
                    data = fh.read()
            except TypeError:
                with zf.open(
                    target, "r", pwd=(password.encode("utf-8") if password else None)
                ) as fh:
                    data = fh.read()
        finally:
            zf.close()

        t_norm = _norm_posix_lower(target)
        if t_norm.endswith(".json"):
            obj = json.loads(data.decode(encoding or "utf-8", errors="replace"))
            return pd.json_normalize(obj)
        else:
            text = data.decode(encoding or "utf-8", errors="replace")
            return read_tabular_text(text, **read_csv_kwargs)

    # ---- Diretório ----
    if container.is_dir():
        files = list(container.rglob("*"))
        table = {_norm_posix_lower(str(p)): p for p in files if p.is_file()}
        target_path = None
        for pat in patterns:
            pat_l = _norm_posix_lower(pat)
            for key, p in table.items():
                if fnmatch.fnmatch(key, pat_l) or key.endswith(pat_l):
                    target_path = p
                    break
            if target_path:
                break
        if not target_path:
            raise CsvReadError(f"No files matching {patterns} in directory {container}")

        if target_path.suffix.lower() == ".json":
            obj = json.loads(target_path.read_text(encoding=encoding or "utf-8"))
            return pd.json_normalize(obj)
        return pd.read_csv(target_path, **read_csv_kwargs)

    # ---- Arquivo plano ----
    if container.suffix.lower() == ".json":
        obj = json.loads(container.read_text(encoding=encoding or "utf-8"))
        return pd.json_normalize(obj)
    return pd.read_csv(container, **read_csv_kwargs)
