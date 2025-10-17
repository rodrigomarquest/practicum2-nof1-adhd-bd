# screentime_ios_backup.py
# SPDX-License-Identifier: MIT
# Extração de Screen Time a partir de um backup local do iPhone (não criptografado).

import os
import sys
import sqlite3
import shutil
import glob
import csv
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

CORE_DATA_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

def _guess_backup_roots() -> List[Path]:
    roots = []
    # macOS
    mac = Path.home() / "Library" / "Application Support" / "MobileSync" / "Backup"
    if mac.exists():
        roots.append(mac)
    # Windows (iTunes clássico)
    win_appdata = os.getenv("APPDATA")
    if win_appdata:
        p = Path(win_appdata) / "Apple Computer" / "MobileSync" / "Backup"
        if p.exists():
            roots.append(p)
    # Windows (Microsoft Store)
    win_user = os.getenv("USERPROFILE")
    if win_user:
        p = Path(win_user) / "Apple" / "MobileSync" / "Backup"
        if p.exists():
            roots.append(p)
    return roots

def find_latest_backup_dir(user_supplied: Optional[str] = None) -> Path:
    if user_supplied:
        p = Path(user_supplied).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Backup path not found: {p}")
        # Se passou o <UDID> dir, use-o diretamente; se passou a pasta 'Backup', pegue o mais recente
        if (p / "Manifest.db").exists():
            return p
        # pegar subpastas
        candidates = [d for d in p.iterdir() if d.is_dir() and (d / "Manifest.db").exists()]
        if not candidates:
            raise FileNotFoundError(f"No valid backup dirs with Manifest.db under: {p}")
        return max(candidates, key=lambda d: d.stat().st_mtime)
    # auto-descoberta
    roots = _guess_backup_roots()
    candidates = []
    for r in roots:
        for d in r.iterdir():
            if d.is_dir() and (d / "Manifest.db").exists():
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError("No iOS backup found. Create a local (unencrypted) backup first.")
    return max(candidates, key=lambda d: d.stat().st_mtime)

def _query_manifest_for_screentime_sqlite(manifest_db: Path) -> Optional[Tuple[str, str]]:
    """
    Retorna (fileID, relativePath) do ScreenTime.sqlite se encontrado.
    Tentamos diferentes padrões porque o caminho pode variar entre iOS.
    """
    con = sqlite3.connect(str(manifest_db))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        # Tabela em backups modernos é 'Files' (antigos: 'Files' também, mas com layout ligeiro diferente).
        # Buscamos qualquer coisa terminando com ScreenTime.sqlite
        cur.execute("""
            SELECT fileID, domain, relativePath
            FROM Files
            WHERE relativePath LIKE '%ScreenTime.sqlite'
        """)
        row = cur.fetchone()
        if row:
            return row["fileID"], row["relativePath"]
        # fallback mais amplo (alguns iOS armazenam subcomponentes em pastas ScreenTime)
        cur.execute("""
            SELECT fileID, domain, relativePath
            FROM Files
            WHERE relativePath LIKE '%ScreenTime%.sqlite'
               OR relativePath LIKE '%ScreenTime/%'
        """)
        row = cur.fetchone()
        if row:
            return row["fileID"], row["relativePath"]
        return None
    finally:
        con.close()

def _copy_hashed_file_from_backup(backup_dir: Path, file_id: str, dest: Path) -> None:
    """
    Em backups iOS, o conteúdo fica em BACKUP/<2 first chars>/<fileID>
    """
    shard = file_id[:2]
    src = backup_dir / shard / file_id
    if not src.exists():
        # Alguns backups antigos não shard-eiam; tenta direto.
        src = backup_dir / file_id
        if not src.exists():
            raise FileNotFoundError(f"File content not found for {file_id}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

def coredata_secs_to_dt_utc(value: float) -> datetime:
    return CORE_DATA_EPOCH + timedelta(seconds=float(value))

def maybe_parse_timestamp(val):
    """
    Converte valores de tempo comuns em iOS/CoreData/Unix para ISO string UTC.
    Regras:
      - int/float grande (~> 1e9) tratado como Unix epoch (segundos/milisegundos)
      - colunas com nome 'Z*DATE*' ou 'START*'/'END*' muitas vezes são CoreData secs
    """
    if val is None:
        return None
    # já string? retorna
    if isinstance(val, str):
        return val
    # número?
    try:
        v = float(val)
    except Exception:
        return val
    # Heurística: se muito grande (provável ms), normaliza para seg
    if v > 1e12:
        v = v / 1000.0
    # Entre 2001-01-01 e ~2050 → pode ser Unix epoch (>= 2001)
    try:
        dt = datetime.fromtimestamp(v, tz=timezone.utc)
        if dt.year >= 2001 and dt.year <= 2100:
            return dt.isoformat()
    except Exception:
        pass
    # Tenta como CoreData (epoch 2001)
    try:
        dt = coredata_secs_to_dt_utc(v)
        if 2001 <= dt.year <= 2100:
            return dt.isoformat()
    except Exception:
        pass
    # fallback: retorna bruto
    return val

def dump_sqlite_to_csvs(sqlite_path: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    out = {}
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r["name"] for r in cur.fetchall()]
        for t in tables:
            cur2 = con.cursor()
            cur2.execute(f"SELECT * FROM '{t}'")
            rows = cur2.fetchall()
            if not rows:
                continue
            cols = rows[0].keys()

            # normalização leve de tempos
            norm_rows = []
            for r in rows:
                d = {}
                for c in cols:
                    v = r[c]
                    # normaliza timestamps se o nome da coluna sugerir que é tempo
                    if re.search(r"(DATE|TIME|START|END|TIMESTAMP)", c, re.IGNORECASE):
                        d[c] = maybe_parse_timestamp(v)
                    else:
                        d[c] = v
                norm_rows.append(d)

            out_csv = out_dir / f"{t}.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(norm_rows)
            out[t] = out_csv
    finally:
        con.close()
    return out

def _coalesce_numeric(row, keys: List[str]) -> float:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except Exception:
                continue
    return 0.0

def _coalesce_str(row, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k]:
            return str(row[k])
    return None

def build_daily_aggregate(sqlite_path: Path, out_csv: Path) -> None:
    """
    Best-effort: procura tabelas típicas e colunas indicativas.
    Objetivo: (date, bundle_id, usage_seconds, notifications, pickups)
    """
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    daily: Dict[Tuple[str, str], Dict[str, float]] = {}
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r["name"] for r in cur.fetchall()]

        # 1) varredura genérica: qualquer tabela que tenha tempo + bundle/app + duração
        candidate_time_cols = re.compile(r"(DATE|DAY|START|END|TIME)", re.IGNORECASE)
        candidate_bundle_cols = re.compile(r"(BUNDLE|IDENTIFIER|APP)", re.IGNORECASE)
        candidate_usage_cols = re.compile(r"(DURATION|SECONDS|USAGE)", re.IGNORECASE)
        candidate_notif_cols = re.compile(r"(NOTIF)", re.IGNORECASE)
        candidate_pickup_cols = re.compile(r"(PICKUP)", re.IGNORECASE)

        for t in tables:
            # pega colunas
            info = con.execute(f"PRAGMA table_info('{t}')").fetchall()
            colnames = [i[1] for i in info]  # (cid, name, type, notnull, dflt, pk)
            # filtros rápidos
            if not any(candidate_bundle_cols.search(c) for c in colnames):
                continue
            if not any(candidate_time_cols.search(c) for c in colnames):
                continue

            # lê conteúdo
            rows = con.execute(f"SELECT * FROM '{t}'").fetchall()
            if not rows:
                continue

            # Para cada linha, tenta extrair (date, bundle, usage/notif/pickup)
            for r in rows:
                row = dict(zip(colnames, r))
                # date: procura colunas com "DAY"/"DATE" prioritariamente
                date_val = None
                for c in colnames:
                    if re.search(r"(DAY|DATE)", c, re.IGNORECASE):
                        date_val = maybe_parse_timestamp(row[c])
                        break
                # fallback para START/END ⇒ usa START
                if not date_val:
                    for c in colnames:
                        if re.search(r"(START)", c, re.IGNORECASE):
                            date_val = maybe_parse_timestamp(row[c])
                            break
                if not date_val:
                    continue
                # normaliza para YYYY-MM-DD
                try:
                    dt = datetime.fromisoformat(str(date_val).replace("Z", "+00:00")).astimezone(timezone.utc)
                    day = dt.date().isoformat()
                except Exception:
                    continue

                bundle = _coalesce_str(row, [c for c in colnames if candidate_bundle_cols.search(c)])
                if not bundle:
                    continue

                usage = _coalesce_numeric(row, [c for c in colnames if candidate_usage_cols.search(c)])
                notifs = _coalesce_numeric(row, [c for c in colnames if candidate_notif_cols.search(c)])
                pickups = _coalesce_numeric(row, [c for c in colnames if candidate_pickup_cols.search(c)])

                key = (day, bundle)
                if key not in daily:
                    daily[key] = {"usage_seconds": 0.0, "notifications": 0.0, "pickups": 0.0}
                daily[key]["usage_seconds"] += usage
                daily[key]["notifications"] += notifs
                daily[key]["pickups"] += pickups

        # escreve CSV
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date", "bundle_id", "usage_seconds", "notifications", "pickups"])
            w.writeheader()
            for (day, bundle), vals in sorted(daily.items()):
                w.writerow({
                    "date": day,
                    "bundle_id": bundle,
                    "usage_seconds": round(vals["usage_seconds"], 3),
                    "notifications": int(vals["notifications"]),
                    "pickups": int(vals["pickups"]),
                })
    finally:
        con.close()

def extract_screentime_from_backup(backup_dir: Optional[str], out_root: str = "artifacts/screentime") -> Dict[str, Path]:
    backup = find_latest_backup_dir(backup_dir)
    manifest = backup / "Manifest.db"
    hit = _query_manifest_for_screentime_sqlite(manifest)
    if not hit:
        raise FileNotFoundError("ScreenTime.sqlite not found in this backup. Refaça o backup sem criptografia e tente novamente.")
    file_id, rel = hit
    out_root = Path(out_root)
    sqlite_out = out_root / "ScreenTime.sqlite"
    _copy_hashed_file_from_backup(backup, file_id, sqlite_out)

    # Dump de todas as tabelas p/ CSV
    raw_dir = out_root / "raw_dump"
    dumps = dump_sqlite_to_csvs(sqlite_out, raw_dir)

    # Agregado diário por app
    daily_csv = out_root / "screentime_daily.csv"
    build_daily_aggregate(sqlite_out, daily_csv)

    return {
        "sqlite": sqlite_out,
        "daily": daily_csv,
        "raw_dir": raw_dir
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract ScreenTime.sqlite from iOS backup and build daily aggregates.")
    ap.add_argument("--backup-dir", type=str, default=None,
                    help="Path to the iOS backup dir or to the 'Backup' root. If omitted, auto-detects latest.")
    ap.add_argument("--out", type=str, default="artifacts/screentime",
                    help="Output directory for artifacts.")
    args = ap.parse_args()

    res = extract_screentime_from_backup(args.backup_dir, args.out)
    print("✅ Screen Time extraction done.")
    print("SQLite:", res["sqlite"])
    print("Daily :", res["daily"])
    print("Raw   :", res["raw_dir"])
