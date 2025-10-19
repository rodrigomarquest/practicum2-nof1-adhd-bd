#!/usr/bin/env python3
import os
import sys
import glob
import sqlite3
import datetime as dt
from pathlib import Path
from dateutil import tz
import traceback
import csv
from collections import defaultdict

# ==== Config =================================================================
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", "decrypted_output"))
KNOW_DIR = os.path.join(OUT_DIR, "knowledgec")
OUT_MAIN = os.path.join(KNOW_DIR, "usage_daily_from_knowledgec.csv")
# Alias para o teu ETL atual:
PLISTS_DIR = os.path.join(OUT_DIR, "screentime_plists")
OUT_ALIAS = os.path.join(PLISTS_DIR, "health_usage_daily.csv")
LOG_PATH = os.path.join(KNOW_DIR, "parse_knowledgec.log")

TZ_LOCAL = tz.gettz("Europe/Dublin")

# Cocoa epoch (Apple): seconds since 2001-01-01 00:00:00 UTC
COCOA_EPOCH = dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc)

def log(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"[{ts}] {msg}\n")

def cocoa_to_dt(seconds):
    """Convert Cocoa time (float seconds since 2001-01-01 UTC) to aware datetime."""
    try:
        return COCOA_EPOCH + dt.timedelta(seconds=float(seconds))
    except Exception:
        return None

def to_local_date(dtime: dt.datetime):
    if not isinstance(dtime, dt.datetime):
        return None
    return dtime.astimezone(TZ_LOCAL).date()

def get_columns(cursor, table):
    cols = {}
    for r in cursor.execute(f"PRAGMA table_info({table})"):
        cols[r[1].lower()] = r[1]
    return cols

def scan_db(path):
    """Return list of (start_dt_utc, end_dt_utc, bundle_guess) usage intervals from one KnowledgeC.db."""
    out = []
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Quick sanity: presence of ZOBJECT
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0].lower(): r[0] for r in cur.fetchall()}
        if "zobject" not in tables:
            log(f"⚠️  No ZOBJECT in {path}. Skipping.")
            con.close()
            return out

        zobject = tables["zobject"]
        cols = get_columns(cur, zobject)

        # Find likely columns (defensive mapping)
        col_stream   = cols.get("zstreamname") or cols.get("streamname") or cols.get("zstream") or "ZSTREAMNAME"
        col_start    = cols.get("zstartdate")  or cols.get("startdate")  or "ZSTARTDATE"
        col_end      = cols.get("zenddate")    or cols.get("enddate")    or "ZENDDATE"
        col_valstr   = cols.get("zvaluestring") or cols.get("valuestring") or "ZVALUESTRING"
        col_struct   = cols.get("zstructuredmetadata") or cols.get("structuredmetadata") or None

        # Optional join to ZSTRUCTUREDMETADATA for bundleId-ish strings
        zmeta = tables.get("zstructuredmetadata")
        meta_cols = {}
        if zmeta:
            meta_cols = get_columns(cur, zmeta)
        col_meta_bundle = meta_cols.get("zbundleid") or meta_cols.get("bundleid")
        col_meta_dom    = meta_cols.get("zdomain") or meta_cols.get("domain")

        # Heuristic: Foreground app usage stream names commonly include:
        # "com.apple.springboard.foregroundapplication"
        # Some builds have variants; we will filter in Python after query.
        q = f"""
        SELECT {col_stream} AS stream, {col_start} AS s, {col_end} AS e,
               {col_valstr} AS vstr,
               {col_struct} AS smeta
        FROM {zobject}
        WHERE ({col_start} IS NOT NULL AND {col_end} IS NOT NULL)
        """
        try:
            rows = list(cur.execute(q))
        except Exception as e:
            log(f"⚠️  Query failed on {path}: {e}")
            con.close()
            return out

        # If possible, prepare a map from ZSTRUCTUREDMETADATA to bundle hints
        meta_map = {}
        if zmeta and col_struct and (col_meta_bundle or col_meta_dom):
            try:
                rows_meta = list(cur.execute(f"SELECT Z_PK as pk, "
                                             f"{col_meta_bundle or 'NULL'} as b, "
                                             f"{col_meta_dom or 'NULL'} as d "
                                             f"FROM {zmeta}"))
                for r in rows_meta:
                    meta_map[r["pk"]] = (r["b"], r["d"])
            except Exception:
                pass

        for r in rows:
            stream = (r["stream"] or "").lower()
            if "foreground" not in stream and "springboard" not in stream and "application" not in stream:
                # keep only likely foreground usage
                continue
            s = cocoa_to_dt(r["s"])
            e = cocoa_to_dt(r["e"])
            if not s or not e or e <= s:
                continue

            # Guess a bundle/app label
            bundle_guess = None
            if r["vstr"]:
                bundle_guess = str(r["vstr"])
            elif r["smeta"] is not None and r["smeta"] in meta_map:
                b, dmn = meta_map.get(r["smeta"], (None, None))
                bundle_guess = b or dmn
            out.append((s, e, bundle_guess))
        con.close()
    except Exception:
        log(f"⚠️  Exception parsing {path}")
        log(traceback.format_exc())
    return out

def main():
    if not os.path.isdir(KNOW_DIR):
        log(f"⚠️  KnowledgeC dir not found: {KNOW_DIR}")
        log("    (Execute extract-knowledgec quando o Manifest trouxer KnowledgeC.db)")
        # Mesmo assim, gera CSV vazio com header mínimo
        write_empty()
        return

    dbs = glob.glob(os.path.join(KNOW_DIR, "**", "*.db"), recursive=True)
    if not dbs:
        log("⚠️  No *.db found under knowledgec/. Producing empty CSV.")
        write_empty()
        return

    log(f"🔎 Found {len(dbs)} KnowledgeC DB candidate(s). Parsing...")
    intervals = []
    for db in dbs:
        intervals.extend(scan_db(db))

    if not intervals:
        log("⚠️  No foreground app intervals detected. Producing empty CSV.")
        write_empty()
        return

    # Aggregate by local day
    totals = defaultdict(float)
    for s, e, _bundle in intervals:
        # Split intervals across days if needed
        cur = s
        while cur.date() != e.date():
            day_end = dt.datetime(cur.year, cur.month, cur.day, 23, 59, 59, tzinfo=cur.tzinfo)
            dur = (day_end - cur).total_seconds()
            d_local = to_local_date(cur)
            if d_local:
                totals[d_local] += max(dur, 0.0)
            cur = day_end + dt.timedelta(seconds=1)
        # remainder on last day
        d_local = to_local_date(e)
        if d_local:
            totals[d_local] += max((e - cur).total_seconds(), 0.0)

    # Build rows
    rows = []
    for d, seconds in sorted(totals.items()):
        rows.append({
            "participant_id": "P000001",
            "date": d.isoformat(),
            "total_screen_minutes": round(seconds / 60.0, 2),
            "num_sessions": 0,
            "first_unlock_time": 0,
            "last_use_time": 0,
            "late_use_minutes_23_07": 0,
            "deep_night_bursts": 0,
            "social_minutes": 0,
            "entertainment_minutes": 0,
            "productivity_minutes": 0,
            "focus_mode_overlap_minutes": 0
        })

    # Ensure output dirs
    Path(KNOW_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLISTS_DIR).mkdir(parents=True, exist_ok=True)

    # Write main + alias
    write_csv(OUT_MAIN, rows)
    write_csv(OUT_ALIAS, rows)

    log(f"✅ Wrote: {OUT_MAIN} rows={len(rows)}")
    log(f"✅ Alias: {OUT_ALIAS}")

def write_empty():
    Path(KNOW_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLISTS_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    write_csv(OUT_MAIN, rows)
    write_csv(OUT_ALIAS, rows)
    log(f"✅ Wrote empty CSVs: {OUT_MAIN} and alias {OUT_ALIAS}")

def write_csv(path, rows):
    header = [
        "participant_id","date","total_screen_minutes","num_sessions","first_unlock_time","last_use_time",
        "late_use_minutes_23_07","deep_night_bursts","social_minutes","entertainment_minutes",
        "productivity_minutes","focus_mode_overlap_minutes"
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("❌ Fatal error in parse_knowledgec_usage.py")
        log(traceback.format_exc())
        sys.exit(1)
