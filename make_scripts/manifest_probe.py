#!/usr/bin/env python3
"""
Probe an iOS Manifest.db to enumerate tables, row counts, top domains,
and list DeviceActivity candidate files. Writes deterministic outputs under
EXTRACTED_DIR/ios/:
 - manifest_probe.json
 - manifest_top_domains.csv
 - manifest_deviceactivity.tsv

Behavior: deterministic ordering, idempotent (no timestamps), metadata-only (no binary copying).
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter


def safe_table_name(name):
    # basic sanitation for SQL identifiers
    return '"{}"'.format(name.replace('"','""'))


def main():
    parser = argparse.ArgumentParser(description='Probe iOS Manifest.db')
    parser.add_argument('--manifest-db', dest='manifest_db', default=os.environ.get('IOS_MANIFEST_DB'))
    parser.add_argument('--out-dir', dest='out_dir', default=os.environ.get('EXTRACTED_DIR'))
    parser.add_argument('--participant', dest='participant', default=os.environ.get('PARTICIPANT'))
    parser.add_argument('--snapshot', dest='snapshot', default=os.environ.get('SNAPSHOT_DATE'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    out_ios_dir = os.path.join(args.out_dir or '.', 'ios')
    os.makedirs(out_ios_dir, exist_ok=True)

    json_path = os.path.join(out_ios_dir, 'manifest_probe.json')
    csv_domains = os.path.join(out_ios_dir, 'manifest_top_domains.csv')
    tsv_device = os.path.join(out_ios_dir, 'manifest_deviceactivity.tsv')

    tables = []
    row_counts = {}
    top_domains_list = []
    device_rows = []
    # defaults to avoid UnboundLocalError when DB missing
    rows = []
    relcol = None
    has_deviceactivity = False
    has_knowledgec = False

    if not args.manifest_db or not os.path.exists(args.manifest_db):
        # No DB: write empty/headers
        tables = []
        row_counts = {}
        top_domains_list = []
        device_rows = []
    else:
        conn = sqlite3.connect(args.manifest_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # list tables
        cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
        tables = [r['name'] for r in cur.fetchall()]

        # row counts (deterministic order by table name)
        for t in sorted(tables):
            try:
                cur.execute(f"SELECT COUNT(*) as c FROM {safe_table_name(t)}")
                row_counts[t] = int(cur.fetchone()['c'])
            except Exception:
                row_counts[t] = None

        # find candidate files table: look for table with columns fileID/fileId and relativePath/domain
        files_table = None
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info({safe_table_name(t)})")
                cols = [r['name'] for r in cur.fetchall()]
                cols_l = [c.lower() for c in cols]
                if ('fileid' in cols_l or 'file_id' in cols_l) and ('relativepath' in cols_l or 'relative_path' in cols_l):
                    files_table = t
                    break
            except Exception:
                continue

        if files_table:
            # collect domain counts and deviceactivity candidates
            # Normalize column names
            cur.execute(f"PRAGMA table_info({safe_table_name(files_table)})")
            cols = [r['name'] for r in cur.fetchall()]
            col_map = {c.lower(): c for c in cols}
            fileid_col = col_map.get('fileid') or col_map.get('file_id') or col_map.get('file')
            relcol = col_map.get('relativepath') or col_map.get('relative_path') or col_map.get('relative')
            domain_col = col_map.get('domain')
            size_col = col_map.get('size') or col_map.get('filelen') or col_map.get('file_size') or col_map.get('filesize')

            # Query all rows deterministically sorted by relativePath
            try:
                cur.execute(f"SELECT * FROM {safe_table_name(files_table)} ORDER BY {relcol} ASC")
                rows = cur.fetchall()
            except Exception:
                rows = []

            domains = []
            for r in rows:
                rel = r[relcol] if relcol in r.keys() else None
                dom = r[domain_col] if domain_col in r.keys() else ''
                fid = r[fileid_col] if fileid_col in r.keys() else ''
                size = r[size_col] if (size_col and size_col in r.keys()) else None
                domains.append(dom or '')
                # DeviceActivity detection
                if rel and ('DeviceActivity' in rel or 'DeviceActivity' in (rel or '')):
                    device_rows.append({'fileID': fid, 'domain': dom or '', 'relativePath': rel, 'fileSize': size or 0})
                # KnowledgeC detection: file name contains KnowledgeC and .db
                if rel and 'KnowledgeC' in rel:
                    # treat as knowledgec presence by later flag
                    pass

            # top domains
            cnt = Counter(domains)
            # deterministic ordering: sort by count desc then domain asc
            top_domains_list = [{'domain': d, 'count': cnt[d]} for d in sorted(cnt.keys(), key=lambda k: (-cnt[k], k))]

        else:
            # no files table found
            top_domains_list = []
            device_rows = []

        # determine has_deviceactivity and has_knowledgec
        has_deviceactivity = len(device_rows) > 0
        # knowledgec: search for any relativePath containing KnowledgeC or filenames KnowledgeC*.db
        has_knowledgec = False
        # rows and relcol may be undefined if files_table absent; guard accordingly
        if files_table and 'rows' in locals() and relcol:
            for r in rows:
                rel = r[relcol] if relcol in r.keys() else ''
                if rel and 'KnowledgeC' in rel:
                    has_knowledgec = True
                    break

        conn.close()

    # Prepare output JSON
    manifest_probe = {
        'tables': sorted(tables),
        'row_counts': {k: row_counts[k] for k in sorted(row_counts.keys())},
        'top_domains': top_domains_list,
        'has_deviceactivity': bool(device_rows),
        'has_knowledgec': bool(has_knowledgec),
    }

    # dry-run prints
    if args.dry_run:
        print(json.dumps(manifest_probe, indent=2, sort_keys=True))
        return

    # Write outputs deterministically
    # JSON
    tmpj = json_path + '.tmp'
    with open(tmpj, 'w', encoding='utf-8') as fj:
        json.dump(manifest_probe, fj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        fj.write('\n')
    os.replace(tmpj, json_path)

    # CSV top domains
    tmpc = csv_domains + '.tmp'
    with open(tmpc, 'w', encoding='utf-8') as fc:
        fc.write('domain,count\n')
        for item in top_domains_list:
            fc.write(f"{item['domain']},{item['count']}\n")
    os.replace(tmpc, csv_domains)

    # TSV device activity (header even when empty)
    tmpt = tsv_device + '.tmp'
    with open(tmpt, 'w', encoding='utf-8') as ft:
        ft.write('fileID\tdomain\trelativePath\tfileSize\n')
        # sort device_rows deterministically
        for dr in sorted(device_rows, key=lambda x: (x['relativePath'] or '', x['fileID'] or '')):
            ft.write(f"{dr['fileID']}\t{dr['domain']}\t{dr['relativePath']}\t{dr['fileSize']}\n")
    os.replace(tmpt, tsv_device)

    print(f'Wrote manifest probe outputs to {out_ios_dir}')


if __name__ == '__main__':
    main()
