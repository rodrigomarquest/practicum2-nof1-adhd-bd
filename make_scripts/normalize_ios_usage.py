#!/usr/bin/env python3
"""
Normalize iOS DeviceActivity (and KnowledgeC if present) into UTC-aligned events.
Writes:
 - $(NORMALIZED_DIR)/ios/ios_usage_events.csv
 - $(NORMALIZED_DIR)/ios/knowledgec_events.csv (header-only if not present)
 - manifests: ios_usage_events_manifest.json, knowledgec_events_manifest.json

This implementation is conservative: it reads discovery outputs (manifest_deviceactivity.tsv,
manifest_probe.json, backup_paths.json). If DeviceActivity rows exist, real parsing is NOT implemented
here (requires complex DB parsing). For now, if rows exist, the script writes placeholder rows
with zero usage_seconds but still records input SHAs and row counts. If no rows, writes header-only CSVs.

All writes are atomic and deterministic.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data_bytes):
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data_bytes)
    os.replace(tmp, str(path))


def write_csv_header(path, header):
    # write header-only csv deterministically
    b = (','.join(header) + '\n').encode('utf-8')
    atomic_write(path, b)


def write_csv_rows(path, header, rows):
    # deterministic ordering by timestamp_utc then bundle_id
    rows_sorted = sorted(rows, key=lambda r: (r.get('timestamp_utc') or '', r.get('bundle_id') or ''))
    out = [','.join(header) + '\n']
    for r in rows_sorted:
        out.append(','.join(str(r.get(c, '')) for c in header) + '\n')
    atomic_write(path, ''.join(out).encode('utf-8'))


def write_manifest(path, inputs_shas, row_count):
    manifest = {
        'inputs': {k: inputs_shas[k] for k in sorted(inputs_shas.keys())},
        'rows': int(row_count),
    }
    jb = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
    atomic_write(path, jb.encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description='Normalize iOS usage events (DeviceActivity/KnowledgeC)')
    parser.add_argument('--extracted-dir', dest='extracted_dir', default=os.environ.get('EXTRACTED_DIR'))
    parser.add_argument('--normalized-dir', dest='normalized_dir', default=os.environ.get('NORMALIZED_DIR'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    extracted = Path(args.extracted_dir or '.') / 'ios'
    normalized = Path(args.normalized_dir or '.') / 'ios'
    normalized.mkdir(parents=True, exist_ok=True)

    # Discovery artifacts
    manifest_probe = extracted / 'manifest_probe.json'
    manifest_deviceactivity = extracted / 'manifest_deviceactivity.tsv'
    backup_paths = extracted / 'backup_paths.json'

    inputs = [p for p in (manifest_probe, manifest_deviceactivity, backup_paths) if p.exists()]
    inputs_shas = {str(p.name): sha256_of_file(str(p)) for p in inputs}

    # Default outputs
    ios_csv = normalized / 'ios_usage_events.csv'
    ios_manifest = normalized / 'ios_usage_events_manifest.json'
    knowledge_csv = normalized / 'knowledgec_events.csv'
    knowledge_manifest = normalized / 'knowledgec_events_manifest.json'

    # Prepare header schema
    ios_header = ['timestamp_utc','bundle_id','category','usage_seconds','source']
    knowledge_header = ['timestamp_utc','bundle_id','category','usage_seconds','source']

    # Read deviceactivity tsv rows if present
    rows = []
    if manifest_deviceactivity.exists():
        with open(manifest_deviceactivity, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
        if len(lines) > 1:
            # There are discovered files; placeholder behavior: create no event rows because parsing
            # DeviceActivity DB is non-trivial. Instead, record zero event rows but include inputs in manifest.
            # If you want actual parsing, implement DB parsing here.
            # For now, we produce header-only CSV but include inputs SHAs.
            rows = []

    # For knowledgec we look at manifest_probe.json
    has_knowledgec = False
    if manifest_probe.exists():
        try:
            with open(manifest_probe, 'r', encoding='utf-8') as f:
                mp = json.load(f)
            has_knowledgec = bool(mp.get('has_knowledgec'))
        except Exception:
            has_knowledgec = False

    # Dry-run: print intended actions
    if args.dry_run:
        print('Would write:', ios_csv)
        print('Would write manifest:', ios_manifest)
        print('Would write knowledge CSV (if present):', knowledge_csv if has_knowledgec else '(none)')
        print('Inputs sha:', inputs_shas)
        return

    # Write ios csv (header-only or rows)
    if rows:
        # Actually write rows (not implemented)
        write_csv_rows(ios_csv, ios_header, rows)
        row_count = len(rows)
    else:
        write_csv_header(ios_csv, ios_header)
        row_count = 0

    # Write knowledgec csv header-only if not parsed
    if has_knowledgec:
        # parsing not implemented; write header-only
        write_csv_header(knowledge_csv, knowledge_header)
        knowledge_rows = 0
    else:
        # ensure file absent or header-only
        if knowledge_csv.exists():
            write_csv_header(knowledge_csv, knowledge_header)
        knowledge_rows = 0

    # Write manifests
    write_manifest(ios_manifest, inputs_shas, row_count)
    write_manifest(knowledge_manifest, inputs_shas, knowledge_rows)

    print(f'Wrote normalized iOS usage outputs to {normalized}')


if __name__ == '__main__':
    main()
