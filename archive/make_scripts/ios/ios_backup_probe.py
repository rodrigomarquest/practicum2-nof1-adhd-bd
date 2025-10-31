#!/usr/bin/env python3
"""Probe the iOS decrypted backup root and write a deterministic JSON with path hints.

Writes only metadata JSON under EXTRACTED_DIR/ios/backup_paths.json
"""
import argparse
import json
import os
import sys
from datetime import datetime


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe iOS decrypted backup and emit metadata JSON")
    parser.add_argument("--backup-root", dest="backup_root", default=os.environ.get('IOS_BACKUP_DIR'))
    parser.add_argument("--manifest-db", dest="manifest_db", default=os.environ.get('IOS_MANIFEST_DB'))
    parser.add_argument("--out-dir", dest="out_dir", default=os.environ.get('EXTRACTED_DIR'))
    parser.add_argument("--participant", dest="participant", default=os.environ.get('PARTICIPANT'))
    parser.add_argument("--snapshot", dest="snapshot", default=os.environ.get('SNAPSHOT_DATE'))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.backup_root:
        print("ERROR: backup_root not provided (flag or IOS_BACKUP_DIR env)", file=sys.stderr)
        sys.exit(2)

    out_dir = os.path.join(args.out_dir or '.', 'ios')
    os.makedirs(out_dir, exist_ok=True)

    # Resolve manifest path; allow relative inside backup_root
    manifest_db = args.manifest_db or os.path.join(args.backup_root, 'Manifest.db')

    # Build metadata dict with deterministic keys (we will sort when dumping)
    # If the file already exists, preserve its timestamp_utc to ensure idempotence.
    out_path = os.path.join(out_dir, 'backup_paths.json')
    existing_ts = None
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                existing_ts = existing.get('timestamp_utc')
        except Exception:
            existing_ts = None

    meta = {
        'backup_root': os.path.normpath(args.backup_root),
        'manifest_db': os.path.normpath(manifest_db),
        'deviceactivity_glob': 'Library/DeviceActivity/**',
        'knowledgec_glob': '**/KnowledgeC*.db',
        'participant': args.participant or '',
        'snapshot_date': args.snapshot or '',
        'timestamp_utc': existing_ts or datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # If dry-run, print what would be written
    if args.dry_run:
        print('DRY RUN: would write:', out_path)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return

    # Write deterministically: sorted keys, ensure stable separators and no trailing spaces
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        f.write('\n')
    os.replace(tmp, out_path)
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
