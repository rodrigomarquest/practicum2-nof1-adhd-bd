#!/usr/bin/env python3
"""Create provenance reports for ETL outputs.

Scans the following directories (ENV or CLI args):
 - NORMALIZED_DIR -> per-metric CSVs
 - PROCESSED_DIR -> daily aggregates
 - JOINED_DIR -> hybrid merges
 - AI_SNAPSHOT_DIR -> AI snapshot files

Writes:
 - provenance/etl_provenance_report.csv  (detailed inventory)
 - provenance/etl_provenance_checks.csv  (simple checks: counts, date ranges)
 - provenance/data_audit_summary.md      (human-readable summary)

Supports --dry-run to write placeholder files without error.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
from datetime import datetime
import os
from typing import List, Dict, Any, Optional


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def scan_dir(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return items
    for p in sorted(root.rglob('*')):
        if p.is_file():
            st = p.stat()
            items.append({
                'path': str(p),
                'relpath': str(p.relative_to(root)),
                'size_bytes': st.st_size,
                'modified': datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z',
            })
    return items


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def summarize_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {'count': 0, 'size_bytes': 0, 'min_modified': None, 'max_modified': None}
    sizes = [i['size_bytes'] for i in items]
    times = [i['modified'] for i in items]
    return {
        'count': len(items),
        'size_bytes': sum(sizes),
        'min_modified': min(times),
        'max_modified': max(times),
    }


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description='Provenance audit for ETL outputs')
    p.add_argument('--participant', help='Participant id to scope the audit, e.g. P000001')
    p.add_argument('--snapshot', help='Optional snapshot date YYYY-MM-DD to include ai snapshot folder')
    p.add_argument('--normalized-dir')
    p.add_argument('--processed-dir')
    p.add_argument('--joined-dir')
    p.add_argument('--ai-snapshot-dir')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    # ENV fallback
    # Participant-scoped defaults
    participant = args.participant or os.environ.get('PARTICIPANT') or 'P000001'
    # allow explicit dirs to override; otherwise derive from participant (and optional snapshot)
    norm = Path(args.normalized_dir or os.environ.get('NORMALIZED_DIR') or f'data/etl/{participant}/normalized')
    proc = Path(args.processed_dir or os.environ.get('PROCESSED_DIR') or f'data/etl/{participant}/processed')
    join = Path(args.joined_dir or os.environ.get('JOINED_DIR') or f'data/etl/{participant}/joined')
    if args.snapshot:
        ai = Path(args.ai_snapshot_dir or os.environ.get('AI_SNAPSHOT_DIR') or f'data/ai/{participant}/snapshots/{args.snapshot}')
    else:
        ai = Path(args.ai_snapshot_dir or os.environ.get('AI_SNAPSHOT_DIR') or f'data/ai/{participant}/snapshots')

    out_dir = Path('provenance')
    out_dir.mkdir(parents=True, exist_ok=True)

    # participant-scoped output filenames (suffix snapshot if provided)
    snap_suffix = f'_{args.snapshot}' if args.snapshot else ''
    report_csv = out_dir / f'etl_provenance_report_{participant}{snap_suffix}.csv'
    checks_csv = out_dir / f'etl_provenance_checks_{participant}{snap_suffix}.csv'
    summary_md = out_dir / f'data_audit_summary_{participant}{snap_suffix}.md'

    report_rows: List[Dict[str, Any]] = []
    checks_rows: List[Dict[str, Any]] = []

    # If dry-run, still create empty placeholder outputs
    if args.dry_run:
        # create empty CSVs with headers
        # participant-scoped output filenames (suffix snapshot if provided)
        snap_suffix = f'_{args.snapshot}' if args.snapshot else ''
        report_csv = out_dir / f'etl_provenance_report_{participant}{snap_suffix}.csv'
        checks_csv = out_dir / f'etl_provenance_checks_{participant}{snap_suffix}.csv'
        summary_md = out_dir / f'data_audit_summary_{participant}{snap_suffix}.md'

        write_csv(report_csv, [], ['stage','path','relpath','size_bytes','modified'])
        write_csv(checks_csv, [], ['stage','count','size_bytes','min_modified','max_modified'])
        # write a small markdown summary for dry-run
        summary_md.write_text(f'# Data audit summary for {participant}{snap_suffix}\n\nDry-run executed at ' + now_iso() + '\n', encoding='utf-8')
        print('Dry-run: wrote placeholder provenance files to', out_dir)
        return

    # Scan each stage
    normalized_items = scan_dir(norm)
    processed_items = scan_dir(proc)
    joined_items = scan_dir(join)
    ai_items = scan_dir(ai)

    # Build report rows
    for items, stage_name in [ (normalized_items, 'normalized'), (processed_items, 'processed'), (joined_items, 'joined'), (ai_items, 'ai_snapshot') ]:
        for it in items:
            report_rows.append({
                'stage': stage_name,
                'path': it['path'],
                'relpath': it['relpath'],
                'size_bytes': it['size_bytes'],
                'modified': it['modified'],
            })
        summary = summarize_items(items)
        checks_rows.append({
            'stage': stage_name,
            'count': summary['count'],
            'size_bytes': summary['size_bytes'],
            'min_modified': summary['min_modified'] or '',
            'max_modified': summary['max_modified'] or '',
        })

    # write outputs
    write_csv(report_csv, report_rows, ['stage','path','relpath','size_bytes','modified'])
    write_csv(checks_csv, checks_rows, ['stage','count','size_bytes','min_modified','max_modified'])

    # write a small markdown summary
    md = [f'# Data audit summary for {participant}{snap_suffix}\n\nGenerated: {now_iso()}\n\n']
    for ck in checks_rows:
        md.append(f"## Stage: {ck['stage']}\n- Files: {ck['count']}\n- Total size bytes: {ck['size_bytes']}\n- Range: {ck['min_modified']} → {ck['max_modified']}\n\n")
    summary_md.write_text('\n'.join(md), encoding='utf-8')

    print('Wrote provenance reports to', out_dir)


if __name__ == '__main__':
    main()
