#!/usr/bin/env python3
"""Idempotence harness for A6 -> A7 -> A8

Runs selected stages twice for a given participant+snapshot and compares
SHA256 of outputs declared in each stage's manifest. Exits non-zero on
mismatch and prints diffs.

Usage:
  tests/idempotence_a6_a7_a8.py --participant P000001 --snapshot 2025-09-29 --stages A6,A7,A8 [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def sha256_of_file(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


STAGE_TO_MAKE = {
    'A6': 'run-a6-apple',
    'A7': 'run-a7-apple',
    'A8': 'run-a8-labels',
}


def manifest_path_for(stage: str, pid: str, snap: str) -> Optional[Path]:
    # Known locations used by runners
    if stage == 'A6':
        return Path('data') / 'etl' / pid / 'snapshots' / snap / 'processed' / 'apple' / 'etl_qc_summary_manifest.json'
    if stage == 'A7':
        return Path('data') / 'etl' / pid / 'snapshots' / snap / 'processed' / 'apple' / 'a7_manifest.json'
    if stage == 'A8':
        return Path('data') / 'ai' / pid / 'snapshots' / snap / 'labels_manifest.json'
    return None


def collect_outputs_from_manifest(man_p: Path) -> Dict[str, Optional[str]]:
    if not man_p.exists():
        return {}
    try:
        obj = json.loads(man_p.read_text(encoding='utf-8'))
    except Exception:
        return {}
    outs = {}
    # manifest formats vary; try common keys
    if isinstance(obj, dict):
        outputs = obj.get('outputs') or obj.get('files') or []
        # outputs may be list of dicts or dict(metric->path)
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, dict) and 'path' in v:
                    p = Path(v['path'])
                    outs[str(p)] = v.get('sha256') or sha256_of_file(p)
                else:
                    p = Path(v)
                    outs[str(p)] = sha256_of_file(p)
        elif isinstance(outputs, list):
            for it in outputs:
                if isinstance(it, dict) and 'path' in it:
                    p = Path(it['path'])
                    outs[str(p)] = it.get('sha256') or sha256_of_file(p)
                elif isinstance(it, str):
                    p = Path(it)
                    outs[str(p)] = sha256_of_file(p)
    return outs


def run_make_target(target: str, pid: str, snap: str, dry_run: bool) -> Tuple[int, str]:
    cmd = ['make', target, f'PARTICIPANT={pid}', f'SNAPSHOT_DATE={snap}', f'DRY_RUN={1 if dry_run else 0}']
    # run and capture stdout/stderr
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def diff_maps(m1: Dict[str, Optional[str]], m2: Dict[str, Optional[str]]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    keys = sorted(set(list(m1.keys()) + list(m2.keys())))
    diffs = []
    for k in keys:
        v1 = m1.get(k)
        v2 = m2.get(k)
        if v1 != v2:
            diffs.append((k, v1, v2))
    return diffs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--stages', required=True, help='Comma-separated stages: A6,A7,A8')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    pid = args.participant
    snap = args.snapshot
    stages = [s.strip().upper() for s in args.stages.split(',') if s.strip()]
    dry = args.dry_run

    overall_ok = True
    total_diffs = []

    for stage in stages:
        if stage not in STAGE_TO_MAKE:
            print(f'Unknown stage: {stage}', file=sys.stderr)
            sys.exit(2)

        target = STAGE_TO_MAKE[stage]

        print(f'Running {stage} (first run): make {target} PARTICIPANT={pid} SNAPSHOT_DATE={snap} DRY_RUN={1 if dry else 0}')
        rc, out = run_make_target(target, pid, snap, dry)
        if rc != 0:
            print(f'First run for {stage} failed (exit {rc}). Output:\n{out}', file=sys.stderr)
            sys.exit(rc)

        man = manifest_path_for(stage, pid, snap)
        m1 = collect_outputs_from_manifest(man) if man is not None else {}

        print(f'Running {stage} (second run): make {target} PARTICIPANT={pid} SNAPSHOT_DATE={snap} DRY_RUN={1 if dry else 0}')
        rc, out = run_make_target(target, pid, snap, dry)
        if rc != 0:
            print(f'Second run for {stage} failed (exit {rc}). Output:\n{out}', file=sys.stderr)
            sys.exit(rc)

        m2 = collect_outputs_from_manifest(man) if man is not None else {}

        diffs = diff_maps(m1, m2)
        if diffs:
            overall_ok = False
            print(f'Idempotence FAILED for stage {stage}. Mismatched files:')
            for path, h1, h2 in diffs:
                print(f'- {path}\n   run1: {h1}\n   run2: {h2}')
            total_diffs.extend([(stage, path, h1, h2) for path, h1, h2 in diffs])
        else:
            print(f'Stage {stage} idempotent (no diffs)')

    if not overall_ok:
        print('\nIdempotence check: FAILED', file=sys.stderr)
        sys.exit(3)
    print('\nIdempotence check: PASSED')
    sys.exit(0)


if __name__ == '__main__':
    main()
