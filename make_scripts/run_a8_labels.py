#!/usr/bin/env python3
"""Prompt A8: Label Integration (Synthetic/EMA) per-snapshot

Reads the modeling features_daily.csv and a label source (synthetic or EMA),
merges labels by UTC day, resolves multiple labels per day by mode (most
frequent) with fallback to max score, preserves raw_label/score columns if
present, writes features_daily_labeled.csv and labels_manifest.json.

Usage:
  make_scripts/apple/run_a8_labels.py --participant P000001 --snapshot 2025-09-29 [--label-path PATH] [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import platform
import subprocess
import sys
from make_scripts.utils.snapshot_lock import SnapshotLock, SnapshotLockError

try:
    from etl_modules.common.progress import progress_bar
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def progress_bar(total, desc: str = "", unit: str = "items"):
        class _B:
            def update(self, n=1):
                return None
            def close(self):
                return None
        yield _B()

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_csv_from_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + out_path.name + '.tmp.', dir=str(out_path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, str(out_path))

def choose_label(group: pd.DataFrame) -> Dict[str, Any]:
    """Resolve multiple label rows for one date: choose mode; if tie, pick label with max score."""
    # prefer explicit 'label' column, else raw_label
    if 'label' in group.columns and group['label'].notna().any():
        labels = group['label'].dropna().astype(str)
    elif 'raw_label' in group.columns and group['raw_label'].notna().any():
        labels = group['raw_label'].dropna().astype(str)
    else:
        return {'label': None, 'raw_label': None, 'score': None}

    # compute mode(s)
    mode_counts = labels.value_counts()
    top_count = mode_counts.iloc[0]
    modes = mode_counts[mode_counts == top_count].index.tolist()
    if len(modes) == 1:
        chosen = modes[0]
    else:
        # tie: pick among tied labels the one with max score if available
        if 'score' in group.columns:
            sub = group[group.get('label', group.get('raw_label')) .isin(modes)]
            if not sub.empty:
                # pick label with max score; handle missing scores
                sub_scores = sub[['label','raw_label','score']].copy()
                sub_scores['score'] = pd.to_numeric(sub_scores['score'], errors='coerce').fillna(-float('inf'))
                idx = sub_scores['score'].idxmax()
                chosen = sub.loc[idx].get('label') or sub.loc[idx].get('raw_label')
            else:
                chosen = modes[0]
        else:
            chosen = modes[0]

    # find representative raw_label and score
    sel = group[(group.get('label', group.get('raw_label')).astype(str) == str(chosen))]
    raw_label = None
    score = None
    if 'raw_label' in sel.columns and sel['raw_label'].notna().any():
        raw_label = sel['raw_label'].dropna().astype(str).iloc[0]
    if 'score' in sel.columns and sel['score'].notna().any():
        try:
            score = float(pd.to_numeric(sel['score'], errors='coerce').max())
        except Exception:
            score = None

    return {'label': chosen, 'raw_label': raw_label, 'score': score}

def run_label_merge(pid: str, snap: str, label_path_arg: str | None = None, dry_run: bool = False, lock_timeout: int = 3600, force_lock: bool = False) -> Dict[str, Any]:
    base = Path('data') / 'ai' / pid / 'snapshots' / snap
    features_path = base / 'features_daily.csv'
    # candidate label sources
    synthetic = base / 'state_of_mind_synthetic.csv'
    # allow override
    label_path = Path(label_path_arg) if label_path_arg else (synthetic if synthetic.exists() else None)

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    stages = ['read_labels','validate','merge','stats','write']
    with progress_bar(total=len(stages), desc='A8 stages', unit='steps') as pbar:
        pbar.update(1)  # read_labels
        if not features_path.exists():
            raise FileNotFoundError(f'features_daily.csv not found: {features_path}')
        features = pd.read_csv(features_path, dtype={'date_utc': str})

        # ensure date_utc string
        if 'date_utc' not in features.columns:
            # try to coerce date column
            if 'date' in features.columns:
                features['date_utc'] = pd.to_datetime(features['date']).dt.date.astype(str)
            else:
                raise ValueError('features_daily.csv missing date_utc')

        pbar.update(1)  # validate
        if label_path is None or not label_path.exists():
            # no labels available; we'll write empty labeled file with same rows
            labels_df = pd.DataFrame(columns=['date_utc','label'])
        else:
            labels_df = pd.read_csv(label_path, dtype=str)
            # coerce date column to date_utc if necessary
            if 'date_utc' not in labels_df.columns:
                if 'date' in labels_df.columns:
                    labels_df['date_utc'] = pd.to_datetime(labels_df['date']).dt.date.astype(str)
                else:
                    raise ValueError('label file missing date/date_utc column')

        pbar.update(1)  # merge
        # group labels by date_utc and resolve multiples
        if not labels_df.empty:
            grouped = labels_df.groupby('date_utc')
            resolved = []
            for dt, g in grouped:
                r = choose_label(g)
                r['date_utc'] = dt
                resolved.append(r)
            resolved_df = pd.DataFrame(resolved)
        else:
            resolved_df = pd.DataFrame(columns=['date_utc','label','raw_label','score'])

        # merge with features (left join) but do not mutate features; create new df
        merged = features.merge(resolved_df, how='left', on='date_utc')

        pbar.update(1)  # stats
        rows_total = len(merged)
        rows_labeled = merged['label'].notna().sum() if 'label' in merged.columns else 0
        class_dist = merged['label'].value_counts(dropna=True).to_dict() if 'label' in merged.columns else {}

        pbar.update(1)  # write
        out_path = base / 'features_daily_labeled.csv'
        manifest_path = base / 'labels_manifest.json'
        if dry_run:
            print('DRY_RUN: would write', out_path, 'rows=', rows_total)
            # try to populate git info for preview
            try:
                commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
                prod_suffix = commit[:7]
                git_preview = {'commit': commit, 'branch': subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], stderr=subprocess.DEVNULL).decode().strip(), 'dirty': bool(subprocess.check_output(['git','status','--porcelain'], stderr=subprocess.DEVNULL).decode().strip())}
            except Exception:
                prod_suffix = 'unknown'
                git_preview = 'unknown'
            manifest = {
                'schema_version': '1.1', 'producer': f"make_scripts/{Path(__file__).name} {prod_suffix}", 'run_id': run_id, 'participant': pid, 'snapshot': snap,
                'git': git_preview,
                'system': {'os': platform.platform(), 'python_version': platform.python_version(), 'tz': datetime.now(timezone.utc).astimezone().tzname()},
                'args': {'cli': sys.argv, 'env_overrides': {}},
                'rows_total': rows_total, 'rows_labeled': int(rows_labeled), 'class_distribution': class_dist,
                'inputs': [ {'path': str(features_path), 'sha256': None, 'rows': None}, {'path': str(label_path), 'sha256': None, 'rows': None} if label_path is not None else None ],
                'outputs': [ {'path': str(out_path), 'sha256': None, 'rows': rows_total}, {'path': str(manifest_path), 'sha256': None, 'rows': None} ]
            }
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            # protect writes with snapshot lock
            snapshot_root = Path('data') / 'ai' / pid / 'snapshots' / snap
            lock = SnapshotLock(snapshot_root, 'ai', pid, snap, timeout_sec=lock_timeout, force=force_lock)
            try:
                lock.acquire()
            except SnapshotLockError as e:
                print(str(e), file=sys.stderr)
                sys.exit(5)
            try:
                # atomic write labeled CSV
                atomic_write_csv_from_df(merged, out_path)
                # build enriched manifest with provenance
                inputs_list = []
                inputs_list.append({'path': str(features_path), 'sha256': sha256_of_file(features_path) if features_path.exists() else None, 'rows': None})
                if label_path is not None:
                    inputs_list.append({'path': str(label_path), 'sha256': sha256_of_file(label_path) if label_path.exists() else None, 'rows': None})

                outputs_list = [{'path': str(out_path), 'sha256': sha256_of_file(out_path), 'rows': rows_total}, {'path': str(manifest_path), 'sha256': None, 'rows': None}]

                manifest = {
                    'schema_version': '1.1', 'producer': f"make_scripts/{Path(__file__).name}", 'run_id': run_id, 'participant': pid, 'snapshot': snap,
                    'git': None,
                    'system': {'os': platform.platform(), 'python_version': platform.python_version(), 'tz': datetime.now(timezone.utc).astimezone().tzname()},
                    'args': {'cli': sys.argv, 'env_overrides': {}},
                    'rows_total': rows_total, 'rows_labeled': int(rows_labeled), 'class_distribution': class_dist,
                    'inputs': inputs_list,
                    'outputs': outputs_list,
                }
                try:
                    commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
                    branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
                    dirty = bool(subprocess.check_output(['git','status','--porcelain'], stderr=subprocess.DEVNULL).decode().strip())
                    manifest['git'] = {'commit': commit, 'branch': branch, 'dirty': dirty}
                except Exception:
                    manifest['git'] = 'unknown'
                # attach producer with commit short if available
                git = manifest.get('git')
                if isinstance(git, dict):
                    manifest['producer'] = f"make_scripts/{Path(__file__).name} {git.get('commit','')[:7]}"
                # atomic manifest write
                fd, tmp = tempfile.mkstemp(prefix='.' + manifest_path.name + '.tmp.', dir=str(manifest_path.parent))
                os.close(fd)
                with open(tmp, 'w', encoding='utf-8') as fh:
                    json.dump(manifest, fh, indent=2, sort_keys=True)
                os.replace(tmp, str(manifest_path))
            finally:
                lock.release()
                print('Released lock:', lock.lock_path)

    # ensure JSON-serializable types (pandas may use numpy ints)
    try:
        rt = int(rows_total)
    except Exception:
        rt = rows_total
    try:
        rl = int(rows_labeled)
    except Exception:
        rl = rows_labeled
    return {'out': str(out_path), 'manifest': str(manifest_path), 'rows_total': rt, 'rows_labeled': rl}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--label-path', required=False)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force-lock', action='store_true', help='Override stale snapshot lock')
    p.add_argument('--lock-timeout', type=int, default=3600, help='Seconds before a lock is considered stale')
    args = p.parse_args(argv)
    res = run_label_merge(args.participant, args.snapshot, args.label_path, dry_run=args.dry_run, lock_timeout=args.lock_timeout, force_lock=args.force_lock)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
