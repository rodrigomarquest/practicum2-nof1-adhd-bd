#!/usr/bin/env python3
"""PX: Build reproducibility checksum manifest for project state.

Writes project_state_manifest.json which includes SHA256 of all manifests, Makefile git commit hash (if available), and pip freeze snapshot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash(file: Path) -> str | None:
    try:
        out = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=%H', '--', str(file)], stderr=subprocess.DEVNULL)
        return out.decode('utf-8').strip()
    except Exception:
        return None


def pip_freeze() -> str:
    try:
        out = subprocess.check_output(['pip', 'freeze'], stderr=subprocess.DEVNULL)
        return out.decode('utf-8')
    except Exception:
        return ''


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args(argv)

    root = Path(args.repo_root)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # collect manifests in repo
    manifests = list(root.glob('**/*_manifest.json'))
    manifests = [m for m in manifests if m.is_file()]
    man_info = []
    for m in sorted(manifests):
        man_info.append({'path': str(m.relative_to(root)), 'sha256': sha256_file(m)})

    makefile_hash = git_commit_hash(root / 'Makefile')
    freeze = pip_freeze()

    out = {
        'generated_at': '',
        'manifests': man_info,
        'makefile_commit': makefile_hash,
        'pip_freeze': freeze
    }
    outpath = outdir / 'project_state_manifest.json'
    outpath.write_text(json.dumps(out, sort_keys=True, indent=2), encoding='utf-8')
    print('WROTE:', outpath)


if __name__ == '__main__':
    main()
