#!/usr/bin/env python3
"""Build a manifest.json for files under a directory.

Writes manifest.json which contains array of objects with path (relative), size_bytes and sha256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict:
    files = sorted([p for p in root.rglob('*') if p.is_file()])
    entries = []
    total = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        sha = sha256_file(p)
        entries.append({'path': rel, 'size_bytes': size, 'sha256': sha})
        total += size
    return {'files': entries, 'count': len(entries), 'total_bytes': total}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('dir', help='Directory to scan')
    p.add_argument('--out', default='manifest.json', help='Output manifest file name')
    args = p.parse_args(argv)
    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        print(f'No such directory: {root}')
        return 2
    manifest = build_manifest(root)
    outp = Path(args.out) 
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    kb = manifest['total_bytes'] / 1024.0
    print(f"Wrote {outp} — {manifest['count']} files, {kb:.1f} KB total")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
