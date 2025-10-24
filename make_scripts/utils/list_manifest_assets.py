#!/usr/bin/env python3
"""Read a manifest.json and print space-separated list of asset paths (relative to manifest dir).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='Path to manifest.json')
    args = p.parse_args(argv)
    m = Path(args.manifest)
    if not m.exists():
        return 0
    try:
        data = json.loads(m.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'Error reading manifest: {e}', file=sys.stderr)
        return 2
    root = m.parent
    files = [root / f['path'] for f in data.get('files', [])]
    # print space-separated paths (shell-friendly)
    out = ' '.join(str(p) for p in files)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
