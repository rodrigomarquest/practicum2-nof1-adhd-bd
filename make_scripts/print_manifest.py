#!/usr/bin/env python3
"""Print a manifest.json in a human-readable form.
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
        print('(no manifest)')
        return 0
    try:
        data = json.loads(m.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'Error reading manifest: {e}', file=sys.stderr)
        return 2
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
