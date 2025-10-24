#!/usr/bin/env python3
"""Fail if 'exports/' appears anywhere in the repo except allowed files.

Allowed exceptions: migrate_layout.py and explicit deprecation messages.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List


ALLOWED = {
    'make_scripts/migrate_layout.py',
    'make_scripts/lint_deprecated_exports.py',
    'Makefile',
    'make_scripts/lint_layout.py',
}


def _find_deprecated(root: Path) -> List[str]:
    found: List[str] = []
    for fp in root.rglob('*'):
        if fp.is_file():
            try:
                txt = fp.read_text(encoding='utf8', errors='ignore')
            except Exception:
                continue
            rel = fp.relative_to(root).as_posix()
            if 'exports/' in txt and rel not in ALLOWED:
                found.append(rel)
    return found


def main(argv=None) -> int:
    root = Path('.').resolve()
    found = _find_deprecated(root)
    if found:
        print('ERROR: Deprecated use of "exports/" found in these files:')
        for f in sorted(found):
            print(' -', f)
        return 2
    print('lint_deprecated_exports: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
