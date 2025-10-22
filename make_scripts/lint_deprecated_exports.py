#!/usr/bin/env python3
"""Fail if 'exports/' appears anywhere in the repo except allowed files.

Allowed exceptions: migrate_layout.py and explicit deprecation messages.
"""
import sys
from pathlib import Path

ALLOWED = {
    'make_scripts/migrate_layout.py',
    'make_scripts/lint_deprecated_exports.py',
    'Makefile',
    'make_scripts/lint_layout.py',
}

root = Path('.').resolve()
found = []
for p in root.rglob('*'):
    if p.is_file():
        try:
            txt = p.read_text(encoding='utf8', errors='ignore')
        except Exception:
            continue
        rel = p.relative_to(root).as_posix()
        if 'exports/' in txt and rel not in ALLOWED:
            found.append(rel)

if found:
    print('ERROR: Deprecated use of "exports/" found in these files:')
    for f in sorted(found):
        print(' -', f)
    sys.exit(2)

print('lint_deprecated_exports: OK')
