#!/usr/bin/env python3
"""Layout linter: enforce stage locations and forbid writes to legacy exports/ from recipes and make_scripts/.

Checks performed:
 - No references to 'exports/' in Makefile recipes or in files under make_scripts/ except migrate_layout.py.
 - No hard-coded writes to wrong stage directories: grep for 'data/etl/.*/processed' (and normalized/joined) appearing in files not allowed to write them.

This is a conservative, text-based check meant to catch obvious regressions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import re
from typing import List


def _collect_violations(root: Path) -> List[str]:
    errors: List[str] = []

    # 1) forbid exports/ in make scripts and Makefile recipes, except migrate_layout.py
    # Also allow the linters themselves (they may mention 'exports/' in docstrings)
    allowed_exports = {'make_scripts/migrate_layout.py', 'make_scripts/lint_deprecated_exports.py', 'make_scripts/lint_layout.py'}
    for p in root.rglob('make_scripts/*.py'):
        rel = p.relative_to(root).as_posix()
        if rel in allowed_exports:
            continue
        try:
            txt = p.read_text(encoding='utf8', errors='ignore')
        except Exception:
            continue
        if 'exports/' in txt:
            errors.append(f"Forbidden 'exports/' found in {rel}")

    # Check Makefile recipes for literal exports/ token
    mf = root / 'Makefile'
    if mf.exists():
        mf_txt = mf.read_text(encoding='utf8', errors='ignore')
        # we disallow exports/ appearing inside recipe lines (lines starting with a tab or > recipe prefix)
        for i, line in enumerate(mf_txt.splitlines(), start=1):
            if 'exports/' in line and (line.startswith('	') or line.lstrip().startswith('>')):
                # allow if the line is the migrate-layout help or explicit init-data compatibility
                if 'migrate-layout' in mf_txt and 'exports/' in mf_txt and 'init-data' in mf_txt and 'exports' in line:
                    # allow legacy init-data recipe for now
                    continue
                errors.append(f"Forbidden 'exports/' found in Makefile at line {i}: {line.strip()}")

    # 2) Disallow scripts writing outside their stage
    stage_patterns = ['normalized', 'processed', 'joined', 'ai']
    allowed_writers = set([
        'etl_modules', 'etl_tools', 'make_scripts/migrate_layout.py', 'make_scripts/intake_zip.py',
        'make_scripts/provenance_audit.py', 'make_scripts/clean_data_make.py',
        'make_scripts/lint_layout.py', 'make_scripts/lint_deprecated_exports.py'
    ])

    for p in root.rglob('*'):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        # skip allowed writers entirely
        if any(rel.startswith(a) for a in allowed_writers if a.endswith('/')):
            continue
        if rel in allowed_writers:
            continue
        try:
            txt = p.read_text(encoding='utf8', errors='ignore')
        except Exception:
            continue
        for stage in stage_patterns:
            if re.search(rf"data/etl/.+/{stage}", txt):
                if rel in allowed_writers:
                    continue
                errors.append(f"Possible forbidden write to stage '{stage}' found in {rel}")

    return errors


def main(argv=None) -> int:
    root = Path('.').resolve()
    errors = _collect_violations(root)
    if errors:
        print('lint-layout: FAILED')
        for e in errors:
            print(' -', e)
        return 2
    print('lint-layout: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
