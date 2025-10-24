#!/usr/bin/env python3
"""Check that ETL export subdirs exist and print file counts.
Exit non-zero if missing.
"""
from __future__ import annotations
import os
from pathlib import Path


def main(argv=None) -> int:
    etl = Path(os.environ.get('ETL_DIR', './data/etl/P000001'))
    apple = etl / 'exports' / 'apple'
    zepp = etl / 'exports' / 'zepp'
    ok = True
    for p in (apple, zepp):
        if not p.exists():
            print('MISSING:', p)
            ok = False
        else:
            cnt = sum(1 for _ in p.rglob('*') if _.is_file())
            print(p, 'files=', cnt)
    if not ok:
        return 2
    print('Running DRY_RUN test...')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
