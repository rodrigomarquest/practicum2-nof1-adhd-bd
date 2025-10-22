#!/usr/bin/env python3
"""Check that ETL export subdirs exist and print file counts.
Exit non-zero if missing.
"""
import os
import sys
from pathlib import Path

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
    sys.exit(2)
print('Running DRY_RUN test...')
sys.exit(0)
