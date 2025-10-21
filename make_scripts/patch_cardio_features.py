#!/usr/bin/env python3
"""Patch cardio_features.py to add atomic CSV write helper (extracted from heredoc).

This script is conservative: it backs up the target (caller should do that), then
inserts small helper functions and replaces occurrences of DataFrame.to_csv(...) with
_write_atomic_csv(...). It's idempotent: if _write_atomic_csv already exists, it skips.
"""
from pathlib import Path
import re


def main():
    p = Path("etl_modules/cardiovascular/cardio_features.py")
    if not p.exists():
        print(f"File not found: {p}")
        raise SystemExit(1)
    s = p.read_text(encoding="utf-8")

    # If helpers already present, skip
    if "_write_atomic_csv" in s:
        print("SKIP: cardio_features.py already has atomic helpers")
        return

    # Ensure imports include modules we need for helpers
    s = s.replace('import math, os', 'import math, os, tempfile, hashlib')

    # Insert helper functions immediately before the definition of update_features_daily
    helper = (
        "def _sha256_file(path: str):\n"
        "    import hashlib\n"
        "    h = hashlib.sha256()\n"
        "    with open(path, 'rb') as f:\n"
        "        for chunk in iter(lambda: f.read(1024*1024), b''):\n"
        "            h.update(chunk)\n"
        "    return h.hexdigest()\n\n"
        "def _write_atomic_csv(df, out_path: str):\n"
        "    import os, tempfile\n"
        "    d = os.path.dirname(out_path) or '.'\n"
        "    fd, tmp = tempfile.mkstemp(prefix='.tmp_', dir=d, suffix='.csv')\n"
        "    os.close(fd)\n"
        "    try:\n"
        "        df.to_csv(tmp, index=False)\n"
        "        os.replace(tmp, out_path)\n"
        "    finally:\n"
        "        if os.path.exists(tmp):\n"
        "            try: os.remove(tmp)\n"
        "            except: pass\n\n"
    )

    if 'def update_features_daily(' in s:
        s = s.replace('def update_features_daily(', helper + 'def update_features_daily(', 1)
    else:
        # fallback: append helpers at top
        s = helper + s

    # Replace occurrences of DataFrame.to_csv(...) with _write_atomic_csv(...)
    s = re.sub(r"\.to_csv\(([^)]+)\)", r"__CSV__(\1)", s)
    s = s.replace('__CSV__(', '_write_atomic_csv(')

    p.write_text(s, encoding="utf-8")
    print("WROTE: cardio_features.py (atomic CSV)")


if __name__ == '__main__':
    main()
