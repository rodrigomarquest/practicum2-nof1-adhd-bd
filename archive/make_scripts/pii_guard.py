#!/usr/bin/env python3
"""Lightweight PII & consent guard.

Provides validate_participant_ready(pid) and CLI wrapper that prints non-blocking
warnings if consent file or raw export is missing or oversized.
"""
from __future__ import annotations

from pathlib import Path
import yaml


def _read_participants_yaml(path: Path) -> dict:
    try:
        with path.open('r', encoding='utf8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def validate_participant_ready(pid: str) -> bool:
    """Return True if checks pass; print warnings otherwise (non-blocking).

    Checks:
      - participants.yaml has non-empty consent field for pid
      - data/raw/<pid>/apple/export.zip exists and size < 1.5GB (warn if bigger)

    This function never raises; it prints warnings and returns boolean pass/fail.
    """
    ok = True
    base = Path('data') / 'raw' / pid / 'apple'
    participants_f = Path('participants.yaml')

    # consent check
    participants = _read_participants_yaml(participants_f)
    info = participants.get(pid) if isinstance(participants, dict) else None
    consent = None
    if info and isinstance(info, dict):
        consent = info.get('consent') or info.get('consent_given') or info.get('consent_date')

    if not consent:
        print('WARNING: participant', pid, 'missing consent info in participants.yaml (do not commit raw files).')
        ok = False

    # raw export check
    export = base / 'export.zip'
    if not export.exists():
        print(f'WARNING: expected raw export not found: {export} (ensure raw data present)')
        ok = False
    else:
        try:
            size = export.stat().st_size
            # 1.5 GiB = 1.5 * 1024**3
            if size > int(1.5 * 1024 ** 3):
                print(f'WARNING: raw export {export} is large ({size} bytes) > 1.5GB; be cautious with PII handling')
        except Exception:
            print('WARNING: cannot stat', export)

    return ok


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='PII & consent guard (non-blocking)')
    p.add_argument('--participant', required=True)
    args = p.parse_args(argv)
    validate_participant_ready(args.participant)


if __name__ == '__main__':
    main()
