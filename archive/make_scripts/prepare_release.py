#!/usr/bin/env python3
"""Prepare or execute a GitHub release create command based on an asset manifest.

Reads <asset_dir>/manifest.json with structure {'files': [{'path': rel, ...}, ...]}
and either prints the exact gh command (dry-run) or executes it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def gh_available() -> bool:
    return shutil.which('gh') is not None


def gh_auth_ok() -> bool:
    try:
        subprocess.check_output(['gh', 'auth', 'status'], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--notes-file', required=True)
    p.add_argument('--asset-dir', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    asset_dir = Path(args.asset_dir)
    manifest_path = asset_dir / 'manifest.json'
    files = []
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest.get('files', []):
            files.append(str(asset_dir / entry['path']))

    # Verify gh availability/auth
    if not args.dry_run:
        if not gh_available():
            print('ERROR: gh CLI not found; install GitHub CLI and authenticate', file=sys.stderr)
            return 2
        if not gh_auth_ok():
            print('ERROR: gh auth status failed; ensure you are authenticated', file=sys.stderr)
            return 3
    else:
        if not gh_available():
            print('WARNING: gh CLI not found (dry-run)', file=sys.stderr)
        if not gh_auth_ok():
            print('WARNING: gh auth status unknown or unauthenticated (dry-run)', file=sys.stderr)

    # Build command string
    cmd = ['gh', 'release', 'create', args.tag, '--draft', '--title', args.title, '--notes-file', args.notes_file]
    cmd.extend(files)
    cmd_str = ' '.join(['"{}"'.format(x) if ' ' in x or '"' in x else x for x in cmd])

    if args.dry_run:
        print('(DRY RUN) gh command:')
        print(cmd_str)
        print('\nAssets from manifest:')
        if manifest is None:
            print('(no manifest found)')
        else:
            for f in manifest.get('files', []):
                print('-', f.get('path'), f.get('size_bytes'), f.get('sha256'))
        return 0

    # Execute the gh command
    try:
        print('Executing:', cmd_str)
        subprocess.check_call(cmd)
        return 0
    except subprocess.CalledProcessError as e:
        print('ERROR: gh command failed with code', e.returncode, file=sys.stderr)
        return e.returncode
    except Exception as e:
        print('ERROR executing gh:', e, file=sys.stderr)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
