#!/usr/bin/env python3
"""Render release notes from templates/docs into dist/release_notes_<TAG>.md

Usage: render_release_from_templates.py --version VERSION --tag TAG [--title TITLE] [--summary SUMMARY] [--branch BRANCH] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding='utf-8')
    os.replace(str(tmp), str(path))


def render_template(tpl: str, mapping: dict) -> str:
    out = tpl
    for k, v in mapping.items():
        out = out.replace('{{' + k + '}}', str(v))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--title', default='')
    p.add_argument('--summary', default='')
    p.add_argument('--branch', default='main')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    tpl_path = Path('docs') / 'release_template.md'
    if not tpl_path.exists():
        print('ERROR: template not found:', tpl_path, file=sys.stderr)
        return 2

    tpl = tpl_path.read_text(encoding='utf-8')
    now = datetime.datetime.now(datetime.timezone.utc)
    mapping = {
        'VERSION': args.version,
        'TITLE': args.title or args.version,
        'DATE_UTC': now.isoformat(),
        'BRANCH': args.branch,
        'SUMMARY': args.summary or ('Release ' + args.version),
    }

    rendered = render_template(tpl, mapping)
    out_path = Path('dist') / f'release_notes_{args.tag}.md'

    # atomic deterministic write
    if args.dry_run:
        print(f'DRY RUN: would write {out_path}')
        # print a short header
        header = '\n'.join(rendered.splitlines()[:6])
        print(header)
        # still write the file to dist as per requirements
        try:
            atomic_write(out_path, rendered)
        except Exception as e:
            print('ERROR writing file in dry-run:', e, file=sys.stderr)
            return 3
        return 0

    try:
        atomic_write(out_path, rendered)
    except Exception as e:
        print('ERROR writing release notes:', e, file=sys.stderr)
        return 4
    print('Wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
