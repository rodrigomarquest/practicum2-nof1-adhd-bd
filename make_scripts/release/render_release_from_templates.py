#!/usr/bin/env python3
"""Render release notes using an internal academic-style template.

Consumes:
- tag and version
- optional title, branch, author, project
- pre-release summary file at reports/pre_release/changes_since_<LATEST_TAG>.md

Writes: docs/release_notes/release_notes_<TAG>.md

No external templating libraries; uses stdlib only.
"""
from __future__ import annotations
import argparse
import datetime
import os
from pathlib import Path
import sys

TEMPLATE = """
🚀 {{ NEXT_TAG }} – {{ title }}
Release date: {{ today_yyyy_mm_dd }}
Branch: {{ branch }}
Author: {{ author }}
Project: {{ project }}

🔧 Summary
{{ summary }}

🧩 Highlights
{{ highlights }}

🧠 Next Steps
{{ next_steps }}

🧾 Citation
Teixeira, R. M. (2025). N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2).
National College of Ireland.

⚖️ License: CC BY-NC-SA 4.0
Supervisor: Dr. Agatha Mattos
Student ID: 24130664
Maintainer: Rodrigo Marques Teixeira
"""


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name('.' + path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding='utf-8')
    os.replace(str(tmp), str(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--title', default='')
    p.add_argument('--branch', default='main')
    p.add_argument('--author', default='Rodrigo Marques Teixeira')
    p.add_argument('--project', default='Practicum2 – N-of-1 ADHD + BD')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    # Determine latest tag from env if provided or by looking at reports path
    latest_tag = os.environ.get('LATEST_TAG') or 'v0.0.0'
    pre_rel_path = Path('reports') / 'pre_release' / f'changes_since_{latest_tag}.md'
    if not pre_rel_path.exists():
        print('NEED:')
        print('- Missing pre-release changes summary at reports/pre_release/changes_since_{}.md'.format(latest_tag))
        print('- Ask VADER to summarize changes since {} and save there.'.format(latest_tag))
        return 2

    summary_text = pre_rel_path.read_text(encoding='utf-8')
    # Heuristics: split into summary, highlights, next_steps by headings if present
    highlights = ''
    next_steps = ''
    summary = summary_text.strip()
    # Very small parsing: if file contains 'Highlights:' or 'Highlights\n', split
    if 'Highlights:' in summary_text:
        parts = summary_text.split('Highlights:', 1)
        summary = parts[0].strip()
        rem = parts[1]
        if 'Next Steps:' in rem:
            h, n = rem.split('Next Steps:', 1)
            highlights = h.strip()
            next_steps = n.strip()
        else:
            highlights = rem.strip()
    elif 'Next Steps:' in summary_text:
        parts = summary_text.split('Next Steps:', 1)
        summary = parts[0].strip()
        next_steps = parts[1].strip()

    now = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    mapping = {
        'NEXT_TAG': args.tag,
        'title': args.title or args.tag,
        'today_yyyy_mm_dd': now,
        'branch': args.branch,
        'author': args.author,
        'project': args.project,
        'summary': summary,
        'highlights': highlights or '(none)',
        'next_steps': next_steps or '(none)'
    }

    out = TEMPLATE
    for k, v in mapping.items():
        out = out.replace('{{ ' + k + ' }}', str(v))
        out = out.replace('{{' + k + '}}', str(v))

    out_path = Path('docs') / 'release_notes' / f'release_notes_{args.tag}.md'
    if args.dry_run:
        print('DRY RUN: would write', out_path)
        print('\n'.join(out.splitlines()[:20]))
        return 0
    try:
        atomic_write(out_path, out)
    except Exception as e:
        print('ERROR writing release notes:', e, file=sys.stderr)
        return 3
    print('Wrote', out_path.as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
