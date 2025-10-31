#!/usr/bin/env python3
"""Minimal renderer used by Makefile release-draft.

Writes:
 - docs/release_notes/release_notes_<tag>.md
 - dist/changelog/CHANGELOG.dryrun.md

Accepts same args as requested and an optional --out to also write the combined
changelog to an extra path. Safe for dry-run mode.
"""
import argparse
import subprocess
import pathlib
import sys
import textwrap
import datetime


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def get_last_tag() -> str:
    try:
        return sh("git describe --tags --abbrev=0 2>/dev/null")
    except subprocess.CalledProcessError:
        return ""


def get_changes_since(tag: str) -> str:
    if tag:
        return sh(f"git log --pretty=oneline {tag}..HEAD || true")
    else:
        return sh("git log --pretty=oneline || true")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--branch", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None, help="Optional extra path to write combined output")
    args = p.parse_args()

    now = datetime.datetime.utcnow().isoformat() + "Z"
    last_tag = get_last_tag()
    changes = get_changes_since(last_tag)

    rn_dir = pathlib.Path("docs/release_notes")
    rn_dir.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path("dist/changelog")
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Release {args.tag}"
    summary = args.summary or "Auto-generated draft. See CHANGES_SINCE_LAST_TAG.txt."
    branch = args.branch or sh("git rev-parse --abbrev-ref HEAD")

    release_notes = textwrap.dedent(f"""\
    # 🚀 {args.tag} — {title}

    **Release date (UTC):** {now}  
    **Branch:** `{branch}`

    ## 🔧 Summary
    {summary}

    ## 📜 Changes since last tag
    ```text
    {changes}
    ```
    """)

    rn_file = rn_dir / f"release_notes_{args.tag}.md"
    rn_file.write_text(release_notes, encoding="utf-8")

    changelog_dry = textwrap.dedent(f"""\
    Wrote {rn_file.as_posix()}
    # CHANGELOG (dry-run) for {args.tag}
    ## Changes since {last_tag or '<no previous tag>'}
    ```text
    {changes}
    ```
    """)
    (out_dir / "CHANGELOG.dryrun.md").write_text(changelog_dry, encoding="utf-8")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(changelog_dry, encoding="utf-8")

    if not args.dry_run:
        print(f"Wrote {rn_file.as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Render release notes from templates/docs into docs/release_notes/release_notes_<TAG>.md

Usage: render_release_from_templates.py --version VERSION --tag TAG [--title TITLE] [--summary SUMMARY] [--branch BRANCH] [--dry-run]

By default the renderer writes into `docs/release_notes/` (not `dist/`). When --dry-run is passed
the script will NOT write files to disk; it will print a short preview instead.
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
    out_path = Path('docs') / 'release_notes' / f'release_notes_{args.tag}.md'

    # Dry-run should not modify the working tree — print a short preview and exit
    if args.dry_run:
        print(f'DRY RUN: would write {out_path}')
        header = '\n'.join(rendered.splitlines()[:6])
        print(header)
        return 0

    # atomic deterministic write into docs/release_notes
    try:
        # If the file already exists and content is identical, avoid rewriting
        if out_path.exists():
            existing = out_path.read_text(encoding='utf-8')
            if existing == rendered:
                print('No change in release notes for', out_path.name)
                return 0
        atomic_write(out_path, rendered)
    except Exception as e:
        print('ERROR writing release notes:', e, file=sys.stderr)
        return 4
    print('Wrote', out_path.as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
