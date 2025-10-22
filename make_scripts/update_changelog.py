#!/usr/bin/env python3
"""Update CHANGELOG.md by inserting/updating a top entry derived from release notes.

Follows the layout specified in the prompt. No external deps.
"""
from __future__ import annotations
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def utc_now_iso() -> str:
    # timezone-aware UTC
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def read_release_notes(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


# Simple parser for markdown headings and sections
HEADING_RE = re.compile(r'^(#{2,})\s*(.+)$', flags=re.MULTILINE)


def extract_sections(md: str) -> Dict[str, str]:
    # Finds top-level headings (##) and captures their content until next ##
    sections: Dict[str, str] = {}
    if not md:
        return sections
    # Normalize line endings
    md = md.replace('\r\n', '\n')
    # Find ## headings
    parts = HEADING_RE.split(md)
    # parts layout: [pre, hashes1, heading1, body1, hashes2, heading2, body2, ...]
    it = iter(parts)
    pre = next(it)
    for hashes, heading, body in zip(it, it, it):
        h = heading.strip().lower()
        sections[h] = body.strip()
    return sections


def extract_bullets(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    bullets: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith('- ') or ln.startswith('* '):
            bullets.append(ln[2:].strip())
        elif ln:
            # treat non-heading paragraphs as a bullet paragraph
            bullets.append(ln)
    return bullets


GROUPS = [
    ('feat', '✨ feat', '✨'),
    ('fix', '🐛 fix', '🐛'),
    ('chore', '🧹 chore', '🧹'),
    ('refactor', '🧰 refactor', '🧰'),
    ('docs', '📚 docs', '📚'),
    ('test', '🧪 test', '🧪'),
    ('ci/build', '⚙️ ci/build', '⚙️'),
]


EMOJI_MAP: Dict[str, str] = {
    'feat': '✨ feat',
    'fix': '🐛 fix',
    'chore': '🧹 chore',
    'refactor': '🧰 refactor',
    'docs': '📚 docs',
    'test': '🧪 test',
    'ci': '⚙️ ci/build',
    'build': '⚙️ ci/build',
}


def build_changelog_entry(tag: str, date_iso: str, sections: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append(f'## [{tag}] - {date_iso}')
    lines.append('')
    # Summary
    summary = sections.get('summary') or sections.get('summary\n') or ''
    if summary:
        lines.append('### 🔧 Summary')
        lines.append('')
        for b in extract_bullets(summary):
            lines.append(f'- {b}')
        lines.append('')
    # Highlights
    highlights = sections.get('highlights') or ''
    if highlights:
        lines.append('### 🧩 Highlights')
        lines.append('')
        for ln in highlights.splitlines():
            if ln.strip():
                lines.append(ln.rstrip())
        lines.append('')
    # Changes grouped
    changes = sections.get('changes') or ''
    grouped: Dict[str, List[str]] = {g[0]: [] for g in GROUPS}
    grouped['other'] = []
    if changes:
        # Classify bullets by conventional commit type where possible
        for ln in changes.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith('- ') or s.startswith('* '):
                val = s[2:].strip()
                m = re.match(r'(?P<type>[^(:]+)(?:\(.+?\))?:\s*(?P<desc>.+)', val)
                if m:
                    t = m.group('type').strip().lower()
                    desc = m.group('desc').strip()
                    # direct match
                    if t in grouped:
                        grouped[t].append(desc)
                    else:
                        # common short names (feat, fix, docs, etc.)
                        if t in EMOJI_MAP:
                            grouped.setdefault(t, []).append(desc)
                        else:
                            grouped['other'].append(val)
                else:
                    grouped['other'].append(val)
            else:
                # treat as a paragraph bullet
                grouped['other'].append(s)
    # Append groups with emojis
    lines.append('### ✍️ Changes')
    lines.append('')
    any_group = False
    for key, title, emoji in GROUPS:
        items = grouped.get(key, [])
        if items:
            any_group = True
            # Use emoji map when available
            key_label = key.split('/')[0]
            display = EMOJI_MAP.get(key_label, title)
            lines.append(f'#### {display}')
            lines.append('')
            for it in items:
                lines.append(f'- {it}')
            lines.append('')
    # other
    if grouped.get('other'):
        any_group = True
        lines.append('#### 📦 misc')
        lines.append('')
        for it in grouped['other']:
            lines.append(f'- {it}')
        lines.append('')
    if not any_group and changes:
        # If no groups were detected but changes text exists, put all bullets under misc
        bullets = extract_bullets(changes)
        if bullets:
            lines.append('#### 📦 misc')
            lines.append('')
            for b in bullets:
                lines.append(f'- {b}')
            lines.append('')
    # Next Steps
    next_steps = sections.get('next steps') or sections.get('next-steps') or sections.get('next') or ''
    if next_steps:
        lines.append('### 🧠 Next Steps')
        lines.append('')
        for b in extract_bullets(next_steps):
            lines.append(f'- {b}')
        lines.append('')
    # Citation
    citation = sections.get('citation') or ''
    if citation:
        lines.append('### 🧾 Citation')
        lines.append('')
        lines.append(citation.strip())
        lines.append('')
    lines.append('---')
    lines.append('')
    return '\n'.join(lines)


def find_previous_tag(tag: str) -> Optional[str]:
    # Try git describe --tags --abbrev=0 <tag>^
    if not tag:
        return None
    r = run(['git', 'rev-parse', '--verify', '--quiet', f'refs/tags/{tag}'])
    if r.returncode == 0:
        r2 = run(['git', 'describe', '--tags', '--abbrev=0', f'{tag}^'])
        if r2.returncode == 0:
            return r2.stdout.strip()
    # fallback: list tags sorted by date
    r3 = run(['git', 'tag', '--sort=-creatordate'])
    if r3.returncode != 0:
        return None
    tags = [l for l in r3.stdout.splitlines() if l.strip()]
    if tag in tags:
        idx = tags.index(tag)
        if idx+1 < len(tags):
            return tags[idx+1]
    if len(tags) > 1:
        return tags[1]
    return None


def repo_owner_repo() -> Optional[Tuple[str, str]]:
    # Try to parse 'origin' remote URL
    r = run(['git', 'remote', 'get-url', 'origin'])
    if r.returncode != 0:
        return None
    url = r.stdout.strip()
    # handle ssh git@github.com:owner/repo.git and https urls
    m = re.match(r'git@github.com:(?P<owner>[^/]+)/(?P<repo>[^.]+)(\.git)?', url)
    if m:
        return m.group('owner'), m.group('repo')
    m2 = re.match(r'https?://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)(\.git)?', url)
    if m2:
        return m2.group('owner'), m2.group('repo')
    return None


def update_footer_links(out: Path, tag: str, prev_tag: Optional[str]):
    # Read existing file and remove any existing compare-link block so update is idempotent
    text = out.read_text(encoding='utf-8') if out.exists() else ''
    # Remove last compare block if present (a divider '---' followed by any compare links or Full changelog block)
    cleaned = re.sub(r'(?s)\n---\n\n.*$', '', text).rstrip()
    owner_repo = repo_owner_repo()
    if not owner_repo:
        print('Warning: could not determine owner/repo from git remote; skipping compare links')
        out.write_text(cleaned + '\n')
        return
    owner, repo = owner_repo
    if prev_tag:
        compare_url = f'https://github.com/{owner}/{repo}/compare/{prev_tag}...{tag}'
        link_text = f'---\n\n[Full changelog]({compare_url})\n'
    else:
        # No previous tag found: link to commit list for the tag
        commits_url = f'https://github.com/{owner}/{repo}/commits/{tag}'
        link_text = f'---\n\n[Commits for {tag}]({commits_url})\n'
    out.write_text(cleaned + '\n\n' + link_text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--version')
    p.add_argument('--since-tag')
    p.add_argument('--release-notes')
    p.add_argument('--outfile', default='CHANGELOG.md')
    p.add_argument('--date-utc')
    p.add_argument('--allow-missing-notes', action='store_true')
    args = p.parse_args(argv)

    tag = args.tag
    version = args.version or (tag[1:] if tag.startswith('v') else tag)
    since_tag = args.since_tag
    rn = Path(args.release_notes) if args.release_notes else Path(f'dist/release_notes_{tag}.md')
    out = Path(args.outfile)
    date_iso = args.date_utc or utc_now_iso()

    rn_text = read_release_notes(rn)
    if not rn_text and not args.allow_missing_notes:
        print(f'Error: release notes not found at {rn}', file=sys.stderr)
        return 2
    sections = extract_sections(rn_text)
    entry = build_changelog_entry(tag, date_iso, sections)

    # Ensure CHANGELOG exists
    if not out.exists():
        out.write_text('# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n', encoding='utf-8')
    orig = out.read_text(encoding='utf-8')
    # Remove existing entry for tag if present
    if f'## [{tag}]' in orig:
        # Simple replace: locate the start and end (next '## [' or EOF)
        start = orig.find(f'## [{tag}]')
        next_idx = orig.find('\n## [', start+1)
        if next_idx == -1:
            new_text = orig[:start] + entry
        else:
            new_text = orig[:start] + entry + orig[next_idx:]
    else:
        # Insert after header intro (after first blank line after top header)
        hdr = '# Changelog'
        if orig.startswith(hdr):
            # find first double newline after the header
            idx = orig.find('\n\n')
            if idx == -1:
                new_text = hdr + '\n\n' + entry + '\n' + orig[len(hdr):]
            else:
                new_text = orig[:idx+2] + entry + '\n' + orig[idx+2:]
        else:
            new_text = entry + '\n' + orig
    out.write_text(new_text, encoding='utf-8')

    # Update footer links using git
    prev = args.since_tag or find_previous_tag(tag)
    update_footer_links(out, tag, prev)

    print(f'Updated {out} with entry for {tag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
