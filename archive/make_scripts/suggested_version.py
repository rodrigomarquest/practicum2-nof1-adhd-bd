#!/usr/bin/env python3
"""Suggest a semantic version bump based on Conventional Commits since last tag.

Rules (simple heuristic):
- If any commit body contains 'BREAKING CHANGE' or subject contains '!:': suggest major bump
- Else if any commit type is 'feat': suggest minor bump
- Else: suggest patch bump

Usage: make suggest-version (Makefile target calls this script)
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from typing import List


def run(cmd: List[str]):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_latest_tag() -> str:
    r = run(['git', 'describe', '--tags', '--abbrev=0'])
    if r.returncode == 0:
        return r.stdout.strip()
    return ''


def get_commits_since(tag: str) -> List[dict]:
    fmt = '%H%x01%s%x01%b'
    rng = f'{tag}..HEAD' if tag else 'HEAD~200..HEAD'
    r = run(['git', 'log', '--pretty=format:' + fmt, rng])
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split('\x01')
        if len(parts) < 3:
            continue
        out.append({'sha': parts[0], 'subject': parts[1], 'body': parts[2]})
    return out


def bump_version(ver: str, level: str) -> str:
    # ver like 0.2.0 or 0.2.0-alpha; split off suffix
    main = ver
    suffix = ''
    if '-' in ver:
        main, suffix = ver.split('-', 1)
    parts = main.split('.')
    if len(parts) != 3:
        # fallback
        parts = ['0','0','0']
    major, minor, patch = map(int, parts)
    if level == 'major':
        major += 1
        minor = 0
        patch = 0
    elif level == 'minor':
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def analyze_commits(commits: List[dict]):
    """Return (level, reason, triggers) where triggers is a list of strings.

    level is one of 'major', 'minor', 'patch'.
    """
    breaking = False
    has_feat = False
    triggers: List[str] = []
    for c in commits:
        subj = (c.get('subject') or '').strip()
        body = (c.get('body') or '').strip()
        short = c.get('sha', '')[:7]
        # Breaking in body
        if 'BREAKING CHANGE' in body or 'BREAKING-CHANGE' in body or re.search(r'\bBREAKING\b', body):
            breaking = True
            triggers.append(f"- {short} {subj}")
        # conventional commits: ! in subject after type/scope
        if re.search(r'^[^:]+![: ]', subj) or re.search(r':.*!$', subj) or re.search(r'\)!', subj):
            breaking = True
            if f"- {short} {subj}" not in triggers:
                triggers.append(f"- {short} {subj}")
        # type detection
        m = re.match(r'^(?P<type>[^(:!]+)(?:\(.+?\))?(!)?:', subj)
        if m:
            t = m.group('type')
            if t == 'feat':
                has_feat = True
                triggers.append(f"- {short} {subj}")

    if breaking:
        return 'major', 'major (BREAKING CHANGE found)', triggers
    if has_feat:
        return 'minor', 'minor (feat commits found)', triggers
    return 'patch', 'patch (no feat/breaking commits found)', triggers


def suggest(commits: List[dict], current_tag: str, preserve_suffix: bool = True, suffix: str = '') -> tuple:
    """Return (suggested_version_str, reason, triggers, since_tag)

    If preserve_suffix is True and current_tag contains a -suffix, preserve it unless a --suffix is explicitly provided.
    """
    # derive current version from tag (strip leading v)
    curtag = current_tag or ''
    curver = curtag[1:] if curtag.startswith('v') else (curtag or '0.0.0')
    # detect existing suffix on current version
    existing_suffix = ''
    if '-' in curver:
        main, existing_suffix = curver.split('-', 1)
        curver_main = main
    else:
        curver_main = curver

    level, reason, triggers = analyze_commits(commits)
    base_suggested = bump_version(curver_main, level)

    # decide final suffix
    final_suffix = ''
    if preserve_suffix and existing_suffix:
        final_suffix = existing_suffix
    if suffix:
        # normalize provided suffix without leading '-'
        norm = suffix.lstrip('-')
        if final_suffix:
            # append with a hyphen separator
            final_suffix = f"{final_suffix}-{norm}"
        else:
            final_suffix = norm

    suggested = base_suggested + (f"-{final_suffix}" if final_suffix else '')
    # normalize suggested: ensure no double-dashes
    suggested = suggested.replace('--', '-')
    since = current_tag or ''
    return suggested, reason, triggers, since


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='')
    p.add_argument('--preserve-suffix', dest='preserve_suffix', action='store_true', default=True,
                   help='Preserve existing pre-release suffix from current tag (default: True)')
    p.add_argument('--no-preserve-suffix', dest='preserve_suffix', action='store_false',
                   help='Do not preserve existing pre-release suffix')
    p.add_argument('--suffix', default='', help='Append this suffix to suggested version (e.g. "-provenance-alpha")')
    p.add_argument('--json', action='store_true', help='Emit JSON with suggestion details')
    args = p.parse_args(argv)

    tag = args.tag or get_latest_tag()
    commits = get_commits_since(tag)
    suggestion, reason, triggers, since = suggest(commits, tag) if False else suggest(commits, tag, preserve_suffix=args.preserve_suffix, suffix=args.suffix)
    # Print in human-readable form
    if args.json:
        import json
        out = {
            'suggested': suggestion,
            'reason': reason,
            'triggers': triggers,
            'since': since,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"suggested: {suggestion}")
        print(f"reason: {reason}")
        print("triggers:")
        for t in triggers:
            print(t)
        if since:
            print(f"since: {since}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
