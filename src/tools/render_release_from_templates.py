#!/usr/bin/env python3
"""Renderer used by Makefile release-draft.

This augments the previous minimal renderer by filling a canonical
release-note template and producing a concise commit-grouped summary
for the SUMMARY and HIGHLIGHTS sections.

It preserves --dry-run and --out options and still writes a
dist/changelog/CHANGELOG.dryrun.md file for review.
"""
import argparse
import subprocess
import pathlib
import sys
import datetime
import textwrap
from collections import defaultdict


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
    return sh("git log --pretty=oneline || true")


def parse_commits(changes_text: str):
    """Return list of (sha, message) tuples from git log --pretty=oneline output."""
    commits = []
    for line in changes_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            sha, msg = parts
        else:
            sha = parts[0]
            msg = ""
        commits.append((sha, msg))
    return commits


KNOWN_PREFIXES = [
    "feat:",
    "fix:",
    "chore:",
    "docs:",
    "refactor:",
    "test:",
    "perf:",
    "ci:",
]


def group_commits(commits):
    groups = defaultdict(list)
    for sha, msg in commits:
        low = msg.lower()
        found = None
        for p in KNOWN_PREFIXES:
            if low.startswith(p):
                found = p.rstrip(":")
                # strip the prefix from message for display
                display = msg[len(p) :].strip()
                break
        if not found:
            found = "other"
            display = msg
        groups[found].append((sha, display))
    return groups


def summarize_group(name, items, max_bullets=3):
    # pick up to max_bullets representative messages
    bullets = []
    seen = set()
    for _, msg in items:
        s = msg.strip()
        if not s:
            continue
        # truncate long messages
        s_short = (s[:140] + "...") if len(s) > 140 else s
        if s_short in seen:
            continue
        seen.add(s_short)
        bullets.append(s_short)
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        bullets = ["(no concise messages available)"]
    # return header + bullets
    header = f"### {name.capitalize()} ({len(items)})"
    body = "\n".join([f"- {b}" for b in bullets])
    return header + "\n" + body


TEMPLATE = textwrap.dedent(
    """
# 🚀 {VERSION} – {TITLE}

**Release date:** {DATE}  
**Branch:** {BRANCH}  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🧩 Summary
{SUMMARY_PARAGRAPH}

---

## 🌿 Highlights
{HIGHLIGHTS}

---

## 🧱 Build & Provenance
* Commit range: {COMMIT_RANGE}
* Tag: {TAG}
* Renderer: `render_release_from_templates.py`

---

## 🔎 Appendix – Changes since last tag
{CHANGELOG_SNIPPET}

"""
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--branch", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    now = datetime.datetime.utcnow().isoformat() + "Z"
    last_tag = get_last_tag()
    changes = get_changes_since(last_tag)
    commits = parse_commits(changes)
    commit_count = len(commits)

    rn_dir = pathlib.Path("docs/release_notes")
    rn_dir.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path("dist/changelog")
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Release {args.tag}"
    summary_arg = args.summary or "Auto-generated draft."
    branch = args.branch or sh("git rev-parse --abbrev-ref HEAD")

    # Group commits and prepare highlights
    groups = group_commits(commits)
    # prepare a short summary paragraph
    if last_tag:
        commit_range = f"{last_tag}..HEAD"
    else:
        commit_range = "HEAD (no previous tag)"

    group_counts = ", ".join([f"{k}:{len(v)}" for k, v in groups.items()])
    summary_paragraph = (
        f"{summary_arg}\n\nSummary of commits since {last_tag or 'start'}: {group_counts}."
    )

    # build highlights: 2-3 bullets per group
    highlights_sections = []
    for k in sorted(groups.keys(), key=lambda x: (x != "feat", x)):
        sec = summarize_group(k, groups[k], max_bullets=3)
        highlights_sections.append(sec)
    highlights_text = "\n\n".join(highlights_sections)

    # changelog snippet: raw git log truncated to reasonable length
    changelog_snippet = "```text\n" + changes.strip() + "\n```" if changes.strip() else "(no commits)"

    rendered = TEMPLATE.format(
        VERSION=args.version,
        TITLE=title,
        DATE=now,
        BRANCH=branch,
        SUMMARY_PARAGRAPH=summary_paragraph,
        HIGHLIGHTS=highlights_text,
        COMMIT_RANGE=commit_range,
        TAG=args.tag,
        CHANGELOG_SNIPPET=changelog_snippet,
    )

    # write dry-run changelog for review
    rn_file = rn_dir / f"release_notes_v{args.version}.md"
    (out_dir / "CHANGELOG.dryrun.md").write_text(
        f"Wrote {rn_file.as_posix()}\n\n{changes}\n", encoding="utf-8"
    )

    if args.dry_run:
        print(f"DRY RUN: would write {rn_file.as_posix()}")
        print(rendered[:1200])
        return 0

    rn_file.write_text(rendered, encoding="utf-8")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(rendered, encoding="utf-8")

    print(f"[OK] Wrote {rn_file.as_posix()} (canonical template)")
    print(f"[OK] Summarized {commit_count} commits since {last_tag or 'initial commit'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
