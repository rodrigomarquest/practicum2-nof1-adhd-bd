#!/usr/bin/env python3
"""Minimal renderer used by Makefile release-draft.

Writes docs/release_notes/release_notes_<tag>.md and dist/changelog/CHANGELOG.dryrun.md.
"""
import argparse
import subprocess
import pathlib
import sys
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
    return sh("git log --pretty=oneline || true")


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

    rn_dir = pathlib.Path("docs/release_notes")
    rn_dir.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path("dist/changelog")
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Release {args.tag}"
    summary = args.summary or "Auto-generated draft."
    branch = args.branch or sh("git rev-parse --abbrev-ref HEAD")

    lines = [
        f"# 🚀 {args.tag} — {title}",
        "",
        f"**Release date (UTC):** {now}",
        f"**Branch:** `{branch}`",
        "",
        "## Summary",
        summary,
        "",
        "## Changes since last tag",
        "```text",
        changes,
        "```",
    ]

    rendered = "\n".join(lines) + "\n"

    rn_file = rn_dir / f"release_notes_{args.tag}.md"
    (out_dir / "CHANGELOG.dryrun.md").write_text(
        f"Wrote {rn_file.as_posix()}\n\n{changes}\n", encoding="utf-8"
    )

    if args.dry_run:
        print(f"DRY RUN: would write {rn_file.as_posix()}")
        print("\n".join(lines[:6]))
        return 0

    rn_file.write_text(rendered, encoding="utf-8")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(rendered, encoding="utf-8")
    print(f"Wrote {rn_file.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
