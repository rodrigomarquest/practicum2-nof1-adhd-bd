#!/usr/bin/env python3
"""Insert a new section into CHANGELOG.md using docs/changelog_template.md.

Usage: tools/update_changelog.py --version VERSION --tag TAG [--title TITLE] [--summary SUMMARY_SHORT] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import os
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def get_previous_tag() -> str | None:
    try:
        # try annotated or lightweight previous tag
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "HEAD^"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        try:
            # fallback to previous tag reachable
            out = subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except Exception:
            return None


def build_section(template: str, mapping: dict) -> str:
    s = template
    for k, v in mapping.items():
        s = s.replace("{{" + k + "}}", str(v))
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--summary", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    tpl_path = Path("docs") / "changelog_template.md"
    changelog_path = Path("CHANGELOG.md")
    if not tpl_path.exists():
        print("ERROR: changelog template not found", file=sys.stderr)
        return 2
    tpl = tpl_path.read_text(encoding="utf-8")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    mapping = {
        "VERSION": args.version,
        "DATE_UTC": now,
        "TITLE": args.title or args.version,
        "SUMMARY_SHORT": args.summary or "--",
        "ADDED": "--",
        "CHANGED": "--",
        "FIXED": "--",
        "TESTS": "--",
        "NOTES": "--",
    }
    section = build_section(tpl, mapping).strip() + "\n\n"

    # read existing changelog (create if missing)
    existing = (
        changelog_path.read_text(encoding="utf-8")
        if changelog_path.exists()
        else "# Changelog\n\n"
    )

    # idempotent: if section already exists (by version line), no-op
    if (
        f"## [{args.version}]" in existing
        or args.version in existing.splitlines()[0:30]
    ):
        print("No update: section for version already present")
        return 0

    # insert after main heading (first line that startswith '# ')
    parts = existing.splitlines(keepends=True)
    insert_at = 1
    for i, line in enumerate(parts):
        if line.startswith("# "):
            insert_at = i + 1
            break

    new_text = "".join(parts[:insert_at]) + "\n" + section + "".join(parts[insert_at:])

    # append comparison link at bottom
    prev = get_previous_tag()
    repo_root = "https://github.com/<owner>/<repo>"
    compare_link = (
        f"[{args.version}]: {repo_root}/compare/{prev or ''}...{args.version}\n"
    )
    if compare_link not in new_text:
        new_text = new_text.rstrip() + "\n\n" + compare_link + "\n"

    if args.dry_run:
        # print a concise diff preview (show inserted header lines)
        print("DRY RUN: would insert section:")
        print("\n".join(section.splitlines()[:8]))
        return 0

    try:
        atomic_write(changelog_path, new_text)
    except Exception as e:
        print("ERROR writing CHANGELOG.md:", e, file=sys.stderr)
        return 3
    print("Updated CHANGELOG.md with version", args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
