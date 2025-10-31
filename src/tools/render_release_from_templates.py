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


def get_changes_detailed(tag: str) -> str:
    """Return git log lines with short-sha, author, and subject separated by TAB."""
    if tag:
        return sh(f"git log --pretty=format:%h%x09%an%x09%s {tag}..HEAD || true")
    return sh("git log --pretty=format:%h%x09%an%x09%s || true")


def parse_commits(changes_text: str):
    """Return list of (sha, author, message) tuples from git log with TAB-separated fields."""
    commits = []
    for line in changes_text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            sha, author, msg = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            sha, author = parts[0].strip(), parts[1].strip()
            msg = ""
        elif parts:
            sha = parts[0].strip()
            author = "(unknown)"
            msg = ""
        else:
            continue
        commits.append((sha, author, msg))
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
    p.add_argument("--last-tag", dest="last_tag", default=None,
                   help="Explicit last tag to compute changelog range (optional).")
    p.add_argument("--branch", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # compute last_tag: if explicit provided use it; otherwise pick the newest tag
    # that is not the target tag (args.tag). This avoids using the target tag as base
    # when the tag already exists locally.
    def compute_last_tag(explicit_last_tag, target_tag):
        if explicit_last_tag:
            return explicit_last_tag
        try:
            tags_out = sh("git tag --sort=-creatordate || true")
        except Exception:
            tags_out = ""
        tags = [t.strip() for t in tags_out.splitlines() if t.strip()]
        for t in tags:
            if t != target_tag:
                return t
        # fallback to git describe --tags
        lt = get_last_tag()
        if lt and lt != target_tag:
            return lt
        return ""

    last_tag = compute_last_tag(args.last_tag, args.tag)

    # get detailed commits (sha, author, subject)
    changes_detailed = get_changes_detailed(last_tag)
    commits = parse_commits(changes_detailed)
    commit_count = len(commits)

    rn_dir = pathlib.Path("docs/release_notes")
    rn_dir.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path("dist/changelog")
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Release {args.tag}"
    summary_arg = args.summary or "Auto-generated draft."
    branch = args.branch or sh("git rev-parse --abbrev-ref HEAD")
    git_user = sh("git config user.name || true") or "(unknown)"

    # Group commits and prepare highlights
    # group_commits expects list of (sha, author, msg)
    groups = defaultdict(list)
    for sha, author, msg in commits:
        low = msg.lower()
        found = None
        display = msg
        for p in KNOWN_PREFIXES:
            if low.startswith(p):
                found = p.rstrip(":")
                display = msg[len(p) :].strip()
                break
        if not found:
            found = "other"
        groups[found].append((sha, author, display))

    if last_tag:
        commit_range = f"{last_tag}..HEAD"
    else:
        commit_range = "HEAD (no previous tag)"

    # counts by group
    group_counts = ", ".join([f"{k}:{len(v)}" for k, v in groups.items() if len(v) > 0]) or "none"
    # compute unique authors count
    unique_authors = len(set([a for _, a, _ in commits]))
    summary_paragraph = (
        f"{summary_arg}\n\n**Since {last_tag or 'start'}:** {commit_count} commits · {group_counts} · {unique_authors} authors."
    )

    # build highlights: 2-3 bullets per group
    highlights_sections = []
    for k in sorted(groups.keys(), key=lambda x: (x != "feat", x)):
        # summarize_group expects items as (sha, display) pairs
        items = [(sha, disp) for sha, _, disp in groups[k]]
        sec = summarize_group(k, items, max_bullets=3)
        highlights_sections.append(sec)
    highlights_text = "\n\n".join(highlights_sections)

    # prepare full changelog block (oneline format) for appendix
    full_changelog = get_changes_since(last_tag)
    changelog_snippet = "```text\n" + full_changelog.strip() + "\n```" if full_changelog.strip() else "(no commits)"

    # metrics: counts by type and top authors
    by_type_items = sorted(((k, len(v)) for k, v in groups.items() if len(v) > 0), key=lambda x: -x[1])
    by_type_inline = ", ".join([f"{k}:{cnt}" for k, cnt in by_type_items]) or "none"
    by_type_multiline = "\n".join([f"- {k}: {cnt}" for k, cnt in by_type_items]) or "- none"

    # top authors
    authors = defaultdict(int)
    for sha, author, _ in commits:
        authors[author] += 1
    top_authors = sorted(authors.items(), key=lambda x: -x[1])[:5]
    top_authors_text = ", ".join([f"{n} ({c})" for n, c in top_authors]) or "(none)"

    # Prepare context for template rendering
    template_path = pathlib.Path(__file__).parent / "templates" / "release_canonical.md.j2"

    stats_block = textwrap.dedent(
        """
    - Total: {total}
    {by_type}
    - Top authors: {top_authors}
    """
    ).format(total=commit_count, by_type=by_type_multiline, top_authors=top_authors_text)

    next_steps = "\n".join([
        "- Integrate Zepp ETL",
        "- Add CI release tests",
        "- Retrain multi-snapshot models",
    ])

    context = {
        "VERSION": args.version,
        "TITLE": title,
        "DATE": now,
        "BRANCH": branch,
        "LAST_TAG": last_tag or "",
        "COMMIT_COUNT": commit_count,
        "BY_TYPE_LINE": by_type_inline,
        "AUTHOR_COUNT": unique_authors,
        "TOP_AUTHORS": top_authors_text,
        "HIGHLIGHTS_BLOCKS": highlights_text,
        "STATS_BLOCK": stats_block,
        "NEXT_STEPS": next_steps,
        "TAG": args.tag,
        "CHANGELOG_SNIPPET": changelog_snippet,
        "SUMMARY_PARAGRAPH": summary_paragraph,
    }

    if template_path.exists():
        try:
            import jinja2

            tpl = jinja2.Template(template_path.read_text(encoding="utf-8"))
            rendered = tpl.render(**context)
        except Exception:
            # Fallback: naive placeholder replacement for {{ VAR }} patterns
            tpl_text = template_path.read_text(encoding="utf-8")
            rendered = tpl_text
            for k, v in context.items():
                rendered = rendered.replace("{{ %s }}" % k, str(v))
                rendered = rendered.replace("{{%s}}" % k, str(v))
    else:
        # fallback to the built-in TEMPLATE
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

    # write dry-run changelog for review (full oneline log)
    rn_file = rn_dir / f"release_notes_v{args.version}.md"
    (out_dir / "CHANGELOG.dryrun.md").write_text(
        f"Range: {commit_range}\n\n{full_changelog}\n", encoding="utf-8"
    )

    if args.dry_run:
        print(f"DRY RUN: would write {rn_file.as_posix()}")
        print(rendered[:1200])
        return 0

    rn_file.write_text(rendered, encoding="utf-8")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(full_changelog + "\n", encoding="utf-8")

    print(f"[OK] Wrote {rn_file.as_posix()} (canonical template)")
    print(f"[OK] Summarized {commit_count} commits since {last_tag or 'initial commit'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
