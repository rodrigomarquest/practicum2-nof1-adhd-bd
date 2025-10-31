"""Generate a deterministic repository reorganization plan for MM2.

This script scans the repository tree, applies heuristics to decide which
files under `make_scripts/` should move to domain subfolders (apple, ios,
zepp, utils) or be archived, and writes a deterministic JSON plan to
tools/audit/reorg_plan.json. It supports dry-run mode (DRY_RUN=1 env or
--dry-run flag) which only writes the plan and does not perform moves.

This tool must not move any data under data/raw/** and only references
relative repo paths.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "data/raw",
    "__pycache__",
}


def is_excluded(p: pathlib.Path) -> bool:
    rel = p.relative_to(REPO_ROOT)
    parts = rel.parts
    for ex in EXCLUDE_DIRS:
        ex_parts = pathlib.Path(ex).parts
        if parts[: len(ex_parts)] == ex_parts:
            return True
    return False


def scan_repo() -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for root, dirs, fnames in os.walk(REPO_ROOT, topdown=True):
        # prune excluded dirs
        relroot = pathlib.Path(root).relative_to(REPO_ROOT)
        if any(str(relroot).startswith(ex) for ex in EXCLUDE_DIRS if ex):
            # skip this entire subtree
            dirs[:] = []
            continue
        # remove excluded subdirs from traversal
        dirs[:] = [d for d in dirs if not is_excluded(pathlib.Path(root) / d)]
        for f in fnames:
            fp = pathlib.Path(root) / f
            if is_excluded(fp):
                continue
            files.append(fp)
    return sorted(files)


def make_move_plan(files: List[pathlib.Path]) -> List[Dict]:
    plan: List[Dict] = []
    for fp in files:
        rel = fp.relative_to(REPO_ROOT)
        rel_str = str(rel).replace("\\", "/")
        entry = {
            "src_path": rel_str,
            "dst_path": None,
            "action": "keep",
            "reason": "default: keep",
        }

        # Heuristics for make_scripts reorg
        if rel.parts[0] == "make_scripts":
            name = fp.name.lower()
            # map by filename patterns
            if any(x in name for x in ("zepp", "zepp_", "zepp-")):
                dst = pathlib.Path("make_scripts/zepp") / fp.name
                entry.update(
                    {"dst_path": str(dst), "action": "move", "reason": "zepp domain"}
                )
            elif any(
                x in name
                for x in ("ios", "ios_", "ios-", "manifest_probe", "backup_probe")
            ):
                dst = pathlib.Path("make_scripts/ios") / fp.name
                entry.update(
                    {"dst_path": str(dst), "action": "move", "reason": "ios domain"}
                )
            elif any(
                x in name
                for x in (
                    "apple",
                    "a6_",
                    "a7_",
                    "a8_",
                    "apple-",
                    "run_a6",
                    "run_a7",
                    "run_a8",
                )
            ):
                dst = pathlib.Path("make_scripts/apple") / fp.name
                entry.update(
                    {"dst_path": str(dst), "action": "move", "reason": "apple domain"}
                )
            elif any(
                x in name
                for x in (
                    "progress",
                    "atomic",
                    "snapshot_lock",
                    "hash",
                    "manifest",
                    "manifests",
                )
            ):
                dst = pathlib.Path("make_scripts/utils") / fp.name
                entry.update(
                    {"dst_path": str(dst), "action": "move", "reason": "utils domain"}
                )
            elif name.endswith(".py") and fp.name in ("__init__.py",):
                entry.update({"action": "keep", "reason": "package init"})
            else:
                # candidate for legacy
                dst = pathlib.Path("make_scripts/legacy") / fp.name
                entry.update(
                    {
                        "dst_path": str(dst),
                        "action": "archive",
                        "reason": "legacy/one-off candidate",
                    }
                )

        # root-level archive candidates
        elif rel.parts[0] in (
            "etl_pipeline_legacy.py",
            "etl_pipeline.py.bak",
            "fix_cardio_files.sh",
        ) or rel.suffix in (".bak",):
            dst = pathlib.Path("archive") / rel
            entry.update(
                {
                    "dst_path": str(dst),
                    "action": "archive",
                    "reason": "root legacy file",
                }
            )

        plan.append(entry)

    # deduplicate: ensure deterministic ordering
    plan_sorted = sorted(plan, key=lambda e: (e["src_path"]))
    return plan_sorted


def write_plan(plan: List[Dict], out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # ensure deterministic JSON
    with open(out, "w", encoding="utf8") as fh:
        json.dump({"plan": plan}, fh, indent=2, sort_keys=True)


def print_table(plan: List[Dict]) -> None:
    # terse table: src -> action -> dst
    print("src_path,action,dst_path,reason")
    for e in plan:
        print(
            f"{e['src_path']},{e['action']},{e.get('dst_path') or ''},{e.get('reason','')}"
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Only write plan; do not move files"
    )
    parser.add_argument(
        "--out",
        default="tools/audit/reorg_plan.json",
        help="Output plan path (repo-relative)",
    )
    args = parser.parse_args(argv or sys.argv[1:])

    dry_run_env = os.environ.get("DRY_RUN", "0") == "1"
    dry_run = args.dry_run or dry_run_env

    files = scan_repo()
    plan = make_move_plan(files)

    out_path = REPO_ROOT / args.out
    # Print terse table
    print_table(plan)

    if dry_run:
        print(f"DRY_RUN enabled: writing plan only to {out_path}")
        write_plan(plan, out_path)
        print("Plan written (dry-run). No file moves performed.")
        return 0

    # Not dry-run: as a safety we still only write the plan; actual move step is manual
    print(
        f"Not dry-run. Writing plan to {out_path} (no automatic moves performed by this tool)."
    )
    write_plan(plan, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
