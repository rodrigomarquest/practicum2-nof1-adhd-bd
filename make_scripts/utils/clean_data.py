#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe cleanup of snapshot outputs and provenance outputs within the repository.

Usage (from repo root):
    python -m make_scripts.utils.clean_data --snapshot data_ai/P000001/snapshots --provenance notebooks/eda_outputs/provenance

This script is conservative by default. It only removes files inside the repository and keeps parent
directories. Use --dry-run to preview deletions, and the data_etl PID deletion requires typing 'YES'.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import List


def safe_rm_tree(target: Path, repo_root: Path, dry_run: bool = False) -> list[Path]:
    """Remove target tree only if it is inside repo_root. Returns list of removed (or would-be removed) paths.

    When dry_run=True, nothing is deleted; instead the function prints what would be removed and returns the
    list of candidates.
    """
    try:
        target_resolved = target.resolve()
        repo_resolved = repo_root.resolve()
        if not str(target_resolved).startswith(str(repo_resolved)):
            print(f"Refusing to remove outside repo: {target}")
            return []
        if not target.exists():
            return []

        removed: list[Path] = []
        if target.is_dir():
            children = list(target.iterdir())
            removed.extend(children)
            if dry_run:
                list_children_for_dryrun(target, children)
                return removed
            _remove_children(children)
            return removed
        else:
            removed.append(target)
            if dry_run:
                print(f"[dry-run] Would remove file: {target}")
                return removed
            target.unlink()
            return removed
    except Exception as e:
        print(f"Error removing {target}: {e}")
        return []


def list_children_for_dryrun(parent: Path, children: List[Path]) -> None:
    print(f"[dry-run] Would remove the following inside {parent}:")
    for child in children:
        print(f"  - {child}")


def _remove_children(children: List[Path]) -> None:
    for child in children:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prompt_yes(question: str, default: bool = False) -> bool:
    try:
        resp = input(question).strip()
    except EOFError:
        return default
    if resp == '':
        return default
    return resp.lower() in ('y', 'yes')


def prompt_exact(question: str, token: str) -> bool:
    try:
        resp = input(question)
    except EOFError:
        return False
    return resp.strip() == token


def infer_pid(snapshot_path: Path) -> str | None:
    parts = list(snapshot_path.parts)
    # support both repository layout variants: data_ai/PID/... and data/ai/PID/...
    if 'data_ai' in parts:
        try:
            i = parts.index('data_ai')
            return parts[i + 1]
        except Exception:
            return None
    # alternative layout: data/ai/PID/...
    if len(parts) >= 3:
        for i in range(len(parts) - 2):
            if parts[i] == 'data' and parts[i + 1] == 'ai':
                return parts[i + 2]
    return None


def process_snapshot(snap: Path, repo_root: Path, dry_run: bool) -> list[Path]:
    print(f"Cleaning snapshot outputs: {snap}")
    removed_snap = safe_rm_tree(snap, repo_root, dry_run=dry_run)
    if removed_snap:
        print("Snapshot would be cleaned (dry-run)." if dry_run else "Snapshot cleaned.")
    else:
        print("No snapshot dir to clean or nothing removed.")
    return removed_snap


def process_provenance(prov: Path, repo_root: Path, assume_yes: bool, dry_run: bool) -> list[Path]:
    do_prov = assume_yes or prompt_yes(f"Remove provenance outputs at {prov}? [y/N]: ")
    if do_prov:
        print(f"Cleaning provenance outputs: {prov}")
        removed_prov = safe_rm_tree(prov, repo_root, dry_run=dry_run)
        if removed_prov:
            print("Provenance would be cleaned (dry-run)." if dry_run else "Provenance cleaned.")
        else:
            print("No provenance outputs to clean or nothing removed.")
        return removed_prov
    else:
        print("Skipping provenance cleanup.")
        return []


def process_data_etl(snap: Path, repo_root: Path, dry_run: bool) -> list[Path]:
    pid = infer_pid(snap)
    detl_path = (repo_root / 'data_etl' / pid) if pid else (repo_root / 'data_etl')
    do_detl = prompt_exact(
        f"Also remove data_etl for pid '{pid or ''}' at {detl_path}? This is destructive. Type 'YES' to proceed: ",
        'YES',
    )
    if do_detl:
        print(f"Cleaning data_etl outputs: {detl_path}")
        removed_detl = safe_rm_tree(detl_path, repo_root, dry_run=dry_run)
        if removed_detl:
            print("data_etl would be cleaned (dry-run)." if dry_run else "data_etl cleaned.")
        else:
            print("No data_etl dir to clean or nothing removed.")
        return removed_detl
    else:
        print("Skipping data_etl cleanup.")
        return []


def process_data_etl_noninteractive(snap: Path, repo_root: Path, dry_run: bool) -> list[Path]:
    pid = infer_pid(snap)
    detl_path = (repo_root / 'data_etl' / pid) if pid else (repo_root / 'data_etl')
    print(f"(non-interactive) Cleaning data_etl outputs: {detl_path}")
    removed_detl = safe_rm_tree(detl_path, repo_root, dry_run=dry_run)
    if removed_detl:
        print("data_etl would be cleaned (dry-run)." if dry_run else "data_etl cleaned.")
    else:
        print("No data_etl dir to clean or nothing removed.")
    return removed_detl


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', default='data_ai/P000001/snapshots')
    p.add_argument('--provenance', default='notebooks/eda_outputs/provenance')
    p.add_argument('--yes', '-y', action='store_true', help='Assume yes to all prompts')
    p.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting')
    p.add_argument('--delete-data-etl', action='store_true', help='Non-interactive: allow deleting data_etl (requires explicit flag)')
    p.add_argument('--audit-log', default=None, help='Path to write an audit log of deletions (defaults to provenance/cleanup_log_TIMESTAMP.txt)')
    args = p.parse_args(argv)

    repo_root = Path.cwd()
    snap = Path(args.snapshot)
    prov = Path(args.provenance)

    removed_paths: list[Path] = []
    removed_paths.extend(process_snapshot(snap, repo_root, args.dry_run) or [])
    removed_paths.extend(process_provenance(prov, repo_root, args.yes, args.dry_run) or [])
    # data_etl deletion can be controlled non-interactively
    if args.delete_data_etl:
        # perform without interactive prompt
        removed_paths.extend(process_data_etl_noninteractive(snap, repo_root, args.dry_run))
    else:
        removed_paths.extend(process_data_etl(snap, repo_root, args.dry_run) or [])

    # write audit log if requested or if deletions occurred
    if args.audit_log or removed_paths:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        default_log = Path('notebooks/eda_outputs/provenance') / f'cleanup_log_{ts}.txt'
        log_path = Path(args.audit_log) if args.audit_log else default_log
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('w', encoding='utf8') as fh:
                fh.write(f"cleanup run: {ts}\n")
                fh.write(f"dry_run: {args.dry_run}\n")
                for p in removed_paths:
                    fh.write(str(p) + "\n")
            print(f"Wrote audit log: {log_path}")
        except Exception as e:
            print(f"Failed to write audit log {log_path}: {e}")


if __name__ == '__main__':
    main()
