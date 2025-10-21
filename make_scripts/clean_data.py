#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe cleanup of snapshot outputs and provenance outputs within the repository.

Usage (from repo root):
    python make_scripts/clean_data.py --snapshot data_ai/P000001/snapshots --provenance notebooks/eda_outputs/provenance
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import sys


def safe_rm_tree(target: Path, repo_root: Path) -> bool:
    """Remove target tree only if it is inside repo_root. Returns True if removed."""
    try:
        target_resolved = target.resolve()
        repo_resolved = repo_root.resolve()
        if not str(target_resolved).startswith(str(repo_resolved)):
            print(f"Refusing to remove outside repo: {target}")
            return False
        if not target.exists():
            return False
        # remove contents but keep the parent dir
        if target.is_dir():
            for child in target.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            return True
        else:
            target.unlink()
            return True
    except Exception as e:
        print(f"Error removing {target}: {e}")
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', default='data_ai/P000001/snapshots')
    p.add_argument('--provenance', default='notebooks/eda_outputs/provenance')
    args = p.parse_args(argv)

    repo_root = Path.cwd()
    snap = Path(args.snapshot)
    prov = Path(args.provenance)

    print(f"Cleaning snapshot outputs: {snap}")
    removed_snap = safe_rm_tree(snap, repo_root)
    print("Snapshot cleaned." if removed_snap else "No snapshot dir to clean or nothing removed.")

    print(f"Cleaning provenance outputs: {prov}")
    removed_prov = safe_rm_tree(prov, repo_root)
    print("Provenance cleaned." if removed_prov else "No provenance outputs to clean or nothing removed.")


if __name__ == '__main__':
    main()
