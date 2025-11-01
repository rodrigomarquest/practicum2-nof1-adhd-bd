#!/usr/bin/env python3
"""Move existing 'snapshots/<date>' directories up one level to '<date>'.

Usage:
  python scripts/migrate_snapshots.py [--dry-run] [--pid PID]

This will search under data/etl/<PID>/snapshots/* and move each snapshot
folder to data/etl/<PID>/<snapshot> preserving contents. It will skip targets
that already exist unless --force is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = Path("data/etl")
    pids = [args.pid] if args.pid else [p.name for p in root.iterdir() if p.is_dir()]

    for pid in pids:
        src_root = root / pid / "snapshots"
        if not src_root.exists():
            print(f"[skip] no snapshots dir for {pid}")
            continue
        for snap in sorted([d for d in src_root.iterdir() if d.is_dir()]):
            dst = root / pid / snap.name
            print(f"{pid}: {snap} -> {dst}")
            if dst.exists():
                if not args.force:
                    print(f"[skip] destination exists: {dst}")
                    continue
                # else remove destination first
                print(f"[force] removing existing {dst}")
                if not args.dry_run:
                    shutil.rmtree(dst)
            if args.dry_run:
                continue
            shutil.move(str(snap), str(dst))
        # remove snapshots dir if empty
        try:
            if not any((root / pid / "snapshots").iterdir()):
                if args.dry_run:
                    print(f"[dry-run] would remove {root / pid / 'snapshots'}")
                else:
                    (root / pid / "snapshots").rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
