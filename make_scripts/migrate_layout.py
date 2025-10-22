#!/usr/bin/env python3
"""migrate_layout.py

Safely migrate legacy ETL export folders into the new raw/ layout.
Writes an audit JSON to provenance/migrate_layout_<UTC_COMPACT>.json

Usage (examples):
  python make_scripts/migrate_layout.py --participant P000001 --dry-run
  python make_scripts/migrate_layout.py --legacy-etl-dir data/etl/P000001 --raw-dir data/raw/P000001

Precedence: CLI args > ENV vars > defaults
"""
from __future__ import annotations

import argparse
import os
import shutil
import hashlib
import json
import datetime
from datetime import timezone
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def timestamp_compact_utc() -> str:
    # Use timezone-aware UTC timestamp to avoid deprecation warnings
    return datetime.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def copy_or_move(src: Path, dst: Path, dry_run: bool) -> Dict[str, Any]:
    action = "moved"
    if dry_run:
        action = "would_move"
        return {
            "action": action,
            "src": str(src),
            "dst": str(dst),
            "size": src.stat().st_size,
            "sha256": sha256_of_file(src),
        }

    ensure_parent(dst)
    # handle collisions by renaming
    if dst.exists():
        ts = timestamp_compact_utc()
        dst = dst.with_name(f"{dst.stem}_migrated_{ts}{dst.suffix}")

    # attempt to preserve mtime
    mtime = src.stat().st_mtime
    shutil.move(str(src), str(dst))
    try:
        os.utime(dst, (mtime, mtime))
    except Exception:
        pass

    return {
        "action": action,
        "src": str(src),
        "dst": str(dst),
        "size": dst.stat().st_size,
        "sha256": sha256_of_file(dst),
    }


def find_legacy_paths(etl_dir: Path) -> Dict[str, List[Path]]:
    results = {"apple": [], "zepp": []}
    exports_dir = etl_dir / "exports"
    if not exports_dir.exists():
        return results

    apple_dir = exports_dir / "apple"
    zepp_dir = exports_dir / "zepp"
    if apple_dir.exists():
        for p in apple_dir.rglob("*"):
            if p.is_file():
                results["apple"].append(p)
    if zepp_dir.exists():
        for p in zepp_dir.rglob("*"):
            if p.is_file():
                results["zepp"].append(p)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--participant")
    p.add_argument("--legacy-etl-dir")
    p.add_argument("--raw-dir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--non-interactive", action="store_true")
    args = p.parse_args(argv)

    # ENV defaults
    env = os.environ
    participant = args.participant or env.get("PARTICIPANT") or "P000001"
    legacy_etl_dir = Path(args.legacy_etl_dir or env.get("ETL_DIR") or f"data/etl/{participant}")
    raw_dir = Path(args.raw_dir or env.get("RAW_DIR") or f"data/raw/{participant}")
    dry_run = args.dry_run or env.get("DRY_RUN") == "1"
    non_interactive = args.non_interactive or env.get("NON_INTERACTIVE") == "1"

    now_ts = timestamp_compact_utc()
    provenance_dir = Path("provenance")
    provenance_dir.mkdir(exist_ok=True)
    audit_file = provenance_dir / f"migrate_layout_{now_ts}.json"

    actions: List[Dict[str, Any]] = []

    legacy = find_legacy_paths(legacy_etl_dir)

    # map and move files
    for kind in ("apple", "zepp"):
        for src in legacy.get(kind, []):
            # compute relative path under exports/<kind>/
            rel = src.relative_to(legacy_etl_dir / "exports" / kind)
            dst = raw_dir / kind / rel
            # On dry run just report
            if dry_run:
                actions.append({"kind": kind, "result": copy_or_move(src, dst, dry_run=True)})
                print(f"would move {src} -> {dst}")
                continue

            # interactive confirmation if files are large and not non_interactive
            size = src.stat().st_size
            if size > 100 * 1024 * 1024 and not non_interactive:
                resp = input(f"File {src} is >100MB ({size} bytes). Move? [y/N]: ")
                if resp.lower() != "y":
                    actions.append({"kind": kind, "result": {"action": "skipped", "src": str(src), "reason": "user_declined"}})
                    continue

            # do the move and record hashes
            before_hash = sha256_of_file(src)
            res = copy_or_move(src, dst, dry_run=False)
            res["sha256_before"] = before_hash
            actions.append({"kind": kind, "result": res})

    # Extra: look for runs/*/raw/apple_inapp and offer to copy export.xml into extracted
    runs_dir = legacy_etl_dir / "runs"
    if runs_dir.exists():
        for run_sub in runs_dir.iterdir():
            if not run_sub.is_dir():
                continue
            apple_inapp = run_sub / "raw" / "apple_inapp"
            if apple_inapp.exists():
                # find export.xml sibling if present
                export_xml = run_sub / "raw" / "export.xml"
                if export_xml.exists():
                    target = Path(raw_dir) / "apple" / "export.xml"
                    if dry_run:
                        print(f"would copy {export_xml} -> {target}")
                        actions.append({"kind": "apple", "result": {"action": "would_copy_export_xml", "src": str(export_xml), "dst": str(target)}})
                    else:
                        ensure_parent(target)
                        if target.exists():
                            ts = timestamp_compact_utc()
                            target = target.with_name(f"{target.stem}_migrated_{ts}{target.suffix}")
                        shutil.copy2(str(export_xml), str(target))
                        actions.append({"kind": "apple", "result": {"action": "copied_export_xml", "src": str(export_xml), "dst": str(target), "sha256": sha256_of_file(target)}})

    audit = {
        "participant": participant,
        "legacy_etl_dir": str(legacy_etl_dir),
        "raw_dir": str(raw_dir),
        "dry_run": bool(dry_run),
        "timestamp": now_ts,
        "actions": actions,
    }

    # write audit JSON always (even for dry-run)
    with audit_file.open("w", encoding="utf8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"Wrote audit: {audit_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
