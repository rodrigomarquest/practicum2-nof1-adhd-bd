#!/usr/bin/env python3
"""Command line runner for ETL tasks.

This script exposes a small CLI with an `extract` subcommand that unifies
Apple (inapp/itunes/autoextract) and Zepp exports into a single snapshot run.

It delegates the heavy lifting to `src.etl_pipeline.extract_run` while
displaying a Timer header/footer (using `src.domains.common.progress.Timer`).
"""
from __future__ import annotations
import sys
import argparse


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="etl_runner")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("extract", help="Extract Apple/Zepp sources")
    # Support both --pid and legacy --participant used by Makefile
    p.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    p.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    p.add_argument("--snapshot", default="auto", help="Snapshot id YYYY-MM-DD or 'auto' (default)")
    p.add_argument("--auto-zip", action="store_true", dest="auto_zip", help="Auto-discover zips")
    p.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run discovery only")
    # legacy/compat flags forwarded by Makefile; accept and ignore
    p.add_argument("--cutover", required=False, help=argparse.SUPPRESS)
    p.add_argument("--tz_before", required=False, help=argparse.SUPPRESS)
    p.add_argument("--tz_after", required=False, help=argparse.SUPPRESS)

    p_full = sub.add_parser("full", help="Run full ETL (delegates to src.etl_pipeline.main)")
    p_full.add_argument("--pid", required=True)
    p_full.add_argument("--snapshot", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "full":
        # delegate to the existing etl.main() to preserve behavior
        try:
            import src.etl_pipeline as etl
            sys.argv = ["etl_pipeline", "full", "--participant", args.pid, "--snapshot", args.snapshot]
            etl.main()
            return 0
        except Exception:
            import traceback
            traceback.print_exc()
            return 4

    if args.cmd != "extract":
        parser.print_help()
        return 2

    # call the new extract_run() in src.etl_pipeline
    try:
        import src.domains.common.progress as progress_mod
        import src.etl_pipeline as etl
    except Exception:
        # best-effort import path if running from project root
        try:
            import etl_pipeline as etl  # type: ignore
        except Exception:
            raise

    # Resolve snapshot and run inside the Timer for consistent UI
    snap_arg = args.snapshot
    pid = args.pid
    auto_zip = bool(args.auto_zip)
    dry_run = bool(args.dry_run)

    # Use Timer from src.domains.common.progress if available
    Timer = getattr(progress_mod, "Timer", None)
    if Timer is None:
        # fallback to a no-op context manager
        from contextlib import nullcontext as Timer

    cmd_desc = f"etl extract [{pid}/{snap_arg}]"
    with Timer(cmd_desc):
        try:
            rc = etl.extract_run(pid=pid, snapshot_arg=snap_arg, auto_zip=auto_zip, dry_run=dry_run)
            return int(rc)
        except SystemExit as se:
            return int(se.code) if isinstance(se.code, int) else 1
        except Exception:
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
