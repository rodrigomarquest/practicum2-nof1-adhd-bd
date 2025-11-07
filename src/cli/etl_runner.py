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
import os
import logging


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # initialize logger for extract CLI. honor ETL_LOG_LEVEL env var once.
    logger = logging.getLogger("etl.extract")
    # Configure basic logging once; default to INFO when ETL_LOG_LEVEL not set.
    lvl_name = os.getenv("ETL_LOG_LEVEL", "INFO")
    try:
        lvl = getattr(logging, lvl_name.upper(), logging.INFO)
    except Exception:
        lvl = logging.INFO
    logging.basicConfig(level=lvl)

    parser = argparse.ArgumentParser(prog="etl_runner")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("extract", help="Extract Apple/Zepp sources")
    # Support both --pid and legacy --participant used by Makefile
    p.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    p.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    p.add_argument("--snapshot", default="auto", help="Snapshot id YYYY-MM-DD or 'auto' (default)")
    p.add_argument("--auto-zip", action="store_true", dest="auto_zip", help="Auto-discover zips")
    p.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run discovery only")
    p.add_argument("--zepp-zip-password", dest="zepp_zip_password", default=None, help="Password for Zepp encrypted ZIP files")

    p_full = sub.add_parser("full", help="Run full ETL (delegates to src.etl_pipeline.main)")
    p_full.add_argument("--pid", required=True)
    p_full.add_argument("--snapshot", required=True)

    p_join = sub.add_parser("join", help="Join per-domain features into canonical joined CSV")
    p_join.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    p_join.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    p_join.add_argument("--snapshot", default="auto", help="Snapshot id YYYY-MM-DD or 'auto' (default)")
    p_join.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run (no writes)")

    p_enrich = sub.add_parser("enrich", help="Run global enrichment over the canonical joined CSV")
    p_enrich.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    p_enrich.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    p_enrich.add_argument("--snapshot", default="auto", help="Snapshot id YYYY-MM-DD or 'auto' (default)")
    p_enrich.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run (no writes)")

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

    if args.cmd == "join":
        try:
            import src.etl_pipeline as etl
        except Exception:
            try:
                import etl_pipeline as etl  # type: ignore
            except Exception:
                raise

        pid = args.pid
        snap = args.snapshot
        if snap == "auto":
            snap = etl.resolve_snapshot(snap, pid)
        snap_dir = etl.etl_snapshot_root(pid, snap)
        rc = etl.join_run(snap_dir, dry_run=bool(args.dry_run))
        return int(rc)

    if args.cmd == "enrich":
        try:
            import src.etl_pipeline as etl
        except Exception:
            try:
                import etl_pipeline as etl  # type: ignore
            except Exception:
                raise

        try:
            # try the project import path first
            from src.domains.enriched import enrich_global as enrich_mod
        except Exception:
            try:
                from domains.enriched import enrich_global as enrich_mod  # type: ignore
            except Exception:
                # last-resort: import by module name
                import importlib

                enrich_mod = importlib.import_module("src.domains.enriched.enrich_global")

        pid = args.pid
        snap = args.snapshot
        if snap == "auto":
            snap = etl.resolve_snapshot(snap, pid)
        snap_dir = etl.etl_snapshot_root(pid, snap)
        rc = enrich_mod.enrich_run(snap_dir, dry_run=bool(args.dry_run))
        return int(rc)

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
            # Pre-discovery log: show discovered sources (vendor/variant) so
            # the CLI clearly reports what will be processed. This mirrors the
            # behaviour of extract_run(dry_run=True) but prints a concise list.

            try:
                sources = etl.discover_sources(pid, auto_zip=auto_zip)
                if sources:
                    print(f"INFO: discovered {len(sources)} source(s) for pid={pid} snapshot={snap_arg}")
                    for s in sources:
                        v = s.get("vendor")
                        va = s.get("variant")
                        zp = s.get("zip_path")
                        print(f" - {v}/{va}: {zp}")
            except Exception:
                # best-effort discovery logging; do not block execution
                pass

            # Zepp password: log a WARNING if missing (do not crash; extract_run
            # will also handle missing password and skip Zepp). Use logger so the
            # message is visible on stderr by default and not echo the password.
            zepp_pwd = args.zepp_zip_password or os.getenv("ZEPP_ZIP_PASSWORD")
            if zepp_pwd:
                # Set env var so extract_run can access it
                os.environ["ZEPP_ZIP_PASSWORD"] = zepp_pwd
            elif os.getenv("ZEPP_ZIP_PASSWORD") in (None, ""):
                logger = logging.getLogger("etl.extract")
                msg = "ZEPP_ZIP_PASSWORD not set; Zepp cloud exports will be skipped."
                logger.warning(msg)
                # also print to stderr so it's visible to users and tests
                print(f"WARNING: {msg}", file=sys.stderr)

            rc = etl.extract_run(pid=pid, snapshot_arg=snap_arg, auto_zip=auto_zip, dry_run=dry_run)
            # extract_run uses exit codes: 0 (ok), 2 (no sources produced)
            return int(rc)
        except SystemExit as se:
            return int(se.code) if isinstance(se.code, int) else 1
        except Exception:
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
