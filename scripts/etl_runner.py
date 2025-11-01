#!/usr/bin/env python3
"""Wrapper to run ETL commands with small pre-checks and Timer wrapper for extract.

Usage: python scripts/etl_runner.py <cmd> --participant PID --snapshot SNAPSHOT --cutover DATE --tz_before TZ --tz_after TZ
"""
from __future__ import annotations
import sys
import os


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: etl_runner.py <cmd> [--participant PID] [--snapshot SNAPSHOT] ...")
        return 2
    cmd = argv[0]
    # Extract common args by simple parsing (we forward them to etl module)
    # We'll look for --participant and --snapshot values
    def get_arg(name: str) -> str | None:
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None

    pid = get_arg("--participant")
    snap = get_arg("--snapshot")

    if cmd == "full":
        if not pid or not snap:
            print("ERROR: full run requires --participant and --snapshot")
            return 2
        # check extracted data
        extracted_dir = os.path.join("data", "etl", pid, snap, "extracted")
        if not os.path.isdir(extracted_dir):
            print(f"ERROR: extracted data not found at {extracted_dir}. Run extract first: make etl ETL_CMD=extract PID={pid} SNAPSHOT={snap}")
            return 3
        # run full etl directly
        try:
            import src.etl_pipeline as etl
            sys.argv = ["etl_pipeline"] + argv
            etl.main()
            return 0
        except Exception:
            import traceback

            traceback.print_exc()
            return 4

    elif cmd == "extract":
        # delegate to run_etl_with_timer for consistent Timer display
        try:
            import scripts.run_etl_with_timer as runner
        except Exception:
            # try import by path
            try:
                from run_etl_with_timer import _run_with_timer as _run
                return _run(argv)
            except Exception:
                import traceback

                traceback.print_exc()
                return 5
        # call internal helper
        return runner._run_with_timer(argv)

    else:
        print(f"Unknown ETL cmd: {cmd}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
