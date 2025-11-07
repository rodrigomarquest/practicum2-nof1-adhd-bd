#!/usr/bin/env python3
"""Run src.etl_pipeline inside a Timer to show our progress header/footer.

This wrapper imports the package and calls main() so internal progress
bars (from src.domains.common.progress) continue to work as designed.
"""
from __future__ import annotations
import sys

def _run_with_timer(argv: list[str]) -> int:
    # Import late so PYTHONPATH can be set by Makefile
    try:
        from src.domains.common.progress import Timer
        import src.etl_pipeline as etl
    except Exception as e:
        print(f"[run_etl_with_timer] import failed: {e}")
        return 2

    # Prepare sys.argv for the etl module's argparse
    sys.argv = ["etl_pipeline"] + argv
    cmd_desc = f"etl {' '.join(argv[:1]) if argv else ''}"
    with Timer(cmd_desc):
        try:
            etl.main()
            return 0
        except SystemExit as se:
            # argparse or the module may call sys.exit()
            return int(se.code) if isinstance(se.code, int) else 1
        except Exception:
            import traceback

            traceback.print_exc()
            return 1


def main() -> int:
    return _run_with_timer(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
