#!/usr/bin/env python3
"""Wrapper for Zepp parsing to keep Make thin.

This script mirrors the existing Makefile behavior but provides a proper CLI and --dry-run.
"""
from __future__ import annotations
import argparse
import subprocess
import sys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".zip file or directory input")
    p.add_argument("--outdir-root", required=True, help="Output root directory")
    p.add_argument("--tz", default=None)
    p.add_argument("--tz-before", dest="tz_before", default=None)
    p.add_argument("--tz-after", dest="tz_after", default=None)
    p.add_argument("--participant", default=None)
    p.add_argument("--cutover", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cmd = [sys.executable, "etl_modules/parse_zepp_export.py", "--input", args.input, "--outdir-root", args.outdir_root]
    if args.tz:
        cmd += ["--tz", args.tz]
    if args.tz_before:
        cmd += ["--tz_before", args.tz_before]
    if args.tz_after:
        cmd += ["--tz_after", args.tz_after]
    if args.participant:
        cmd += ["--participant", args.participant]
    if args.cutover:
        cmd += ["--cutover", args.cutover]
    if args.password:
        cmd += ["--password", args.password]

    print("Running:", " ".join(cmd))
    if args.dry_run:
        print("DRY-RUN: not executing parse command")
        return 0

    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
