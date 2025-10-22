#!/usr/bin/env python3
"""Run compare_zepp_apple + plot_sleep_compare for the 'lite' flow.

This collects arguments and runs the two scripts sequentially. Supports --dry-run.
"""
from __future__ import annotations
import argparse
import subprocess
import sys


def run(cmd):
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--pid', required=True)
    p.add_argument('--snap', required=True)
    p.add_argument('--policy', default='best_of_day')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    compare_cmd = [sys.executable, 'etl_tools/compare_zepp_apple.py', '--pid', args.pid, '--zepp-root', f'data_etl/{args.pid}/zepp_processed', '--apple-dir', f'data_ai/{args.pid}/snapshots/{args.snap}', '--out-dir', f'data_ai/{args.pid}/snapshots/{args.snap}/hybrid_join', '--sleep-policy', args.policy]
    plot_cmd = [sys.executable, 'etl_tools/plot_sleep_compare.py', '--join', f'data_ai/{args.pid}/snapshots/{args.snap}/hybrid_join/join_hybrid_daily.csv', '--outdir', f'data_ai/{args.pid}/snapshots/{args.snap}/hybrid_join/plots']

    print('compare command:', ' '.join(compare_cmd))
    print('plot command:   ', ' '.join(plot_cmd))

    if args.dry_run:
        print('DRY-RUN: not executing commands')
        return 0

    rc = run(compare_cmd)
    if rc != 0:
        return rc
    return run(plot_cmd)


if __name__ == '__main__':
    raise SystemExit(main())
