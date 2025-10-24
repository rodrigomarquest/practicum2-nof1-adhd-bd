#!/usr/bin/env python3
"""Top-level shim for aggregate_join_ios_daily.

Delegates to `make_scripts.ios.aggregate_join_ios_daily` to centralize
implementation while preserving the top-level entrypoint.
"""
from __future__ import annotations
import sys


def main(argv=None):
    from make_scripts.ios.aggregate_join_ios_daily import main as ios_main
    return ios_main(argv)


if __name__ == '__main__':
    sys.exit(main())
