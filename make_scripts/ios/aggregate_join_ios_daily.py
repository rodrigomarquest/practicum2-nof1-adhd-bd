#!/usr/bin/env python3
"""Wrapper for aggregate_join_ios_daily moved into ios domain."""
from __future__ import annotations
try:
    from make_scripts.ios.aggregate_join_ios_daily import main as _main
    def main():
        return _main()
except Exception:
    def main():
        print('aggregate_join_ios_daily not available; original module missing')

if __name__ == '__main__':
    main()
