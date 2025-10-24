#!/usr/bin/env python3
"""Wrapper for normalize_ios_usage moved into ios domain."""
from __future__ import annotations
try:
    from make_scripts.ios.normalize_ios_usage import main as _main
    def main():
        return _main()
except Exception:
    def main():
        print('normalize_ios_usage not available; original module missing')

if __name__ == '__main__':
    main()
