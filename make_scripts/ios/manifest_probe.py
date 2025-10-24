#!/usr/bin/env python3
"""Wrapper for manifest_probe moved into ios domain."""
from __future__ import annotations
try:
    from make_scripts.ios.manifest_probe import main as _main
    def main():
        return _main()
except Exception:
    def main():
        print('manifest_probe not available; original module missing')

if __name__ == '__main__':
    main()
