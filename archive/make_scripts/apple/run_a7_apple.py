#!/usr/bin/env python3
"""Wrapper for run_a7_apple moved into apple domain."""
from __future__ import annotations
try:
    from make_scripts.apple.run_a7_apple import main as _main
    def main():
        return _main()
except Exception:
    def main():
        print('run_a7_apple not available; original module missing')

if __name__ == '__main__':
    main()
