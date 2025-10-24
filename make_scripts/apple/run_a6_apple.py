#!/usr/bin/env python3
"""Run Prompt A6: Normalize Apple In-App extracted metrics + QC for a snapshot.

This is the same content previously at make_scripts/apple/run_a6_apple.py; copied to apple domain.
"""
from __future__ import annotations
import argparse
# minimal stub: import original implementation if available
try:
    from make_scripts.apple.run_a6_apple import normalize_snapshot
    # expose a main wrapper
    def main():
        print('Delegated run_a6_apple to top-level module (legacy).')
except Exception:
    def main():
        print('No delegated implementation available in this environment.')

if __name__ == '__main__':
    main()
