#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.run_a7_apple

Preserves the historical import path `make_scripts.run_a7_apple` and
delegates the implementation to `make_scripts.apple.run_a7_apple`.
"""
from __future__ import annotations
import sys


def main(argv=None):
    from make_scripts.apple.run_a7_apple import main as apple_main
    return apple_main(argv)


if __name__ == '__main__':
    sys.exit(main())
