#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.run_a8_labels

Preserves the historical import path `make_scripts.run_a8_labels` and
delegates the implementation to `make_scripts.apple.run_a8_labels`.
"""
from __future__ import annotations
import sys


def main(argv=None):
    from make_scripts.apple.run_a8_labels import main as apple_main
    return apple_main(argv)


if __name__ == '__main__':
    sys.exit(main())
