#!/usr/bin/env python3
"""Utilities for manifest handling in make_scripts.utils.

This module delegates to the canonical implementation in
``make_scripts.utils.print_manifest``. It exists so callers can import
``make_scripts.utils.manifests.main`` while the concrete implementation
lives in `print_manifest.py`.
"""
from __future__ import annotations
try:
    from make_scripts.utils.print_manifest import main as _main
except Exception:
    _main = None


def main(argv=None):
    if _main is not None:
        return _main(argv)
    print('print_manifest not available; original module missing')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
