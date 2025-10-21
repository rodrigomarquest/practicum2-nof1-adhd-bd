#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print Python and key package versions.

Usage: python make_scripts/env_versions.py
"""
from __future__ import annotations
import importlib
import sys


def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, '__version__', 'unknown')
    except Exception:
        return 'not installed'


def main():
    print('python', sys.version.splitlines()[0])
    print('pandas', ver('pandas'))
    print('numpy', ver('numpy'))
    print('matplotlib', ver('matplotlib'))

if __name__ == '__main__':
    main()
