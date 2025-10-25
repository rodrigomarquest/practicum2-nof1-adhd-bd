#!/usr/bin/env python3
"""Import-sweep grouped by domain for make_scripts.

This script walks all modules under `make_scripts` and attempts to import
each one, grouping results by the first domain component (e.g. `apple`,
`zepp`, `ios`, or top-level modules grouped under 'root'). It prints a
JSON summary of successes and any tracebacks.
"""
import importlib
import importlib.util
import importlib.machinery
import pkgutil
import traceback
import json
import sys
import os

# Ensure repo root is on sys.path so `make_scripts` can be imported from tools/
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import make_scripts

results = {}
total_ok = 0
total_err = 0

for finder, name, ispkg in pkgutil.walk_packages(make_scripts.__path__, prefix=make_scripts.__name__ + '.'):
    # name is full like 'make_scripts.apple.foo'
    parts = name.split('.')
    domain = parts[1] if len(parts) > 1 else 'root'
    results.setdefault(domain, {'ok': 0, 'err': 0, 'errors': {}})
    try:
        importlib.import_module(name)
        results[domain]['ok'] += 1
        total_ok += 1
    except Exception:
        tb = traceback.format_exc()
        results[domain]['err'] += 1
        results[domain]['errors'][name] = tb
        total_err += 1

summary = {'total_ok': total_ok, 'total_err': total_err, 'by_domain': results}
print(json.dumps(summary, indent=2))
