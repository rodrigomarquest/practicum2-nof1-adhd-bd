#!/usr/bin/env python3
"""Simple linter for Makefile herd-doc tokens.
Exits with code 2 if heredoc tokens '<<' or '<<-' are found.
"""
import sys
from pathlib import Path
import re

mfpath = Path('Makefile')
if not mfpath.exists():
    print('Makefile not found', file=sys.stderr)
    sys.exit(1)
text = mfpath.read_text(encoding='utf-8')
# Look for << or <<- occurrences (common heredoc markers)
if re.search(r'(^|\s)<<-?', text):
    print('ERROR: heredoc token found in Makefile. Move multi-line logic to make_scripts/*.py and call it from Make.')
    sys.exit(2)
print('lint-make: OK (no heredoc tokens found)')
