#!/usr/bin/env python3
import re
from pathlib import Path

M = Path('Makefile')
bak = Path('Makefile.bak')
text = M.read_text()
bak.write_text(text)

lines = text.splitlines()
out = []
in_recipe = False
target_re = re.compile(r'^[^\s].*:\s*(#.*)?$')

for i, line in enumerate(lines):
    stripped = line.strip()
    # detect target lines (start of a rule)
    if target_re.match(line):
        in_recipe = True
        out.append(line)
        continue

    # blank line ends recipe context
    if stripped == '':
        in_recipe = False
        out.append(line)
        continue

    if in_recipe:
        # if already starts with recipe prefix '>' leave it
        if line.lstrip().startswith('>'):
            out.append(line)
            continue
        # comments in recipes leave as-is (but prefix to be safe?) keep as-is
        if stripped.startswith('#'):
            out.append(line)
            continue
        # For any other non-empty line in a recipe, prefix with '> '
        out.append('> ' + line)
    else:
        out.append(line)

new = '\n'.join(out) + '\n'
if new != text:
    M.write_text(new)
    print('Patched Makefile recipes; backup written to', bak.name)
else:
    print('No changes made; recipes already normalized.')
