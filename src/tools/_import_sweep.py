#!/usr/bin/env python3
"""Small import sweep for make_scripts modules (used by the agent)."""
import pkgutil
import importlib
import traceback
#!/usr/bin/env python3
"""Small import sweep for make_scripts modules (used by the agent)."""
import pkgutil
import importlib
import traceback
import json

names = [m.name for m in pkgutil.iter_modules(['make_scripts'])]
ok = 0
err = 0
errs = {}
importlib.invalidate_caches()
for n in names:
    try:
        importlib.import_module('make_scripts.' + n)
        ok += 1
    except Exception:
        errs[n] = traceback.format_exc()
        err += 1

print('IMPORT_SWEEP_OK', ok, 'ERR', err)
print(json.dumps(errs, indent=2))

