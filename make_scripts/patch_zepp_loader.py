#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.zepp.patch_zepp_loader"""
from importlib import import_module
_m = import_module('make_scripts.zepp.patch_zepp_loader')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
#!/usr/bin/env python3
"""Patch zepp/loader.py to respect ZEPPOVERRIDE_DIR env var (extracted from heredoc).
"""
from pathlib import Path
import os


def main():
    p = Path("etl_modules/cardiovascular/zepp/loader.py")
    if not p.exists():
        print(f"File not found: {p}")
        raise SystemExit(1)
    s = p.read_text(encoding="utf-8")

    if "ZEPPOVERRIDE_DIR" not in s:
        s = s.replace('def _first_existing(paths: List[str]) -> Optional[str]:', 'import os\n\ndef _first_existing(paths: List[str]) -> Optional[str]:')
        s = s.replace(
            'candidates = [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_hr_daily.csv"),',
            'envdir = os.environ.get("ZEPPOVERRIDE_DIR")\n        candidates = ([os.path.join(envdir, "zepp_hr_daily.csv")] if envdir else []) + [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_hr_daily.csv"),'
        )
        s = s.replace(
            'candidates = [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_daily_features.csv"),',
            'envdir = os.environ.get("ZEPPOVERRIDE_DIR")\n        candidates = ([os.path.join(envdir, "zepp_daily_features.csv")] if envdir else []) + [\n            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_daily_features.csv"),'
        )
        p.write_text(s, encoding="utf-8")
        print("WROTE: zepp/loader.py (env ZEPPOVERRIDE_DIR support)")
    else:
        print("SKIP: zepp/loader.py already supports ZEPPOVERRIDE_DIR")


if __name__ == '__main__':
    main()
