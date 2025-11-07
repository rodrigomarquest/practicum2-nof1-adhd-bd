import importlib
import sys

CANDIDATES = [
    "etl_modules.config",
    "etl_modules.io_utils",
    "etl_modules.common",
    "etl_modules.common.progress",
    "etl_modules.common.io",
    "etl_modules.common.adapters",
    "etl_modules.common.segments",
    "etl_modules.cardiovascular",
    "etl_modules.cardiovascular.cardio_etl",
    "etl_modules.cardiovascular.cardio_features",
    "etl_modules.cardiovascular.apple.loader",
    "etl_modules.cardiovascular.zepp.loader",
    "etl_modules.parse_zepp_export",
    "etl_modules.zepp_join",
    "etl_modules.iphone_backup",
]

failed = []
for name in CANDIDATES:
    try:
        importlib.import_module(name)
        print("[OK]", name)
    except Exception as e:
        print("[FAIL]", name, "-", repr(e))
        failed.append((name, repr(e)))

if failed:
    print("\nMissing or broken shims:")
    for n, err in failed:
        print(" -", n, ":", err)
    sys.exit(1)
else:
    print("\nAll shim imports resolved.")
