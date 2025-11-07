# Backward-compat shim -> new location under src/domains
from importlib import import_module as _imp

# Map legacy names to new modules
_MAP = {
    "apple_raw_to_per_metric": "src.domains.apple_raw_to_per_metric",
    "config": "src.domains.config",
    "io_utils": "src.domains.io_utils",
    "parse_zepp_export": "src.domains.parse_zepp_export",
    "zepp_join": "src.domains.zepp_join",
    # cardiovascular subpackages
    "cardiovascular": "src.domains.cardiovascular",
    "cardiovascular.cardio_etl": "src.domains.cardiovascular.cardio_etl",
    "cardiovascular.cardio_features": "src.domains.cardiovascular.cardio_features",
    "cardiovascular.apple.loader": "src.domains.cardiovascular.apple.loader",
    "cardiovascular.zepp.loader": "src.domains.cardiovascular.zepp.loader",
    "common": "src.domains.common",
    "common.adapters": "src.domains.common.adapters",
    "common.io": "src.domains.common.io",
    "common.progress": "src.domains.common.progress",
    "common.segments": "src.domains.common.segments",
    "extract_screen_time": "src.domains.extract_screen_time",
    "iphone_backup": "src.domains.iphone_backup",
}


def __getattr__(name):
    full = _MAP.get(name)
    if not full:
        raise AttributeError(f"etl_modules shim: unknown attribute {name!r}")
    return _imp(full)


def __dir__():
    return sorted(list(_MAP))
