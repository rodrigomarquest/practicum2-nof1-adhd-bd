from pathlib import Path


def etl_snapshot_dir(pid: str, snap: str, base: str | Path = "data/etl") -> Path:
    """Return the canonical ETL snapshot directory for a participant.

    Historically some tools wrote to data/etl/<pid>/snapshots/<snap>/..., but the
    preferred canonical layout is data/etl/<pid>/<snap>/.... This helper centralizes
    the decision and can be used by tools to read/write snapshots without hardcoding
    the intermediate "snapshots" path.

    It returns the preferred path (without the intermediate 'snapshots').
    """
    basep = Path(base)
    return basep / pid / snap


def etl_snapshot_dir_legacy(pid: str, snap: str, base: str | Path = "data/etl") -> Path:
    """Return the legacy path that includes the intermediate 'snapshots' folder.

    Use when implementing a migration or a backward-compatibility shim.
    """
    basep = Path(base)
    return basep / pid / "snapshots" / snap
