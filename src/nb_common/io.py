from pathlib import Path
import json


def resolve_slug_path(slug: str, data_root: str):
    root = Path(data_root)
    p = root / slug / "features_daily_labeled.csv"
    return p


def write_run_config(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(cfg, fh, indent=2)
