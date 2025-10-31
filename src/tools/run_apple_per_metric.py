import sys
from pathlib import Path
from etl_modules.apple_raw_to_per_metric import export_per_metric
import os
from src.domains.common.io import per_metric_dir, etl_snapshot_root

RAW = Path(os.environ.get("ETL_DIR", "data_etl"))
RAW_RAW = Path(os.environ.get("RAW_DIR", "data/raw"))


def canon_snap(s: str) -> str:
    s = s.strip()
    return (
        s
        if len(s) == 10 and s[4] == "-" and s[7] == "-"
        else f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    )


def find_xml(pid: str, snap_iso: str):
    cands = [
        RAW / pid / "snapshots" / snap_iso / "export.xml",
        RAW / pid / "snapshots" / snap_iso.replace("-", "") / "export.xml",
        RAW / pid / snap_iso / "export.xml",
        RAW / pid / snap_iso.replace("-", "") / "export.xml",
        # refactored/extracted layout under ETL_DIR
        RAW / pid / "extracted" / "apple" / "export.xml",
        RAW / pid / "extracted" / "export.xml",
        # consolidated RAW layout (zip or extracted)
        RAW_RAW / pid / "apple" / "export.zip",
        RAW_RAW / pid / "apple" / "export.xml",
        # legacy fallback
        Path("data/etl") / pid / "extracted" / "apple" / "export.xml",
        Path("data/etl") / pid / "extracted" / "export.xml",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def ensure_out(pid: str, snap_iso: str) -> Path:
    # Use canonical ETL layout for per-metric outputs
    out_root = etl_snapshot_root(pid, snap_iso)
    return per_metric_dir(pid, snap_iso)


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python run_apple_per_metric.py <PARTICIPANT> <SNAPSHOT> <CUTOVER> <TZ_BEFORE> <TZ_AFTER>"
        )
        sys.exit(1)
    pid, snap, cutover, tz_before, tz_after = sys.argv[1:]
    snap_iso = canon_snap(snap)
    xml = find_xml(pid, snap_iso)
    if xml is None:
        print("export.xml not found.")
        sys.exit(2)
    outdir = ensure_out(pid, snap_iso)
    written = export_per_metric(xml, outdir, cutover, tz_before, tz_after)
    if not written:
        print("No per-metric files were produced.")
        sys.exit(3)
    for k, v in written.items():
        print(" -", k, "->", v)


if __name__ == "__main__":
    main()
