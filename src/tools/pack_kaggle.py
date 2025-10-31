import sys
import os
import zipfile
import pathlib


def main():
    pid = os.environ.get("PID") or os.environ.get("PARTICIPANT") or "P000001"
    snap = os.environ.get("SNAPSHOT") or "2025-09-29"
    base = pathlib.Path(f"data/etl/{pid}/snapshots/{snap}/joined")
    files = list(base.glob("features_daily*.csv"))
    v = base / "version_log_enriched.csv"
    if v.exists():
        files.append(v)
    readme = pathlib.Path("README.md")
    if readme.exists():
        files.append(readme)
    if not files:
        print(f"[pack-kaggle] no files in {base}", file=sys.stderr)
        sys.exit(1)
    outdir = pathlib.Path("dist/assets")
    outdir.mkdir(parents=True, exist_ok=True)
    zipfn = outdir / f"{pid}_{snap}_ai.zip"
    with zipfile.ZipFile(zipfn, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    print(f"[pack-kaggle] wrote {zipfn}")


if __name__ == "__main__":
    main()
