Snapshot date incoherence across sources (Apple Health export, iTunes backup, AutoExport, Zepp)

Description

We observed inconsistent snapshot dates across different input sources for the same participant/snapshot, causing the ETL to treat files as belonging to different snapshots and leading to misaligned outputs.

Sources involved

- Apple Health Export (in-app .zip)
- Apple iTunes Backup (exported files inside backup tree)
- Apple AutoExport app (AutoExport CSVs)
- Zepp export (device/zip from Zepp/Zepp app)

Observed behavior (example)

- Participant: `P000001` Snapshot argument: `2025-09-29`
- `data/raw/P000001/apple/export/...apple_health_export_20251022T061854Z.zip` (Apple export has internal timestamp 2025-10-22)
- `data/raw/P000001/itunes_backup/...` (itunes backup files have different modification dates)
- `data/raw/P000001/autoexport/...` (AutoExport files may have another timestamp)
- `data/raw/P000001/zepp/...zip` (Zepp archive timestamp differs)

When running `etl extract` with `--snapshot 2025-09-29` the ETL currently discovers and extracts these files, but the files' internal timestamps and/or file naming dates do not consistently match the requested snapshot date. This leads to:

- Extracted outputs with inconsistent time ranges across per-metric CSVs
- Possible incorrect joins or filtering later in the pipeline
- Confusion about which input file is authoritative for the snapshot date

Expected behavior (proposal)

- ETL should record (and persist in `extract_manifest.json`) the original source filename, file-size, and original modification timestamp for each imported archive/file.
- ETL should use a clear rule to determine the canonical snapshot date for a given run, for example (in order):
  1. Use explicit `--snapshot` parameter when provided (current behavior), but validate and warn if no source file falls into the given date range.
  2. If no explicit `--snapshot` is provided, infer snapshot date from the most-recent file that matches the participant and the configured source priority (Apple export > AutoExport > iTunes backup > Zepp), but write the inference into the manifest and log a warning.
  3. Provide a small migration/compatibility helper to map legacy `snapshots/`-style layouts into `data/etl/<PID>/<YYYY-MM-DD>/` while preserving source metadata.

Reproduction steps

1. Place the following files under `data/raw/P000001/` (already present in this repo):
   - `apple/export/...apple_health_export_20251022T061854Z.zip`
   - `zepp/...zip`
   - any available `itunes_backup` and `autoexport` inputs
2. Run: `python -m scripts.run_etl_with_timer extract --participant P000001 --snapshot 2025-09-29 --auto-zip`
3. Inspect `data/etl/P000001/2025-09-29/extract_manifest.json` and `data/etl/P000001/2025-09-29/extracted/*` and observe that source file timestamps differ from the provided `--snapshot` and from each other.

Suggested labels

- bug
- data-quality
- help-wanted (optional)

Additional notes

- I can provide CLI logs and exact file paths / md5/sha256 hashes if helpful.
- We should decide whether the `--snapshot` parameter is a strict requirement (ETL should fail if sources don't match) or a best-effort mapping with warnings.

Assignee: (none)
