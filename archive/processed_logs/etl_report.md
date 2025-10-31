ETL report: apple

Parallel execution and locking

- Running the same ETL target (e.g. `etl-apple-parse`) in parallel for the same participant/normalized path is not recommended.
- The ETL writers use a lightweight lockfile convention (`<file>.lock`) to avoid concurrent writes. This reduces the chance of corruption, but it does not guarantee that parallel runs produce the most recent snapshot in the case of races.
- Recommended: run ETL stages serially for a given participant/target. Locks are in place, but serial execution is the safest.

Stable CSVs & schema versioning

- CSV rows are written in deterministic order (sorted by primary time key) and columns are written in a fixed explicit order to ensure byte-stability across repeated runs.
- Numeric values are written with 6 decimal places where applicable (e.g. heart rate variability `sdnn_ms`).
- Every manifest includes `schema_version` (see `make_scripts/io_utils.py::SCHEMA_VERSION`). When the schema changes, bump the `SCHEMA_VERSION` constant. The bump will be incorporated in the manifest metadata and idempotency keys, forcing a clean recomputation.

Dry-run behavior

- Use `--dry-run` to preview writes. The dry-run now inspects manifests and will print UP-TO-DATE when outputs are present and match current content.

Notes

- The idempotency approach prefers file-content stability: manifests are not rewritten if the file SHA is unchanged. If you want stricter behaviour (require idempotency_key equality), adjust the idempotency check in the stage marker handling.
