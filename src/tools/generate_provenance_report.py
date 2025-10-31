#!/usr/bin/env python3
"""Generate a simple provenance report for a snapshot folder.

Writes:
- notebooks/eda_outputs/provenance/etl_provenance_report.csv
- notebooks/eda_outputs/provenance/data_audit_summary.md

This is intentionally conservative and uses heuristics to detect date columns.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from src.domains.common.io import etl_snapshot_root


def find_csvs(snapshot_dir: Path):
    return sorted(snapshot_dir.glob("**/*.csv"))


def guess_date_columns(df: pd.DataFrame):
    # heuristics: column names containing 'date', 'time', 'timestamp' or 'ts'
    candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ("date", "time", "timestamp", "ts"))
    ]
    # prefer columns that parse as datetimes (use utc=True for consistent tz handling)
    dates = []
    for c in candidates:
        try:
            s = pd.to_datetime(df[c].dropna().iloc[:100], utc=True, errors="coerce")
            if s.notna().any():
                dates.append(c)
        except Exception:
            continue
    return dates


def analyze_file(p: Path):
    info = {"file": str(p)}
    # attempt to read with pandas; on failure we will still compute row hashes by streaming
    try:
        df = pd.read_csv(p, low_memory=False)
        info["rows"] = str(int(len(df)))
        info["ncols"] = str(int(df.shape[1]))
        info["columns"] = ";".join(map(str, df.columns.tolist()))
        info["dtypes"] = ";".join(f"{c}:{str(t)}" for c, t in df.dtypes.items())
        info["null_counts"] = ";".join(
            f"{c}:{int(df[c].isna().sum())}" for c in df.columns
        )

        # primary time column selection (priority list)
        priority = ["date", "day", "datetime", "timestamp", "start_time", "end_time"]
        primary = None
        for key in priority:
            for c in df.columns:
                if c.lower() == key or key == c.lower():
                    primary = c
                    break
            if primary:
                break
        if primary is None:
            # fallback to heuristic guess
            date_cols = guess_date_columns(df)
            primary = date_cols[0] if date_cols else None

        notes = []
        if primary:
            try:
                raw_sample = df[primary].dropna().astype(str).head(200).tolist()
                tz_indicators = any(
                    (("+" in v and ":" in v) or v.endswith("Z")) for v in raw_sample
                )
                ts_utc = pd.to_datetime(df[primary], utc=True, errors="coerce")
                coerced_na = int(ts_utc.isna().sum())
                if tz_indicators and coerced_na > 0:
                    notes.append("tz_mixed")
                ts = ts_utc.dropna()
                if not ts.empty:
                    info["primary_time_col"] = primary
                    # start_date / end_date in ISO-8601 UTC
                    start = ts.min().tz_convert("UTC")
                    end = ts.max().tz_convert("UTC")
                    info["start_date"] = start.isoformat()
                    info["end_date"] = end.isoformat()
                    # coverage_days: if filename contains 'daily' treat as daily (inclusive span)
                    if "daily" in p.name.lower():
                        coverage_days = (end.date() - start.date()).days + 1
                    else:
                        coverage_days = int(ts.dt.date.nunique())
                    info["coverage_days"] = str(int(coverage_days))
                    info["tz_info"] = "UTC"
                    # anomalies
                    if ts.duplicated().any():
                        notes.append("duplicate_timestamps")
                    # non-monotonic: compare to sorted
                    if not ts.is_monotonic_increasing:
                        notes.append("non_monotonic_time")
            except Exception:
                pass

        # units mapping (per-file JSON string)
        units = detect_units_for_columns(df.columns.tolist())
        if units:
            info["units"] = json.dumps(units, ensure_ascii=False)

        # compute deterministic per-row SHA256 sample and count
        try:
            import hashlib

            hashes = df.astype(str).apply(
                lambda r: hashlib.sha256(
                    "|".join([str(r[c]) for c in df.columns]).encode("utf-8")
                ).hexdigest(),
                axis=1,
            )
            info["hash_count"] = str(int(len(hashes)))
            sample = hashes.head(5).tolist()
            info["hash_sha256_sample"] = json.dumps(sample)
        except Exception:
            pass

        if notes:
            info["notes"] = ";".join(notes)

        return info

    except Exception as e:
        # On read error, attempt to stream file lines and compute per-line hashes
        err = str(e)
        info["notes"] = f"error:{err}"
        try:
            import hashlib

            hashes = []
            count = 0
            with p.open("rb") as fh:
                for raw in fh:
                    line = raw.rstrip(b"\r\n")
                    h = hashlib.sha256(line).hexdigest()
                    if len(hashes) < 5:
                        hashes.append(h)
                    count += 1
            info["rows"] = str(int(count))
            info["hash_count"] = str(int(count))
            info["hash_sha256_sample"] = json.dumps(hashes)
        except Exception as e2:
            info["notes"] = f"error:{err};stream_error:{e2}"
        return info


def detect_units_for_columns(columns: list[str]) -> dict:
    """Heuristic mapping from column name to units.

    columns: list of column names (strings)
    returns: dict mapping column -> unit string
    """
    mapping = {
        ("hr", "heart_rate", "median_hr", "avg_hr", "mean_hr", "resting_hr"): "bpm",
        ("hrv", "rmssd", "rmssd", "sdnn"): "ms",
        (
            "sleep_",
            "waso",
            "total_sleep_time",
            "duration_minutes",
            "screentime",
            "app_minutes",
        ): "minutes",
        ("sleep_efficiency",): "%",
        ("temperature", "skin_temp"): "°C",
    }
    units: dict = {}
    for c in columns:
        unit = None
        lc = c.lower()
        for keys, u in mapping.items():
            for key in keys:
                if key in lc:
                    unit = u
                    break
            if unit:
                break
        if unit:
            units[c] = unit
    return units


def run_consistency_checks(snapshot_dir: Path, out_csv: Path):
    """Run Apple vs Zepp consistency checks and write etl_provenance_checks.csv."""
    snapshot = Path(snapshot_dir)
    csv_rows = []

    # helper to find file by name (exact or contains)
    def find_file(name):
        # search for exact filename first, else any file containing name
        candidates = list(snapshot.glob(f"**/{name}"))
        if candidates:
            return candidates[0]
        # fallback contains
        for p in snapshot.rglob("*.csv"):
            if name in p.name:
                return p
        return None

    pairs = [
        ("HR", "health_hr_daily.csv", "zepp_hr_daily.csv", 3),
        ("HRV", "health_hrv_daily.csv", "zepp_hrv_daily.csv", 6),
    ]

    # compute global snapshot span (days) from available files
    global_mins = []
    global_maxs = []
    for p in snapshot.rglob("*.csv"):
        try:
            df = pd.read_csv(p, low_memory=False)
            cols = guess_date_columns(df)
            if cols:
                col = cols[0]
                s = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
                if not s.empty:
                    global_mins.append(s.min())
                    global_maxs.append(s.max())
        except Exception:
            continue
    if global_mins and global_maxs:
        global_span = (min(global_mins), max(global_maxs))
        snapshot_length_days = int((global_span[1] - global_span[0]).days) + 1
    else:
        snapshot_length_days = 0

    for scope, a_name, b_name, tol in pairs:
        try:
            a_file = find_file(a_name)
            b_file = find_file(b_name)
            if a_file is None or b_file is None:
                # mark missing as error row
                csv_rows.append(
                    {
                        "check_id": f"{scope}_missing",
                        "check_type": "consistency",
                        "scope": scope,
                        "metric": scope,
                        "a_file": str(a_file) if a_file else "",
                        "b_file": str(b_file) if b_file else "",
                        "value": "",
                        "status": "error",
                        "details": "missing_file",
                    }
                )
                continue

            da = pd.read_csv(a_file, low_memory=False)
            db = pd.read_csv(b_file, low_memory=False)

            # detect date cols and parse
            ca = guess_date_columns(da)
            cb = guess_date_columns(db)
            if not ca or not cb:
                raise RuntimeError("no_date_col")
            ca = ca[0]
            cb = cb[0]
            da[ca] = pd.to_datetime(da[ca], utc=True, errors="coerce")
            db[cb] = pd.to_datetime(db[cb], utc=True, errors="coerce")

            # pick metric column (first numeric)
            numa = da.select_dtypes(include="number").columns.tolist()
            numb = db.select_dtypes(include="number").columns.tolist()
            if not numa or not numb:
                raise RuntimeError("no_numeric_metric")
            ma = numa[0]
            mb = numb[0]

            # aggregate to daily by date (UTC)
            da["__d"] = da[ca].dt.tz_convert("UTC").dt.date
            db["__d"] = db[cb].dt.tz_convert("UTC").dt.date
            sa = da.groupby("__d")[ma].median()
            sb = db.groupby("__d")[mb].median()
            joined = pd.concat([sa, sb], axis=1, join="inner").dropna()
            joined.columns = ["a", "b"]
            overlap_days = len(joined)
            if overlap_days == 0:
                raise RuntimeError("no_overlap")
            diffs = (joined["a"] - joined["b"]).dropna()
            median_diff = float(diffs.median())
            p10 = float(diffs.quantile(0.1))
            p50 = float(diffs.quantile(0.5))
            p90 = float(diffs.quantile(0.9))
            n = int(len(diffs))

            # status rules
            threshold = (
                60 if snapshot_length_days >= 60 else max(1, snapshot_length_days)
            )
            if overlap_days >= threshold and abs(median_diff) <= tol:
                status = "ok"
            else:
                status = "warn"

            csv_rows.append(
                {
                    "check_id": f"{scope}_consistency",
                    "check_type": "consistency",
                    "scope": scope,
                    "metric": scope,
                    "a_file": str(a_file),
                    "b_file": str(b_file),
                    "value": json.dumps(
                        {
                            "median_diff": median_diff,
                            "p10": p10,
                            "p50": p50,
                            "p90": p90,
                            "n": n,
                        }
                    ),
                    "status": status,
                    "details": f"overlap_days={overlap_days}",
                }
            )
        except Exception as e:
            csv_rows.append(
                {
                    "check_id": f"{scope}_error",
                    "check_type": "consistency",
                    "scope": scope,
                    "metric": scope,
                    "a_file": str(a_file) if "a_file" in locals() and a_file else "",
                    "b_file": str(b_file) if "b_file" in locals() and b_file else "",
                    "value": "",
                    "status": "error",
                    "details": str(e),
                }
            )

    # write checks CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "check_id",
        "check_type",
        "scope",
        "metric",
        "a_file",
        "b_file",
        "value",
        "status",
        "details",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def save_coverage_plot(report_csv: Path, out_png: Path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(report_csv)
    if "coverage_days" in df.columns:
        df["coverage_days"] = pd.to_numeric(
            df["coverage_days"], errors="coerce"
        ).fillna(0)
    else:
        df["coverage_days"] = 0
    df = df.sort_values("coverage_days")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, max(2, len(df) * 0.25)))
    plt.barh(df["file"], df["coverage_days"])
    plt.xlabel("coverage_days")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def update_markdown(
    summary_md: Path, checks_csv: Path, coverage_png: Path, report_csv: Path
):
    # Read existing summary or create new
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    if summary_md.exists():
        txt = summary_md.read_text(encoding="utf-8")
    else:
        txt = ""

    # overview if missing
    if "Overview" not in txt:
        # basic counts
        rpt = pd.read_csv(report_csv)
        n_files = len(rpt)
        min_dates = (
            rpt["min_date"].dropna().tolist() if "min_date" in rpt.columns else []
        )
        max_dates = (
            rpt["max_date"].dropna().tolist() if "max_date" in rpt.columns else []
        )
        if min_dates and max_dates:
            span = f"{min(min_dates)} → {max(max_dates)}"
        else:
            span = "n/a"
        header = f"# Overview\n\n- Files: {n_files}\n- Coverage span: {span}\n\n"
        txt = header + txt

    # Coverage section with PNG reference
    coverage_section = "\n## Coverage\n\n"
    coverage_section += f"![coverage]({coverage_png.as_posix()})\n\n"

    # Anomalies section
    rpt = pd.read_csv(report_csv)
    anomalies = []
    if "notes" in rpt.columns:
        for i, row in rpt.iterrows():
            if pd.notna(row["notes"]):
                anomalies.append(f"- {row['file']}: {row['notes']}")
    anomalies_section = "\n## Anomalies\n\n"
    if anomalies:
        anomalies_section += "\n".join(anomalies) + "\n\n"
    else:
        anomalies_section += "None found.\n\n"

    # Apple vs Zepp checks table
    checks = pd.read_csv(checks_csv)
    rows = []
    findings = []
    for i, r in checks.iterrows():
        if r["status"] in ("warn", "error"):
            findings.append(f"- {r['check_id']}: {r['status']} ({r['details']})")
        # attempt to extract median_diff from value
        val = r.get("value", "")
        median = ""
        try:
            j = json.loads(val) if val else {}
            median = j.get("median_diff", "")
        except Exception:
            median = ""
        rows.append(
            f"| {r['scope']} | {r['a_file']} | {r['b_file']} | {r.get('details','')} | {median} | {r['status']} |"
        )

    checks_section = "\n## Apple vs Zepp Consistency Checks\n\n"
    checks_section += (
        "| scope | a_file | b_file | days_overlap | median_diff | status |\n"
    )
    checks_section += "|---|---:|---:|---:|---:|---:|\n"
    checks_section += "\n".join(rows) + "\n\n"

    findings_section = "\n## Findings & Recommendations\n\n"
    if findings:
        findings_section += "\n".join(findings) + "\n\n"
    else:
        findings_section += "- No WARN/ERROR items detected.\n\n"

    # append or replace sections at the end
    txt = txt + coverage_section + anomalies_section + checks_section + findings_section
    summary_md.write_text(txt, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    # default to a canonical data/etl path for the demo snapshot
    p.add_argument(
        "--snapshot",
        default=str(etl_snapshot_root("P000001", "2025-09-29")),
    )
    p.add_argument("--outdir", default="notebooks/eda_outputs/provenance")
    args = p.parse_args(argv)

    snapshot = Path(args.snapshot)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_csvs(snapshot)
    rows = []
    for f in files:
        print("ANALYZE:", f)
        info = analyze_file(f)
        rows.append(info)

    # write CSV (flatten rows with missing keys)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    out_csv = outdir / "etl_provenance_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    # summary markdown
    summary = outdir / "data_audit_summary.md"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write("# Data Audit Summary\n\n")
        fh.write(f"* Snapshot: {snapshot}\n")
        fh.write(f"* Files scanned: {len(files)}\n\n")
        fh.write("## Per-file overview\n\n")
        for r in rows:
            fh.write(
                f"- {r.get('file')}: rows={r.get('rows','?')} cols={r.get('ncols','?')}\n"
            )

    print("WROTE:", out_csv, summary)


if __name__ == "__main__":
    main()
