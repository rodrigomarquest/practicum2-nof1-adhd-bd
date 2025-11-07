import os
import subprocess
from pathlib import Path
import csv
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

PID = "PTEST01"
SNAP = "2025-11-06"


def write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def run_module(module: str, pid: str, snap: str, dry_run: int, env=None):
    cmd = [sys.executable, "-m", module, "--pid", pid, "--snapshot", snap, "--dry-run", str(dry_run)]
    env2 = os.environ.copy()
    env2.update({"PYTHONPATH": str(REPO_ROOT / "src")})
    if env:
        env2.update(env)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc


def test_cardio_sleep_cli(tmp_path):
    # create synthetic HEARTRATE and SLEEP CSVs under data/etl/<PID>/<SNAP>/extracted/zepp/cloud
    base = REPO_ROOT / "data" / "etl" / PID / SNAP / "extracted" / "zepp" / "cloud"
    hr_dir = base / "HEARTRATE"
    sl_dir = base / "SLEEP"

    # HEARTRATE CSV: timestamp,value
    hr_csv = hr_dir / "hr.csv"
    write_csv(hr_csv, ["timestamp", "value"], [["2025-11-06T01:00:00Z", 60], ["2025-11-06T02:00:00Z", 70]])

    # SLEEP CSV: date,total_hours,deep_hours,light_hours,rem_hours
    sl_csv = sl_dir / "sleep.csv"
    write_csv(sl_csv, ["date", "total_hours", "deep_hours", "light_hours", "rem_hours"], [["2025-11-06", 7.5, 1.2, 5.8, 0.5]])

    # Dry-run cardio
    p = run_module("domains.cardiovascular.cardio_from_extracted", PID, SNAP, 1)
    assert p.returncode == 0, f"cardio dry-run failed: {p.stderr}\n{p.stdout}"
    assert "[dry-run] cardio seed" in p.stdout

    # Dry-run sleep
    p = run_module("domains.sleep.sleep_from_extracted", PID, SNAP, 1)
    assert p.returncode == 0, f"sleep dry-run failed: {p.stderr}\n{p.stdout}"
    assert "[dry-run] sleep seed" in p.stdout

    # Real run cardio
    p = run_module("domains.cardiovascular.cardio_from_extracted", PID, SNAP, 0)
    assert p.returncode == 0, f"cardio real run failed: {p.stderr}\n{p.stdout}"
    out_cardio = REPO_ROOT / "data" / "etl" / PID / SNAP / "features" / "cardio" / "features_daily.csv"
    assert out_cardio.exists(), "cardio features file not created"

    # Real run sleep
    p = run_module("domains.sleep.sleep_from_extracted", PID, SNAP, 0)
    assert p.returncode == 0, f"sleep real run failed: {p.stderr}\n{p.stdout}"
    out_sleep = REPO_ROOT / "data" / "etl" / PID / SNAP / "features" / "sleep" / "features_daily.csv"
    assert out_sleep.exists(), "sleep features file not created"

    # QC files
    qc_cardio = REPO_ROOT / "data" / "etl" / PID / SNAP / "qc" / "cardio_seed_qc.csv"
    qc_sleep = REPO_ROOT / "data" / "etl" / PID / SNAP / "qc" / "sleep_seed_qc.csv"
    assert qc_cardio.exists(), "cardio QC file not created"
    assert qc_sleep.exists(), "sleep QC file not created"