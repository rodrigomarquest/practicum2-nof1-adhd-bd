#!/usr/bin/env python3
"""Test helper that verifies `make etl-one` exits 0 when export.xml is missing.

Behavior:
- Create a temporary data_etl/<PID>/snapshots/<SNAP> without export.xml
- Move any existing data_etl out of the way
- Run `make etl-one` with DRY_RUN=1 and appropriate PID/SNAP
- Restore original data_etl and cleanup
- Exit with non-zero if `make etl-one` returned non-zero
"""
import os
import shutil
import subprocess
import sys
import tempfile

PID = os.environ.get("TEST_PID", "PTEST123")
SNAP = os.environ.get("TEST_SNAP", "2025-01-01")

cwd = os.getcwd()
backup = None
tmpdir = None

def main():
    tmpdir = None
    backup = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="etl_test_")
        # create minimal layout
        os.makedirs(os.path.join(tmpdir, PID, "snapshots", SNAP), exist_ok=True)

        if os.path.exists("data_etl"):
            backup = os.path.abspath("data_etl.bak_for_test")
            # ensure unique backup name
            i = 0
            while os.path.exists(backup):
                i += 1
                backup = f"data_etl.bak_for_test.{i}"
            shutil.move("data_etl", backup)

        shutil.move(tmpdir, "data_etl")

        # run make etl-one using DRY_RUN and explicit PID/SNAP
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "PID": PID, "SNAP": SNAP})

        print("Running: make etl-one (dry-run) with temporary data_etl")
        res = subprocess.run(["make", "-s", "etl-one", f"PID={PID}", f"SNAP={SNAP}"], cwd=cwd, env=env)

        if res.returncode != 0:
            print(f"test-etl-one-skip: FAILED (make returned {res.returncode})")
            sys.exit(res.returncode)

        print("test-etl-one-skip: OK (etl-one skipped as expected)")
        sys.exit(0)

    finally:
        # Restore original data_etl
        try:
            if os.path.exists("data_etl"):
                shutil.rmtree("data_etl")
        except Exception:
            pass
        if backup and os.path.exists(backup):
            try:
                shutil.move(backup, "data_etl")
            except Exception:
                print("Warning: failed to restore original data_etl", file=sys.stderr)

        # cleanup tmpdir if still present
        try:
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
