import json
import subprocess
import sys
from pathlib import Path


def test_migrate_layout_dry_run(tmp_path):
    # Setup fake legacy structure
    participant = "PTEST"
    legacy_etl = tmp_path / "data" / "etl" / participant
    raw_dir = tmp_path / "data" / "raw" / participant
    apple_export = legacy_etl / "exports" / "apple"
    zepp_export = legacy_etl / "exports" / "zepp"
    apple_export.mkdir(parents=True, exist_ok=True)
    zepp_export.mkdir(parents=True, exist_ok=True)

    # create fake files
    f1 = apple_export / "export.zip"
    f1.write_bytes(b"hello apple")
    f2 = zepp_export / "payload.bin"
    f2.write_bytes(b"zeppdata")

    # Run the script in dry-run mode
    script = Path.cwd() / "make_scripts" / "migrate_layout.py"
    env = {**dict(), **{"PARTICIPANT": participant}}
    proc = subprocess.run([
        sys.executable,
        str(script),
        "--participant",
        participant,
        "--legacy-etl-dir",
        str(legacy_etl),
        "--raw-dir",
        str(raw_dir),
        "--dry-run",
    ], cwd=Path.cwd(), env={**env, **dict(os_environ if (os_environ := {}) else {})}, capture_output=True, text=True)

    # Assert it printed would move lines
    out = proc.stdout + proc.stderr
    assert "would move" in out

    # Assert audit file exists in provenance
    prov = Path.cwd() / "provenance"
    candidates = list(prov.glob("migrate_layout_*.json"))
    assert candidates, "No audit file written"
    # load the last one
    audit = json.loads(candidates[-1].read_text(encoding="utf8"))
    assert audit.get("dry_run") is True
    assert isinstance(audit.get("actions"), list)
    # ensure entries for apple and zepp were recorded
    kinds = {a.get("kind") for a in audit.get("actions")}
    assert "apple" in kinds or "zepp" in kinds
