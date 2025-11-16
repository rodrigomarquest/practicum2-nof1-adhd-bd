import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cli.etl_runner as etl_runner


def run_main_args(args):
    # helper to call main and raise SystemExit with its return code
    with pytest.raises(SystemExit) as se:
        raise SystemExit(etl_runner.main(args))
    return se.value.code


def test_extract_no_sources(monkeypatch, capsys):
    # Scenario A: no sources discovered -> exit code 2, no discovered print
    monkeypatch.setenv("ETL_TZ_GUARD", "0")

    # stub discover_sources to return empty
    import src.etl_pipeline as etl_mod

    monkeypatch.setattr(etl_mod, "discover_sources", lambda pid, auto_zip=True: [])

    code = run_main_args(["extract", "--pid", "P000001", "--snapshot", "2025-11-01", "--dry-run", "1"])
    assert int(code) == 2
    out, err = capsys.readouterr()
    # ensure CLI-level discovered header not printed (extract_run may still
    # print a DRY RUN line)
    assert "info: discovered" not in out.lower()


def test_extract_with_sources_and_no_zepp_password(monkeypatch, capsys):
    # Scenario B: apple + zepp discovered, ZEPP_ZIP_PASSWORD missing -> exit 0
    monkeypatch.setenv("ETL_TZ_GUARD", "0")
    # ensure no password in env
    monkeypatch.delenv("ZEPP_ZIP_PASSWORD", raising=False)

    import src.etl_pipeline as etl_mod

    def fake_discover(pid, auto_zip=True):
        return [
            {"vendor": "apple", "variant": "inapp", "zip_path": Path("data/raw/P000001/apple/export/apple.zip")},
            {"vendor": "zepp", "variant": "cloud", "zip_path": Path("data/raw/P000001/zepp/zepp.zip")},
        ]

    monkeypatch.setattr(etl_mod, "discover_sources", fake_discover)
    # stub extract_run to short-circuit
    monkeypatch.setattr(etl_mod, "extract_run", lambda pid, snapshot_arg, auto_zip=True, dry_run=False: 0)

    code = run_main_args(["extract", "--pid", "P000001", "--snapshot", "2025-11-01", "--dry-run", "1"])
    assert int(code) == 0
    out, err = capsys.readouterr()
    # discovered header should appear and mention both vendors/variants
    assert "discovered" in out.lower()
    assert "apple/inapp" in out.lower()
    assert "zepp/cloud" in out.lower()
    # WARNING about missing ZEPP_ZIP_PASSWORD should have been logged (likely stderr)
    combined = (out + err).lower()
    assert "zepp_zip_password not set" in combined
