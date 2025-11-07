import os
import json
import logging
from pathlib import Path
import pytest

# ensure project src on PYTHONPATH during tests via pytest invocation
from domains.cda import parse_cda

CDA_GOOD = '''<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <component>
    <section>
      <entry>
        <observation>
          <code code="X1" displayName="Test1"/>
        </observation>
      </entry>
      <entry>
        <observation>
          <code code="X2" displayName="Test2"/>
        </observation>
      </entry>
    </section>
  </component>
</ClinicalDocument>
'''

CDA_BAD = "<ClinicalDocument><broken></ClinicalDocument>"


@pytest.fixture(autouse=True)
def disable_tz_guard(monkeypatch):
    # disable timezone guard in tests
    monkeypatch.setenv("ETL_TZ_GUARD", "0")
    yield


def run_probe_and_write(snapshot_root: Path, cda_path: Path, home_tz: str = "UTC"):
    logger = logging.getLogger("etl.extract")
    try:
        summary = parse_cda.cda_probe(cda_path, home_tz=home_tz)
        n_obs = int(summary.get("n_observation", 0) or 0)
        enriched = dict(summary or {})
        enriched["snapshot_dir"] = str(snapshot_root)
        enriched["source_path"] = str(cda_path.resolve())
        enriched["home_tz"] = home_tz or "unknown"
        enriched["n_observation"] = n_obs
        enriched["n_section"] = int(enriched.get("n_section", 0) or 0)
        enriched["codes"] = enriched.get("codes", {}) or {}
        if n_obs > 0:
            enriched["status"] = "ok"
            logger.info("CDA parsed: %s (obs=%d) [snapshot=%s]", str(cda_path.resolve()), n_obs, str(snapshot_root))
        else:
            enriched["status"] = "empty"
            logger.warning("CDA parsed but empty: %s [snapshot=%s]", str(cda_path.resolve()), str(snapshot_root))
        parse_cda.write_cda_qc(snapshot_root, enriched)
    except Exception as e:
        logger.warning("CDA parse failed: %s (%s: %s) [snapshot=%s]", str(cda_path.resolve()), type(e).__name__, str(e), str(snapshot_root))
        enriched = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "n_section": 0,
            "n_observation": 0,
            "codes": {},
            "snapshot_dir": str(snapshot_root),
            "source_path": str(cda_path.resolve()),
            "home_tz": home_tz or "unknown",
        }
        parse_cda.write_cda_qc(snapshot_root, enriched)


def test_cda_pipeline_ok(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    # create fake snapshot
    snapshot_root = tmp_path / "data" / "etl" / "P000001" / "2099-01-02"
    cda_dir = snapshot_root / "extracted" / "apple" / "inapp" / "apple_health_export"
    cda_dir.mkdir(parents=True, exist_ok=True)
    cda_path = cda_dir / "export_cda.xml"
    cda_path.write_text(CDA_GOOD, encoding="utf-8")

    run_probe_and_write(snapshot_root, cda_path, home_tz="UTC")

    # logs contain CDA parsed
    assert any("CDA parsed" in r.message for r in caplog.records)

    # qc files exist
    jsonp = snapshot_root / "qc" / "cda_summary.json"
    csvp = snapshot_root / "qc" / "cda_summary.csv"
    assert jsonp.exists()
    assert csvp.exists()

    j = json.loads(jsonp.read_text(encoding="utf-8"))
    assert j.get("status") == "ok"
    assert j.get("n_observation", 0) >= 1
    assert j.get("snapshot_dir") == str(snapshot_root)
    assert j.get("source_path") == str(cda_path.resolve())
    # ts_utc should be present and end with Z (or +00:00 tolerated)
    assert "ts_utc" in j and isinstance(j.get("ts_utc"), str) and j.get("ts_utc")
    assert j.get("ts_utc").endswith("Z") or j.get("ts_utc").endswith("+00:00")


def test_cda_pipeline_malformed(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    snapshot_root = tmp_path / "data" / "etl" / "P000001" / "2099-01-02"
    cda_dir = snapshot_root / "extracted" / "apple" / "inapp" / "apple_health_export"
    cda_dir.mkdir(parents=True, exist_ok=True)
    cda_path = cda_dir / "export_cda.xml"
    cda_path.write_text(CDA_BAD, encoding="utf-8")

    run_probe_and_write(snapshot_root, cda_path, home_tz="UTC")

    # logs should include parse failed warning or empty warning
    assert any("CDA parse failed" in r.message or "CDA parsed but empty" in r.message for r in caplog.records)

    jsonp = snapshot_root / "qc" / "cda_summary.json"
    assert jsonp.exists()
    j = json.loads(jsonp.read_text(encoding="utf-8"))
    assert j.get("status") in ("error", "empty")
    assert j.get("snapshot_dir") == str(snapshot_root)
    assert j.get("source_path") == str(cda_path.resolve())
    # ts_utc present and formatted
    assert "ts_utc" in j and isinstance(j.get("ts_utc"), str) and j.get("ts_utc")
    assert j.get("ts_utc").endswith("Z") or j.get("ts_utc").endswith("+00:00")


def test_cda_pipeline_recovered(tmp_path, caplog):
    # CDA with trailing junk after the closing ClinicalDocument should exercise recovery
    caplog.set_level(logging.WARNING)
    CDA_TRAILING = CDA_GOOD + "\nTHIS_IS_TRAILING_JUNK\n<notxml>bad</notxml>"

    snapshot_root = tmp_path / "data" / "etl" / "P000001" / "2099-01-03"
    cda_dir = snapshot_root / "extracted" / "apple" / "inapp" / "apple_health_export"
    cda_dir.mkdir(parents=True, exist_ok=True)
    cda_path = cda_dir / "export_cda.xml"
    cda_path.write_text(CDA_TRAILING, encoding="utf-8")

    run_probe_and_write(snapshot_root, cda_path, home_tz="UTC")

    jsonp = snapshot_root / "qc" / "cda_summary.json"
    assert jsonp.exists()
    j = json.loads(jsonp.read_text(encoding="utf-8"))

    # status remains one of the canonical values
    assert j.get("status") in ("ok", "empty", "error")
    assert j.get("snapshot_dir") == str(snapshot_root)
    assert j.get("source_path") == str(cda_path.resolve())

    # recover metrics
    assert "recover" in j and isinstance(j.get("recover"), dict)
    rec = j.get("recover")
    assert rec.get("used") is True
    assert rec.get("method") in ("lxml", "salvage_first_root")

    # ts_utc present and formatted
    assert "ts_utc" in j and isinstance(j.get("ts_utc"), str) and j.get("ts_utc")
    assert j.get("ts_utc").endswith("Z") or j.get("ts_utc").endswith("+00:00")

    # CSV should include flat recovery keys
    csvp = snapshot_root / "qc" / "cda_summary.csv"
    assert csvp.exists()
    txt = csvp.read_text(encoding="utf-8")
    assert "recover_used" in txt
    assert "recover_method" in txt
    