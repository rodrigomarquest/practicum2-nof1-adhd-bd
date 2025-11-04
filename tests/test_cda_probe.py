import tempfile
from pathlib import Path
from domains.cda import parse_cda

CDA_SAMPLE = '''<?xml version="1.0" encoding="UTF-8"?>
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


def test_cda_probe_basic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "export_cda.xml"
        p.write_text(CDA_SAMPLE, encoding="utf-8")
        summary = parse_cda.cda_probe(p)
        assert summary.get("n_section", 0) >= 1
        assert summary.get("n_observation", 0) >= 1
        assert isinstance(summary.get("codes", {}), dict)
