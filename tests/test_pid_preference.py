import zipfile
import importlib
from pathlib import Path


def make_zip(path: Path, name: str, contents: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as zf:
        for fname, data in contents.items():
            zf.writestr(fname, data)


def test_pipeline_prefers_provided_participant(tmp_path, monkeypatch, capsys):
    # Setup two raw folders: provided (P00001) and canonical (P000001)
    provided = 'P00001'
    canonical = 'P000001'

    raw_root = tmp_path / 'data' / 'raw'
    etl_root = tmp_path / 'data' / 'etl'

    # create a valid zepp zip only under the provided participant
    prov_zip = raw_root / provided / 'zepp' / 'zepp_prov.zip'
    make_zip(prov_zip, 'zepp_prov.zip', {'hr.csv': 'ts,value\n2025-01-01T00:00:00Z,1'})

    # also create a (different) zip under canonical to ensure both exist
    canon_zip = raw_root / canonical / 'zepp' / 'zepp_canon.zip'
    make_zip(canon_zip, 'zepp_canon.zip', {'hr.csv': 'ts,value\n2025-01-02T00:00:00Z,2'})

    # reload module and monkeypatch paths
    import etl_pipeline
    importlib.reload(etl_pipeline)
    monkeypatch.setattr(etl_pipeline, 'RAW_ARCHIVE', raw_root)
    monkeypatch.setattr(etl_pipeline, 'ETL_ROOT', etl_root)

    # prepare argv to call extract for the provided pid
    monkeypatch.setenv('NON_INTERACTIVE', '1')
    monkeypatch.setattr('sys.argv', ['etl_pipeline.py', 'extract',
                                     '--participant', provided,
                                     '--snapshot', '2025-01-01',
                                     '--cutover', '2024-01-01',
                                     '--tz_before', 'UTC',
                                     '--tz_after', 'UTC',
                                     '--auto-zip'])

    # run
    res = etl_pipeline.main()
    assert res == 0

    # capture stdout and ensure the INFO line indicates the provided participant
    captured = capsys.readouterr()
    assert f"INFO: using participant '{provided}'" in captured.out

    # ensure ETL wrote to the provided participant folder, not canonical
    prov_target = etl_root / provided / 'snapshots' / '2025-01-01' / 'extracted' / 'zepp'
    canon_target = etl_root / canonical / 'snapshots' / '2025-01-01' / 'extracted' / 'zepp'
    assert prov_target.exists(), f"expected extraction under provided PID: {prov_target}"
    assert not canon_target.exists(), f"did not expect extraction under canonical PID: {canon_target}"
