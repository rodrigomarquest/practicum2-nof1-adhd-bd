import shutil

import pyzipper


def create_aes_zip(path: str, password: bytes):
    with pyzipper.AESZipFile(path, 'w', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(password)
        zf.writestr('SLEEP/SLEEP_1.csv', 'ts,value\n2025-01-01T00:00:00Z,1')


def test_etl_extract_zepp_encrypted(tmp_path, monkeypatch):
    # prepare AES-encrypted zepp zip in tmp
    zip_path = tmp_path / 'zepp_enc.zip'
    password = b'secretpw'
    create_aes_zip(str(zip_path), password)

    # setup temporary repo-like raw and etl roots inside tmp_path
    participant = 'P900000'
    raw_root = tmp_path / 'data' / 'raw'
    etl_root = tmp_path / 'data' / 'etl'
    dest_dir = raw_root / participant / 'zepp'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / zip_path.name
    shutil.copy(str(zip_path), str(dest_zip))

    # monkeypatch module-level paths so the ETL writes to tmp_path instead of repo
    import importlib
    import etl_pipeline
    importlib.reload(etl_pipeline)
    monkeypatch.setattr(etl_pipeline, 'RAW_ARCHIVE', raw_root)
    monkeypatch.setattr(etl_pipeline, 'ETL_ROOT', etl_root)

    # run extract in-process via main() to avoid spawning a subprocess
    argv = [
        'etl_pipeline.py', 'extract',
        '--participant', participant,
        '--snapshot', '2025-01-01',
        '--cutover', '2024-01-01',
        '--tz_before', 'UTC',
        '--tz_after', 'UTC',
        '--auto-zip',
        '--zepp-password', password.decode(),
    ]
    monkeypatch.setenv('NON_INTERACTIVE', '1')
    monkeypatch.setattr('sys.argv', argv)

    # call main()
    res = etl_pipeline.main()
    assert res == 0

    # assert extraction target exists and sha file present inside tmp_path
    target = etl_root / participant / 'snapshots' / '2025-01-01' / 'extracted' / 'zepp'
    assert target.exists(), f"Zepp extracted dir not found: {target}"
    sha = target / 'zepp_zip.sha256'
    assert sha.exists(), 'sha256 file not written'
