import json
import os
import subprocess
import tempfile

import pyzipper


def create_aes_zip(path, password: bytes):
    # create a small AES-encrypted zip with one CSV file
    with pyzipper.AESZipFile(path, 'w', compression=pyzipper.ZIP_DEFLATED,
                              encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(password)
        zf.writestr('SLEEP/SLEEP_1.csv', 'ts,value\n2025-10-22T00:00:00Z,1')


def run_inventory(zip_path, out_dir, password: str):
    cmd = [
        os.path.abspath('.venv/Scripts/python'),
        os.path.abspath('make_scripts/zepp/zepp_zip_inventory.py'),
        '--zip-path', zip_path,
        '--out-dir', out_dir,
        '--participant', 'P900000'
    ]
    # pass password via env to simulate ZEPP_ZIP_PASSWORD
    env = os.environ.copy()
    env['ZEPP_ZIP_PASSWORD'] = password
    subprocess.check_call(cmd, env=env)


def test_aes_validation(tmp_path):
    zip_path = tmp_path / 'test_aes.zip'
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    password = b'secret123'

    create_aes_zip(str(zip_path), password)
    # run inventory; should validate using pyzipper and write manifest
    run_inventory(str(zip_path), str(out_dir), password.decode())

    manifest_path = out_dir / 'zepp_zip_inventory.json'
    assert manifest_path.exists(), 'manifest not written'
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        manifest = json.load(fh)

    assert manifest.get('encrypted') is True
    assert manifest.get('password_provided') is True
    assert manifest.get('password_validated') is True
    assert manifest.get('password_validation_method') == 'pyzipper'
