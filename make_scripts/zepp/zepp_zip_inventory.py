#!/usr/bin/env python3
"""Inventory a Zepp export zip file (metadata only)."""
import argparse
import hashlib
import json
import os
import sys
import time
from zipfile import ZipFile, is_zipfile, BadZipFile
try:
    import pyzipper  # optional, used to read AES-encrypted zip members
except Exception:
    pyzipper = None

# Use env var ZEPP_ZIP_PASSWORD to indicate user provided password (metadata-only script)
ZEPP_ZIP_PASSWORD = os.environ.get('ZEPP_ZIP_PASSWORD')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, text, encoding='utf-8'):
    tmp = path + '.tmp'
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(tmp, 'w', encoding=encoding) as fh:
        fh.write(text)
    os.replace(tmp, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--zip-path', required=False, default='', help='Path to zip file; if omitted script will search default data/raw/<participant>/zepp for latest')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--password', required=False, help='Password for encrypted zip (optional). If not provided and needed, script will prompt')
    p.add_argument('--participant', required=False, help='Participant id to locate default raw dir (default P000001)')
    return p.parse_args(argv)


def select_zip_path(zip_path, participant):
    if zip_path:
        return zip_path
    default_dir = os.path.join('data', 'raw', participant, 'zepp')
    if not os.path.isdir(default_dir):
        print('Default Zepp dir not found:', default_dir, file=sys.stderr)
        sys.exit(2)
    cand = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if os.path.isfile(os.path.join(default_dir, f))]
    if not cand:
        print('No files found in default Zepp dir:', default_dir, file=sys.stderr)
        sys.exit(2)
    chosen = max(cand, key=lambda p: os.path.getmtime(p))
    print('Auto-selected latest Zepp file:', chosen)
    return chosen


def open_ziplist(zip_path):
    if not is_zipfile(zip_path):
        print('Not a zip file:', zip_path, file=sys.stderr)
        sys.exit(2)
    try:
        z = ZipFile(zip_path, 'r')
        namelist = z.infolist()
    except BadZipFile:
        print('Bad zip file:', zip_path, file=sys.stderr)
        sys.exit(2)
    except RuntimeError:
        print('Archive unreadable (possibly encrypted). If encrypted set ZEPP_ZIP_PASSWORD and retry.', file=sys.stderr)
        sys.exit(3)
    return z, namelist


def validate_password_if_needed(z, namelist, zip_path, provided_password):
    import getpass
    encrypted_members = [info for info in namelist if info.flag_bits & 0x1]
    if not encrypted_members:
        return provided_password, False, None

    # ensure a password value (prompt if needed)
    pwd = provided_password
    if not pwd:
        try:
            pwd = getpass.getpass('Archive contains encrypted entries; enter ZEPP_ZIP_PASSWORD: ')
        except Exception:
            pwd = None
    if not pwd:
        print('Archive contains encrypted entries; password not provided. Set ZEPP_ZIP_PASSWORD or pass --password.', file=sys.stderr)
        sys.exit(3)

    # validate by reading first encrypted member
    first = encrypted_members[0]
    zpwd = pwd.encode('utf-8') if isinstance(pwd, str) else pwd
    try:
        with z.open(first.filename, pwd=zpwd) as fh:
            _ = fh.read(1)
        return pwd, True, 'zipfile'
    except RuntimeError:
        if pyzipper is None:
            print('Warning: could not validate password with stdlib zipfile and pyzipper is not installed; AES validation skipped.\n'
                  'Install pyzipper (pip install pyzipper) to enable AES-encrypted archive validation.', file=sys.stderr)
            return pwd, None, 'skipped_pyzipper_missing'
        try:
            with pyzipper.AESZipFile(zip_path, 'r') as az:
                az.pwd = zpwd
                _ = az.read(first.filename, pwd=zpwd)
            return pwd, True, 'pyzipper'
        except RuntimeError:
            print('Provided password invalid for encrypted AES entries.', file=sys.stderr)
            sys.exit(4)
        except Exception as e:
            print('Error validating AES-encrypted archive with pyzipper:', str(e), file=sys.stderr)
            sys.exit(4)


def build_manifest(zip_path, size, sha, namelist, provided_password, password_validated, validation_method):
    files = [{'path': info.filename, 'size': info.file_size} for info in namelist]
    candidates = [f['path'] for f in files if any(x in f['path'].lower() for x in ['heart_rate', 'hrv', 'sleep', 'emoti', 'temperature', 'rate'])]
    mtime = os.path.getmtime(zip_path)
    timestamp_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mtime))
    manifest = {
        'archive_name': os.path.basename(zip_path),
        'size_bytes': size,
        'sha256': sha,
        'encrypted': bool([info for info in namelist if info.flag_bits & 0x1]),
        'password_provided': bool(provided_password),
        'password_validated': password_validated,
        'password_validation_method': validation_method,
        'n_files': len(files),
        'files': files,
        'candidate_paths': sorted(candidates),
        'timestamp_utc': timestamp_utc
    }
    return manifest


def write_outputs(out_dir, manifest, namelist):
    inventory_json = os.path.join(out_dir, 'zepp_zip_inventory.json')
    filelist_tsv = os.path.join(out_dir, 'zepp_zip_filelist.tsv')
    atomic_write(inventory_json, json.dumps(manifest, sort_keys=True, indent=2))
    lines = ['path\tsize']
    files = [{'path': info.filename, 'size': info.file_size} for info in namelist]
    for f in sorted(files, key=lambda x: x['path']):
        lines.append(f"{f['path']}\t{f['size']}")
    atomic_write(filelist_tsv, '\n'.join(lines) + '\n')
    print('Wrote inventory:', inventory_json)
    print('Wrote filelist:', filelist_tsv)


def main(argv=None):
    args = parse_args(argv)
    zip_path = select_zip_path(args.zip_path or '', args.participant or os.environ.get('PARTICIPANT', 'P000001'))
    out_dir = args.out_dir
    if not os.path.exists(zip_path):
        print('ZIP not found:', zip_path, file=sys.stderr)
        sys.exit(2)
    size = os.path.getsize(zip_path)
    sha = sha256_file(zip_path)
    z, namelist = open_ziplist(zip_path)
    provided_pwd = args.password or ZEPP_ZIP_PASSWORD
    provided_pwd, pwd_validated, validation_method = validate_password_if_needed(z, namelist, zip_path, provided_pwd)
    manifest = build_manifest(zip_path, size, sha, namelist, provided_pwd, pwd_validated, validation_method)
    write_outputs(out_dir, manifest, namelist)


if __name__ == '__main__':
    main()
