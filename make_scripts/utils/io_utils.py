"""Core IO utilities: atomic writes, checksums, idempotency keys, and locks.

No extra dependencies; Windows-safe. Designed to be imported by other make_scripts/* helpers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA_VERSION = "2025-10-provenance-v1"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def _fsync_file_and_parent(fp_path: Path):
    # fsync file
    try:
        with fp_path.open('rb') as f:
            try:
                os.fsync(f.fileno())
            except Exception:
                # some platforms or filesystems may not support fsync on text mode
                pass
    except Exception:
        # file may not be readable yet; ignore
        pass
    # fsync parent dir if possible
    try:
        dirfd = os.open(str(fp_path.parent), os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except Exception:
        pass


def write_atomic_text(path: Path, text: str, encoding: str = 'utf-8') -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding=encoding, newline='') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # replace atomically
        os.replace(tmp, str(path))
        _fsync_file_and_parent(path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


def write_atomic_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
        _fsync_file_and_parent(path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


def write_atomic_csv(path: Path, rows: Iterable[Iterable[Any]], headers: Optional[Iterable[str]] = None) -> None:
    """Write CSV atomically. rows is an iterable of iterables, headers optional."""
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if headers:
                w.writerow(list(headers))
            for r in rows:
                w.writerow(list(r))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
        _fsync_file_and_parent(path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


def manifest(path_final: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Write <path_final>.manifest.json with metadata. Computes sha256 and row/col if file is CSV.

    meta may include units, source, schema_version override, etc.
    Returns the manifest dict.
    """
    p = Path(path_final)
    out = dict(meta)
    out.setdefault('schema_version', SCHEMA_VERSION)
    # compute sha256
    try:
        out['sha256'] = sha256_of_file(p)
    except Exception:
        out['sha256'] = None
    # try to count rows/cols and min/max timestamps from first column if CSV
    rows = 0
    cols = None
    min_ts = None
    max_ts = None
    try:
        with p.open('r', encoding='utf-8', newline='') as f:
            r = csv.reader(f)
            hdr = next(r, None)
            if hdr is not None:
                cols = len(hdr)
            for row in r:
                rows += 1
                if row:
                    ts = row[0]
                    if ts:
                        if min_ts is None or ts < min_ts:
                            min_ts = ts
                        if max_ts is None or ts > max_ts:
                            max_ts = ts
    except Exception:
        pass
    out.setdefault('rows', rows)
    out.setdefault('cols', cols)
    out.setdefault('date_min', min_ts)
    out.setdefault('date_max', max_ts)
    out.setdefault('generated_at_utc', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))

    # write manifest atomically next to file, but avoid rewriting if nothing changed.
    manifest_path = p.with_name(p.name + '.manifest.json')
    try:
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding='utf-8'))
                # If the existing manifest already refers to the same file sha,
                # consider it unchanged and return it without rewriting. This
                # preserves older idempotency_key metadata but avoids churn of
                # generated_at_utc when the content hasn't changed.
                same_sha = existing.get('sha256') == out.get('sha256')
                if same_sha:
                    return existing
            except Exception:
                # fall through to rewrite if existing manifest is unreadable
                pass
    except Exception:
        # ignore filesystem races; attempt to write below
        pass

    # Write new manifest (sorted keys for deterministic output)
    write_atomic_text(manifest_path, json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return out


def idempotency_key(source_sha256: str, params: Dict[str, Any]) -> str:
    """Create a stable idempotency key from source_sha256 and params by canonical JSON+sha256."""
    payload = {'source_sha256': source_sha256, 'params': params}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


@contextmanager
def with_lock(lock_path: Path):
    """Context manager that creates an exclusive lock file. Removes it on exit.

    Honors LOCK_TIMEOUT_SEC environment variable (default 300).
    """
    lock_path = Path(lock_path)
    ensure_dir(lock_path.parent)
    timeout = float(os.environ.get('LOCK_TIMEOUT_SEC', '300'))
    start = time.time()
    fd = None
    while True:
        try:
            # O_CREAT|O_EXCL ensures atomic create fail if exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            # write pid and timestamp
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'pid': os.getpid(), 'ts': datetime.now(timezone.utc).isoformat()}))
            fd = None
            break
        except FileExistsError:
            if (time.time() - start) >= timeout:
                raise TimeoutError(f'Could not acquire lock {lock_path} within {timeout}s')
            time.sleep(0.1)
        except Exception:
            # unknown error acquiring lock; re-raise
            raise
    try:
        yield
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
