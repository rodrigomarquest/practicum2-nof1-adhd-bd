#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts/release/version_guard.py

The Makefile historically invoked `make_scripts/version_guard.py`. During
refactoring the implementation was moved to `make_scripts/release/`. This
shim preserves backward compatibility by executing the moved script.
"""
from pathlib import Path
import runpy
import sys
import os


def main():
    # Allow callers to bypass checks in CI/dry-run by setting SKIP_VERSION_GUARD=1
    if os.environ.get('SKIP_VERSION_GUARD') in ('1', 'true', 'True'):
        print('SKIP_VERSION_GUARD set: skipping version guard checks')
        return 0
    here = Path(__file__).resolve().parent
    target = here / "release" / "version_guard.py"
    if not target.exists():
        print(f"Missing compatibility target: {target}", file=sys.stderr)
        return 2
    try:
        # execute the target as a script
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit:
        # allow the delegated script to exit with its own status
        raise
    except Exception as exc:  # pragma: no cover - best-effort shim
        print(f"Error running {target}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""Simple guard to validate release/tag state before releasing.

Checks:
- Working tree is clean (unless --allow-dirty)
- The requested tag does not already exist

Exits non-zero on failures with helpful messages.
"""

import argparse
import json
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def check_clean(allow_dirty: bool) -> bool:
    if allow_dirty:
        print("--allow-dirty: skipping clean working tree check")
        return True
    # Use git status --porcelain to detect uncommitted changes
    r = run(["git", "status", "--porcelain"])
    if r.returncode != 0:
        print("ERROR: git status failed:", r.stderr.strip(), file=sys.stderr)
        return False
    if r.stdout.strip() == "":
        print("Working tree clean.")
        return True
    else:
        print("ERROR: Working tree has uncommitted changes. Commit or stash them, or pass --allow-dirty.")
        print(r.stdout)
        return False


def tag_exists(tag: str) -> bool:
    # git rev-parse --verify --quiet refs/tags/<tag>
    r = run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"])
    # rev-parse returns 0 if exists, non-zero otherwise
    exists = r.returncode == 0
    if exists:
        print(f"ERROR: Tag '{tag}' already exists.")
    else:
        print(f"Tag '{tag}' not found (OK).")
    return exists


def tag_exists_remote(tag: str, remote: str = 'origin') -> bool:
    # Use git ls-remote --tags <remote> <tag>
    try:
        r = run(["git", "ls-remote", "--tags", remote, tag])
        if r.returncode != 0:
            # ls-remote failed (remote may not exist or network issue)
            print(f"Warning: git ls-remote failed for remote '{remote}': {r.stderr.strip()}")
            return False
        out = r.stdout.strip()
        exists = bool(out)
        if exists:
            print(f"ERROR: Tag '{tag}' already exists on remote '{remote}'.")
        else:
            print(f"Tag '{tag}' not found on remote '{remote}' (OK).")
        return exists
    except Exception as e:
        print(f"Warning: exception while checking remote tags: {e}")
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description="Guard release/tag creation: ensure clean tree and tag non-existence.")
    p.add_argument("--tag", required=True, help="Tag to verify (e.g. v0.1.0)")
    p.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted changes (skip clean check)")
    p.add_argument("--remote-check", dest="remote_check", action="store_true", help="Check remote for tag existence (default on)")
    p.add_argument("--no-remote-check", dest="remote_check", action="store_false", help="Do not check remote for tag existence")
    p.set_defaults(remote_check=True)
    p.add_argument("--remote", default="origin", help="Remote name to check (default: origin)")
    p.add_argument("--json", action="store_true", help="Emit a small JSON status block to stdout")
    args = p.parse_args(argv)

    status = {
        'dirty': False,
        'tag_exists_local': False,
        'tag_exists_remote': False,
    }
    status['dirty'] = not check_clean(args.allow_dirty)
    status['tag_exists_local'] = tag_exists(args.tag)
    status['tag_exists_remote'] = False
    if args.remote_check:
        status['tag_exists_remote'] = tag_exists_remote(args.tag, args.remote)

    ok = (not status['dirty']) and (not status['tag_exists_local']) and (not status['tag_exists_remote'])

    if args.json:
        print(json.dumps(status))

    if not ok:
        print("version_guard: checks failed", file=sys.stderr)
        sys.exit(2)
    print("version_guard: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
