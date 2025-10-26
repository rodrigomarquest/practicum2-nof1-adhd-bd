#!/usr/bin/env python3
"""Guard release/tag creation: ensure clean tree and tag non-existence.

This is a copy of make_scripts/version_guard.py moved into make_scripts/release for release-related tools.
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
    r = run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"])
    exists = r.returncode == 0
    if exists:
        print(f"ERROR: Tag '{tag}' already exists.")
    else:
        print(f"Tag '{tag}' not found (OK).")
    return exists


def tag_exists_remote(tag: str, remote: str = 'origin') -> bool:
    try:
        r = run(["git", "ls-remote", "--tags", remote, tag])
        if r.returncode != 0:
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
