#!/usr/bin/env python3
"""Create a virtualenv and optionally install requirements.

This encapsulates the long bash one-liner from the Makefile so Make stays thin.

Usage examples:
  python make_scripts/venv_create.py --python python                # create .venv
  python make_scripts/venv_create.py --python python --dev --no-install
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def run(cmd, check=True, shell=False):
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=shell)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--python", default="python", help="Python executable to use for venv creation")
    p.add_argument("--dev", action="store_true", help="Install dev requirements")
    p.add_argument("--kaggle", action="store_true", help="Use Kaggle constraints")
    p.add_argument("--win", action="store_true", help="Add Windows constraints")
    p.add_argument("--no-install", dest="no_install", action="store_true", help="Create venv but skip pip install")
    args = p.parse_args(argv)

    venv_dir = Path('.venv')
    py = args.python

    print(f"Creating virtual environment {venv_dir} using {py}")
    # Create venv
    run([py, "-m", "venv", str(venv_dir)])

    # Determine venv python path (Windows vs POSIX)
    if (venv_dir / 'Scripts' / 'python.exe').exists():
        venv_py = venv_dir / 'Scripts' / 'python.exe'
    elif (venv_dir / 'Scripts' / 'python').exists():
        venv_py = venv_dir / 'Scripts' / 'python'
    else:
        venv_py = venv_dir / 'bin' / 'python'

    print('Using', venv_py)

    if args.no_install:
        print('Skipping pip install (--no-install).')
        return

    # Build pip install commands
    # Upgrade pip/setuptools/wheel first
    run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    install_req = "-r requirements.txt"
    if args.dev:
        install_req = "-r requirements-dev.txt"

    constraints = []
    if args.kaggle:
        constraints.extend(["-c", "constraints-kaggle.txt"])
    if args.win:
        constraints.extend(["-c", "constraints-windows.txt"])

    cmd = [str(venv_py), "-m", "pip", "install"] + install_req.split() + constraints
    run(cmd)


if __name__ == '__main__':
    main()
