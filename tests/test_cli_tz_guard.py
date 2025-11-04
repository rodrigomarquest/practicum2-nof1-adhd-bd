import sys
import os
import importlib
import pytest


def _import_main():
    mod = importlib.import_module("src.cli.etl_runner")
    return getattr(mod, "main")


def test_no_flags_no_exit(monkeypatch):
    """Default invocation with no TZ/CUTOVER flags should not SystemExit."""
    monkeypatch.setattr(sys, "argv", ["etl_runner"])
    main = _import_main()
    # calling main() directly should not raise SystemExit
    rc = main(argv=[])
    assert isinstance(rc, int)


def test_cutover_warn_no_exit(monkeypatch, capsys):
    """Passing --cutover should print a warning but not exit by default."""
    monkeypatch.setattr(sys, "argv", ["etl_runner", "--cutover=2024-01-01"])
    monkeypatch.delenv("ETL_STRICT_TZ", raising=False)
    monkeypatch.delenv("CI", raising=False)
    main = _import_main()
    rc = main(argv=[])
    out = capsys.readouterr().out
    assert "[warn] Deprecated TZ/CUTOVER argument detected:" in out
    # parser prints help and returns 2 when no subcommand provided
    assert rc == 2


def test_tz_before_strict_env_exits(monkeypatch, capsys):
    """When ETL_STRICT_TZ=1 is set, deprecated TZ flags cause exit(2)."""
    monkeypatch.setattr(sys, "argv", ["etl_runner", "--tz-before=America/Sao_Paulo"])
    monkeypatch.setenv("ETL_STRICT_TZ", "1")
    main = _import_main()
    with pytest.raises(SystemExit) as ei:
        main(argv=[])
    assert ei.value.code == 2
    out = capsys.readouterr().out
    assert "[warn] Deprecated TZ/CUTOVER argument detected:" in out
