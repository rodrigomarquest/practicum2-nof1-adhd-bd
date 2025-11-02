"""CLI entrypoint package for executable scripts moved from /scripts.

This package contains thin wrappers that call into `src.*` modules and
are intended to be run as `python -m cli.etl_runner` so Makefile and other
invokers don't rely on a scripts/ directory.
"""

__all__ = ["etl_runner", "run_etl_with_timer"]
