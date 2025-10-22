# Makefile calling pattern for make_scripts/

This project follows a strict pattern for Make targets that invoke Python helpers under `make_scripts/`.

## Principles

- Keep Make targets thin: they validate inputs, export inline environment variables, echo the full command, then call a Python helper.
- Avoid `eval` or constructing large shell snippets inside Make recipes. Complex logic belongs in `make_scripts/*.py`.
- Provide per-target `*_ARGS` variables to allow flag-style invocation without exporting environment variables.
- Echo the full command before running it for reproducibility.

## Examples

ENV-only style (preferred when setting multiple env vars):

DRY_RUN=1 NON_INTERACTIVE=1 make clean-data

Flags passthrough style (use per-target \*\_ARGS):

make clean-data CLEAN_ARGS='--dry-run --non-interactive'

## Venv helper

Long bash one-liners for venv creation were moved to `make_scripts/venv_create.py`.

Usage from Make:

make venv
make venv DEV=1
make venv NO_INSTALL=1 # creates .venv but skips pip install

Direct invocation (for debugging):

python make_scripts/venv_create.py --python python --dev

## Notes

- When updating Makefile targets, follow the same pattern used by `clean-data`, `env-versions`, and `weekly-report`.
- Use `make help` to see examples for venv and data targets.
