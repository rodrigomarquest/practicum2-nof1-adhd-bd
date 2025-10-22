#!/usr/bin/env bash
# Activate .venv and ensure the interactive prompt shows the env name in Git Bash
# Idempotent: safe to source multiple times
set -euo pipefail
#!/usr/bin/env bash
# make_scripts/activate_venv.sh
#
# Convenience wrapper to activate the repository .venv and launch an interactive
# shell (Git Bash) with the virtualenv visible in the prompt. Designed to be
# sourced or executed from Git Bash/MSYS. It is idempotent and safe to re-run.
#
# Behavior:
# - locates the usual venv activation scripts (.venv/bin/activate or
#   .venv/Scripts/activate)
# - ensures VIRTUAL_ENV_DISABLE_PROMPT is unset so activation may affect PS1
# - preserves existing PROMPT_COMMAND if present; only sets a safe no-op when
#   PROMPT_COMMAND is empty
# - prepends a venv prefix to PS1 when not already present
# - by default execs an interactive shell (replaces the caller). If
#   ACTIVATE_NOEXEC=1 is set the script will not spawn an interactive shell and
#   instead prints status (useful for tests and CI).

set -euo pipefail

VENV_DIR=".venv"
ACTIVATE_POSIX="$VENV_DIR/bin/activate"
ACTIVATE_WIN_BASH="$VENV_DIR/Scripts/activate"

# Locate activation script
ACTIVATE=""
if [ -f "$ACTIVATE_POSIX" ]; then
  ACTIVATE="$ACTIVATE_POSIX"
elif [ -f "$ACTIVATE_WIN_BASH" ]; then
  ACTIVATE="$ACTIVATE_WIN_BASH"
fi

if [ -z "$ACTIVATE" ]; then
  echo "Error: virtualenv activate script not found under $VENV_DIR; run 'make venv' first." >&2
  exit 2
fi

# Allow activation to update the prompt
export VIRTUAL_ENV_DISABLE_PROMPT=0

# Source the activation script (POSIX-style activation works under Git Bash)
# shellcheck disable=SC1090
. "$ACTIVATE"

# If VIRTUAL_ENV was set by the activate script, make minimal, safe prompt changes.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  # envname typically '.venv' or the base dir; prefer VIRTUAL_ENV_PROMPT if provided
  envname="${VIRTUAL_ENV_PROMPT:-${VIRTUAL_ENV##*/}}"

  # Only modify PS1 if it doesn't already contain the env marker
  if [ -n "${PS1:-}" ]; then
    if ! echo "$PS1" | grep -q "(${envname})" >/dev/null 2>&1; then
      # Prepend the env prefix, preserving the remainder of PS1
      PS1="(${envname}) ${PS1}"
    fi
  else
    # A sensible default when PS1 is empty
    PS1="(${envname}) \u@\h:\w\$ "
  fi

  # Preserve PROMPT_COMMAND if set; otherwise set a safe no-op so shells that
  # read it do not fail. Avoid injecting backslash escapes or newlines.
  if [ -z "${PROMPT_COMMAND:-}" ]; then
    PROMPT_COMMAND=:
  fi

  export PS1 PROMPT_COMMAND
fi

echo "venv activated: ${VIRTUAL_ENV:-}" 

# Exec an interactive shell unless in test mode
if [ "${ACTIVATE_NOEXEC:-0}" = "1" ]; then
  echo "ACTIVATE_NOEXEC=1 set; not launching interactive shell (testing mode)."
  return 0 2>/dev/null || exit 0
else
  exec "$SHELL" -i
fi

## Ensure the venv prefix survives prompt frameworks that re-set PS1 via PROMPT_COMMAND
## Define a tiny function that will ensure PS1 contains the prefix and append it to
## PROMPT_COMMAND so it runs after other prompt helpers.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  # use the chosen envname
  envname="${VIRTUAL_ENV_PROMPT:-${VIRTUAL_ENV##*/}}"

  # define the function only if it's not already defined
  if ! command -v __venv_ensure_prefix >/dev/null 2>&1; then
    __venv_ensure_prefix() {
      # If PS1 does not start with the prefix, prepend it
      case "$PS1" in
        "(${envname})"* ) ;; # already has prefix
        "(${envname} "* ) ;; # variant
        * ) PS1="(${envname}) ${PS1:-\u@\h:\w\$ }" ;;
      esac
    }
    export -f __venv_ensure_prefix 2>/dev/null || true
  fi

  # Append our function to PROMPT_COMMAND so it runs after other helpers; avoid duplication
  case "${PROMPT_COMMAND:-}" in
    *__venv_ensure_prefix* ) ;;
    "" ) PROMPT_COMMAND=__venv_ensure_prefix ;;
    * ) PROMPT_COMMAND="${PROMPT_COMMAND};__venv_ensure_prefix" ;;
  esac
  export PROMPT_COMMAND
fi
