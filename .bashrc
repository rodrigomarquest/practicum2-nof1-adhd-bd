# Project-local shell helpers for practicum2
# Adds convenient aliases to activate/deactivate the project's virtualenv (.venv)

# Use in project root: source ./.bashrc
# Define robust shell functions so they behave correctly regardless of how the shell
# evaluates aliases/expansions. These functions look for the project's .venv in the
# current working directory and source the right activate script.
venvon() {
  # prefer Windows-style Scripts/activate (Git Bash / MSYS) but fall back to bin/activate
  if [ -f "$PWD/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1090
    source "$PWD/.venv/Scripts/activate"
    # Some Windows/Git-Bash activate scripts don't export VIRTUAL_ENV reliably.
    if [ -z "$VIRTUAL_ENV" ] && [ -d "$PWD/.venv/Scripts" ]; then
      export VIRTUAL_ENV="$PWD/.venv"
      export PATH="$VIRTUAL_ENV/Scripts:$PATH"
      hash -r 2>/dev/null || true
    fi
    return $?
  elif [ -f "$PWD/.venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$PWD/.venv/bin/activate"
    return $?
  else
    echo ".venv not found in $PWD"
    return 2
  fi
}

venvoff() {
  if command -v deactivate >/dev/null 2>&1; then
    deactivate
    return $?
  else
    echo "venv not active or 'deactivate' not available"
    return 1
  fi
}
