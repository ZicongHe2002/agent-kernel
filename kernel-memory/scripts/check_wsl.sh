#!/usr/bin/env bash
# Capture installation, tests, and demos in one text file; keep the terminal quiet.
set -euo pipefail

INSTALL_SPEC='.[dev]'
case "${1:-}" in
  '') ;;
  --with-jax) INSTALL_SPEC='.[dev,jax]' ;;
  *) printf 'Usage: bash scripts/check_wsl.sh [--with-jax]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Usage: bash scripts/check_wsl.sh [--with-jax]\n' >&2
  exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
mkdir -p -- "$PROJECT_DIR/.demo/checks"
LOG_FILE="$PROJECT_DIR/.demo/checks/latest.txt"
PYTHON="$PROJECT_DIR/.venv/bin/python"
ACTIVE_STAGE='setup'
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Descriptor 3 preserves the terminal; everything else, including stderr, goes to the log.
exec 3>&1
exec > "$LOG_FILE" 2>&1

announce() {
  printf '%s\n' "$*"
  printf '%s\n' "$*" >&3
}

finish() {
  local result=$?
  trap - EXIT
  if (( result == 0 )); then
    announce "[$(date -Is)] DONE: all stages completed (exit=0)."
  else
    announce "[$(date -Is)] STOPPED: $ACTIVE_STAGE (exit=$result); later stages were not run."
  fi
  announce "Full output: $LOG_FILE"
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
  ACTIVE_STAGE="$1"
  shift
  announce "[$(date -Is)] START: $ACTIVE_STAGE"
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  local result=0
  "$@" || result=$?
  announce "[$(date -Is)] END: $ACTIVE_STAGE (exit=$result)"
  if (( result != 0 )); then
    exit "$result"
  fi
}

announce "Full output: $LOG_FILE (replaced on each run)"
printf 'Project: %s\nSystem: %s\n' "$PROJECT_DIR" "$(uname -a)"
printf 'Plan: install dependencies -> environment -> pytest -> P0 demo -> CPU demo\n'

if [[ ! -x "$PYTHON" ]]; then
  if [[ -e "$PROJECT_DIR/.venv" ]]; then
    announce 'ERROR: .venv exists but has no executable Linux bin/python. Use a WSL venv in a separate checkout from Windows.'
    exit 2
  fi
  run_stage 'Create Linux venv' python3 -m venv .venv
fi

run_stage 'Install test dependencies' "$PYTHON" -m pip install --disable-pip-version-check --no-input -e "$INSTALL_SPEC"
run_stage 'Environment and dependency imports' "$PYTHON" - "$INSTALL_SPEC" <<'PY'
import importlib
import importlib.util
from importlib.metadata import version
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
for module, distribution in (
    ("kernel_memory", "kernel-memory"),
    ("pytest", "pytest"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("jsonschema", "jsonschema"),
):
    importlib.import_module(module)
    print(f"{distribution}: {version(distribution)}")
if "jax" in sys.argv[1] or importlib.util.find_spec("jax") is not None:
    importlib.import_module("jax")
    print(f"jax: {version('jax')}")
else:
    print("jax: not installed; optional real-JAX tests will be skipped (use --with-jax to include them)")
PY
run_stage 'Pytest' "$PYTHON" -m pytest tests -q --color=no
run_stage 'P0 demo' bash "$PROJECT_DIR/scripts/demo_p0.sh"
run_stage 'CPU demo' bash "$PROJECT_DIR/scripts/demo_cpu.sh"
