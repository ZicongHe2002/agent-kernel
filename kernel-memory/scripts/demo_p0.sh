#!/usr/bin/env bash
# Offline P0 demonstration: import the synthetic handoff bundle, validate deeply, rebuild the
# trajectory, query, compare, evaluate a decision (blocked: fixtures never become production),
# and export agent context. No network, TPU, or model access. All numbers are fixture values,
# not measurements.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-$HERE/.demo/p0-memory}"
KMEM="$HERE/.venv/bin/kmem"
if [ ! -x "$KMEM" ]; then KMEM="$HERE/.venv/bin/python -m kernel_memory.cli.main"; fi

rm -rf "$ROOT"
echo "== init =="
$KMEM --root "$ROOT" init --json
echo "== import fixture bundle (explicit --allow-fixture; provenance stays 'fixture') =="
$KMEM --root "$ROOT" import-bundle "$HERE/fixtures/handoff/examples/demo_bundle.json" \
  --artifact-root "$HERE/fixtures/handoff" --allow-fixture --json
echo "== second import is idempotent =="
$KMEM --root "$ROOT" import-bundle "$HERE/fixtures/handoff/examples/demo_bundle.json" \
  --artifact-root "$HERE/fixtures/handoff" --allow-fixture --json
echo "== deep validation =="
$KMEM --root "$ROOT" validate --deep --json
echo "== trajectory (deterministic rebuild) =="
$KMEM --root "$ROOT" trajectory --config cfg-demo --rebuild --json
$KMEM --root "$ROOT" trajectory --config cfg-demo --verify --json
echo "== query: untested commits (commit-demo-b has no run) =="
$KMEM --root "$ROOT" query --config cfg-demo --record-type commit --run-status not_run --json
echo "== query: tiling changes =="
$KMEM --root "$ROOT" query --config cfg-demo --component tiling --json
echo "== compare run-demo-a vs baseline (fixture arithmetic: speedup 100/90, reduction 10%) =="
$KMEM --root "$ROOT" compare --candidate run-demo-a --baseline run-demo-baseline --json || true
echo "== decide (expected: blocked, FIXTURE_NOT_ELIGIBLE + INSUFFICIENT_CONFIRMATION_PAIRS) =="
PAIRS="$ROOT/../p0-pairs.json"
printf '[{"candidate_run": "run-demo-a", "baseline_run": "run-demo-baseline"}]\n' > "$PAIRS"
$KMEM --root "$ROOT" decide --candidate commit-demo-a --pairs "$PAIRS" --dry-run --json || true
echo "== export agent context =="
$KMEM --root "$ROOT" export-context --config cfg-demo --max-records 30 --json
echo "== status / pending integrations =="
$KMEM --root "$ROOT" status --json
echo "P0 demo complete. Store: $ROOT"
