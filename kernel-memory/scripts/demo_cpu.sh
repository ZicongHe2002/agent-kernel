#!/usr/bin/env bash
# Real CPU demonstration: registers the demo vector-add kernel/config, a baseline whose identity is
# the content address of the actually loaded demo source, a local trial context, then EXECUTES and
# MEASURES vector addition on this host (backend "cpu", provenance "trusted_worker"), compares a
# runtime-override variant against the baseline, evaluates the promotion policy, and runs the
# MockPlanner orchestration with a small budget. This proves execution integration on CPU only.
# It says nothing about MLA or TPU performance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-$HERE/.demo/cpu-memory}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-10}"
KMEM="$HERE/.venv/bin/kmem"
if [ ! -x "$KMEM" ]; then KMEM="$HERE/.venv/bin/python -m kernel_memory.cli.main"; fi
OUT="$ROOT/../cpu-demo-files"

rm -rf "$ROOT" "$OUT"
mkdir -p "$OUT"
echo "== init =="
$KMEM --root "$ROOT" init --json
echo "== default protocol/verifier + demo source content address =="
$KMEM --root "$ROOT" cpu-demo-defaults --out "$OUT" --repetitions "$REPS" --warmup "$WARMUP" --json
echo "== register kernel and config (normalizer: f32 alias -> float32) =="
$KMEM --root "$ROOT" register-kernel --kernel-id demo_vector_add --display-name "CPU demo vector add" \
  --adapter-id cpu-demo-v1 --notes "Real CPU demonstration operator; not MLA." --json
CFG_JSON=$($KMEM --root "$ROOT" register-config --kernel-id demo_vector_add --problem '{"n": 4096, "dtype": "f32"}' --tag cpu-demo --json)
echo "$CFG_JSON"
CFG=$(printf '%s' "$CFG_JSON" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["config"]["record_id"])')
echo "== baseline: numpy vector add at the loaded-source content address =="
$KMEM --root "$ROOT" add-baseline --config "$CFG" --baseline-id cpu-demo-reference \
  --repo-uid local:kernel-memory-cpu-demo --commit cpu-demo-source \
  --entrypoint kernel_memory.adapters.cpu_demo:vector_add_numpy --role both \
  --description "numpy vector add; identity = sha256 content address of the loaded demo source" --json
echo "== execute baseline (real measurement on this host) =="
$KMEM --root "$ROOT" run --subject baseline-cpu-demo-reference --backend cpu \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" \
  --session-id session-cpu-1 --pair-id pair-cpu-1 --role baseline --request-id request-cpu-baseline-1 --json
echo "== execute chunked variant (runtime override on the same source => new variant_digest) =="
$KMEM --root "$ROOT" run --subject baseline-cpu-demo-reference --backend cpu \
  --entrypoint kernel_memory.adapters.cpu_demo:vector_add_numpy_chunked --overrides '{"chunk": 512}' \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" \
  --session-id session-cpu-1 --pair-id pair-cpu-1 --role candidate --request-id request-cpu-chunked-1 --json
echo "== replaying the same request id does not execute again (idempotent) =="
$KMEM --root "$ROOT" run --subject baseline-cpu-demo-reference --backend cpu \
  --entrypoint kernel_memory.adapters.cpu_demo:vector_add_numpy_chunked --overrides '{"chunk": 512}' \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" \
  --session-id session-cpu-1 --pair-id pair-cpu-1 --role candidate --request-id request-cpu-chunked-1 --json
echo "== deliberately incorrect candidate: executes but fails correctness (never promotable) =="
$KMEM --root "$ROOT" run --subject baseline-cpu-demo-reference --backend cpu \
  --entrypoint kernel_memory.adapters.cpu_demo:vector_add_wrong \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" --request-id request-cpu-wrong-1 --json
echo "== compare variant vs baseline (same comparison key) =="
$KMEM --root "$ROOT" compare --candidate run-request-cpu-chunked-1-a1 --baseline run-request-cpu-baseline-1-a1 --json || true
echo "== decide with one pair (expected: inconclusive, INSUFFICIENT_CONFIRMATION_PAIRS) =="
printf '[{"candidate_run": "run-request-cpu-chunked-1-a1", "baseline_run": "run-request-cpu-baseline-1-a1"}]\n' > "$OUT/pairs.json"
$KMEM --root "$ROOT" decide --candidate baseline-cpu-demo-reference --pairs "$OUT/pairs.json" --json || true
echo "== TPU backend on this host: explicit BackendUnavailable (exit 5), no fallback =="
$KMEM --root "$ROOT" run --subject baseline-cpu-demo-reference --backend jax_tpu \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" --request-id request-tpu-1 --json || echo "exit=$? (expected 5 or 7)"
echo "== deep validation of real runs =="
$KMEM --root "$ROOT" validate --deep --json
echo "== trajectory =="
$KMEM --root "$ROOT" trajectory --config "$CFG" --rebuild --json
echo "== MockPlanner orchestration (dry run, then a small real budget) =="
printf '{"max_candidates": 3, "max_execution_attempts": 3, "max_model_calls": 0, "max_wall_time_seconds": 600, "max_concurrent_runners": 1}\n' > "$OUT/budget.json"
$KMEM --root "$ROOT" optimize --config "$CFG" --planner mock --subject baseline-cpu-demo-reference \
  --baseline-run run-request-cpu-baseline-1-a1 --backend cpu \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" --budget "$OUT/budget.json" --dry-run --json
$KMEM --root "$ROOT" optimize --config "$CFG" --planner mock --subject baseline-cpu-demo-reference \
  --baseline-run run-request-cpu-baseline-1-a1 --backend cpu \
  --protocol "$OUT/cpu_protocol.json" --verifier "$OUT/cpu_verifier.json" --budget "$OUT/budget.json" --job-id job-cpu-demo --json
echo "== export context =="
$KMEM --root "$ROOT" export-context --config "$CFG" --json
echo "CPU demo complete. Store: $ROOT"
