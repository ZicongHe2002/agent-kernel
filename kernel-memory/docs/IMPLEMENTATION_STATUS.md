# Implementation status

Project: `agent-kernel/kernel-memory/` (git repo `ZicongHe2002/agent-kernel`) · Python 3.11.8 venv at `.venv` ·
Updated 2026-09-24 · Spec: `../kernel_memory_ai_handoff/docs/IMPLEMENTATION.en.md` (read-only input; its
`MANIFEST.sha256` verifies unchanged).

Read this file first when resuming. It records what is done, what was actually executed, what is unexecuted and
why, and the minimum configuration a user still has to supply.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M0 workspace inspection, project root, venv, ADR-0001 (+ amendment for the move to `agent-kernel/`) | done | `docs/adr/ADR-0001-workspace-boundary.md`, `requirements.lock.txt` |
| M1 contracts and identity (RFC 8785 JCS, hashes, strict models, problem normalizers) | done | `tests/test_jcs.py` (4 golden vectors + RFC 8785 appendix cases), `test_hashing.py` (every fixture hash reproduces), `test_models.py`, `test_problems.py` |
| M2 authoritative storage (atomic publish, idempotency, artifacts, journal, integrity, recovery, SQLite index, lock) | done | `test_storage.py`, `test_storage_recovery.py` (crash injection), `test_storage_concurrency.py` (threads + second process + `LOCK_TIMEOUT`), `test_layout.py` |
| M3 validation/import, trajectory/query/context, compare/decide | done | `test_validation.py` (86), `test_importer.py` (all 12 handoff negative cases), `test_trajectory.py`, `test_query.py`, `test_context.py`, `test_compare.py`, `test_decide.py`, `test_policy.py` |
| M4 CLI + offline demonstration | done | `kmem` with 25 subcommands, `test_cli.py` (exit codes 0/2/3/4/5/7), `docs/evidence/demo_p0.log` |
| M5 read-only GitHub client + `collect-pr` (pagination, 250-limit, force push, shared source) | done offline | `test_github_client.py`, `test_collect.py` with `tests/fixtures/github/*.json`; **live collection unexecuted** |
| M6 real CPU demo adapter, JAX/TPU interface, request ledger, runner | done | `test_cpu_demo.py` (real execution), `test_jax_tpu.py` (T29 executes: `BackendUnavailable` on this CPU-only host), `test_ledger.py`, `test_runner.py`, `docs/evidence/demo_cpu.log` |
| M7 orchestration (budgets, ledger, MockPlanner, cancel/restart, decisions) | done | `test_budget.py`, `test_coordinator.py`; `kmem optimize` runs 3 real CPU rounds and stops on the candidate budget |
| M8 real MLA kernel + TPU + LLO | **blocked** | no MLA source/ABI, no TPU device, no execution authorization, no LLO sample; explicit refusal paths tested |

## Commands actually executed and their results (2026-09-24, this Mac, CPU only)

```bash
.venv/bin/python -m pytest tests -q -p no:cacheprovider --junitxml .demo/junit.xml
# 1892 passed in 59.24s — 0 failed, 0 skipped, 38 test files (1272 test functions, some parametrized)
.venv/bin/python scripts/acceptance_report.py --junit .demo/junit.xml --out docs/ACCEPTANCE.md
# 32/32 acceptance scenarios mapped to executed tests; T02, T09, T17, T29 marked partial (external prerequisite missing)
bash scripts/demo_p0.sh .demo/p0-memory              # exit 0 → docs/evidence/demo_p0.log
REPS=50 WARMUP=10 bash scripts/demo_cpu.sh .demo/cpu-memory   # exit 0 → docs/evidence/demo_cpu.log
.venv/bin/python ../kernel_memory_ai_handoff/tools/validate_handoff.py ../kernel_memory_ai_handoff --report .demo/handoff_validation.json
# status passed, 18 records, 4 golden vectors → docs/evidence/handoff_validation.json (handoff directory left byte-identical)
```

Offline P0 demo (`docs/evidence/demo_p0.log`, fixture numbers only): import 18 records + 7 artifacts; re-import 18
idempotent / 0 published; `validate --deep` ok (0 errors, 4 informational warnings about fixture provenance);
`trajectory --rebuild/--verify` deterministic; `query --run-status not_run` → `commit-demo-a-in-102`, `commit-demo-b`;
`query --component tiling` → `commit-demo-a`; `compare run-demo-a run-demo-baseline` → COMPARABLE, speedup 1.1111,
latency reduction 10.0 %, blocker `FIXTURE_NOT_ELIGIBLE`; `decide` → **blocked** (`FIXTURE_NOT_ELIGIBLE`,
`INSUFFICIENT_CONFIRMATION_PAIRS`), `is_production false`.

Real CPU demo (`docs/evidence/demo_cpu.log`; host-synchronized numbers on this Mac, **not** kernel performance):
kernel `demo_vector_add`, config n=4096 f32 (hash `sha256:8085a979…`), baseline identity = content address of the loaded
demo source (`sha256:61ae1e5d…`). Baseline run `trusted_worker`/`cpu`, succeeded, correctness pass, 50 samples,
median 1.125 µs; chunked variant (new `variant_digest`) succeeded, median 7.333 µs; replaying the request id executed
nothing new; the deliberately wrong entrypoint is `succeeded` + correctness **fail** (0/4 cases, max abs error
0.00195); compare → COMPARABLE, speedup 0.1534, confirmation-eligible; decide → inconclusive
(`INSUFFICIENT_CONFIRMATION_PAIRS`, `PAIR_SPEEDUP_BELOW_THRESHOLD`); `--backend jax_tpu` → exit 7
`TPU_EXECUTION_NOT_AUTHORIZED` (authorization precedes the device probe; with authorization the same host yields exit 5
`BackendUnavailable`, tested); `validate --deep` ok on the real runs; `optimize --planner mock` → dry run 3 proposals,
real run 3 rounds (all succeeded/pass/COMPARABLE, annotated), stop `BUDGET_CANDIDATES_EXHAUSTED`; MockPlanner output
is orchestration evidence only, not optimization effectiveness.

## Unexecuted integrations and the exact missing input

| Integration | Status | Missing input (nothing else blocks it) |
|---|---|---|
| Live GitHub collection | unexecuted (offline fixtures tested) | `github_repository` (`OWNER/REPO`), a read-only token in the env var named by `github_token_env_name`, `permissions.allow_network=true` |
| TPU execution | unexecuted (refusal path executed) | a TPU device visible to JAX, `permissions.allow_tpu_execution=true` (+ per-request authorization), `kernel_entrypoint` resolvable by a configured resolver |
| `mla_forward` configs | refused (exit 5, 7 unresolved items listed) | a problem schema + normalizer derived from the real MLA source (`problem_schema_path`), trusted reference (`trusted_reference_entrypoint`), approved tolerances/suite (`approved_verifier_path`) |
| LLO analysis | unsupported (`LloAnalysisAdapter` raises `UnsupportedFormat`) | a real LLO format specification or sample; then a new analysis adapter with its own `parser_version` |
| Model-driven planner | unexecuted (`MODEL_PROVIDER_NOT_CONFIGURED`) | `model_provider`, credentials in the env var named by `model_api_key_env_name`, `permissions.allow_model_api_calls=true` |
| Remote PR write actions | disabled by design | `permissions.allow_remote_write=true` plus explicit user authorization; no code path performs remote writes today |
| v0.1 data migration | exercised on synthetic input only | a real v0.1 export; the reader is tolerant and must be confirmed against it (`docs/MIGRATION.md`) |

## Repository state

* Root `.gitignore` added; `.venv`, `__pycache__`, `egg-info`, `.DS_Store` untracked from the index (8103 staged
  deletions, **not committed**; 108 legitimate files tracked + 41 new untracked files from this work). Nothing was
  committed or pushed; that decision is the user's.
* macOS quirk: the editable-install `.pth` keeps being re-flagged hidden (Python 3.11.4+ then skips it). Two
  independent fallbacks are in place: `site-packages/kernel_memory` symlink → `src/kernel_memory` (so `kmem` works from
  any cwd) and `PYTHONPATH=$HERE/src` exported by the demo scripts; `tests/conftest.py` also adds `src/`.

## Minimum remaining user configuration

Copy `fixtures/handoff/examples/project_settings.template.json` to a settings file and fill, in this order of value:
`memory_root`; for GitHub: `github_repository`, token env var, `allow_network`; for TPU/MLA: `source_repository`,
`kernel_entrypoint`, `problem_schema_path`, `trusted_reference_entrypoint`, `approved_verifier_path`,
`allow_tpu_execution`; for a model planner: `model_provider`, `model_api_key_env_name`, `allow_model_api_calls`.
Everything else runs offline today.

## Next action

None required for P0–P2. For M8: obtain the MLA callable and ABI from the real repository, write the `mla_forward`
problem schema/normalizer and an entrypoint resolver, register the trusted reference and approved tolerances, then run
`kmem run --backend jax_tpu` on a TPU host with authorization; report TPU results separately from these CPU results.

History: wave 1 (missing modules + 8 fixes) and waves 2/2b (test suites + docs) were executed by parallel agents with
disjoint file ownership; earlier interrupted attempts are described in `docs/adr/ADR-0001-workspace-boundary.md`.
