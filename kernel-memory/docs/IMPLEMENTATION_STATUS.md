# Implementation status

Project: `kernel-agent/kernel-memory/` · Python 3.11.8 venv at `.venv` · Updated 2026-09-08

Read this file first when resuming. It records what is done, what was actually executed, what is
unexecuted and why, and the next action.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M0 workspace inspection, project root, venv, ADR-0001 | done | `docs/adr/ADR-0001-workspace-boundary.md`, `requirements.lock.txt` |
| M1 contracts and identity (JCS, hashes, models, normalizers) | done, smoke-tested | golden vectors 4/4 match; RFC 8785 appendix cases pass; fixture bundle round-trips byte-identically |
| M2 authoritative storage (atomic publish, idempotency, recovery, artifacts, journal, integrity, SQLite index) | done, smoke-tested | shuffled 18-record import; re-import idempotent; conflict/missing-ref/traversal rejected; tamper detected; pending manifest recovered |
| M3 validation/import, trajectory/query/context, compare/decide | in progress | — |
| M4 CLI + offline demo | pending | — |
| M5 GitHub read-only client + collect-pr (offline fixtures) | pending | — |
| M6 CPU demo adapter (real execution), JAX/TPU interface, request ledger | pending | — |
| M7 orchestration (budgets, MockPlanner, cancel/restart) | pending | — |
| M8 real MLA kernel + TPU | **blocked: no MLA source/ABI, no TPU, no authorization** | explicit `BackendUnavailable` path |

## Commands actually executed so far

```bash
python3 -m venv .venv
.venv/bin/pip install "jsonschema>=4.18,<5" "pytest>=7.4" "numpy>=1.26" "PyYAML>=6.0"   # network available; versions in requirements.lock.txt
.venv/bin/python <scratch>/check_jcs.py      # 4/4 golden vectors, RFC 8785 number/string/ordering cases: pass
.venv/bin/python <scratch>/check_models.py   # 18 records validated + round-tripped; negative cases rejected
.venv/bin/python <scratch>/check_store.py    # publish/idempotency/conflict/artifacts/tamper/recovery: pass
```

Formal pytest suite: not yet executed (being written).

## Unexecuted integrations and the exact missing input

| Integration | Status | Missing input |
|---|---|---|
| Live GitHub collection | unexecuted | `GITHUB_TOKEN` + repository name; client is tested offline only |
| TPU execution | unexecuted | TPU device, `allow_tpu_execution`, real MLA callable/ABI |
| MLA config registration | refused (exit 5) | backend-specific problem schema derived from the real repository |
| LLO analysis | unsupported | a real LLO sample/format specification |
| Model-driven planner | unexecuted | provider credentials + `allow_model_api_calls`; MockPlanner only |
| Remote PR write actions | disabled | `allow_remote_write` and explicit user authorization |

## Next action

Implement M3–M7 modules and their tests in parallel (see `docs/DESIGN.md` for ownership), run the full pytest
suite, fix failures, then build the CLI and offline/CPU demonstrations.
