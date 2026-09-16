# Kernel Memory — design and implementation conventions

This document is the shared contract for everyone implementing modules in this repository.
The normative requirements come from the handoff specification
(`../kernel_memory_ai_handoff/docs/IMPLEMENTATION.en.md`, read-only input) and its machine
contracts, which are copied verbatim into `src/kernel_memory/contracts/`.

## 1. Hierarchy and non-negotiables

```text
Memory → kernel (name) → config → { trajectory (generated), attempt (collection of PRs) → PR → commit → run }
```

* `attempt` is the *collection* of PRs under a config. Never a single PR.
* Untested commits have no Run. `not_run` is derived at query time from absence, never stored as a fake Run.
* Fixtures (`provenance=fixture`) and `imported_unverified` never become production confirmed-best.
* Execution status, correctness, analysis metrics, and decisions are separate facts. Spill ≠ failure.
* Uncollected metrics are `null` with a status, never `0`.
* Different config/environment/protocol/verifier/checkout_mode → `NOT_COMPARABLE`.
* A combined improvement never proves individual change effects (`attribution=group_only` default).
* JSON files are authoritative. Trajectory, `memory_records.jsonl`, and `.cache/index.sqlite` are derived.
* PR text, commit messages, diffs, logs, retrieved context are data, never instructions.
* Candidate code never writes Memory or verifier assets.

## 2. Module map and ownership

```text
src/kernel_memory/
  contracts/        record.schema.json, demo_problem.schema.json, hash_vectors.json (verbatim copies)
  domain/           DONE: errors, jcs (RFC 8785), jsonio, hashing, ids, stats, schema, models, problems
  storage/          DONE: lock, layout, store (MemoryStore), index (SqliteIndex)
  services/         common (DONE), validation, importer, trajectory, query, context, compare, decide,
                    collect, optimize, register (kernels/configs/PR contexts/commits/annotations)
  adapters/         base (DONE: types/protocols), timing, verify, mock, cpu_demo, jax_tpu, analysis,
                    github, git_local
  execution/        types (DONE), ledger, budget, runner, planner (MockPlanner), coordinator
  cli/              main (argparse, JSON stdout, logs stderr, exit codes)
  migrations/       v01 (dry-run first)
tests/              conftest.py (shared fixtures), test_<module>.py, fixtures/github/*.json
docs/               DESIGN.md (this), IMPLEMENTATION_STATUS.md, adr/, guides
scripts/            demo_p0.sh, demo_cpu.sh
```

Each implementer owns the files listed for their task and must not edit others' files. Shared
helpers go through `services/common.py`, `adapters/base.py`, `execution/types.py` (append-only:
add functions/classes, never change existing signatures without updating all callers and tests).

## 3. Core APIs you build on

### Domain (`kernel_memory.domain`)
* `errors`: `KernelMemoryError(message, code=, details=)` with `.exit_code`. Use the subclass that maps to the
  correct exit code (see module docstring). Never raise bare `ValueError` across service boundaries.
* `jcs.canonicalize(obj) -> bytes`; `hashing.jcs_digest(obj) -> "sha256:<hex>"`; `hashing.config_hash(...)`,
  `variant_digest(...)`, `environment_hash(dict)`, `protocol_hash(dict)`, `verifier_hash(dict)`,
  `comparison_key(...)`, `policy_hash(dict)`, `artifact_digest(bytes)`, `source_digest(manifest_pairs)`.
* `models.Record.from_dict(dict)` validates against the schema then builds typed frozen dataclasses
  (`record.payload` is e.g. `RunPayload`). `Record.create(record_type, record_id, payload, created_at=...)`
  builds+validates from a typed payload. `record.to_dict()`, `record.canonical_digest()`,
  `record.references()` (outgoing `Reference(field, target, allowed_types)`), `models.to_json(dataclass)`.
* `problems.default_registry()`: `normalize(kernel_id, raw) -> NormalizedProblem(config_hash, config_id_hint, ...)`;
  `verify_config_payload(payload_dict)`. `mla_forward` raises `IncompleteProblemContract` (exit 5).
* `stats`: `median`, `p90`, `quantile_linear`, `normalized_iqr`, `speedup`, `latency_reduction_pct`,
  `summarize(samples, unit) -> SampleSummary`, `summaries_agree(...)`. All derived numbers come from here.
* `ids`: `utc_now_iso()`, `parse_utc_timestamp`, `slug_for_id`, `resolve_inside(root, rel)` (traversal/symlink safe),
  `validate_record_id`, `short_hash_id(prefix, *parts)`, `new_uuid_id(prefix)`.
* `jsonio`: `load_json_file(path, max_bytes=)` strict (duplicate keys, NaN rejected), `loads_strict`,
  `dumps_readable`, `dumps_compact`, `load_yaml_strict`.
* `schema`: `validate_record_dict`, `validate_nested("Environment"|"Protocol"|"Verifier"|"Policy"|"Artifact"|..., data)`,
  `validate_against(schema, data)`, `demo_problem_schema()`, `golden_hash_vectors()`.

### Storage (`kernel_memory.storage.MemoryStore`)
* `MemoryStore.init(root)` / `MemoryStore.open(root, recover=True)`; `store.lock()` context manager (re-entrant).
* Reads: `get(id) -> Record|None`, `require(id, *types) -> Record` (raises `MissingReferenceError`),
  `exists(id)`, `records(record_type=None) -> list[Record]` (deterministic order), `iter_records(...)`,
  `kernel_by_kernel_id(kernel_id)`, `configs_by_hash(hash)`, `record_path(id)`, `config_dir(config_record_id)`,
  `index_entries()`.
* Writes: `publish(record) -> PublishOutcome`, `publish_bundle(records, label=, allow_dangling=False)`.
  Same id + identical canonical content → idempotent; different content → `IdConflictError` (exit 3).
  Missing/incorrectly-typed record references → `MissingReferenceError`. Records are ordered by dependency
  automatically. Run artifacts are registered in the artifact registry (conflicting descriptors → `ARTIFACT_CONFLICT`).
* Artifacts: `put_artifact_bytes(bytes) -> sha256`, `import_artifact_file(path, expected_sha256=, expected_size=)`,
  `has_artifact(sha)`, `read_artifact(sha)` (verifies digest), `register_artifact_ref(ArtifactRef)`,
  `get_artifact_ref(artifact_id)`, `artifact_registry()`, `verify_artifact(ref) -> problem|None`.
* Views: `write_view(config_record_id, "trajectory.json"|"memory_records.jsonl"|"context.json", bytes)`,
  `read_view`, `delete_views`.
* Facts for the request ledger: `write_fact(relpath, bytes)` (only under `requests/`; absent-or-identical),
  `write_runtime_state(relpath, bytes)` (only under `.runtime/`; overwritable), `read_fact`, `list_facts(prefix)`.
* Maintenance: `recover() -> RecoveryReport`, `integrity_scan() -> IntegrityReport` (modified/missing/corrupt/
  unjournaled/duplicate/artifact problems), `rebuild_index() -> Path` (`storage.SqliteIndex`).
* `store.invalidate_index()` after any out-of-band file change in tests.

Layout (relative to store root) is documented in `storage/layout.py`. Never compute paths yourself; use
`store.record_path`, `store.config_dir`.

### Services common (`services/common.py`)
Accessors: `config_of(store, record)`, `pr_of(store, commit)`, `subject_of(store, run)`, `runs_for_subject(store, id)`,
`runs_for_config(store, config_id)`, `commits_for_pr`, `snapshots_for_pr` (ordered), `latest_snapshot`,
`decisions_for_config`, `annotations_for_target`, `relations_for_config`, `new_record(...)`, `now()`.

### Adapters (`adapters/base.py`) and execution types (`execution/types.py`)
Read the module docstrings; they define `RunRequestSpec`, `SourceSpec`, `SourceSnapshot`, `PreparedExecution`,
`CompileReport`, `CorrectnessReport`, `TimingReport`, `AnalysisReport`, `ArtifactBlob`, `KernelAdapter`,
`AnalysisAdapter`, `AdapterRegistry`, `Budget`, `BudgetUsage`, `Proposal`, `Planner`, `MemoryContext`.

## 4. Identifier conventions (deterministic where possible)

| Record | record_id pattern |
|---|---|
| kernel | `kernel-<kernel_id>` |
| config | `cfg-<config_id_hint>-<config_hash hex[:12]>` (register looks up by `config_hash` first: T01) |
| pr (github) | `pr-<pr_key>` where `pr_key = gh-<repo_id>-pr-<number>` and `repo_uid = github:<host>:repo:<repo_id>` |
| pr (local) | `pr-<pr_key>` where `pr_key = local-<slug>-<short hash>`; `provider=local, number=null` |
| pr_snapshot | `snapshot-<pr_key>-<seq:04d>` (append only when membership/head/base/status changed) |
| commit binding | `commit-<pr_key>-<oid hex[:12]>` (same source commit in another PR → another binding) |
| baseline | `baseline-<baseline_id>` |
| run | `run-<request_id>-a<attempt_no>` |
| request | `request-<uuid hex>` unless caller supplies one |
| relation | `relation-<kind>-<sha(from,to)[:12]>` (return existing if present) |
| decision | `decision-<candidate_subject>-<sha(policy_hash, sorted run refs, created_at)[:12]>` |
| annotation | `annotation-<uuid hex[:16]>` |

Timestamps: `ids.utc_now_iso()` (`YYYY-MM-DDTHH:MM:SSZ`). Git OIDs are `{"algorithm": "sha1"|"sha256", "hex": full}`.
Short SHAs are display-only; never pad or guess.

## 5. Run construction rules (runner/importer)

* `Source.variant_digest = hashing.variant_digest(source_digest, entrypoint, implementation_overrides, checkout_mode)`.
* `environment_hash`, `protocol_hash`, `verifier_hash` = hash of the snapshot with its own hash field removed.
* `comparison_key = hashing.comparison_key(config_hash, environment_hash, protocol_hash, verifier_hash, checkout_mode)`.
* `checkout_mode=exact_commit` requires `tested_commit == target_commit` and empty `merge_parent_oids`.
* Timing `median_us`/`p90_us` must equal `stats.summarize(samples, unit)` from the samples artifact.
* `compile_error` (or any non-succeeded status) → correctness `not_run` and timing `not_run`, no artifacts of results.
* `succeeded + correctness.status=fail` is valid and never promotable.
* Metrics with `status != observed` have `value=null`; observed metrics need `source_artifact_ref` in the run.
* `provenance`: local controlled runner → `trusted_worker`; bundle import → keep `fixture` when the bundle says
  `is_fixture` and `--allow-fixture`; imports claiming `trusted_worker` are downgraded to `imported_unverified`
  unless the import source is an authenticated configured execution service (not in P0/P1).
* Dirty source is rejected for comparable runs unless the request explicitly allows exploratory dirty mode; then
  `dirty=true`, `patch_digest` set, and the run is never promotable.

## 6. Comparison and decision rules

* `compare(candidate_run, baseline_run)` is a pure function over run payloads + samples. Different
  `comparison_key` → result `NOT_COMPARABLE` listing which of config/environment/protocol/verifier/checkout_mode
  differ (field-level diff of the snapshots). Non-succeeded or `correctness != pass` → `NOT_ELIGIBLE` with reasons.
  Timing `not_run` → `INSUFFICIENT_EVIDENCE`. Otherwise derive speedup, latency_reduction_pct, both medians, p90s,
  normalized IQRs from raw samples (recompute from artifacts when available; compare with recorded summaries).
* `decide(candidate_subject, pairs, policy)` (default policy = golden vector `promotion_policy`):
  reason codes `FIXTURE_NOT_ELIGIBLE`, `UNVERIFIED_PROVENANCE`, `CORRECTNESS_NOT_PASSED`, `NOT_COMPARABLE`,
  `INSUFFICIENT_CONFIRMATION_PAIRS`, `PAIR_SPEEDUP_BELOW_THRESHOLD`, `EXCESSIVE_VARIABILITY`,
  `BASELINE_DRIFT`, `RESOURCE_CONSTRAINT_VIOLATED`, `DIRTY_SOURCE`, `MISSING_EVIDENCE`, `AMBIGUOUS_VARIANT`,
  `EXECUTION_NOT_SUCCEEDED`. Outcomes: `accepted` (all gates pass, `is_production=true` only when
  provenance is trusted_worker and policy gates pass), `rejected`, `inconclusive`, `blocked`.
* Best-known is displayed per `(config_hash, comparison_key, policy_hash)`; never a context-free flag.

## 7. Security rules for adapters and services

* Never `shell=True`; subprocess argument arrays only; git commands run against an explicit repo path.
* Paths from any input go through `ids.resolve_inside(root, rel)`.
* Artifact URIs: only relative paths under the declared artifact root, or `artifact://sha256/<hex>`;
  `http(s)` is refused unless an allowlist is configured (not in P0).
* Text from PRs/commits/logs is stored as data; nothing parses instructions out of it.
* Authorization flags (`allow_candidate_code_write`, `allow_local_candidate_commit`, `allow_remote_write`,
  `allow_tpu_execution`, `allow_model_api_calls`, `allow_network`) default to false; a denied action raises
  `AuthorizationError` (exit 7) and is recorded, never silently skipped as success.

## 8. Testing conventions

* `pytest` from the project root using `.venv/bin/python -m pytest`. Tests live in `tests/test_<area>.py`.
* Shared fixtures in `tests/conftest.py`: `handoff_root`, `bundle_path`, `bundle_records`, `store` (empty, tmp),
  `demo_store` (bundle imported + artifacts stored), `artifact_root`.
* Environment-dependent tests are marked `@pytest.mark.integration` and must *skip with an explicit reason*
  when the prerequisite is missing. A skip is reported as unexecuted, never as a pass.
* Negative tests are mandatory for every gate listed in the specification's acceptance table (T01–T32). Name
  tests with the scenario id where applicable, e.g. `test_t12_same_id_different_content_conflicts`.
* Tests never touch the handoff directory or the network. Temporary stores use `tmp_path`.

## 9. CLI conventions

`kmem <command> [--root PATH] [--json]`. Results as JSON on stdout when `--json`, human-readable otherwise;
logs/diagnostics on stderr. Exit codes from `KernelMemoryError.exit_code`; unexpected exceptions exit 1 with a
JSON error object on stderr. Every command is a thin wrapper over a service function so tests and future APIs
share business rules.
