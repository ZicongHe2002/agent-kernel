# Configuration guide

Everything runs offline by default. External integrations are switched on one at a time through a
project settings file plus environment variables for credentials. No credential is ever written to
Memory or to the settings file.

## 1. Settings file

Start from the handoff template (`fixtures/handoff/examples/project_settings.template.json`) and save
it as, for example, `kmem.settings.json` next to your Memory root. Loader: `kernel_memory.settings.load_settings`.

| Key | Meaning |
|---|---|
| `memory_root` | Store directory (relative paths resolve against the settings file). |
| `source_repository` | Local path of the kernel repository (for `record-commit`, source digests, reconciliation). |
| `kernel_id`, `kernel_entrypoint` | Real kernel identity and `module:function` entrypoint. Not guessed; supplied by you. |
| `problem_schema_path` | Backend-specific problem schema derived from the real repository (required before `mla_forward` configs can be registered). |
| `trusted_reference_entrypoint`, `approved_verifier_path` | Reference implementation and approved tolerances/suite for correctness. |
| `github_repository`, `github_token_env_name` | `owner/repo` and the *name* of the environment variable holding a read-only token. |
| `model_provider`, `model_api_key_env_name` | Planner provider and the *name* of the environment variable holding its key. |
| `llo_adapter` | Analysis adapter id for a real LLO format (none exists yet). |
| `permissions` | Safety switches, all false by default except `allow_project_code_write` and `allow_local_cpu_tests`. |
| `budget` | Optimization budgets (candidates, execution attempts, model calls, wall time, concurrency). |

Keys containing `token`, `secret`, `password`, `api_key`, or `credential` are rejected unless they end in
`_env_name`. Unknown keys are rejected.

## 2. GitHub read-only collection

Code: `kernel_memory.adapters.github.GitHubClient`, `kernel_memory.services.collect.collect_pr`.

* Offline by construction: the client takes an injectable transport. Tests use `FixtureTransport`.
* Live requests require `permissions.allow_network = true` **and** `UrllibTransport`. Without the flag the
  client raises `NETWORK_NOT_AUTHORIZED` (exit 5) before any request.
* Token: `export GITHUB_TOKEN=...` (or the variable named in `github_token_env_name`). Public repositories
  work without a token but are rate limited; the client backs off on 403/429 and retries 5xx.
* The client has no write methods. `allow_remote_write` gates nothing here because nothing writes.
* Known limits, handled explicitly: the PR-commits endpoint returns at most 250 commits; the compare
  endpoint is capped too. Coverage is recorded as `partial` unless a local clone (`source_repository`)
  lets `collect-pr` reconcile the full commit set via `git rev-list HEAD --not BASE`. Shallow clones stay `partial`.
* Force pushes append a new snapshot; old snapshots, commits, and runs are retained.

Minimum to run live: `github_repository`, `GITHUB_TOKEN` in the environment, `allow_network: true`.
Status today: **unexecuted live** (no repository or token was available); offline fixtures are tested.

## 3. CPU demonstration adapter (real execution)

Code: `kernel_memory.adapters.cpu_demo.CpuDemoAdapter` (backend `cpu`). Needs `numpy`. Executes vector
addition, verifies against an independent pure-Python reference with fixed tolerances, records raw
`perf_counter_ns` samples with `timing_method=host_synchronized`, and labels the environment `cpu`.
It proves execution integration, not MLA or TPU performance. Entry points are allowlisted; nothing is
imported from request strings.

## 4. JAX / TPU adapter

Code: `kernel_memory.adapters.jax_tpu.JaxAdapter(required_platform="tpu")` (backend `jax_tpu`).

Prerequisites, all checked explicitly and reported as `BackendUnavailable` / `AuthorizationError` /
`PrerequisiteMissingError` / `IncompleteProblemContract` when absent:

1. `jax` importable and `jax.default_backend() == "tpu"` with TPU devices. CPU JAX is **never** relabelled as TPU.
2. `permissions.allow_tpu_execution = true` (settings) and `authorization.allow_tpu_execution = true` on the request.
3. `kernel_entrypoint` resolvable through the configured entrypoint resolver, returning the callable, an input
   factory, the trusted reference callable, and the required output names (`O` vs `O+LSE` are different contracts).
4. A complete problem schema for `mla_forward` derived from the real source (`problem_schema_path`). The built-in
   `MlaForwardProblem` refuses to normalise anything until then (exit 5, listing the unresolved items).
5. Approved tolerances and suite (`approved_verifier_path`).

Benchmarks exclude compilation, call `jax.block_until_ready` on the whole output pytree, and consume all
required outputs. Profiling (`device_profiler`) runs as a separate protocol/scope, never mixed with host timing.
Status today: **unexecuted** (no TPU, no MLA source/ABI, no authorization). This machine has CPU-only JAX 0.10.2,
and the adapter reports exactly that.

## 5. Analysis / LLO

Code: `kernel_memory.adapters.analysis`. Metrics carry name, status, value (null unless observed), unit,
kind (`measurement` / `static_estimate` / `derived`), scope, source artifact, parser id/version, and a
definition. `LloAnalysisAdapter` accepts `llo_dump` artifacts and raises `UnsupportedFormat`: no LLO format
specification or sample exists, so no parser was invented. Supplying a format sample means implementing a new
adapter with its own `parser_version`; existing metrics are never rewritten, only re-annotated.
`MockSpillAnalysisAdapter` (`adapter_id=mock-spill`) parses only the fixture format `mock-spill-v1` used by the
synthetic bundle's spill report; it exists to exercise the metric pipeline offline, not to analyse real kernels. Its
one metric, `register_spill_vmem_static_bytes`, is a compiler-side *static estimate* (`kind=static_estimate`), not a
measurement of HBM traffic, and `0` is reported only when the report literally says `0`.

## 6. Model-driven planner

Code: `kernel_memory.execution.planner`. `MockPlanner` exercises the orchestration state machine with
low-risk runtime-override proposals. `UnavailableModelPlanner` raises `MODEL_PROVIDER_NOT_CONFIGURED`
until `model_provider`, the API key environment variable, and `permissions.allow_model_api_calls` are set.
Even then, an accepted candidate is a Memory decision, not a deployment: code writes, local commits, and
remote PR actions each need their own permission (`allow_candidate_code_write`, `allow_local_candidate_commit`,
`allow_remote_write`).

## 7. Budgets

Defaults (configurable, not promises): 8 candidates, 24 execution attempts, 12 model calls, 1800 s wall
time, 1 concurrent runner, stop after 3 consecutive execution failures or 4 rounds without a confirmable
improvement. Usage is persisted in the request ledger and restored after restart.

## 8. Venv troubleshooting (macOS)

If `.venv/bin/python -c "import kernel_memory"` fails with `ModuleNotFoundError`, the editable-install
`.pth` file has the macOS hidden flag and Python 3.11.4+ skips hidden `.pth` files:

```bash
chflags nohidden .venv/lib/python3.11/site-packages/*.pth
```

`tests/conftest.py` also adds `src/` to `sys.path` as a fallback so the test suite is unaffected. Because the flag
was observed to reappear, a `.pth`-independent fallback is also installed: a symlink
`.venv/lib/python3.11/site-packages/kernel_memory -> ../../../../src/kernel_memory` (recreate it with `ln -s` after
rebuilding the venv), and the demo scripts export `PYTHONPATH=$HERE/src`.
