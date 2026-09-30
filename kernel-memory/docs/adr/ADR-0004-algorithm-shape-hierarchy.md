# ADR-0004: Algorithm level, shapes, and a kernel-wide trajectory directory

**Status:** accepted · **Date:** 2026-09-28 · **Supersedes:** the per-config trajectory placement of ADR-0003 §4;
amends ADR-0001 ("contracts copied verbatim")

## Context

The user's corrected whiteboard (2026-09-28) defines the Memory hierarchy as:

```text
kernel/
├── trajectory/                 # generated view for the whole kernel
├── algorithm_1/
│   ├── method_summary
│   ├── shape_1/  { attempt/ (PR, commit), code/ (optional), result/ (performance, HLO, LLO, ...), ... }
│   ├── shape_2/
│   └── shape_3/
├── algorithm_2/
└── algorithm_3/
```

The handoff specification (`kernel_memory_ai_handoff/docs/IMPLEMENTATION.en.md`, read-only input) fixes its own
precedence rule in §0.1, line 19: *"Precedence: subsequent explicit user instructions → the latest confirmed
hierarchy → this specification and its companion contracts → the historical PDF."* The whiteboard is a subsequent
explicit user instruction, so it outranks the §2 tree (`name → config → {trajectory, attempt}`) and the §4 layout
that placed `trajectory.json` / `memory_records.jsonl` under `configs/<config-slug>/`. The rule also says a prose /
schema conflict must be reported and repaired with regression tests rather than chosen silently; this ADR is that
report.

## Decision

1. **Hierarchy.** `kernel → algorithm → shape (= config record) → attempt (PR → commit → run)`. A new record type
   `algorithm` carries `kernel_id`, `algorithm_id`, `display_name`, `method_summary` (authored text, data — never a
   measurement), `summary_author` (`human | agent | program`) and `tags`. `config.payload.algorithm_ref` is required.
   The record type keeps the name `config` in code and contracts; "shape" is the user-facing word.
2. **Identity unchanged where it matters.** `config_hash` still hashes only the problem (§5.2: *config says what to
   compute*); the algorithm is a coarse "how", like tiles. The same shape under two algorithms is therefore two
   config records with the *same* `config_hash` and `comparison_key`; that is what makes cross-algorithm comparison
   honest. New config ids are `cfg-<algorithm_id>-<config_id_hint>-<hash12>`; legacy ids are never rewritten.
   T01 (same normalized problem registered twice → same record) holds per algorithm. Decisions remain single-config
   (`DECISION_CONFIG_MISMATCH` unchanged); a cross-algorithm "best method" is only ever a generated table in the
   kernel trajectory, never a Decision.
3. **Physical layout (layout_version 2)** follows the whiteboard literally: `kernels/<kernel>/<algorithm-slug>/
   <config-slug>/…` with `kernel.json`, `annotations/` and `trajectory/` as reserved siblings of the algorithm
   directories (`RESERVED_SLUG` rejects algorithm ids whose slug is `trajectory`, `annotations`, `kernel.json`, and
   config ids whose slug is `annotations`, `algorithm.json`). `attempt/`, `baselines/`, `relations/`, `decisions/`,
   `annotations/` under a shape are unchanged.
4. **Trajectory at the kernel level.** All generated views live under `kernels/<kernel>/trajectory/`:
   `trajectory.json` (`kernel-trajectory-v1`: per-algorithm shape summaries plus a per-shape cross-algorithm table
   keyed by `(config_hash, comparison_key, policy_hash)`), `memory_records.jsonl` (kernel, algorithms, every shape),
   and `shapes/<config-slug>/{trajectory.json, memory_records.jsonl}` (the per-shape `trajectory-v2` view, which
   gains `algorithm_ref` / `algorithm_id` and two generated sections realising the board's `code` and `result`).
   Nothing under `trajectory/` is a record; `is_record_file` excludes the subtree by directory. Views never merge
   measurements across shapes and derive no numbers.
5. **`code/` and `result/` are generated sections, not moved records.** `code` = the commit's content-addressed diff
   artifact plus every run's source/variant digests; `result` = every run's execution status, correctness, timing,
   analysis metrics (nulls stay null), artifact descriptors, HLO artifact slots and the LLO status (`unsupported`).
   The JAX adapter additionally captures real `Lowered.as_text()` (StableHLO) and `Compiled.as_text()` (optimized
   HLO) as `stablehlo_text` / `compiled_hlo_text` text artifacts when JAX returns them, labelled by the backend that
   actually compiled; absence is recorded, never invented. No LLO parser exists.
6. **Contract versioning.** `contracts/record.schema.json` is now the project's `0.3.0` contract. The verbatim
   handoff contract is retained byte-for-byte as `contracts/legacy/record.schema.v0.2.0.json` (sha256
   `e24f4cd5bc52f6f0b8b8c40ee123f2760e3a1371bae9e15b981c16e092ecbf77`) and is used only to validate legacy input.
   `demo_problem.schema.json`, `hash_vectors.json` and the fixture bundle stay verbatim; the handoff directory is
   untouched and its `MANIFEST.sha256` keeps verifying.
7. **Legacy data is upgraded, never edited.** A v0.2 bundle is validated against the legacy contract, then upgraded in
   memory (`migrations.v02.upgrade_v02_records`): every record gets `schema_version 0.3.0`, every config gets
   `algorithm_ref`, and one placeholder algorithm per kernel is appended — `algorithm-<kernel_id>-unspecified`,
   `method_summary` literally `"unspecified (imported from v0.2)"`, `summary_author "program"`, tag `imported-v02`,
   `created_at` copied from the kernel record so the upgrade is deterministic and re-imports stay idempotent. The
   placeholder states that no method was described; it is never presented as knowledge. A layout-1 store is refused
   on open and migrated into a *new* root with `kmem migrate-v02` (export → upgrade → import; provenance preserved as
   stored, because a store-to-store move of already-admitted facts is not an untrusted import).

## What the whiteboard overrides, and what stays

| Spec statement | Status |
|---|---|
| §2 tree: trajectory and attempt under `config` | overridden: trajectory under the kernel; attempt under the shape |
| §4 layout: `configs/<config-slug>/trajectory.json`, `memory_records.jsonl` | overridden: `kernels/<k>/trajectory/…` |
| §0.1 U1 naming (kernel → config) | extended with the algorithm level; config = shape |
| §5.2 `config_hash` identity | unchanged |
| T01–T03, T24 acceptance | unchanged in meaning; re-mapped under the new layout (`docs/ACCEPTANCE.md`) |
| Immutability, provenance gates, fixture exclusion, null metrics | unchanged |

## Deferred (documented, not decided here)

* `pr-gh-<repo>-pr-<n>` and `baseline-<id>` ids are global, so one GitHub PR cannot be attached to two shapes; a
  per-shape scoping would change existing ids.
* Cross-algorithm Decisions; a SQLite `algorithm_ref` column; `execution/*` changes (the coordinator only needs
  `config_ref` / `config_hash`).

## Consequences

* Existing `.demo/` stores are layout 1 and must be regenerated or migrated with `kmem migrate-v02`.
* `kmem register-config` needs an algorithm: `--algorithm` (optional when the kernel has exactly one).
* `kmem trajectory` takes `--kernel` or `--config`; the CLI, `docs/DESIGN.md`, `RECOVERY.md`, `MIGRATION.md`,
  `CONFIGURATION.md`, both READMEs and the evidence logs describe the new layout.
