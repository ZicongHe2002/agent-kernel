# Migrating v0.1 data to v0.2

Code: `kernel_memory.migrations.v01` (`read_v01_source`, `migrate_v01`, `MigrationReport`,
`NullResolver`, `MappingResolver`); CLI: `kmem migrate-v01`. Specification: section 20 of
`../../kernel_memory_ai_handoff/docs/IMPLEMENTATION.en.md`; acceptance scenario T31.
Tests: `tests/test_migration_v01.py` (synthetic input only).

## 1. Purpose and scope

The migration maps **real, user-supplied v0.1 data** into v0.2 records without inventing anything
the old data does not contain. It is dry-run by default and never modifies the original files.

Two boundaries are binding:

* The historical PDF that describes the v0.1 design is a **design example, not a dataset**. Its
  tables and numbers are not imported by anyone.
* **No real v0.1 export exists in this workspace.** The reader below was written from the concepts
  named in specification section 20 and is deliberately tolerant. It **must be confirmed against
  the real export** (field names, shapes, identifier formats) before any `--apply` run. Every
  report carries this reminder as its first note.

## 2. Expected v0.1 input structure

`read_v01_source(path)` accepts:

* a single `.json`, `.yaml` or `.yml` file whose top-level object carries any subset of the keys
  `kernels`, `configs`, `attempts`, `revisions`, `results`, `trajectory`;
* a directory (top level only, no recursion) of such files; list-valued keys from several files are
  concatenated, object-valued keys are merged, and the same scalar/sub-key defined twice is an
  error (`V01_SOURCE_CONFLICT`, exit 2);
* a file named after a top-level key (`kernels.json`, `revisions.yaml`, `trajectory.json`, ...)
  whose whole content is that key's value.

Each concept is a list of objects (an object keyed by id is also accepted and converted; the
conversion is reported as a note). The fields the migration understands:

| Key | Entry id | Fields read |
|---|---|---|
| `kernels` | `kernel_id` | `kernel_id`, `display_name`, `adapter_id` |
| `configs` | `config_id` | `config_id`, `kernel_id`, `problem` (object) |
| `attempts` | `attempt_id` | `attempt_id`, `kernel_id`, `config_id`, `pr_number`, `repo`, `title`, `hypothesis`, `parent_attempt_id`, `selected_revision` |
| `revisions` | `revision_id` | `revision_id`, `attempt_id`, `commit_sha`, `parent_sha`, `changes[]` (`change_id`, `component`, `key`, `before`, `after`, `rationale`, `extraction_source`, `attribution`), `summary` |
| `results` | `revision_id` | `revision_id`, `status`, `latency_us`, `speedup`, `spill_bytes`, `excessive_spill`, `environment`, `notes` |
| `trajectory` | — | ignored entirely: a derived view, rebuilt from migrated facts |

Anything else is **reported, never silently dropped**: unknown top-level keys and unknown per-entry
fields appear in `discarded_fields` with the reason `unknown v0.1 field; not migrated`. Files are
parsed strictly (duplicate keys, NaN, custom YAML tags are rejected); symlinked directory entries
are refused (`UnsafePathError`, exit 7); files above the JSON size limit are refused (exit 2).

A minimal synthetic example in the accepted shape (fictional identifiers; see
`tests/test_migration_v01.py::synthetic_v01` for the full one used by the tests):

```json
{
  "kernels":   [{"kernel_id": "demo_vector_add", "display_name": "vector add", "adapter_id": "legacy-cpu"}],
  "configs":   [{"config_id": "cfg-1", "kernel_id": "demo_vector_add", "problem": {"n": 16, "dtype": "f32"}}],
  "attempts":  [{"attempt_id": "att-1", "config_id": "cfg-1", "repo": "org/repo", "pr_number": 7,
                 "title": "tiling", "selected_revision": "rev-1"}],
  "revisions": [{"revision_id": "rev-1", "attempt_id": "att-1",
                 "commit_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", "summary": "first try"}],
  "results":   [{"revision_id": "rev-1", "status": "ok", "latency_us": 123.4, "spill_bytes": 65536}]
}
```

## 3. Mapping rules (specification section 20, as implemented)

| v0.1 concept | Spec section 20 mapping | Implemented rule |
|---|---|---|
| Kernel | kernel / name | `kernel` record `kernel-<kernel_id>`. `display_name` defaults to the id; `adapter_id` defaults to `unknown-v01` and is marked unverified in `contract_notes`. An existing store kernel with the same `kernel_id` is reused (T01) and the v0.1 display name / adapter are not applied. |
| Config | (config under the kernel) | The `problem` object is normalised through the **registered problem adapter** for that kernel; `config_hash` is recomputed, never copied. Record id `cfg-<config_id_hint>-<hash[:12]>`; tag `migrated-v01`. An existing store config with the same `config_hash` is reused (T01). A kernel without a complete adapter has all its configs **rejected** with `no problem adapter for kernel; cannot verify semantics` — `mla_forward` stays rejected until a real problem schema exists (see `CONFIGURATION.md` section 4). |
| Attempt (PR) | PR inside the attempt collection | `pr` record inside the config's attempt collection. `provider=github` with `number=pr_number` **only** when `repo` maps through `--repo-uid-map` to a `github:<host>:repo:<id>` repo_uid (record id `pr-gh-<repo_id>-pr-<number>`). Otherwise `provider=local`, `number=null`, `pr_key=local-<slug>-<hash>`; an unmapped `repo` becomes a placeholder `local:<slug>` repo_uid and the fact is reported in `discarded_fields`/`notes`. |
| Revision | Commit binding | `commit` record `commit-<pr_key>-<oid[:12]>` bound to the migrated PR. A full 40-hex (sha1) or 64-hex (sha256) `commit_sha` is accepted as given. A **short sha is resolved through the resolver or left unresolved** — never zero-padded, never guessed; an ambiguous prefix is also unresolved. An unresolved revision produces **no commit record** and is listed in `unresolved`. Non-hex or missing shas are rejections, not unresolved. `parent_sha` follows the same rules; an unresolved parent leaves `git_parent_oids` empty with a note. `source_available=false`; `changes` are validated, `attribution` is forced to `group_only`, `extraction_source` is `unknown` unless the export says `explicit`; missing `summary` is recorded empty with `summary_author=collector`. |
| `parent_attempt_id` | optimization-origin relationship; unresolved if the exact commit is unknown | `relation` (`kind=optimization_origin`) **only when both endpoints resolve to concrete commit records**: the parent's selected revision (or its single revision) and the child's selected (or single) revision. Otherwise `unresolved` with the reason `exact origin commit unknown: ...`. When only the parent endpoint resolves, `pr.origin_ref` on the child PR is still set and the relation is reported unresolved. |
| `selected_revision` | Decision with evidence/context/policy; missing conditions do not imply confirmation | **Never a Decision.** The conditions are always missing (no policy, no comparable runs, no evidence), so it becomes a `note` annotation on the selected commit (`author_kind=human`, `confidence=unverified`, text `v0.1 selected_revision=...; no policy/evidence-backed decision migrated`). Selected revisions that are not part of the attempt, or whose sha did not resolve, are `unresolved`. |
| `result.status` | separate execution and correctness when supported; otherwise retain unverified status | **Never a Run.** v0.2 Runs require environment/protocol/verifier snapshots, raw samples and tested-source identity, none of which v0.1 has. The whole result object is embedded **verbatim as JSON data** in one `note` annotation on the commit (`confidence=unverified`), together with per-field notes. `status` is never turned into `execution_status` or a correctness verdict; `latency_us`/`speedup` are never timing or derived comparisons. |
| `excessive_spill` | preserve original diagnosis; do not infer execution failure | Kept as the diagnosis label `v0.1 diagnosis: excessive_spill` inside the result annotation. Execution failure is not inferred (spill is not failure). |
| old `spill_bytes` | retain original value/source; unknown semantics cannot become precise HBM traffic | Retained verbatim with the note `semantics unknown; not HBM traffic`. It never becomes an analysis `Metric`. |
| old `trajectory.json` | rebuild from migrated authoritative facts | Discarded (`discarded_fields.top_level.trajectory`); rebuild with `kmem trajectory --config <cfg> --rebuild` after `--apply`. |

The report therefore contains `runs=0` and `decisions=0` by design; every v0.1 result and
selection is an `unverified` annotation whose text is data, not a fact about execution.

## 4. What is never inferred

* Full object ids from short shas (no padding, no prefix guessing, no "most likely" match).
* `execution_status`, correctness, timing summaries or comparisons from `result.status`,
  `latency_us` or `speedup`.
* Execution failure from `excessive_spill`, or HBM traffic from `spill_bytes`.
* A `config_hash` for a problem whose semantics no adapter can verify.
* A Decision, `is_production`, or best-known status from `selected_revision`.
* An `optimization_origin` relation whose endpoints are not both concrete commits.
* Environment/protocol/verifier snapshots from free-form `environment` descriptions.
* Isolated per-change attribution (`group_only` is forced; a combined result never proves
  individual effects).
* Any field the reader does not know: it is reported as discarded, never mapped by guess.

Text from the export (titles, hypotheses, summaries, notes, results) is stored as data and never
interpreted as instructions.

## 5. How to run

Dry run (default; opens no store, writes nothing, prints the full report):

```bash
.venv/bin/kmem migrate-v01 /path/to/v01-export --json
```

Apply into a store (the store must exist: `kmem --root R init`):

```bash
.venv/bin/kmem --root /path/to/memory migrate-v01 /path/to/v01-export --apply \
  --repo-uid-map '{"org/repo": "github:github.com:repo:42"}' \
  --resolver map:@short-shas.json --json
```

Flags:

| Flag | Meaning |
|---|---|
| `PATH` | v0.1 export: one JSON/YAML file or a directory of them (section 2). |
| `--apply` | Publish the produced records (`publish_bundle`, `allow_dangling=False`) into `--root`. Without it nothing is written. |
| `--repo-uid-map JSON \| @file.json` | v0.1 `repo` names → v0.2 `repo_uid`. Only `github:<host>:repo:<id>` values give GitHub PR identities; everything else is `local`. |
| `--resolver map:@file.json` | Offline short-sha table: keys `"<repo_uid>|<shortsha>"`, values full 40/64-hex ids. A prefix that matches two entries is ambiguous, hence unresolved. |
| `--resolver git:<repo path>` | Resolve through a local clone with `git rev-parse --verify` (`kernel_memory.adapters.git_local.LocalGitRepo`); an "ambiguous argument" failure is reported as ambiguous. |
| (no `--resolver`) | `NullResolver`: every short sha stays unresolved. This is the safe default. |

Python API, when you need to preview against an existing store or control the timestamp:

```python
from pathlib import Path
from kernel_memory.migrations.v01 import migrate_v01, MappingResolver
from kernel_memory.storage import MemoryStore

store = MemoryStore.open(Path("memory"))
report = migrate_v01(Path("v01-export"), store=store, dry_run=True,
                     repo_uid_map={"org/repo": "github:github.com:repo:42"},
                     resolver=MappingResolver({("github:github.com:repo:42", "beef12"): "beef12..."}),
                     created_at="2026-09-24T00:00:00Z")
print(report.record_counts(), report.unresolved, report.rejections)
```

With `store=` and `dry_run=True` the migration additionally reports T01 reuse of existing
kernels/configs and pre-checks `ID_CONFLICT` against published records, still without writing.
Note that the CLI dry run passes no store, so it cannot report reuse or conflicts.

## 6. Idempotency and conflicts

All produced records share one `created_at`. Migration annotations get a deterministic
`annotation-<16 hex>` id derived from the v0.1 identity; kernels, configs, PRs, commits and
relations use the deterministic ids of `DESIGN.md` section 4. Consequently:

* Re-running with the **same input and the same `created_at`** is idempotent: the store reports
  every record as `idempotent`, counts do not change.
* A **changed input** (or a different `created_at`) that maps to an already published id raises
  `ID_CONFLICT` (exit 3) before anything is written; published records are immutable, so nothing is
  overwritten and nothing is partially applied. This is the conflict guard working, not data loss.
* The CLI does not expose `created_at` (it uses the current time), so a second `--apply` of the
  same export from the CLI fails with `ID_CONFLICT` once the first has succeeded. Treat that as
  "already migrated" and verify with `kmem status`; use the Python API with an explicit
  `created_at` when you need a repeatable apply.
* Records with missing references are refused (`MISSING_REFERENCE`, exit 3); duplicate v0.1 ids
  keep the first occurrence and reject the rest with a reason.

## 7. The report

`MigrationReport.to_dict()` (what `--json` prints):

| Field | Content |
|---|---|
| `source_path`, `source_file_digests` | Input path and `sha256:<hex>` of the **original bytes** of every input file. |
| `dry_run` | Whether anything could have been written. |
| `identity_map` | One row per v0.1 entity: `v01_kind`, `v01_id`, `v02_record_type`, `v02_record_id` (null when not produced), `status` (`mapped` / `unresolved` / `rejected`). |
| `retained_fields` | Per concept: which v0.1 field went where (e.g. `results.spill_bytes` → annotation text with the "not HBM traffic" note). |
| `discarded_fields` | Per concept: which fields were not migrated and why (unknown fields, `trajectory`, unmapped `repo`, `pr_number` without a GitHub repo_uid, dropped malformed changes). |
| `unresolved` | Short shas, parents, selected revisions, results and parent links that could not be tied to a concrete commit, each with a reason. |
| `rejections` | Entries that were refused (missing/invalid ids, unknown kernel or config, no problem adapter, non-hex sha, type errors), each with a reason. |
| `records` / `record_counts` | The produced v0.2 record dicts and their counts by type (`run` and `decision` are never present). |
| `annotations_for_unverified` | Number of unverified annotations created for results and selections. |
| `notes` | Human-readable remarks: the tolerant-reader reminder, defaults applied, T01 reuse, placeholder repo_uids, what was published. |
| `publish_outcome` | `null` on a dry run; otherwise the store's publish outcome (published vs idempotent ids). |

## 8. Artifact retention changed between v0.1 and v0.2

The old policy treated raw timing samples as optional. v0.2 (specification section 13) retains
structured results, lightweight timing samples, correctness reports and key decision evidence
**permanently by default** (`retention=permanent`), allows tiered retention only for large
profiles/IR/LLO dumps (`retain_for_decision`, `expiring`), and keeps tombstones
(`availability=expired`) so that missing evidence is never confused with evidence that was never
collected. Consequences for migrated data:

* v0.1 results carry no samples, so they cannot satisfy this policy and are imported as unverified
  annotations, not as Runs (sections 3 and 4). Nothing about them is "expired"; it was never collected.
* Every measurement made after migration goes through `kmem run` (or a bundle import with explicit
  provenance) and is stored with content-addressed artifacts under the new retention classes.
* `kmem validate --deep` reports absent or corrupt evidence as `artifact_problems`; a Decision
  depending on it is blocked (`MISSING_EVIDENCE`) rather than silently degraded.

## 9. Verify before and after

1. Digest the originals and keep the values: `shasum -a 256 /path/to/v01-export/*` must equal the
   report's `source_file_digests`. Re-check after the run; the migration never writes to `PATH`.
2. Dry-run first and read `unresolved` and `rejections` until every entry is understood. Supply a
   resolver or a repo_uid map where the data supports it; do not "fix" the export to make numbers
   fit.
3. Record the store state: `kmem --root R status --json` (record counts by type) and
   `kmem --root R validate --deep --json` (must report `ok`).
4. `--apply`, then repeat step 3. The count delta per type must equal the dry run's
   `record_counts`, and deep validation must still be `ok`.
5. Rebuild derived views: `kmem --root R trajectory --config <cfg> --rebuild --json`, then
   `--verify`. The old `trajectory.json` is not consulted.
6. Optional: re-run the migration from the Python API with the same `created_at`; the publish
   outcome must list every record as idempotent and the counts must be unchanged.

Fixture behaviour is unaffected by migration: migrated records carry no `provenance`, and only
Runs carry one. Migrated data therefore never becomes a production confirmed-best by itself.
