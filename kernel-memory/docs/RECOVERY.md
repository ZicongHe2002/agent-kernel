# Recovery, integrity, and consistency operations

This guide covers the authoritative store (`MemoryStore`), what can go wrong, and how each
situation is repaired without ever deleting published experimental facts.

## What is authoritative

| Path | Role |
|---|---|
| `kernels/**/…json` (except views) | Authoritative records. Immutable once published. |
| `artifacts/sha256/<p>/<digest>` | Content-addressed evidence bytes. |
| `artifacts/registry/*.json` | Artifact descriptors by `artifact_id`. |
| `requests/**` | Request ledger and job events (append-only facts). |
| `journal/records.jsonl` | Publication journal: `(record_id, relpath, canonical digest, txn)`. Integrity evidence. |
| `.runtime/` | Lock, pending manifests, staging, sealed manifests, leases. Not facts. |
| `.cache/index.sqlite` | Disposable index. Delete freely; rebuild with `kmem reindex`. |
| `…/trajectory.json`, `…/memory_records.jsonl` | Generated views. Delete freely; rebuild with `kmem trajectory --rebuild`. |

## Publication protocol (why partial writes cannot corrupt history)

1. Validate every record (schema, references, artifact-descriptor consistency).
2. Stage each record to `.runtime/staging/<txn>/` with `fsync`.
3. Write a checksummed pending manifest `.runtime/pending/<txn>.json`.
4. For each record: `rename` into place, `fsync` the directory, append to the journal.
5. Remove the manifest (completion marker), then the staging directory.

A crash between steps leaves either nothing visible (before step 4) or a pending manifest that
recovery can finish (during step 4). The same id with identical content is idempotent; the same id
with different content is `ID_CONFLICT` (exit 3) and nothing is written.

## `kmem recover --root PATH`

Runs `MemoryStore.recover()` under the store lock and prints an audit report:

* `completed_txns`: pending manifests whose staged files were moved into place and journaled.
* `sealed_txns`: manifests that cannot be completed (staged file missing/corrupt, checksum mismatch).
  They are moved to `.runtime/sealed/` with a `*.report.json`. Records already published by that
  transaction remain; nothing is deleted.
* `records_completed`, `journal_repaired`: files that were in place but not journaled get journal entries.
* `temp_files_removed`: `.<name>.tmp-*` leftovers (never published) are removed.

`MemoryStore.open()` runs recovery automatically when pending manifests or stray temp files exist.
Recovery is idempotent: running it twice changes nothing the second time.

### Run published, job state not updated (T25)

If a Run `run-<request>-a<n>` exists but the request ledger still shows `claimed`/`running`,
`RequestLedger.reconcile()` (invoked by `kmem recover`) appends the terminal `finished` event pointing
at the existing run. The measurement is not repeated. A genuine re-measurement needs a new request
(`kmem run` with a new request id) or a new attempt number.

### Late result from an expired worker (T26)

Every claim carries a monotonically increasing fencing token. A worker finishing with a stale token
gets `LEASE_LOST`; its payload is appended as a `late_result_quarantined` event (evidence), and no Run
is published or overwritten.

## `kmem validate --root PATH --deep`

Deep validation = schema + cross-record invariants + hash recomputation + sample/summary consistency +
artifact presence/digests + the integrity scan:

| Finding | Meaning | Action |
|---|---|---|
| `modified` | File digest differs from the journal | Hand-edited history. Restore from backup or create a superseding Annotation/Decision; never accept the edit silently. |
| `missing` | Journaled record file absent | Restore from backup. Views/index are unaffected but the fact is lost until restored. |
| `corrupt` | Unparseable/invalid record file | Restore from backup. |
| `unjournaled` | Record file with no journal entry | Usually a manual copy; `kmem recover` journals it if valid. |
| `duplicate_ids` | Same record id in two files | Manual investigation; one is a manual copy. |
| `artifact_problems` | Missing/corrupt/size-mismatched evidence | Confirmation decisions that depend on it are blocked (`MISSING_EVIDENCE`). Re-import the artifact bytes if available. |

## Locks and concurrency

One `fcntl` lock (`.runtime/lock`) guards mutation, recovery, and consistent reads on one machine.
Two processes may publish concurrently and serialize on the lock. Multiple hosts writing one directory
over a network filesystem are unsupported. A lock wait beyond the timeout raises `LOCK_TIMEOUT` (exit 6).

## Backups

Copy the whole store directory while holding no writer (or after `kmem recover`). Everything needed to
rebuild views and indexes is inside `kernels/`, `artifacts/`, `requests/`, and `journal/`.
