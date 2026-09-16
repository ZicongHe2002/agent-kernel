# ADR-0003: Readable JSON is the only truth; slugged layout; journal for integrity

**Status:** accepted · **Date:** 2026-09-08

## Decisions

1. **JSON, not YAML.** Authoritative records are `*.json` written with stable indentation and sorted keys. The
   canonical identity of a record is its RFC 8785 digest, independent of formatting. YAML input (optional) is
   normalised to the same JSON value; it is never a second source of truth.
2. **Directory names are slugs of record IDs** (`ids.slug_for_id`), lower-cased and suffixed with a short hash
   when the id is not already a safe lower-case slug. This keeps case-insensitive filesystems (macOS) safe and
   keeps identities in the files, not the paths.
3. **Publication protocol.** Stage (temp file + fsync) → checksummed pending manifest → atomic rename →
   directory sync → journal append (`journal/records.jsonl`) → manifest completion. Same id + same content is
   idempotent; different content is `ID_CONFLICT`. Published records are immutable; the CLI never edits them.
4. **Recovery** completes or seals pending manifests idempotently, removes stray temp files (never published),
   repairs journal gaps, and never deletes published facts. `MemoryStore.open()` runs recovery under the lock when
   pending work exists.
5. **Integrity scan** compares files against the journal: modified (digest mismatch), missing (journaled but
   absent), corrupt (unparseable/invalid), unjournaled (file without journal entry), duplicate ids, artifact
   problems. Hand-edited history is detected, not silently accepted.
6. **SQLite index is disposable.** `.cache/index.sqlite` is rebuilt from JSON; `SqliteIndex.is_fresh` compares a
   fingerprint of all record digests. Nothing authoritative is written to SQLite.
7. **Single-machine, single coordinating writer.** One `fcntl.flock` protocol for mutation, recovery, and
   consistent reads. Network filesystems and multi-host writers are unsupported by design.
