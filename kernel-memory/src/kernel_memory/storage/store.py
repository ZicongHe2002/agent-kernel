"""Authoritative JSON store with atomic publication, idempotency, recovery, and artifacts.

Guarantees (specification sections 4, 13, 15):

* Readable JSON files are the only source of truth. Views and the SQLite index are derived.
* Publication: validate -> stage (temp file + fsync) -> checksummed pending manifest ->
  atomic rename into place -> directory sync -> journal append -> manifest completion.
* Same record ID with identical canonical content is idempotent; different content is an
  ``IdConflictError``. Published records are immutable; the integrity scan detects edits.
* Recovery completes or seals pending manifests idempotently and never deletes published
  facts. Stray temp files (never published) are removed.
* Artifacts are content-addressed by SHA-256 of the original bytes; descriptors are kept
  in an artifact registry keyed by artifact_id so that commit ``diff_artifact_ref`` and run
  evidence references resolve consistently (ADR-0002).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator

from ..domain.errors import (
    ConflictError,
    IdConflictError,
    InputError,
    InvariantViolation,
    MissingReferenceError,
    UnsafePathError,
)
from ..domain.hashing import artifact_digest, is_sha256_ref, sha256_bytes
from ..domain.ids import fsync_directory, resolve_inside, utc_now_iso, validate_record_id
from ..domain.jsonio import dumps_compact, dumps_readable, load_json_file, loads_strict
from ..domain.models import TYPE_ORDER, ArtifactRef, Record, to_json
from ..domain.schema import SCHEMA_VERSION, validate_nested
from . import layout
from .lock import StoreLock

STORE_VERSION = "0.2.0"
LAYOUT_VERSION = 1
DEFAULT_MAX_ARTIFACT_BYTES = 200_000_000


@dataclass(frozen=True)
class IndexEntry:
    record_id: str
    record_type: str
    relpath: str
    digest: str


@dataclass
class PublishOutcome:
    txn_id: str
    published: list[str] = field(default_factory=list)
    idempotent: list[str] = field(default_factory=list)
    artifacts_registered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "published": list(self.published),
            "idempotent": list(self.idempotent),
            "artifacts_registered": list(self.artifacts_registered),
        }


@dataclass
class RecoveryReport:
    completed_txns: list[str] = field(default_factory=list)
    sealed_txns: list[dict[str, Any]] = field(default_factory=list)
    records_completed: list[str] = field(default_factory=list)
    journal_repaired: list[str] = field(default_factory=list)
    temp_files_removed: list[str] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_txns": self.completed_txns,
            "sealed_txns": self.sealed_txns,
            "records_completed": self.records_completed,
            "journal_repaired": self.journal_repaired,
            "temp_files_removed": self.temp_files_removed,
            "problems": self.problems,
        }


@dataclass
class IntegrityReport:
    records_checked: int = 0
    artifacts_checked: int = 0
    modified: list[dict[str, Any]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    corrupt: list[dict[str, Any]] = field(default_factory=list)
    unjournaled: list[dict[str, Any]] = field(default_factory=list)
    duplicate_ids: list[dict[str, Any]] = field(default_factory=list)
    artifact_problems: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.modified or self.missing or self.corrupt or self.unjournaled or self.duplicate_ids or self.artifact_problems
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "records_checked": self.records_checked,
            "artifacts_checked": self.artifacts_checked,
            "modified": self.modified,
            "missing": self.missing,
            "corrupt": self.corrupt,
            "unjournaled": self.unjournaled,
            "duplicate_ids": self.duplicate_ids,
            "artifact_problems": self.artifact_problems,
        }


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._lock = StoreLock(self.root / layout.RUNTIME_DIR / "lock")
        self._index: dict[str, IndexEntry] | None = None
        self._kernel_index: dict[str, str] = {}
        self._record_cache: dict[str, Record] = {}

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def init(cls, root: Path) -> "MemoryStore":
        root = Path(root)
        manifest = root / layout.MANIFEST_FILE
        if root.exists():
            if manifest.exists():
                store = cls(root)
                store._check_manifest()
                return store
            entries = [p for p in root.iterdir() if p.name != ".DS_Store"]
            if entries:
                raise InputError(
                    f"refusing to initialise a store inside non-empty directory {root} (no manifest.json present)",
                    code="ROOT_NOT_EMPTY",
                    details={"root": str(root), "entries": sorted(p.name for p in entries)[:20]},
                )
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root)
        for sub in (layout.KERNELS_DIR, layout.ARTIFACTS_DIR, layout.REQUESTS_DIR, layout.JOURNAL_DIR, layout.RUNTIME_DIR):
            (store.root / sub).mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "store_version": STORE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "layout_version": LAYOUT_VERSION,
            "hash_version": "jcs-sha256-v1",
            "store_id": uuid.uuid4().hex,
            "created_at": utc_now_iso(),
            "notes": "Readable JSON files under kernels/ are authoritative. trajectory.json, memory_records.jsonl and .cache/ are derived and rebuildable.",
        }
        store._atomic_write(store.root / layout.MANIFEST_FILE, dumps_readable(manifest_data).encode("utf-8"), overwrite=False)
        return store

    @classmethod
    def open(cls, root: Path, *, recover: bool = True) -> "MemoryStore":
        store = cls(root)
        store._check_manifest()
        if recover:
            with store.lock():
                if store._pending_manifests():
                    store.recover()
        return store

    def _check_manifest(self) -> dict[str, Any]:
        manifest = self.root / layout.MANIFEST_FILE
        if not manifest.is_file():
            raise InputError(f"{self.root} is not a kernel-memory store (manifest.json missing)", code="NOT_A_STORE")
        data = load_json_file(manifest)
        if not isinstance(data, dict) or data.get("layout_version") != LAYOUT_VERSION:
            raise InputError(
                f"unsupported store layout version in {manifest}: {data.get('layout_version') if isinstance(data, dict) else data!r}",
                code="UNSUPPORTED_STORE",
            )
        return data

    def manifest(self) -> dict[str, Any]:
        return self._check_manifest()

    def lock(self) -> StoreLock:
        return self._lock

    # ------------------------------------------------------------------ paths
    def _abs(self, relpath: PurePosixPath | str) -> Path:
        return resolve_inside(self.root, str(relpath))

    def _runtime(self, *parts: str) -> Path:
        return self.root.joinpath(layout.RUNTIME_DIR, *parts)

    def _journal_path(self) -> Path:
        return self.root / layout.JOURNAL_DIR / "records.jsonl"

    # ------------------------------------------------------------------ low-level atomic writes
    def _atomic_write(self, dest: Path, data: bytes, *, overwrite: bool) -> str:
        """Write bytes atomically. Returns 'written', 'identical', or raises ConflictError."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            existing = dest.read_bytes()
            if existing == data:
                return "identical"
            if not overwrite:
                raise ConflictError(f"destination exists with different content: {dest.relative_to(self.root)}", code="FILE_CONFLICT")
        tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
        fsync_directory(dest.parent)
        return "written"

    # ------------------------------------------------------------------ index
    def _ensure_index(self) -> dict[str, IndexEntry]:
        if self._index is None:
            self._index, self._kernel_index, self._record_cache = self._scan_records()
        return self._index

    def invalidate_index(self) -> None:
        self._index = None
        self._kernel_index = {}
        self._record_cache = {}

    def _scan_records(self) -> tuple[dict[str, IndexEntry], dict[str, str], dict[str, Record]]:
        index: dict[str, IndexEntry] = {}
        kernels: dict[str, str] = {}
        cache: dict[str, Record] = {}
        kernels_root = self.root / layout.KERNELS_DIR
        if not kernels_root.exists():
            return index, kernels, cache
        for path in sorted(kernels_root.rglob("*.json")):
            rel = PurePosixPath(path.relative_to(self.root).as_posix())
            if not layout.is_record_file(rel):
                continue
            try:
                record = Record.from_dict(load_json_file(path))
            except Exception as exc:  # corrupt files are reported by integrity_scan, skipped here
                raise InvariantViolation(
                    f"corrupt or invalid record file {rel}: {exc}", code="CORRUPT_RECORD_FILE", details={"path": str(rel)}
                ) from exc
            if record.record_id in index:
                raise InvariantViolation(
                    f"record id {record.record_id!r} appears in two files: {index[record.record_id].relpath} and {rel}",
                    code="DUPLICATE_RECORD_FILE",
                )
            index[record.record_id] = IndexEntry(record.record_id, record.record_type, str(rel), record.canonical_digest())
            cache[record.record_id] = record
            if record.record_type == "kernel":
                kernels[record.payload.kernel_id] = record.record_id
        return index, kernels, cache

    def index_entries(self) -> list[IndexEntry]:
        return sorted(self._ensure_index().values(), key=lambda e: (TYPE_ORDER[e.record_type], e.record_id))

    # ------------------------------------------------------------------ reading
    def get(self, record_id: str) -> Record | None:
        index = self._ensure_index()
        entry = index.get(record_id)
        if entry is None:
            return None
        cached = self._record_cache.get(record_id)
        if cached is not None:
            return cached
        record = Record.from_dict(load_json_file(self.root / entry.relpath))
        self._record_cache[record_id] = record
        return record

    def exists(self, record_id: str) -> bool:
        return record_id in self._ensure_index()

    def require(self, record_id: str, *allowed_types: str) -> Record:
        record = self.get(record_id)
        if record is None:
            raise MissingReferenceError(f"record {record_id!r} not found", details={"record_id": record_id})
        if allowed_types and record.record_type not in allowed_types:
            raise MissingReferenceError(
                f"record {record_id!r} has type {record.record_type!r}, expected {list(allowed_types)}",
                details={"record_id": record_id, "record_type": record.record_type},
            )
        return record

    def iter_records(self, record_type: str | None = None) -> Iterator[Record]:
        for entry in self.index_entries():
            if record_type is None or entry.record_type == record_type:
                record = self.get(entry.record_id)
                if record is not None:
                    yield record

    def records(self, record_type: str | None = None) -> list[Record]:
        return list(self.iter_records(record_type))

    def kernel_by_kernel_id(self, kernel_id: str) -> Record | None:
        self._ensure_index()
        rid = self._kernel_index.get(kernel_id)
        return self.get(rid) if rid else None

    def configs_by_hash(self, config_hash: str) -> list[Record]:
        return [r for r in self.iter_records("config") if r.payload.config_hash == config_hash]

    def record_path(self, record_id: str) -> Path:
        entry = self._ensure_index().get(record_id)
        if entry is None:
            raise MissingReferenceError(f"record {record_id!r} not found")
        return self.root / entry.relpath

    def config_dir(self, config_record_id: str) -> Path:
        config = self.require(config_record_id, "config")
        rel = layout.config_dir_for(config, self.get, self.kernel_by_kernel_id)
        return self.root / rel

    # ------------------------------------------------------------------ publication
    def publish(self, record: Record, *, label: str | None = None) -> PublishOutcome:
        return self.publish_bundle([record], label=label)

    def publish_bundle(self, records: Iterable[Record], *, label: str | None = None, allow_dangling: bool = False) -> PublishOutcome:
        records = list(records)
        for record in records:
            if not isinstance(record, Record):
                raise InputError("publish_bundle expects Record instances (validate first)")
        txn_id = f"txn-{uuid.uuid4().hex}"
        outcome = PublishOutcome(txn_id=txn_id)
        with self.lock():
            self.recover_if_pending()
            index = self._ensure_index()
            # 1. Collapse identical duplicates, reject conflicting duplicates and stored conflicts.
            by_id: dict[str, Record] = {}
            for record in records:
                validate_record_id(record.record_id)
                digest = record.canonical_digest()
                if record.record_id in by_id:
                    if by_id[record.record_id].canonical_digest() != digest:
                        raise IdConflictError(
                            f"bundle contains record id {record.record_id!r} twice with different content",
                            details={"record_id": record.record_id},
                        )
                    continue
                by_id[record.record_id] = record
            new_records: dict[str, Record] = {}
            for rid, record in by_id.items():
                existing = index.get(rid)
                if existing is not None:
                    if existing.digest != record.canonical_digest():
                        raise IdConflictError(
                            f"record {rid!r} already exists with different content; published records are immutable",
                            details={"record_id": rid, "existing_digest": existing.digest, "new_digest": record.canonical_digest()},
                        )
                    outcome.idempotent.append(rid)
                else:
                    new_records[rid] = record

            def resolver(record_id: str) -> Record | None:
                return new_records.get(record_id) or self.get(record_id)

            def kernel_resolver(kernel_id: str) -> Record | None:
                for rec in new_records.values():
                    if rec.record_type == "kernel" and rec.payload.kernel_id == kernel_id:
                        return rec
                return self.kernel_by_kernel_id(kernel_id)

            # 2. Reference checks and dependency ordering.
            ordered = self._dependency_order(new_records, resolver, allow_dangling=allow_dangling)
            # 3. Artifact registry consistency (before writing anything).
            pending_artifact_refs: dict[str, ArtifactRef] = {}
            for record in ordered:
                if record.record_type == "run":
                    for ref in record.payload.artifacts:
                        self._check_artifact_ref(ref, pending_artifact_refs)
                        pending_artifact_refs[ref.artifact_id] = ref
            # 4. Compute destinations in dependency order.
            plan: list[tuple[Record, PurePosixPath, str]] = []
            for record in ordered:
                rel = layout.record_relpath(record, resolver, kernel_resolver)
                dest = self._abs(rel)
                if dest.exists():
                    raise IdConflictError(
                        f"destination {rel} already exists for a different record id", details={"path": str(rel)}
                    )
                plan.append((record, rel, record.canonical_digest()))
            if not plan:
                for ref in pending_artifact_refs.values():
                    if self.register_artifact_ref(ref):
                        outcome.artifacts_registered.append(ref.artifact_id)
                return outcome
            # 5. Stage files and write the checksummed pending manifest.
            staging = self._runtime("staging", txn_id)
            staging.mkdir(parents=True, exist_ok=True)
            entries: list[dict[str, Any]] = []
            for n, (record, rel, digest) in enumerate(plan):
                staged = staging / f"{n:06d}.json"
                self._atomic_write(staged, dumps_readable(record.to_dict()).encode("utf-8"), overwrite=True)
                entries.append(
                    {"record_id": record.record_id, "record_type": record.record_type, "relpath": str(rel), "digest": digest, "staged": staged.name}
                )
            manifest = {"txn_id": txn_id, "created_at": utc_now_iso(), "label": label, "records": entries}
            manifest["manifest_digest"] = sha256_bytes(dumps_compact(manifest).encode("utf-8"))
            pending_dir = self._runtime("pending")
            pending_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(pending_dir / f"{txn_id}.json", dumps_readable(manifest).encode("utf-8"), overwrite=False)
            # 6. Commit: rename into place, journal, then complete the manifest.
            self._apply_manifest(manifest, staging, outcome.published)
            for ref in pending_artifact_refs.values():
                if self.register_artifact_ref(ref):
                    outcome.artifacts_registered.append(ref.artifact_id)
            self._complete_manifest(txn_id)
        return outcome

    def _dependency_order(self, new_records: dict[str, Record], resolver: Callable[[str], Record | None], *, allow_dangling: bool) -> list[Record]:
        # Kahn's algorithm over intra-bundle references, deterministic tie-break by (type order, id).
        deps: dict[str, set[str]] = {rid: set() for rid in new_records}
        for rid, record in new_records.items():
            for ref in record.record_references():
                target = ref.target
                if target in new_records:
                    if new_records[target].record_type not in ref.allowed_types:
                        raise MissingReferenceError(
                            f"{record.record_type} {rid!r} field {ref.field} refers to {target!r} of type "
                            f"{new_records[target].record_type!r}; expected {list(ref.allowed_types)}"
                        )
                    deps[rid].add(target)
                else:
                    existing = resolver(target)
                    if existing is None:
                        if allow_dangling:
                            continue
                        raise MissingReferenceError(
                            f"{record.record_type} {rid!r} field {ref.field} refers to missing record {target!r}",
                            details={"record_id": rid, "field": ref.field, "target": target},
                        )
                    if existing.record_type not in ref.allowed_types:
                        raise MissingReferenceError(
                            f"{record.record_type} {rid!r} field {ref.field} refers to {target!r} of type "
                            f"{existing.record_type!r}; expected {list(ref.allowed_types)}"
                        )
            if record.record_type == "config":
                for other_id, other in new_records.items():
                    if other.record_type == "kernel" and other.payload.kernel_id == record.payload.kernel_id:
                        deps[rid].add(other_id)
        ordered: list[Record] = []
        remaining = dict(deps)
        while remaining:
            ready = sorted(
                (rid for rid, d in remaining.items() if not d),
                key=lambda rid: (TYPE_ORDER[new_records[rid].record_type], rid),
            )
            if not ready:
                raise InvariantViolation(
                    "reference cycle inside bundle", code="REFERENCE_CYCLE", details={"records": sorted(remaining)}
                )
            for rid in ready:
                ordered.append(new_records[rid])
                del remaining[rid]
            for d in remaining.values():
                d.difference_update(ready)
        return ordered

    def _apply_manifest(self, manifest: dict[str, Any], staging: Path, published: list[str]) -> None:
        for entry in manifest["records"]:
            dest = self._abs(entry["relpath"])
            staged = staging / entry["staged"]
            if dest.exists():
                existing = Record.from_dict(load_json_file(dest))
                if existing.canonical_digest() != entry["digest"]:
                    raise IdConflictError(
                        f"destination {entry['relpath']} holds different content than the pending manifest",
                        details={"record_id": entry["record_id"]},
                    )
            else:
                if not staged.exists():
                    raise ConflictError(f"staged file missing for {entry['record_id']}", code="STAGING_MISSING")
                staged_record = Record.from_dict(load_json_file(staged))
                if staged_record.canonical_digest() != entry["digest"]:
                    raise ConflictError(f"staged content digest mismatch for {entry['record_id']}", code="STAGING_CORRUPT")
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, dest)
                fsync_directory(dest.parent)
            self._journal_append(entry, manifest["txn_id"])
            published.append(entry["record_id"])
            record = Record.from_dict(load_json_file(dest))
            index = self._ensure_index()
            index[record.record_id] = IndexEntry(record.record_id, record.record_type, entry["relpath"], entry["digest"])
            self._record_cache[record.record_id] = record
            if record.record_type == "kernel":
                self._kernel_index[record.payload.kernel_id] = record.record_id

    def _journal_append(self, entry: dict[str, Any], txn_id: str) -> None:
        journal = self._journal_path()
        journal.parent.mkdir(parents=True, exist_ok=True)
        line = dumps_compact(
            {
                "record_id": entry["record_id"],
                "record_type": entry["record_type"],
                "relpath": entry["relpath"],
                "digest": entry["digest"],
                "txn_id": txn_id,
                "published_at": utc_now_iso(),
            }
        )
        with open(journal, "ab") as fh:
            fh.write(line.encode("utf-8") + b"\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _journal_entries(self) -> list[dict[str, Any]]:
        journal = self._journal_path()
        if not journal.exists():
            return []
        entries: list[dict[str, Any]] = []
        for raw in journal.read_bytes().splitlines():
            if not raw.strip():
                continue
            try:
                entries.append(loads_strict(raw))
            except Exception:
                entries.append({"corrupt_line": raw.decode("utf-8", errors="replace")})
        return entries

    def _complete_manifest(self, txn_id: str) -> None:
        manifest_path = self._runtime("pending", f"{txn_id}.json")
        if manifest_path.exists():
            manifest_path.unlink()
            fsync_directory(manifest_path.parent)
        staging = self._runtime("staging", txn_id)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    def _pending_manifests(self) -> list[Path]:
        pending = self._runtime("pending")
        if not pending.exists():
            return []
        return sorted(p for p in pending.iterdir() if p.suffix == ".json" and not p.name.startswith("."))

    def recover_if_pending(self) -> RecoveryReport | None:
        if self._pending_manifests() or self._stray_temp_files():
            return self.recover()
        return None

    def _stray_temp_files(self) -> list[Path]:
        strays: list[Path] = []
        for base in (layout.KERNELS_DIR, layout.ARTIFACTS_DIR, layout.REQUESTS_DIR):
            root = self.root / base
            if root.exists():
                strays.extend(p for p in root.rglob(".*.tmp-*") if p.is_file())
        return strays

    # ------------------------------------------------------------------ recovery
    def recover(self) -> RecoveryReport:
        report = RecoveryReport()
        with self.lock():
            for stray in self._stray_temp_files():
                stray.unlink()
                report.temp_files_removed.append(str(stray.relative_to(self.root)))
            for manifest_path in self._pending_manifests():
                txn_id = manifest_path.stem
                try:
                    manifest = load_json_file(manifest_path)
                    recorded = manifest.get("manifest_digest")
                    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
                    if recorded != sha256_bytes(dumps_compact(body).encode("utf-8")):
                        raise InvariantViolation("pending manifest checksum mismatch", code="MANIFEST_CORRUPT")
                except Exception as exc:
                    sealed = self._seal_manifest(manifest_path, reason=f"unreadable manifest: {exc}")
                    report.sealed_txns.append(sealed)
                    continue
                staging = self._runtime("staging", txn_id)
                incomplete: list[dict[str, Any]] = []
                for entry in manifest["records"]:
                    dest = self._abs(entry["relpath"])
                    staged = staging / entry.get("staged", "")
                    if dest.exists():
                        try:
                            digest = Record.from_dict(load_json_file(dest)).canonical_digest()
                        except Exception as exc:
                            incomplete.append({"record_id": entry["record_id"], "reason": f"destination corrupt: {exc}"})
                            continue
                        if digest != entry["digest"]:
                            incomplete.append({"record_id": entry["record_id"], "reason": "destination holds different content"})
                            continue
                        if not self._journaled(entry["record_id"], entry["digest"]):
                            self._journal_append(entry, txn_id)
                            report.journal_repaired.append(entry["record_id"])
                    elif staged.exists():
                        try:
                            digest = Record.from_dict(load_json_file(staged)).canonical_digest()
                        except Exception as exc:
                            incomplete.append({"record_id": entry["record_id"], "reason": f"staged file corrupt: {exc}"})
                            continue
                        if digest != entry["digest"]:
                            incomplete.append({"record_id": entry["record_id"], "reason": "staged digest mismatch"})
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(staged, dest)
                        fsync_directory(dest.parent)
                        self._journal_append(entry, txn_id)
                        report.records_completed.append(entry["record_id"])
                    else:
                        incomplete.append({"record_id": entry["record_id"], "reason": "neither destination nor staged file exists"})
                if incomplete:
                    sealed = self._seal_manifest(manifest_path, reason="incomplete records", incomplete=incomplete)
                    report.sealed_txns.append(sealed)
                else:
                    self._complete_manifest(txn_id)
                    report.completed_txns.append(txn_id)
            # Journal repair for files that were renamed into place but never journaled.
            self.invalidate_index()
            journaled = {(e.get("record_id"), e.get("digest")) for e in self._journal_entries()}
            for entry in self.index_entries():
                if (entry.record_id, entry.digest) not in journaled:
                    self._journal_append(
                        {"record_id": entry.record_id, "record_type": entry.record_type, "relpath": entry.relpath, "digest": entry.digest},
                        "recovery",
                    )
                    if entry.record_id not in report.journal_repaired:
                        report.journal_repaired.append(entry.record_id)
        return report

    def _journaled(self, record_id: str, digest: str) -> bool:
        return any(e.get("record_id") == record_id and e.get("digest") == digest for e in self._journal_entries())

    def _seal_manifest(self, manifest_path: Path, *, reason: str, incomplete: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        sealed_dir = self._runtime("sealed")
        sealed_dir.mkdir(parents=True, exist_ok=True)
        target = sealed_dir / manifest_path.name
        os.replace(manifest_path, target)
        report = {"txn_id": manifest_path.stem, "reason": reason, "incomplete": incomplete or [], "sealed_manifest": str(target.relative_to(self.root))}
        self._atomic_write(sealed_dir / f"{manifest_path.stem}.report.json", dumps_readable(report).encode("utf-8"), overwrite=True)
        return report

    # ------------------------------------------------------------------ integrity
    def integrity_scan(self, *, verify_artifacts: bool = True) -> IntegrityReport:
        report = IntegrityReport()
        with self.lock():
            journal = [e for e in self._journal_entries() if "record_id" in e]
            latest: dict[str, dict[str, Any]] = {}
            for e in journal:
                latest[e["record_id"]] = e
            seen_paths: dict[str, str] = {}
            kernels_root = self.root / layout.KERNELS_DIR
            files = sorted(kernels_root.rglob("*.json")) if kernels_root.exists() else []
            found_ids: set[str] = set()
            for path in files:
                rel = PurePosixPath(path.relative_to(self.root).as_posix())
                if not layout.is_record_file(rel):
                    continue
                report.records_checked += 1
                try:
                    record = Record.from_dict(load_json_file(path))
                except Exception as exc:
                    report.corrupt.append({"path": str(rel), "error": str(exc)})
                    continue
                if record.record_id in seen_paths:
                    report.duplicate_ids.append({"record_id": record.record_id, "paths": [seen_paths[record.record_id], str(rel)]})
                seen_paths[record.record_id] = str(rel)
                found_ids.add(record.record_id)
                digest = record.canonical_digest()
                entry = latest.get(record.record_id)
                if entry is None:
                    report.unjournaled.append({"record_id": record.record_id, "path": str(rel)})
                elif entry["digest"] != digest:
                    report.modified.append({"record_id": record.record_id, "path": str(rel), "journal_digest": entry["digest"], "file_digest": digest})
                elif entry["relpath"] != str(rel):
                    report.modified.append({"record_id": record.record_id, "path": str(rel), "journal_relpath": entry["relpath"], "reason": "moved"})
            for rid, entry in latest.items():
                if rid not in found_ids:
                    report.missing.append({"record_id": rid, "expected_path": entry["relpath"]})
            if verify_artifacts:
                for ref in self.artifact_registry():
                    report.artifacts_checked += 1
                    problem = self.verify_artifact(ref)
                    if problem is not None:
                        report.artifact_problems.append(problem)
        return report

    # ------------------------------------------------------------------ artifacts
    def artifact_path(self, sha256_ref: str) -> Path:
        if not is_sha256_ref(sha256_ref):
            raise InputError(f"invalid sha256 reference: {sha256_ref!r}", code="INVALID_DIGEST")
        return self.root / layout.artifact_relpath(sha256_ref)

    def has_artifact(self, sha256_ref: str) -> bool:
        return self.artifact_path(sha256_ref).is_file()

    def put_artifact_bytes(self, data: bytes, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES) -> str:
        if len(data) > max_bytes:
            raise InputError(f"artifact of {len(data)} bytes exceeds limit {max_bytes}", code="ARTIFACT_TOO_LARGE")
        digest = artifact_digest(data)
        dest = self.artifact_path(digest)
        with self.lock():
            if dest.exists():
                if artifact_digest(dest.read_bytes()) != digest:
                    raise InvariantViolation(f"stored artifact {digest} is corrupt", code="ARTIFACT_CORRUPT")
                return digest
            self._atomic_write(dest, data, overwrite=False)
        return digest

    def import_artifact_file(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
        max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> str:
        path = Path(path)
        if not path.is_file():
            raise InputError(f"artifact file not found: {path}", code="ARTIFACT_MISSING")
        size = path.stat().st_size
        if size > max_bytes:
            raise InputError(f"artifact {path} is {size} bytes, above limit {max_bytes}", code="ARTIFACT_TOO_LARGE")
        data = path.read_bytes()
        digest = artifact_digest(data)
        if expected_sha256 is not None and digest != expected_sha256:
            raise InvariantViolation(
                f"artifact checksum mismatch for {path.name}: expected {expected_sha256}, got {digest}",
                code="ARTIFACT_CHECKSUM_MISMATCH",
                details={"path": str(path), "expected": expected_sha256, "actual": digest},
            )
        if expected_size is not None and size != expected_size:
            raise InvariantViolation(
                f"artifact size mismatch for {path.name}: expected {expected_size}, got {size}",
                code="ARTIFACT_SIZE_MISMATCH",
            )
        return self.put_artifact_bytes(data, max_bytes=max_bytes)

    def read_artifact(self, sha256_ref: str) -> bytes:
        dest = self.artifact_path(sha256_ref)
        if not dest.is_file():
            raise MissingReferenceError(f"artifact {sha256_ref} is not stored", code="ARTIFACT_MISSING")
        data = dest.read_bytes()
        if artifact_digest(data) != sha256_ref:
            raise InvariantViolation(f"stored artifact {sha256_ref} is corrupt", code="ARTIFACT_CORRUPT")
        return data

    def _check_artifact_ref(self, ref: ArtifactRef, pending: dict[str, ArtifactRef]) -> None:
        validate_record_id(ref.artifact_id, what="artifact_id")
        if not is_sha256_ref(ref.sha256):
            raise InputError(f"artifact {ref.artifact_id} has invalid sha256 {ref.sha256!r}", code="INVALID_DIGEST")
        other = pending.get(ref.artifact_id) or self.get_artifact_ref(ref.artifact_id)
        if other is not None and to_json(other) != to_json(ref):
            raise ConflictError(
                f"artifact {ref.artifact_id!r} is already registered with a different descriptor",
                code="ARTIFACT_CONFLICT",
                details={"artifact_id": ref.artifact_id, "existing": to_json(other), "new": to_json(ref)},
            )

    def register_artifact_ref(self, ref: ArtifactRef) -> bool:
        """Register/verify an artifact descriptor. Returns True if newly registered."""
        data = to_json(ref)
        validate_nested("Artifact", data)
        dest = self.root / layout.artifact_registry_relpath(ref.artifact_id)
        with self.lock():
            existing = self.get_artifact_ref(ref.artifact_id)
            if existing is not None:
                if to_json(existing) != data:
                    raise ConflictError(
                        f"artifact {ref.artifact_id!r} is already registered with a different descriptor",
                        code="ARTIFACT_CONFLICT",
                    )
                return False
            self._atomic_write(dest, dumps_readable(data).encode("utf-8"), overwrite=False)
        return True

    def get_artifact_ref(self, artifact_id: str) -> ArtifactRef | None:
        dest = self.root / layout.artifact_registry_relpath(artifact_id)
        if not dest.is_file():
            return None
        data = load_json_file(dest)
        validate_nested("Artifact", data)
        return ArtifactRef(**data)

    def artifact_registry(self) -> list[ArtifactRef]:
        registry = self.root / layout.ARTIFACTS_DIR / "registry"
        if not registry.exists():
            return []
        refs: list[ArtifactRef] = []
        for path in sorted(registry.glob("*.json")):
            if path.name.startswith("."):
                continue
            data = load_json_file(path)
            validate_nested("Artifact", data)
            refs.append(ArtifactRef(**data))
        return refs

    def verify_artifact(self, ref: ArtifactRef) -> dict[str, Any] | None:
        """Return a problem dict if the referenced artifact is missing or corrupt, else None."""
        dest = self.artifact_path(ref.sha256)
        if not dest.is_file():
            if ref.availability == "present":
                return {"artifact_id": ref.artifact_id, "sha256": ref.sha256, "problem": "missing", "availability": ref.availability}
            return None
        data = dest.read_bytes()
        if artifact_digest(data) != ref.sha256:
            return {"artifact_id": ref.artifact_id, "sha256": ref.sha256, "problem": "corrupt"}
        if len(data) != ref.size_bytes:
            return {"artifact_id": ref.artifact_id, "sha256": ref.sha256, "problem": "size_mismatch", "size": len(data)}
        return None

    # ------------------------------------------------------------------ generated views
    def write_view(self, config_record_id: str, name: str, data: bytes) -> Path:
        if name not in layout.VIEW_FILE_NAMES:
            raise InputError(f"{name!r} is not a permitted view file name", code="INVALID_VIEW")
        dest = self.config_dir(config_record_id) / name
        with self.lock():
            self._atomic_write(dest, data, overwrite=True)
        return dest

    def read_view(self, config_record_id: str, name: str) -> bytes | None:
        dest = self.config_dir(config_record_id) / name
        return dest.read_bytes() if dest.is_file() else None

    def delete_views(self, config_record_id: str) -> list[str]:
        removed: list[str] = []
        cfg = self.config_dir(config_record_id)
        for name in sorted(layout.VIEW_FILE_NAMES):
            p = cfg / name
            if p.exists():
                p.unlink()
                removed.append(name)
        return removed

    # ------------------------------------------------------------------ generic fact files (request ledger)
    def write_fact(self, relpath: str, data: bytes) -> str:
        """Atomically write an immutable fact file. Returns 'written' or 'identical'; conflicts raise."""
        dest = self._abs(relpath)
        if not dest.is_relative_to(self.root / layout.REQUESTS_DIR) and not dest.is_relative_to(self.root / layout.RUNTIME_DIR):
            raise UnsafePathError(f"write_fact is limited to {layout.REQUESTS_DIR}/ and {layout.RUNTIME_DIR}/, got {relpath!r}")
        with self.lock():
            return self._atomic_write(dest, data, overwrite=False)

    def write_runtime_state(self, relpath: str, data: bytes) -> str:
        """Overwritable runtime state (leases, spool) under .runtime/ — not experiment facts."""
        dest = self._abs(relpath)
        if not dest.is_relative_to(self.root / layout.RUNTIME_DIR):
            raise UnsafePathError(f"runtime state is limited to {layout.RUNTIME_DIR}/, got {relpath!r}")
        with self.lock():
            return self._atomic_write(dest, data, overwrite=True)

    def read_fact(self, relpath: str) -> bytes | None:
        dest = self._abs(relpath)
        return dest.read_bytes() if dest.is_file() else None

    def list_facts(self, relprefix: str) -> list[str]:
        base = self._abs(relprefix)
        if not base.exists():
            return []
        return sorted(str(PurePosixPath(p.relative_to(self.root).as_posix())) for p in base.rglob("*") if p.is_file() and not p.name.startswith("."))

    # ------------------------------------------------------------------ SQLite index (disposable)
    def rebuild_index(self) -> Path:
        from .index import SqliteIndex

        return SqliteIndex.rebuild(self)
