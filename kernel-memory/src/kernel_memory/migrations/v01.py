"""Explicit v0.1 -> v0.2 migration (specification section 20, acceptance T31).

Public API
----------
* ``ShaResolver`` (Protocol): ``resolve(repo_uid, sha) -> GitOid | None`` and
  ``is_ambiguous(repo_uid, sha) -> bool``. ``NullResolver`` never resolves anything;
  ``MappingResolver`` resolves from an explicit ``{(repo_uid, sha_prefix): full_hex}`` table
  (tests / offline use). A resolver backed by a local clone (``adapters.git_local``) can be
  adapted by the CLI later: it only has to implement these two methods (``git rev-parse
  --verify <sha>^{commit}`` for ``resolve`` and the "ambiguous argument" failure for
  ``is_ambiguous``). This module deliberately does not import ``adapters.git_local``.
* ``MigrationReport`` (dataclass, ``to_dict()``): identity map, retained/discarded fields,
  unresolved data, rejection reasons, produced v0.2 record dicts, source-file digests.
* ``read_v01_source(path) -> dict``: tolerant reader for a v0.1 export (single JSON/YAML file
  or a directory of them). See ``docs/MIGRATION.md`` for the expected structure. The
  structure is *inferred from the concepts named in specification section 20*; no real v0.1
  dataset exists in this workspace, so the reader must be confirmed against real
  user-supplied data before an ``--apply`` run.
* ``migrate_v01(path, *, store, registry, resolver, repo_uid_map, dry_run=True, created_at)``
  -> ``MigrationReport``. Default is a dry run that writes nothing.

Guarantees
----------
* Every produced record is validated with ``Record.from_dict`` (via ``services.common.new_record``);
  identifiers follow ``docs/DESIGN.md`` section 4. Migration annotations use a deterministic
  ``annotation-<16 hex>`` id derived from the v0.1 identity so that re-running the migration
  with the same ``created_at`` is idempotent and a changed input surfaces as ``IdConflictError``.
* Short SHAs are resolved through the resolver or left unresolved. They are never padded
  with zeros or guessed (T31). An unresolved revision produces no commit record.
* A v0.1 ``result`` never becomes a ``run``: v0.2 Runs need environment/protocol/verifier
  snapshots, raw samples and tested-source identity. The original result fields are embedded
  verbatim (as JSON data) in an ``unverified`` note annotation on the commit. ``spill_bytes``
  is kept with the note "semantics unknown; not HBM traffic"; ``excessive_spill`` is kept as
  the diagnosis label "v0.1 diagnosis: excessive_spill" and never turned into an execution
  failure; ``result.status`` never becomes ``execution_status`` or correctness.
* ``selected_revision`` becomes a note annotation, never a Decision (no policy/evidence).
* ``parent_attempt_id`` becomes an ``optimization_origin`` relation only when both endpoints
  resolve to concrete commit records; otherwise it is reported as unresolved with the reason
  "exact origin commit unknown".
* Configs are normalised through the problem registry; kernels without a (complete) problem
  adapter have their configs rejected ("no problem adapter for kernel; cannot verify
  semantics"). No ``config_hash`` is ever fabricated for unknown semantics (``mla_forward``
  stays unresolved). Existing store kernels/configs are reused by ``kernel_id`` / ``config_hash``.
* Unknown top-level keys and unknown per-entry fields are reported as discarded, never dropped
  silently. ``trajectory`` is always discarded (it is rebuilt from authoritative facts).
* All text from the v0.1 export (titles, summaries, notes) is stored as data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..domain.errors import (
    IdConflictError,
    IncompleteProblemContract,
    InputError,
    MissingReferenceError,
    UnsafePathError,
)
from ..domain.hashing import sha256_bytes
from ..domain.ids import (
    parse_utc_timestamp,
    resolve_inside,
    short_hash_id,
    slug_for_id,
    utc_now_iso,
    validate_record_id,
)
from ..domain.jsonio import DEFAULT_MAX_BYTES, dumps_compact, load_json_file, load_yaml_strict
from ..domain.models import (
    AnnotationPayload,
    Change,
    CommitPayload,
    ConfigPayload,
    GitOid,
    KernelPayload,
    PrPayload,
    Record,
    RelationPayload,
)
from ..domain.problems import ProblemRegistry, default_registry
from ..services.common import new_record
from ..storage.store import MemoryStore

# --------------------------------------------------------------------------------------
# Expected v0.1 structure (inferred from specification section 20; confirm against real data)
# --------------------------------------------------------------------------------------
V01_TOP_LEVEL_KEYS: tuple[str, ...] = ("kernels", "configs", "attempts", "revisions", "results", "trajectory")
V01_ENTRY_ID_FIELD: dict[str, str] = {
    "kernels": "kernel_id",
    "configs": "config_id",
    "attempts": "attempt_id",
    "revisions": "revision_id",
    "results": "revision_id",
}
KERNEL_FIELDS = frozenset({"kernel_id", "display_name", "adapter_id"})
CONFIG_FIELDS = frozenset({"config_id", "kernel_id", "problem"})
ATTEMPT_FIELDS = frozenset(
    {"attempt_id", "kernel_id", "config_id", "pr_number", "repo", "title", "hypothesis", "parent_attempt_id", "selected_revision"}
)
REVISION_FIELDS = frozenset({"revision_id", "attempt_id", "commit_sha", "parent_sha", "changes", "summary"})
RESULT_FIELDS = frozenset({"revision_id", "status", "latency_us", "speedup", "spill_bytes", "excessive_spill", "environment", "notes"})
CHANGE_FIELDS = frozenset({"change_id", "component", "key", "before", "after", "rationale", "extraction_source", "attribution"})

DEFAULT_ADAPTER_ID = "unknown-v01"
MIGRATION_LABEL = "migrate-v01"
SPILL_BYTES_NOTE = "semantics unknown; not HBM traffic"
EXCESSIVE_SPILL_NOTE = "v0.1 diagnosis: excessive_spill"
RESULT_TEXT_PREFIX = "v0.1 result (unverified, not a Run): "
ORIGIN_UNKNOWN = "exact origin commit unknown"
NO_ADAPTER_REASON = "no problem adapter for kernel; cannot verify semantics"

_SOURCE_SUFFIXES = {".json", ".yaml", ".yml"}
_HEX = re.compile(r"^[0-9a-f]+$")
_GITHUB_REPO_UID = re.compile(r"^github:[A-Za-z0-9._-]+:repo:([0-9]+)$")
_LOCAL_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------------------
# SHA resolvers
# --------------------------------------------------------------------------------------
@runtime_checkable
class ShaResolver(Protocol):
    """Resolve short Git SHAs against a specific repository. Never pad, never guess."""

    def resolve(self, repo_uid: str, sha: str) -> GitOid | None: ...

    def is_ambiguous(self, repo_uid: str, sha: str) -> bool: ...


class NullResolver:
    """Resolves nothing: every short SHA stays unresolved (the safe default)."""

    def resolve(self, repo_uid: str, sha: str) -> GitOid | None:
        return None

    def is_ambiguous(self, repo_uid: str, sha: str) -> bool:
        return False


def _oid_from_full_hex(hex_value: str) -> GitOid | None:
    if len(hex_value) == 40:
        return GitOid(algorithm="sha1", hex=hex_value)
    if len(hex_value) == 64:
        return GitOid(algorithm="sha256", hex=hex_value)
    return None


class MappingResolver:
    """Offline resolver over an explicit ``{(repo_uid, sha_prefix): full_hex}`` table.

    A short SHA resolves when exactly one full hex in the same repository extends it; two or
    more matching full hexes make it ambiguous. Full hexes must be 40 (sha1) or 64 (sha256)
    lower-case hex characters and each key prefix must actually be a prefix of its value.
    """

    def __init__(self, mapping: dict[tuple[str, str], str]) -> None:
        self._entries: list[tuple[str, str]] = []
        for key, full in dict(mapping).items():
            if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key)):
                raise InputError(f"MappingResolver keys must be (repo_uid, sha_prefix) tuples, got {key!r}", code="INVALID_RESOLVER_MAP")
            repo_uid, prefix = key
            if not isinstance(full, str):
                raise InputError(f"MappingResolver value for {key!r} must be a hex string", code="INVALID_RESOLVER_MAP")
            full_l = full.strip().lower()
            prefix_l = prefix.strip().lower()
            if not _HEX.match(full_l) or _oid_from_full_hex(full_l) is None:
                raise InputError(f"MappingResolver value for {key!r} is not a full 40/64-hex object id", code="INVALID_RESOLVER_MAP")
            if not prefix_l or not _HEX.match(prefix_l) or not full_l.startswith(prefix_l):
                raise InputError(f"MappingResolver prefix {prefix!r} is not a hex prefix of {full!r}", code="INVALID_RESOLVER_MAP")
            self._entries.append((repo_uid, full_l))

    def _candidates(self, repo_uid: str, sha: str) -> set[str]:
        needle = sha.strip().lower()
        if not needle or not _HEX.match(needle):
            return set()
        return {full for repo, full in self._entries if repo == repo_uid and full.startswith(needle)}

    def resolve(self, repo_uid: str, sha: str) -> GitOid | None:
        candidates = self._candidates(repo_uid, sha)
        if len(candidates) != 1:
            return None
        return _oid_from_full_hex(next(iter(candidates)))

    def is_ambiguous(self, repo_uid: str, sha: str) -> bool:
        return len(self._candidates(repo_uid, sha)) > 1


def resolve_sha(raw: Any, repo_uid: str, resolver: ShaResolver) -> tuple[GitOid | None, str | None]:
    """Turn a v0.1 SHA string into a ``GitOid`` or a failure reason.

    Full 40/64-hex values are accepted as given (algorithm by length). Anything shorter goes
    through the resolver; ambiguous or unresolvable values return ``(None, reason)``.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "commit sha missing or not a string"
    text = raw.strip().lower()
    if not _HEX.match(text):
        return None, f"sha {raw!r} is not hexadecimal"
    full = _oid_from_full_hex(text)
    if full is not None:
        return full, None
    if resolver.is_ambiguous(repo_uid, text):
        return None, f"short sha {text!r} is ambiguous in {repo_uid}; not padded or guessed"
    resolved = resolver.resolve(repo_uid, text)
    if resolved is None:
        return None, f"short sha {text!r} could not be resolved uniquely in {repo_uid}; not padded or guessed"
    if not isinstance(resolved, GitOid) or _oid_from_full_hex(resolved.hex.lower()) is None or not resolved.hex.lower().startswith(text):
        raise InputError(
            f"resolver returned an object id that does not extend short sha {text!r}",
            code="RESOLVER_INVALID",
            details={"repo_uid": repo_uid, "sha": text, "resolved": getattr(resolved, "hex", None)},
        )
    return GitOid(algorithm=resolved.algorithm, hex=resolved.hex.lower()), None


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
@dataclass
class MigrationReport:
    source_path: str
    source_file_digests: dict[str, str]
    dry_run: bool
    identity_map: list[dict[str, Any]] = field(default_factory=list)
    retained_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    discarded_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    annotations_for_unverified: int = 0
    notes: list[str] = field(default_factory=list)
    publish_outcome: dict[str, Any] | None = None

    def record_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record["record_type"]] = counts.get(record["record_type"], 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_file_digests": dict(self.source_file_digests),
            "dry_run": self.dry_run,
            "identity_map": list(self.identity_map),
            "retained_fields": {k: dict(v) for k, v in self.retained_fields.items()},
            "discarded_fields": {k: dict(v) for k, v in self.discarded_fields.items()},
            "unresolved": list(self.unresolved),
            "rejections": list(self.rejections),
            "records": list(self.records),
            "record_counts": self.record_counts(),
            "annotations_for_unverified": self.annotations_for_unverified,
            "notes": list(self.notes),
            "publish_outcome": self.publish_outcome,
        }


# --------------------------------------------------------------------------------------
# Source reading
# --------------------------------------------------------------------------------------
def list_v01_source_files(path: Path) -> list[tuple[str, Path]]:
    """Return ``(relative name, absolute path)`` for every JSON/YAML file of a v0.1 export.

    ``path`` may be a single file or a directory (top level only). Directory entries are
    resolved with ``ids.resolve_inside`` so symlinked entries are refused, not followed.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"v0.1 source not found: {path}", code="FILE_NOT_FOUND", details={"path": str(path)})
    if path.is_file():
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            raise InputError(
                f"v0.1 source file must be .json/.yaml/.yml, got {path.name!r}", code="UNSUPPORTED_SOURCE", details={"path": str(path)}
            )
        return [(path.name, path)]
    if not path.is_dir():
        raise InputError(f"v0.1 source is neither a file nor a directory: {path}", code="UNSUPPORTED_SOURCE")
    files: list[tuple[str, Path]] = []
    for entry in sorted(path.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        try:
            resolved = resolve_inside(path, entry.name)
        except UnsafePathError as exc:
            raise UnsafePathError(f"refusing v0.1 source entry {entry.name!r}: {exc}", details={"path": str(entry)}) from exc
        if not resolved.is_file():
            continue
        files.append((entry.name, resolved))
    if not files:
        raise InputError(f"no .json/.yaml/.yml files found in v0.1 source directory {path}", code="EMPTY_SOURCE")
    return files


def source_file_digests(path: Path) -> dict[str, str]:
    """``sha256:<hex>`` of the original bytes of every input file (relative name -> digest)."""
    return {name: sha256_bytes(abs_path.read_bytes()) for name, abs_path in list_v01_source_files(path)}


def _load_source_file(abs_path: Path) -> Any:
    if abs_path.suffix.lower() == ".json":
        return load_json_file(abs_path)
    size = abs_path.stat().st_size
    if size > DEFAULT_MAX_BYTES:
        raise InputError(f"input {abs_path} is {size} bytes, above the limit of {DEFAULT_MAX_BYTES}", code="INPUT_TOO_LARGE")
    try:
        text = abs_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{abs_path} is not valid UTF-8: {exc}", code="INVALID_UTF8") from exc
    return load_yaml_strict(text)


def _merge_key(merged: dict[str, Any], key: str, value: Any, relname: str, origin: dict[str, str]) -> None:
    if key not in merged:
        merged[key] = value
        origin[key] = relname
        return
    existing = merged[key]
    if isinstance(existing, list) and isinstance(value, list):
        existing.extend(value)
        return
    if isinstance(existing, dict) and isinstance(value, dict):
        for sub_key, sub_value in value.items():
            if sub_key in existing:
                raise InputError(
                    f"v0.1 key {key!r}.{sub_key!r} is defined in both {origin[key]} and {relname}",
                    code="V01_SOURCE_CONFLICT",
                    details={"key": key, "sub_key": sub_key, "files": [origin[key], relname]},
                )
            existing[sub_key] = sub_value
        return
    raise InputError(
        f"v0.1 key {key!r} has conflicting definitions in {origin[key]} and {relname}",
        code="V01_SOURCE_CONFLICT",
        details={"key": key, "files": [origin[key], relname]},
    )


def read_v01_source(path: Path) -> dict[str, Any]:
    """Read a v0.1 export into one dict keyed by top-level v0.1 concept.

    Accepted shapes (tolerant, must be confirmed against real data):

    * one JSON/YAML file whose object carries any subset of ``kernels``, ``configs``,
      ``attempts``, ``revisions``, ``results``, ``trajectory`` (plus unknown keys, which are
      reported as discarded by ``migrate_v01``);
    * a directory of such files; list-valued keys from several files are concatenated;
    * a file named after a top-level key (``kernels.json``, ``trajectory.json``, ...) whose
      whole content is that key's value.

    Every file is parsed strictly (duplicate keys, NaN and custom YAML tags are rejected).
    """
    merged: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for relname, abs_path in list_v01_source_files(path):
        data = _load_source_file(abs_path)
        stem = Path(relname).stem.lower()
        if stem in V01_TOP_LEVEL_KEYS and not (isinstance(data, dict) and stem in data):
            _merge_key(merged, stem, data, relname, origin)
        elif isinstance(data, dict):
            for key, value in data.items():
                _merge_key(merged, str(key), value, relname, origin)
        else:
            raise InputError(
                f"v0.1 source file {relname} must contain a JSON object (or be named after a top-level key), got {type(data).__name__}",
                code="V01_SOURCE_INVALID",
                details={"file": relname},
            )
    return merged


# --------------------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------------------
def _local_slug(text: str) -> str:
    slug = _LOCAL_SLUG_UNSAFE.sub("-", text.lower()).strip("-")
    return slug[:32].strip("-") or "attempt"


@dataclass
class _AttemptCtx:
    attempt_id: str
    config_ref: str
    kernel_id: str
    pr_key: str
    pr_record_id: str
    repo_uid: str
    provider: str
    number: int | None
    title: str
    hypothesis: str | None
    parent_attempt_id: str | None
    selected_revision: str | None
    revision_ids: list[str] = field(default_factory=list)
    origin_ref: str | None = None


class _Migration:
    def __init__(
        self,
        source: dict[str, Any],
        *,
        store: MemoryStore | None,
        registry: ProblemRegistry,
        resolver: ShaResolver,
        repo_uid_map: dict[str, str],
        created_at: str,
        report: MigrationReport,
    ) -> None:
        self.source = source
        self.store = store
        self.registry = registry
        self.resolver = resolver
        self.repo_uid_map = repo_uid_map
        self.created_at = created_at
        self.report = report
        self._records: dict[str, Record] = {}
        self._origins: dict[str, list[str]] = {}
        self.kernel_record_ids: dict[str, str] = {}  # v0.1 kernel_id -> kernel record id (new or reused)
        self.config_record_ids: dict[str, str] = {}  # v0.1 config_id -> config record id (new or reused)
        self.config_kernel: dict[str, str] = {}  # v0.1 config_id -> kernel_id
        self.attempts: dict[str, _AttemptCtx] = {}
        self.revision_attempt: dict[str, str] = {}  # every revision seen with a migrated attempt
        self.revision_commit: dict[str, str] = {}  # only revisions whose sha resolved

    # ------------------------------------------------------------------ report helpers
    def _retain(self, kind: str, field_name: str, where: str) -> None:
        self.report.retained_fields.setdefault(kind, {})[field_name] = where

    def _discard(self, kind: str, field_name: str, reason: str) -> None:
        self.report.discarded_fields.setdefault(kind, {})[field_name] = reason

    def _map(self, v01_kind: str, v01_id: str, v02_type: str, v02_id: str | None, status: str) -> None:
        self.report.identity_map.append(
            {"v01_kind": v01_kind, "v01_id": v01_id, "v02_record_type": v02_type, "v02_record_id": v02_id, "status": status}
        )

    def _reject(self, v01_kind: str, v01_id: str, v02_type: str, reason: str) -> None:
        self.report.rejections.append({"v01_kind": v01_kind, "v01_id": v01_id, "reason": reason})
        self._map(v01_kind, v01_id, v02_type, None, "rejected")

    def _unresolved(self, v01_kind: str, v01_id: str, v02_type: str, reason: str) -> None:
        self.report.unresolved.append({"v01_kind": v01_kind, "v01_id": v01_id, "reason": reason})
        self._map(v01_kind, v01_id, v02_type, None, "unresolved")

    def _note(self, text: str) -> None:
        self.report.notes.append(text)

    def _add_record(self, record: Record, v01_id: str) -> None:
        existing = self._records.get(record.record_id)
        if existing is not None:
            if existing.canonical_digest() != record.canonical_digest():
                raise IdConflictError(
                    f"v0.1 entries {self._origins[record.record_id]} and {v01_id!r} map to record {record.record_id!r} with different content",
                    details={"record_id": record.record_id, "v01_ids": self._origins[record.record_id] + [v01_id]},
                )
            self._origins[record.record_id].append(v01_id)
            return
        self._records[record.record_id] = record
        self._origins[record.record_id] = [v01_id]

    def _known(self, record_id: str) -> bool:
        return record_id in self._records or (self.store is not None and self.store.exists(record_id))

    # ------------------------------------------------------------------ entry helpers
    def _entries(self, kind: str) -> list[Any]:
        value = self.source.get(kind)
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            id_field = V01_ENTRY_ID_FIELD[kind]
            entries: list[Any] = []
            for key, item in value.items():
                if isinstance(item, dict) and id_field not in item:
                    item = {id_field: key, **item}
                entries.append(item)
            self._note(f"v0.1 {kind!r} was an object keyed by {id_field}; converted to a list of {len(entries)} entries")
            return entries
        raise InputError(
            f"v0.1 {kind!r} must be a list of objects, got {type(value).__name__}", code="V01_SOURCE_INVALID", details={"key": kind}
        )

    def _unknown_fields(self, kind: str, entry: dict[str, Any], known: frozenset[str]) -> None:
        for name in sorted(set(entry) - known):
            self._discard(kind, name, "unknown v0.1 field; not migrated")

    @staticmethod
    def _entry_id(entry: Any, id_field: str, index: int) -> str:
        if isinstance(entry, dict) and isinstance(entry.get(id_field), str) and entry[id_field]:
            return entry[id_field]
        return f"<{id_field} missing at index {index}>"

    # ------------------------------------------------------------------ stages
    def run(self) -> None:
        self._check_top_level()
        self._migrate_kernels()
        self._migrate_configs()
        self._collect_attempts()
        self._migrate_revisions()
        self._resolve_origins()
        self._build_pr_records()
        self._migrate_selected_revisions()
        self._migrate_results()
        self._migrate_parent_links()
        self._finalize()

    def _check_top_level(self) -> None:
        for key in sorted(self.source):
            if key not in V01_TOP_LEVEL_KEYS:
                self._discard("top_level", key, "unknown v0.1 top-level key; not migrated")
        if "trajectory" in self.source:
            self._discard("top_level", "trajectory", "v0.1 trajectory.json is a derived view; rebuilt from migrated authoritative facts")
        for kind in ("kernels", "configs", "attempts", "revisions", "results"):
            self._entries(kind)  # type check only

    def _migrate_kernels(self) -> None:
        for index, entry in enumerate(self._entries("kernels")):
            v01_id = self._entry_id(entry, "kernel_id", index)
            if not isinstance(entry, dict):
                self._reject("kernel", v01_id, "kernel", "kernel entry is not an object")
                continue
            self._unknown_fields("kernels", entry, KERNEL_FIELDS)
            kernel_id = entry.get("kernel_id")
            if not isinstance(kernel_id, str) or not kernel_id:
                self._reject("kernel", v01_id, "kernel", "kernel_id missing or not a string")
                continue
            try:
                validate_record_id(kernel_id, what="kernel_id")
            except InputError as exc:
                self._reject("kernel", v01_id, "kernel", f"kernel_id is not a valid identifier: {exc.message}")
                continue
            if kernel_id in self.kernel_record_ids:
                self._reject("kernel", v01_id, "kernel", f"duplicate kernel_id {kernel_id!r}; first occurrence kept")
                continue
            self._retain("kernels", "kernel_id", "kernel.kernel_id / record id kernel-<kernel_id>")
            existing = self.store.kernel_by_kernel_id(kernel_id) if self.store is not None else None
            if existing is not None:
                self.kernel_record_ids[kernel_id] = existing.record_id
                self._map("kernel", v01_id, "kernel", existing.record_id, "mapped")
                self._note(f"kernel {kernel_id!r}: reused existing store record {existing.record_id!r}; v0.1 display_name/adapter_id not applied")
                continue
            display_name = entry.get("display_name")
            if display_name is not None and not isinstance(display_name, str):
                self._reject("kernel", v01_id, "kernel", "display_name must be a string")
                continue
            adapter_id = entry.get("adapter_id")
            if adapter_id is not None and not isinstance(adapter_id, str):
                self._reject("kernel", v01_id, "kernel", "adapter_id must be a string")
                continue
            if "display_name" in entry:
                self._retain("kernels", "display_name", "kernel.display_name")
            if adapter_id:
                try:
                    validate_record_id(adapter_id, what="adapter_id")
                except InputError as exc:
                    self._reject("kernel", v01_id, "kernel", f"adapter_id is not a valid identifier: {exc.message}")
                    continue
                self._retain("kernels", "adapter_id", "kernel.adapter_id (unverified)")
            else:
                self._note(f"kernel {kernel_id!r}: no adapter_id in v0.1; using {DEFAULT_ADAPTER_ID!r}")
            record_id = f"kernel-{kernel_id}"
            payload = KernelPayload(
                kernel_id=kernel_id,
                display_name=display_name or kernel_id,
                adapter_id=adapter_id or DEFAULT_ADAPTER_ID,
                contract_notes=(
                    f"Migrated from v0.1 (kernel_id={kernel_id}). Adapter and problem contract were not verified by the "
                    "migration; register a real adapter before executing runs. Historical dimensions are context, not an ABI."
                ),
            )
            self._add_record(new_record("kernel", record_id, payload, created_at=self.created_at), v01_id)
            self.kernel_record_ids[kernel_id] = record_id
            self._map("kernel", v01_id, "kernel", record_id, "mapped")

    def _migrate_configs(self) -> None:
        for index, entry in enumerate(self._entries("configs")):
            v01_id = self._entry_id(entry, "config_id", index)
            if not isinstance(entry, dict):
                self._reject("config", v01_id, "config", "config entry is not an object")
                continue
            self._unknown_fields("configs", entry, CONFIG_FIELDS)
            config_id = entry.get("config_id")
            kernel_id = entry.get("kernel_id")
            problem = entry.get("problem")
            if not isinstance(config_id, str) or not config_id:
                self._reject("config", v01_id, "config", "config_id missing or not a string")
                continue
            if config_id in self.config_record_ids or config_id in self.config_kernel:
                self._reject("config", v01_id, "config", f"duplicate config_id {config_id!r}; first occurrence kept")
                continue
            if not isinstance(kernel_id, str) or not kernel_id:
                self._reject("config", v01_id, "config", "kernel_id missing or not a string")
                continue
            if not isinstance(problem, dict):
                self._reject("config", v01_id, "config", "problem missing or not an object")
                continue
            self._retain("configs", "config_id", "identity_map.v01_id (v0.2 config_id is the normalized config_id_hint)")
            self._retain("configs", "kernel_id", "config.kernel_id")
            self._retain("configs", "problem", "config.problem after registry normalization; config_hash recomputed")
            if kernel_id not in self.kernel_record_ids:
                self._reject("config", v01_id, "config", f"kernel {kernel_id!r} was not migrated (missing or rejected kernel entry)")
                continue
            try:
                normalized = self.registry.normalize(kernel_id, problem)
            except IncompleteProblemContract as exc:
                self._reject("config", v01_id, "config", f"{NO_ADAPTER_REASON} ({exc.message})")
                continue
            except InputError as exc:
                self._reject("config", v01_id, "config", f"problem rejected by the {kernel_id!r} normalizer: {exc.message}")
                continue
            self.config_kernel[config_id] = kernel_id
            existing = self.store.configs_by_hash(normalized.config_hash) if self.store is not None else []
            if existing:
                self.config_record_ids[config_id] = existing[0].record_id
                self._map("config", v01_id, "config", existing[0].record_id, "mapped")
                self._note(f"config {config_id!r}: same config_hash as existing store record {existing[0].record_id!r}; reused (T01)")
                continue
            record_id = f"cfg-{normalized.config_id_hint}-{normalized.config_hash[len('sha256:'):][:12]}"
            try:
                validate_record_id(record_id)
            except InputError as exc:
                self._reject("config", v01_id, "config", f"derived config record id invalid: {exc.message}")
                continue
            payload = ConfigPayload(
                kernel_id=kernel_id,
                config_id=normalized.config_id_hint,
                problem_schema_id=normalized.problem_schema_id,
                problem_schema_digest=normalized.problem_schema_digest,
                problem=normalized.problem,
                config_hash=normalized.config_hash,
                tags=["migrated-v01"],
            )
            record = new_record("config", record_id, payload, created_at=self.created_at)
            if record_id in self._records:
                self._note(f"config {config_id!r}: same config_hash as v0.1 config(s) {self._origins[record_id]}; one v0.2 config (T01)")
            self._add_record(record, v01_id)
            self.config_record_ids[config_id] = record_id
            self._map("config", v01_id, "config", record_id, "mapped")

    def _collect_attempts(self) -> None:
        for index, entry in enumerate(self._entries("attempts")):
            v01_id = self._entry_id(entry, "attempt_id", index)
            if not isinstance(entry, dict):
                self._reject("attempt", v01_id, "pr", "attempt entry is not an object")
                continue
            self._unknown_fields("attempts", entry, ATTEMPT_FIELDS)
            attempt_id = entry.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                self._reject("attempt", v01_id, "pr", "attempt_id missing or not a string")
                continue
            if attempt_id in self.attempts:
                self._reject("attempt", v01_id, "pr", f"duplicate attempt_id {attempt_id!r}; first occurrence kept")
                continue
            config_id = entry.get("config_id")
            if not isinstance(config_id, str) or config_id not in self.config_record_ids:
                self._reject("attempt", v01_id, "pr", f"config {config_id!r} was not migrated (missing or rejected config)")
                continue
            kernel_id = entry.get("kernel_id")
            if kernel_id is not None and kernel_id != self.config_kernel[config_id]:
                self._reject("attempt", v01_id, "pr", f"attempt kernel_id {kernel_id!r} disagrees with config {config_id!r} kernel {self.config_kernel[config_id]!r}")
                continue
            title = entry.get("title")
            if title is None:
                title = f"v0.1 attempt {attempt_id} (no title recorded)"
                self._note(f"attempt {attempt_id!r}: no title in v0.1; placeholder title used")
            elif not isinstance(title, str):
                self._reject("attempt", v01_id, "pr", "title must be a string")
                continue
            hypothesis = entry.get("hypothesis")
            if hypothesis is not None and not isinstance(hypothesis, str):
                self._reject("attempt", v01_id, "pr", "hypothesis must be a string or null")
                continue
            pr_number = entry.get("pr_number")
            if pr_number is not None and (isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number < 1):
                self._reject("attempt", v01_id, "pr", "pr_number must be a positive integer or null")
                continue
            repo = entry.get("repo")
            if repo is not None and not isinstance(repo, str):
                self._reject("attempt", v01_id, "pr", "repo must be a string or null")
                continue
            parent = entry.get("parent_attempt_id")
            if parent is not None and not isinstance(parent, str):
                self._reject("attempt", v01_id, "pr", "parent_attempt_id must be a string or null")
                continue
            selected = entry.get("selected_revision")
            if selected is not None and not isinstance(selected, str):
                self._reject("attempt", v01_id, "pr", "selected_revision must be a string or null")
                continue
            self._retain("attempts", "attempt_id", "identity_map.v01_id / pr_key derivation")
            self._retain("attempts", "config_id", "pr.config_ref")
            if kernel_id is not None:
                self._retain("attempts", "kernel_id", "consistency check against config.kernel_id")
            self._retain("attempts", "title", "pr.title")
            if "hypothesis" in entry:
                self._retain("attempts", "hypothesis", "pr.hypothesis")
            if parent is not None:
                self._retain("attempts", "parent_attempt_id", "relation optimization_origin (+ pr.origin_ref) only when endpoints resolve; else unresolved")
            if selected is not None:
                self._retain("attempts", "selected_revision", "note annotation on the commit (never a Decision)")

            config_ref = self.config_record_ids[config_id]
            repo_uid = self.repo_uid_map.get(repo) if repo is not None else None
            if repo is not None and repo_uid is None:
                repo_uid = f"local:{slug_for_id(repo)}"
                self._note(f"attempt {attempt_id!r}: repo {repo!r} has no repo_uid mapping; using placeholder repo_uid {repo_uid!r}")
                self._discard("attempts", "repo", "no repo_uid mapping for this value; placeholder local: repo_uid used")
            elif repo is not None:
                self._retain("attempts", "repo", "pr.repo_uid via --repo-uid-map")
            if repo_uid is None:
                repo_uid = "local:unknown-v01"
            github_match = _GITHUB_REPO_UID.match(repo_uid)
            if pr_number is not None and github_match is not None:
                provider, number = "github", pr_number
                pr_key = f"gh-{github_match.group(1)}-pr-{pr_number}"
                self._retain("attempts", "pr_number", "pr.number (provider github)")
            else:
                provider, number = "local", None
                pr_key = short_hash_id(f"local-{_local_slug(attempt_id)}", "v01-attempt", config_ref, attempt_id, length=12)
                if pr_number is not None:
                    self._discard("attempts", "pr_number", "kept only in notes: repo_uid is not a github:<host>:repo:<id> mapping; provider=local, number=null")
                    self._note(f"attempt {attempt_id!r}: pr_number {pr_number} not applied (repo_uid {repo_uid!r} is not a GitHub repo_uid)")
            self.attempts[attempt_id] = _AttemptCtx(
                attempt_id=attempt_id,
                config_ref=config_ref,
                kernel_id=self.config_kernel[config_id],
                pr_key=pr_key,
                pr_record_id=f"pr-{pr_key}",
                repo_uid=repo_uid,
                provider=provider,
                number=number,
                title=title,
                hypothesis=hypothesis,
                parent_attempt_id=parent,
                selected_revision=selected,
            )

    def _normalize_change(self, raw: Any, revision_id: str, index: int) -> Change | None:
        label = f"changes[{revision_id}][{index}]"
        if not isinstance(raw, dict):
            self._discard("revisions", label, "change entry is not an object; dropped")
            return None
        for name in sorted(set(raw) - CHANGE_FIELDS):
            self._discard("revisions", f"{label}.{name}", "unknown change field; not migrated")
        component = raw.get("component")
        if not isinstance(component, str) or not component:
            self._discard("revisions", label, "change without a string component; dropped")
            return None
        if "before" not in raw or "after" not in raw:
            self._discard("revisions", label, "change without before/after; dropped")
            return None
        key = raw.get("key")
        if key is not None and not isinstance(key, str):
            self._discard("revisions", label, "change key must be a string or null; dropped")
            return None
        rationale = raw.get("rationale", "")
        if not isinstance(rationale, str):
            self._discard("revisions", label, "change rationale must be a string; dropped")
            return None
        change_id = raw.get("change_id")
        if change_id is None:
            change_id = f"change-v01-{slug_for_id(revision_id)}-{index}"
        if not isinstance(change_id, str):
            self._discard("revisions", label, "change_id must be a string; dropped")
            return None
        try:
            validate_record_id(change_id, what="change_id")
        except InputError:
            self._discard("revisions", label, f"change_id {change_id!r} is not a valid identifier; dropped")
            return None
        if raw.get("attribution") not in (None, "group_only"):
            self._note(f"revision {revision_id!r}: change {change_id!r} attribution {raw.get('attribution')!r} overridden to group_only (a combined result never proves individual effects)")
        extraction_source = "explicit" if raw.get("extraction_source") == "explicit" else "unknown"
        return Change(
            change_id=change_id,
            component=component,
            key=key,
            before=raw["before"],
            after=raw["after"],
            rationale=rationale,
            extraction_source=extraction_source,
            attribution="group_only",
        )

    def _migrate_revisions(self) -> None:
        for index, entry in enumerate(self._entries("revisions")):
            v01_id = self._entry_id(entry, "revision_id", index)
            if not isinstance(entry, dict):
                self._reject("revision", v01_id, "commit", "revision entry is not an object")
                continue
            self._unknown_fields("revisions", entry, REVISION_FIELDS)
            revision_id = entry.get("revision_id")
            if not isinstance(revision_id, str) or not revision_id:
                self._reject("revision", v01_id, "commit", "revision_id missing or not a string")
                continue
            if revision_id in self.revision_attempt:
                self._reject("revision", v01_id, "commit", f"duplicate revision_id {revision_id!r}; first occurrence kept")
                continue
            attempt_id = entry.get("attempt_id")
            ctx = self.attempts.get(attempt_id) if isinstance(attempt_id, str) else None
            if ctx is None:
                self._reject("revision", v01_id, "commit", f"attempt {attempt_id!r} was not migrated (missing or rejected attempt)")
                continue
            self._retain("revisions", "revision_id", "identity_map.v01_id")
            self._retain("revisions", "attempt_id", "commit.pr_ref")
            self._retain("revisions", "commit_sha", "commit.commit_oid (full sha as given; short sha via resolver or unresolved)")
            self.revision_attempt[revision_id] = attempt_id
            ctx.revision_ids.append(revision_id)
            oid, failure = resolve_sha(entry.get("commit_sha"), ctx.repo_uid, self.resolver)
            if oid is None:
                reason = failure or "commit sha unresolved"
                malformed = ("not hexadecimal" in reason) or reason.startswith("commit sha missing")
                if malformed:
                    self._reject("revision", v01_id, "commit", reason)
                else:
                    self._unresolved("revision", v01_id, "commit", reason)
                continue
            parents: list[GitOid] = []
            if entry.get("parent_sha") is not None:
                self._retain("revisions", "parent_sha", "commit.git_parent_oids (same resolution rules; unresolved -> [] with a note)")
                parent_oid, parent_failure = resolve_sha(entry.get("parent_sha"), ctx.repo_uid, self.resolver)
                if parent_oid is None:
                    self._note(f"revision {revision_id!r}: parent_sha unresolved ({parent_failure}); git_parent_oids left empty")
                    self._unresolved("revision.parent_sha", revision_id, "commit.git_parent_oids", parent_failure or "parent sha unresolved")
                else:
                    parents.append(parent_oid)
            changes: list[Change] = []
            raw_changes = entry.get("changes")
            if raw_changes is not None:
                self._retain("revisions", "changes", "commit.changes (validated entries; attribution forced group_only; extraction_source unknown unless explicit)")
                if not isinstance(raw_changes, list):
                    self._discard("revisions", f"changes[{revision_id}]", "changes is not a list; dropped")
                else:
                    for change_index, raw_change in enumerate(raw_changes):
                        change = self._normalize_change(raw_change, revision_id, change_index)
                        if change is not None:
                            changes.append(change)
            summary = entry.get("summary")
            if summary is not None and not isinstance(summary, str):
                self._reject("revision", v01_id, "commit", "summary must be a string or null")
                continue
            if summary is not None:
                self._retain("revisions", "summary", "commit.summary (summary_author human)")
                summary_author = "human"
            else:
                summary, summary_author = "", "collector"
                self._note(f"revision {revision_id!r}: no summary in v0.1; empty summary recorded (author collector)")
            record_id = f"commit-{ctx.pr_key}-{oid.hex[:12]}"
            payload = CommitPayload(
                pr_ref=ctx.pr_record_id,
                repo_uid=ctx.repo_uid,
                commit_oid=oid,
                git_parent_oids=parents,
                diff_base_oid=None,
                source_available=False,
                change_status="recorded" if changes else "not_extracted",
                changes=changes,
                summary=summary,
                summary_author=summary_author,
                diff_artifact_ref=None,
            )
            self._add_record(new_record("commit", record_id, payload, created_at=self.created_at), v01_id)
            self.revision_commit[revision_id] = record_id
            self._map("revision", v01_id, "commit", record_id, "mapped")

    def _endpoint_commit(self, ctx: _AttemptCtx) -> tuple[str | None, str]:
        """The one concrete commit an attempt stands for: its selected revision or its single revision."""
        if ctx.selected_revision is not None:
            if ctx.selected_revision not in ctx.revision_ids:
                return None, f"selected_revision {ctx.selected_revision!r} is not a revision of attempt {ctx.attempt_id!r}"
            commit_id = self.revision_commit.get(ctx.selected_revision)
            if commit_id is None:
                return None, f"selected_revision {ctx.selected_revision!r} of attempt {ctx.attempt_id!r} has no migrated commit (sha unresolved)"
            return commit_id, ""
        if not ctx.revision_ids:
            return None, f"attempt {ctx.attempt_id!r} has no revisions"
        if len(ctx.revision_ids) == 1:
            commit_id = self.revision_commit.get(ctx.revision_ids[0])
            if commit_id is None:
                return None, f"the single revision {ctx.revision_ids[0]!r} of attempt {ctx.attempt_id!r} has no migrated commit (sha unresolved)"
            return commit_id, ""
        return None, f"attempt {ctx.attempt_id!r} has {len(ctx.revision_ids)} revisions and no selected_revision"

    def _resolve_origins(self) -> None:
        for ctx in self.attempts.values():
            if ctx.parent_attempt_id is None:
                continue
            parent = self.attempts.get(ctx.parent_attempt_id)
            if parent is None:
                continue  # reported in _migrate_parent_links
            origin, _ = self._endpoint_commit(parent)
            ctx.origin_ref = origin

    def _build_pr_records(self) -> None:
        for ctx in self.attempts.values():
            payload = PrPayload(
                config_ref=ctx.config_ref,
                pr_key=ctx.pr_key,
                repo_uid=ctx.repo_uid,
                provider=ctx.provider,
                number=ctx.number,
                title=ctx.title,
                hypothesis=ctx.hypothesis,
                origin_ref=ctx.origin_ref,
            )
            self._add_record(new_record("pr", ctx.pr_record_id, payload, created_at=self.created_at), ctx.attempt_id)
            self._map("attempt", ctx.attempt_id, "pr", ctx.pr_record_id, "mapped")

    def _annotation(self, *, target_ref: str, text: str, id_parts: tuple[str, ...], v01_kind: str, v01_id: str) -> None:
        record_id = short_hash_id("annotation", "v01", *id_parts, length=16)
        payload = AnnotationPayload(
            target_ref=target_ref,
            category="note",
            text=text,
            author_kind="human",
            evidence_refs=[],
            confidence="unverified",
            supersedes_ref=None,
        )
        self._add_record(new_record("annotation", record_id, payload, created_at=self.created_at), v01_id)
        self.report.annotations_for_unverified += 1
        self._map(v01_kind, v01_id, "annotation", record_id, "mapped")

    def _migrate_selected_revisions(self) -> None:
        for ctx in self.attempts.values():
            if ctx.selected_revision is None:
                continue
            commit_id = self.revision_commit.get(ctx.selected_revision)
            if ctx.selected_revision not in ctx.revision_ids:
                self._unresolved("selected_revision", ctx.attempt_id, "annotation", f"selected_revision {ctx.selected_revision!r} is not a revision of attempt {ctx.attempt_id!r}")
                continue
            if commit_id is None:
                self._unresolved("selected_revision", ctx.attempt_id, "annotation", f"selected_revision {ctx.selected_revision!r} has no migrated commit (sha unresolved)")
                continue
            self._annotation(
                target_ref=commit_id,
                text=f"v0.1 selected_revision={ctx.selected_revision}; no policy/evidence-backed decision migrated",
                id_parts=("selected_revision", ctx.attempt_id, commit_id),
                v01_kind="selected_revision",
                v01_id=ctx.attempt_id,
            )

    def _migrate_results(self) -> None:
        for index, entry in enumerate(self._entries("results")):
            revision_id = entry.get("revision_id") if isinstance(entry, dict) else None
            v01_id = f"{revision_id if isinstance(revision_id, str) else '<revision_id missing>'}#result[{index}]"
            if not isinstance(entry, dict):
                self._reject("result", v01_id, "annotation", "result entry is not an object")
                continue
            self._unknown_fields("results", entry, RESULT_FIELDS)
            if not isinstance(revision_id, str) or not revision_id:
                self._reject("result", v01_id, "annotation", "revision_id missing or not a string")
                continue
            self._retain("results", "revision_id", "annotation.target_ref (the migrated commit)")
            for name in ("status", "latency_us", "speedup", "spill_bytes", "excessive_spill", "environment", "notes"):
                if name in entry:
                    self._retain("results", name, RESULT_FIELD_DISPOSITION[name])
            if revision_id not in self.revision_attempt:
                self._reject("result", v01_id, "annotation", f"revision {revision_id!r} was not migrated (missing or rejected revision)")
                continue
            commit_id = self.revision_commit.get(revision_id)
            if commit_id is None:
                self._unresolved("result", v01_id, "annotation", f"revision {revision_id!r} has no migrated commit (sha unresolved); result not migrated (T31)")
                continue
            field_notes: dict[str, str] = {}
            if "status" in entry:
                field_notes["status"] = "v0.1 result.status retained as-is; execution status and correctness are not inferred from it"
            for name in ("latency_us", "speedup"):
                if name in entry:
                    field_notes[name] = "historical number without environment/protocol/raw samples/tested-source identity; not a v0.2 measurement"
            if "spill_bytes" in entry:
                field_notes["spill_bytes"] = SPILL_BYTES_NOTE
            if "excessive_spill" in entry:
                field_notes["excessive_spill"] = f"{EXCESSIVE_SPILL_NOTE} (original diagnosis label; execution failure is not inferred)"
            if "environment" in entry:
                field_notes["environment"] = "free-form v0.1 description; not an Environment snapshot"
            body = {
                "kind": "v0.1 result (unverified; not a v0.2 Run)",
                "v01_revision_id": revision_id,
                "v01_result": entry,
                "field_notes": field_notes,
            }
            try:
                text = RESULT_TEXT_PREFIX + dumps_compact(body)
            except (TypeError, ValueError) as exc:
                self._reject("result", v01_id, "annotation", f"result is not JSON-serialisable: {exc}")
                continue
            self._annotation(
                target_ref=commit_id,
                text=text,
                id_parts=("result", commit_id, dumps_compact(entry)),
                v01_kind="result",
                v01_id=v01_id,
            )

    def _migrate_parent_links(self) -> None:
        for ctx in self.attempts.values():
            if ctx.parent_attempt_id is None:
                continue
            parent = self.attempts.get(ctx.parent_attempt_id)
            if parent is None:
                self._unresolved("parent_attempt_id", ctx.attempt_id, "relation", f"{ORIGIN_UNKNOWN}: parent attempt {ctx.parent_attempt_id!r} was not migrated")
                continue
            origin, origin_reason = self._endpoint_commit(parent)
            if origin is None:
                self._unresolved("parent_attempt_id", ctx.attempt_id, "relation", f"{ORIGIN_UNKNOWN}: {origin_reason}")
                continue
            child, child_reason = self._endpoint_commit(ctx)
            if child is None:
                self._unresolved(
                    "parent_attempt_id",
                    ctx.attempt_id,
                    "relation",
                    f"{ORIGIN_UNKNOWN} for a concrete child commit: {child_reason}; pr.origin_ref={origin} recorded on the PR only",
                )
                continue
            if parent.config_ref != ctx.config_ref:
                self._note(f"attempt {ctx.attempt_id!r}: origin attempt {parent.attempt_id!r} belongs to another config; relation filed under the child config")
            record_id = short_hash_id("relation-optimization_origin", origin, child, length=12)
            payload = RelationPayload(
                config_ref=ctx.config_ref,
                kind="optimization_origin",
                from_ref=origin,
                to_ref=child,
                evidence_refs=[],
                rationale=(
                    f"Migrated from v0.1 parent_attempt_id={ctx.parent_attempt_id}: both endpoints resolved to concrete commits "
                    "(parent's selected or single revision -> child's selected or single revision). No evidence migrated."
                ),
            )
            self._add_record(new_record("relation", record_id, payload, created_at=self.created_at), ctx.attempt_id)
            self._map("parent_attempt_id", ctx.attempt_id, "relation", record_id, "mapped")

    def _finalize(self) -> None:
        records = self.records()
        for record in records:
            for ref in record.record_references():
                if not self._known(ref.target):
                    raise MissingReferenceError(
                        f"{record.record_type} {record.record_id!r} field {ref.field} refers to missing record {ref.target!r}",
                        details={"record_id": record.record_id, "field": ref.field, "target": ref.target},
                    )
            if self.store is not None:
                existing = self.store.get(record.record_id)
                if existing is not None and existing.canonical_digest() != record.canonical_digest():
                    raise IdConflictError(
                        f"record {record.record_id!r} already exists with different content; published records are immutable",
                        details={
                            "record_id": record.record_id,
                            "existing_digest": existing.canonical_digest(),
                            "new_digest": record.canonical_digest(),
                            "v01_ids": self._origins.get(record.record_id, []),
                        },
                    )
        self.report.records = [record.to_dict() for record in records]
        counts = self.report.record_counts()
        self._note(
            "produced records: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none")
            + "; runs=0 and decisions=0 by design (v0.1 results/selections are unverified annotations)"
        )

    def records(self) -> list[Record]:
        return sorted(self._records.values(), key=lambda r: (r.record_type, r.record_id))


RESULT_FIELD_DISPOSITION: dict[str, str] = {
    "status": "annotation text only (verbatim); NOT run.execution_status / correctness",
    "latency_us": "annotation text only (verbatim); NOT run.timing (no raw samples, protocol, environment)",
    "speedup": "annotation text only (verbatim); NOT a derived comparison (no comparable runs)",
    "spill_bytes": f"annotation text only (verbatim) with note '{SPILL_BYTES_NOTE}'; NOT an analysis metric",
    "excessive_spill": f"annotation text only as diagnosis label '{EXCESSIVE_SPILL_NOTE}'; no execution failure inferred",
    "environment": "annotation text only (verbatim); NOT an Environment snapshot (no environment_hash)",
    "notes": "annotation text only (verbatim)",
}


def migrate_v01(
    path: Path,
    *,
    store: MemoryStore | None = None,
    registry: ProblemRegistry | None = None,
    resolver: ShaResolver | None = None,
    repo_uid_map: dict[str, str] | None = None,
    dry_run: bool = True,
    created_at: str | None = None,
) -> MigrationReport:
    """Migrate a v0.1 export at ``path`` into v0.2 records (dry run by default).

    * ``store``: used to reuse existing kernels/configs and to detect conflicts; records are
      published (``publish_bundle(..., allow_dangling=False)``) only when ``dry_run`` is False.
    * ``registry``: problem registry (default ``problems.default_registry()``).
    * ``resolver``: short-SHA resolver (default ``NullResolver`` -> every short SHA unresolved).
    * ``repo_uid_map``: v0.1 repository names -> v0.2 ``repo_uid`` (e.g. ``github:github.com:repo:42``).
    * ``created_at``: timestamp for all produced records (default now). Pass the same value to
      re-run idempotently.
    """
    path = Path(path)
    stamp = created_at or utc_now_iso()
    parse_utc_timestamp(stamp)
    if store is not None and not isinstance(store, MemoryStore):
        raise InputError("store must be a MemoryStore", code="INVALID_STORE")
    mapping: dict[str, str] = {}
    for name, repo_uid in dict(repo_uid_map or {}).items():
        if not isinstance(name, str) or not isinstance(repo_uid, str):
            raise InputError("repo_uid_map must map repository names (str) to repo_uid strings", code="INVALID_REPO_UID_MAP")
        validate_record_id(repo_uid, what="repo_uid")
        mapping[name] = repo_uid
    digests = source_file_digests(path)
    source = read_v01_source(path)
    report = MigrationReport(source_path=str(path), source_file_digests=digests, dry_run=bool(dry_run))
    report.notes.append(
        "v0.1 input structure is inferred from specification section 20 and read tolerantly; confirm it against the real export before --apply"
    )
    migration = _Migration(
        source,
        store=store,
        registry=registry or default_registry(),
        resolver=resolver or NullResolver(),
        repo_uid_map=mapping,
        created_at=stamp,
        report=report,
    )
    migration.run()
    if dry_run or store is None:
        report.notes.append("dry run: nothing was written" if dry_run else "no store given: nothing was written")
        return report
    outcome = store.publish_bundle(migration.records(), label=MIGRATION_LABEL, allow_dangling=False)
    report.publish_outcome = outcome.to_dict()
    report.notes.append(f"published {len(outcome.published)} record(s), {len(outcome.idempotent)} already present (idempotent)")
    return report


__all__ = [
    "ShaResolver",
    "NullResolver",
    "MappingResolver",
    "MigrationReport",
    "resolve_sha",
    "list_v01_source_files",
    "source_file_digests",
    "read_v01_source",
    "migrate_v01",
    "V01_TOP_LEVEL_KEYS",
    "DEFAULT_ADAPTER_ID",
    "SPILL_BYTES_NOTE",
    "EXCESSIVE_SPILL_NOTE",
    "RESULT_TEXT_PREFIX",
    "ORIGIN_UNKNOWN",
    "NO_ADAPTER_REASON",
]
