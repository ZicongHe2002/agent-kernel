"""v0.2 -> v0.3 upgrade: the algorithm level (ADR-0004).

Schema ``0.3.0`` inserts an ``algorithm`` record between ``kernel`` and ``config`` and makes
``config.payload.algorithm_ref`` required. Records published under ``0.2.0`` are never edited in
place (they are immutable); instead legacy *input* is upgraded in memory before it is validated
against the current contract and published.

Public API
----------
``upgrade_v02_records(dicts, *, algorithm_resolver=None) -> (dicts, UpgradeReport)``
    Pure, deterministic, order-preserving. For every record dict:

    * ``schema_version`` ``"0.2.0"`` becomes ``"0.3.0"`` (``0.3.0`` input passes through untouched;
      any other value is left alone so validation reports it).
    * a ``config`` whose payload lacks ``algorithm_ref`` is linked to the kernel's placeholder
      algorithm ``algorithm-<kernel_id>-unspecified``; ``config_hash`` is never recomputed and
      the config's ``record_id`` never changes.
    * for every ``kernel`` dict in the input one placeholder algorithm record is synthesized and
      APPENDED at the end (original bundle indices stay valid for error reporting), with
      ``created_at`` copied from that kernel dict (never ``now()``, so re-imports are idempotent).
      When a config's kernel is absent from the input, ``algorithm_resolver(kernel_id, algorithm_id)``
      (for example ``store.algorithm_by_id``) is consulted; when it returns nothing the placeholder is
      synthesized with ``created_at`` = the earliest ``created_at`` among that kernel's configs in
      the input (deterministic). A record with the placeholder id already present is never
      duplicated.
    * unknown fields and malformed items are preserved untouched so that validation raises the
      same errors as before.

    Applying the function twice equals applying it once.
``upgrade_bundle_v02(bundle, *, algorithm_resolver=None) -> (bundle, UpgradeReport)``
    Same for a whole bundle document (``bundle_version`` -> ``0.3.0``; artifacts untouched).
``default_algorithm_record_id(kernel_id)``, ``default_algorithm_payload(kernel_id)``,
``default_algorithm_record(kernel_id, created_at)``
    The placeholder. Its ``method_summary`` states literally that no method was described; it
    is authored by ``program`` and tagged ``imported-v02``. It is a placeholder, never knowledge.

Store migration (layout 1 -> 2) lives in ``read_legacy_store`` / ``migrate_v02_store`` further below
(``StoreMigrationReport``, ``LegacyStoreSnapshot``); see their docstrings for the refusal codes.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ..domain.errors import InputError, InvariantViolation, SchemaValidationError
from ..domain.hashing import artifact_digest, jcs_digest
from ..domain.ids import utc_now_iso
from ..domain.jsonio import dumps_readable, load_json_file, loads_strict
from ..domain.models import ArtifactRef, Record, to_json
from ..domain.schema import LEGACY_SCHEMA_VERSION, RECORD_TYPES, SCHEMA_VERSION, validate_legacy_record_dict, validate_nested

DEFAULT_ALGORITHM_ID = "unspecified"
LEGACY_METHOD_SUMMARY = "unspecified (imported from v0.2)"
LEGACY_SUMMARY_AUTHOR = "program"
LEGACY_TAGS: tuple[str, ...] = ("imported-v02",)
TARGET_SCHEMA_VERSION = SCHEMA_VERSION

AlgorithmResolver = Callable[[str, str], Any]


def default_algorithm_record_id(kernel_id: str) -> str:
    return f"algorithm-{kernel_id}-{DEFAULT_ALGORITHM_ID}"


def default_algorithm_payload(kernel_id: str) -> dict[str, Any]:
    return {
        "kernel_id": kernel_id,
        "algorithm_id": DEFAULT_ALGORITHM_ID,
        "display_name": DEFAULT_ALGORITHM_ID,
        "method_summary": LEGACY_METHOD_SUMMARY,
        "summary_author": LEGACY_SUMMARY_AUTHOR,
        "tags": list(LEGACY_TAGS),
    }


def default_algorithm_record(kernel_id: str, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "record_type": "algorithm",
        "record_id": default_algorithm_record_id(kernel_id),
        "created_at": created_at,
        "payload": default_algorithm_payload(kernel_id),
    }


@dataclass
class UpgradeReport:
    from_version: str = LEGACY_SCHEMA_VERSION
    to_version: str = TARGET_SCHEMA_VERSION
    records_in: int = 0
    records_rewritten: list[str] = field(default_factory=list)  # schema_version bumped
    configs_linked: list[str] = field(default_factory=list)  # algorithm_ref added
    records_synthesized: list[str] = field(default_factory=list)  # placeholder algorithms appended
    records_untouched: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.records_rewritten or self.configs_linked or self.records_synthesized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "records_in": self.records_in,
            "records_rewritten": list(self.records_rewritten),
            "configs_linked": list(self.configs_linked),
            "records_synthesized": list(self.records_synthesized),
            "records_untouched": self.records_untouched,
            "placeholder_method_summary": LEGACY_METHOD_SUMMARY,
        }


def _is_record_like(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("payload"), dict)


def upgrade_v02_records(dicts: list[Any], *, algorithm_resolver: AlgorithmResolver | None = None) -> tuple[list[Any], UpgradeReport]:
    report = UpgradeReport(records_in=len(dicts))
    out: list[Any] = copy.deepcopy(list(dicts))
    present_ids = {d.get("record_id") for d in out if isinstance(d, dict)}
    kernels_in_input: dict[str, str] = {}  # kernel_id -> created_at
    config_created: dict[str, list[str]] = {}  # kernel_id -> created_at values of configs lacking algorithm_ref
    for item in out:
        if not _is_record_like(item):
            report.records_untouched += 1
            continue
        version = item.get("schema_version")
        if version == LEGACY_SCHEMA_VERSION:
            item["schema_version"] = TARGET_SCHEMA_VERSION
            report.records_rewritten.append(str(item.get("record_id")))
        elif version != TARGET_SCHEMA_VERSION:
            report.records_untouched += 1
        rtype = item.get("record_type")
        payload = item["payload"]
        kernel_id = payload.get("kernel_id")
        if rtype == "kernel" and isinstance(kernel_id, str) and isinstance(item.get("created_at"), str):
            kernels_in_input.setdefault(kernel_id, item["created_at"])
        if rtype == "config" and "algorithm_ref" not in payload and isinstance(kernel_id, str):
            # Insert right after kernel_id so the readable JSON mirrors the contract's field order.
            rebuilt: dict[str, Any] = {}
            for key, value in payload.items():
                rebuilt[key] = value
                if key == "kernel_id":
                    rebuilt["algorithm_ref"] = default_algorithm_record_id(kernel_id)
            item["payload"] = rebuilt
            report.configs_linked.append(str(item.get("record_id")))
            if isinstance(item.get("created_at"), str):
                config_created.setdefault(kernel_id, []).append(item["created_at"])
            else:
                config_created.setdefault(kernel_id, [])
    # Synthesize placeholders: kernels in the input first (created_at copied), then kernels only
    # referenced by configs (resolver consulted, else earliest config created_at).
    for kernel_id, created_at in kernels_in_input.items():
        rid = default_algorithm_record_id(kernel_id)
        if rid in present_ids:
            continue
        out.append(default_algorithm_record(kernel_id, created_at))
        present_ids.add(rid)
        report.records_synthesized.append(rid)
    for kernel_id, stamps in sorted(config_created.items()):
        rid = default_algorithm_record_id(kernel_id)
        if rid in present_ids or kernel_id in kernels_in_input:
            continue
        if algorithm_resolver is not None and algorithm_resolver(kernel_id, DEFAULT_ALGORITHM_ID) is not None:
            continue
        if not stamps:
            continue
        out.append(default_algorithm_record(kernel_id, min(stamps)))
        present_ids.add(rid)
        report.records_synthesized.append(rid)
    return out, report


def upgrade_bundle_v02(bundle: dict[str, Any], *, algorithm_resolver: AlgorithmResolver | None = None) -> tuple[dict[str, Any], UpgradeReport]:
    if not isinstance(bundle, dict) or not isinstance(bundle.get("records"), list):
        raise InputError("bundle must be an object with a 'records' list", code="INVALID_BUNDLE")
    upgraded = dict(bundle)
    records, report = upgrade_v02_records(bundle["records"], algorithm_resolver=algorithm_resolver)
    upgraded["records"] = records
    if upgraded.get("bundle_version", LEGACY_SCHEMA_VERSION) == LEGACY_SCHEMA_VERSION:
        upgraded["bundle_version"] = TARGET_SCHEMA_VERSION
    return upgraded, report


# --------------------------------------------------------------------------------------
# Legacy store (layout_version 1) reading and migration into a new root
# --------------------------------------------------------------------------------------
LEGACY_VIEW_FILE_NAMES = frozenset({"trajectory.json", "memory_records.jsonl", "context.json"})
LEGACY_LAYOUT_VERSION = 1
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class LegacyStoreSnapshot:
    """Read-only picture of a layout-1 store (``read_legacy_store``). Nothing here is ever written back."""

    root: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]  # raw 0.2.0 dicts in file order (only files that validate against the legacy contract)
    relpaths: dict[str, str]  # record_id -> relpath
    journal: list[dict[str, Any]]  # parsed journal entries (corrupt lines are reported as problems, not kept)
    integrity_problems: list[dict[str, Any]]
    artifact_blobs: dict[str, Path]  # sha256 ref -> path (digest-verified)
    artifact_descriptors: list[dict[str, Any]]
    request_files: list[Path]
    pending_state: list[str]
    embedded_record_requests: list[str] = field(default_factory=list)  # requests/** files that embed a full record document
    notes: list[str] = field(default_factory=list)


@dataclass
class StoreMigrationReport:
    source_root: str
    dest_root: str
    dry_run: bool
    records_read: int = 0
    records_rewritten: int = 0
    records_linked: int = 0
    records_synthesized: list[str] = field(default_factory=list)
    records_published: int = 0
    records_idempotent: int = 0
    artifacts_found: int = 0
    artifacts_copied: int = 0
    descriptors_found: int = 0
    descriptors_copied: int = 0
    requests_found: int = 0
    requests_copied: int = 0
    kernel_views_published: list[str] = field(default_factory=list)
    integrity_problems: list[dict[str, Any]] = field(default_factory=list)
    destination_integrity_ok: bool | None = None
    forced: bool = False
    provenance: str = "preserved"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root": self.source_root,
            "dest_root": self.dest_root,
            "dry_run": self.dry_run,
            "records_read": self.records_read,
            "records_rewritten": self.records_rewritten,
            "records_linked": self.records_linked,
            "records_synthesized": list(self.records_synthesized),
            "records_published": self.records_published,
            "records_idempotent": self.records_idempotent,
            "artifacts_found": self.artifacts_found,
            "artifacts_copied": self.artifacts_copied,
            "descriptors_found": self.descriptors_found,
            "descriptors_copied": self.descriptors_copied,
            "requests_found": self.requests_found,
            "requests_copied": self.requests_copied,
            "kernel_views_published": list(self.kernel_views_published),
            "integrity_problems": list(self.integrity_problems),
            "destination_integrity_ok": self.destination_integrity_ok,
            "forced": self.forced,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }


def _legacy_is_record_file(rel: PurePosixPath) -> bool:
    """Layout-1 rule: any ``*.json`` under kernels/ except the per-config view files and dot files."""
    if not rel.parts or rel.parts[0] != "kernels":
        return False
    name = rel.name
    return name.endswith(".json") and not name.startswith(".") and name not in LEGACY_VIEW_FILE_NAMES


def _looks_like_record_document(parsed: Any) -> bool:
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("payload"), dict)
        and parsed.get("record_type") in RECORD_TYPES
        and isinstance(parsed.get("schema_version"), str)
        and isinstance(parsed.get("record_id"), str)
    )


def _has_files(directory: Path) -> bool:
    return directory.exists() and any(p.is_file() and not p.name.startswith(".") for p in directory.rglob("*"))


def read_legacy_store(root: Path) -> LegacyStoreSnapshot:
    """Read a layout-1 (v0.2) store WITHOUT writing anything (no lock file, no index, no recovery).

    ``NOT_A_STORE`` when there is no manifest, ``NOT_A_LEGACY_STORE`` when ``layout_version`` is not 1.
    Every record file is validated against the verbatim 0.2.0 contract and compared with the journal;
    findings go to ``integrity_problems`` with ``problem`` in {corrupt, duplicate, modified (incl.
    ``reason: moved``), unjournaled, missing, corrupt_journal_line, artifact_corrupt,
    artifact_unaddressable, corrupt_descriptor}. Request files that embed a full record document are
    listed in ``embedded_record_requests``.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise InputError(f"{root} is not a kernel-memory store (manifest.json missing)", code="NOT_A_STORE", details={"root": str(root)})
    manifest = load_json_file(manifest_path)
    found = manifest.get("layout_version") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or found != LEGACY_LAYOUT_VERSION:
        raise InputError(
            f"{root} is not a layout-version-1 (v0.2) store; found layout_version={found!r}",
            code="NOT_A_LEGACY_STORE",
            details={"root": str(root), "layout_version": found},
        )
    notes: list[str] = []
    pending_state = [f".runtime/{sub}" for sub in ("pending", "jobs") if _has_files(root / ".runtime" / sub)]
    if _has_files(root / ".runtime" / "sealed"):
        notes.append("source holds sealed manifests under .runtime/sealed; they are diagnostics of the old store and are not migrated")

    problems: list[dict[str, Any]] = []
    journal: list[dict[str, Any]] = []
    jpath = root / "journal" / "records.jsonl"
    if jpath.is_file():
        for line_no, raw in enumerate(jpath.read_bytes().splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                entry = loads_strict(raw)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                problems.append({"problem": "corrupt_journal_line", "line_no": line_no, "error": str(exc)})
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("record_id"), str):
                problems.append({"problem": "corrupt_journal_line", "line_no": line_no, "error": "entry is not an object with a record_id"})
                continue
            journal.append(entry)
    journaled: dict[str, dict[str, Any]] = {}
    for entry in journal:
        journaled[entry["record_id"]] = entry  # latest entry per record id wins

    records: list[dict[str, Any]] = []
    relpaths: dict[str, str] = {}
    kernels_root = root / "kernels"
    files = sorted(kernels_root.rglob("*.json")) if kernels_root.exists() else []
    for path in files:
        rel = PurePosixPath(path.relative_to(root).as_posix())
        if not _legacy_is_record_file(rel):
            continue
        try:
            data = load_json_file(path)
            validate_legacy_record_dict(data)
        except Exception as exc:  # noqa: BLE001 - any unreadable/invalid file is one finding
            problems.append({"path": str(rel), "problem": "corrupt", "error": str(exc)})
            continue
        rid = data["record_id"]
        if rid in relpaths:
            problems.append({"record_id": rid, "problem": "duplicate", "paths": [relpaths[rid], str(rel)]})
            continue
        digest = jcs_digest(data)
        entry = journaled.get(rid)
        if entry is None:
            problems.append({"record_id": rid, "path": str(rel), "problem": "unjournaled"})
        elif entry.get("digest") != digest:
            problems.append({"record_id": rid, "path": str(rel), "problem": "modified", "journal_digest": entry.get("digest"), "file_digest": digest})
        elif entry.get("relpath") != str(rel):
            problems.append({"record_id": rid, "path": str(rel), "problem": "modified", "reason": "moved", "journal_relpath": entry.get("relpath")})
        records.append(data)
        relpaths[rid] = str(rel)
    for rid, entry in journaled.items():
        if rid not in relpaths:
            problems.append({"record_id": rid, "problem": "missing", "expected_path": entry.get("relpath")})

    blobs: dict[str, Path] = {}
    cas = root / "artifacts" / "sha256"
    if cas.exists():
        for path in sorted(p for p in cas.rglob("*") if p.is_file() and not p.name.startswith(".")):
            rel = str(PurePosixPath(path.relative_to(root).as_posix()))
            if not _SHA256_HEX.fullmatch(path.name):
                problems.append({"path": rel, "problem": "artifact_unaddressable"})
                continue
            ref = f"sha256:{path.name}"
            actual = artifact_digest(path.read_bytes())
            if actual != ref:
                problems.append({"path": rel, "problem": "artifact_corrupt", "declared": ref, "actual": actual})
                continue
            blobs[ref] = path
    descriptors: list[dict[str, Any]] = []
    registry = root / "artifacts" / "registry"
    if registry.exists():
        for path in sorted(registry.glob("*.json")):
            if path.name.startswith("."):
                continue
            rel = str(PurePosixPath(path.relative_to(root).as_posix()))
            try:
                descriptor = load_json_file(path)
            except Exception as exc:  # noqa: BLE001
                problems.append({"path": rel, "problem": "corrupt_descriptor", "error": str(exc)})
                continue
            if not isinstance(descriptor, dict):
                problems.append({"path": rel, "problem": "corrupt_descriptor", "error": "descriptor is not an object"})
                continue
            descriptors.append(descriptor)
    requests: list[Path] = []
    embedded: list[str] = []
    rq = root / "requests"
    if rq.exists():
        requests = sorted(p for p in rq.rglob("*") if p.is_file() and not p.name.startswith("."))
        for path in requests:
            try:
                parsed = loads_strict(path.read_bytes())
            except Exception:  # noqa: BLE001 - non-JSON request files are copied verbatim
                continue
            if _looks_like_record_document(parsed):
                embedded.append(str(PurePosixPath(path.relative_to(root).as_posix())))
    return LegacyStoreSnapshot(
        root=root,
        manifest=manifest,
        records=records,
        relpaths=relpaths,
        journal=journal,
        integrity_problems=problems,
        artifact_blobs=blobs,
        artifact_descriptors=descriptors,
        request_files=requests,
        pending_state=pending_state,
        embedded_record_requests=embedded,
        notes=notes,
    )


def _check_destination(source_root: Path, dest_root: Path) -> None:
    dest_resolved = dest_root.resolve()
    if dest_resolved == source_root or dest_resolved.is_relative_to(source_root) or source_root.is_relative_to(dest_resolved):
        raise InputError(
            f"destination {dest_root} must be a separate directory: not the source store, not inside it, not containing it",
            code="INVALID_DESTINATION",
            details={"source": str(source_root), "dest": str(dest_resolved)},
        )
    if dest_root.exists():
        if not dest_root.is_dir():
            raise InputError(f"destination {dest_root} exists and is not a directory", code="INVALID_DESTINATION", details={"dest": str(dest_resolved)})
        entries = sorted(p.name for p in dest_root.iterdir() if p.name != ".DS_Store")
        if entries:
            raise InputError(
                f"destination {dest_root} must be absent or an empty directory (migration never merges into an existing store)",
                code="ROOT_NOT_EMPTY",
                details={"dest": str(dest_resolved), "entries": entries[:20]},
            )


def migrate_v02_store(source_root: Path, dest_root: Path, *, dry_run: bool = True, force: bool = False) -> StoreMigrationReport:
    """Migrate a layout-1 (v0.2) store into a NEW root at layout 2. The source is never written.

    Order (every refusal happens before anything is written to the destination):
      1. ``read_legacy_store`` (read-only; the source is never opened as a ``MemoryStore``);
      2. ``MIGRATE_PENDING_STATE`` when ``.runtime/pending`` or ``.runtime/jobs`` hold files;
      3. ``EMBEDDED_RECORD_IN_REQUEST`` when a ``requests/**`` file embeds a full record document
         (facts are never rewritten during migration; not overridable);
      4. ``SOURCE_INTEGRITY`` when the source disagrees with its own journal (modified / moved / corrupt /
         unjournaled / missing / duplicate records, corrupt blobs or descriptors) unless ``force``, which
         records the findings in the report and the new manifest;
      5. ``INVALID_DESTINATION`` (destination is the source, inside it or containing it) / ``ROOT_NOT_EMPTY``;
      6. in-memory upgrade (``upgrade_v02_records``; ``SchemaValidationError`` if an upgraded record does
         not validate against 0.3.0) and registry descriptor checks (``ARTIFACT_DESCRIPTOR_INVALID`` /
         ``ARTIFACT_DESCRIPTOR_CONFLICT`` against the run records' own descriptors);
      7. ``dry_run`` (the default) returns here with every count filled in;
      8. ``MemoryStore.init(dest)``, ``publish_bundle``, CAS blobs (digest-verified again), registry
         descriptors, ``requests/**`` verbatim through ``write_fact``, ``migrated_from`` in the new manifest,
         one kernel trajectory per kernel (lazy import of ``services.trajectory.publish_kernel_trajectory``;
         a note when unavailable or failing — views never invalidate migrated facts), integrity scan.
    Provenance is preserved as stored: a store-to-store move of already-admitted facts is not an
    untrusted import. Old per-config view files, ``journal/``, ``.cache/`` and ``.runtime/`` are not copied.
    """
    from ..storage.store import DEFAULT_MAX_ARTIFACT_BYTES, MemoryStore

    source_root = Path(source_root).resolve()
    dest_root = Path(dest_root)
    snapshot = read_legacy_store(source_root)
    report = StoreMigrationReport(source_root=str(source_root), dest_root=str(dest_root), dry_run=dry_run)
    report.records_read = len(snapshot.records)
    report.artifacts_found = len(snapshot.artifact_blobs)
    report.descriptors_found = len(snapshot.artifact_descriptors)
    report.requests_found = len(snapshot.request_files)
    report.integrity_problems = list(snapshot.integrity_problems)
    report.notes.extend(snapshot.notes)
    if snapshot.pending_state:
        raise InvariantViolation(
            f"source store {source_root} has pending runtime state ({', '.join(snapshot.pending_state)}); "
            "recover or finish it with the 0.2 tooling before migrating",
            code="MIGRATE_PENDING_STATE",
            details={"pending": snapshot.pending_state},
        )
    if snapshot.embedded_record_requests:
        raise InvariantViolation(
            f"request file(s) {snapshot.embedded_record_requests[:5]} embed a full record document; "
            "refusing to rewrite facts during migration",
            code="EMBEDDED_RECORD_IN_REQUEST",
            details={"paths": snapshot.embedded_record_requests[:50]},
        )
    if snapshot.integrity_problems:
        if not force:
            raise InvariantViolation(
                f"source store {source_root} fails its own integrity check ({len(snapshot.integrity_problems)} problem(s)); "
                "pass force=True (--force) to migrate anyway and record the findings",
                code="SOURCE_INTEGRITY",
                details={"problems": snapshot.integrity_problems[:50], "count": len(snapshot.integrity_problems)},
            )
        report.forced = True
        report.notes.append(f"forced: {len(snapshot.integrity_problems)} source integrity problem(s) recorded; records migrated as found")
    _check_destination(source_root, dest_root)

    upgraded, up = upgrade_v02_records(snapshot.records)  # no resolver: a v0.2 store holds no algorithm records
    report.records_rewritten = len(up.records_rewritten)
    report.records_linked = len(up.configs_linked)
    report.records_synthesized = list(up.records_synthesized)
    records: list[Record] = []
    for item in upgraded:
        try:
            records.append(Record.from_dict(item))
        except InputError as exc:
            rid = item.get("record_id") if isinstance(item, dict) else None
            raise SchemaValidationError(
                f"record {rid!r} ({snapshot.relpaths.get(rid, 'synthesized')}) does not validate against {SCHEMA_VERSION} after the "
                f"{LEGACY_SCHEMA_VERSION} upgrade: {exc.message}",
                code=exc.code,
                details={**exc.details, "record_id": rid, "path": snapshot.relpaths.get(rid)},
            ) from exc
    declared: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.record_type == "run":
            for ref in record.payload.artifacts:
                declared[ref.artifact_id] = to_json(ref)
    descriptor_refs: list[ArtifactRef] = []
    for descriptor in snapshot.artifact_descriptors:
        try:
            validate_nested("Artifact", descriptor)
            ref = ArtifactRef(**descriptor)
        except (InputError, TypeError) as exc:
            raise InvariantViolation(
                f"artifact registry descriptor {descriptor.get('artifact_id')!r} in the source store is invalid: {exc}",
                code="ARTIFACT_DESCRIPTOR_INVALID",
                details={"artifact_id": descriptor.get("artifact_id")},
            ) from exc
        other = declared.get(ref.artifact_id)
        if other is not None and other != to_json(ref):
            raise InvariantViolation(
                f"artifact registry descriptor {ref.artifact_id!r} differs from the descriptor declared by its run record",
                code="ARTIFACT_DESCRIPTOR_CONFLICT",
                details={"artifact_id": ref.artifact_id, "registry": to_json(ref), "run": other},
            )
        descriptor_refs.append(ref)
    if dry_run:
        report.notes.append("dry run: nothing was written; run with --apply to create the destination store")
        return report

    store = MemoryStore.init(dest_root)
    outcome = store.publish_bundle(records, label=f"migrate-v02:{source_root.name}")
    report.records_published = len(outcome.published)
    report.records_idempotent = len(outcome.idempotent)
    for ref_digest, path in sorted(snapshot.artifact_blobs.items()):
        data = path.read_bytes()
        if artifact_digest(data) != ref_digest:
            raise InvariantViolation(f"artifact blob {ref_digest} in the source store is corrupt", code="ARTIFACT_CORRUPT", details={"path": str(path)})
        store.put_artifact_bytes(data, max_bytes=max(DEFAULT_MAX_ARTIFACT_BYTES, len(data)))
        report.artifacts_copied += 1
    for ref in descriptor_refs:
        store.register_artifact_ref(ref)  # idempotent for the descriptors publish_bundle already registered
        report.descriptors_copied += 1
    for path in snapshot.request_files:
        rel = PurePosixPath(path.relative_to(source_root).as_posix())
        store.write_fact(str(rel), path.read_bytes())
        report.requests_copied += 1
    manifest_path = store.root / "manifest.json"
    manifest = load_json_file(manifest_path)
    manifest["migrated_from"] = {
        "root": str(source_root),
        "store_id": snapshot.manifest.get("store_id"),
        "store_version": snapshot.manifest.get("store_version"),
        "layout_version": snapshot.manifest.get("layout_version"),
        "journal_entries": len(snapshot.journal),
        "integrity_problems": len(snapshot.integrity_problems),
        "forced": report.forced,
        "provenance": report.provenance,
        "migrated_at": utc_now_iso(),
    }
    store._atomic_write(manifest_path, dumps_readable(manifest).encode("utf-8"), overwrite=True)
    try:
        from ..services.trajectory import publish_kernel_trajectory
    except ImportError:
        publish_kernel_trajectory = None
        report.notes.append(
            "kernel trajectories not published: services.trajectory.publish_kernel_trajectory is unavailable; "
            "run `kmem trajectory --kernel <kernel_id> --rebuild` later"
        )
    if publish_kernel_trajectory is not None:
        for kernel in store.records("kernel"):
            kernel_id = kernel.payload.kernel_id
            try:
                publish_kernel_trajectory(store, kernel_id, force=True)
                report.kernel_views_published.append(kernel_id)
            except Exception as exc:  # noqa: BLE001 - a view failure never invalidates migrated facts
                report.notes.append(f"kernel trajectory for {kernel_id!r} not published: {type(exc).__name__}: {exc}")
    scan = store.integrity_scan()
    report.destination_integrity_ok = scan.ok
    if not scan.ok:
        report.notes.append(f"destination integrity scan reported problems: {json.dumps(scan.to_dict(), sort_keys=True)[:500]}")
    return report


__all__ = [
    "DEFAULT_ALGORITHM_ID",
    "LEGACY_METHOD_SUMMARY",
    "LEGACY_SUMMARY_AUTHOR",
    "LEGACY_TAGS",
    "UpgradeReport",
    "default_algorithm_record_id",
    "default_algorithm_payload",
    "default_algorithm_record",
    "upgrade_v02_records",
    "upgrade_bundle_v02",
    "LegacyStoreSnapshot",
    "StoreMigrationReport",
    "read_legacy_store",
    "migrate_v02_store",
]
