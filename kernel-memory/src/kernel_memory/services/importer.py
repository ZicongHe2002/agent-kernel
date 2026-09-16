"""Idempotent bundle import and export (specification sections 6, 9.2, 13, 15, 19, 22).

A bundle is a JSON object ``{"bundle_version", "is_fixture", "description", "records": [...],
"artifacts": [...]}`` (only ``records`` is required; any other top-level key is rejected).
``records`` are wire records; ``artifacts`` (optional, produced by ``export_bundle(...,
include_artifacts=True)``) are ``{"sha256", "size_bytes", "encoding": "base64", "data"}`` blobs.

Public API
----------
``ImportReport``
    ``bundle_path``, ``is_fixture``, ``records_total``, ``records_published``,
    ``records_idempotent``, ``artifacts_stored``, ``artifacts_idempotent`` (both counted per
    *present* artifact reference; ``blobs_stored`` / ``blobs_idempotent`` count distinct
    content-addressed blobs), ``provenance_downgraded`` (run ids rewritten to
    ``imported_unverified``), ``issues`` (warnings from validation), ``dry_run``, ``txn_id``
    (``None`` for dry runs and when nothing new was published). ``.ok`` is true when no
    error-severity issue is present (a returned report never carries errors: errors abort).
``import_bundle(store, bundle_path, *, artifact_root=None, allow_fixture=False,
                allow_missing_artifacts=False, trusted_source=False, registry=None,
                dry_run=False, max_bundle_bytes=50_000_000, max_artifact_bytes=200_000_000)``
    All-or-nothing pipeline; nothing is written before every check passed:
      1. strict JSON load (size limit, duplicate keys, NaN/Infinity rejected) -> ``InputError``;
      2. envelope check (object, ``records`` non-empty list, ``is_fixture`` bool) -> ``InputError``
         (``INVALID_BUNDLE`` / ``UNKNOWN_BUNDLE_FIELD`` / ``EMPTY_BUNDLE``);
      3. schema validation of every record -> ``SchemaValidationError`` naming bundle, index, id;
      4. provenance policy: fixture data (bundle ``is_fixture`` or any ``provenance=fixture`` run)
         requires ``allow_fixture`` (``InputError`` ``FIXTURE_REQUIRES_FLAG``); in a fixture bundle
         every run must be a fixture (``InvariantViolation`` ``FIXTURE_CLAIMS_TRUST``); in a
         non-fixture bundle runs claiming ``trusted_worker`` are deterministically rewritten to
         ``imported_unverified`` unless ``trusted_source`` (recorded in ``provenance_downgraded``);
      5. artifact resolution for every run artifact with ``availability=present``: embedded blob,
         ``artifact://sha256/<hex>`` (must already be in the CAS), or a relative path resolved with
         ``ids.resolve_inside(artifact_root or bundle directory, uri)`` (traversal/absolute ->
         ``UnsafePathError``, other URI schemes -> ``SecurityPolicyError``); bytes are verified
         against sha256 and size (``InvariantViolation`` ``ARTIFACT_CHECKSUM_MISMATCH`` /
         ``ARTIFACT_SIZE_MISMATCH``); a missing file is ``InputError`` ``MISSING_ARTIFACT`` unless
         ``allow_missing_artifacts`` (then a warning; ``deep_validate`` later reports
         ``MISSING_EVIDENCE``);
      6. ``validation.validate_records`` over the in-memory set with the store as external resolver;
         any error aborts with ``InvariantViolation`` whose ``code`` is the first error's code and
         whose ``details.issues`` lists the first errors;
      7. ``dry_run`` returns the report without writing (an ``IdConflictError`` is still raised for
         same-id/different-content records); otherwise ``store.publish_bundle`` (``IdConflictError``
         propagates) followed by ``store.put_artifact_bytes`` for every resolved blob.
    Re-importing the same bundle is idempotent (all records idempotent, nothing rewritten).
``export_bundle(store, *, config_ref=None, include_artifacts=False) -> dict``
    Deterministic bundle of all records (or the kernel, config and every record owned by
    ``config_ref``), sorted by (type order, record_id); ``is_fixture`` is true when any exported run
    is a fixture; artifact bytes are embedded base64 only when ``include_artifacts``.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.errors import (
    IdConflictError,
    InputError,
    InvariantViolation,
    MissingReferenceError,
    SchemaValidationError,
    SecurityPolicyError,
    UnsupportedFormat,
)
from ..domain.hashing import artifact_digest, is_sha256_ref
from ..domain.ids import resolve_inside
from ..domain.jsonio import load_json_file
from ..domain.models import TYPE_ORDER, ArtifactRef, Record
from ..domain.problems import ProblemRegistry
from ..storage.store import MemoryStore
from .common import config_of
from .validation import ERROR, WARNING, Issue, validate_records

BUNDLE_VERSION = "0.2.0"
ALLOWED_BUNDLE_KEYS = frozenset({"bundle_version", "is_fixture", "description", "records", "artifacts"})
ALLOWED_EMBEDDED_KEYS = frozenset({"sha256", "size_bytes", "encoding", "data", "media_type"})
ARTIFACT_URI_PREFIX = "artifact://sha256/"
REFUSED_URI_SCHEMES = ("http:", "https:", "ftp:", "ftps:", "file:", "data:", "s3:", "gs:", "ssh:", "git:")
DOWNGRADED_PROVENANCE = "imported_unverified"


@dataclass
class ImportReport:
    bundle_path: str
    is_fixture: bool
    records_total: int
    records_published: int
    records_idempotent: int
    artifacts_stored: int
    artifacts_idempotent: int
    provenance_downgraded: list[str]
    issues: list[Issue]
    dry_run: bool
    txn_id: str | None
    blobs_stored: int = 0
    blobs_idempotent: int = 0
    published_ids: list[str] = dataclasses.field(default_factory=list)
    idempotent_ids: list[str] = dataclasses.field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle_path": self.bundle_path,
            "is_fixture": self.is_fixture,
            "records_total": self.records_total,
            "records_published": self.records_published,
            "records_idempotent": self.records_idempotent,
            "artifacts_stored": self.artifacts_stored,
            "artifacts_idempotent": self.artifacts_idempotent,
            "blobs_stored": self.blobs_stored,
            "blobs_idempotent": self.blobs_idempotent,
            "provenance_downgraded": list(self.provenance_downgraded),
            "published_ids": list(self.published_ids),
            "idempotent_ids": list(self.idempotent_ids),
            "issues": [i.to_dict() for i in self.issues],
            "warning_count": len(self.warnings),
            "dry_run": self.dry_run,
            "txn_id": self.txn_id,
        }


@dataclass(frozen=True)
class _Envelope:
    records: list[Any]
    is_fixture: bool
    embedded: list[Any]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _parse_envelope(data: Any, bundle_path: Path) -> _Envelope:
    where = str(bundle_path)
    if not isinstance(data, dict):
        raise InputError(f"{where}: bundle must be a JSON object", code="INVALID_BUNDLE", details={"bundle": where})
    unknown = sorted(set(data) - ALLOWED_BUNDLE_KEYS)
    if unknown:
        raise InputError(
            f"{where}: unknown bundle fields {unknown}; permitted: {sorted(ALLOWED_BUNDLE_KEYS)}",
            code="UNKNOWN_BUNDLE_FIELD",
            details={"bundle": where, "unknown": unknown},
        )
    records = data.get("records")
    if not isinstance(records, list):
        raise InputError(f"{where}: bundle field 'records' must be a list", code="INVALID_BUNDLE", details={"bundle": where})
    if not records:
        raise InputError(f"{where}: bundle contains no records", code="EMPTY_BUNDLE", details={"bundle": where})
    is_fixture = data.get("is_fixture", False)
    if not isinstance(is_fixture, bool):
        raise InputError(f"{where}: bundle field 'is_fixture' must be a boolean", code="INVALID_BUNDLE", details={"bundle": where})
    version = data.get("bundle_version", BUNDLE_VERSION)
    if not isinstance(version, str):
        raise InputError(f"{where}: bundle field 'bundle_version' must be a string", code="INVALID_BUNDLE", details={"bundle": where})
    if version != BUNDLE_VERSION:
        raise UnsupportedFormat(
            f"{where}: bundle_version {version!r} is not supported (expected {BUNDLE_VERSION})",
            code="UNSUPPORTED_BUNDLE_VERSION",
            details={"bundle": where, "bundle_version": version},
        )
    description = data.get("description", "")
    if not isinstance(description, str):
        raise InputError(f"{where}: bundle field 'description' must be a string", code="INVALID_BUNDLE", details={"bundle": where})
    embedded = data.get("artifacts", [])
    if not isinstance(embedded, list):
        raise InputError(f"{where}: bundle field 'artifacts' must be a list", code="INVALID_BUNDLE", details={"bundle": where})
    return _Envelope(records=records, is_fixture=is_fixture, embedded=embedded)


def _materialize_records(items: list[Any], bundle_path: Path) -> list[Record]:
    records: list[Record] = []
    for index, item in enumerate(items):
        record_id = item.get("record_id") if isinstance(item, dict) else None
        try:
            records.append(Record.from_dict(item))
        except InputError as exc:
            details = dict(exc.details)
            details.update({"bundle": str(bundle_path), "index": index, "record_id": record_id})
            raise SchemaValidationError(
                f"{bundle_path.name}: records[{index}] ({record_id!r}) is invalid: {exc.message}",
                code=exc.code,
                details=details,
            ) from exc
    return records


def _decode_embedded_artifacts(items: list[Any], bundle_path: Path, max_artifact_bytes: int) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for index, item in enumerate(items):
        where = f"{bundle_path.name}: artifacts[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{where} must be an object", code="INVALID_BUNDLE")
        unknown = sorted(set(item) - ALLOWED_EMBEDDED_KEYS)
        if unknown:
            raise InputError(f"{where} has unknown fields {unknown}", code="INVALID_BUNDLE", details={"unknown": unknown})
        sha = item.get("sha256")
        if not is_sha256_ref(sha):
            raise InputError(f"{where}: invalid sha256 reference {sha!r}", code="INVALID_BUNDLE")
        if item.get("encoding") != "base64":
            raise InputError(f"{where}: unsupported artifact encoding {item.get('encoding')!r}", code="UNSUPPORTED_ARTIFACT_ENCODING")
        raw = item.get("data")
        if not isinstance(raw, str):
            raise InputError(f"{where}: 'data' must be a base64 string", code="INVALID_BUNDLE")
        if len(raw) > max_artifact_bytes * 2:
            raise InputError(f"{where}: embedded artifact exceeds the size limit {max_artifact_bytes}", code="ARTIFACT_TOO_LARGE")
        try:
            blob = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InputError(f"{where}: invalid base64 data: {exc}", code="INVALID_BUNDLE") from exc
        if len(blob) > max_artifact_bytes:
            raise InputError(f"{where}: embedded artifact of {len(blob)} bytes exceeds the limit {max_artifact_bytes}", code="ARTIFACT_TOO_LARGE")
        digest = artifact_digest(blob)
        if digest != sha:
            raise InvariantViolation(
                f"{where}: embedded artifact bytes hash to {digest}, not the declared {sha}",
                code="ARTIFACT_CHECKSUM_MISMATCH",
                details={"expected": sha, "actual": digest},
            )
        size = item.get("size_bytes")
        if size is not None and size != len(blob):
            raise InvariantViolation(f"{where}: embedded artifact has {len(blob)} bytes, declared {size}", code="ARTIFACT_SIZE_MISMATCH")
        out[sha] = blob
    return out


# --------------------------------------------------------------------------------------
# Provenance policy
# --------------------------------------------------------------------------------------
def _apply_provenance_policy(
    records: list[Record], *, is_fixture_bundle: bool, allow_fixture: bool, trusted_source: bool, bundle_path: Path
) -> tuple[list[Record], list[str]]:
    runs = [r for r in records if r.record_type == "run"]
    fixture_runs = [r.record_id for r in runs if r.payload.provenance == "fixture"]
    if (is_fixture_bundle or fixture_runs) and not allow_fixture:
        raise InputError(
            f"{bundle_path.name} contains fixture data (is_fixture={is_fixture_bundle}, fixture runs={len(fixture_runs)}); "
            "pass allow_fixture=True (--allow-fixture) to import synthetic records",
            code="FIXTURE_REQUIRES_FLAG",
            details={"bundle": str(bundle_path), "is_fixture": is_fixture_bundle, "fixture_runs": fixture_runs[:50]},
        )
    if is_fixture_bundle:
        claiming = [r.record_id for r in runs if r.payload.provenance != "fixture"]
        if claiming:
            raise InvariantViolation(
                f"{bundle_path.name} is declared is_fixture but runs {claiming} claim non-fixture provenance; "
                "fixture data cannot self-assert trusted execution",
                code="FIXTURE_CLAIMS_TRUST",
                details={"bundle": str(bundle_path), "records": claiming},
            )
        return records, []
    downgraded: list[str] = []
    out: list[Record] = []
    for record in records:
        if record.record_type == "run" and record.payload.provenance == "trusted_worker" and not trusted_source:
            data = record.to_dict()
            data["payload"]["provenance"] = DOWNGRADED_PROVENANCE
            out.append(Record.from_dict(data))
            downgraded.append(record.record_id)
        else:
            out.append(record)
    return out, downgraded


# --------------------------------------------------------------------------------------
# Artifact resolution
# --------------------------------------------------------------------------------------
def _store_bytes(store: MemoryStore, sha256_ref: str) -> bytes | None:
    try:
        path = store.artifact_path(sha256_ref)
    except InputError:
        return None
    return path.read_bytes() if path.is_file() else None


def _present_refs(records: list[Record]) -> list[tuple[Record, int, ArtifactRef]]:
    out: list[tuple[Record, int, ArtifactRef]] = []
    for record in records:
        if record.record_type != "run":
            continue
        for index, ref in enumerate(record.payload.artifacts):
            if ref.availability == "present":
                out.append((record, index, ref))
    return out


def _resolve_artifacts(
    store: MemoryStore,
    records: list[Record],
    *,
    root: Path,
    embedded: dict[str, bytes],
    allow_missing: bool,
    max_artifact_bytes: int,
) -> tuple[dict[str, bytes], set[str], list[Issue]]:
    """Return (sha -> verified bytes, sha set that is missing but tolerated, warnings)."""
    resolved: dict[str, bytes] = dict(embedded)
    missing: set[str] = set()
    issues: list[Issue] = []

    def missing_artifact(record: Record, index: int, ref: ArtifactRef, reason: str, **extra: Any) -> None:
        details = {"record_id": record.record_id, "artifact_id": ref.artifact_id, "uri": ref.uri, "sha256": ref.sha256, **extra}
        if allow_missing:
            missing.add(ref.sha256)
            issues.append(
                Issue(WARNING, "MISSING_ARTIFACT", f"run {record.record_id!r}: artifact {ref.artifact_id!r} {reason}; imported without evidence", record.record_id, f"artifacts[{index}]", details)
            )
            return
        raise InputError(
            f"run {record.record_id!r}: artifact {ref.artifact_id!r} {reason} (pass allow_missing_artifacts to import anyway)",
            code="MISSING_ARTIFACT",
            details=details,
        )

    for record, index, ref in _present_refs(records):
        if ref.sha256 in resolved:
            continue
        uri = ref.uri
        if uri.startswith(ARTIFACT_URI_PREFIX):
            hexpart = uri[len(ARTIFACT_URI_PREFIX):]
            if f"sha256:{hexpart}" != ref.sha256:
                raise InvariantViolation(
                    f"run {record.record_id!r}: artifact {ref.artifact_id!r} uri {uri!r} names a different digest than sha256 {ref.sha256}",
                    code="ARTIFACT_URI_MISMATCH",
                    details={"record_id": record.record_id, "artifact_id": ref.artifact_id, "uri": uri, "sha256": ref.sha256},
                )
            if store.has_artifact(ref.sha256):
                blob = store.read_artifact(ref.sha256)  # raises ARTIFACT_CORRUPT on a damaged blob
                _check_size(record, ref, blob)
                resolved[ref.sha256] = blob
            else:
                missing_artifact(record, index, ref, "is not in the content-addressed store")
            continue
        lowered = uri.lower()
        if "://" in uri or lowered.startswith(REFUSED_URI_SCHEMES):
            raise SecurityPolicyError(
                f"run {record.record_id!r}: artifact {ref.artifact_id!r} uri {uri!r} uses a scheme that is not permitted; "
                "only relative paths under the artifact root or artifact://sha256/<hex> are accepted",
                code="EXTERNAL_URI_REFUSED",
                details={"record_id": record.record_id, "artifact_id": ref.artifact_id, "uri": uri},
            )
        path = resolve_inside(root, uri)  # UnsafePathError (exit 7) for traversal / absolute / symlink escapes
        if not path.is_file():
            missing_artifact(record, index, ref, f"file {uri!r} does not exist under the artifact root", path=str(path))
            continue
        size = path.stat().st_size
        if size > max_artifact_bytes:
            raise InputError(
                f"run {record.record_id!r}: artifact {ref.artifact_id!r} is {size} bytes, above the limit {max_artifact_bytes}",
                code="ARTIFACT_TOO_LARGE",
                details={"record_id": record.record_id, "artifact_id": ref.artifact_id, "size": size, "limit": max_artifact_bytes},
            )
        blob = path.read_bytes()
        digest = artifact_digest(blob)
        if digest != ref.sha256:
            raise InvariantViolation(
                f"run {record.record_id!r}: artifact {ref.artifact_id!r} file {uri!r} hashes to {digest}, not the declared {ref.sha256}",
                code="ARTIFACT_CHECKSUM_MISMATCH",
                details={"record_id": record.record_id, "artifact_id": ref.artifact_id, "uri": uri, "expected": ref.sha256, "actual": digest},
            )
        _check_size(record, ref, blob)
        resolved[ref.sha256] = blob
    return resolved, missing, issues


def _check_size(record: Record, ref: ArtifactRef, blob: bytes) -> None:
    if len(blob) != ref.size_bytes:
        raise InvariantViolation(
            f"run {record.record_id!r}: artifact {ref.artifact_id!r} has {len(blob)} bytes, declared size_bytes {ref.size_bytes}",
            code="ARTIFACT_SIZE_MISMATCH",
            details={"record_id": record.record_id, "artifact_id": ref.artifact_id, "expected": ref.size_bytes, "actual": len(blob)},
        )


# --------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------
def import_bundle(
    store: MemoryStore,
    bundle_path: Path,
    *,
    artifact_root: Path | None = None,
    allow_fixture: bool = False,
    allow_missing_artifacts: bool = False,
    trusted_source: bool = False,
    registry: ProblemRegistry | None = None,
    dry_run: bool = False,
    max_bundle_bytes: int = 50_000_000,
    max_artifact_bytes: int = 200_000_000,
) -> ImportReport:
    """Import a bundle into ``store``. See the module docstring for the pipeline and error codes."""
    bundle_path = Path(bundle_path)
    data = load_json_file(bundle_path, max_bytes=max_bundle_bytes)
    envelope = _parse_envelope(data, bundle_path)
    records = _materialize_records(envelope.records, bundle_path)
    records, downgraded = _apply_provenance_policy(
        records,
        is_fixture_bundle=envelope.is_fixture,
        allow_fixture=allow_fixture,
        trusted_source=trusted_source,
        bundle_path=bundle_path,
    )
    embedded = _decode_embedded_artifacts(envelope.embedded, bundle_path, max_artifact_bytes)
    root = Path(artifact_root) if artifact_root is not None else bundle_path.parent
    resolved, missing, issues = _resolve_artifacts(
        store,
        records,
        root=root,
        embedded=embedded,
        allow_missing=allow_missing_artifacts,
        max_artifact_bytes=max_artifact_bytes,
    )

    def reader(sha256_ref: str) -> bytes | None:
        blob = resolved.get(sha256_ref)
        if blob is not None:
            return blob
        return _store_bytes(store, sha256_ref)

    report = validate_records(
        records,
        artifact_reader=reader,
        artifact_registry=store.get_artifact_ref,
        registry=registry,
        external_resolver=store.get,
        kernel_resolver=store.kernel_by_kernel_id,
        missing_evidence_severity=WARNING if allow_missing_artifacts else ERROR,
    )
    if not report.ok:
        errors = report.errors
        raise InvariantViolation(
            f"{bundle_path.name} failed deep validation with {len(errors)} error(s); first: [{errors[0].code}] {errors[0].message}",
            code=errors[0].code,
            details={
                "bundle": str(bundle_path),
                "error_count": len(errors),
                "codes": sorted({e.code for e in errors}),
                "issues": [e.to_dict() for e in errors[:25]],
            },
        )
    issues.extend(report.issues)

    present = _present_refs(records)
    is_fixture = envelope.is_fixture or any(r.record_type == "run" and r.payload.provenance == "fixture" for r in records)

    if dry_run:
        published: list[str] = []
        idempotent: list[str] = []
        for record in records:
            existing = store.get(record.record_id)
            if existing is None:
                published.append(record.record_id)
            elif existing.canonical_digest() == record.canonical_digest():
                idempotent.append(record.record_id)
            else:
                raise IdConflictError(
                    f"record {record.record_id!r} already exists with different content; published records are immutable",
                    details={"record_id": record.record_id, "existing_digest": existing.canonical_digest(), "new_digest": record.canonical_digest()},
                )
        blobs_existing = {sha for sha in resolved if store.has_artifact(sha)}
        blobs_new = set(resolved) - blobs_existing
        return ImportReport(
            bundle_path=str(bundle_path),
            is_fixture=is_fixture,
            records_total=len(records),
            records_published=len(published),
            records_idempotent=len(idempotent),
            artifacts_stored=sum(1 for _, _, ref in present if ref.sha256 in blobs_new),
            artifacts_idempotent=sum(1 for _, _, ref in present if ref.sha256 in blobs_existing),
            provenance_downgraded=downgraded,
            issues=issues,
            dry_run=True,
            txn_id=None,
            blobs_stored=len(blobs_new),
            blobs_idempotent=len(blobs_existing),
            published_ids=published,
            idempotent_ids=idempotent,
        )

    outcome = store.publish_bundle(records, label=f"import:{bundle_path.name}")
    blobs_existing: set[str] = set()
    blobs_new: set[str] = set()
    for sha in sorted(resolved):
        if store.has_artifact(sha):
            store.read_artifact(sha)  # verifies the stored blob; raises ARTIFACT_CORRUPT if damaged
            blobs_existing.add(sha)
        else:
            store.put_artifact_bytes(resolved[sha], max_bytes=max_artifact_bytes)
            blobs_new.add(sha)
    return ImportReport(
        bundle_path=str(bundle_path),
        is_fixture=is_fixture,
        records_total=len(records),
        records_published=len(outcome.published),
        records_idempotent=len(outcome.idempotent),
        artifacts_stored=sum(1 for _, _, ref in present if ref.sha256 in blobs_new),
        artifacts_idempotent=sum(1 for _, _, ref in present if ref.sha256 in blobs_existing),
        provenance_downgraded=downgraded,
        issues=issues,
        dry_run=False,
        txn_id=outcome.txn_id if outcome.published else None,
        blobs_stored=len(blobs_new),
        blobs_idempotent=len(blobs_existing),
        published_ids=list(outcome.published),
        idempotent_ids=list(outcome.idempotent),
    )


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def export_bundle(store: MemoryStore, *, config_ref: str | None = None, include_artifacts: bool = False) -> dict[str, Any]:
    """Build a deterministic bundle dict from the store (see module docstring)."""
    if config_ref is None:
        records = store.records()
    else:
        config = store.require(config_ref, "config")
        selected: list[Record] = [config]
        kernel = store.kernel_by_kernel_id(config.payload.kernel_id)
        if kernel is not None:
            selected.append(kernel)
        for record in store.records():
            if record.record_type in ("kernel", "config"):
                continue
            try:
                owner = config_of(store, record)
            except MissingReferenceError:
                continue
            if owner.record_id == config_ref:
                selected.append(record)
        records = selected
    records = sorted(records, key=lambda r: (TYPE_ORDER[r.record_type], r.record_id))
    runs = [r for r in records if r.record_type == "run"]
    is_fixture = any(r.payload.provenance == "fixture" for r in runs)
    description = "Kernel Memory export" + (f" of config {config_ref}" if config_ref else "")
    if is_fixture:
        description += "; contains fixture runs (synthetic, never measured)"
    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "is_fixture": is_fixture,
        "description": description,
        "records": [r.to_dict() for r in records],
    }
    if include_artifacts:
        blobs: dict[str, dict[str, Any]] = {}
        for run in runs:
            for ref in run.payload.artifacts:
                if ref.availability != "present" or ref.sha256 in blobs or not store.has_artifact(ref.sha256):
                    continue
                data = store.read_artifact(ref.sha256)
                blobs[ref.sha256] = {
                    "sha256": ref.sha256,
                    "size_bytes": len(data),
                    "media_type": ref.media_type,
                    "encoding": "base64",
                    "data": base64.b64encode(data).decode("ascii"),
                }
        bundle["artifacts"] = [blobs[sha] for sha in sorted(blobs)]
    return bundle


__all__ = ["ImportReport", "import_bundle", "export_bundle", "BUNDLE_VERSION"]
