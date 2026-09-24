"""Authoritative store: lifecycle, atomic publication, idempotency (T12), artifacts (T27),
integrity scan, views, fact files (T28), and the disposable SQLite index (T24).

Specification sections 4, 13, 15; docs/DESIGN.md section 3; ADR-0002/ADR-0003.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path, PurePosixPath

import pytest

from conftest import import_demo_bundle, record_dict
from kernel_memory.domain.errors import (
    ConflictError,
    IdConflictError,
    InputError,
    InvariantViolation,
    MissingReferenceError,
    UnsafePathError,
)
from kernel_memory.domain.hashing import artifact_digest, sha256_bytes
from kernel_memory.domain.jsonio import dumps_readable
from kernel_memory.domain.models import TYPE_ORDER, ArtifactRef, Record
from kernel_memory.storage import MemoryStore, SqliteIndex
from kernel_memory.storage.store import LAYOUT_VERSION

ALL_IDS = {
    "kernel-demo", "cfg-demo", "baseline-demo", "pr-demo-101", "commit-demo-a", "commit-demo-b",
    "snapshot-demo-101", "pr-demo-102", "commit-demo-c", "commit-demo-a-in-102", "snapshot-demo-102",
    "relation-demo-origin", "run-demo-baseline", "run-demo-a", "run-demo-c", "run-demo-c-failure",
    "decision-demo-blocked", "annotation-demo-grouped",
}
ARTIFACT_IDS = {
    "run-demo-baseline-samples", "run-demo-baseline-correctness", "run-demo-a-samples", "run-demo-a-correctness",
    "run-demo-c-samples", "run-demo-c-correctness", "mock-spill-report",
}


# ----------------------------------------------------------------------------- helpers
def shuffled(records: list[Record], seed: int = 7) -> list[Record]:
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


def annotation(
    bundle_dicts: list[dict],
    record_id: str,
    *,
    target: str = "commit-demo-a",
    text: str | None = None,
    evidence: list[str] | None = None,
    supersedes: str | None = None,
) -> Record:
    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["record_id"] = record_id
    data["payload"]["target_ref"] = target
    data["payload"]["text"] = text if text is not None else f"note {record_id}"
    data["payload"]["evidence_refs"] = list(evidence or [])
    data["payload"]["supersedes_ref"] = supersedes
    return Record.from_dict(data)


def run_like(bundle_dicts: list[dict], record_id: str, *, base: str = "run-demo-a", **payload: object) -> Record:
    data = record_dict(bundle_dicts, base)
    data["record_id"] = record_id
    data["payload"]["request_id"] = f"request-{record_id}"
    for key, value in payload.items():
        data["payload"][key] = value
    return Record.from_dict(data)


def fact_files(root: Path) -> list[str]:
    """Every file under the store except the lock/runtime area (which is not a fact)."""
    return sorted(
        str(PurePosixPath(p.relative_to(root).as_posix()))
        for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0] == ".runtime"
    )


def journal_lines(store: MemoryStore) -> list[dict]:
    path = store.root / "journal" / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def rewrite_record_file(path: Path, mutate) -> None:
    data = json.loads(path.read_text("utf-8"))
    mutate(data)
    path.write_text(dumps_readable(data), encoding="utf-8")


# ============================================================================= lifecycle
def test_init_creates_manifest_and_directories(tmp_path: Path) -> None:
    store = MemoryStore.init(tmp_path / "memory")
    manifest = store.manifest()
    assert manifest["layout_version"] == LAYOUT_VERSION
    assert manifest["store_version"] == "0.2.0"
    assert manifest["hash_version"] == "jcs-sha256-v1"
    for sub in ("kernels", "artifacts", "requests", "journal", ".runtime"):
        assert (store.root / sub).is_dir()
    assert store.records() == []
    assert store.integrity_scan().ok


def test_init_refuses_non_empty_non_store_directory(tmp_path: Path) -> None:
    root = tmp_path / "not-a-store"
    root.mkdir()
    (root / "something.txt").write_text("hello")
    with pytest.raises(InputError) as info:
        MemoryStore.init(root)
    assert info.value.code == "ROOT_NOT_EMPTY"
    assert info.value.exit_code == 2
    assert "something.txt" in info.value.details["entries"]
    assert not (root / "manifest.json").exists()
    assert sorted(p.name for p in root.iterdir()) == ["something.txt"]


def test_init_is_idempotent_on_existing_store(tmp_path: Path, bundle_records: list[Record]) -> None:
    first = MemoryStore.init(tmp_path / "memory")
    first.publish_bundle(bundle_records)
    store_id = first.manifest()["store_id"]
    again = MemoryStore.init(tmp_path / "memory")
    assert again.manifest()["store_id"] == store_id
    assert {r.record_id for r in again.records()} == ALL_IDS


def test_open_refuses_non_store(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "kernels").mkdir()
    for root in (plain, tmp_path / "does-not-exist"):
        with pytest.raises(InputError) as info:
            MemoryStore.open(root)
        assert info.value.code == "NOT_A_STORE"
        assert info.value.exit_code == 2


def test_open_refuses_unsupported_layout_version(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    MemoryStore.init(root)
    manifest = root / "manifest.json"
    data = json.loads(manifest.read_text("utf-8"))
    data["layout_version"] = LAYOUT_VERSION + 99
    manifest.write_text(json.dumps(data))
    with pytest.raises(InputError) as info:
        MemoryStore.open(root)
    assert info.value.code == "UNSUPPORTED_STORE"


def test_open_existing_store_sees_records(tmp_path: Path, bundle_records: list[Record]) -> None:
    MemoryStore.init(tmp_path / "memory").publish_bundle(bundle_records)
    reopened = MemoryStore.open(tmp_path / "memory")
    assert {r.record_id for r in reopened.records()} == ALL_IDS
    assert reopened.integrity_scan(verify_artifacts=False).ok
    # Artifacts declared present but never imported are a diagnostic, not silently accepted (T27).
    full = reopened.integrity_scan()
    assert not full.ok
    assert {p["problem"] for p in full.artifact_problems} == {"missing"}
    assert {p["artifact_id"] for p in full.artifact_problems} == ARTIFACT_IDS


# ============================================================================= publication
def test_publish_bundle_in_shuffled_order(store: MemoryStore, bundle_records: list[Record]) -> None:
    outcome = store.publish_bundle(shuffled(bundle_records), label="shuffled")
    assert sorted(outcome.published) == sorted(ALL_IDS)
    assert outcome.idempotent == []
    assert sorted(outcome.artifacts_registered) == sorted(ARTIFACT_IDS)
    assert outcome.txn_id.startswith("txn-")
    # Dependency order: parents are always published before children.
    position = {rid: i for i, rid in enumerate(outcome.published)}
    for record in store.records():
        for ref in record.record_references():
            assert position[ref.target] < position[record.record_id], (record.record_id, ref.target)
    # The journal holds one entry per record with the canonical digest and the transaction id.
    lines = journal_lines(store)
    assert len(lines) == 18
    assert {(e["record_id"], e["digest"]) for e in lines} == {(r.record_id, r.canonical_digest()) for r in bundle_records}
    assert {e["txn_id"] for e in lines} == {outcome.txn_id}
    assert store.integrity_scan(verify_artifacts=False).ok
    # Nothing pending or staged remains after a completed transaction.
    assert not list((store.root / ".runtime" / "pending").glob("*.json"))
    assert not (store.root / ".runtime" / "staging" / outcome.txn_id).exists()


def test_publish_returns_records_bitwise_stable(store: MemoryStore, bundle_records: list[Record]) -> None:
    store.publish_bundle(bundle_records)
    for original in bundle_records:
        stored = store.get(original.record_id)
        assert stored is not None
        assert stored.to_dict() == original.to_dict()
        assert stored.canonical_digest() == original.canonical_digest()


def test_reimport_is_fully_idempotent(demo_store: MemoryStore, bundle_records: list[Record], artifact_root: Path) -> None:
    before_files = fact_files(demo_store.root)
    before_bytes = {rel: (demo_store.root / rel).read_bytes() for rel in before_files}
    before_journal = journal_lines(demo_store)
    outcome = demo_store.publish_bundle(shuffled(bundle_records, seed=99), label="again")
    assert outcome.published == []
    assert sorted(outcome.idempotent) == sorted(ALL_IDS)
    assert len(outcome.idempotent) == 18
    assert outcome.artifacts_registered == []
    import_demo_bundle(demo_store, bundle_records, artifact_root)  # artifacts too
    assert fact_files(demo_store.root) == before_files
    assert {rel: (demo_store.root / rel).read_bytes() for rel in before_files} == before_bytes
    assert journal_lines(demo_store) == before_journal
    assert demo_store.integrity_scan().ok


def test_single_publish_is_idempotent(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-once")
    first = demo_store.publish(record)
    second = demo_store.publish(record)
    assert first.published == ["annotation-once"] and first.idempotent == []
    assert second.published == [] and second.idempotent == ["annotation-once"]
    assert sum(1 for e in journal_lines(demo_store) if e["record_id"] == "annotation-once") == 1


def test_identical_duplicate_inside_bundle_collapses(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-dup")
    outcome = demo_store.publish_bundle([record, record, Record.from_dict(record.to_dict())])
    assert outcome.published == ["annotation-dup"]
    assert sum(1 for e in journal_lines(demo_store) if e["record_id"] == "annotation-dup") == 1


def test_t12_same_id_different_content_conflicts_inside_bundle(store: MemoryStore, bundle_records: list[Record], bundle_dicts: list[dict]) -> None:
    variant = record_dict(bundle_dicts, "annotation-demo-grouped")
    variant["payload"]["text"] = "different text, same id"
    bundle = shuffled(bundle_records) + [Record.from_dict(variant)]
    before = fact_files(store.root)
    with pytest.raises(IdConflictError) as info:
        store.publish_bundle(bundle)
    assert info.value.code == "ID_CONFLICT"
    assert info.value.exit_code == 3
    assert info.value.details["record_id"] == "annotation-demo-grouped"
    # Nothing at all was written: no records, no journal, no pending manifest.
    assert store.index_entries() == []
    assert fact_files(store.root) == before
    assert journal_lines(store) == []
    assert not (store.root / ".runtime" / "pending").exists() or not list((store.root / ".runtime" / "pending").iterdir())


def test_t12_same_id_different_content_conflicts_against_stored_record(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    variant = record_dict(bundle_dicts, "run-demo-a")
    variant["payload"]["timing"]["median_us"] = 1  # published facts are immutable
    entries_before = demo_store.index_entries()
    files_before = fact_files(demo_store.root)
    with pytest.raises(IdConflictError) as info:
        demo_store.publish(Record.from_dict(variant))
    assert info.value.code == "ID_CONFLICT"
    assert info.value.exit_code == 3
    assert info.value.details["record_id"] == "run-demo-a"
    assert info.value.details["existing_digest"] != info.value.details["new_digest"]
    assert demo_store.index_entries() == entries_before
    assert fact_files(demo_store.root) == files_before
    assert demo_store.get("run-demo-a").payload.timing.median_us == 90


def test_t12_conflict_in_a_mixed_bundle_writes_nothing(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    good = annotation(bundle_dicts, "annotation-good")
    bad = record_dict(bundle_dicts, "annotation-demo-grouped")
    bad["payload"]["confidence"] = "supported"
    before = fact_files(demo_store.root)
    with pytest.raises(IdConflictError):
        demo_store.publish_bundle([good, Record.from_dict(bad)])
    assert fact_files(demo_store.root) == before
    assert demo_store.get("annotation-good") is None


def test_missing_reference_rejected(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    before = fact_files(demo_store.root)
    with pytest.raises(MissingReferenceError) as info:
        demo_store.publish(annotation(bundle_dicts, "annotation-dangling", target="commit-does-not-exist"))
    assert info.value.code == "MISSING_REFERENCE"
    assert info.value.exit_code == 3
    assert info.value.details["target"] == "commit-does-not-exist"
    assert fact_files(demo_store.root) == before


def test_missing_reference_inside_bundle_writes_nothing(store: MemoryStore, bundle_records: list[Record], bundle_dicts: list[dict]) -> None:
    orphan = annotation(bundle_dicts, "annotation-orphan", evidence=["run-not-in-bundle"])
    with pytest.raises(MissingReferenceError):
        store.publish_bundle(shuffled(bundle_records) + [orphan])
    assert store.index_entries() == []
    assert journal_lines(store) == []


def test_allow_dangling_permits_only_evidence_references(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-evidence", evidence=["run-not-yet-imported"])
    with pytest.raises(MissingReferenceError):
        demo_store.publish(record)
    outcome = demo_store.publish_bundle([record], allow_dangling=True)
    assert outcome.published == ["annotation-evidence"]
    # A structural parent (the target that decides the directory) can never dangle.
    with pytest.raises(MissingReferenceError):
        demo_store.publish_bundle([annotation(bundle_dicts, "annotation-nowhere", target="commit-nope")], allow_dangling=True)


def test_wrong_type_reference_rejected_against_stored_record(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    bad_run = run_like(bundle_dicts, "run-bad-subject", subject_ref="cfg-demo")
    before = demo_store.index_entries()
    with pytest.raises(MissingReferenceError) as info:
        demo_store.publish(bad_run)
    assert info.value.exit_code == 3
    assert "cfg-demo" in str(info.value)
    assert demo_store.index_entries() == before


def test_wrong_type_reference_rejected_inside_bundle(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    first = annotation(bundle_dicts, "annotation-first")
    # supersedes_ref must point at an annotation, not at a commit published in the same bundle.
    second = annotation(bundle_dicts, "annotation-second", supersedes="commit-new")
    commit = record_dict(bundle_dicts, "commit-demo-b")
    commit["record_id"] = "commit-new"
    commit["payload"]["commit_oid"] = {"algorithm": "sha1", "hex": "0000000000000000000000000000000000000099"}
    with pytest.raises(MissingReferenceError) as info:
        demo_store.publish_bundle([second, Record.from_dict(commit), first])
    assert "supersedes_ref" in str(info.value)
    assert demo_store.get("annotation-first") is None
    assert demo_store.get("commit-new") is None


def test_reference_cycle_inside_bundle_rejected(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    a = annotation(bundle_dicts, "annotation-cycle-a", supersedes="annotation-cycle-b")
    b = annotation(bundle_dicts, "annotation-cycle-b", supersedes="annotation-cycle-a")
    before = fact_files(demo_store.root)
    with pytest.raises(InvariantViolation) as info:
        demo_store.publish_bundle([a, b])
    assert info.value.code == "REFERENCE_CYCLE"
    assert info.value.exit_code == 2
    assert sorted(info.value.details["records"]) == ["annotation-cycle-a", "annotation-cycle-b"]
    assert fact_files(demo_store.root) == before


def test_supersedes_chain_without_cycle_publishes_in_order(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    a = annotation(bundle_dicts, "annotation-v1")
    b = annotation(bundle_dicts, "annotation-v2", supersedes="annotation-v1")
    outcome = demo_store.publish_bundle([b, a])
    assert outcome.published == ["annotation-v1", "annotation-v2"]


def test_publish_bundle_rejects_non_record_inputs(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    with pytest.raises(InputError):
        demo_store.publish_bundle([record_dict(bundle_dicts, "annotation-demo-grouped")])  # a dict, not a Record


def test_publish_rejects_invalid_record_id(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["record_id"] = "annotation-with-slash/evil"
    with pytest.raises(InputError):
        demo_store.publish(Record.from_dict(data))


# ============================================================================= artifact registry
def test_conflicting_artifact_descriptors_in_one_bundle_reject_before_any_write(
    store: MemoryStore, bundle_records: list[Record], bundle_dicts: list[dict]
) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-demo-a-again"
    data["payload"]["request_id"] = "request-run-demo-a-again"
    data["payload"]["artifacts"][0]["sha256"] = sha256_bytes(b"different bytes for the same artifact_id")
    conflicting = Record.from_dict(data)
    before = fact_files(store.root)
    with pytest.raises(ConflictError) as info:
        store.publish_bundle(shuffled(bundle_records) + [conflicting])
    assert info.value.code == "ARTIFACT_CONFLICT"
    assert info.value.exit_code == 3
    assert info.value.details["artifact_id"] == "run-demo-a-samples"
    assert store.index_entries() == []
    assert fact_files(store.root) == before
    assert store.artifact_registry() == []
    assert journal_lines(store) == []


def test_conflicting_artifact_descriptor_against_registry_rejected(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-demo-a-rerun"
    data["payload"]["request_id"] = "request-run-demo-a-rerun"
    data["payload"]["artifacts"][1]["size_bytes"] = 999
    entries = demo_store.index_entries()
    files = fact_files(demo_store.root)
    with pytest.raises(ConflictError) as info:
        demo_store.publish(Record.from_dict(data))
    assert info.value.code == "ARTIFACT_CONFLICT"
    assert demo_store.index_entries() == entries
    assert fact_files(demo_store.root) == files
    assert demo_store.get_artifact_ref("run-demo-a-correctness").size_bytes == 105


def test_register_artifact_ref_is_idempotent_and_conflict_checked(demo_store: MemoryStore) -> None:
    ref = demo_store.get_artifact_ref("run-demo-a-samples")
    assert ref is not None
    assert demo_store.register_artifact_ref(ref) is False  # already registered, identical
    fresh = ArtifactRef(
        artifact_id="extra-artifact", kind="compile_log", uri="artifact://sha256/" + sha256_bytes(b"x")[7:],
        sha256=sha256_bytes(b"x"), size_bytes=1, media_type="text/plain", retention="permanent", availability="present",
    )
    assert demo_store.register_artifact_ref(fresh) is True
    assert demo_store.register_artifact_ref(fresh) is False
    with pytest.raises(ConflictError) as info:
        demo_store.register_artifact_ref(ArtifactRef(**{**fresh.__dict__, "size_bytes": 2}))
    assert info.value.code == "ARTIFACT_CONFLICT"
    assert demo_store.get_artifact_ref("extra-artifact").size_bytes == 1
    assert demo_store.get_artifact_ref("never-registered") is None


# ============================================================================= artifact bytes
def test_put_and_read_artifact_round_trip(store: MemoryStore) -> None:
    data = b'{"samples": [1, 2, 3]}\n'
    digest = store.put_artifact_bytes(data)
    assert digest == artifact_digest(data)
    assert digest.startswith("sha256:")
    assert store.has_artifact(digest)
    assert store.read_artifact(digest) == data
    assert store.put_artifact_bytes(data) == digest  # content addressed: no duplicate
    blob = store.artifact_path(digest)
    assert blob.is_file() and blob.read_bytes() == data
    assert len(list((store.root / "artifacts" / "sha256").rglob("*"))) == 2  # one prefix dir + one blob


def test_read_artifact_missing(store: MemoryStore) -> None:
    missing = sha256_bytes(b"never stored")
    assert not store.has_artifact(missing)
    with pytest.raises(MissingReferenceError) as info:
        store.read_artifact(missing)
    assert info.value.code == "ARTIFACT_MISSING"
    assert info.value.exit_code == 3


def test_t27_corrupt_blob_detected_on_read(store: MemoryStore) -> None:
    data = b"evidence bytes"
    digest = store.put_artifact_bytes(data)
    store.artifact_path(digest).write_bytes(b"tampered")
    with pytest.raises(InvariantViolation) as info:
        store.read_artifact(digest)
    assert info.value.code == "ARTIFACT_CORRUPT"
    # Re-storing the genuine bytes must not silently paper over the corruption either.
    with pytest.raises(InvariantViolation) as info2:
        store.put_artifact_bytes(data)
    assert info2.value.code == "ARTIFACT_CORRUPT"


def test_put_artifact_bytes_enforces_size_limit(store: MemoryStore) -> None:
    with pytest.raises(InputError) as info:
        store.put_artifact_bytes(b"x" * 11, max_bytes=10)
    assert info.value.code == "ARTIFACT_TOO_LARGE"
    assert not list((store.root / "artifacts").rglob("*"))


def test_artifact_path_rejects_invalid_reference(store: MemoryStore) -> None:
    for bad in ("abc", "sha256:xyz", "md5:" + "0" * 32, "sha256:" + "0" * 63):
        with pytest.raises(InputError) as info:
            store.artifact_path(bad)
        assert info.value.code == "INVALID_DIGEST"


def test_import_artifact_file_round_trip(store: MemoryStore, artifact_root: Path, bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-a")
    ref = run.payload.artifact_by_id()["run-demo-a-samples"]
    digest = store.import_artifact_file(artifact_root / ref.uri, expected_sha256=ref.sha256, expected_size=ref.size_bytes)
    assert digest == ref.sha256
    assert store.read_artifact(digest) == (artifact_root / ref.uri).read_bytes()


def test_import_artifact_file_checksum_mismatch(store: MemoryStore, tmp_path: Path) -> None:
    src = tmp_path / "samples.json"
    src.write_bytes(b"[1,2,3]")
    with pytest.raises(InvariantViolation) as info:
        store.import_artifact_file(src, expected_sha256=sha256_bytes(b"something else"))
    assert info.value.code == "ARTIFACT_CHECKSUM_MISMATCH"
    assert info.value.exit_code == 2
    assert info.value.details["actual"] == artifact_digest(b"[1,2,3]")
    assert not store.has_artifact(artifact_digest(b"[1,2,3]"))


def test_import_artifact_file_size_mismatch(store: MemoryStore, tmp_path: Path) -> None:
    src = tmp_path / "samples.json"
    src.write_bytes(b"[1,2,3]")
    with pytest.raises(InvariantViolation) as info:
        store.import_artifact_file(src, expected_sha256=artifact_digest(b"[1,2,3]"), expected_size=99)
    assert info.value.code == "ARTIFACT_SIZE_MISMATCH"
    assert not store.has_artifact(artifact_digest(b"[1,2,3]"))


def test_import_artifact_file_missing_or_too_large(store: MemoryStore, tmp_path: Path) -> None:
    with pytest.raises(InputError) as info:
        store.import_artifact_file(tmp_path / "nope.json")
    assert info.value.code == "ARTIFACT_MISSING"
    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * 32)
    with pytest.raises(InputError) as info2:
        store.import_artifact_file(big, max_bytes=16)
    assert info2.value.code == "ARTIFACT_TOO_LARGE"


def test_t27_verify_artifact_reports_missing_corrupt_and_size_mismatch(demo_store: MemoryStore) -> None:
    ok_ref = demo_store.get_artifact_ref("run-demo-a-samples")
    assert demo_store.verify_artifact(ok_ref) is None

    missing_ref = ArtifactRef(**{**ok_ref.__dict__, "artifact_id": "gone", "sha256": sha256_bytes(b"absent")})
    problem = demo_store.verify_artifact(missing_ref)
    assert problem["problem"] == "missing" and problem["artifact_id"] == "gone"
    # An artifact declared as not present is not "missing": absence is the recorded state (tombstone).
    expired = ArtifactRef(**{**missing_ref.__dict__, "availability": "expired"})
    assert demo_store.verify_artifact(expired) is None

    size_ref = ArtifactRef(**{**ok_ref.__dict__, "size_bytes": ok_ref.size_bytes + 1})
    assert demo_store.verify_artifact(size_ref)["problem"] == "size_mismatch"

    demo_store.artifact_path(ok_ref.sha256).write_bytes(b"garbage")
    assert demo_store.verify_artifact(ok_ref)["problem"] == "corrupt"


def test_t27_integrity_scan_reports_artifact_problems(demo_store: MemoryStore) -> None:
    assert demo_store.integrity_scan().ok
    ref = demo_store.get_artifact_ref("run-demo-c-samples")
    demo_store.artifact_path(ref.sha256).write_bytes(b"not the samples")
    report = demo_store.integrity_scan()
    assert not report.ok
    assert report.artifacts_checked == 7
    assert [p["artifact_id"] for p in report.artifact_problems] == ["run-demo-c-samples"]
    assert report.artifact_problems[0]["problem"] == "corrupt"
    # Records themselves are untouched: the diagnostic is scoped to the evidence.
    assert report.modified == [] and report.missing == [] and report.corrupt == []
    assert demo_store.integrity_scan(verify_artifacts=False).ok
    demo_store.artifact_path(ref.sha256).unlink()
    assert demo_store.integrity_scan().artifact_problems[0]["problem"] == "missing"


# ============================================================================= integrity scan
def test_integrity_scan_ok_on_demo_store(demo_store: MemoryStore) -> None:
    report = demo_store.integrity_scan()
    assert report.ok
    assert report.records_checked == 18
    assert report.artifacts_checked == 7
    assert report.to_dict()["ok"] is True


def test_integrity_scan_reports_modified_record(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("annotation-demo-grouped")
    rewrite_record_file(path, lambda d: d["payload"].update({"text": "hand-edited history"}))
    demo_store.invalidate_index()
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert not report.ok
    assert [m["record_id"] for m in report.modified] == ["annotation-demo-grouped"]
    assert report.modified[0]["journal_digest"] != report.modified[0]["file_digest"]
    assert report.missing == [] and report.corrupt == [] and report.unjournaled == []


def test_integrity_scan_reports_missing_record(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("decision-demo-blocked")
    path.unlink()
    demo_store.invalidate_index()
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert not report.ok
    assert report.missing == [{"record_id": "decision-demo-blocked", "expected_path": str(path.relative_to(demo_store.root))}]
    assert report.records_checked == 17


def test_integrity_scan_reports_corrupt_record(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("relation-demo-origin")
    path.write_bytes(b"{ this is not json")
    demo_store.invalidate_index()
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert not report.ok
    assert [c["path"] for c in report.corrupt] == [str(path.relative_to(demo_store.root))]
    # The journaled record is no longer found as a valid record: reported as missing as well.
    assert [m["record_id"] for m in report.missing] == ["relation-demo-origin"]
    # Schema-invalid JSON is corrupt too, not silently accepted.
    path.write_text(json.dumps({"record_type": "relation", "record_id": "relation-demo-origin"}))
    assert len(demo_store.integrity_scan(verify_artifacts=False).corrupt) == 1


def test_integrity_scan_reports_unjournaled_record(demo_store: MemoryStore) -> None:
    src = demo_store.record_path("annotation-demo-grouped")
    dest = src.with_name("annotation-hand-copied.json")
    rewrite_record_file(src, lambda d: None)  # keep the original untouched
    data = json.loads(src.read_text("utf-8"))
    data["record_id"] = "annotation-hand-copied"
    dest.write_text(dumps_readable(data), encoding="utf-8")
    demo_store.invalidate_index()
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert not report.ok
    assert report.unjournaled == [{"record_id": "annotation-hand-copied", "path": str(dest.relative_to(demo_store.root))}]
    assert report.modified == [] and report.duplicate_ids == []
    assert report.records_checked == 19


def test_integrity_scan_reports_duplicate_ids(demo_store: MemoryStore) -> None:
    src = demo_store.record_path("annotation-demo-grouped")
    dest = src.with_name("annotation-demo-grouped-copy.json")
    dest.write_bytes(src.read_bytes())
    demo_store.invalidate_index()
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert not report.ok
    assert len(report.duplicate_ids) == 1
    dup = report.duplicate_ids[0]
    assert dup["record_id"] == "annotation-demo-grouped"
    assert sorted(dup["paths"]) == sorted([str(src.relative_to(demo_store.root)), str(dest.relative_to(demo_store.root))])
    # The in-memory scan refuses ambiguous stores rather than picking one file silently.
    with pytest.raises(InvariantViolation) as info:
        demo_store.records()
    assert info.value.code == "DUPLICATE_RECORD_FILE"


def test_integrity_scan_ignores_generated_views(demo_store: MemoryStore) -> None:
    demo_store.write_view("cfg-demo", "trajectory.json", b'{"nodes": []}')
    demo_store.write_view("cfg-demo", "memory_records.jsonl", b"{}\n")
    demo_store.write_view("cfg-demo", "context.json", b"{}")
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert report.ok
    assert report.records_checked == 18
    assert len(demo_store.index_entries()) == 18


# ============================================================================= reads
def test_records_are_ordered_by_type_then_id(demo_store: MemoryStore) -> None:
    ids = [r.record_id for r in demo_store.records()]
    keyed = [(TYPE_ORDER[r.record_type], r.record_id) for r in demo_store.records()]
    assert keyed == sorted(keyed)
    assert ids[0] == "kernel-demo" and ids[1] == "cfg-demo"
    assert [r.record_id for r in demo_store.records("run")] == ["run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure"]
    assert [e.record_id for e in demo_store.index_entries()] == ids


def test_get_require_exists_and_lookups(demo_store: MemoryStore) -> None:
    assert demo_store.get("nope") is None
    assert demo_store.exists("run-demo-a") and not demo_store.exists("nope")
    assert demo_store.require("run-demo-a", "run").record_type == "run"
    with pytest.raises(MissingReferenceError) as info:
        demo_store.require("run-demo-a", "commit")
    assert info.value.details["record_type"] == "run"
    with pytest.raises(MissingReferenceError):
        demo_store.require("nope")
    with pytest.raises(MissingReferenceError):
        demo_store.record_path("nope")
    assert demo_store.kernel_by_kernel_id("demo_vector_add").record_id == "kernel-demo"
    assert demo_store.kernel_by_kernel_id("unknown") is None
    cfg = demo_store.require("cfg-demo")
    assert [c.record_id for c in demo_store.configs_by_hash(cfg.payload.config_hash)] == ["cfg-demo"]
    assert demo_store.configs_by_hash(sha256_bytes(b"other")) == []


def test_invalidate_index_picks_up_out_of_band_files(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-out-of-band")
    dest = demo_store.record_path("annotation-demo-grouped").with_name("annotation-out-of-band.json")
    dest.write_text(dumps_readable(record.to_dict()), encoding="utf-8")
    assert demo_store.get("annotation-out-of-band") is None  # cached index
    demo_store.invalidate_index()
    assert demo_store.get("annotation-out-of-band").canonical_digest() == record.canonical_digest()


# ============================================================================= views
def test_write_read_delete_views(demo_store: MemoryStore) -> None:
    assert demo_store.read_view("cfg-demo", "trajectory.json") is None
    path = demo_store.write_view("cfg-demo", "trajectory.json", b'{"v": 1}')
    assert path == demo_store.config_dir("cfg-demo") / "trajectory.json"
    assert demo_store.read_view("cfg-demo", "trajectory.json") == b'{"v": 1}'
    demo_store.write_view("cfg-demo", "trajectory.json", b'{"v": 2}')  # views are overwritable
    assert demo_store.read_view("cfg-demo", "trajectory.json") == b'{"v": 2}'
    demo_store.write_view("cfg-demo", "memory_records.jsonl", b"{}\n")
    assert demo_store.delete_views("cfg-demo") == ["memory_records.jsonl", "trajectory.json"]
    assert demo_store.read_view("cfg-demo", "trajectory.json") is None
    assert demo_store.delete_views("cfg-demo") == []
    # Views never enter the authoritative index.
    assert {e.record_id for e in demo_store.index_entries()} == ALL_IDS


def test_write_view_rejects_non_view_names_and_non_configs(demo_store: MemoryStore) -> None:
    with pytest.raises(InputError) as info:
        demo_store.write_view("cfg-demo", "config.json", b"{}")
    assert info.value.code == "INVALID_VIEW"
    with pytest.raises(InputError):
        demo_store.write_view("cfg-demo", "../trajectory.json", b"{}")
    with pytest.raises(MissingReferenceError):
        demo_store.write_view("commit-demo-a", "trajectory.json", b"{}")
    assert demo_store.get("cfg-demo").payload.config_id == "demo-n16-f32"


# ============================================================================= fact files (T28)
def test_write_fact_absent_or_identical(store: MemoryStore) -> None:
    assert store.write_fact("requests/request-1/request.json", b'{"a": 1}') == "written"
    assert store.write_fact("requests/request-1/request.json", b'{"a": 1}') == "identical"
    assert store.read_fact("requests/request-1/request.json") == b'{"a": 1}'
    assert store.read_fact("requests/request-1/missing.json") is None


def test_write_fact_conflicts_on_different_bytes(store: MemoryStore) -> None:
    store.write_fact("requests/request-1/request.json", b'{"a": 1}')
    with pytest.raises(ConflictError) as info:
        store.write_fact("requests/request-1/request.json", b'{"a": 2}')
    assert info.value.code == "FILE_CONFLICT"
    assert info.value.exit_code == 3
    assert store.read_fact("requests/request-1/request.json") == b'{"a": 1}'


@pytest.mark.parametrize(
    "relpath",
    [
        "kernels/demo_vector_add/kernel.json",
        "artifacts/registry/x.json",
        "manifest.json",
        "journal/records.jsonl",
        "requests/../kernels/x.json",
        "requests/a/../../manifest.json",
        "/etc/passwd",
        "../outside.json",
        "requests/evil\x00.json",
        "",
    ],
)
def test_t28_write_fact_refuses_paths_outside_requests(store: MemoryStore, relpath: str) -> None:
    before = fact_files(store.root)
    with pytest.raises(UnsafePathError) as info:
        store.write_fact(relpath, b"{}")
    assert info.value.exit_code == 7
    assert info.value.code == "UNSAFE_PATH"
    assert fact_files(store.root) == before
    assert (store.root / "manifest.json").read_bytes() != b"{}"


def test_t28_write_fact_refuses_symlink_escape(store: MemoryStore, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store.root / "requests" / "link")
    with pytest.raises(UnsafePathError):
        store.write_fact("requests/link/leak.json", b"{}")
    assert list(outside.iterdir()) == []


def test_write_runtime_state_overwrites_and_is_limited_to_runtime(store: MemoryStore) -> None:
    assert store.write_runtime_state(".runtime/leases/req-1.json", b"v1") == "written"
    assert store.write_runtime_state(".runtime/leases/req-1.json", b"v2") == "written"
    assert store.write_runtime_state(".runtime/leases/req-1.json", b"v2") == "identical"
    assert store.read_fact(".runtime/leases/req-1.json") == b"v2"
    for bad in ("requests/request-1/request.json", "kernels/x.json", ".runtime/../manifest.json", "/tmp/x"):
        with pytest.raises(UnsafePathError) as info:
            store.write_runtime_state(bad, b"{}")
        assert info.value.exit_code == 7
    assert store.read_fact("requests/request-1/request.json") is None


def test_list_facts(store: MemoryStore) -> None:
    assert store.list_facts("requests") == []
    store.write_fact("requests/request-b/request.json", b"{}")
    store.write_fact("requests/request-a/request.json", b"{}")
    store.write_fact("requests/request-a/events/0001-event-x.json", b"{}")
    (store.root / "requests" / "request-a" / ".hidden").write_bytes(b"")
    assert store.list_facts("requests") == [
        "requests/request-a/events/0001-event-x.json",
        "requests/request-a/request.json",
        "requests/request-b/request.json",
    ]
    assert store.list_facts("requests/request-a") == [
        "requests/request-a/events/0001-event-x.json",
        "requests/request-a/request.json",
    ]
    assert store.list_facts("requests/request-zzz") == []
    with pytest.raises(UnsafePathError):
        store.list_facts("../")


# ============================================================================= SQLite index
def _index(store: MemoryStore) -> SqliteIndex:
    return SqliteIndex(SqliteIndex.default_path(store))


def test_index_absent_is_not_fresh(demo_store: MemoryStore) -> None:
    idx = _index(demo_store)
    assert not idx.exists()
    assert idx.fingerprint() is None
    assert idx.is_fresh(demo_store) is False


def test_index_rebuild_fresh_stale_fresh(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    path = demo_store.rebuild_index()
    assert path == demo_store.root / ".cache" / "index.sqlite"
    idx = _index(demo_store)
    assert idx.exists() and idx.is_fresh(demo_store) is True
    demo_store.publish(annotation(bundle_dicts, "annotation-stale-maker"))
    assert idx.is_fresh(demo_store) is False
    demo_store.rebuild_index()
    assert idx.is_fresh(demo_store) is True
    assert "annotation-stale-maker" in idx.record_ids(record_type="annotation")


def test_index_commits_by_component(demo_store: MemoryStore) -> None:
    demo_store.rebuild_index()
    idx = _index(demo_store)
    assert idx.commits_by_component("tiling") == ["commit-demo-a"]
    assert idx.commits_by_component("pipeline") == ["commit-demo-c"]
    assert idx.commits_by_component("layout") == ["commit-demo-c"]
    assert idx.commits_by_component("no-such-component") == []


def test_index_record_ids_filters(demo_store: MemoryStore) -> None:
    demo_store.rebuild_index()
    idx = _index(demo_store)
    assert idx.record_ids() == sorted(ALL_IDS)
    assert idx.record_ids(record_type="run") == ["run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure"]
    assert idx.record_ids(record_type="kernel") == ["kernel-demo"]
    assert idx.record_ids(subject_ref="commit-demo-c") == ["run-demo-c", "run-demo-c-failure"]
    assert idx.record_ids(subject_ref="commit-demo-b") == []
    assert idx.record_ids(config_ref="cfg-demo") == [
        "baseline-demo", "decision-demo-blocked", "pr-demo-101", "pr-demo-102", "relation-demo-origin",
        "run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure",
    ]
    assert idx.record_ids(record_type="run", config_ref="cfg-demo", subject_ref="baseline-demo") == ["run-demo-baseline"]
    assert idx.record_ids(record_type="nope") == []


def test_t24_delete_cache_and_views_then_rebuild_identical(demo_store: MemoryStore) -> None:
    demo_store.write_view("cfg-demo", "trajectory.json", b'{"nodes": []}')
    demo_store.write_view("cfg-demo", "memory_records.jsonl", b"{}\n")
    demo_store.rebuild_index()
    idx = _index(demo_store)
    rows_before = list(idx.rows())
    fingerprint_before = idx.fingerprint()
    assert len(rows_before) == 18
    assert {(r[0], r[1], r[2], r[3]) for r in rows_before} == {
        (e.record_id, e.record_type, e.relpath, e.digest) for e in demo_store.index_entries()
    }
    # Delete every derived file.
    import shutil

    shutil.rmtree(demo_store.root / ".cache")
    assert demo_store.delete_views("cfg-demo") == ["memory_records.jsonl", "trajectory.json"]
    assert not idx.exists() and idx.is_fresh(demo_store) is False
    # Authoritative facts are untouched and the index reconstructs deterministically.
    assert demo_store.integrity_scan().ok
    demo_store.rebuild_index()
    assert list(idx.rows()) == rows_before
    assert idx.fingerprint() == fingerprint_before
    assert idx.is_fresh(demo_store) is True
    assert idx.commits_by_component("tiling") == ["commit-demo-a"]


def test_index_survives_garbage_cache_file(demo_store: MemoryStore) -> None:
    demo_store.rebuild_index()
    idx = _index(demo_store)
    assert idx.is_fresh(demo_store) is True
    idx.path.write_bytes(b"this is not a sqlite database \x00\x01\x02" * 40)
    assert idx.exists()
    assert idx.fingerprint() is None
    assert idx.is_fresh(demo_store) is False
    # Truncated file too.
    idx.path.write_bytes(b"")
    assert idx.is_fresh(demo_store) is False
    demo_store.rebuild_index()
    assert idx.is_fresh(demo_store) is True
    assert len(list(idx.rows())) == 18


def test_index_of_empty_store(store: MemoryStore) -> None:
    store.rebuild_index()
    idx = _index(store)
    assert idx.is_fresh(store) is True
    assert list(idx.rows()) == []
    assert idx.record_ids() == []
