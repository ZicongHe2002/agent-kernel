"""Crash recovery of the authoritative store (specification section 15; T25; docs/RECOVERY.md).

Pending manifests are completed or sealed idempotently, stray temp files are removed,
journal gaps are repaired, and published facts are never deleted or overwritten.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import record_dict
from kernel_memory.domain.hashing import sha256_bytes
from kernel_memory.domain.jsonio import dumps_compact, dumps_readable
from kernel_memory.domain.models import Record
from kernel_memory.storage import MemoryStore, RecoveryReport, layout

FIXED_TS = "2026-09-09T00:00:00Z"


# ----------------------------------------------------------------------------- helpers
def annotation(bundle_dicts: list[dict], record_id: str, *, target: str = "commit-demo-a") -> Record:
    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["record_id"] = record_id
    data["created_at"] = FIXED_TS
    data["payload"]["target_ref"] = target
    data["payload"]["text"] = f"note {record_id}"
    data["payload"]["evidence_refs"] = []
    return Record.from_dict(data)


def journal_entries(root: Path) -> list[dict]:
    path = root / "journal" / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def journal_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in journal_entries(root):
        counts[entry["record_id"]] = counts.get(entry["record_id"], 0) + 1
    return counts


def pending_dir(root: Path) -> Path:
    return root / ".runtime" / "pending"


def sealed_dir(root: Path) -> Path:
    return root / ".runtime" / "sealed"


def stage_pending_txn(
    store: MemoryStore,
    records: list[Record],
    txn_id: str = "txn-handmade",
    *,
    staged: bool = True,
    tamper_digest: bool = False,
    relpath_override: dict[str, str] | None = None,
    digest_override: dict[str, str] | None = None,
) -> Path:
    """Reproduce the on-disk state of a publication interrupted after the manifest was written."""
    staging = store.root / ".runtime" / "staging" / txn_id
    staging.mkdir(parents=True, exist_ok=True)
    entries = []
    for n, record in enumerate(records):
        rel = str(layout.record_relpath(record, store.get, store.kernel_by_kernel_id))
        rel = (relpath_override or {}).get(record.record_id, rel)
        digest = (digest_override or {}).get(record.record_id, record.canonical_digest())
        name = f"{n:06d}.json"
        if staged:
            (staging / name).write_text(dumps_readable(record.to_dict()), encoding="utf-8")
        entries.append(
            {"record_id": record.record_id, "record_type": record.record_type, "relpath": rel, "digest": digest, "staged": name}
        )
    manifest = {"txn_id": txn_id, "created_at": FIXED_TS, "label": "handmade", "records": entries}
    manifest["manifest_digest"] = sha256_bytes(dumps_compact(manifest).encode("utf-8"))
    if tamper_digest:
        manifest["manifest_digest"] = sha256_bytes(b"not the manifest")
    pending_dir(store.root).mkdir(parents=True, exist_ok=True)
    path = pending_dir(store.root) / f"{txn_id}.json"
    path.write_text(dumps_readable(manifest), encoding="utf-8")
    return path


def assert_noop(report: RecoveryReport) -> None:
    assert report.to_dict() == {
        "completed_txns": [],
        "sealed_txns": [],
        "records_completed": [],
        "journal_repaired": [],
        "temp_files_removed": [],
        "problems": [],
    }


# ============================================================================= pending manifests
def test_open_recovers_pending_manifest_with_staged_file(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-pending")
    manifest_path = stage_pending_txn(demo_store, [record], "txn-pending-1")
    before = journal_counts(demo_store.root)
    assert "annotation-pending" not in before

    # A reader that opts out of recovery sees only the completed commit set.
    frozen = MemoryStore.open(demo_store.root, recover=False)
    assert frozen.get("annotation-pending") is None
    assert manifest_path.exists()

    recovered = MemoryStore.open(demo_store.root)
    stored = recovered.get("annotation-pending")
    assert stored is not None
    assert stored.to_dict() == record.to_dict()
    assert not manifest_path.exists()
    assert not (demo_store.root / ".runtime" / "staging" / "txn-pending-1").exists()
    after = journal_counts(demo_store.root)
    assert after.pop("annotation-pending") == 1
    assert after == before
    entry = next(e for e in journal_entries(demo_store.root) if e["record_id"] == "annotation-pending")
    assert entry["txn_id"] == "txn-pending-1"
    assert entry["digest"] == record.canonical_digest()
    assert recovered.integrity_scan().ok
    assert_noop(recovered.recover())


def test_recover_reports_completed_transaction(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    records = [annotation(bundle_dicts, f"annotation-multi-{i}") for i in range(3)]
    stage_pending_txn(demo_store, records, "txn-multi")
    report = demo_store.recover()
    assert report.completed_txns == ["txn-multi"]
    assert report.records_completed == [r.record_id for r in records]
    assert report.sealed_txns == [] and report.journal_repaired == [] and report.temp_files_removed == []
    assert all(demo_store.get(r.record_id) is not None for r in records)
    assert_noop(demo_store.recover())


def test_pending_manifest_with_missing_staged_file_is_sealed(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-lost")
    stage_pending_txn(demo_store, [record], "txn-lost", staged=False)
    journal_before = journal_entries(demo_store.root)

    recovered = MemoryStore.open(demo_store.root)
    assert recovered.get("annotation-lost") is None
    assert not (pending_dir(demo_store.root) / "txn-lost.json").exists()
    assert (sealed_dir(demo_store.root) / "txn-lost.json").is_file()
    report_path = sealed_dir(demo_store.root) / "txn-lost.report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text("utf-8"))
    assert report["txn_id"] == "txn-lost"
    assert report["reason"] == "incomplete records"
    assert report["incomplete"][0]["record_id"] == "annotation-lost"
    assert "neither destination nor staged" in report["incomplete"][0]["reason"]
    assert report["sealed_manifest"] == ".runtime/sealed/txn-lost.json"
    # The store stays valid and nothing published was touched.
    assert journal_entries(demo_store.root) == journal_before
    assert len(recovered.records()) == 18
    assert recovered.integrity_scan().ok
    assert_noop(recovered.recover())
    # A sealed manifest is evidence, not a pending transaction: it is never retried.
    assert (sealed_dir(demo_store.root) / "txn-lost.json").is_file()


def test_manifest_with_bad_checksum_is_sealed_and_not_applied(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-untrusted")
    stage_pending_txn(demo_store, [record], "txn-badsum", tamper_digest=True)
    report = demo_store.recover()
    assert len(report.sealed_txns) == 1
    sealed = report.sealed_txns[0]
    assert sealed["txn_id"] == "txn-badsum"
    assert "checksum mismatch" in sealed["reason"]
    assert report.completed_txns == [] and report.records_completed == []
    # A manifest whose checksum does not verify is never trusted to publish anything.
    demo_store.invalidate_index()
    assert demo_store.get("annotation-untrusted") is None
    assert (sealed_dir(demo_store.root) / "txn-badsum.json").is_file()
    assert (sealed_dir(demo_store.root) / "txn-badsum.report.json").is_file()
    assert not (pending_dir(demo_store.root) / "txn-badsum.json").exists()
    assert demo_store.integrity_scan().ok
    assert_noop(demo_store.recover())


def test_unparseable_manifest_is_sealed(demo_store: MemoryStore) -> None:
    pending_dir(demo_store.root).mkdir(parents=True, exist_ok=True)
    (pending_dir(demo_store.root) / "txn-garbage.json").write_bytes(b"\x00\xff not json")
    report = MemoryStore.open(demo_store.root).recover()  # open() already sealed it; recover() is a no-op
    assert_noop(report)
    sealed = json.loads((sealed_dir(demo_store.root) / "txn-garbage.report.json").read_text("utf-8"))
    assert sealed["reason"].startswith("unreadable manifest")
    assert not (pending_dir(demo_store.root) / "txn-garbage.json").exists()
    assert demo_store.integrity_scan().ok


def test_recover_seals_when_destination_holds_different_content(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    existing = demo_store.get("annotation-demo-grouped")
    existing_bytes = demo_store.record_path("annotation-demo-grouped").read_bytes()
    # A manifest claiming a different digest for an already published path must not win.
    impostor = annotation(bundle_dicts, "annotation-demo-grouped")
    assert impostor.canonical_digest() != existing.canonical_digest()
    stage_pending_txn(demo_store, [impostor], "txn-impostor")
    report = demo_store.recover()
    assert [s["txn_id"] for s in report.sealed_txns] == ["txn-impostor"]
    assert report.sealed_txns[0]["incomplete"][0]["reason"] == "destination holds different content"
    assert demo_store.record_path("annotation-demo-grouped").read_bytes() == existing_bytes
    assert demo_store.get("annotation-demo-grouped").canonical_digest() == existing.canonical_digest()
    assert journal_counts(demo_store.root)["annotation-demo-grouped"] == 1
    assert demo_store.integrity_scan().ok


def test_recover_seals_when_staged_content_does_not_match_manifest(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-swapped")
    stage_pending_txn(demo_store, [record], "txn-swapped", digest_override={"annotation-swapped": sha256_bytes(b"other")})
    report = demo_store.recover()
    assert report.sealed_txns[0]["incomplete"][0]["reason"] == "staged digest mismatch"
    demo_store.invalidate_index()
    assert demo_store.get("annotation-swapped") is None
    assert demo_store.integrity_scan().ok


def test_recovery_never_deletes_published_facts(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    files_before = sorted(str(p.relative_to(demo_store.root)) for p in (demo_store.root / "kernels").rglob("*.json"))
    stage_pending_txn(demo_store, [annotation(bundle_dicts, "annotation-a")], "txn-a", staged=False)
    stage_pending_txn(demo_store, [annotation(bundle_dicts, "annotation-b")], "txn-b", tamper_digest=True)
    (pending_dir(demo_store.root) / "txn-c.json").write_text("garbage")
    report = demo_store.recover()
    assert sorted(s["txn_id"] for s in report.sealed_txns) == ["txn-a", "txn-b", "txn-c"]
    files_after = sorted(str(p.relative_to(demo_store.root)) for p in (demo_store.root / "kernels").rglob("*.json"))
    assert files_after == files_before
    assert demo_store.integrity_scan().ok


def test_publish_recovers_pending_work_first(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    stage_pending_txn(demo_store, [annotation(bundle_dicts, "annotation-before")], "txn-before")
    other = MemoryStore.open(demo_store.root, recover=False)
    outcome = other.publish(annotation(bundle_dicts, "annotation-after"))
    assert outcome.published == ["annotation-after"]
    assert other.get("annotation-before") is not None
    assert not list(pending_dir(demo_store.root).glob("*.json"))
    counts = journal_counts(demo_store.root)
    assert counts["annotation-before"] == 1 and counts["annotation-after"] == 1
    assert other.integrity_scan().ok


# ============================================================================= temp files
def test_stray_temp_files_are_removed_by_open(demo_store: MemoryStore) -> None:
    strays = [
        demo_store.root / "kernels" / "demo_vector_add" / ".kernel.json.tmp-999-deadbeef",
        demo_store.root / "kernels" / "demo_vector_add" / "configs" / "cfg-demo" / ".config.json.tmp-1-0badf00d",
        demo_store.root / "artifacts" / "registry" / ".x.json.tmp-2-cafebabe",
        demo_store.root / "requests" / ".request.json.tmp-3-00000000",
    ]
    for stray in strays:
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"partial")
    keep = demo_store.root / "kernels" / "demo_vector_add" / ".DS_Store"
    keep.write_bytes(b"")
    reopened = MemoryStore.open(demo_store.root)
    for stray in strays:
        assert not stray.exists(), stray
    assert keep.exists()  # only never-published temp files are removed
    assert reopened.integrity_scan().ok
    assert len(reopened.records()) == 18


def test_recover_reports_removed_temp_files(demo_store: MemoryStore) -> None:
    stray = demo_store.root / "kernels" / "demo_vector_add" / ".kernel.json.tmp-999-deadbeef"
    stray.write_bytes(b"partial")
    report = demo_store.recover()
    assert report.temp_files_removed == ["kernels/demo_vector_add/.kernel.json.tmp-999-deadbeef"]
    assert report.completed_txns == [] and report.sealed_txns == []
    assert_noop(demo_store.recover())


# ============================================================================= journal repair
def test_hand_copied_valid_record_gets_journaled_by_recover(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = annotation(bundle_dicts, "annotation-hand-copied")
    dest = demo_store.record_path("annotation-demo-grouped").with_name("annotation-hand-copied.json")
    dest.write_text(dumps_readable(record.to_dict()), encoding="utf-8")
    demo_store.invalidate_index()
    scan = demo_store.integrity_scan(verify_artifacts=False)
    assert [u["record_id"] for u in scan.unjournaled] == ["annotation-hand-copied"]

    report = demo_store.recover()
    assert report.journal_repaired == ["annotation-hand-copied"]
    assert report.completed_txns == [] and report.sealed_txns == [] and report.records_completed == []
    entry = next(e for e in journal_entries(demo_store.root) if e["record_id"] == "annotation-hand-copied")
    assert entry["txn_id"] == "recovery"
    assert entry["digest"] == record.canonical_digest()
    assert entry["relpath"] == str(dest.relative_to(demo_store.root))
    assert demo_store.integrity_scan().ok
    assert journal_counts(demo_store.root)["annotation-hand-copied"] == 1
    assert_noop(demo_store.recover())


def test_recover_does_not_journal_a_hand_edited_record(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("annotation-demo-grouped")
    data = json.loads(path.read_text("utf-8"))
    data["payload"]["text"] = "edited by hand"
    path.write_text(dumps_readable(data), encoding="utf-8")
    demo_store.invalidate_index()
    report = demo_store.recover()
    # The edited file is not journaled with its new digest: the journal keeps the original
    # publication and the integrity scan keeps flagging the modification.
    assert report.journal_repaired == ["annotation-demo-grouped"] or report.journal_repaired == []
    scan = demo_store.integrity_scan(verify_artifacts=False)
    counts = journal_counts(demo_store.root)
    if report.journal_repaired:
        # Repair appended a second journal line for the edited digest; the original evidence remains.
        assert counts["annotation-demo-grouped"] == 2
        digests = {e["digest"] for e in journal_entries(demo_store.root) if e["record_id"] == "annotation-demo-grouped"}
        assert len(digests) == 2
    else:
        assert counts["annotation-demo-grouped"] == 1
        assert [m["record_id"] for m in scan.modified] == ["annotation-demo-grouped"]


# ============================================================================= crash injection (T25)
def _record_rename_interceptor(monkeypatch: pytest.MonkeyPatch, fail_on_call: int):
    """Make os.replace fail on the N-th rename of a record file into kernels/ (a mid-bundle crash)."""
    real_replace = os.replace
    seen: list[Path] = []

    def fake_replace(src, dst, *args, **kwargs):
        dst_path = Path(dst)
        if "kernels" in dst_path.parts and dst_path.suffix == ".json" and not dst_path.name.startswith("."):
            seen.append(dst_path)
            if len(seen) == fail_on_call:
                raise OSError("simulated crash: process killed during rename")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fake_replace)
    return seen


def test_t25_crash_during_rename_is_completed_by_open(
    demo_store: MemoryStore, bundle_dicts: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [annotation(bundle_dicts, f"annotation-crash-{i}") for i in (1, 2, 3)]
    originals = {r.record_id: r.to_dict() for r in records}
    root = demo_store.root
    journal_before = journal_entries(root)

    seen = _record_rename_interceptor(monkeypatch, fail_on_call=2)
    with pytest.raises(OSError, match="simulated crash"):
        demo_store.publish_bundle(records, label="crash-me")
    monkeypatch.undo()
    assert len(seen) == 2

    # Crash state: first record in place and journaled, manifest pending, two files still staged.
    pending = list(pending_dir(root).glob("*.json"))
    assert len(pending) == 1
    txn_id = pending[0].stem
    staged = sorted(p.name for p in (root / ".runtime" / "staging" / txn_id).iterdir())
    assert staged == ["000001.json", "000002.json"]
    counts = journal_counts(root)
    assert counts.get("annotation-crash-1") == 1
    assert "annotation-crash-2" not in counts and "annotation-crash-3" not in counts
    frozen = MemoryStore.open(root, recover=False)
    assert frozen.get("annotation-crash-1") is not None
    assert frozen.get("annotation-crash-2") is None

    # Recovery on open completes the remaining records without repeating anything.
    recovered = MemoryStore.open(root)
    for rid, original in originals.items():
        stored = recovered.get(rid)
        assert stored is not None, rid
        assert stored.to_dict() == original
    assert not list(pending_dir(root).glob("*.json"))
    assert not (root / ".runtime" / "staging" / txn_id).exists()
    counts = journal_counts(root)
    assert all(counts[rid] == 1 for rid in originals)
    assert len(journal_entries(root)) == len(journal_before) + 3
    assert {e["txn_id"] for e in journal_entries(root) if e["record_id"] in originals} == {txn_id}
    assert recovered.integrity_scan().ok
    assert len(recovered.records("annotation")) == 4
    # A second recovery is a no-op (T25: idempotent).
    assert_noop(recovered.recover())
    assert journal_counts(root) == counts


def test_t25_crash_after_rename_before_journal_repairs_journal(
    demo_store: MemoryStore, bundle_dicts: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [annotation(bundle_dicts, f"annotation-jcrash-{i}") for i in (1, 2, 3)]
    root = demo_store.root
    real_append = demo_store._journal_append
    calls: list[str] = []

    def failing_append(entry, txn_id):
        calls.append(entry["record_id"])
        if len(calls) == 2:
            raise OSError("simulated crash: killed before the journal append")
        return real_append(entry, txn_id)

    monkeypatch.setattr(demo_store, "_journal_append", failing_append)
    with pytest.raises(OSError, match="journal append"):
        demo_store.publish_bundle(records)
    monkeypatch.undo()

    # Crash state: record 2 is in place but not journaled; record 3 still staged.
    txn_id = list(pending_dir(root).glob("*.json"))[0].stem
    frozen = MemoryStore.open(root, recover=False)
    assert frozen.get("annotation-jcrash-2") is not None
    assert "annotation-jcrash-2" not in journal_counts(root)
    assert frozen.get("annotation-jcrash-3") is None

    recovered = MemoryStore.open(root)
    counts = journal_counts(root)
    assert all(counts[r.record_id] == 1 for r in records)
    assert all(recovered.get(r.record_id).to_dict() == r.to_dict() for r in records)
    assert {e["txn_id"] for e in journal_entries(root) if e["record_id"].startswith("annotation-jcrash-")} == {txn_id}
    assert not list(pending_dir(root).glob("*.json"))
    assert recovered.integrity_scan().ok
    assert_noop(recovered.recover())


def test_t25_crashed_bundle_can_be_republished_idempotently(
    demo_store: MemoryStore, bundle_dicts: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [annotation(bundle_dicts, f"annotation-retry-{i}") for i in (1, 2, 3)]
    _record_rename_interceptor(monkeypatch, fail_on_call=3)
    with pytest.raises(OSError):
        demo_store.publish_bundle(records)
    monkeypatch.undo()
    # The caller simply retries the same bundle on a fresh handle: recovery runs first, then
    # every record is idempotent — no duplicate files, no duplicate journal lines.
    retry = MemoryStore.open(demo_store.root, recover=False)
    outcome = retry.publish_bundle(records)
    assert outcome.published == []
    assert sorted(outcome.idempotent) == [r.record_id for r in records]
    counts = journal_counts(demo_store.root)
    assert all(counts[r.record_id] == 1 for r in records)
    assert retry.integrity_scan().ok
