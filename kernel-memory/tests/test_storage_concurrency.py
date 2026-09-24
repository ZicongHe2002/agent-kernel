"""Storage concurrency: one cross-process locking protocol (spec section 15, T25-adjacent).

Competing writers -- threads that each own a ``MemoryStore`` instance, or a second OS
process -- must serialize on ``.runtime/lock``. Afterwards every record exists exactly
once, ``integrity_scan`` is clean and ``journal/records.jsonl`` holds exactly one line
per record id. ``multiprocessing`` is deliberately avoided (macOS spawn re-imports test
modules); the cross-process case uses ``subprocess``. Tests coordinate with events and
ordering, never with wall-clock speed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from kernel_memory.domain.errors import ExecutionInfrastructureError
from kernel_memory.domain.models import AnnotationPayload, Record
from kernel_memory.services.common import new_record
from kernel_memory.storage import MemoryStore, layout
from kernel_memory.storage.lock import StoreLock

ROOT = Path(__file__).resolve().parents[1]
TARGET = "baseline-demo"  # a fixture record every annotation below targets
PER_WRITER = 25


# ----------------------------------------------------------------------------- helpers
def make_annotation(record_id: str, *, target: str = TARGET) -> Record:
    payload = AnnotationPayload(
        target_ref=target,
        category="note",
        text=f"concurrency probe {record_id}",
        author_kind="program",
        evidence_refs=[],
        confidence="unverified",
        supersedes_ref=None,
    )
    return new_record("annotation", record_id, payload)


def ids_for(prefix: str, count: int = PER_WRITER) -> list[str]:
    return [f"annotation-{prefix}-{i:03d}" for i in range(count)]


def journal_ids(root: Path) -> list[str]:
    journal = root / layout.JOURNAL_DIR / "records.jsonl"
    return [json.loads(line)["record_id"] for line in journal.read_text("utf-8").splitlines() if line.strip()]


def assert_serialized_outcome(root: Path, expected_new: list[str]) -> None:
    """Every expected record exists exactly once, journal has one line per id, store is clean."""
    fresh = MemoryStore.open(root)
    for rid in expected_new:
        record = fresh.get(rid)
        assert record is not None, f"{rid} was not published"
        assert record.record_type == "annotation"
    counts = Counter(journal_ids(root))
    assert [rid for rid in expected_new if counts[rid] != 1] == []
    assert max(counts.values()) == 1, "journal must hold exactly one line per record id"
    assert set(expected_new) <= set(counts)
    report = fresh.integrity_scan()
    assert report.ok, report.to_dict()
    assert fresh.recover_if_pending() is None, "no pending manifests or stray temp files may remain"
    staging = root / layout.RUNTIME_DIR / "staging"
    assert not staging.exists() or list(staging.iterdir()) == [], "staging directories must be cleaned up"


# ----------------------------------------------------------------------------- (1) threads, separate instances
def test_t25_two_threads_with_own_store_instances_serialize(demo_store: MemoryStore) -> None:
    root = demo_store.root
    writers = {"alpha": ids_for("alpha"), "beta": ids_for("beta")}
    start = threading.Barrier(len(writers))
    errors: list[BaseException] = []

    def worker(name: str) -> None:
        try:
            store = MemoryStore.open(root, recover=False)  # own instance, own lock fd
            start.wait(timeout=10)
            for rid in writers[name]:
                outcome = store.publish(make_annotation(rid))
                assert outcome.published == [rid], outcome.to_dict()
        except BaseException as exc:  # pragma: no cover - surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,), name=f"writer-{name}") for name in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert [t.name for t in threads if t.is_alive()] == []
    assert errors == [], errors
    expected = [rid for ids in writers.values() for rid in ids]
    assert len(expected) == 2 * PER_WRITER
    assert_serialized_outcome(root, expected)
    # Read paths keep a per-instance snapshot; the documented refresh points are the public
    # invalidate_index() and every mutation (publish_bundle re-reads the journal stamp before
    # resolving references), so the idle fixture instance must see the other writers' records
    # through both routes.
    stale_view = demo_store.get(expected[0])
    followup = "annotation-followup-on-foreign-record"
    outcome = demo_store.publish(make_annotation(followup, target=expected[0]))  # target written by another instance
    assert outcome.published == [followup], outcome.to_dict()
    assert demo_store.get(expected[0]) is not None and demo_store.get(expected[-1]) is not None
    demo_store.invalidate_index()
    assert {rid for rid in expected} <= {r.record_id for r in demo_store.records("annotation")}
    assert stale_view is None or stale_view.record_id == expected[0]


# ----------------------------------------------------------------------------- (2) a second OS process
CHILD_SCRIPT = r"""
import sys
from pathlib import Path
from kernel_memory.domain.models import AnnotationPayload
from kernel_memory.services.common import new_record
from kernel_memory.storage import MemoryStore

root, count, ready = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
store = MemoryStore.open(root, recover=False)
ready.write_text("ready", encoding="utf-8")
for i in range(count):
    rid = f"annotation-child-{i:03d}"
    payload = AnnotationPayload(
        target_ref="baseline-demo", category="note", text=f"concurrency probe {rid}", author_kind="program",
        evidence_refs=[], confidence="unverified", supersedes_ref=None,
    )
    outcome = store.publish(new_record("annotation", rid, payload))
    assert outcome.published == [rid], outcome.to_dict()
print("child-done", count)
"""


def test_t25_second_process_and_parent_thread_serialize(demo_store: MemoryStore, tmp_path: Path) -> None:
    root = demo_store.root
    ready = tmp_path / "child-ready"
    child_ids = ids_for("child")
    parent_ids = ids_for("parent")
    result: dict[str, object] = {}

    def run_child() -> None:
        try:
            result["completed"] = subprocess.run(
                [sys.executable, "-c", CHILD_SCRIPT, str(root), str(PER_WRITER), str(ready)],
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                timeout=120,
                capture_output=True,
                text=True,
            )
        except BaseException as exc:  # pragma: no cover - surfaced through the assertion below
            result["error"] = exc

    child = threading.Thread(target=run_child, name="child-runner")
    child.start()
    # Wait until the child has opened the store so both writers actually overlap.
    ready_seen = threading.Event()

    def watch_ready() -> None:
        while child.is_alive() and not ready.exists():
            child.join(timeout=0.01)
        ready_seen.set()

    watcher = threading.Thread(target=watch_ready, name="ready-watcher")
    watcher.start()
    assert ready_seen.wait(timeout=60)
    parent = MemoryStore.open(root, recover=False)
    for rid in parent_ids:
        outcome = parent.publish(make_annotation(rid))
        assert outcome.published == [rid], outcome.to_dict()
    child.join(timeout=130)
    watcher.join(timeout=10)
    assert not child.is_alive()
    if "error" in result:
        exc = result["error"]
        detail = getattr(exc, "stderr", "") or ""
        pytest.fail(f"child process failed: {exc!r}\n{detail}")
    completed = result["completed"]
    assert isinstance(completed, subprocess.CompletedProcess)
    assert completed.returncode == 0, completed.stderr
    assert f"child-done {PER_WRITER}" in completed.stdout
    assert_serialized_outcome(root, child_ids + parent_ids)


# ----------------------------------------------------------------------------- (3) lock timeout
def test_lock_timeout_raises_infrastructure_error_and_releases_thread_lock(tmp_path: Path) -> None:
    path = tmp_path / layout.RUNTIME_DIR / "lock"
    held = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def holder() -> None:
        lock = StoreLock(path)
        lock.acquire()
        held.set()
        release.wait(timeout=30)
        lock.release()
        done.set()

    t = threading.Thread(target=holder, name="holder")
    t.start()
    assert held.wait(timeout=10)

    second = StoreLock(path, timeout_seconds=0.2, poll_interval=0.01)
    with pytest.raises(ExecutionInfrastructureError) as info:
        second.acquire()
    assert info.value.code == "LOCK_TIMEOUT"
    assert info.value.exit_code == 6
    assert info.value.to_dict()["error"] == "LOCK_TIMEOUT"
    assert info.value.to_dict()["exit_code"] == 6
    assert not second.held
    # A failed acquire must not leave the per-instance thread lock held: another thread
    # must be able to acquire the same instance once the holder lets go.
    acquired_elsewhere = threading.Event()

    def late_acquirer() -> None:
        second.acquire()
        acquired_elsewhere.set()
        second.release()

    late = threading.Thread(target=late_acquirer, name="late")
    release.set()
    assert done.wait(timeout=10)
    t.join(timeout=10)
    late.start()
    assert acquired_elsewhere.wait(timeout=10)
    late.join(timeout=10)
    assert not second.held
    # And the main thread can now acquire it too.
    second.acquire()
    assert second.held
    second.release()
    assert not second.held


# ----------------------------------------------------------------------------- (4) re-entrancy
def test_lock_is_reentrant_per_instance_and_unbalanced_release_raises(tmp_path: Path) -> None:
    path = tmp_path / layout.RUNTIME_DIR / "lock"
    lock = StoreLock(path)
    with pytest.raises(RuntimeError):
        lock.release()
    assert not lock.held

    other = StoreLock(path, timeout_seconds=0.1, poll_interval=0.01)
    lock.acquire()
    lock.acquire()
    assert lock.held
    with pytest.raises(ExecutionInfrastructureError) as info:
        other.acquire()
    assert info.value.code == "LOCK_TIMEOUT"
    lock.release()
    assert lock.held, "one release of a doubly-acquired lock must keep it held"
    with pytest.raises(ExecutionInfrastructureError):
        other.acquire()
    lock.release()
    assert not lock.held
    other.acquire()
    assert other.held
    other.release()
    with pytest.raises(RuntimeError):
        lock.release()
    with pytest.raises(RuntimeError):
        other.release()

    with lock:
        with lock:
            assert lock.held
        assert lock.held
    assert not lock.held


# ----------------------------------------------------------------------------- (5) held lock blocks a concurrent publish
def test_publish_blocks_until_foreign_lock_is_released(demo_store: MemoryStore) -> None:
    root = demo_store.root
    holder = MemoryStore.open(root, recover=False)
    writer = MemoryStore.open(root, recover=False)
    rid = "annotation-blocked-by-holder"
    order: list[str] = []
    entered = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    holder.lock().acquire()
    try:

        def publish() -> None:
            try:
                entered.set()
                writer.publish(make_annotation(rid))
                order.append("published")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                finished.set()

        t = threading.Thread(target=publish, name="blocked-writer")
        t.start()
        assert entered.wait(timeout=10)
        # Bounded observation, not a speed claim: with the lock held the publish cannot
        # complete, so the record must be absent from disk and the journal.
        assert not finished.wait(timeout=0.5)
        assert rid not in journal_ids(root)
        assert not list((root / layout.KERNELS_DIR).rglob(f"*{rid}*.json"))
        order.append("released")
    finally:
        holder.lock().release()
    assert finished.wait(timeout=60)
    t.join(timeout=10)
    assert errors == [], errors
    assert order == ["released", "published"]
    assert journal_ids(root).count(rid) == 1
    assert holder.get(rid) is not None  # holder never scanned before the release; first scan sees it
    assert MemoryStore.open(root).integrity_scan(verify_artifacts=False).ok


def test_publish_reenters_store_lock_held_by_same_instance(demo_store: MemoryStore) -> None:
    """The store's own lock is re-entrant: publishing inside ``with store.lock()`` must not deadlock."""
    rid = "annotation-reentrant-publish"
    with demo_store.lock():
        assert demo_store.lock().held
        outcome = demo_store.publish(make_annotation(rid))
        assert demo_store.lock().held
    assert not demo_store.lock().held
    assert outcome.published == [rid]
    assert journal_ids(demo_store.root).count(rid) == 1


# ----------------------------------------------------------------------------- (6) lock file placement
def test_lock_file_lives_under_runtime_and_is_not_a_record(demo_store: MemoryStore) -> None:
    root = demo_store.root
    lock_path = demo_store.lock().path
    assert lock_path == root / layout.RUNTIME_DIR / "lock"
    assert lock_path.parent.name == ".runtime"
    before = sorted(r.record_id for r in demo_store.records())
    assert before, "demo store must hold the fixture records"
    with demo_store.lock():
        assert lock_path.is_file()
    assert lock_path.is_file()  # the lock file persists; only the flock is dropped
    rel = PurePosixPath(lock_path.relative_to(root).as_posix())
    assert not layout.is_record_file(rel)
    assert not list((root / layout.KERNELS_DIR).rglob("lock"))
    assert sorted(r.record_id for r in demo_store.records()) == before
    assert sorted(r.record_id for r in MemoryStore.open(root).records()) == before
    report = demo_store.integrity_scan(verify_artifacts=False)
    assert report.ok, report.to_dict()
    assert report.records_checked == len(before)
