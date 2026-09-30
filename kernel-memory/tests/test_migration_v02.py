"""v0.2 -> v0.3 migration tests (ADR-0004): the pure record upgrade and the layout-1 store migration.

The legacy store is built by hand under ``tmp_path`` from the verbatim 0.2.0 fixture dicts with the
layout-1 relpaths of a 0.2 store (``kernels/<k>/configs/<c>/...``), a journal whose digests are
``hashing.jcs_digest`` of the record dicts, the fixture artifacts as CAS blobs plus registry
descriptors, the old per-config view files and a small request ledger. Nothing touches the network or
the handoff directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from conftest import FIXTURE_RECORD_COUNT, PLACEHOLDER_ALGORITHM_ID, record_dict

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, InvariantViolation
from kernel_memory.domain.ids import slug_for_id
from kernel_memory.domain.jsonio import dumps_compact, dumps_readable, load_json_file
from kernel_memory.domain.models import Record
from kernel_memory.migrations.v02 import (
    DEFAULT_ALGORITHM_ID,
    LEGACY_METHOD_SUMMARY,
    LEGACY_SUMMARY_AUTHOR,
    LEGACY_TAGS,
    LegacyStoreSnapshot,
    StoreMigrationReport,
    default_algorithm_record,
    default_algorithm_record_id,
    migrate_v02_store,
    read_legacy_store,
    upgrade_bundle_v02,
    upgrade_v02_records,
)
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore

KERNEL_ID = "demo_vector_add"
LEGACY_TS = "2026-09-08T00:00:00Z"
LEGACY_TXN = "txn-legacy-0001"
LEGACY_STORE_ID = "0123456789abcdef0123456789abcdef"
CFG_DIR = "kernels/demo_vector_add/configs/cfg-demo"
REPORT_KEYS = {
    "source_root", "dest_root", "dry_run", "records_read", "records_rewritten", "records_linked", "records_synthesized",
    "records_published", "records_idempotent", "artifacts_found", "artifacts_copied", "descriptors_found", "descriptors_copied",
    "requests_found", "requests_copied", "kernel_views_published", "integrity_problems", "destination_integrity_ok", "forced",
    "provenance", "notes",
}


# --------------------------------------------------------------------------------------
# helpers: a layout-1 store written by hand
# --------------------------------------------------------------------------------------
def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def tree_digest(root: Path) -> str:
    """sha256 over every (relpath, bytes) pair under ``root``: proves a tree is byte-identical."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _legacy_config_dir(d: dict, by_id: dict[str, dict]) -> str:
    t, p = d["record_type"], d["payload"]
    if t == "config":
        return f"kernels/{slug_for_id(p['kernel_id'])}/configs/{slug_for_id(d['record_id'])}"
    if t in ("pr", "baseline", "run", "relation", "decision"):
        return _legacy_config_dir(by_id[p["config_ref"]], by_id)
    if t in ("commit", "pr_snapshot"):
        return _legacy_config_dir(by_id[p["pr_ref"]], by_id)
    if t == "annotation":
        return _legacy_config_dir(by_id[p["target_ref"]], by_id)
    raise AssertionError(f"{t} has no config directory")


def legacy_relpath(d: dict, by_id: dict[str, dict]) -> str:
    """The layout-1 (git HEAD ``storage/layout.py``) path of a record dict."""
    t, p, slug = d["record_type"], d["payload"], slug_for_id(d["record_id"])
    if t == "kernel":
        return f"kernels/{slug_for_id(p['kernel_id'])}/kernel.json"
    if t == "config":
        return f"{_legacy_config_dir(d, by_id)}/config.json"
    if t == "pr":
        return f"{_legacy_config_dir(d, by_id)}/attempt/{slug}/pr.json"
    if t in ("pr_snapshot", "commit"):
        pr = by_id[p["pr_ref"]]
        pr_dir = f"{_legacy_config_dir(pr, by_id)}/attempt/{slug_for_id(pr['record_id'])}"
        return f"{pr_dir}/snapshots/{slug}.json" if t == "pr_snapshot" else f"{pr_dir}/commits/{slug}/commit.json"
    if t == "baseline":
        return f"{_legacy_config_dir(d, by_id)}/baselines/{slug}/baseline.json"
    if t == "run":
        subject = by_id[p["subject_ref"]]
        subject_dir = legacy_relpath(subject, by_id).rsplit("/", 1)[0]
        return f"{subject_dir}/runs/{slug}/run.json"
    if t in ("relation", "decision"):
        return f"{_legacy_config_dir(d, by_id)}/{t}s/{slug}.json"
    if t == "annotation":
        target = by_id[p["target_ref"]]
        if target["record_type"] == "kernel":
            return f"kernels/{slug_for_id(target['payload']['kernel_id'])}/annotations/{slug}.json"
        return f"{_legacy_config_dir(target, by_id)}/annotations/{slug}.json"
    raise AssertionError(t)


def build_legacy_store(root: Path, legacy_dicts: list[dict], artifact_root: Path, *, with_views: bool = True, with_requests: bool = True) -> dict[str, str]:
    """Write a layout-1 (v0.2) store by hand. Returns record_id -> relpath."""
    by_id = {d["record_id"]: d for d in legacy_dicts}
    relpaths: dict[str, str] = {}
    root.mkdir(parents=True)
    for sub in ("kernels", "artifacts", "requests", "journal", ".runtime"):
        (root / sub).mkdir()
    manifest = {
        "store_version": "0.2.0",
        "schema_version": "0.2.0",
        "layout_version": 1,
        "hash_version": "jcs-sha256-v1",
        "store_id": LEGACY_STORE_ID,
        "created_at": LEGACY_TS,
        "notes": "Readable JSON files under kernels/ are authoritative. trajectory.json, memory_records.jsonl and .cache/ are derived and rebuildable.",
    }
    (root / "manifest.json").write_text(dumps_readable(manifest))
    lines: list[str] = []
    for d in legacy_dicts:
        rel = legacy_relpath(d, by_id)
        relpaths[d["record_id"]] = rel
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_readable(d))
        lines.append(
            dumps_compact(
                {"record_id": d["record_id"], "record_type": d["record_type"], "relpath": rel, "digest": hashing.jcs_digest(d), "txn_id": LEGACY_TXN, "published_at": LEGACY_TS}
            )
        )
    (root / "journal" / "records.jsonl").write_text("\n".join(lines) + "\n")
    (root / "artifacts" / "registry").mkdir()
    for d in legacy_dicts:
        if d["record_type"] != "run":
            continue
        for ref in d["payload"]["artifacts"]:
            data = (artifact_root / ref["uri"]).read_bytes()
            assert hashing.artifact_digest(data) == ref["sha256"]
            hexpart = ref["sha256"].split(":", 1)[1]
            blob = root / "artifacts" / "sha256" / hexpart[:2] / hexpart
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            (root / "artifacts" / "registry" / f"{slug_for_id(ref['artifact_id'])}.json").write_text(dumps_readable(ref))
    if with_views:
        (root / CFG_DIR / "trajectory.json").write_text('{"view_version": "trajectory-v1", "legacy": true}\n')
        (root / CFG_DIR / "memory_records.jsonl").write_text('{"record_id": "cfg-demo"}\n')
    if with_requests:
        req = root / "requests" / "request-legacy-1"
        (req / "events").mkdir(parents=True)
        (req / "request.json").write_text(
            dumps_readable({"ledger_version": 1, "request_id": "request-legacy-1", "state": "completed", "payload": {"kernel_id": KERNEL_ID, "config_ref": "cfg-demo"}})
        )
        (req / "events" / "000001-evt-legacy.json").write_text(dumps_readable({"event_id": "evt-legacy", "kind": "submitted", "at": LEGACY_TS}))
        (root / "requests" / "_keys").mkdir()
        (root / "requests" / "_keys" / "k.json").write_text(dumps_readable({"idempotency_key": "k", "request_id": "request-legacy-1"}))
    return relpaths


@pytest.fixture
def legacy_store(tmp_path: Path, legacy_bundle_dicts: list[dict], artifact_root: Path) -> Path:
    root = tmp_path / "legacy"
    build_legacy_store(root, legacy_bundle_dicts, artifact_root)
    return root


def _apply(source: Path, dest: Path, **kwargs: Any) -> StoreMigrationReport:
    return migrate_v02_store(source, dest, dry_run=False, **kwargs)


def _refused(legacy_store: Path, tmp_path: Path, mutate: Callable[[Path], None]) -> list[dict]:
    """Mutate the source, expect SOURCE_INTEGRITY, prove nothing was written; return the problems."""
    mutate(legacy_store)
    dest = tmp_path / "migrated"
    with pytest.raises(InvariantViolation) as info:
        migrate_v02_store(legacy_store, dest)
    assert info.value.code == "SOURCE_INTEGRITY" and info.value.exit_code == 2 and not dest.exists()
    assert info.value.details["count"] == len(info.value.details["problems"])
    return info.value.details["problems"]


# --------------------------------------------------------------------------------------
# upgrade half (pure function)
# --------------------------------------------------------------------------------------
def test_upgrade_is_deterministic_and_idempotent(legacy_bundle_dicts: list[dict]) -> None:
    once, report = upgrade_v02_records(legacy_bundle_dicts)
    again, _ = upgrade_v02_records(legacy_bundle_dicts)
    twice, second = upgrade_v02_records(once)
    assert once == again == twice and len(once) == FIXTURE_RECORD_COUNT == 19
    assert report.changed and not second.changed
    assert second.records_synthesized == [] and second.configs_linked == [] and second.records_rewritten == []


def test_upgrade_preserves_order_and_bundle_indices(legacy_bundle_dicts: list[dict]) -> None:
    upgraded, report = upgrade_v02_records(legacy_bundle_dicts)
    assert [d["record_id"] for d in upgraded[:18]] == [d["record_id"] for d in legacy_bundle_dicts]
    assert upgraded[1]["record_id"] == "cfg-demo" and upgraded[18]["record_id"] == PLACEHOLDER_ALGORITHM_ID
    assert report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID] and report.configs_linked == ["cfg-demo"]
    assert len(report.records_rewritten) == 18 and report.records_in == 18 and report.records_untouched == 0


def test_upgrade_never_invents_timestamps(legacy_bundle_dicts: list[dict]) -> None:
    upgraded, _ = upgrade_v02_records(legacy_bundle_dicts)
    assert {d["created_at"] for d in upgraded} <= {d["created_at"] for d in legacy_bundle_dicts}
    kernel = next(d for d in legacy_bundle_dicts if d["record_type"] == "kernel")
    assert upgraded[18]["created_at"] == kernel["created_at"] == LEGACY_TS


def test_upgrade_keeps_config_hash_and_places_algorithm_ref_after_kernel_id(legacy_bundle_dicts: list[dict]) -> None:
    upgraded, _ = upgrade_v02_records(legacy_bundle_dicts)
    old = record_dict(legacy_bundle_dicts, "cfg-demo")
    new = next(d for d in upgraded if d["record_id"] == "cfg-demo")
    assert new["payload"]["config_hash"] == old["payload"]["config_hash"] and new["record_id"] == "cfg-demo"
    assert new["schema_version"] == "0.3.0" and new["payload"]["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID
    keys = list(new["payload"])
    assert keys.index("algorithm_ref") == keys.index("kernel_id") + 1
    assert {k: v for k, v in new["payload"].items() if k != "algorithm_ref"} == old["payload"]


def test_upgrade_preserves_unknown_fields_and_never_mutates_input(legacy_bundle_dicts: list[dict]) -> None:
    dicts = clone(legacy_bundle_dicts)
    dicts[1]["unexpected_field"] = True
    dicts[1]["payload"]["extra"] = {"kept": 1}
    dicts.append(42)
    dicts.append({"record_type": "run", "record_id": "no-payload"})
    before = clone(dicts)
    upgraded, report = upgrade_v02_records(dicts)
    assert dicts == before
    cfg = next(d for d in upgraded if isinstance(d, dict) and d.get("record_id") == "cfg-demo")
    assert cfg["unexpected_field"] is True and cfg["payload"]["extra"] == {"kept": 1}
    assert 42 in upgraded and {"record_type": "run", "record_id": "no-payload"} in upgraded
    assert report.records_untouched == 2 and report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID]


def test_upgrade_kernel_less_configs_use_the_resolver_or_the_earliest_config_timestamp(legacy_bundle_dicts: list[dict]) -> None:
    cfg_a = record_dict(legacy_bundle_dicts, "cfg-demo")
    cfg_a["created_at"] = "2026-09-09T00:00:00Z"
    cfg_b = record_dict(legacy_bundle_dicts, "cfg-demo")
    cfg_b["record_id"], cfg_b["created_at"] = "cfg-demo-b", "2026-09-07T12:00:00Z"
    upgraded, report = upgrade_v02_records([cfg_a, cfg_b])
    assert report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID] and upgraded[2]["created_at"] == "2026-09-07T12:00:00Z"
    calls: list[tuple[str, str]] = []

    def resolver(kernel_id: str, algorithm_id: str) -> Record:
        calls.append((kernel_id, algorithm_id))
        return Record.from_dict(default_algorithm_record(kernel_id, LEGACY_TS))

    upgraded, report = upgrade_v02_records([cfg_a, cfg_b], algorithm_resolver=resolver)
    assert report.records_synthesized == [] and len(upgraded) == 2 and calls == [(KERNEL_ID, DEFAULT_ALGORITHM_ID)]
    assert all(d["payload"]["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID for d in upgraded)
    calls.clear()
    _, report = upgrade_v02_records(legacy_bundle_dicts, algorithm_resolver=resolver)
    assert calls == [] and report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID]  # the kernel is in the input


def test_upgrade_of_current_input_is_a_noop(bundle_dicts: list[dict]) -> None:
    upgraded, report = upgrade_v02_records(bundle_dicts)
    assert upgraded == bundle_dicts and not report.changed and report.records_synthesized == []


def test_placeholder_record_is_exact(legacy_bundle_dicts: list[dict]) -> None:
    upgraded, _ = upgrade_v02_records(legacy_bundle_dicts)
    assert upgraded[18] == {
        "schema_version": "0.3.0",
        "record_type": "algorithm",
        "record_id": "algorithm-demo_vector_add-unspecified",
        "created_at": LEGACY_TS,
        "payload": {
            "kernel_id": KERNEL_ID,
            "algorithm_id": "unspecified",
            "display_name": "unspecified",
            "method_summary": "unspecified (imported from v0.2)",
            "summary_author": "program",
            "tags": ["imported-v02"],
        },
    }
    assert LEGACY_METHOD_SUMMARY == "unspecified (imported from v0.2)" and LEGACY_SUMMARY_AUTHOR == "program" and list(LEGACY_TAGS) == ["imported-v02"]
    assert default_algorithm_record_id(KERNEL_ID) == PLACEHOLDER_ALGORITHM_ID
    for d in upgraded:
        Record.from_dict(clone(d))  # every upgraded dict validates against the 0.3.0 contract


def test_upgrade_bundle_bumps_version_and_rejects_non_bundles(legacy_bundle_dicts: list[dict]) -> None:
    bundle, report = upgrade_bundle_v02({"bundle_version": "0.2.0", "is_fixture": True, "description": "x", "records": legacy_bundle_dicts})
    assert bundle["bundle_version"] == "0.3.0" and len(bundle["records"]) == 19 and bundle["description"] == "x"
    assert report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID]
    with pytest.raises(InputError) as info:
        upgrade_bundle_v02({"records": "nope"})
    assert info.value.code == "INVALID_BUNDLE"


# --------------------------------------------------------------------------------------
# store half: reading a layout-1 store
# --------------------------------------------------------------------------------------
def test_legacy_relpaths_follow_the_v02_layout(legacy_store: Path, legacy_bundle_dicts: list[dict]) -> None:
    by_id = {d["record_id"]: d for d in legacy_bundle_dicts}
    assert legacy_relpath(by_id["kernel-demo"], by_id) == "kernels/demo_vector_add/kernel.json"
    assert legacy_relpath(by_id["cfg-demo"], by_id) == f"{CFG_DIR}/config.json"
    assert legacy_relpath(by_id["run-demo-a"], by_id) == f"{CFG_DIR}/attempt/pr-demo-101/commits/commit-demo-a/runs/run-demo-a/run.json"
    assert legacy_relpath(by_id["run-demo-baseline"], by_id) == f"{CFG_DIR}/baselines/baseline-demo/runs/run-demo-baseline/run.json"
    assert legacy_relpath(by_id["snapshot-demo-102"], by_id) == f"{CFG_DIR}/attempt/pr-demo-102/snapshots/snapshot-demo-102.json"
    assert legacy_relpath(by_id["annotation-demo-grouped"], by_id) == f"{CFG_DIR}/annotations/annotation-demo-grouped.json"
    assert (legacy_store / CFG_DIR / "trajectory.json").is_file() and (legacy_store / "journal" / "records.jsonl").is_file()


def test_read_legacy_store_is_read_only(legacy_store: Path, legacy_bundle_dicts: list[dict]) -> None:
    before = tree_digest(legacy_store)
    snapshot = read_legacy_store(legacy_store)
    assert tree_digest(legacy_store) == before and not (legacy_store / ".runtime" / "lock").exists()
    assert isinstance(snapshot, LegacyStoreSnapshot) and snapshot.manifest["layout_version"] == 1
    assert len(snapshot.records) == 18 and len(snapshot.journal) == 18 and snapshot.integrity_problems == [] and snapshot.pending_state == []
    assert [d["record_id"] for d in snapshot.records] == [rid for rid, _ in sorted(snapshot.relpaths.items(), key=lambda kv: Path(kv[1]))]
    assert {d["record_id"] for d in snapshot.records} == {d["record_id"] for d in legacy_bundle_dicts}
    assert all(d["schema_version"] == "0.2.0" for d in snapshot.records)
    assert len(snapshot.artifact_blobs) == 5 and len(snapshot.artifact_descriptors) == 7 and len(snapshot.request_files) == 3
    assert snapshot.embedded_record_requests == [] and snapshot.notes == []
    assert not any(rel.endswith(("trajectory.json", "memory_records.jsonl")) for rel in snapshot.relpaths.values())


def test_not_a_store_and_not_a_legacy_store(tmp_path: Path, store: MemoryStore) -> None:
    with pytest.raises(InputError) as info:
        read_legacy_store(tmp_path / "nothing")
    assert info.value.code == "NOT_A_STORE"
    with pytest.raises(InputError) as info:
        migrate_v02_store(store.root, tmp_path / "dest")  # a layout-2 store is not a migration source
    assert info.value.code == "NOT_A_LEGACY_STORE" and info.value.details["layout_version"] == 2
    assert not (tmp_path / "dest").exists()


def test_source_store_is_refused_by_the_current_reader(legacy_store: Path) -> None:
    before = tree_digest(legacy_store)
    with pytest.raises(InputError) as info:
        MemoryStore.open(legacy_store)
    assert info.value.code == "UNSUPPORTED_STORE" and info.value.exit_code == 2 and "migrate-v02" in info.value.message
    assert tree_digest(legacy_store) == before and not (legacy_store / ".runtime" / "lock").exists()


# --------------------------------------------------------------------------------------
# store half: dry run and apply
# --------------------------------------------------------------------------------------
def test_dry_run_writes_nothing(legacy_store: Path, tmp_path: Path) -> None:
    before = tree_digest(legacy_store)
    dest = tmp_path / "migrated"
    report = migrate_v02_store(legacy_store, dest)  # dry_run is the default
    assert report.dry_run is True and not dest.exists() and tree_digest(legacy_store) == before
    assert report.records_read == 18 and report.records_rewritten == 18 and report.records_linked == 1
    assert report.records_synthesized == [PLACEHOLDER_ALGORITHM_ID]
    assert report.artifacts_found == 5 and report.descriptors_found == 7 and report.requests_found == 3
    assert report.records_published == 0 and report.artifacts_copied == 0 and report.descriptors_copied == 0 and report.requests_copied == 0
    assert report.destination_integrity_ok is None and report.provenance == "preserved" and report.forced is False
    assert report.kernel_views_published == [] and any("dry run" in n for n in report.notes)
    data = report.to_dict()
    assert set(data) == REPORT_KEYS and data["dry_run"] is True and data["records_synthesized"] == [PLACEHOLDER_ALGORITHM_ID]


def test_apply_migrates_into_a_new_root(legacy_store: Path, tmp_path: Path, legacy_bundle_dicts: list[dict]) -> None:
    before = tree_digest(legacy_store)
    dest = tmp_path / "migrated"
    report = _apply(legacy_store, dest)
    assert tree_digest(legacy_store) == before  # the source is never written
    assert report.dry_run is False and report.records_published == 19 and report.records_idempotent == 0
    assert report.artifacts_copied == 5 and report.descriptors_copied == 7 and report.requests_copied == 3
    assert report.destination_integrity_ok is True and report.integrity_problems == [] and report.forced is False
    assert set(report.to_dict()) == REPORT_KEYS
    store = MemoryStore.open(dest)
    assert len(store.records()) == FIXTURE_RECORD_COUNT
    placeholder = store.get(PLACEHOLDER_ALGORITHM_ID)
    assert placeholder is not None and placeholder.payload.method_summary == LEGACY_METHOD_SUMMARY and placeholder.created_at == LEGACY_TS
    cfg = store.get("cfg-demo")
    assert cfg.payload.algorithm_ref == PLACEHOLDER_ALGORITHM_ID and cfg.schema_version == "0.3.0"
    assert store.record_path("cfg-demo").relative_to(store.root) == Path("kernels/demo_vector_add/unspecified/cfg-demo/config.json")
    assert store.record_path(PLACEHOLDER_ALGORITHM_ID).relative_to(store.root) == Path("kernels/demo_vector_add/unspecified/algorithm.json")
    assert not (dest / CFG_DIR).exists()  # old layout and old views are not copied
    for d in legacy_bundle_dicts:
        if d["record_type"] == "run":
            assert store.get(d["record_id"]).payload.provenance == d["payload"]["provenance"] == "fixture"  # preserved as stored
            for ref in d["payload"]["artifacts"]:
                assert store.has_artifact(ref["sha256"]) and store.get_artifact_ref(ref["artifact_id"]) is not None
    assert len(store.artifact_registry()) == 7
    for rel in ("requests/request-legacy-1/request.json", "requests/request-legacy-1/events/000001-evt-legacy.json", "requests/_keys/k.json"):
        assert (dest / rel).read_bytes() == (legacy_store / rel).read_bytes()
    manifest = load_json_file(dest / "manifest.json")
    assert manifest["layout_version"] == 2
    migrated_from = manifest["migrated_from"]
    assert migrated_from["root"] == str(legacy_store.resolve()) and migrated_from["store_id"] == LEGACY_STORE_ID
    assert migrated_from["layout_version"] == 1 and migrated_from["store_version"] == "0.2.0" and migrated_from["journal_entries"] == 18
    assert migrated_from["forced"] is False and migrated_from["provenance"] == "preserved" and migrated_from["integrity_problems"] == 0
    assert store.integrity_scan().ok
    assert deep_validate(store).ok


def test_apply_publishes_kernel_trajectory_when_the_service_exists(legacy_store: Path, tmp_path: Path) -> None:
    dest = tmp_path / "migrated"
    report = _apply(legacy_store, dest)
    try:
        from kernel_memory.services.trajectory import publish_kernel_trajectory  # noqa: F401
    except ImportError:
        assert report.kernel_views_published == []
        assert any("publish_kernel_trajectory" in note for note in report.notes)
    else:
        assert report.kernel_views_published == [KERNEL_ID], report.notes
        assert (dest / "kernels/demo_vector_add/trajectory/trajectory.json").is_file()
        assert MemoryStore.open(dest).integrity_scan().ok  # views are never records


def test_reapply_into_another_root_is_deterministic(legacy_store: Path, tmp_path: Path) -> None:
    _apply(legacy_store, tmp_path / "a")
    _apply(legacy_store, tmp_path / "b")

    def journal(root: Path) -> list[tuple[str, str, str]]:
        lines = (root / "journal" / "records.jsonl").read_text().splitlines()
        return sorted((e["record_id"], e["relpath"], e["digest"]) for e in map(json.loads, lines))

    assert journal(tmp_path / "a") == journal(tmp_path / "b") and len(journal(tmp_path / "a")) == 19
    a, b = MemoryStore.open(tmp_path / "a"), MemoryStore.open(tmp_path / "b")
    assert [r.canonical_digest() for r in a.records()] == [r.canonical_digest() for r in b.records()]


# --------------------------------------------------------------------------------------
# store half: refusals happen before any write
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("pending", [".runtime/pending/txn-1.json", ".runtime/jobs/job-1.json"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_pending_runtime_state_refused(legacy_store: Path, tmp_path: Path, pending: str, dry_run: bool) -> None:
    path = legacy_store / pending
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    before = tree_digest(legacy_store)
    dest = tmp_path / "migrated"
    with pytest.raises(InvariantViolation) as info:
        migrate_v02_store(legacy_store, dest, dry_run=dry_run)
    assert info.value.code == "MIGRATE_PENDING_STATE" and info.value.exit_code == 2
    assert info.value.details["pending"] == [pending.rsplit("/", 1)[0]]
    assert not dest.exists() and tree_digest(legacy_store) == before


def test_tampered_record_is_source_integrity_unless_forced(legacy_store: Path, tmp_path: Path) -> None:
    path = legacy_store / CFG_DIR / "annotations" / "annotation-demo-grouped.json"
    data = json.loads(path.read_text())
    data["payload"]["text"] = "edited after publication"
    path.write_text(json.dumps(data, indent=2) + "\n")
    before = tree_digest(legacy_store)
    dest = tmp_path / "migrated"
    with pytest.raises(InvariantViolation) as info:
        _apply(legacy_store, dest)
    assert info.value.code == "SOURCE_INTEGRITY" and info.value.exit_code == 2
    problems = info.value.details["problems"]
    assert len(problems) == 1 and problems[0]["problem"] == "modified" and problems[0]["record_id"] == "annotation-demo-grouped"
    assert problems[0]["journal_digest"] != problems[0]["file_digest"]
    assert not dest.exists() and tree_digest(legacy_store) == before
    forced = migrate_v02_store(legacy_store, dest, force=True)  # still a dry run
    assert forced.forced is True and forced.integrity_problems == problems and not dest.exists()
    applied = _apply(legacy_store, dest, force=True)
    assert applied.forced is True and applied.records_published == 19 and applied.integrity_problems == problems
    manifest = load_json_file(dest / "manifest.json")
    assert manifest["migrated_from"]["forced"] is True and manifest["migrated_from"]["integrity_problems"] == 1
    assert MemoryStore.open(dest).get("annotation-demo-grouped").payload.text == "edited after publication"  # migrated as found, recorded as forced
    assert tree_digest(legacy_store) == before


def test_unjournaled_stray_record(legacy_store: Path, tmp_path: Path) -> None:
    src = legacy_store / CFG_DIR / "annotations" / "annotation-demo-grouped.json"
    stray = json.loads(src.read_text())
    stray["record_id"] = "annotation-demo-stray"
    problems = _refused(legacy_store, tmp_path, lambda root: (src.parent / "annotation-demo-stray.json").write_text(json.dumps(stray, indent=2)))
    assert problems == [{"record_id": "annotation-demo-stray", "path": f"{CFG_DIR}/annotations/annotation-demo-stray.json", "problem": "unjournaled"}]


def test_missing_record_file(legacy_store: Path, tmp_path: Path) -> None:
    rel = f"{CFG_DIR}/decisions/decision-demo-blocked.json"
    problems = _refused(legacy_store, tmp_path, lambda root: (root / rel).unlink())
    assert problems == [{"record_id": "decision-demo-blocked", "problem": "missing", "expected_path": rel}]


def test_corrupt_record_file(legacy_store: Path, tmp_path: Path) -> None:
    rel = f"{CFG_DIR}/relations/relation-demo-origin.json"
    problems = _refused(legacy_store, tmp_path, lambda root: (root / rel).write_text("{ not json"))
    assert sorted(p["problem"] for p in problems) == ["corrupt", "missing"]  # unreadable file, and its journal entry has no valid file
    assert next(p for p in problems if p["problem"] == "corrupt")["path"] == rel


def test_record_violating_the_legacy_contract_is_corrupt(legacy_store: Path, tmp_path: Path) -> None:
    rel = f"{CFG_DIR}/config.json"

    def mutate(root: Path) -> None:
        data = json.loads((root / rel).read_text())
        data["payload"]["algorithm_ref"] = "x"  # not a 0.2.0 field
        (root / rel).write_text(json.dumps(data, indent=2))

    problems = _refused(legacy_store, tmp_path, mutate)
    assert any(p["problem"] == "corrupt" and p["path"] == rel and "algorithm_ref" in p["error"] for p in problems)


def test_moved_record_file_is_relpath_drift(legacy_store: Path, tmp_path: Path) -> None:
    rel = f"{CFG_DIR}/relations/relation-demo-origin.json"
    moved = f"{CFG_DIR}/relations/moved.json"
    problems = _refused(legacy_store, tmp_path, lambda root: (root / rel).rename(root / moved))
    assert problems == [{"record_id": "relation-demo-origin", "path": moved, "problem": "modified", "reason": "moved", "journal_relpath": rel}]


def test_corrupt_journal_line_and_duplicate_record(legacy_store: Path, tmp_path: Path) -> None:
    def mutate(root: Path) -> None:
        journal = root / "journal" / "records.jsonl"
        journal.write_bytes(journal.read_bytes() + b"{ not json\n")
        src = root / CFG_DIR / "annotations" / "annotation-demo-grouped.json"
        (src.parent / "twin.json").write_bytes(src.read_bytes())

    problems = _refused(legacy_store, tmp_path, mutate)
    assert sorted(p["problem"] for p in problems) == ["corrupt_journal_line", "duplicate"]
    assert next(p for p in problems if p["problem"] == "corrupt_journal_line")["line_no"] == 19
    assert next(p for p in problems if p["problem"] == "duplicate")["record_id"] == "annotation-demo-grouped"


def test_corrupt_blob_unaddressable_blob_and_unreadable_descriptor(legacy_store: Path, tmp_path: Path, legacy_bundle_dicts: list[dict]) -> None:
    samples_hex = record_dict(legacy_bundle_dicts, "run-demo-a")["payload"]["artifacts"][0]["sha256"].split(":", 1)[1]

    def mutate(root: Path) -> None:
        blob = root / "artifacts" / "sha256" / samples_hex[:2] / samples_hex
        blob.write_bytes(blob.read_bytes() + b" ")
        (root / "artifacts" / "registry" / "run-demo-a-samples.json").write_text("{ nope")
        (root / "artifacts" / "sha256" / "zz").mkdir()
        (root / "artifacts" / "sha256" / "zz" / "not-a-digest").write_bytes(b"x")

    problems = _refused(legacy_store, tmp_path, mutate)
    assert sorted(p["problem"] for p in problems) == ["artifact_corrupt", "artifact_unaddressable", "corrupt_descriptor"]
    assert next(p for p in problems if p["problem"] == "artifact_corrupt")["declared"] == f"sha256:{samples_hex}"


@pytest.mark.parametrize("dry_run", [True, False])
def test_embedded_record_in_request_refused_even_with_force(legacy_store: Path, tmp_path: Path, legacy_bundle_dicts: list[dict], dry_run: bool) -> None:
    rel = "requests/request-legacy-1/events/000002-embedded.json"
    (legacy_store / rel).write_text(json.dumps(record_dict(legacy_bundle_dicts, "cfg-demo")))
    before = tree_digest(legacy_store)
    assert read_legacy_store(legacy_store).embedded_record_requests == [rel]
    dest = tmp_path / "migrated"
    with pytest.raises(InvariantViolation) as info:
        migrate_v02_store(legacy_store, dest, dry_run=dry_run, force=True)
    assert info.value.code == "EMBEDDED_RECORD_IN_REQUEST" and info.value.details["paths"] == [rel]
    assert not dest.exists() and tree_digest(legacy_store) == before


@pytest.mark.parametrize("change, code", [({"media_type": "text/plain"}, "ARTIFACT_DESCRIPTOR_CONFLICT"), ({"retention": None}, "ARTIFACT_DESCRIPTOR_INVALID")])
def test_bad_registry_descriptor_refused_before_any_write(legacy_store: Path, tmp_path: Path, change: dict, code: str) -> None:
    path = legacy_store / "artifacts" / "registry" / "run-demo-a-samples.json"
    data = json.loads(path.read_text())
    for key, value in change.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    path.write_text(json.dumps(data, indent=2))
    dest = tmp_path / "migrated"
    with pytest.raises(InvariantViolation) as info:
        _apply(legacy_store, dest)
    assert info.value.code == code and info.value.details["artifact_id"] == "run-demo-a-samples" and not dest.exists()


def test_destination_checks(legacy_store: Path, tmp_path: Path) -> None:
    before = tree_digest(legacy_store)
    for dest in (legacy_store, legacy_store / "inside", tmp_path):  # the source, inside it, containing it
        with pytest.raises(InputError) as info:
            _apply(legacy_store, dest)
        assert info.value.code == "INVALID_DESTINATION" and info.value.exit_code == 2, dest
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file.txt").write_text("x")
    with pytest.raises(InputError) as info:
        _apply(legacy_store, occupied)
    assert info.value.code == "ROOT_NOT_EMPTY" and info.value.details["entries"] == ["file.txt"]
    assert (occupied / "file.txt").read_text() == "x" and tree_digest(legacy_store) == before
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _apply(legacy_store, empty).records_published == 19  # an empty directory is a valid destination
