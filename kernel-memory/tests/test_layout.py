"""Physical store layout (specification section 4; ADR-0003).

Slugged directories derived from record ids, identities inside the files, views and
dotfiles excluded from the authoritative record set, and case-collision-safe slugs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path, PurePosixPath

import pytest

from conftest import import_demo_bundle, record_dict
from kernel_memory.domain.errors import InputError, MissingReferenceError
from kernel_memory.domain.ids import slug_for_id
from kernel_memory.domain.models import Record
from kernel_memory.storage import MemoryStore, layout

CFG = "kernels/demo_vector_add/configs/cfg-demo"
PR101 = f"{CFG}/attempt/pr-demo-101"
PR102 = f"{CFG}/attempt/pr-demo-102"

EXPECTED_RECORD_FILES: dict[str, str] = {
    "kernel-demo": "kernels/demo_vector_add/kernel.json",
    "cfg-demo": f"{CFG}/config.json",
    "pr-demo-101": f"{PR101}/pr.json",
    "snapshot-demo-101": f"{PR101}/snapshots/snapshot-demo-101.json",
    "commit-demo-a": f"{PR101}/commits/commit-demo-a/commit.json",
    "commit-demo-b": f"{PR101}/commits/commit-demo-b/commit.json",
    "run-demo-a": f"{PR101}/commits/commit-demo-a/runs/run-demo-a/run.json",
    "pr-demo-102": f"{PR102}/pr.json",
    "snapshot-demo-102": f"{PR102}/snapshots/snapshot-demo-102.json",
    "commit-demo-c": f"{PR102}/commits/commit-demo-c/commit.json",
    "commit-demo-a-in-102": f"{PR102}/commits/commit-demo-a-in-102/commit.json",
    "run-demo-c": f"{PR102}/commits/commit-demo-c/runs/run-demo-c/run.json",
    "run-demo-c-failure": f"{PR102}/commits/commit-demo-c/runs/run-demo-c-failure/run.json",
    "baseline-demo": f"{CFG}/baselines/baseline-demo/baseline.json",
    "run-demo-baseline": f"{CFG}/baselines/baseline-demo/runs/run-demo-baseline/run.json",
    "relation-demo-origin": f"{CFG}/relations/relation-demo-origin.json",
    "decision-demo-blocked": f"{CFG}/decisions/decision-demo-blocked.json",
    "annotation-demo-grouped": f"{CFG}/annotations/annotation-demo-grouped.json",
}


def shuffled(records: list[Record], seed: int = 20260924) -> list[Record]:
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


@pytest.fixture
def shuffled_store(tmp_path: Path, bundle_records: list[Record], artifact_root: Path) -> MemoryStore:
    store = MemoryStore.init(tmp_path / "memory")
    import_demo_bundle(store, shuffled(bundle_records), artifact_root)
    return store


# ----------------------------------------------------------------------------- record files
def test_expected_files_cover_all_fixture_records(bundle_records: list[Record]) -> None:
    assert len(bundle_records) == 18
    assert set(EXPECTED_RECORD_FILES) == {r.record_id for r in bundle_records}


@pytest.mark.parametrize("record_id,relpath", sorted(EXPECTED_RECORD_FILES.items()))
def test_shuffled_import_produces_exact_layout(shuffled_store: MemoryStore, record_id: str, relpath: str) -> None:
    path = shuffled_store.root / relpath
    assert path.is_file(), relpath
    # Identity lives inside the file, not in the path.
    data = json.loads(path.read_text("utf-8"))
    assert data["record_id"] == record_id
    assert shuffled_store.record_path(record_id) == path


def test_no_unexpected_record_files(shuffled_store: MemoryStore) -> None:
    found = sorted(
        str(PurePosixPath(p.relative_to(shuffled_store.root).as_posix()))
        for p in (shuffled_store.root / "kernels").rglob("*")
        if p.is_file()
    )
    assert found == sorted(EXPECTED_RECORD_FILES.values())


def test_record_relpath_matches_store_paths(shuffled_store: MemoryStore) -> None:
    for record in shuffled_store.records():
        rel = layout.record_relpath(record, shuffled_store.get, shuffled_store.kernel_by_kernel_id)
        assert str(rel) == EXPECTED_RECORD_FILES[record.record_id]
        assert layout.is_record_file(rel)


def test_config_dir_and_kernel_dir(shuffled_store: MemoryStore) -> None:
    assert shuffled_store.config_dir("cfg-demo") == shuffled_store.root / CFG
    assert str(layout.kernel_dir("demo_vector_add")) == "kernels/demo_vector_add"
    with pytest.raises(MissingReferenceError):
        shuffled_store.config_dir("commit-demo-a")  # not a config


# ----------------------------------------------------------------------------- artifact registry
def test_artifact_registry_file_per_artifact_id(shuffled_store: MemoryStore, bundle_records: list[Record]) -> None:
    artifact_ids = {a.artifact_id for r in bundle_records if r.record_type == "run" for a in r.payload.artifacts}
    assert len(artifact_ids) == 7
    for artifact_id in artifact_ids:
        rel = layout.artifact_registry_relpath(artifact_id)
        assert str(rel) == f"artifacts/registry/{slug_for_id(artifact_id)}.json"
        path = shuffled_store.root / rel
        assert path.is_file(), artifact_id
        assert json.loads(path.read_text("utf-8"))["artifact_id"] == artifact_id
    registry_files = sorted(p.name for p in (shuffled_store.root / "artifacts" / "registry").glob("*.json"))
    assert registry_files == sorted(f"{slug_for_id(a)}.json" for a in artifact_ids)
    assert {ref.artifact_id for ref in shuffled_store.artifact_registry()} == artifact_ids


def test_artifact_blob_path_is_content_addressed(shuffled_store: MemoryStore) -> None:
    ref = shuffled_store.get_artifact_ref("run-demo-a-samples")
    assert ref is not None
    digest = ref.sha256.split(":", 1)[1]
    rel = layout.artifact_relpath(ref.sha256)
    assert str(rel) == f"artifacts/sha256/{digest[:2]}/{digest}"
    assert (shuffled_store.root / rel).is_file()
    assert shuffled_store.artifact_path(ref.sha256) == shuffled_store.root / rel


def test_request_dir_uses_slug() -> None:
    assert str(layout.request_dir("request-abc")) == "requests/request-abc"
    assert str(layout.request_dir("Request-ABC")) == f"requests/{slug_for_id('Request-ABC')}"


# ----------------------------------------------------------------------------- is_record_file
@pytest.mark.parametrize(
    "relpath",
    [
        "kernels/demo_vector_add/kernel.json",
        f"{CFG}/config.json",
        f"{PR101}/commits/commit-demo-a/runs/run-demo-a/run.json",
        f"{CFG}/annotations/annotation-x.json",
    ],
)
def test_is_record_file_accepts_authoritative_records(relpath: str) -> None:
    assert layout.is_record_file(PurePosixPath(relpath)) is True


@pytest.mark.parametrize(
    "relpath",
    [
        f"{CFG}/trajectory.json",
        f"{CFG}/memory_records.jsonl",
        f"{CFG}/context.json",
        f"{CFG}/.config.json.tmp-123-abcdef01",
        f"{CFG}/.hidden.json",
        f"{CFG}/notes.txt",
        "artifacts/registry/run-demo-a-samples.json",
        "requests/request-x/request.json",
        "manifest.json",
        "journal/records.jsonl",
        ".runtime/pending/txn-x.json",
    ],
)
def test_is_record_file_excludes_views_dotfiles_and_non_kernel_paths(relpath: str) -> None:
    assert layout.is_record_file(PurePosixPath(relpath)) is False


def test_view_file_names_are_the_three_generated_views() -> None:
    assert layout.VIEW_FILE_NAMES == {"trajectory.json", "memory_records.jsonl", "context.json"}


# ----------------------------------------------------------------------------- slug safety
def test_slug_for_id_lowercase_safe_ids_are_verbatim() -> None:
    for value in ("commit-demo-a", "cfg-demo", "demo_vector_add", "run.1_x-y"):
        assert slug_for_id(value) == value


def test_slug_for_id_case_variants_never_collide() -> None:
    lower, upper, mixed = slug_for_id("run-a"), slug_for_id("RUN-A"), slug_for_id("Run-A")
    assert len({lower, upper, mixed}) == 3
    # Every slug is lower-case so a case-insensitive filesystem cannot merge two of them.
    for s in (lower, upper, mixed):
        assert s == s.lower()
        assert ".." not in s and "/" not in s
    assert slug_for_id("Run-A") == mixed  # deterministic


def test_slug_for_id_punctuation_variants_never_collide() -> None:
    assert slug_for_id("a:b") != slug_for_id("a_b") != slug_for_id("a-b")
    assert slug_for_id("a:b") != slug_for_id("a-b")


def test_slug_for_id_rejects_empty() -> None:
    with pytest.raises(InputError) as info:
        slug_for_id("")
    assert info.value.code == "INVALID_ID"
    assert info.value.exit_code == 2


def test_case_variant_ids_get_distinct_files_in_store(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    base = record_dict(bundle_dicts, "annotation-demo-grouped")
    records = []
    for rid in ("annotation-case", "annotation-Case", "annotation-CASE"):
        data = json.loads(json.dumps(base))
        data["record_id"] = rid
        data["payload"]["text"] = f"variant {rid}"
        records.append(Record.from_dict(data))
    outcome = demo_store.publish_bundle(records)
    assert sorted(outcome.published) == sorted(r.record_id for r in records)
    paths = {rid: demo_store.record_path(rid) for rid in ("annotation-case", "annotation-Case", "annotation-CASE")}
    assert len({str(p).lower() for p in paths.values()}) == 3
    assert paths["annotation-case"].name == "annotation-case.json"
    for rid, path in paths.items():
        assert path.is_file()
        assert json.loads(path.read_text("utf-8"))["record_id"] == rid
        assert demo_store.get(rid).payload.text == f"variant {rid}"
    assert demo_store.integrity_scan().ok


# ----------------------------------------------------------------------------- negative resolution
def test_record_relpath_rejects_run_whose_subject_is_a_config(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-bad-subject"
    data["payload"]["subject_ref"] = "cfg-demo"
    record = Record.from_dict(data)
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(record, demo_store.get, demo_store.kernel_by_kernel_id)
    assert info.value.exit_code == 3


def test_record_relpath_rejects_missing_parent(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "commit-demo-a")
    data["record_id"] = "commit-orphan"
    data["payload"]["pr_ref"] = "pr-does-not-exist"
    record = Record.from_dict(data)
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(record, demo_store.get, demo_store.kernel_by_kernel_id)
    assert info.value.details["record_id"] == "pr-does-not-exist"


def test_config_relpath_rejects_unknown_kernel(store: MemoryStore, bundle_dicts: list[dict]) -> None:
    record = Record.from_dict(record_dict(bundle_dicts, "cfg-demo"))
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(record, store.get, store.kernel_by_kernel_id)
    assert info.value.details["kernel_id"] == "demo_vector_add"


def test_subject_dir_for_rejects_non_subject(demo_store: MemoryStore) -> None:
    config = demo_store.require("cfg-demo")
    with pytest.raises(MissingReferenceError):
        layout.subject_dir_for(config, demo_store.get, demo_store.kernel_by_kernel_id)


def test_kernel_annotation_lives_under_kernel_dir(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["record_id"] = "annotation-kernel-note"
    data["payload"]["target_ref"] = "kernel-demo"
    data["payload"]["evidence_refs"] = []
    record = Record.from_dict(data)
    rel = layout.record_relpath(record, demo_store.get, demo_store.kernel_by_kernel_id)
    assert str(rel) == "kernels/demo_vector_add/annotations/annotation-kernel-note.json"
    demo_store.publish(record)
    assert demo_store.record_path("annotation-kernel-note") == demo_store.root / rel
    with pytest.raises(MissingReferenceError):
        layout.config_dir_for(record, demo_store.get, demo_store.kernel_by_kernel_id)
