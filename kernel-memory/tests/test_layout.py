"""Physical store layout (specification section 4; ADR-0003; ADR-0004 layout_version 2).

Slugged directories derived from record ids, identities inside the files, the algorithm level
between kernel and shape (config), the kernel-wide ``trajectory/`` view tree excluded from the
authoritative record set by directory (never by file name), reserved sibling names, dotfiles
excluded, and case-collision-safe slugs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path, PurePosixPath

import pytest

from conftest import FIXTURE_RECORD_COUNT, PLACEHOLDER_ALGORITHM_ID, import_demo_bundle, record_dict
from kernel_memory.domain.errors import InputError, MissingReferenceError
from kernel_memory.domain.ids import slug_for_id
from kernel_memory.domain.models import Record
from kernel_memory.storage import MemoryStore, layout

ALG = "kernels/demo_vector_add/unspecified"
CFG = f"{ALG}/cfg-demo"
VIEWS = "kernels/demo_vector_add/trajectory"
PR101 = f"{CFG}/attempt/pr-demo-101"
PR102 = f"{CFG}/attempt/pr-demo-102"

EXPECTED_RECORD_FILES: dict[str, str] = {
    "kernel-demo": "kernels/demo_vector_add/kernel.json",
    PLACEHOLDER_ALGORITHM_ID: f"{ALG}/algorithm.json",
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


def kernel_files(store: MemoryStore) -> list[str]:
    return sorted(
        str(PurePosixPath(p.relative_to(store.root).as_posix()))
        for p in (store.root / "kernels").rglob("*")
        if p.is_file()
    )


@pytest.fixture
def shuffled_store(tmp_path: Path, bundle_records: list[Record], artifact_root: Path) -> MemoryStore:
    store = MemoryStore.init(tmp_path / "memory")
    import_demo_bundle(store, shuffled(bundle_records), artifact_root)
    return store


# ----------------------------------------------------------------------------- record files
def test_expected_files_cover_all_fixture_records(bundle_records: list[Record]) -> None:
    assert len(bundle_records) == FIXTURE_RECORD_COUNT
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
    assert kernel_files(shuffled_store) == sorted(EXPECTED_RECORD_FILES.values())


def test_record_relpath_matches_store_paths(shuffled_store: MemoryStore) -> None:
    for record in shuffled_store.records():
        rel = layout.record_relpath(record, shuffled_store.get, shuffled_store.kernel_by_kernel_id)
        assert str(rel) == EXPECTED_RECORD_FILES[record.record_id]
        assert layout.is_record_file(rel)


def test_config_dir_and_kernel_dir(shuffled_store: MemoryStore) -> None:
    assert shuffled_store.config_dir("cfg-demo") == shuffled_store.root / CFG
    assert str(layout.kernel_dir("demo_vector_add")) == "kernels/demo_vector_add"
    assert str(layout.algorithm_dir("demo_vector_add", "unspecified")) == ALG
    with pytest.raises(MissingReferenceError):
        shuffled_store.config_dir("commit-demo-a")  # not a config
    with pytest.raises(MissingReferenceError):
        shuffled_store.config_dir(PLACEHOLDER_ALGORITHM_ID)  # an algorithm is not a shape either


def test_algorithm_dir_slugs_and_view_dirs() -> None:
    assert layout.algorithm_slug("unspecified") == "unspecified"
    assert layout.shape_slug("cfg-demo") == "cfg-demo"
    assert str(layout.algorithm_dir("demo_vector_add", "numpy-add")) == "kernels/demo_vector_add/numpy-add"
    assert str(layout.kernel_view_dir("demo_vector_add")) == VIEWS
    assert str(layout.shape_view_dir("demo_vector_add", "cfg-demo")) == f"{VIEWS}/shapes/cfg-demo"
    assert str(layout.shape_view_dir("demo_vector_add", "Cfg-X")) == f"{VIEWS}/shapes/{slug_for_id('Cfg-X')}"
    # The view tree is a sibling of the algorithm directories, never inside one.
    assert not str(layout.shape_view_dir("demo_vector_add", "cfg-demo")).startswith(ALG)


# ----------------------------------------------------------------------------- reserved slugs
@pytest.mark.parametrize("algorithm_id", sorted(layout.RESERVED_ALGORITHM_SLUGS))
def test_reserved_algorithm_slug_rejected(algorithm_id: str) -> None:
    assert layout.RESERVED_ALGORITHM_SLUGS == {"trajectory", "annotations", "kernel.json"}
    for call in (lambda: layout.algorithm_slug(algorithm_id), lambda: layout.algorithm_dir("demo_vector_add", algorithm_id)):
        with pytest.raises(InputError) as info:
            call()
        assert info.value.code == "RESERVED_SLUG"
        assert info.value.exit_code == 2
        assert info.value.details["algorithm_id"] == algorithm_id
        assert info.value.details["reserved"] == sorted(layout.RESERVED_ALGORITHM_SLUGS)
    # Only the exact slug is reserved: a case variant gets a distinct, non-reserved slug.
    variant = algorithm_id.capitalize()
    assert layout.algorithm_slug(variant) == slug_for_id(variant)
    assert layout.algorithm_slug(variant) not in layout.RESERVED_ALGORITHM_SLUGS


@pytest.mark.parametrize("record_id", sorted(layout.RESERVED_SHAPE_SLUGS))
def test_reserved_shape_slug_rejected(record_id: str) -> None:
    assert layout.RESERVED_SHAPE_SLUGS == {"annotations", "algorithm.json"}
    with pytest.raises(InputError) as info:
        layout.shape_slug(record_id)
    assert info.value.code == "RESERVED_SLUG"
    assert info.value.exit_code == 2
    assert info.value.details["record_id"] == record_id
    assert info.value.details["reserved"] == sorted(layout.RESERVED_SHAPE_SLUGS)


def test_reserved_slugs_rejected_at_publish(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    files_before = kernel_files(demo_store)
    entries_before = demo_store.index_entries()
    algorithm = record_dict(bundle_dicts, PLACEHOLDER_ALGORITHM_ID)
    algorithm["record_id"] = "algorithm-demo_vector_add-trajectory"
    algorithm["payload"]["algorithm_id"] = "trajectory"
    with pytest.raises(InputError) as info:
        demo_store.publish(Record.from_dict(algorithm))
    assert info.value.code == "RESERVED_SLUG"
    assert info.value.exit_code == 2
    config = record_dict(bundle_dicts, "cfg-demo")
    config["record_id"] = "annotations"
    with pytest.raises(InputError) as info2:
        demo_store.publish(Record.from_dict(config))
    assert info2.value.code == "RESERVED_SLUG"
    # Nothing was written and the index is unchanged.
    assert demo_store.get("algorithm-demo_vector_add-trajectory") is None
    assert demo_store.get("annotations") is None
    assert kernel_files(demo_store) == files_before
    assert demo_store.index_entries() == entries_before
    assert demo_store.integrity_scan().ok


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
        "kernels/demo_vector_add/annotations/annotation-k.json",
        f"{ALG}/algorithm.json",
        f"{ALG}/annotations/annotation-alg.json",
        f"{CFG}/config.json",
        f"{PR101}/commits/commit-demo-a/runs/run-demo-a/run.json",
        f"{CFG}/annotations/annotation-x.json",
        # The exclusion is directory-based: a record slug named "trajectory" inside a shape is a record.
        f"{CFG}/attempt/trajectory/pr.json",
        f"{CFG}/attempt/pr-demo-101/commits/trajectory/commit.json",
        # A config id whose slug is "trajectory" is legal under an algorithm (only algorithm ids reserve it).
        f"{ALG}/trajectory/config.json",
    ],
)
def test_is_record_file_accepts_authoritative_records(relpath: str) -> None:
    assert layout.is_record_file(PurePosixPath(relpath)) is True
    assert layout.is_view_path(PurePosixPath(relpath)) is False


@pytest.mark.parametrize(
    "relpath",
    [
        f"{VIEWS}/trajectory.json",
        f"{VIEWS}/memory_records.jsonl",
        f"{VIEWS}/shapes/cfg-demo/trajectory.json",
        f"{VIEWS}/shapes/cfg-demo/memory_records.jsonl",
        f"{VIEWS}/shapes/cfg-demo/context.json",
        f"{VIEWS}/shapes/nested/looks-like-a-record.json",  # the whole subtree, whatever the file name
        f"{CFG}/.config.json.tmp-123-abcdef01",
        f"{CFG}/.hidden.json",
        f"{CFG}/notes.txt",
        f"{ALG}/.algorithm.json.tmp-4-feedface",
        "artifacts/registry/run-demo-a-samples.json",
        "requests/request-x/request.json",
        "manifest.json",
        "journal/records.jsonl",
        ".runtime/pending/txn-x.json",
    ],
)
def test_is_record_file_excludes_views_dotfiles_and_non_kernel_paths(relpath: str) -> None:
    assert layout.is_record_file(PurePosixPath(relpath)) is False


@pytest.mark.parametrize(
    "relpath,expected",
    [
        (f"{VIEWS}/trajectory.json", True),
        (f"{VIEWS}/memory_records.jsonl", True),
        (f"{VIEWS}/shapes/cfg-demo/trajectory.json", True),
        (f"{VIEWS}/shapes/cfg-demo/memory_records.jsonl", True),
        (f"{VIEWS}/shapes/nested/looks-like-a-record.json", True),
        ("kernels/demo_vector_add/kernel.json", False),
        (f"{CFG}/config.json", False),
        (f"{CFG}/attempt/trajectory/pr.json", False),
        (f"{ALG}/trajectory/config.json", False),
        ("trajectory/x.json", False),
        ("kernels/demo_vector_add/trajectory", False),  # the directory itself (3 parts) is not a file path
        ("artifacts/registry/x.json", False),
    ],
)
def test_is_view_path(relpath: str, expected: bool) -> None:
    assert layout.is_view_path(PurePosixPath(relpath)) is expected


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


def test_config_relpath_rejects_missing_algorithm(store: MemoryStore, bundle_records: list[Record]) -> None:
    by_id = {r.record_id: r for r in bundle_records}
    store.publish(by_id["kernel-demo"])
    config = by_id["cfg-demo"]
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(config, store.get, store.kernel_by_kernel_id)
    assert info.value.exit_code == 3
    assert info.value.details["record_id"] == PLACEHOLDER_ALGORITHM_ID
    with pytest.raises(MissingReferenceError) as info2:
        store.publish(config)
    assert info2.value.code == "MISSING_REFERENCE"
    assert info2.value.details["target"] == PLACEHOLDER_ALGORITHM_ID
    assert store.get("cfg-demo") is None
    assert kernel_files(store) == ["kernels/demo_vector_add/kernel.json"]


def test_algorithm_relpath_rejects_unknown_kernel(store: MemoryStore, bundle_records: list[Record]) -> None:
    algorithm = next(r for r in bundle_records if r.record_id == PLACEHOLDER_ALGORITHM_ID)
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(algorithm, store.get, store.kernel_by_kernel_id)
    assert info.value.details == {"record_id": PLACEHOLDER_ALGORITHM_ID, "kernel_id": "demo_vector_add"}
    with pytest.raises(MissingReferenceError):
        store.publish(algorithm)
    assert store.get(PLACEHOLDER_ALGORITHM_ID) is None


def test_algorithm_kernel_mismatch(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    other_kernel = record_dict(bundle_dicts, "kernel-demo")
    other_kernel["record_id"] = "kernel-other"
    other_kernel["payload"]["kernel_id"] = "other_kernel"
    other_algorithm = record_dict(bundle_dicts, PLACEHOLDER_ALGORITHM_ID)
    other_algorithm["record_id"] = "algorithm-other_kernel-unspecified"
    other_algorithm["payload"]["kernel_id"] = "other_kernel"
    outcome = demo_store.publish_bundle([Record.from_dict(other_algorithm), Record.from_dict(other_kernel)])
    assert outcome.published == ["kernel-other", "algorithm-other_kernel-unspecified"]
    assert demo_store.record_path("algorithm-other_kernel-unspecified") == demo_store.root / "kernels/other_kernel/unspecified/algorithm.json"
    files_before = kernel_files(demo_store)

    mismatch = record_dict(bundle_dicts, "cfg-demo")
    mismatch["record_id"] = "cfg-mismatch"
    mismatch["payload"]["algorithm_ref"] = "algorithm-other_kernel-unspecified"  # kernel_id stays demo_vector_add
    record = Record.from_dict(mismatch)
    with pytest.raises(MissingReferenceError) as info:
        layout.record_relpath(record, demo_store.get, demo_store.kernel_by_kernel_id)
    assert info.value.code == "ALGORITHM_KERNEL_MISMATCH"
    assert info.value.exit_code == 3
    assert info.value.details["record_id"] == "cfg-mismatch"
    assert info.value.details["algorithm_ref"] == "algorithm-other_kernel-unspecified"
    with pytest.raises(MissingReferenceError) as info2:
        demo_store.publish(record)
    assert info2.value.code == "ALGORITHM_KERNEL_MISMATCH"
    assert demo_store.get("cfg-mismatch") is None
    assert kernel_files(demo_store) == files_before
    assert demo_store.integrity_scan().ok


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


def test_record_relpath_for_algorithm_and_its_annotations(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    algorithm = demo_store.require(PLACEHOLDER_ALGORITHM_ID, "algorithm")
    rel = layout.record_relpath(algorithm, demo_store.get, demo_store.kernel_by_kernel_id)
    assert str(rel) == f"{ALG}/algorithm.json"
    assert str(layout.algorithm_dir_for(algorithm, demo_store.kernel_by_kernel_id)) == ALG
    assert demo_store.record_path(PLACEHOLDER_ALGORITHM_ID) == demo_store.root / rel
    # The algorithm has no config directory; a config has no algorithm directory.
    with pytest.raises(MissingReferenceError):
        layout.config_dir_for(algorithm, demo_store.get, demo_store.kernel_by_kernel_id)
    with pytest.raises(MissingReferenceError):
        layout.algorithm_dir_for(demo_store.require("cfg-demo"), demo_store.kernel_by_kernel_id)

    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["record_id"] = "annotation-alg-note"
    data["payload"]["target_ref"] = PLACEHOLDER_ALGORITHM_ID
    data["payload"]["evidence_refs"] = []
    note = Record.from_dict(data)
    note_rel = layout.record_relpath(note, demo_store.get, demo_store.kernel_by_kernel_id)
    assert str(note_rel) == f"{ALG}/annotations/annotation-alg-note.json"
    demo_store.publish(note)
    assert demo_store.record_path("annotation-alg-note") == demo_store.root / note_rel
    assert layout.is_record_file(note_rel)
    with pytest.raises(MissingReferenceError):
        layout.config_dir_for(note, demo_store.get, demo_store.kernel_by_kernel_id)
    assert demo_store.integrity_scan().ok
