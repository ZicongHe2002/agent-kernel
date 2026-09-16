"""Bundle import/export tests (specification sections 9.2, 13, 15, 19, 22; scenarios T12, T19, T20, T27, T28, T32).

Includes the twelve handoff negative cases from
``kernel_memory_ai_handoff/tools/test_handoff_negative_cases.py``, reproduced on copies of the
fixture tree under ``tmp_path`` (the handoff directory itself is never touched).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from kernel_memory.domain.errors import (
    IdConflictError,
    InputError,
    InvariantViolation,
    KernelMemoryError,
    MissingReferenceError,
    SchemaValidationError,
    SecurityPolicyError,
    UnsafePathError,
    UnsupportedFormat,
)
from kernel_memory.domain.hashing import artifact_digest
from kernel_memory.domain.models import Record
from kernel_memory.services.importer import ImportReport, export_bundle, import_bundle
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore

BUNDLE_REL = Path("examples") / "demo_bundle.json"
FIXTURE_RUN_IDS = ["run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure"]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def copy_fixtures(fixtures_root: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "handoff"
    shutil.copytree(fixtures_root, dst)
    return dst


def find(bundle: dict, record_id: str) -> dict:
    return next(r for r in bundle["records"] if r["record_id"] == record_id)


def mutate_json(root: Path, func: Callable[[dict], Any]) -> None:
    path = root / BUNDLE_REL
    bundle = json.loads(path.read_text())
    func(bundle)
    path.write_text(json.dumps(bundle, indent=2) + "\n")


def mutate_text(root: Path, old: str, new: str) -> None:
    path = root / BUNDLE_REL
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def do_import(store: MemoryStore, root: Path, **kwargs: Any) -> ImportReport:
    kwargs.setdefault("allow_fixture", True)
    return import_bundle(store, root / BUNDLE_REL, artifact_root=root, **kwargs)


def blob_files(store: MemoryStore) -> list[Path]:
    base = store.root / "artifacts" / "sha256"
    return [p for p in base.rglob("*") if p.is_file()] if base.exists() else []


def assert_nothing_published(store: MemoryStore) -> None:
    store.invalidate_index()
    assert store.records() == []
    assert store.artifact_registry() == []
    assert blob_files(store) == []


def raised_codes(exc: KernelMemoryError) -> set[str]:
    return {exc.code} | set(exc.details.get("codes", []))


def write_bundle(tmp_path: Path, bundle: dict, name: str = "bundle.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------------------
# happy path, idempotency, dry run
# --------------------------------------------------------------------------------------
def test_import_fixture_bundle_into_fresh_store(store: MemoryStore, fixtures_root: Path, bundle_path: Path, bundle_records: list[Record]) -> None:
    report = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    assert report.ok
    assert report.is_fixture is True and report.dry_run is False and report.txn_id is not None
    assert report.records_total == 18 and report.records_published == 18 and report.records_idempotent == 0
    assert report.artifacts_stored == 7 and report.artifacts_idempotent == 0
    assert report.blobs_stored == 5 and report.blobs_idempotent == 0  # three correctness reports share one digest
    assert report.provenance_downgraded == []
    assert {i.code for i in report.issues} == {"FIXTURE_RUN"}
    for record in bundle_records:
        stored = store.get(record.record_id)
        assert stored is not None and stored.canonical_digest() == record.canonical_digest()
        if record.record_type == "run":
            for ref in record.payload.artifacts:
                assert store.has_artifact(ref.sha256)
                assert store.get_artifact_ref(ref.artifact_id) == ref
    data = report.to_dict()
    assert data["ok"] is True and data["records_published"] == 18 and len(data["published_ids"]) == 18
    assert deep_validate(store).ok


def test_import_without_allow_fixture_rejected(store: MemoryStore, fixtures_root: Path, bundle_path: Path) -> None:
    with pytest.raises(InputError) as info:
        import_bundle(store, bundle_path, artifact_root=fixtures_root)
    assert info.value.code == "FIXTURE_REQUIRES_FLAG" and info.value.exit_code == 2
    assert info.value.details["is_fixture"] is True and len(info.value.details["fixture_runs"]) == 4
    assert_nothing_published(store)


def test_reimport_is_idempotent(store: MemoryStore, fixtures_root: Path, bundle_path: Path) -> None:
    first = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    files_before = sorted(str(p) for p in store.root.rglob("*.json"))
    journal_before = (store.root / "journal" / "records.jsonl").read_bytes()
    second = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    assert second.ok
    assert second.records_published == 0 and second.records_idempotent == 18
    assert second.artifacts_stored == 0 and second.artifacts_idempotent == 7
    assert second.blobs_stored == 0 and second.blobs_idempotent == 5
    assert second.txn_id is None and first.txn_id is not None
    assert sorted(str(p) for p in store.root.rglob("*.json")) == files_before
    assert (store.root / "journal" / "records.jsonl").read_bytes() == journal_before


def test_dry_run_writes_nothing(store: MemoryStore, fixtures_root: Path, bundle_path: Path) -> None:
    report = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True, dry_run=True)
    assert report.dry_run is True and report.txn_id is None
    assert report.records_published == 18 and report.records_idempotent == 0
    assert report.artifacts_stored == 7 and report.blobs_stored == 5
    assert_nothing_published(store)
    assert not (store.root / "journal" / "records.jsonl").exists()
    import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    again = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True, dry_run=True)
    assert again.records_published == 0 and again.records_idempotent == 18 and again.artifacts_idempotent == 7


# --------------------------------------------------------------------------------------
# the twelve handoff negative cases
# --------------------------------------------------------------------------------------
def _case_unknown_record_field(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "cfg-demo").update({"unexpected_field": True}))


def _case_tested_source_mismatch(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["source"]["tested_commit"].update({"hex": "f" * 40}))


def _case_unknown_spill_filled_with_zero(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["analysis_metrics"][0].update({"value": 0}))


def _case_incorrect_latency_summary(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["timing"].update({"median_us": 1}))


def _case_fixture_claiming_trusted_worker(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"].update({"provenance": "trusted_worker"}))


def _case_artifact_tampering(root: Path) -> None:
    path = root / "examples" / "artifacts" / "run-demo-a-samples.json"
    path.write_text(path.read_text() + " ")


def _case_wrong_config_hash(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "cfg-demo")["payload"].update({"config_hash": "sha256:" + "0" * 64}))


def _case_cross_pr_membership_error(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "snapshot-demo-102")["payload"].update({"commit_refs": ["commit-demo-a", "commit-demo-c"]}))


def _case_duplicate_record_id(root: Path) -> None:
    mutate_json(root, lambda b: b["records"].append(find(b, "cfg-demo")))


def _case_artifact_path_traversal(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["artifacts"][0].update({"uri": "../../outside.json"}))


def _case_invalid_boolean_dimension(root: Path) -> None:
    mutate_json(root, lambda b: find(b, "cfg-demo")["payload"]["problem"].update({"n": True}))


def _case_duplicate_json_key(root: Path) -> None:
    mutate_text(root, '"bundle_version":', '"bundle_version": "0.2.0", "bundle_version":')


HANDOFF_NEGATIVE_CASES: dict[str, tuple[Callable[[Path], None], type, set[str], int]] = {
    "unknown_record_field": (_case_unknown_record_field, SchemaValidationError, {"SCHEMA_INVALID"}, 2),
    "tested_source_mismatch": (_case_tested_source_mismatch, InvariantViolation, {"EXACT_COMMIT_MISMATCH", "TESTED_SOURCE_MISMATCH"}, 2),
    "unknown_spill_filled_with_zero": (_case_unknown_spill_filled_with_zero, InvariantViolation, {"UNKNOWN_METRIC_NOT_NULL"}, 2),
    "incorrect_latency_summary": (_case_incorrect_latency_summary, InvariantViolation, {"SUMMARY_MISMATCH"}, 2),
    "fixture_claiming_trusted_worker": (_case_fixture_claiming_trusted_worker, InvariantViolation, {"FIXTURE_CLAIMS_TRUST"}, 2),
    "artifact_tampering": (_case_artifact_tampering, InvariantViolation, {"ARTIFACT_CHECKSUM_MISMATCH"}, 2),
    "wrong_config_hash": (_case_wrong_config_hash, InvariantViolation, {"CONFIG_HASH_MISMATCH"}, 2),
    "cross_pr_membership_error": (_case_cross_pr_membership_error, InvariantViolation, {"CROSS_PR_MEMBERSHIP"}, 2),
    "duplicate_record_id": (_case_duplicate_record_id, InvariantViolation, {"DUPLICATE_RECORD_ID"}, 2),
    "artifact_path_traversal": (_case_artifact_path_traversal, UnsafePathError, {"UNSAFE_PATH"}, 7),
    "invalid_boolean_dimension": (_case_invalid_boolean_dimension, InvariantViolation, {"CONFIG_PROBLEM_INVALID", "CONFIG_HASH_MISMATCH"}, 2),
    "duplicate_json_key": (_case_duplicate_json_key, InputError, {"DUPLICATE_JSON_KEY"}, 2),
}


@pytest.mark.parametrize("case", sorted(HANDOFF_NEGATIVE_CASES))
def test_handoff_negative_case_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path, case: str) -> None:
    mutate, exc_type, expected_codes, exit_code = HANDOFF_NEGATIVE_CASES[case]
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate(root)
    with pytest.raises(exc_type) as info:
        do_import(store, root)
    assert info.value.exit_code == exit_code
    assert raised_codes(info.value) & expected_codes, (info.value.code, info.value.details)
    assert_nothing_published(store)


def test_handoff_negative_cases_cover_the_handoff_list() -> None:
    expected = [
        "unknown_record_field", "tested_source_mismatch", "unknown_spill_filled_with_zero", "incorrect_latency_summary",
        "fixture_claiming_trusted_worker", "artifact_tampering", "wrong_config_hash", "cross_pr_membership_error",
        "duplicate_record_id", "artifact_path_traversal", "invalid_boolean_dimension", "duplicate_json_key",
    ]
    assert sorted(HANDOFF_NEGATIVE_CASES) == sorted(expected) and len(expected) == 12


def test_validation_failure_lists_issues(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    _case_cross_pr_membership_error(root)
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    details = info.value.details
    assert details["error_count"] >= 1 and details["codes"] == ["CROSS_PR_MEMBERSHIP"]
    assert details["issues"][0]["record_id"] == "snapshot-demo-102" and details["issues"][0]["severity"] == "error"
    assert info.value.exit_code == 2


def test_schema_error_names_bundle_index_and_record(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    _case_unknown_record_field(root)
    with pytest.raises(SchemaValidationError) as info:
        do_import(store, root)
    assert info.value.details["record_id"] == "cfg-demo" and info.value.details["index"] == 1
    assert info.value.details["bundle"].endswith("demo_bundle.json")


# --------------------------------------------------------------------------------------
# T12 same id, different content
# --------------------------------------------------------------------------------------
def test_t12_modified_record_conflicts_on_reimport(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    do_import(store, root)
    before = store.get("cfg-demo").canonical_digest()
    mutate_json(root, lambda b: find(b, "cfg-demo")["payload"]["tags"].append("changed-after-publication"))
    with pytest.raises(IdConflictError) as info:
        do_import(store, root)
    assert info.value.exit_code == 3 and info.value.code == "ID_CONFLICT"
    assert info.value.details["record_id"] == "cfg-demo"
    with pytest.raises(IdConflictError):
        do_import(store, root, dry_run=True)
    store.invalidate_index()
    assert store.get("cfg-demo").canonical_digest() == before
    assert len(store.records()) == 18


# --------------------------------------------------------------------------------------
# T19 provenance policy
# --------------------------------------------------------------------------------------
def _non_fixture_bundle(fixtures_root: Path, provenance: str) -> dict:
    bundle = json.loads((fixtures_root / BUNDLE_REL).read_text())
    bundle["is_fixture"] = False
    for record in bundle["records"]:
        if record["record_type"] == "run":
            record["payload"]["provenance"] = provenance
    return bundle


def test_t19_trusted_worker_claim_is_downgraded(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    path = write_bundle(tmp_path, _non_fixture_bundle(fixtures_root, "trusted_worker"))
    report = import_bundle(store, path, artifact_root=fixtures_root)  # no allow_fixture needed: nothing is a fixture
    assert report.ok and report.is_fixture is False
    assert sorted(report.provenance_downgraded) == FIXTURE_RUN_IDS
    assert report.records_published == 18
    for run_id in FIXTURE_RUN_IDS:
        assert store.get(run_id).payload.provenance == "imported_unverified"
    assert "FIXTURE_RUN" not in {i.code for i in report.issues}
    again = import_bundle(store, path, artifact_root=fixtures_root)
    assert again.records_published == 0 and again.records_idempotent == 18
    assert sorted(again.provenance_downgraded) == FIXTURE_RUN_IDS


def test_t19_trusted_source_keeps_trusted_worker(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    path = write_bundle(tmp_path, _non_fixture_bundle(fixtures_root, "trusted_worker"))
    report = import_bundle(store, path, artifact_root=fixtures_root, trusted_source=True)
    assert report.provenance_downgraded == []
    assert all(store.get(r).payload.provenance == "trusted_worker" for r in FIXTURE_RUN_IDS)


def test_t19_imported_unverified_is_kept_as_is(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    path = write_bundle(tmp_path, _non_fixture_bundle(fixtures_root, "imported_unverified"))
    report = import_bundle(store, path, artifact_root=fixtures_root)
    assert report.provenance_downgraded == [] and report.records_published == 18


def test_t19_fixture_runs_in_non_fixture_bundle_need_flag(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    bundle = _non_fixture_bundle(fixtures_root, "trusted_worker")
    find(bundle, "run-demo-c")["payload"]["provenance"] = "fixture"
    path = write_bundle(tmp_path, bundle)
    with pytest.raises(InputError) as info:
        import_bundle(store, path, artifact_root=fixtures_root)
    assert info.value.code == "FIXTURE_REQUIRES_FLAG" and info.value.details["fixture_runs"] == ["run-demo-c"]
    assert_nothing_published(store)
    report = import_bundle(store, path, artifact_root=fixtures_root, allow_fixture=True)
    assert report.is_fixture is True
    assert store.get("run-demo-c").payload.provenance == "fixture"
    assert sorted(report.provenance_downgraded) == ["run-demo-a", "run-demo-baseline", "run-demo-c-failure"]


# --------------------------------------------------------------------------------------
# T20 samples versus summaries
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("change", [{"median_us": 1}, {"p90_us": 100}, {"sample_count": 4}])
def test_t20_summary_disagreement_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path, change: dict) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["timing"].update(change))
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    assert raised_codes(info.value) & {"SUMMARY_MISMATCH", "SAMPLE_COUNT_MISMATCH"}
    assert_nothing_published(store)


def test_t20_samples_file_edited_is_checksum_mismatch(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    path = root / "examples" / "artifacts" / "run-demo-a-samples.json"
    path.write_text(json.dumps({"is_fixture": True, "samples": [1, 1, 1, 1, 1], "unit": "microseconds"}))
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    assert info.value.code == "ARTIFACT_CHECKSUM_MISMATCH"
    assert_nothing_published(store)


# --------------------------------------------------------------------------------------
# T27 missing / corrupt artifacts
# --------------------------------------------------------------------------------------
def test_t27_missing_artifact_file(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    (root / "examples" / "artifacts" / "run-demo-a-samples.json").unlink()
    with pytest.raises(InputError) as info:
        do_import(store, root)
    assert info.value.code == "MISSING_ARTIFACT" and info.value.exit_code == 2
    assert info.value.details["artifact_id"] == "run-demo-a-samples"
    assert_nothing_published(store)


def test_t27_allow_missing_artifacts_imports_with_diagnostic(store: MemoryStore, fixtures_root: Path, tmp_path: Path, bundle_records: list[Record]) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    (root / "examples" / "artifacts" / "run-demo-a-samples.json").unlink()
    report = do_import(store, root, allow_missing_artifacts=True)
    assert report.ok and report.records_published == 18
    codes = {i.code for i in report.issues}
    assert {"MISSING_ARTIFACT", "MISSING_EVIDENCE"} <= codes
    assert report.artifacts_stored == 6 and report.blobs_stored == 4
    samples_ref = next(a for a in store.get("run-demo-a").payload.artifacts if a.artifact_id == "run-demo-a-samples")
    assert not store.has_artifact(samples_ref.sha256)
    deep = deep_validate(store)
    assert not deep.ok
    assert any(i.code == "MISSING_EVIDENCE" and i.record_id == "run-demo-a" for i in deep.errors)
    assert "STORE_ARTIFACT_MISSING" in {i.code for i in deep.errors}


def test_t27_artifact_size_mismatch(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["artifacts"][0].update({"size_bytes": 111}))
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    assert info.value.code == "ARTIFACT_SIZE_MISMATCH"
    assert_nothing_published(store)


def test_artifact_too_large(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    with pytest.raises(InputError) as info:
        do_import(store, root, max_artifact_bytes=50)
    assert info.value.code == "ARTIFACT_TOO_LARGE"
    assert_nothing_published(store)


# --------------------------------------------------------------------------------------
# T28 unsafe URIs
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("uri", ["../../outside.json", "examples/../../outside.json", "/etc/passwd", "\\\\server\\share\\x.json"])
def test_t28_traversal_and_absolute_uris_denied(store: MemoryStore, fixtures_root: Path, tmp_path: Path, uri: str) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["artifacts"][0].update({"uri": uri}))
    with pytest.raises(UnsafePathError) as info:
        do_import(store, root)
    assert info.value.exit_code == 7
    assert_nothing_published(store)


@pytest.mark.parametrize("uri", ["https://example.invalid/samples.json", "http://127.0.0.1/x", "file:///etc/passwd", "s3://bucket/key"])
def test_t28_remote_uri_schemes_refused(store: MemoryStore, fixtures_root: Path, tmp_path: Path, uri: str) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["artifacts"][0].update({"uri": uri}))
    with pytest.raises(SecurityPolicyError) as info:
        do_import(store, root)
    assert info.value.exit_code == 7 and info.value.code == "EXTERNAL_URI_REFUSED"
    assert_nothing_published(store)


def test_t28_symlink_escape_denied(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    outside = tmp_path / "outside-samples.json"
    outside.write_bytes((root / "examples" / "artifacts" / "run-demo-a-samples.json").read_bytes())
    link = root / "examples" / "artifacts" / "linked.json"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are not available on this filesystem: {exc}")
    mutate_json(root, lambda b: find(b, "run-demo-a")["payload"]["artifacts"][0].update({"uri": "examples/artifacts/linked.json"}))
    with pytest.raises(UnsafePathError):
        do_import(store, root)
    assert_nothing_published(store)


# --------------------------------------------------------------------------------------
# T32 malformed input
# --------------------------------------------------------------------------------------
def test_t32_duplicate_json_key_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    _case_duplicate_json_key(root)
    with pytest.raises(InputError) as info:
        do_import(store, root)
    assert info.value.code == "DUPLICATE_JSON_KEY" and info.value.exit_code == 2
    assert_nothing_published(store)


def test_t32_nan_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    mutate_text(root, '"median_us": 100,', '"median_us": NaN,')
    with pytest.raises(InputError) as info:
        do_import(store, root)
    assert info.value.code == "NONFINITE_JSON"
    assert_nothing_published(store)


def test_t32_boolean_dimension_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    _case_invalid_boolean_dimension(root)
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    assert "CONFIG_PROBLEM_INVALID" in raised_codes(info.value) and info.value.exit_code == 2
    assert_nothing_published(store)


def test_t32_misspelled_property_rejected(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)

    def misspell(bundle: dict) -> None:
        payload = find(bundle, "run-demo-a")["payload"]
        payload["provenence"] = payload.pop("provenance")

    mutate_json(root, misspell)
    with pytest.raises(SchemaValidationError):
        do_import(store, root)
    assert_nothing_published(store)


def test_bundle_size_limit(store: MemoryStore, fixtures_root: Path, bundle_path: Path) -> None:
    with pytest.raises(InputError) as info:
        import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True, max_bundle_bytes=1000)
    assert info.value.code == "INPUT_TOO_LARGE"


def test_missing_bundle_file(store: MemoryStore, tmp_path: Path) -> None:
    with pytest.raises(InputError) as info:
        import_bundle(store, tmp_path / "nope.json")
    assert info.value.code == "FILE_NOT_FOUND"


@pytest.mark.parametrize(
    "bundle, exc_type, code",
    [
        ([], InputError, "INVALID_BUNDLE"),
        ({"is_fixture": True}, InputError, "INVALID_BUNDLE"),
        ({"records": []}, InputError, "EMPTY_BUNDLE"),
        ({"records": [1], "extra": 1}, InputError, "UNKNOWN_BUNDLE_FIELD"),
        ({"records": [1], "is_fixture": "yes"}, InputError, "INVALID_BUNDLE"),
        ({"records": [1], "bundle_version": "9.9.9"}, UnsupportedFormat, "UNSUPPORTED_BUNDLE_VERSION"),
        ({"records": [1], "artifacts": {}}, InputError, "INVALID_BUNDLE"),
    ],
)
def test_bundle_envelope_errors(store: MemoryStore, tmp_path: Path, bundle: Any, exc_type: type, code: str) -> None:
    path = write_bundle(tmp_path, bundle)
    with pytest.raises(exc_type) as info:
        import_bundle(store, path, allow_fixture=True)
    assert info.value.code == code
    assert_nothing_published(store)


def test_record_that_is_not_an_object_is_schema_error(store: MemoryStore, tmp_path: Path) -> None:
    path = write_bundle(tmp_path, {"records": [42]})
    with pytest.raises(SchemaValidationError) as info:
        import_bundle(store, path)
    assert info.value.details["index"] == 0


# --------------------------------------------------------------------------------------
# artifact:// URIs and atomicity
# --------------------------------------------------------------------------------------
def _rerun_with_cas_uris(fixtures_root: Path) -> dict:
    """A copy of run-demo-a as a second attempt whose artifacts are addressed by digest only."""
    bundle = json.loads((fixtures_root / BUNDLE_REL).read_text())
    run = json.loads(json.dumps(find(bundle, "run-demo-a")))
    run["record_id"] = "run-demo-a-rerun"
    payload = run["payload"]
    payload["request_id"] = "request-run-demo-a-rerun"
    payload["rerun_of"] = "run-demo-a"
    for ref in payload["artifacts"]:
        ref["artifact_id"] = ref["artifact_id"].replace("run-demo-a-", "run-demo-a-rerun-")
        ref["uri"] = "artifact://sha256/" + ref["sha256"].split(":", 1)[1]
    payload["correctness"]["report_artifact_ref"] = "run-demo-a-rerun-correctness"
    payload["timing"]["samples_artifact_ref"] = "run-demo-a-rerun-samples"
    return {"bundle_version": "0.2.0", "is_fixture": True, "records": [run]}


def test_artifact_uri_resolves_from_cas(store: MemoryStore, fixtures_root: Path, bundle_path: Path, tmp_path: Path) -> None:
    import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    path = write_bundle(tmp_path, _rerun_with_cas_uris(fixtures_root))
    report = import_bundle(store, path, allow_fixture=True)
    assert report.ok and report.records_published == 1
    assert report.artifacts_stored == 0 and report.artifacts_idempotent == 2 and report.blobs_stored == 0
    assert store.get("run-demo-a-rerun") is not None
    assert deep_validate(store).ok


def test_artifact_uri_must_already_be_in_cas(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    bundle = json.loads((fixtures_root / BUNDLE_REL).read_text())
    bundle["records"].append(_rerun_with_cas_uris(fixtures_root)["records"][0])
    # Make the digest-addressed blob unavailable in the fixture tree too: point the fixture run at
    # a fresh store where nothing is cached, and drop the file-backed refs that share the digest.
    for record in bundle["records"]:
        if record["record_id"] in ("run-demo-a", "run-demo-c", "run-demo-baseline"):
            for ref in record["payload"]["artifacts"]:
                if ref["kind"] == "correctness_report":
                    ref["availability"] = "expired"
            record["payload"]["correctness"].update({"status": "not_run", "cases_total": 0, "cases_passed": 0, "max_abs_error": None, "max_rel_error": None, "report_artifact_ref": None})
    find(bundle, "run-demo-a-rerun")["payload"]["correctness"].update(
        {"status": "not_run", "cases_total": 0, "cases_passed": 0, "max_abs_error": None, "max_rel_error": None, "report_artifact_ref": None}
    )
    for ref in find(bundle, "run-demo-a-rerun")["payload"]["artifacts"]:
        if ref["kind"] == "correctness_report":
            ref["availability"] = "expired"
    path = write_bundle(tmp_path, bundle)
    with pytest.raises(InputError) as info:
        import_bundle(store, path, artifact_root=fixtures_root, allow_fixture=True)
    assert info.value.code == "MISSING_ARTIFACT" and info.value.details["artifact_id"] == "run-demo-a-rerun-samples"
    assert_nothing_published(store)


def test_artifact_uri_digest_must_match_descriptor(store: MemoryStore, fixtures_root: Path, bundle_path: Path, tmp_path: Path) -> None:
    import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True)
    bundle = _rerun_with_cas_uris(fixtures_root)
    bundle["records"][0]["payload"]["artifacts"][0]["uri"] = "artifact://sha256/" + "0" * 64
    path = write_bundle(tmp_path, bundle)
    with pytest.raises(InvariantViolation) as info:
        import_bundle(store, path, allow_fixture=True)
    assert info.value.code == "ARTIFACT_URI_MISMATCH"
    assert store.get("run-demo-a-rerun") is None


def test_import_is_all_or_nothing(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    root = copy_fixtures(fixtures_root, tmp_path)
    # The very last record of the bundle refers to a missing run: nothing at all may be published.
    mutate_json(root, lambda b: find(b, "annotation-demo-grouped")["payload"].update({"evidence_refs": ["run-nope"]}))
    with pytest.raises(InvariantViolation) as info:
        do_import(store, root)
    assert "MISSING_REFERENCE" in raised_codes(info.value)
    assert_nothing_published(store)


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------
def test_export_bundle_round_trips(demo_store: MemoryStore, tmp_path: Path) -> None:
    exported = export_bundle(demo_store, include_artifacts=True)
    assert exported["bundle_version"] == "0.2.0" and exported["is_fixture"] is True
    assert len(exported["records"]) == 18 and len(exported["artifacts"]) == 5
    assert exported == export_bundle(demo_store, include_artifacts=True)  # deterministic
    ids = [r["record_id"] for r in exported["records"]]
    assert ids[0] == "kernel-demo" and ids[1] == "cfg-demo" and ids == sorted(ids, key=ids.index)
    path = write_bundle(tmp_path, exported, "export.json")
    fresh = MemoryStore.init(tmp_path / "fresh")
    report = import_bundle(fresh, path, allow_fixture=True)  # no artifact_root: the blobs are embedded
    assert report.ok and report.records_published == 18 and report.blobs_stored == 5
    assert sorted(r.canonical_digest() for r in fresh.records()) == sorted(r.canonical_digest() for r in demo_store.records())
    assert sorted(p.name for p in blob_files(fresh)) == sorted(p.name for p in blob_files(demo_store))
    assert deep_validate(fresh).ok


def test_export_without_artifacts_needs_artifact_root_on_import(demo_store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    exported = export_bundle(demo_store)
    assert "artifacts" not in exported
    path = write_bundle(tmp_path, exported, "export.json")
    with pytest.raises(InputError) as info:
        import_bundle(MemoryStore.init(tmp_path / "fresh-a"), path, allow_fixture=True)
    assert info.value.code == "MISSING_ARTIFACT"
    report = import_bundle(MemoryStore.init(tmp_path / "fresh-b"), path, artifact_root=fixtures_root, allow_fixture=True)
    assert report.ok and report.records_published == 18


def test_export_scoped_to_config(demo_store: MemoryStore) -> None:
    exported = export_bundle(demo_store, config_ref="cfg-demo")
    assert len(exported["records"]) == 18
    assert exported["records"][0]["record_id"] == "kernel-demo"
    with pytest.raises(MissingReferenceError):
        export_bundle(demo_store, config_ref="cfg-nope")


def test_export_is_fixture_reflects_runs(store: MemoryStore, fixtures_root: Path, tmp_path: Path) -> None:
    path = write_bundle(tmp_path, _non_fixture_bundle(fixtures_root, "imported_unverified"))
    import_bundle(store, path, artifact_root=fixtures_root)
    exported = export_bundle(store)
    assert exported["is_fixture"] is False
    assert "fixture" not in exported["description"]


def test_embedded_artifact_tamper_rejected(demo_store: MemoryStore, tmp_path: Path) -> None:
    exported = export_bundle(demo_store, include_artifacts=True)
    blob = exported["artifacts"][0]
    blob["data"] = blob["data"][:-4] + "AAA="
    path = write_bundle(tmp_path, exported, "export.json")
    fresh = MemoryStore.init(tmp_path / "fresh")
    with pytest.raises(InvariantViolation) as info:
        import_bundle(fresh, path, allow_fixture=True)
    assert info.value.code == "ARTIFACT_CHECKSUM_MISMATCH"
    assert_nothing_published(fresh)
    assert artifact_digest(b"") != blob["sha256"]


def test_embedded_artifact_envelope_errors(demo_store: MemoryStore, tmp_path: Path) -> None:
    exported = export_bundle(demo_store, include_artifacts=True)
    exported["artifacts"][0]["encoding"] = "hex"
    with pytest.raises(InputError) as info:
        import_bundle(MemoryStore.init(tmp_path / "f1"), write_bundle(tmp_path, exported, "e1.json"), allow_fixture=True)
    assert info.value.code == "UNSUPPORTED_ARTIFACT_ENCODING"
    exported = export_bundle(demo_store, include_artifacts=True)
    exported["artifacts"][0]["data"] = "not base64!!"
    with pytest.raises(InputError) as info:
        import_bundle(MemoryStore.init(tmp_path / "f2"), write_bundle(tmp_path, exported, "e2.json"), allow_fixture=True)
    assert info.value.code == "INVALID_BUNDLE"


def test_import_report_to_dict_shape(store: MemoryStore, fixtures_root: Path, bundle_path: Path) -> None:
    data = import_bundle(store, bundle_path, artifact_root=fixtures_root, allow_fixture=True).to_dict()
    expected_keys = {
        "ok", "bundle_path", "is_fixture", "records_total", "records_published", "records_idempotent", "artifacts_stored",
        "artifacts_idempotent", "blobs_stored", "blobs_idempotent", "provenance_downgraded", "published_ids", "idempotent_ids",
        "issues", "warning_count", "dry_run", "txn_id",
    }
    assert set(data) == expected_keys
    assert data["warning_count"] == 4 and all(i["severity"] == "warning" for i in data["issues"])
