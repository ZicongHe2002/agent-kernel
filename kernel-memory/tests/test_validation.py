"""Deep validation tests (specification sections 6, 7, 11.2, 13; scenarios T04, T06, T14-T17, T19, T20, T27, T32).

Every gate has a negative test built by mutating fixture record dicts (never the shared
fixture objects) and validating with an artifact reader backed by the fixture blobs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from conftest import record_dict

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError
from kernel_memory.domain.models import Record
from kernel_memory.services.validation import (
    ERROR,
    WARNING,
    Issue,
    ValidationReport,
    deep_validate,
    find_cycle,
    validate_records,
)
from kernel_memory.storage import MemoryStore

ZERO_HASH = "sha256:" + "0" * 64
OID = lambda n: {"algorithm": "sha1", "hex": f"{n:040x}"}  # noqa: E731


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
@pytest.fixture
def blobs(artifact_root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in sorted((artifact_root / "examples" / "artifacts").glob("*.json")):
        data = path.read_bytes()
        out[hashing.artifact_digest(data)] = data
    return out


def clone(dicts: list[dict]) -> list[dict]:
    return json.loads(json.dumps(dicts))


def mutated(bundle_dicts: list[dict], record_id: str, fn: Callable[[dict], Any]) -> list[dict]:
    dicts = clone(bundle_dicts)
    fn(next(d for d in dicts if d["record_id"] == record_id))
    return dicts


def validate(dicts: list[dict], blobs: dict[str, bytes], **kwargs: Any) -> ValidationReport:
    records = [Record.from_dict(d) for d in clone(dicts)]
    return validate_records(records, artifact_reader=lambda sha: blobs.get(sha), **kwargs)


def error_codes(report: ValidationReport) -> set[str]:
    return {i.code for i in report.errors}


def warning_codes(report: ValidationReport) -> set[str]:
    return {i.code for i in report.warnings}


def issues_for(report: ValidationReport, record_id: str) -> list[Issue]:
    return [i for i in report.errors if i.record_id == record_id]


def swap_samples_artifact(run: dict, blobs: dict[str, bytes], content: bytes) -> str:
    """Point the run's samples artifact at new bytes (kept in ``blobs``); returns the new digest."""
    sha = hashing.artifact_digest(content)
    blobs[sha] = content
    ref = next(a for a in run["payload"]["artifacts"] if a["artifact_id"] == run["payload"]["timing"]["samples_artifact_ref"])
    ref["sha256"] = sha
    ref["size_bytes"] = len(content)
    return sha


def samples_sha(bundle_dicts: list[dict], run_id: str) -> str:
    run = record_dict(bundle_dicts, run_id)
    ref = next(a for a in run["payload"]["artifacts"] if a["artifact_id"] == run["payload"]["timing"]["samples_artifact_ref"])
    return ref["sha256"]


# --------------------------------------------------------------------------------------
# happy paths and report shape
# --------------------------------------------------------------------------------------
def test_deep_validate_demo_store_ok(demo_store: MemoryStore) -> None:
    report = deep_validate(demo_store)
    assert report.ok, [i.to_dict() for i in report.errors]
    assert report.records_checked == 18
    assert report.runs_checked == 4
    assert report.artifact_checks == 7
    assert report.summary_checks == 6  # three recorded timings x (median, p90)
    assert warning_codes(report) == {"FIXTURE_RUN"}
    data = report.to_dict()
    assert data["ok"] is True and data["error_count"] == 0 and data["warning_count"] == 4
    assert all(set(i) == {"severity", "code", "message", "record_id", "field", "details"} for i in data["issues"])


def test_fixture_set_validates_clean(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(bundle_dicts, blobs)
    assert report.errors == []
    assert warning_codes(report) == {"FIXTURE_RUN"}
    assert {i.record_id for i in report.warnings} == {"run-demo-baseline", "run-demo-a", "run-demo-c", "run-demo-c-failure"}


def test_deep_validate_empty_store(store: MemoryStore) -> None:
    report = deep_validate(store)
    assert report.ok and report.records_checked == 0 and report.issues == []


def test_validate_records_rejects_non_records(blobs: dict[str, bytes]) -> None:
    with pytest.raises(InputError):
        validate_records([{"record_type": "kernel"}], artifact_reader=lambda sha: None)  # type: ignore[list-item]
    with pytest.raises(InputError):
        validate_records([], artifact_reader=lambda sha: None, missing_evidence_severity="fatal")


def test_external_resolver_supplies_store_records(demo_store: MemoryStore, bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    only_run = [record_dict(bundle_dicts, "run-demo-a")]
    alone = validate(only_run, blobs)
    assert "MISSING_REFERENCE" in error_codes(alone)
    resolved = validate(
        only_run,
        blobs,
        external_resolver=demo_store.get,
        kernel_resolver=demo_store.kernel_by_kernel_id,
        artifact_registry=demo_store.get_artifact_ref,
    )
    assert resolved.errors == [], [i.to_dict() for i in resolved.errors]


# --------------------------------------------------------------------------------------
# set-level checks
# --------------------------------------------------------------------------------------
def test_duplicate_record_id(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    dicts = clone(bundle_dicts) + [record_dict(bundle_dicts, "cfg-demo")]
    report = validate(dicts, blobs)
    dup = [i for i in report.errors if i.code == "DUPLICATE_RECORD_ID"]
    assert dup and dup[0].record_id == "cfg-demo" and dup[0].details["identical_content"] is True
    changed = record_dict(bundle_dicts, "cfg-demo")
    changed["payload"]["tags"].append("changed")
    report = validate(clone(bundle_dicts) + [changed], blobs)
    assert any(i.code == "DUPLICATE_RECORD_ID" and i.details["identical_content"] is False for i in report.errors)


def test_missing_reference(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "pr-demo-101", lambda d: d["payload"].update(config_ref="cfg-nope")), blobs)
    issue = next(i for i in report.errors if i.code == "MISSING_REFERENCE")
    assert issue.record_id == "pr-demo-101" and issue.field == "config_ref" and issue.details["target"] == "cfg-nope"


def test_wrong_reference_type(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "pr-demo-101", lambda d: d["payload"].update(config_ref="run-demo-a")), blobs)
    issue = next(i for i in report.errors if i.code == "WRONG_REFERENCE_TYPE")
    assert issue.details["target_type"] == "run" and issue.details["allowed_types"] == ["config"]


def test_duplicate_attempt(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    twin = record_dict(bundle_dicts, "run-demo-a")
    twin["record_id"] = "run-demo-a-twin"
    report = validate(clone(bundle_dicts) + [twin], blobs)
    assert "DUPLICATE_ATTEMPT" in error_codes(report)


def test_duplicate_kernel_id(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    twin = record_dict(bundle_dicts, "kernel-demo")
    twin["record_id"] = "kernel-demo-twin"
    assert "DUPLICATE_KERNEL_ID" in error_codes(validate(clone(bundle_dicts) + [twin], blobs))


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def test_config_hash_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "cfg-demo", lambda d: d["payload"].update(config_hash=ZERO_HASH)), blobs)
    issue = next(i for i in report.errors if i.code == "CONFIG_HASH_MISMATCH")
    assert issue.details["recorded"] == ZERO_HASH and issue.details["computed"].startswith("sha256:")
    assert "COMPARISON_KEY_MISMATCH" in error_codes(report)  # runs hash the config's wrong hash


def test_t32_boolean_dimension_rejected(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "cfg-demo", lambda d: d["payload"]["problem"].update(n=True)), blobs)
    assert "CONFIG_PROBLEM_INVALID" in error_codes(report)


def test_config_problem_not_normalized(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    def denormalize(d: dict) -> None:
        d["payload"]["problem"]["dtype"] = "f32"  # alias, not canonical form
        d["payload"]["config_hash"] = hashing.config_hash(
            kernel_id=d["payload"]["kernel_id"],
            problem_schema_id=d["payload"]["problem_schema_id"],
            problem_schema_digest=d["payload"]["problem_schema_digest"],
            problem=d["payload"]["problem"],
        )

    report = validate(mutated(bundle_dicts, "cfg-demo", denormalize), blobs)
    assert "CONFIG_PROBLEM_INVALID" in error_codes(report)
    assert "CONFIG_HASH_MISMATCH" not in error_codes(report)


def test_missing_kernel(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    dicts = [d for d in clone(bundle_dicts) if d["record_id"] != "kernel-demo"]
    issue = next(i for i in validate(dicts, blobs).errors if i.code == "MISSING_KERNEL")
    assert issue.record_id == "cfg-demo" and issue.details["kernel_id"] == "demo_vector_add"


# --------------------------------------------------------------------------------------
# pr / snapshot
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mutation", [{"number": None}, {"provider": "local", "pr_key": "local-x-1"}])
def test_pr_provider_number_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes], mutation: dict) -> None:
    report = validate(mutated(bundle_dicts, "pr-demo-101", lambda d: d["payload"].update(mutation)), blobs)
    assert "PROVIDER_NUMBER_MISMATCH" in error_codes(report)


def test_pr_key_convention_warning(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "pr-demo-101", lambda d: d["payload"].update(pr_key="gh-1-pr-1")), blobs)
    assert report.ok and "PR_KEY_CONVENTION" in warning_codes(report)


def test_cross_pr_membership(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(
        mutated(bundle_dicts, "snapshot-demo-102", lambda d: d["payload"].update(commit_refs=["commit-demo-a", "commit-demo-c"])), blobs
    )
    issue = next(i for i in report.errors if i.code == "CROSS_PR_MEMBERSHIP")
    assert issue.field == "commit_refs[0]" and issue.details["commit_pr"] == "pr-demo-101"


def test_snapshot_previous_from_other_pr(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "snapshot-demo-102", lambda d: d["payload"].update(previous_snapshot_ref="snapshot-demo-101")), blobs)
    assert "SNAPSHOT_CHAIN_MISMATCH" in error_codes(report)


def test_snapshot_chain_cycle(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    dicts = mutated(bundle_dicts, "snapshot-demo-101", lambda d: d["payload"].update(previous_snapshot_ref="snapshot-demo-102"))
    dicts = mutated(dicts, "snapshot-demo-102", lambda d: d["payload"].update(previous_snapshot_ref="snapshot-demo-101"))
    report = validate(dicts, blobs)
    assert "SNAPSHOT_CHAIN_CYCLE" in error_codes(report)
    assert sum(1 for i in report.errors if i.code == "SNAPSHOT_CHAIN_CYCLE") == 1


def test_snapshot_partial_requires_reason(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "snapshot-demo-101", lambda d: d["payload"].update(enumeration_status="partial")), blobs)
    assert "MISSING_ENUMERATION_REASON" in error_codes(report)
    report = validate(
        mutated(bundle_dicts, "snapshot-demo-101", lambda d: d["payload"].update(enumeration_status="partial", reason="pagination stopped at 250")),
        blobs,
    )
    assert report.ok


def test_snapshot_duplicate_commit_ref(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "snapshot-demo-101", lambda d: d["payload"].update(commit_refs=["commit-demo-a", "commit-demo-a"])), blobs)
    assert "DUPLICATE_COMMIT_REF" in error_codes(report)


# --------------------------------------------------------------------------------------
# commit
# --------------------------------------------------------------------------------------
def test_commit_repo_uid_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "commit-demo-b", lambda d: d["payload"].update(repo_uid="github:github.com:repo:1")), blobs)
    assert any(i.code == "REPO_UID_MISMATCH" and i.record_id == "commit-demo-b" for i in report.errors)


@pytest.mark.parametrize("status", ["not_extracted", "no_code_change"])
def test_unextracted_changes_not_empty(bundle_dicts: list[dict], blobs: dict[str, bytes], status: str) -> None:
    change = record_dict(bundle_dicts, "commit-demo-a")["payload"]["changes"][0]
    report = validate(mutated(bundle_dicts, "commit-demo-b", lambda d: d["payload"].update(change_status=status, changes=[change])), blobs)
    assert "UNEXTRACTED_CHANGES_NOT_EMPTY" in error_codes(report)


def test_duplicate_change_id(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    def dup(d: dict) -> None:
        d["payload"]["changes"][1]["change_id"] = d["payload"]["changes"][0]["change_id"]

    assert "DUPLICATE_CHANGE_ID" in error_codes(validate(mutated(bundle_dicts, "commit-demo-c", dup), blobs))


def test_t04_isolated_attribution_needs_evidence(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    def isolate(d: dict) -> None:
        d["payload"]["changes"][0]["attribution"] = "isolated"

    report = validate(mutated(bundle_dicts, "commit-demo-a", isolate), blobs)
    assert report.ok and "ISOLATED_ATTRIBUTION_UNSUPPORTED" in warning_codes(report)
    # An annotation with evidence targeting the commit supports the claim.
    dicts = mutated(bundle_dicts, "commit-demo-a", isolate)
    dicts = mutated(dicts, "annotation-demo-grouped", lambda d: d["payload"].update(target_ref="commit-demo-a", evidence_refs=["run-demo-a"]))
    assert "ISOLATED_ATTRIBUTION_UNSUPPORTED" not in warning_codes(validate(dicts, blobs))


def test_diff_base_not_parent_warning(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "commit-demo-a", lambda d: d["payload"].update(diff_base_oid=OID(0x77))), blobs)
    assert report.ok and "DIFF_BASE_NOT_PARENT" in warning_codes(report)


# --------------------------------------------------------------------------------------
# relation / annotation
# --------------------------------------------------------------------------------------
def test_self_relation(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "relation-demo-origin", lambda d: d["payload"].update(to_ref="commit-demo-a")), blobs)
    assert "SELF_RELATION" in error_codes(report)


def test_relation_endpoint_type(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "relation-demo-origin", lambda d: d["payload"].update(from_ref="pr-demo-101")), blobs)
    issue = next(i for i in report.errors if i.code == "RELATION_ENDPOINT_TYPE")
    assert issue.field == "from_ref" and issue.details["target_type"] == "pr"
    # inspired_by may point at any record type
    report = validate(mutated(bundle_dicts, "relation-demo-origin", lambda d: d["payload"].update(kind="inspired_by", from_ref="pr-demo-101")), blobs)
    assert "RELATION_ENDPOINT_TYPE" not in error_codes(report)


def test_relation_evidence_must_exist(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "relation-demo-origin", lambda d: d["payload"].update(evidence_refs=["run-nope"])), blobs)
    assert any(i.code == "MISSING_REFERENCE" and i.field == "evidence_refs[0]" for i in report.errors)


def test_annotation_target_and_evidence_must_exist(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "annotation-demo-grouped", lambda d: d["payload"].update(target_ref="commit-nope")), blobs)
    assert any(i.code == "MISSING_REFERENCE" and i.field == "target_ref" for i in report.errors)
    report = validate(mutated(bundle_dicts, "annotation-demo-grouped", lambda d: d["payload"].update(evidence_refs=["run-nope"])), blobs)
    assert any(i.code == "MISSING_REFERENCE" and i.field == "evidence_refs[0]" for i in report.errors)
    report = validate(mutated(bundle_dicts, "annotation-demo-grouped", lambda d: d["payload"].update(supersedes_ref="annotation-demo-grouped")), blobs)
    assert "SELF_REFERENCE" in error_codes(report)


# --------------------------------------------------------------------------------------
# run: source and hashes
# --------------------------------------------------------------------------------------
def test_tested_source_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(target_commit=OID(5))), blobs)
    issue = next(i for i in report.errors if i.code == "TESTED_SOURCE_MISMATCH")
    assert issue.record_id == "run-demo-a" and issue.details["subject"] == "commit-demo-a"


def test_run_config_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    other_cfg = record_dict(bundle_dicts, "cfg-demo")
    other_cfg["record_id"] = "cfg-demo-copy"
    dicts = mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"].update(config_ref="cfg-demo-copy")) + [other_cfg]
    report = validate(dicts, blobs)
    assert any(i.code == "RUN_CONFIG_MISMATCH" and i.record_id == "run-demo-a" for i in report.errors)
    assert "COMPARISON_KEY_MISMATCH" not in error_codes(report)  # same config_hash, so the key still holds


def test_exact_commit_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"]["tested_commit"].update(hex="f" * 40)), blobs)
    assert any(i.code == "EXACT_COMMIT_MISMATCH" and i.field == "source.tested_commit" for i in report.errors)
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(merge_parent_oids=[OID(1)])), blobs)
    assert any(i.code == "EXACT_COMMIT_MISMATCH" and i.field == "source.merge_parent_oids" for i in report.errors)


def test_t08_integration_merge_identity(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(checkout_mode="integration_merge")), blobs)
    assert "INTEGRATION_MERGE_IDENTITY" in error_codes(report)
    assert "INTEGRATION_MERGE_WITHOUT_PARENTS" in warning_codes(report)


def test_variant_hash_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(entrypoint="demo.other:fn")), blobs)
    issue = next(i for i in report.errors if i.code == "VARIANT_HASH_MISMATCH")
    assert issue.details["computed"] == hashing.variant_digest(
        source_digest=record_dict(bundle_dicts, "run-demo-a")["payload"]["source"]["source_digest"],
        entrypoint="demo.other:fn",
        implementation_overrides={},
        checkout_mode="exact_commit",
    )


@pytest.mark.parametrize(
    "snapshot, change",
    [("environment", {"device_count": 2}), ("protocol", {"warmup": 3}), ("verifier", {"nonfinite_policy": "allow"})],
)
def test_t23_snapshot_hash_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes], snapshot: str, change: dict) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"][snapshot].update(change)), blobs)
    issue = next(i for i in report.errors if i.code == "SNAPSHOT_HASH_MISMATCH")
    assert issue.details["snapshot"] == snapshot
    assert "COMPARISON_KEY_MISMATCH" not in error_codes(report)  # recorded snapshot hashes were kept


def test_comparison_key_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"].update(comparison_key=ZERO_HASH)), blobs)
    assert "COMPARISON_KEY_MISMATCH" in error_codes(report)
    assert "DECISION_GROUP_MISMATCH" in error_codes(report)  # the decision references run-demo-a


def test_dirty_without_patch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(dirty=True)), blobs)
    assert "DIRTY_WITHOUT_PATCH" in error_codes(report)
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(dirty=True, patch_digest=ZERO_HASH)), blobs)
    assert "DIRTY_WITHOUT_PATCH" not in error_codes(report)


def test_run_repo_uid_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["source"].update(repo_uid="github:github.com:repo:2")), blobs)
    assert any(i.code == "REPO_UID_MISMATCH" and i.record_id == "run-demo-a" for i in report.errors)


# --------------------------------------------------------------------------------------
# run: artifacts, correctness, timing
# --------------------------------------------------------------------------------------
def test_duplicate_artifact_id(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["artifacts"].append(dict(d["payload"]["artifacts"][0]))), blobs)
    assert "DUPLICATE_ARTIFACT_ID" in error_codes(report)


def test_artifact_descriptor_conflict_across_runs(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    def rename(d: dict) -> None:
        d["payload"]["artifacts"][1]["artifact_id"] = "run-demo-baseline-correctness"  # same id, different uri
        d["payload"]["correctness"]["report_artifact_ref"] = "run-demo-baseline-correctness"

    issue = next(i for i in validate(mutated(bundle_dicts, "run-demo-a", rename), blobs).errors if i.code == "ARTIFACT_DESCRIPTOR_CONFLICT")
    assert issue.details["other_run"] == "run-demo-baseline"


def test_artifact_descriptor_conflict_with_registry(demo_store: MemoryStore, bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    run = record_dict(bundle_dicts, "run-demo-a")
    run["payload"]["artifacts"][0]["uri"] = "examples/artifacts/elsewhere.json"
    report = validate([run], blobs, external_resolver=demo_store.get, artifact_registry=demo_store.get_artifact_ref)
    assert any(i.code == "ARTIFACT_DESCRIPTOR_CONFLICT" and "registered" in i.details for i in report.errors)


def test_dangling_in_run_ref(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(report_artifact_ref="nope")), blobs)
    assert any(i.code == "DANGLING_IN_RUN_REF" and i.field == "correctness.report_artifact_ref" for i in report.errors)
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(samples_artifact_ref="nope")), blobs)
    assert any(i.code == "DANGLING_IN_RUN_REF" and i.field == "timing.samples_artifact_ref" for i in report.errors)
    report = validate(mutated(bundle_dicts, "run-demo-c", lambda d: d["payload"]["analysis_metrics"][0].update(source_artifact_ref="nope")), blobs)
    assert any(i.code == "DANGLING_IN_RUN_REF" and i.field.startswith("analysis_metrics[0]") for i in report.errors)


@pytest.mark.parametrize("counts", [{"cases_passed": 0}, {"cases_total": 0, "cases_passed": 0}])
def test_invalid_pass(bundle_dicts: list[dict], blobs: dict[str, bytes], counts: dict) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(counts)), blobs)
    assert "INVALID_PASS" in error_codes(report)


def test_invalid_correctness_counts(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(cases_passed=2)), blobs)
    assert {"INVALID_CORRECTNESS_COUNTS", "INVALID_PASS"} <= error_codes(report)


def test_missing_correctness_evidence(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(report_artifact_ref=None)), blobs)
    assert "MISSING_CORRECTNESS_EVIDENCE" in error_codes(report)


def test_t15_succeeded_with_fail_is_valid_but_invalid_fail_is_not(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(status="fail", cases_passed=0)), blobs)
    assert issues_for(report, "run-demo-a") == []
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["correctness"].update(status="fail")), blobs)
    assert "INVALID_FAIL" in error_codes(report)


def test_correctness_not_run_with_results(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-c-failure", lambda d: d["payload"]["correctness"].update(cases_total=1, cases_passed=1)), blobs)
    assert "CORRECTNESS_NOT_RUN_WITH_RESULTS" in error_codes(report)


def test_t14_compile_error_with_results_is_fabricated(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"].update(execution_status="compile_error", failure_reason="boom")), blobs)
    issue = next(i for i in report.errors if i.code == "FABRICATED_RESULT")
    assert issue.details == {"execution_status": "compile_error", "correctness_status": "pass", "timing_status": "recorded"}
    # The fixture's own compile failure carries no results and only lacks nothing.
    assert issues_for(validate(bundle_dicts, blobs), "run-demo-c-failure") == []


def test_missing_failure_reason_warning(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-c-failure", lambda d: d["payload"].update(failure_reason=None)), blobs)
    assert report.ok and "MISSING_FAILURE_REASON" in warning_codes(report)


def test_missing_timing_evidence(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(samples_artifact_ref=None)), blobs)
    assert "MISSING_TIMING_EVIDENCE" in error_codes(report)


def test_summary_missing(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(median_us=None)), blobs)
    assert "SUMMARY_MISSING" in error_codes(report) and "SUMMARY_MISMATCH" not in error_codes(report)


def test_t27_missing_evidence(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    sha = samples_sha(bundle_dicts, "run-demo-a")
    partial = {k: v for k, v in blobs.items() if k != sha}
    report = validate(bundle_dicts, partial)
    issue = next(i for i in report.errors if i.code == "MISSING_EVIDENCE")
    assert issue.record_id == "run-demo-a" and issue.details["sha256"] == sha
    assert "SUMMARY_MISMATCH" not in error_codes(report)  # nothing to compare against; never invented
    downgraded = validate(bundle_dicts, partial, missing_evidence_severity="warning")
    assert downgraded.ok and "MISSING_EVIDENCE" in warning_codes(downgraded)


def test_t27_artifact_corrupt(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    sha = samples_sha(bundle_dicts, "run-demo-a")
    tampered = dict(blobs)
    tampered[sha] = blobs[sha] + b" "
    report = validate(bundle_dicts, tampered)
    issue = next(i for i in report.errors if i.code == "ARTIFACT_CORRUPT")
    assert issue.record_id == "run-demo-a" and issue.details["declared"] == sha


def test_artifact_size_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["artifacts"][0].update(size_bytes=999)), blobs)
    assert "ARTIFACT_SIZE_MISMATCH" in error_codes(report)


def test_reader_exception_is_missing_evidence(bundle_dicts: list[dict]) -> None:
    def broken(sha: str) -> bytes | None:
        raise OSError("disk on fire")

    records = [Record.from_dict(d) for d in clone(bundle_dicts)]
    report = validate_records(records, artifact_reader=broken)
    issue = next(i for i in report.errors if i.code == "MISSING_EVIDENCE")
    assert "disk on fire" in issue.details["reader_error"]


def test_evidence_unavailable_tombstone_is_warning(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["artifacts"][0].update(availability="expired")), blobs)
    assert report.ok
    assert "EVIDENCE_UNAVAILABLE" in warning_codes(report)


def test_unknown_unit(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    dicts = clone(bundle_dicts)
    run = next(d for d in dicts if d["record_id"] == "run-demo-a")
    swap_samples_artifact(run, blobs, json.dumps({"samples": [90, 91, 90, 89, 90], "unit": "cycles"}).encode())
    assert "UNKNOWN_UNIT" in error_codes(validate(dicts, blobs))


@pytest.mark.parametrize("samples", [[90, 91, -1, 89, 90], [90, 91, 0, 89, 90], [90, "91", 90, 89, 90], []])
def test_invalid_samples(bundle_dicts: list[dict], blobs: dict[str, bytes], samples: list) -> None:
    dicts = clone(bundle_dicts)
    run = next(d for d in dicts if d["record_id"] == "run-demo-a")
    swap_samples_artifact(run, blobs, json.dumps({"samples": samples, "unit": "microseconds"}).encode())
    assert "INVALID_SAMPLES" in error_codes(validate(dicts, blobs))


@pytest.mark.parametrize("content", [b"not json", b'{"unit": "microseconds"}', b"[90, 91]", b'{"samples": [1, NaN], "unit": "microseconds"}'])
def test_invalid_samples_artifact(bundle_dicts: list[dict], blobs: dict[str, bytes], content: bytes) -> None:
    dicts = clone(bundle_dicts)
    run = next(d for d in dicts if d["record_id"] == "run-demo-a")
    swap_samples_artifact(run, blobs, content)
    assert "INVALID_SAMPLES_ARTIFACT" in error_codes(validate(dicts, blobs))


def test_sample_count_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(sample_count=4)), blobs)
    fields = {i.field for i in report.errors if i.code == "SAMPLE_COUNT_MISMATCH"}
    assert fields == {"timing.sample_count"}
    assert sum(1 for i in report.errors if i.code == "SAMPLE_COUNT_MISMATCH") == 2  # vs repetitions and vs the samples file


def test_unit_conversion_is_applied(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    dicts = clone(bundle_dicts)
    run = next(d for d in dicts if d["record_id"] == "run-demo-a")
    swap_samples_artifact(run, blobs, json.dumps({"samples": [0.090, 0.091, 0.090, 0.089, 0.090], "unit": "milliseconds"}).encode())
    report = validate(dicts, blobs)
    assert issues_for(report, "run-demo-a") == []  # 0.09 ms == 90 us


def test_t20_summary_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(median_us=1)), blobs)
    issue = next(i for i in report.errors if i.code == "SUMMARY_MISMATCH")
    assert issue.details["computed_median_us"] == 90.0 and issue.details["computed_p90_us"] == pytest.approx(90.6)
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["timing"].update(p90_us=91)), blobs)
    assert "SUMMARY_MISMATCH" in error_codes(report)


def test_timing_not_run_with_results(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-c-failure", lambda d: d["payload"]["timing"].update(sample_count=5)), blobs)
    assert "TIMING_NOT_RUN_WITH_RESULTS" in error_codes(report)
    report = validate(mutated(bundle_dicts, "run-demo-c-failure", lambda d: d["payload"]["timing"].update(status="error", median_us=1.0)), blobs)
    assert "TIMING_NOT_RUN_WITH_RESULTS" in error_codes(report)


# --------------------------------------------------------------------------------------
# run: metrics and provenance
# --------------------------------------------------------------------------------------
def test_t16_spill_observed_on_succeeded_run_is_clean(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    run = record_dict(bundle_dicts, "run-demo-c")
    assert run["payload"]["execution_status"] == "succeeded"
    assert run["payload"]["analysis_metrics"][0]["status"] == "observed" and run["payload"]["analysis_metrics"][0]["value"] == 65536
    report = validate(bundle_dicts, blobs)
    assert issues_for(report, "run-demo-c") == []


@pytest.mark.parametrize("status", ["not_collected", "unsupported", "parse_error", "not_applicable"])
def test_t17_uncollected_metric_with_zero_rejected(bundle_dicts: list[dict], blobs: dict[str, bytes], status: str) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-a", lambda d: d["payload"]["analysis_metrics"][0].update(status=status, value=0)), blobs)
    issue = next(i for i in report.errors if i.code == "UNKNOWN_METRIC_NOT_NULL")
    assert issue.details == {"metric": "register_spill_vmem_static_bytes", "status": status, "value": 0}


@pytest.mark.parametrize("change", [{"source_artifact_ref": None}, {"value": None}])
def test_observed_metric_without_evidence(bundle_dicts: list[dict], blobs: dict[str, bytes], change: dict) -> None:
    report = validate(mutated(bundle_dicts, "run-demo-c", lambda d: d["payload"]["analysis_metrics"][0].update(change)), blobs)
    assert "OBSERVED_METRIC_WITHOUT_EVIDENCE" in error_codes(report)


def test_observed_metric_evidence_must_be_readable(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    spill = record_dict(bundle_dicts, "run-demo-c")["payload"]["artifacts"][2]["sha256"]
    partial = {k: v for k, v in blobs.items() if k != spill}
    report = validate(bundle_dicts, partial)
    assert any(i.code == "MISSING_EVIDENCE" and i.details["artifact_id"] == "mock-spill-report" for i in report.errors)


def test_fixture_run_warning(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate([record_dict(bundle_dicts, r) for r in ("kernel-demo", "cfg-demo", "baseline-demo", "run-demo-baseline")], blobs)
    assert report.ok
    assert [i.record_id for i in report.warnings if i.code == "FIXTURE_RUN"] == ["run-demo-baseline"]


# --------------------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------------------
def test_policy_hash_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"]["policy"].update(min_confirm_pairs=1)), blobs)
    issue = next(i for i in report.errors if i.code == "POLICY_HASH_MISMATCH")
    assert issue.details["recorded"] != issue.details["computed"]


def test_decision_group_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"].update(comparison_key=ZERO_HASH)), blobs)
    assert sum(1 for i in report.errors if i.code == "DECISION_GROUP_MISMATCH") == 2


def test_candidate_subject_mismatch(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"].update(candidate_subject_ref="commit-demo-c")), blobs)
    assert "CANDIDATE_SUBJECT_MISMATCH" in error_codes(report)


def test_t19_fixture_production_decision_rejected(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(
        mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"].update(is_production=True, outcome="accepted", reason_codes=[])), blobs
    )
    issue = next(i for i in report.errors if i.code == "FIXTURE_PRODUCTION_DECISION")
    assert issue.details["untrusted_runs"] == ["run-demo-a", "run-demo-baseline"]
    assert {"FIXTURE_DECISION_ACCEPTED", "UNVERIFIED_ACCEPTED"} <= error_codes(report)


def test_production_requires_acceptance(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"].update(is_production=True)), blobs)
    assert {"PRODUCTION_WITHOUT_ACCEPTANCE", "FIXTURE_PRODUCTION_DECISION"} <= error_codes(report)


def test_accepted_without_candidate_runs(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(
        mutated(bundle_dicts, "decision-demo-blocked", lambda d: d["payload"].update(outcome="accepted", candidate_run_refs=[], baseline_run_refs=[])), blobs
    )
    assert "ACCEPTED_WITHOUT_EVIDENCE" in error_codes(report)


# --------------------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------------------
def test_find_cycle_helper() -> None:
    assert find_cycle({"a": {"b"}, "b": {"c"}, "c": set()}) is None
    cycle = find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}})
    assert cycle is not None and cycle[0] == cycle[-1] and set(cycle) == {"a", "b", "c"}
    assert find_cycle({"x": {"x"}}) == ["x", "x"]


def test_t06_origin_cycle_via_relations(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    back = record_dict(bundle_dicts, "relation-demo-origin")
    back["record_id"] = "relation-demo-back"
    back["payload"].update(from_ref="commit-demo-c", to_ref="commit-demo-a")
    report = validate(clone(bundle_dicts) + [back], blobs)
    issue = next(i for i in report.errors if i.code == "ORIGIN_CYCLE")
    assert {"commit-demo-a", "commit-demo-c"} <= set(issue.details["cycle"])


def test_t06_origin_cycle_via_pr_origin_refs(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    # PR 102 branches from commit A (of PR 101); making PR 101 branch from commit C (of PR 102) closes a loop.
    report = validate(mutated(bundle_dicts, "pr-demo-101", lambda d: d["payload"].update(origin_ref="commit-demo-c")), blobs)
    issue = next(i for i in report.errors if i.code == "ORIGIN_CYCLE")
    assert {"pr-demo-101", "pr-demo-102", "commit-demo-a", "commit-demo-c"} <= set(issue.details["cycle"])


def test_git_parent_cycle(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    # commit B's parent is A (..02); making A's parent B (..03) creates 02 -> 03 -> 02.
    report = validate(mutated(bundle_dicts, "commit-demo-a", lambda d: d["payload"].update(git_parent_oids=[OID(3)])), blobs)
    issue = next(i for i in report.errors if i.code == "GIT_PARENT_CYCLE")
    assert issue.details["repo_uid"] == "github:github.com:repo:900001"
    assert {"sha1:" + f"{2:040x}", "sha1:" + f"{3:040x}"} <= set(issue.details["oids"])


def test_parent_oids_inconsistent_across_bindings(bundle_dicts: list[dict], blobs: dict[str, bytes]) -> None:
    report = validate(mutated(bundle_dicts, "commit-demo-a-in-102", lambda d: d["payload"].update(git_parent_oids=[])), blobs)
    issue = next(i for i in report.errors if i.code == "PARENT_OIDS_INCONSISTENT")
    assert issue.details["other_record"] == "commit-demo-a"
    assert "GIT_PARENT_CYCLE" not in error_codes(report)


# --------------------------------------------------------------------------------------
# deep_validate over a store: CAS and file tampering
# --------------------------------------------------------------------------------------
def test_t27_deep_validate_detects_tampered_blob(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    sha = samples_sha(bundle_dicts, "run-demo-a")
    path = demo_store.artifact_path(sha)
    path.write_bytes(path.read_bytes() + b" ")
    report = deep_validate(demo_store)
    assert not report.ok
    assert {"ARTIFACT_CORRUPT", "STORE_ARTIFACT_CORRUPT"} <= error_codes(report)
    assert any(i.code == "ARTIFACT_CORRUPT" and i.record_id == "run-demo-a" for i in report.errors)


def test_t27_deep_validate_detects_missing_blob(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    sha = samples_sha(bundle_dicts, "run-demo-c")
    demo_store.artifact_path(sha).unlink()
    report = deep_validate(demo_store)
    assert {"MISSING_EVIDENCE", "STORE_ARTIFACT_MISSING"} <= error_codes(report)
    assert any(i.code == "MISSING_EVIDENCE" and i.record_id == "run-demo-c" for i in report.errors)
    relaxed = deep_validate(demo_store, verify_artifacts=False)
    assert relaxed.ok and relaxed.artifact_checks == 0 and relaxed.summary_checks == 0


def test_deep_validate_detects_modified_record_file(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("annotation-demo-grouped")
    data = json.loads(path.read_text())
    data["payload"]["text"] = "edited after publication"
    path.write_text(json.dumps(data, indent=2))
    demo_store.invalidate_index()
    report = deep_validate(demo_store)
    assert any(i.code == "STORE_RECORD_MODIFIED" and i.record_id == "annotation-demo-grouped" for i in report.errors)


def test_deep_validate_reports_corrupt_record_file_without_raising(demo_store: MemoryStore) -> None:
    path = demo_store.record_path("annotation-demo-grouped")
    path.write_text("{ not json")
    demo_store.invalidate_index()
    report = deep_validate(demo_store)
    assert not report.ok
    assert "STORE_RECORD_CORRUPT" in error_codes(report)


def test_deep_validate_detects_unjournaled_record(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    src = demo_store.record_path("annotation-demo-grouped")
    stray = json.loads(src.read_text())
    stray["record_id"] = "annotation-demo-stray"
    (src.parent / "annotation-demo-stray.json").write_text(json.dumps(stray, indent=2))
    demo_store.invalidate_index()
    report = deep_validate(demo_store)
    assert any(i.code == "STORE_RECORD_UNJOURNALED" and i.record_id == "annotation-demo-stray" for i in report.errors)


def test_issue_and_report_serialisation() -> None:
    report = ValidationReport()
    report.add(ERROR, "X", "bad", "r1", "f", a=1)
    report.add(WARNING, "Y", "meh", None)
    assert not report.ok and report.codes(ERROR) == {"X"} and report.codes() == {"X", "Y"}
    data = report.to_dict()
    assert data["issues"][0] == {"severity": "error", "code": "X", "message": "bad", "record_id": "r1", "field": "f", "details": {"a": 1}}
    assert data["issues"][1]["record_id"] is None and data["error_count"] == 1 and data["warning_count"] == 1
