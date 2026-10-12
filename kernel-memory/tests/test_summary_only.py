"""Summary imports retain results without relaxing the ordinary evidence contract."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import PLACEHOLDER_ALGORITHM_ID, record_dict

from kernel_memory.domain.errors import SchemaValidationError
from kernel_memory.domain.hashing import jcs_digest
from kernel_memory.domain.models import Record
from kernel_memory.services.compare import compare_runs
from kernel_memory.services.importer import import_bundle
from kernel_memory.services.validation import deep_validate, validate_records
from kernel_memory.storage import MemoryStore


@pytest.fixture
def summary_dicts(bundle_dicts: list[dict]) -> list[dict]:
    result = [
        record_dict(bundle_dicts, record_id)
        for record_id in ("kernel-demo", PLACEHOLDER_ALGORITHM_ID, "cfg-demo", "baseline-demo", "run-demo-baseline")
    ]
    payload = result[-1]["payload"]
    payload.update(
        provenance="imported_unverified",
        evidence_mode="summary_only",
        artifacts=[],
        session_id=None,
        pair_id=None,
        role_in_pair=None,
    )
    payload["correctness"]["report_artifact_ref"] = None
    payload["timing"]["samples_artifact_ref"] = None
    payload["analysis_metrics"][0].update(status="observed", value=128, source_artifact_ref=None)
    return result


def _records(dicts: list[dict]) -> list[Record]:
    return [Record.from_dict(d) for d in dicts]


def _errors(dicts: list[dict]) -> set[str]:
    return {issue.code for issue in validate_records(_records(dicts), artifact_reader=lambda _: None).errors}


def test_legacy_wire_records_roundtrip_and_digest_are_unchanged(bundle_dicts: list[dict]) -> None:
    for original in bundle_dicts:
        record = Record.from_dict(original)
        assert record.to_dict() == original
        assert record.canonical_digest() == jcs_digest(original)
        if record.record_type == "run":
            assert record.payload.evidence_mode is None
            assert "evidence_mode" not in record.to_dict()["payload"]
            assert Record.create("run", record.record_id, record.payload, created_at=record.created_at).to_dict() == original


def test_summary_import_deep_validation_and_idempotency(
    summary_dicts: list[dict], store: MemoryStore, tmp_path: Path
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"bundle_version": "0.3.0", "records": summary_dicts}), encoding="utf-8")
    report = import_bundle(store, path, trusted_source=True)
    assert report.ok and report.records_published == 5
    assert report.artifacts_stored == report.blobs_stored == 0
    stored = store.require("run-demo-baseline", "run")
    assert stored.to_dict() == summary_dicts[-1]
    assert stored.payload.provenance == "imported_unverified"
    assert stored.payload.correctness.status == "pass"
    assert stored.payload.timing.median_us == 100
    assert stored.payload.analysis_metrics[0].value == 128
    assert store.artifact_registry() == []
    assert not any(p.is_file() for p in (store.root / "artifacts").rglob("*"))
    checked = deep_validate(store)
    assert checked.ok, [issue.to_dict() for issue in checked.errors]
    assert checked.runs_checked == 1
    assert checked.artifact_checks == checked.summary_checks == 0
    assert {issue.code for issue in checked.warnings} == {"SUMMARY_ONLY_RUN"}
    second = import_bundle(store, path)
    assert second.records_idempotent == 5 and second.records_published == 0


@pytest.mark.parametrize("mode", [None, "full", "summary", "", True])
def test_unknown_or_null_explicit_evidence_mode_is_rejected(summary_dicts: list[dict], mode: Any) -> None:
    summary_dicts[-1]["payload"]["evidence_mode"] = mode
    with pytest.raises(SchemaValidationError):
        _records(summary_dicts)


@pytest.mark.parametrize("provenance", ["trusted_worker", "fixture"])
def test_summary_cannot_claim_other_provenance(summary_dicts: list[dict], provenance: str) -> None:
    summary_dicts[-1]["payload"]["provenance"] = provenance
    with pytest.raises(SchemaValidationError):
        _records(summary_dicts)


@pytest.mark.parametrize("field", ["artifacts", "correctness", "timing", "analysis_metrics"])
def test_summary_cannot_keep_evidence_references(
    summary_dicts: list[dict], bundle_dicts: list[dict], field: str
) -> None:
    payload = summary_dicts[-1]["payload"]
    if field == "artifacts":
        payload[field] = record_dict(bundle_dicts, "run-demo-baseline")["payload"][field]
    elif field == "analysis_metrics":
        payload[field][0]["source_artifact_ref"] = "discarded-artifact"
    else:
        key = "report_artifact_ref" if field == "correctness" else "samples_artifact_ref"
        payload[field][key] = "discarded-artifact"
    with pytest.raises(SchemaValidationError):
        _records(summary_dicts)


def test_typed_summary_cannot_bypass_trust_and_attachment_checks(summary_dicts: list[dict], bundle_dicts: list[dict]) -> None:
    records = _records(summary_dicts)
    record = records[-1]
    original = Record.from_dict(record_dict(bundle_dicts, "run-demo-baseline"))
    records[-1] = dataclasses.replace(
        record,
        payload=dataclasses.replace(record.payload, provenance="trusted_worker", artifacts=original.payload.artifacts),
    )
    errors = {issue.code for issue in validate_records(records, artifact_reader=lambda _: None).errors}
    assert {"SUMMARY_ONLY_PROVENANCE", "SUMMARY_ONLY_WITH_EVIDENCE"} <= errors


def test_ordinary_import_still_requires_evidence(summary_dicts: list[dict]) -> None:
    del summary_dicts[-1]["payload"]["evidence_mode"]
    assert {
        "MISSING_CORRECTNESS_EVIDENCE",
        "MISSING_TIMING_EVIDENCE",
        "OBSERVED_METRIC_WITHOUT_EVIDENCE",
    } <= _errors(summary_dicts)


def test_summary_cannot_be_confirmed_as_a_production_improvement(summary_dicts: list[dict]) -> None:
    run = _records(summary_dicts)[-1]
    comparison = compare_runs(run, run)
    assert not comparison.confirmation_eligible
    assert {"UNVERIFIED_PROVENANCE", "MISSING_SAMPLES"} <= set(comparison.confirmation_blockers)


@pytest.mark.parametrize(
    ("field", "changes", "code"),
    [
        ("correctness", {"cases_passed": 2}, "INVALID_CORRECTNESS_COUNTS"),
        ("correctness", {"cases_passed": 0}, "INVALID_PASS"),
        ("correctness", {"status": "fail"}, "INVALID_FAIL"),
        ("correctness", {"status": "not_run"}, "CORRECTNESS_NOT_RUN_WITH_RESULTS"),
        ("timing", {"sample_count": 0}, "INVALID_SAMPLE_COUNT"),
        ("timing", {"sample_count": 4}, "SAMPLE_COUNT_MISMATCH"),
        ("timing", {"median_us": None}, "SUMMARY_MISSING"),
        ("timing", {"median_us": 0}, "INVALID_TIMING_SUMMARY"),
        ("timing", {"p90_us": 99}, "INVALID_TIMING_SUMMARY"),
        ("timing", {"status": "not_run"}, "TIMING_NOT_RUN_WITH_RESULTS"),
        ("analysis_metrics", {"value": None}, "OBSERVED_METRIC_WITHOUT_EVIDENCE"),
        ("analysis_metrics", {"status": "not_collected"}, "UNKNOWN_METRIC_NOT_NULL"),
    ],
)
def test_summary_result_invariants_remain_required(
    summary_dicts: list[dict], field: str, changes: dict, code: str
) -> None:
    payload = summary_dicts[-1]["payload"]
    target = payload[field][0] if field == "analysis_metrics" else payload[field]
    target.update(changes)
    assert code in _errors(summary_dicts)


def test_failed_execution_cannot_keep_success_numbers(summary_dicts: list[dict]) -> None:
    summary_dicts[-1]["payload"].update(execution_status="runtime_error", failure_reason="failed")
    assert "FABRICATED_RESULT" in _errors(summary_dicts)


@pytest.mark.parametrize("field", ["correctness", "timing", "analysis_metrics"])
def test_summary_unknown_result_status_is_rejected(summary_dicts: list[dict], field: str) -> None:
    payload = summary_dicts[-1]["payload"]
    target = payload[field][0] if field == "analysis_metrics" else payload[field]
    target["status"] = "probably_ok"
    with pytest.raises(SchemaValidationError):
        _records(summary_dicts)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_typed_summary_rejects_nonfinite_values(summary_dicts: list[dict], number: float) -> None:
    records = _records(summary_dicts)
    run = records[-1]
    payload = run.payload
    records[-1] = dataclasses.replace(
        run,
        payload=dataclasses.replace(
            payload,
            timing=dataclasses.replace(payload.timing, median_us=number),
            analysis_metrics=[dataclasses.replace(payload.analysis_metrics[0], value=number)],
        ),
    )
    errors = {issue.code for issue in validate_records(records, artifact_reader=lambda _: None).errors}
    assert {"INVALID_TIMING_SUMMARY", "INVALID_METRIC_VALUE"} <= errors
