"""Pairwise run comparison (specification 11.2 and 12.1, DESIGN section 6; scenarios T14 T15 T16 T18 T23).

``compare_runs`` is pure; ``compare_in_store`` adds store lookups and sample-blob recomputation. Golden
values come from the fixture bundle (medians 90 us / 100 us / 88 us). Trusted runs are produced by the
real ``LocalRunner`` through ``tests/synthetic_runs.py`` so every hash is derived as in production.
"""
from __future__ import annotations

import json
import math
from typing import Any

import pytest

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, KernelMemoryError, MissingReferenceError
from kernel_memory.services.compare import (
    ABSENT,
    COMPARABLE,
    INSUFFICIENT_EVIDENCE,
    NOT_COMPARABLE,
    NOT_ELIGIBLE,
    SnapshotDifference,
    compare_in_store,
    compare_runs,
    diff_snapshots,
    load_run_samples,
)

from conftest import record_dict
from synthetic_runs import cloned_run_dict, publish_dict, reconcile_run_hashes, trusted_run

US = "microseconds"
DEMO_A_SAMPLES = [90, 91, 90, 89, 90]
DEMO_BASELINE_SAMPLES = [100, 101, 100, 99, 100]


def _fields(differences: list[SnapshotDifference]) -> set[tuple[str, str]]:
    return {(d.group, d.field) for d in differences}


def _samples_path(store, run_id: str, artifact_id: str):
    run = store.get(run_id)
    ref = run.payload.artifact_by_id()[artifact_id]
    return store.artifact_path(ref.sha256)


# --------------------------------------------------------------------------------------
# Golden: fixture candidate versus fixture baseline
# --------------------------------------------------------------------------------------
def test_golden_demo_a_vs_baseline_is_comparable(demo_store):
    result = compare_in_store(demo_store, "run-demo-a", "run-demo-baseline")
    assert result.status == COMPARABLE
    assert result.reasons == []
    assert result.identity_differences() == []
    assert result.comparison_key_candidate == result.comparison_key_baseline
    derived = result.derived
    assert derived is not None
    assert math.isclose(derived["speedup"], 100 / 90)
    assert math.isclose(result.speedup, 100 / 90)
    assert math.isclose(derived["latency_reduction_pct"], 10.0)
    assert derived["candidate_median_us"] == 90.0
    assert derived["baseline_median_us"] == 100.0
    assert math.isclose(derived["candidate_p90_us"], 90.6)
    assert math.isclose(derived["baseline_p90_us"], 100.6)
    assert derived["candidate_sample_count"] == 5 and derived["baseline_sample_count"] == 5
    assert derived["recomputed_from_samples"] is True
    assert derived["candidate_normalized_iqr"] is not None and derived["baseline_normalized_iqr"] is not None
    assert derived["unit"] == US
    # Fixture provenance never confirms anything, however good the numbers look.
    assert result.confirmation_eligible is False
    assert "FIXTURE_NOT_ELIGIBLE" in result.confirmation_blockers
    assert NOT_COMPARABLE not in result.confirmation_blockers


def test_golden_result_to_dict_is_json_and_carries_speedup(demo_store):
    result = compare_in_store(demo_store, "run-demo-a", "run-demo-baseline")
    data = result.to_dict()
    json.dumps(data)  # serialisable for the CLI
    assert data["status"] == COMPARABLE
    assert data["candidate_run_ref"] == "run-demo-a" and data["baseline_run_ref"] == "run-demo-baseline"
    assert math.isclose(data["speedup"], 100 / 90)
    assert data["confirmation_eligible"] is False
    assert "FIXTURE_NOT_ELIGIBLE" in data["confirmation_blockers"]
    assert {d["group"] for d in data["differences"]} <= {"variant"}


def test_golden_fixture_samples_load_from_cas(demo_store):
    samples, warnings = load_run_samples(demo_store, demo_store.get("run-demo-a"), side="candidate")
    assert warnings == []
    assert samples == (DEMO_A_SAMPLES, US)


# --------------------------------------------------------------------------------------
# T18: identity field changes make runs NOT_COMPARABLE (field-level diff names group and field)
# --------------------------------------------------------------------------------------
T18_CASES = [
    ("environment__accelerator_model", "TPU-SYNTHETIC-V9", "environment", "accelerator_model"),
    ("protocol__warmup", 3, "protocol", "warmup"),
    ("protocol__measurement_scope", "kernel_only", "protocol", "measurement_scope"),
    ("verifier__tolerances", {"atol": "1e-6", "rtol": "0"}, "verifier", "tolerances.atol"),
    ("source__checkout_mode", "integration_merge", "checkout_mode", "source.checkout_mode"),
]


@pytest.mark.parametrize("override, value, group, dotted", T18_CASES, ids=[c[0] for c in T18_CASES])
def test_t18_identity_change_is_not_comparable(demo_store, bundle_dicts, override, value, group, dotted):
    original = demo_store.get("run-demo-a")
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id=f"run-demo-a-{group}", **{override: value})
    clone = publish_dict(demo_store, data)
    # The clone is self-consistent (hashes recomputed) and lands in another comparability group.
    assert clone.payload.comparison_key != original.payload.comparison_key
    result = compare_in_store(demo_store, clone.record_id, "run-demo-baseline")
    assert result.status == NOT_COMPARABLE
    assert result.derived is None and result.speedup is None
    assert result.confirmation_eligible is False
    assert result.confirmation_blockers == [NOT_COMPARABLE]
    assert (group, dotted) in _fields(result.identity_differences())
    assert result.reasons == [f"{NOT_COMPARABLE}({group})"]
    assert not any(w.endswith("_INCONSISTENT") or w.startswith("RECORDED_COMPARISON_KEY_MISMATCH") for w in result.warnings)
    dumped = result.to_dict()
    assert any(d["group"] == group and d["field"] == dotted for d in dumped["differences"])


def test_t18_difference_values_name_candidate_and_baseline(demo_store, bundle_dicts):
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-env2", environment__accelerator_model="OTHER-ACCEL")
    clone = publish_dict(demo_store, data)
    result = compare_in_store(demo_store, clone.record_id, "run-demo-baseline")
    diff = next(d for d in result.identity_differences() if d.field == "accelerator_model")
    assert diff.group == "environment"
    assert diff.candidate == "OTHER-ACCEL"
    assert diff.baseline == "MOCK-NO-HARDWARE"
    assert diff.to_dict() == {"group": "environment", "field": "accelerator_model", "candidate": "OTHER-ACCEL", "baseline": "MOCK-NO-HARDWARE"}


def test_t18_reconciled_clone_hashes_follow_domain_hashing(bundle_dicts):
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-check", protocol__warmup=7)
    payload = data["payload"]
    assert payload["protocol"]["protocol_hash"] == hashing.protocol_hash(payload["protocol"])
    assert payload["environment"]["environment_hash"] == hashing.environment_hash(payload["environment"])
    assert payload["verifier"]["verifier_hash"] == hashing.verifier_hash(payload["verifier"])
    cfg = record_dict(bundle_dicts, "cfg-demo")["payload"]["config_hash"]
    assert payload["comparison_key"] == hashing.comparison_key(
        config_hash=cfg,
        environment_hash=payload["environment"]["environment_hash"],
        protocol_hash=payload["protocol"]["protocol_hash"],
        verifier_hash=payload["verifier"]["verifier_hash"],
        checkout_mode=payload["source"]["checkout_mode"],
    )
    fixture = record_dict(bundle_dicts, "run-demo-a")["payload"]
    assert payload["protocol"]["protocol_hash"] != fixture["protocol"]["protocol_hash"]
    assert payload["comparison_key"] != fixture["comparison_key"]


def test_t18_unreconciled_clone_is_flagged_inconsistent(bundle_dicts):
    """A record whose snapshot changed but whose hashes did not is still NOT_COMPARABLE, with warnings."""
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-stale", reconcile=False, environment__accelerator_model="X")
    result = compare_runs(data, record_dict(bundle_dicts, "run-demo-baseline"))
    assert result.status == NOT_COMPARABLE
    assert ("environment", "accelerator_model") in _fields(result.identity_differences())
    assert "ENVIRONMENT_HASH_INCONSISTENT" in result.warnings
    assert "COMPARISON_KEY_INCONSISTENT" in result.warnings


def test_t18_manual_reconciliation_matches_helper(bundle_dicts):
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-demo-a-manual"
    data["payload"]["environment"]["accelerator_model"] = "MANUAL"
    cfg = record_dict(bundle_dicts, "cfg-demo")["payload"]["config_hash"]
    reconcile_run_hashes(data, config_hash=cfg)
    helper = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-manual", environment__accelerator_model="MANUAL")
    assert data["payload"]["comparison_key"] == helper["payload"]["comparison_key"]
    assert data["payload"]["environment"]["environment_hash"] == helper["payload"]["environment"]["environment_hash"]


TRUSTED_CASES = [
    ({"environment_overrides": {"accelerator_model": "OTHER-ACCEL"}}, "environment", "accelerator_model"),
    ({"protocol": {"warmup": 4}}, "protocol", "warmup"),
    ({"protocol": {"measurement_scope": "fixture_only"}}, "protocol", "measurement_scope"),
    ({"verifier": {"tolerances": {"atol": "1e-6", "rtol": "0"}}}, "verifier", "tolerances.atol"),
    ({"checkout_mode": "integration_merge"}, "checkout_mode", "source.checkout_mode"),
]


@pytest.mark.parametrize("kwargs, group, dotted", TRUSTED_CASES, ids=[c[1] + ":" + c[2] for c in TRUSTED_CASES])
def test_t18_trusted_runs_with_different_identity_are_not_comparable(demo_store, kwargs, group, dotted):
    cand = trusted_run(demo_store, request_id="t18-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", **kwargs)
    base = trusted_run(demo_store, request_id="t18-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline")
    assert cand.payload.comparison_key != base.payload.comparison_key
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == NOT_COMPARABLE
    assert (group, dotted) in _fields(result.identity_differences())
    assert result.reasons == [f"{NOT_COMPARABLE}({group})"]
    assert result.derived is None


def test_t18_negative_same_identity_other_commit_is_comparable(demo_store):
    """Different subjects in one comparability group compare; only variant fields differ (informational)."""
    result = compare_in_store(demo_store, "run-demo-c", "run-demo-a")
    assert result.status == COMPARABLE
    assert result.identity_differences() == []
    variant_fields = {d.field for d in result.variant_differences()}
    assert "source.variant_digest" in variant_fields and "source.source_digest" in variant_fields
    assert math.isclose(result.speedup, 90 / 88)


def test_t18_different_config_is_not_comparable(demo_store, bundle_dicts):
    """A config hash difference is an identity difference of group ``config`` even when snapshots match."""
    cand = record_dict(bundle_dicts, "run-demo-a")
    base = record_dict(bundle_dicts, "run-demo-baseline")
    cfg = record_dict(bundle_dicts, "cfg-demo")["payload"]["config_hash"]
    other = hashing.sha256_bytes(b"another-config")
    result = compare_runs(cand, base, candidate_config_hash=other, baseline_config_hash=cfg)
    assert result.status == NOT_COMPARABLE
    assert ("config", "config_hash") in _fields(result.identity_differences())
    assert "RECORDED_COMPARISON_KEY_MISMATCH(candidate)" in result.warnings


# --------------------------------------------------------------------------------------
# T23: tolerance change -> verifier_hash change -> NOT_COMPARABLE
# --------------------------------------------------------------------------------------
def test_t23_tolerance_change_changes_verifier_hash(demo_store, bundle_dicts):
    original = demo_store.get("run-demo-a")
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-tol", verifier__tolerances={"atol": "1e-6", "rtol": "0"})
    clone = publish_dict(demo_store, data)
    assert clone.payload.verifier.verifier_hash != original.payload.verifier.verifier_hash
    assert clone.payload.protocol.protocol_hash == original.payload.protocol.protocol_hash
    assert clone.payload.environment.environment_hash == original.payload.environment.environment_hash
    result = compare_in_store(demo_store, clone.record_id, "run-demo-baseline")
    assert result.status == NOT_COMPARABLE
    assert result.reasons == [f"{NOT_COMPARABLE}(verifier)"]
    diff = next(d for d in result.identity_differences() if d.field == "tolerances.atol")
    assert (diff.group, diff.candidate, diff.baseline) == ("verifier", "1e-6", "0")
    assert ("verifier", "tolerances.rtol") not in _fields(result.differences)


def test_t23_trusted_verifier_tolerance_change_is_not_comparable(demo_store):
    strict = trusted_run(demo_store, request_id="t23-strict", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline")
    loose = trusted_run(
        demo_store, request_id="t23-loose", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate",
        verifier={"tolerances": {"atol": "1e-3", "rtol": "1e-3"}},
    )
    assert loose.payload.verifier.verifier_hash != strict.payload.verifier.verifier_hash
    result = compare_in_store(demo_store, loose.record_id, strict.record_id)
    assert result.status == NOT_COMPARABLE
    assert {("verifier", "tolerances.atol"), ("verifier", "tolerances.rtol")} <= _fields(result.identity_differences())


def test_t23_negative_unchanged_tolerances_keep_verifier_hash(demo_store, bundle_dicts):
    original = demo_store.get("run-demo-a")
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-same-tol", verifier__tolerances={"atol": "0", "rtol": "0"})
    clone = publish_dict(demo_store, data)
    assert clone.payload.verifier.verifier_hash == original.payload.verifier.verifier_hash
    assert clone.payload.comparison_key == original.payload.comparison_key
    result = compare_in_store(demo_store, clone.record_id, "run-demo-baseline")
    assert result.status == COMPARABLE
    assert result.identity_differences() == []


# --------------------------------------------------------------------------------------
# T14: failed execution -> NOT_ELIGIBLE
# --------------------------------------------------------------------------------------
def test_t14_failed_candidate_is_not_eligible(demo_store):
    result = compare_in_store(demo_store, "run-demo-c-failure", "run-demo-baseline")
    assert result.status == NOT_ELIGIBLE
    assert "EXECUTION_NOT_SUCCEEDED(candidate)" in result.reasons
    assert "CORRECTNESS_NOT_PASSED(candidate)" in result.reasons  # correctness not_run is not a pass
    assert result.derived is None and result.speedup is None
    assert result.confirmation_eligible is False
    assert result.confirmation_blockers == [NOT_ELIGIBLE]
    assert result.identity_differences() == []  # same comparability group; eligibility is the failing gate


def test_t14_failed_baseline_side_is_named(demo_store):
    result = compare_in_store(demo_store, "run-demo-a", "run-demo-c-failure")
    assert result.status == NOT_ELIGIBLE
    assert "EXECUTION_NOT_SUCCEEDED(baseline)" in result.reasons
    assert not any(r.endswith("(candidate)") for r in result.reasons)


def test_t14_comparability_is_checked_before_eligibility(bundle_dicts):
    data = cloned_run_dict(bundle_dicts, "run-demo-c-failure", new_id="run-demo-c-failure-env", environment__accelerator_model="X")
    result = compare_runs(data, record_dict(bundle_dicts, "run-demo-baseline"))
    assert result.status == NOT_COMPARABLE
    assert not any(r.startswith("EXECUTION_NOT_SUCCEEDED") for r in result.reasons)


def test_t14_trusted_compile_error_is_not_eligible(demo_store):
    cand = trusted_run(demo_store, request_id="t14-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", compile_status="compile_error")
    base = trusted_run(demo_store, request_id="t14-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline")
    assert cand.payload.execution_status != "succeeded"
    assert cand.payload.timing.status != "recorded"
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == NOT_ELIGIBLE
    assert "EXECUTION_NOT_SUCCEEDED(candidate)" in result.reasons
    assert result.derived is None


# --------------------------------------------------------------------------------------
# T15: correctness failure -> NOT_ELIGIBLE (timing evidence is never consulted)
# --------------------------------------------------------------------------------------
def test_t15_correctness_fail_is_not_eligible(demo_store):
    cand = trusted_run(demo_store, request_id="t15-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", correctness="fail", samples=(1.0,) * 5)
    base = trusted_run(demo_store, request_id="t15-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline")
    assert cand.payload.execution_status == "succeeded"
    assert cand.payload.correctness.status == "fail"
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == NOT_ELIGIBLE
    assert result.reasons == ["CORRECTNESS_NOT_PASSED(candidate)"]
    assert result.derived is None  # a 10x "speedup" of a wrong kernel is never derived
    assert result.confirmation_eligible is False
    assert result.confirmation_blockers == [NOT_ELIGIBLE]


def test_t15_negative_trusted_pass_is_confirmation_eligible(demo_store):
    cand = trusted_run(demo_store, request_id="t15-ok-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="t15-ok-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline", samples=(10.0,) * 5)
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert result.confirmation_blockers == []
    assert result.confirmation_eligible is True
    assert result.derived["recomputed_from_samples"] is True
    assert math.isclose(result.speedup, 10 / 9)
    assert result.derived["candidate_normalized_iqr"] == 0.0


# --------------------------------------------------------------------------------------
# T16: an observed resource metric never affects comparability
# --------------------------------------------------------------------------------------
def test_t16_spill_observed_run_is_comparable(demo_store):
    run = demo_store.get("run-demo-c")
    spill = next(m for m in run.payload.analysis_metrics if m.name == "register_spill_vmem_static_bytes")
    assert spill.status == "observed" and spill.value == 65536
    result = compare_in_store(demo_store, "run-demo-c", "run-demo-baseline")
    assert result.status == COMPARABLE
    assert result.identity_differences() == []
    assert math.isclose(result.speedup, 100 / 88)
    assert result.derived["recomputed_from_samples"] is True
    assert result.confirmation_blockers == ["FIXTURE_NOT_ELIGIBLE"]


# --------------------------------------------------------------------------------------
# Evidence degradation: missing / corrupt sample blobs
# --------------------------------------------------------------------------------------
def test_missing_samples_blob_falls_back_to_recorded_summaries(demo_store):
    _samples_path(demo_store, "run-demo-a", "run-demo-a-samples").unlink()
    result = compare_in_store(demo_store, "run-demo-a", "run-demo-baseline")
    assert result.status == COMPARABLE
    assert "MISSING_SAMPLES(candidate)" in result.warnings
    assert "DERIVED_FROM_RECORDED_SUMMARIES(candidate)" in result.warnings
    assert "DERIVED_FROM_RECORDED_SUMMARIES(baseline)" not in result.warnings
    assert "MISSING_SAMPLES" in result.confirmation_blockers
    assert result.confirmation_eligible is False
    derived = result.derived
    assert derived["recomputed_from_samples"] is False
    assert math.isclose(derived["speedup"], 100 / 90)
    assert derived["candidate_normalized_iqr"] is None
    assert derived["baseline_normalized_iqr"] is not None


def test_missing_samples_blob_blocks_a_trusted_pair(demo_store):
    cand = trusted_run(demo_store, request_id="miss-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="miss-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline", samples=(10.0,) * 5)
    assert compare_in_store(demo_store, cand.record_id, base.record_id).confirmation_eligible is True
    ref = cand.payload.artifact_by_id()[cand.payload.timing.samples_artifact_ref]
    demo_store.artifact_path(ref.sha256).unlink()
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert result.confirmation_eligible is False
    assert result.confirmation_blockers == ["MISSING_SAMPLES"]


def test_corrupt_samples_blob_warns_and_degrades(demo_store):
    path = _samples_path(demo_store, "run-demo-baseline", "run-demo-baseline-samples")
    path.write_bytes(b'{"samples": [1, 2, 3], "unit": "microseconds"}')
    result = compare_in_store(demo_store, "run-demo-a", "run-demo-baseline")
    assert result.status == COMPARABLE
    assert "CORRUPT_SAMPLES(baseline)" in result.warnings
    assert "MISSING_SAMPLES" in result.confirmation_blockers
    assert result.derived["recomputed_from_samples"] is False
    assert result.derived["baseline_median_us"] == 100.0  # tampered bytes are never used


def test_load_run_samples_without_reference(demo_store):
    samples, warnings = load_run_samples(demo_store, demo_store.get("run-demo-c-failure"), side="candidate")
    assert samples is None
    assert warnings == []  # timing not recorded: nothing to load, nothing to warn about


# --------------------------------------------------------------------------------------
# compare_runs with explicit samples (pure)
# --------------------------------------------------------------------------------------
def test_compare_runs_recomputes_from_explicit_samples(demo_store):
    result = compare_runs(
        demo_store.get("run-demo-a"),
        demo_store.get("run-demo-baseline"),
        candidate_samples=(DEMO_A_SAMPLES, US),
        baseline_samples=(DEMO_BASELINE_SAMPLES, US),
    )
    assert result.status == COMPARABLE
    assert result.derived["recomputed_from_samples"] is True
    assert math.isclose(result.speedup, 100 / 90)
    assert math.isclose(result.derived["candidate_p90_us"], 90.6)
    assert result.confirmation_blockers == ["FIXTURE_NOT_ELIGIBLE"]


def test_compare_runs_detects_summary_mismatch(demo_store):
    result = compare_runs(
        demo_store.get("run-demo-a"),
        demo_store.get("run-demo-baseline"),
        candidate_samples=([80, 81, 80, 79, 80], US),  # recorded median is 90
        baseline_samples=(DEMO_BASELINE_SAMPLES, US),
    )
    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.reasons == ["SUMMARY_MISMATCH(candidate)"]
    assert result.derived is None
    assert result.confirmation_eligible is False
    assert result.confirmation_blockers == [INSUFFICIENT_EVIDENCE]


def test_compare_runs_detects_sample_count_mismatch(demo_store):
    result = compare_runs(
        demo_store.get("run-demo-a"),
        demo_store.get("run-demo-baseline"),
        candidate_samples=(DEMO_A_SAMPLES + [90], US),  # recorded sample_count is 5
        baseline_samples=(DEMO_BASELINE_SAMPLES, US),
    )
    assert result.status == INSUFFICIENT_EVIDENCE
    assert "SAMPLE_COUNT_MISMATCH(candidate)" in result.reasons
    assert result.derived is None


def test_compare_runs_baseline_mismatch_is_named_by_side(demo_store):
    result = compare_runs(
        demo_store.get("run-demo-a"),
        demo_store.get("run-demo-baseline"),
        candidate_samples=(DEMO_A_SAMPLES, US),
        baseline_samples=([200, 201, 200, 199, 200], US),
    )
    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.reasons == ["SUMMARY_MISMATCH(baseline)"]


def test_compare_runs_without_samples_uses_recorded_summaries(demo_store):
    result = compare_runs(demo_store.get("run-demo-a"), demo_store.get("run-demo-baseline"))
    assert result.status == COMPARABLE
    assert result.derived["recomputed_from_samples"] is False
    assert math.isclose(result.speedup, 100 / 90)
    assert {"DERIVED_FROM_RECORDED_SUMMARIES(candidate)", "DERIVED_FROM_RECORDED_SUMMARIES(baseline)"} <= set(result.warnings)
    assert "MISSING_SAMPLES" in result.confirmation_blockers
    assert result.derived["candidate_normalized_iqr"] is None


def test_compare_runs_rejects_invalid_sample_tuples(demo_store):
    result = compare_runs(
        demo_store.get("run-demo-a"),
        demo_store.get("run-demo-baseline"),
        candidate_samples=([], US),
        baseline_samples=(DEMO_BASELINE_SAMPLES, "furlongs"),
    )
    assert result.status == INSUFFICIENT_EVIDENCE
    assert len(result.reasons) == 2
    assert result.reasons[0].endswith("(candidate)") and result.reasons[1].endswith("(baseline)")
    assert result.derived is None


def test_compare_runs_timing_not_recorded_is_insufficient_evidence(bundle_dicts):
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-notiming")
    data["payload"]["timing"] = {"status": "not_run", "sample_count": 0, "median_us": None, "p90_us": None, "samples_artifact_ref": None}
    result = compare_runs(data, record_dict(bundle_dicts, "run-demo-baseline"))
    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.reasons == ["TIMING_NOT_RECORDED(candidate)"]
    assert result.derived is None


def test_compare_runs_accepts_dicts_and_payloads(bundle_dicts):
    cand = record_dict(bundle_dicts, "run-demo-a")
    base = record_dict(bundle_dicts, "run-demo-baseline")
    by_record = compare_runs(cand, base)
    assert by_record.status == COMPARABLE
    assert (by_record.candidate_run_ref, by_record.baseline_run_ref) == ("run-demo-a", "run-demo-baseline")
    by_payload = compare_runs(cand["payload"], base["payload"])
    assert by_payload.status == COMPARABLE
    assert (by_payload.candidate_run_ref, by_payload.baseline_run_ref) == ("<candidate>", "<baseline>")
    assert math.isclose(by_payload.speedup, 100 / 90)


def test_compare_runs_rejects_non_run_inputs(demo_store):
    baseline = demo_store.get("run-demo-baseline")
    with pytest.raises(InputError) as info:
        compare_runs(demo_store.get("cfg-demo"), baseline)
    assert info.value.code == "INVALID_RUN" and info.value.exit_code == 2
    with pytest.raises(InputError):
        compare_runs(42, baseline)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        compare_runs({"record_type": "config", "payload": {}}, baseline)


def test_compare_in_store_missing_reference_raises(demo_store):
    with pytest.raises(MissingReferenceError) as info:
        compare_in_store(demo_store, "run-does-not-exist", "run-demo-baseline")
    assert info.value.exit_code == 3
    with pytest.raises(MissingReferenceError):
        compare_in_store(demo_store, "run-demo-a", "run-does-not-exist")


def test_compare_in_store_wrong_record_type_raises(demo_store):
    with pytest.raises(KernelMemoryError) as info:
        compare_in_store(demo_store, "cfg-demo", "run-demo-baseline")
    assert info.value.exit_code in (2, 3)


def test_compare_in_store_unresolved_config_warns(demo_store, bundle_dicts):
    """A run whose config record is not a config degrades to comparison-key evidence with a warning."""
    data = cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-nocfg", reconcile=False)
    result = compare_runs(data, record_dict(bundle_dicts, "run-demo-baseline"), candidate_config_hash=None, baseline_config_hash=None)
    assert result.status == COMPARABLE  # keys still equal; config hashes unknown are not a difference
    assert result.identity_differences() == []


# --------------------------------------------------------------------------------------
# diff_snapshots
# --------------------------------------------------------------------------------------
def test_diff_snapshots_nested_dicts_yield_dotted_paths():
    a: dict[str, Any] = {"software": {"python": "3.11", "jax": "0.10"}, "device_count": 1, "extra": True, "same": {"deep": {"x": 1}}}
    b: dict[str, Any] = {"software": {"python": "3.12"}, "device_count": 1.0, "same": {"deep": {"x": 1}}}
    diffs = diff_snapshots(a, b, "environment")
    assert [(d.field, d.candidate, d.baseline) for d in diffs] == [
        ("extra", True, ABSENT),
        ("software.jax", "0.10", ABSENT),
        ("software.python", "3.11", "3.12"),
    ]
    assert all(d.group == "environment" for d in diffs)


def test_diff_snapshots_exclude_is_top_level_only_and_prefix_applies():
    a = {"environment_hash": "sha256:a", "software": {"environment_hash": "x"}}
    b = {"environment_hash": "sha256:b", "software": {"environment_hash": "y"}}
    diffs = diff_snapshots(a, b, "environment", exclude=("environment_hash",))
    assert [d.field for d in diffs] == ["software.environment_hash"]
    prefixed = diff_snapshots({"tile": 128}, {"tile": 256}, "variant", prefix="source.implementation_overrides.")
    assert [(d.group, d.field, d.candidate, d.baseline) for d in prefixed] == [("variant", "source.implementation_overrides.tile", 128, 256)]


def test_diff_snapshots_json_value_semantics():
    assert diff_snapshots({"n": 1}, {"n": 1.0}, "protocol") == []
    assert [d.field for d in diff_snapshots({"flag": True}, {"flag": 1}, "protocol")] == ["flag"]
    assert [d.field for d in diff_snapshots({"n": 1}, {"n": "1"}, "protocol")] == ["n"]
    assert [d.field for d in diff_snapshots({"x": {"y": 1}}, {"x": 1}, "protocol")] == ["x"]
    assert diff_snapshots(None, None, "protocol") == []
    assert [d.field for d in diff_snapshots(None, {"k": 1}, "protocol")] == ["k"]


# --------------------------------------------------------------------------------------
# Variants of one subject stay comparable; blockers are reported, never promoted to NOT_COMPARABLE
# --------------------------------------------------------------------------------------
def test_two_variants_of_one_subject_stay_comparable_with_variant_differences(demo_store):
    tile_128 = trusted_run(demo_store, request_id="var-128", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", overrides={"tile": 128}, samples=(9.0,) * 5)
    tile_256 = trusted_run(demo_store, request_id="var-256", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="baseline", overrides={"tile": 256}, samples=(10.0,) * 5)
    assert tile_128.payload.source.variant_digest != tile_256.payload.source.variant_digest
    assert tile_128.payload.comparison_key == tile_256.payload.comparison_key
    result = compare_in_store(demo_store, tile_128.record_id, tile_256.record_id)
    assert result.status == COMPARABLE
    assert result.identity_differences() == []
    variants = {(d.field, d.candidate, d.baseline) for d in result.variant_differences()}
    assert ("source.implementation_overrides.tile", 128, 256) in variants
    assert any(f == "source.variant_digest" for f, _, _ in variants)
    assert all(d.group == "variant" for d in result.differences)
    assert math.isclose(result.speedup, 10 / 9)
    assert result.confirmation_eligible is True


def test_missing_pair_ids_block_confirmation_but_stay_comparable(demo_store):
    cand = trusted_run(demo_store, request_id="noid-cand", subject_ref="commit-demo-a", session_id=None, pair_id=None, role=None, samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="noid-base", subject_ref="baseline-demo", session_id=None, pair_id=None, role=None, samples=(10.0,) * 5)
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert result.confirmation_blockers == ["MISSING_PAIR_IDS"]
    assert result.confirmation_eligible is False


def test_differing_pair_ids_warn(demo_store):
    cand = trusted_run(demo_store, request_id="pid-cand", subject_ref="commit-demo-a", session_id="s1", pair_id="p1", role="candidate", samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="pid-base", subject_ref="baseline-demo", session_id="s2", pair_id="p2", role="baseline", samples=(10.0,) * 5)
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert "PAIR_IDS_DIFFER" in result.warnings
    assert result.confirmation_eligible is True  # a warning, not a blocker; decide counts distinct pairs


def test_dirty_source_blocks_confirmation(demo_store):
    cand = trusted_run(demo_store, request_id="dirty-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", dirty=True, samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="dirty-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline", samples=(10.0,) * 5)
    assert cand.payload.source.dirty is True
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert result.confirmation_blockers == ["DIRTY_SOURCE"]


def test_unverified_provenance_blocks_confirmation(demo_store, bundle_dicts):
    clone = publish_dict(demo_store, cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-unverified", provenance="imported_unverified"))
    result = compare_in_store(demo_store, clone.record_id, "run-demo-baseline")
    assert result.status == COMPARABLE
    assert "UNVERIFIED_PROVENANCE" in result.confirmation_blockers
    assert "FIXTURE_NOT_ELIGIBLE" in result.confirmation_blockers  # the fixture baseline
    assert result.confirmation_eligible is False


def test_unknown_required_environment_fields_block_confirmation(demo_store):
    env = {"unknown_required_fields": ["accelerator_model"]}
    cand = trusted_run(demo_store, request_id="unk-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", environment_overrides=env, samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="unk-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline", environment_overrides=env, samples=(10.0,) * 5)
    result = compare_in_store(demo_store, cand.record_id, base.record_id)
    assert result.status == COMPARABLE
    assert result.confirmation_blockers == ["UNKNOWN_ENVIRONMENT_FIELDS"]
