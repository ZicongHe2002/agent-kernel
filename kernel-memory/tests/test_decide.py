"""Promotion decisions (specification 12.2 and 13, DESIGN sections 4 and 6; scenarios T17 T19 T21 T22).

Every decision is derived from stored evidence only. Trusted pairs come from ``tests/synthetic_runs.py``
(real ``LocalRunner`` + in-test adapter); fixture runs come from the demo bundle and are never promotable.
"""
from __future__ import annotations

import math
import re

import pytest

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, MissingReferenceError, SchemaValidationError
from kernel_memory.services.decide import (
    AMBIGUOUS_VARIANT,
    BASELINE_DRIFT,
    CANDIDATE_SUBJECT_MISMATCH,
    CORRECTNESS_NOT_PASSED,
    DIRTY_SOURCE,
    EXCESSIVE_VARIABILITY,
    EXECUTION_NOT_SUCCEEDED,
    FIXTURE_NOT_ELIGIBLE,
    INSUFFICIENT_CONFIRMATION_PAIRS,
    MISSING_EVIDENCE,
    NOT_COMPARABLE,
    PAIR_SPEEDUP_BELOW_THRESHOLD,
    RESOURCE_CONSTRAINT_UNVERIFIABLE,
    RESOURCE_CONSTRAINT_VIOLATED,
    UNVERIFIED_PROVENANCE,
    DecisionDraft,
    append_decision,
    best_known,
    decision_record_id,
    evaluate_candidate,
    supersedes_chain,
)
from kernel_memory.services.policy import default_policy, policy_hash
from kernel_memory.services.trajectory import build_trajectory
from kernel_memory.services.validation import deep_validate

from synthetic_runs import cloned_run_dict, publish_dict, trusted_pairs, trusted_run

DEMO_PAIR = [{"candidate_run": "run-demo-a", "baseline_run": "run-demo-baseline"}]
DECISION_ID = re.compile(r"^decision-commit-demo-a-[0-9a-f]{12}$")
SPILL = "register_spill_vmem_static_bytes"
T0 = "2026-09-10T00:00:00Z"
T1 = "2026-09-11T00:00:00Z"
T2 = "2026-09-12T00:00:00Z"


def spill_policy(max_value: int = 32768, **extra):
    policy = default_policy()
    policy["hard_resource_constraints"] = [{"metric": SPILL, "max_value": max_value}]
    policy.update(extra)
    return policy


def permissive_policy(**extra):
    """Fixture runs allowed, one pair suffices, trusted workers not required (never production)."""
    policy = default_policy()
    policy.update({"allow_fixture": True, "require_trusted_worker": False, "min_confirm_pairs": 1})
    policy.update(extra)
    return policy


# --------------------------------------------------------------------------------------
# Fixture reproduction (the bundle's decision-demo-blocked)
# --------------------------------------------------------------------------------------
def test_fixture_pair_reproduces_blocked_decision(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy())
    assert draft.outcome == "blocked"
    assert FIXTURE_NOT_ELIGIBLE in draft.reason_codes
    assert INSUFFICIENT_CONFIRMATION_PAIRS in draft.reason_codes
    assert draft.is_production is False
    fixture = demo_store.get("decision-demo-blocked").payload
    assert draft.reason_codes == list(fixture.reason_codes)
    assert draft.policy_hash == fixture.policy_hash
    assert draft.comparison_key == fixture.comparison_key
    assert draft.config_ref == "cfg-demo"
    assert draft.candidate_run_refs == ["run-demo-a"] and draft.baseline_run_refs == ["run-demo-baseline"]
    evaluation = draft.pair_evaluations[0]
    assert math.isclose(evaluation.pair_speedup, 100 / 90)
    assert evaluation.passed is False and FIXTURE_NOT_ELIGIBLE in evaluation.reasons


def test_fixture_decision_publishes_and_deep_validates(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy())
    record = append_decision(demo_store, draft, created_at=T0)
    assert record.record_type == "decision"
    assert DECISION_ID.match(record.record_id)
    assert record.created_at == T0
    stored = demo_store.get(record.record_id)
    assert stored is not None and stored.payload.outcome == "blocked"
    assert stored.payload.supersedes_decision_ref is None
    assert stored.payload.is_production is False
    report = deep_validate(demo_store)
    assert report.ok, [issue.__dict__ for issue in report.errors]


def test_default_policy_is_used_when_none_given(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR)
    assert draft.policy == default_policy()
    assert draft.policy_hash == policy_hash(default_policy())


def test_fixture_allowed_policy_removes_only_the_fixture_gate(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, permissive_policy(min_confirm_pairs=3))
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS]
    single = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, permissive_policy())
    assert single.outcome == "accepted"
    assert single.reason_codes == []
    assert single.is_production is False  # fixture evidence is never production


# --------------------------------------------------------------------------------------
# T17: hard resource constraints (null is never zero)
# --------------------------------------------------------------------------------------
def test_t17_null_metric_is_unverifiable_not_zero(demo_store):
    run = demo_store.get("run-demo-a")
    metric = next(m for m in run.payload.analysis_metrics if m.name == SPILL)
    assert metric.status == "not_collected" and metric.value is None
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, spill_policy())
    assert draft.outcome == "blocked"
    assert RESOURCE_CONSTRAINT_UNVERIFIABLE in draft.reason_codes
    assert RESOURCE_CONSTRAINT_VIOLATED not in draft.reason_codes
    assert any("null is not zero" in note for note in draft.notes)


def test_t17_observed_metric_over_bound_is_violated(demo_store):
    pair = [{"candidate_run": "run-demo-c", "baseline_run": "run-demo-baseline"}]
    draft = evaluate_candidate(demo_store, "commit-demo-c", pair, spill_policy())
    assert RESOURCE_CONSTRAINT_VIOLATED in draft.reason_codes
    assert RESOURCE_CONSTRAINT_UNVERIFIABLE not in draft.reason_codes
    assert draft.outcome == "blocked"  # fixture provenance still blocks
    assert any("65536" in note and "32768" in note for note in draft.notes)


def test_t17_constraint_outcomes_under_permissive_policy(demo_store):
    pair_c = [{"candidate_run": "run-demo-c", "baseline_run": "run-demo-baseline"}]
    violated = evaluate_candidate(demo_store, "commit-demo-c", pair_c, permissive_policy(hard_resource_constraints=[{"metric": SPILL, "max_value": 32768}]))
    assert (violated.outcome, violated.reason_codes) == ("rejected", [RESOURCE_CONSTRAINT_VIOLATED])
    unverifiable = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, permissive_policy(hard_resource_constraints=[{"metric": SPILL, "max_value": 32768}]))
    assert (unverifiable.outcome, unverifiable.reason_codes) == ("blocked", [RESOURCE_CONSTRAINT_UNVERIFIABLE])
    at_bound = evaluate_candidate(demo_store, "commit-demo-c", pair_c, permissive_policy(hard_resource_constraints=[{"metric": SPILL, "max_value": 65536}]))
    assert (at_bound.outcome, at_bound.reason_codes) == ("accepted", [])
    below_min = evaluate_candidate(demo_store, "commit-demo-c", pair_c, permissive_policy(hard_resource_constraints=[{"metric": SPILL, "min_value": 70000}]))
    assert (below_min.outcome, below_min.reason_codes) == ("rejected", [RESOURCE_CONSTRAINT_VIOLATED])


def test_t17_constraint_scope_and_kind_must_match(demo_store):
    pair_c = [{"candidate_run": "run-demo-c", "baseline_run": "run-demo-baseline"}]
    wrong_scope = permissive_policy(hard_resource_constraints=[{"metric": SPILL, "max_value": 32768, "scope": "whole_program"}])
    draft = evaluate_candidate(demo_store, "commit-demo-c", pair_c, wrong_scope)
    assert draft.reason_codes == [RESOURCE_CONSTRAINT_UNVERIFIABLE]
    right_scope = permissive_policy(hard_resource_constraints=[{"metric": SPILL, "max_value": 32768, "scope": "compiled_kernel", "kind": "static_estimate"}])
    draft = evaluate_candidate(demo_store, "commit-demo-c", pair_c, right_scope)
    assert draft.reason_codes == [RESOURCE_CONSTRAINT_VIOLATED]
    absent = permissive_policy(hard_resource_constraints=[{"metric": "no_such_metric", "max_value": 1}])
    draft = evaluate_candidate(demo_store, "commit-demo-c", pair_c, absent)
    assert draft.reason_codes == [RESOURCE_CONSTRAINT_UNVERIFIABLE]


def test_t17_trusted_runs_without_metrics_are_unverifiable(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="t17")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, spill_policy())
    assert draft.outcome == "blocked"
    assert draft.reason_codes == [RESOURCE_CONSTRAINT_UNVERIFIABLE]
    assert draft.is_production is False


# --------------------------------------------------------------------------------------
# T21: insufficient pairs and excessive variability are inconclusive
# --------------------------------------------------------------------------------------
def test_t21_single_trusted_pair_is_inconclusive(demo_store):
    pairs = trusted_pairs(demo_store, n=1, prefix="t21")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS]
    assert draft.is_production is False
    evaluation = draft.pair_evaluations[0]
    assert evaluation.passed is True
    assert math.isclose(evaluation.pair_speedup, 10 / 9)
    assert any("1 distinct pair(s)" in note and "requires 3" in note for note in draft.notes)


def test_t21_two_pairs_are_still_inconclusive(demo_store):
    pairs = trusted_pairs(demo_store, n=2, prefix="t21b")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS]


def test_t21_noisy_candidate_samples_are_excessive_variability(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="noisy", candidate_samples=(7.0, 8.0, 9.0, 10.0, 11.0))
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [EXCESSIVE_VARIABILITY]
    for evaluation in draft.pair_evaluations:
        assert evaluation.candidate_iqr > 0.05 and evaluation.baseline_iqr == 0.0
        assert evaluation.passed is False
    assert any(note.startswith(EXCESSIVE_VARIABILITY) and "candidate" in note for note in draft.notes)


def test_t21_noisy_baseline_samples_are_excessive_variability(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="noisyb", baseline_samples=(8.0, 9.0, 10.0, 11.0, 12.0))
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert EXCESSIVE_VARIABILITY in draft.reason_codes
    assert any(note.startswith(EXCESSIVE_VARIABILITY) and "baseline" in note for note in draft.notes)


def test_t21_negative_variability_at_threshold_is_accepted(demo_store):
    # sorted [9.8, 9.9, 10, 10.1, 10.2]: q25 = 9.9, q75 = 10.1, normalized IQR = 0.02 <= 0.05
    pairs = trusted_pairs(demo_store, n=3, prefix="calm", candidate_samples=(8.0,) * 5, baseline_samples=(10.2, 9.8, 10.0, 10.1, 9.9))
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "accepted"
    assert draft.reason_codes == []


# --------------------------------------------------------------------------------------
# T22: three clean trusted pairs are accepted as production; the negatives for each gate
# --------------------------------------------------------------------------------------
def test_t22_three_trusted_pairs_accepted_as_production(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="t22")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "accepted"
    assert draft.reason_codes == []
    assert draft.is_production is True
    assert draft.evaluated_by == "program"
    assert all(e.passed for e in draft.pair_evaluations)
    assert len({e.pair_key for e in draft.pair_evaluations}) == 3
    record = append_decision(demo_store, draft, created_at=T0)
    assert DECISION_ID.match(record.record_id)
    expected_digest = hashing.jcs_digest(
        {"policy_hash": draft.policy_hash, "run_refs": sorted(draft.candidate_run_refs + draft.baseline_run_refs), "created_at": T0}
    )
    assert record.record_id == f"decision-commit-demo-a-{expected_digest[len('sha256:'):][:12]}"
    assert record.record_id == decision_record_id("commit-demo-a", draft.policy_hash, draft.candidate_run_refs + draft.baseline_run_refs, T0)
    payload = record.payload
    assert payload.outcome == "accepted" and payload.is_production is True
    assert list(payload.reason_codes) == []
    assert payload.policy_hash == policy_hash(default_policy())
    assert deep_validate(demo_store).ok
    known = best_known(demo_store, "cfg-demo")
    assert [k["decision_ref"] for k in known] == [record.record_id]
    entry = known[0]
    assert entry["config_hash"] == demo_store.get("cfg-demo").payload.config_hash
    assert entry["comparison_key"] == draft.comparison_key
    assert entry["policy_hash"] == draft.policy_hash
    assert entry["is_production"] is True and entry["supersedes"] == []
    trajectory = build_trajectory(demo_store, "cfg-demo")
    assert record.record_id in [b["decision_ref"] for b in trajectory["best_known"]]


def test_t22_decision_id_depends_on_policy_runs_and_time():
    base = decision_record_id("commit-demo-a", "sha256:" + "0" * 64, ["run-b", "run-a"], T0)
    assert DECISION_ID.match(base)
    assert base == decision_record_id("commit-demo-a", "sha256:" + "0" * 64, ["run-a", "run-b", "run-a"], T0)  # order/duplicates ignored
    assert base != decision_record_id("commit-demo-a", "sha256:" + "0" * 64, ["run-a", "run-b"], T1)
    assert base != decision_record_id("commit-demo-a", "sha256:" + "1" * 64, ["run-a", "run-b"], T0)
    assert base != decision_record_id("commit-demo-a", "sha256:" + "0" * 64, ["run-a"], T0)


def test_t22_pair_speedup_below_threshold_is_rejected(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="slow", candidate_samples=(100.0,) * 5, baseline_samples=(101.0,) * 5)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "rejected"
    assert draft.reason_codes == [PAIR_SPEEDUP_BELOW_THRESHOLD]
    assert draft.is_production is False
    assert all(math.isclose(e.pair_speedup, 1.01) and not e.passed for e in draft.pair_evaluations)


def test_t22_one_slow_pair_rejects_the_whole_set(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="oneslow", candidate_samples=[(9.0,) * 5, (9.0,) * 5, (10.0,) * 5])
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "rejected"
    assert draft.reason_codes == [PAIR_SPEEDUP_BELOW_THRESHOLD]
    assert [e.passed for e in draft.pair_evaluations] == [True, True, False]


def test_t22_speedup_exactly_at_threshold_is_accepted(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="edge", candidate_samples=(100.0,) * 5, baseline_samples=(102.0,) * 5)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "accepted"
    assert draft.reason_codes == []


def test_t22_speedup_below_threshold_with_too_few_pairs_stays_inconclusive(demo_store):
    pairs = trusted_pairs(demo_store, n=1, prefix="slow1", candidate_samples=(100.0,) * 5, baseline_samples=(101.0,) * 5)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS, PAIR_SPEEDUP_BELOW_THRESHOLD]


def test_t22_baseline_drift_is_inconclusive(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="drift", baseline_samples=[(10.0,) * 5, (10.8,) * 5, (10.0,) * 5])
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [BASELINE_DRIFT]
    assert any(note.startswith(BASELINE_DRIFT) and "1.08" in note for note in draft.notes)


def test_t22_negative_baseline_drift_within_bound_is_accepted(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="steady", baseline_samples=[(10.0,) * 5, (10.4,) * 5, (10.0,) * 5])
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "accepted"
    assert draft.reason_codes == []


def test_t22_repeated_pair_ids_do_not_count_as_independent(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="rep", session_ids=["session-rep"] * 3, pair_ids=["pair-rep"] * 3)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS]
    assert sum(note.startswith("REPEATED_PAIR_NOT_INDEPENDENT") for note in draft.notes) == 2
    assert any("1 distinct pair(s) of 3 supplied" in note for note in draft.notes)


def test_t22_pairs_without_ids_are_not_counted(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="noids", session_ids=[None] * 3, pair_ids=[None] * 3)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "inconclusive"
    assert draft.reason_codes == [INSUFFICIENT_CONFIRMATION_PAIRS]
    assert sum(note.startswith("PAIR_IDS_MISSING") for note in draft.notes) == 3


def test_t22_ambiguous_candidate_variant_is_blocked(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="amb", candidate_overrides=[{"tile": 128}, {"tile": 256}, None])
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "blocked"
    assert draft.reason_codes == [AMBIGUOUS_VARIANT]
    assert draft.is_production is False


def test_t22_negative_consistent_variant_is_accepted(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="onevar", candidate_overrides=[{"tile": 128}] * 3)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "accepted"
    assert draft.is_production is True
    assert all(e.comparison.variant_differences() for e in draft.pair_evaluations)


def test_t22_imported_unverified_run_is_blocked(demo_store, bundle_dicts):
    clone = publish_dict(demo_store, cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-demo-a-unverified", provenance="imported_unverified"))
    assert clone.payload.provenance == "imported_unverified"
    pair = [{"candidate_run": clone.record_id, "baseline_run": "run-demo-baseline"}]
    draft = evaluate_candidate(demo_store, "commit-demo-a", pair, default_policy())
    assert draft.outcome == "blocked"
    assert UNVERIFIED_PROVENANCE in draft.reason_codes
    assert draft.is_production is False
    relaxed = evaluate_candidate(demo_store, "commit-demo-a", pair, permissive_policy())
    assert UNVERIFIED_PROVENANCE not in relaxed.reason_codes
    assert relaxed.outcome == "accepted" and relaxed.is_production is False


def test_t22_dirty_source_is_blocked(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="dirty", candidate_kwargs={"dirty": True})
    assert all(demo_store.get(p["candidate_run"]).payload.source.dirty is True for p in pairs)
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "blocked"
    assert draft.reason_codes == [DIRTY_SOURCE]
    assert draft.is_production is False


def test_t22_dirty_baseline_also_blocks(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="dirtyb", baseline_kwargs={"dirty": True})
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.reason_codes == [DIRTY_SOURCE]


def test_t22_trusted_worker_not_required_is_never_production(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="relaxed")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, permissive_policy(min_confirm_pairs=3))
    assert draft.outcome == "accepted"
    assert draft.is_production is False
    assert any("not production" in note for note in draft.notes)


# --------------------------------------------------------------------------------------
# Other blocking gates: subject mismatch, comparability, execution/correctness, missing evidence
# --------------------------------------------------------------------------------------
def test_candidate_subject_mismatch_is_blocked(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-b", DEMO_PAIR, default_policy())
    assert draft.outcome == "blocked"
    assert CANDIDATE_SUBJECT_MISMATCH in draft.reason_codes
    assert draft.candidate_subject_ref == "commit-demo-b"


def test_not_comparable_pair_is_blocked_with_field_notes(demo_store):
    cand = trusted_run(demo_store, request_id="nc-cand", subject_ref="commit-demo-a", session_id="s", pair_id="p", role="candidate", environment_overrides={"accelerator_model": "OTHER"}, samples=(9.0,) * 5)
    base = trusted_run(demo_store, request_id="nc-base", subject_ref="baseline-demo", session_id="s", pair_id="p", role="baseline", samples=(10.0,) * 5)
    draft = evaluate_candidate(demo_store, "commit-demo-a", [{"candidate_run": cand.record_id, "baseline_run": base.record_id}], default_policy())
    assert draft.outcome == "blocked"
    assert NOT_COMPARABLE in draft.reason_codes
    assert any("accelerator_model" in note for note in draft.notes)
    assert draft.pair_evaluations[0].pair_speedup is None


def test_explicit_comparison_key_mismatch_is_blocked(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="ck")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy(), comparison_key="sha256:" + "0" * 64)
    assert draft.outcome == "blocked"
    assert NOT_COMPARABLE in draft.reason_codes
    assert draft.comparison_key == "sha256:" + "0" * 64


def test_failed_execution_is_blocked(demo_store):
    pair = [{"candidate_run": "run-demo-c-failure", "baseline_run": "run-demo-baseline"}]
    draft = evaluate_candidate(demo_store, "commit-demo-c", pair, permissive_policy())
    assert draft.outcome == "blocked"
    assert EXECUTION_NOT_SUCCEEDED in draft.reason_codes
    assert CORRECTNESS_NOT_PASSED in draft.reason_codes
    assert draft.pair_evaluations[0].pair_speedup is None


def test_correctness_failure_is_blocked(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="wrong", candidate_kwargs={"correctness": "fail"})
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "blocked"
    assert CORRECTNESS_NOT_PASSED in draft.reason_codes
    # An ineligible pair carries no timing evidence, so it never counts as a confirmation pair either.
    assert INSUFFICIENT_CONFIRMATION_PAIRS in draft.reason_codes
    assert set(draft.reason_codes) == {CORRECTNESS_NOT_PASSED, INSUFFICIENT_CONFIRMATION_PAIRS}
    assert all(e.pair_speedup is None and not e.passed for e in draft.pair_evaluations)


def test_missing_samples_blob_is_missing_evidence(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="gone")
    cand = demo_store.get(pairs[0]["candidate_run"])
    ref = cand.payload.artifact_by_id()[cand.payload.timing.samples_artifact_ref]
    demo_store.artifact_path(ref.sha256).unlink()
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    assert draft.outcome == "blocked"
    assert draft.reason_codes == [MISSING_EVIDENCE]
    assert any(note.startswith(MISSING_EVIDENCE) and "MISSING_SAMPLES" in note for note in draft.notes)


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------
def test_evaluate_candidate_input_errors(demo_store):
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", [], default_policy())
    assert info.value.code == "NO_PAIRS" and info.value.exit_code == 2
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy(), evaluated_by="robot")
    assert info.value.code == "INVALID_EVALUATOR"
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, None, DEMO_PAIR, default_policy())
    assert info.value.code == "MISSING_SUBJECT"
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", [{"candidate_run": "run-demo-a"}], default_policy())
    assert info.value.code == "INVALID_PAIRS"
    with pytest.raises(MissingReferenceError):
        evaluate_candidate(demo_store, "commit-missing", DEMO_PAIR, default_policy())
    with pytest.raises(MissingReferenceError):
        evaluate_candidate(demo_store, "commit-demo-a", [{"candidate_run": "run-missing", "baseline_run": "run-demo-baseline"}], default_policy())
    with pytest.raises(SchemaValidationError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, {"policy_id": "broken"})
    assert info.value.code == "INVALID_POLICY"
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy(), config_ref="cfg-other")
    assert info.value.code == "CONFIG_MISMATCH"


def test_pair_spellings_and_keyword_form_are_equivalent(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="forms")
    by_dict = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    by_alias = evaluate_candidate(demo_store, "commit-demo-a", [{"candidate": p["candidate_run"], "baseline": p["baseline_run"]} for p in pairs], default_policy())
    by_tuple = evaluate_candidate(demo_store, "commit-demo-a", [(p["candidate_run"], p["baseline_run"]) for p in pairs], default_policy())
    by_kw = evaluate_candidate(
        demo_store, "commit-demo-a", None, default_policy(),
        candidate_run_refs=[p["candidate_run"] for p in pairs], baseline_run_refs=[p["baseline_run"] for p in pairs],
    )
    assert by_dict.outcome == by_alias.outcome == by_tuple.outcome == by_kw.outcome == "accepted"
    assert by_dict.to_payload() == by_alias.to_payload() == by_tuple.to_payload() == by_kw.to_payload()
    with pytest.raises(InputError) as info:
        evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy(), candidate_run_refs=[pairs[0]["candidate_run"]])
    assert info.value.code == "INVALID_PAIRS"


def test_human_override_is_recorded(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy(), evaluated_by="human_override")
    assert draft.evaluated_by == "human_override"
    record = append_decision(demo_store, draft, created_at=T0)
    assert record.payload.evaluated_by == "human_override"


# --------------------------------------------------------------------------------------
# append_decision: idempotency, supersedes, best_known
# --------------------------------------------------------------------------------------
def test_append_decision_same_created_at_is_idempotent(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="idem")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    first = append_decision(demo_store, draft, created_at=T0)
    second = append_decision(demo_store, draft, created_at=T0)
    assert first.record_id == second.record_id
    assert first.to_dict() == second.to_dict()
    assert [d.record_id for d in demo_store.records("decision") if d.record_id == first.record_id] == [first.record_id]
    later = append_decision(demo_store, draft, created_at=T1)
    assert later.record_id != first.record_id
    assert deep_validate(demo_store).ok


def test_append_decision_rejects_non_drafts_and_unknown_supersedes(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="bad")
    draft = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    with pytest.raises(InputError) as info:
        append_decision(demo_store, draft.to_dict())  # type: ignore[arg-type]
    assert info.value.code == "INVALID_DRAFT"
    with pytest.raises(MissingReferenceError):
        append_decision(demo_store, draft, supersedes_decision_ref="decision-missing", created_at=T0)
    with pytest.raises(MissingReferenceError):
        append_decision(demo_store, draft, supersedes_decision_ref="run-demo-a", created_at=T0)


def test_supersedes_chain_and_best_known(demo_store):
    first_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="gen1"), default_policy())
    first = append_decision(demo_store, first_draft, created_at=T0)
    assert [k["decision_ref"] for k in best_known(demo_store, "cfg-demo")] == [first.record_id]

    second_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="gen2"), default_policy())
    second = append_decision(demo_store, second_draft, supersedes_decision_ref=first.record_id, created_at=T1)
    assert second.payload.supersedes_decision_ref == first.record_id
    assert second_draft.outcome == "accepted" and second.payload.is_production is True
    known = best_known(demo_store, "cfg-demo")
    assert [k["decision_ref"] for k in known] == [second.record_id]
    assert known[0]["supersedes"] == [first.record_id]
    assert supersedes_chain(demo_store, second) == [first.record_id]
    assert supersedes_chain(demo_store, first) == []

    third_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="gen3"), default_policy())
    third = append_decision(demo_store, third_draft, supersedes_decision_ref=second.record_id, created_at=T2)
    assert supersedes_chain(demo_store, third) == [second.record_id, first.record_id]
    known = best_known(demo_store, "cfg-demo")
    assert [k["decision_ref"] for k in known] == [third.record_id]
    assert known[0]["supersedes"] == [second.record_id, first.record_id]
    assert demo_store.get(first.record_id) is not None  # superseded decisions stay in the ledger
    assert deep_validate(demo_store).ok
    assert [b["decision_ref"] for b in build_trajectory(demo_store, "cfg-demo")["best_known"]] == [third.record_id]


def test_superseding_decision_wins_even_when_older(demo_store):
    """Supersession is explicit: a decision superseded by an earlier-stamped one is no longer best-known."""
    newer_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="new"), default_policy())
    newer = append_decision(demo_store, newer_draft, created_at=T1)
    older_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="old"), default_policy())
    older = append_decision(demo_store, older_draft, supersedes_decision_ref=newer.record_id, created_at=T0)
    assert [k["decision_ref"] for k in best_known(demo_store, "cfg-demo")] == [older.record_id]


def test_best_known_ignores_non_production_and_filters(demo_store):
    assert best_known(demo_store, "cfg-demo") == []  # the bundle's blocked decision never counts
    rejected_draft = evaluate_candidate(
        demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="rej", candidate_samples=(100.0,) * 5, baseline_samples=(101.0,) * 5), default_policy()
    )
    assert rejected_draft.outcome == "rejected"
    append_decision(demo_store, rejected_draft, created_at=T0)
    assert best_known(demo_store, "cfg-demo") == []
    accepted_draft = evaluate_candidate(demo_store, "commit-demo-a", trusted_pairs(demo_store, n=3, prefix="acc"), default_policy())
    accepted = append_decision(demo_store, accepted_draft, created_at=T1)
    assert [k["decision_ref"] for k in best_known(demo_store, "cfg-demo")] == [accepted.record_id]
    assert best_known(demo_store, "cfg-demo", policy_hash=accepted_draft.policy_hash)[0]["decision_ref"] == accepted.record_id
    assert best_known(demo_store, "cfg-demo", policy_hash="sha256:" + "f" * 64) == []
    assert best_known(demo_store, "cfg-demo", comparison_key=accepted_draft.comparison_key)[0]["decision_ref"] == accepted.record_id
    assert best_known(demo_store, "cfg-demo", comparison_key="sha256:" + "f" * 64) == []
    with pytest.raises(MissingReferenceError):
        best_known(demo_store, "cfg-missing")


def test_best_known_is_per_policy_hash_group(demo_store):
    pairs = trusted_pairs(demo_store, n=3, prefix="pol")
    strict = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy())
    looser = evaluate_candidate(demo_store, "commit-demo-a", pairs, default_policy() | {"policy_id": "loose-v1", "min_pair_speedup": "1.01"})
    assert strict.policy_hash != looser.policy_hash
    a = append_decision(demo_store, strict, created_at=T0)
    b = append_decision(demo_store, looser, created_at=T0)
    assert a.record_id != b.record_id
    known = best_known(demo_store, "cfg-demo")
    assert {k["decision_ref"] for k in known} == {a.record_id, b.record_id}
    assert {k["policy_id"] for k in known} == {"default-confirm-v1", "loose-v1"}


def test_draft_payload_and_dict_shapes(demo_store):
    draft = evaluate_candidate(demo_store, "commit-demo-a", DEMO_PAIR, default_policy())
    assert isinstance(draft, DecisionDraft)
    payload = draft.to_payload()
    assert payload["reason_codes"] == sorted(set(draft.reason_codes))
    assert payload["supersedes_decision_ref"] is None
    assert payload["policy"] == default_policy()
    dumped = draft.to_dict()
    assert dumped["outcome"] == "blocked"
    assert dumped["pair_evaluations"][0]["comparison"]["status"] == "COMPARABLE"
    assert dumped["pair_evaluations"][0]["pair_key"] == ["session-1", "pair-1", "session-1", "pair-1"]
