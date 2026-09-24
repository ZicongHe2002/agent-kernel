"""Correctness verification primitives (specification section 10; adapters/verify.py).

The acceptance rule ``abs(c - r) <= atol + rtol * abs(r)`` is applied to every element of
the complete output pytree; structure, shape and dtype must match; non-finite values follow
an explicit policy; the relative-error diagnostic states its epsilon and never replaces the
rule. Tolerance boundaries use binary-exact values so the tests are deterministic.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from kernel_memory.adapters import verify
from kernel_memory.adapters.verify import (
    ACCEPTANCE_RULE,
    NONFINITE_POLICIES,
    REL_ERROR_EPSILON,
    PytreeStructure,
    VerifyOutcome,
    correctness_report_bytes,
    elementwise_close,
    parse_tolerance,
    verify_pytree_structure,
)
from kernel_memory.domain.errors import InputError

NAN = float("nan")
INF = float("inf")


def close(candidate: Any, reference: Any, *, atol: Any = "0", rtol: Any = "0", policy: str = "reject_unexpected") -> VerifyOutcome:
    return elementwise_close(candidate, reference, atol=atol, rtol=rtol, nonfinite_policy=policy)


def strict_json(data: bytes) -> dict[str, Any]:
    """Parse and refuse the non-standard NaN/Infinity tokens (the artifact must be strict JSON)."""

    def refuse(token: str) -> Any:
        raise AssertionError(f"non-strict JSON token {token!r} in artifact")

    return json.loads(data.decode("utf-8"), parse_constant=refuse)


# ------------------------------------------------------------------------------ exact pass
def test_exact_match_passes_with_zero_errors() -> None:
    ref = np.array([0.5, -1.25, 3.0, 0.0], dtype=np.float32)
    outcome = close(ref.copy(), ref)
    assert outcome.passed is True
    assert outcome.mismatch_count == 0
    assert outcome.checked_count == 4
    assert outcome.max_abs_error == 0.0
    assert outcome.max_rel_error == 0.0
    assert outcome.message is None
    assert outcome.rel_error_epsilon == REL_ERROR_EPSILON


def test_scalars_are_a_single_root_leaf() -> None:
    outcome = close(1.5, 1.5)
    assert outcome.passed and outcome.checked_count == 1
    failing = close(1.5, 2.5)
    assert failing.passed is False and failing.mismatch_count == 1
    assert "<root>" in (failing.message or "")


def test_array_likes_exposing_dunder_array_are_leaves() -> None:
    class ArrayLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            return np.asarray(self._values, dtype=np.float32)

    assert close(ArrayLike([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32)).passed is True
    assert close(ArrayLike([1.0, 2.5]), np.array([1.0, 2.0], dtype=np.float32)).passed is False


def test_nested_python_lists_are_one_array_leaf() -> None:
    reference = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert close([[1, 2], [3, 4]], reference).passed is True
    outcome = close([[1, 2], [3, 5]], reference)
    assert outcome.passed is False and outcome.mismatch_count == 1
    assert "[1,1]" in (outcome.message or "")


def test_to_dict_reports_epsilon_and_counts() -> None:
    doc = close(np.ones(3), np.ones(3)).to_dict()
    assert doc == {
        "passed": True,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "rel_error_epsilon": REL_ERROR_EPSILON,
        "mismatch_count": 0,
        "checked_count": 3,
        "message": None,
    }


# ------------------------------------------------------------------------------ tolerance boundaries
def test_difference_exactly_atol_passes_and_slightly_above_fails() -> None:
    reference = np.array([1.0, 2.0, -4.0], dtype=np.float64)
    exactly = reference + 0.5  # exact in binary floating point
    assert close(exactly, reference, atol="0.5").passed is True
    assert close(exactly, reference, atol=0.5).passed is True
    above = reference + (0.5 + 2.0**-40)
    outcome = close(above, reference, atol="0.5")
    assert outcome.passed is False
    assert outcome.mismatch_count == 3
    assert outcome.max_abs_error == pytest.approx(0.5 + 2.0**-40)


def test_difference_exactly_rtol_times_reference_passes_and_above_fails() -> None:
    reference = np.array([8.0, -16.0], dtype=np.float64)
    candidate = np.array([9.0, -18.0], dtype=np.float64)  # diff = 0.125 * |ref| exactly
    assert close(candidate, reference, rtol="0.125").passed is True
    above = candidate + np.array([2.0**-40, 0.0])
    outcome = close(above, reference, rtol="0.125")
    assert outcome.passed is False and outcome.mismatch_count == 1
    assert "[0]" in (outcome.message or "")


def test_atol_and_rtol_combine_additively() -> None:
    reference = np.array([4.0])
    candidate = np.array([4.0 + 0.5 + 0.5])  # atol 0.5 + rtol 0.125*4 = 1.0
    assert close(candidate, reference, atol="0.5", rtol="0.125").passed is True
    assert close(candidate, reference, atol="0.5", rtol="0").passed is False
    assert close(candidate, reference, atol="0", rtol="0.125").passed is False


def test_diagnostic_relative_error_never_replaces_the_acceptance_rule() -> None:
    reference = np.array([1.0e6])
    candidate = np.array([1.0e6 + 1.0])
    outcome = close(candidate, reference)  # atol = rtol = 0: any difference fails
    assert outcome.passed is False
    assert outcome.max_rel_error == pytest.approx(1.0e-6)  # tiny, but the rule still governs
    assert outcome.max_abs_error == 1.0


def test_failure_message_states_rule_tolerances_policy_and_first_mismatch() -> None:
    outcome = close(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]), atol="0.1", policy="reject_all")
    message = outcome.message or ""
    assert message.startswith("1 of 3 elements violate")
    assert ACCEPTANCE_RULE in message
    assert "atol=0.1" in message and "rtol=0.0" in message
    assert "nonfinite_policy=reject_all" in message
    assert "[1]: candidate=2.0 reference=2.5" in message


# ------------------------------------------------------------------------------ structural mismatches
def test_shape_mismatch_fails_with_distinguishing_message() -> None:
    outcome = close(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32))
    assert outcome.passed is False
    assert "SHAPE_MISMATCH" in (outcome.message or "")
    assert "(3,)" in outcome.message and "(2,)" in outcome.message
    assert "DTYPE_MISMATCH" not in outcome.message and "STRUCTURE_MISMATCH" not in outcome.message
    assert outcome.max_abs_error is None and outcome.max_rel_error is None
    assert outcome.checked_count == 0


def test_dtype_mismatch_fails_with_distinguishing_message() -> None:
    outcome = close(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float32))
    assert outcome.passed is False
    assert "DTYPE_MISMATCH" in (outcome.message or "")
    assert "float64" in outcome.message and "float32" in outcome.message
    assert "SHAPE_MISMATCH" not in outcome.message
    assert outcome.max_abs_error is None


def test_structure_mismatch_dict_versus_array() -> None:
    outcome = close({"o": np.zeros(2)}, np.zeros(2))
    assert outcome.passed is False
    assert "STRUCTURE_MISMATCH" in (outcome.message or "")
    assert "candidate is dict, reference is leaf" in outcome.message
    assert "SHAPE_MISMATCH" not in outcome.message and "DTYPE_MISMATCH" not in outcome.message


def test_structure_mismatch_missing_required_key_is_named() -> None:
    reference = {"o": np.zeros(2), "lse": np.zeros(1)}
    outcome = close({"o": np.zeros(2)}, reference)
    assert outcome.passed is False
    assert "STRUCTURE_MISMATCH" in (outcome.message or "")
    assert "missing keys ['lse']" in outcome.message
    extra = close({"o": np.zeros(2), "lse": np.zeros(1), "extra": np.zeros(1)}, reference)
    assert extra.passed is False and "unexpected keys ['extra']" in (extra.message or "")


def test_list_and_tuple_are_distinct_containers_and_lengths_must_match() -> None:
    arrays = [np.zeros(2), np.ones(2)]
    assert close(list(arrays), list(arrays)).passed is True
    assert close(tuple(arrays), tuple(arrays)).passed is True
    kind = close(tuple(arrays), list(arrays))
    assert kind.passed is False and "candidate is tuple, reference is list" in (kind.message or "")
    length = close([np.zeros(2)], list(arrays))
    assert length.passed is False and "list length 1 vs 2" in (length.message or "")


def test_unsupported_leaf_types_fail_structurally() -> None:
    outcome = close({"o": "text"}, {"o": "text"})
    assert outcome.passed is False
    assert "STRUCTURE_MISMATCH" in (outcome.message or "") and "unsupported:str" in outcome.message


def test_non_numeric_dtype_is_unsupported() -> None:
    outcome = close(np.array(["a", "b"]), np.array(["a", "b"]))
    assert outcome.passed is False
    assert "UNSUPPORTED_DTYPE" in (outcome.message or "")


def test_empty_outputs_with_matching_structure_pass_with_note() -> None:
    outcome = close(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert outcome.passed is True and outcome.checked_count == 0
    assert outcome.max_abs_error is None and outcome.max_rel_error is None
    assert "no elements compared" in (outcome.message or "")


# ------------------------------------------------------------------------------ non-finite policies
def test_nan_in_candidate_where_reference_finite_fails_under_reject_unexpected() -> None:
    outcome = close(np.array([1.0, NAN, 3.0]), np.array([1.0, 2.0, 3.0]), policy="reject_unexpected")
    assert outcome.passed is False
    assert outcome.mismatch_count == 1
    assert "[1]: candidate=nan reference=2.0" in (outcome.message or "")
    # diagnostics cover only the finite pairs
    assert outcome.max_abs_error == 0.0 and outcome.checked_count == 3


def test_inf_in_candidate_where_reference_finite_fails_under_every_policy() -> None:
    for policy in NONFINITE_POLICIES:
        outcome = close(np.array([INF]), np.array([1.0]), policy=policy)
        assert outcome.passed is False, policy


def test_matching_positive_inf_passes_under_match_reference() -> None:
    reference = np.array([INF, 1.0, -INF])
    outcome = close(reference.copy(), reference, policy="match_reference")
    assert outcome.passed is True and outcome.mismatch_count == 0
    # reject_unexpected shares the rule: expected sentinels must match exactly
    assert close(reference.copy(), reference, policy="reject_unexpected").passed is True


def test_infinity_sign_must_agree_under_match_reference() -> None:
    outcome = close(np.array([-INF]), np.array([INF]), policy="match_reference")
    assert outcome.passed is False and outcome.mismatch_count == 1


def test_nan_matches_nan_under_match_reference_but_finite_does_not() -> None:
    assert close(np.array([NAN, 1.0]), np.array([NAN, 1.0]), policy="match_reference").passed is True
    outcome = close(np.array([0.0, 1.0]), np.array([NAN, 1.0]), policy="match_reference")
    assert outcome.passed is False and outcome.mismatch_count == 1
    assert close(np.array([0.0]), np.array([NAN]), policy="reject_unexpected").passed is False


def test_reject_all_fails_on_any_non_finite_on_either_side() -> None:
    finite = np.array([1.0, 2.0])
    assert close(finite, finite, policy="reject_all").passed is True
    assert close(np.array([INF, 2.0]), np.array([INF, 2.0]), policy="reject_all").passed is False
    assert close(np.array([NAN, 2.0]), np.array([NAN, 2.0]), policy="reject_all").passed is False
    assert close(finite, np.array([-INF, 2.0]), policy="reject_all").passed is False
    outcome = close(np.array([1.0, NAN]), finite, policy="reject_all")
    assert outcome.passed is False and outcome.mismatch_count == 1


def test_unknown_nonfinite_policy_is_input_error() -> None:
    with pytest.raises(InputError) as excinfo:
        elementwise_close(np.ones(1), np.ones(1), atol="0", rtol="0", nonfinite_policy="accept_all")
    assert excinfo.value.code == "INVALID_NONFINITE_POLICY"
    assert excinfo.value.details["known"] == list(NONFINITE_POLICIES)


# ------------------------------------------------------------------------------ pytrees
def test_dict_pytree_passes_and_reports_failing_leaf_path() -> None:
    reference = {"o": np.arange(6, dtype=np.float32).reshape(2, 3), "lse": np.array([0.5, 1.5], dtype=np.float32)}
    candidate = {"o": reference["o"].copy(), "lse": reference["lse"].copy()}
    outcome = close(candidate, reference)
    assert outcome.passed is True and outcome.checked_count == 8

    candidate["lse"] = np.array([0.5, 2.5], dtype=np.float32)
    failing = close(candidate, reference)
    assert failing.passed is False and failing.mismatch_count == 1
    assert "lse[1]: candidate=2.5 reference=1.5" in (failing.message or "")
    assert failing.max_abs_error == 1.0
    assert failing.max_rel_error == pytest.approx(1.0 / 1.5)


def test_list_pytree_and_nested_containers() -> None:
    reference = [np.ones(2), {"inner": (np.zeros(1), np.array(3.0))}]
    candidate = [np.ones(2), {"inner": (np.zeros(1), np.array(3.0))}]
    assert close(candidate, reference).passed is True
    candidate[1]["inner"] = (np.zeros(1), np.array(4.0))
    failing = close(candidate, reference)
    assert failing.passed is False
    assert "[1].inner[1]" in (failing.message or "")


def test_none_leaves_match_each_other_only() -> None:
    assert close({"o": np.ones(1), "aux": None}, {"o": np.ones(1), "aux": None}).passed is True
    outcome = close({"o": np.ones(1), "aux": np.ones(1)}, {"o": np.ones(1), "aux": None})
    assert outcome.passed is False and "candidate is leaf, reference is none" in (outcome.message or "")


def test_complex_outputs_compare_in_complex128() -> None:
    reference = np.array([1 + 2j, -3j], dtype=np.complex64)
    assert close(reference.copy(), reference).passed is True
    outcome = close(np.array([1 + 2j, -3.5j], dtype=np.complex64), reference)
    assert outcome.passed is False and outcome.max_abs_error == pytest.approx(0.5)


# ------------------------------------------------------------------------------ tolerances
def test_string_tolerances_are_parsed_with_decimal() -> None:
    assert parse_tolerance("1e-6", "atol") == float(Decimal("1e-6"))
    assert parse_tolerance("0.1") == 0.1
    assert parse_tolerance(" 2.5 ") == 2.5
    assert parse_tolerance("0") == 0.0
    assert parse_tolerance(3) == 3.0
    assert parse_tolerance(0.25) == 0.25


@pytest.mark.parametrize("value", ["abc", "", "1e", "NaN", "Infinity", "-Infinity", "-1", -0.5, float("nan"), float("inf"), True, None, [1]])
def test_invalid_tolerances_are_input_errors(value: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        parse_tolerance(value, "rtol")
    assert excinfo.value.code == "INVALID_TOLERANCE"
    assert "rtol" in excinfo.value.message


def test_elementwise_close_rejects_invalid_tolerances_before_comparing() -> None:
    with pytest.raises(InputError) as excinfo:
        elementwise_close(np.ones(1), np.ones(1), atol="-1", rtol="0", nonfinite_policy="reject_all")
    assert excinfo.value.code == "INVALID_TOLERANCE" and "atol" in excinfo.value.message
    with pytest.raises(InputError):
        elementwise_close(np.ones(1), np.ones(1), atol="0", rtol="oops", nonfinite_policy="reject_all")


def test_string_tolerance_boundary_matches_decimal_semantics() -> None:
    reference = np.array([0.0])
    candidate = np.array([0.1])
    # "0.1" -> float(Decimal("0.1")) == 0.1 exactly as the float literal, so the boundary passes
    assert close(candidate, reference, atol="0.1").passed is True
    assert close(np.array([0.1 + 2.0**-50]), reference, atol="0.1").passed is False


# ------------------------------------------------------------------------------ relative error epsilon
def test_relative_error_epsilon_is_reported_and_used_as_denominator_floor() -> None:
    assert REL_ERROR_EPSILON == 1e-12
    outcome = close(np.array([1e-13]), np.array([0.0]), atol="1")
    assert outcome.passed is True
    assert outcome.rel_error_epsilon == 1e-12
    # denominator = max(|0.0|, eps) = 1e-12  ->  1e-13 / 1e-12 = 0.1
    assert outcome.max_rel_error == pytest.approx(0.1)
    assert outcome.max_abs_error == pytest.approx(1e-13)


def test_max_errors_are_maxima_over_finite_pairs() -> None:
    outcome = close(np.array([1.0, 2.0, 10.0]), np.array([1.0, 2.5, 8.0]), atol="5")
    assert outcome.passed is True
    assert outcome.max_abs_error == 2.0
    assert outcome.max_rel_error == pytest.approx(0.25)  # 2/8 > 0.5/2.5


# ------------------------------------------------------------------------------ verify_pytree_structure
def test_verify_pytree_structure_aligns_leaves_with_paths() -> None:
    reference = {"o": np.zeros((2, 2)), "aux": [np.ones(1), {"b": 3.0}]}
    structure = verify_pytree_structure(reference, reference)
    assert isinstance(structure, PytreeStructure)
    assert structure.matches is True and structure.problems == []
    assert [path for path, _, _ in structure.leaves] == ["aux[0]", "aux[1].b", "o"]
    for _, cand, ref in structure.leaves:
        assert cand is ref


def test_verify_pytree_structure_reports_every_problem() -> None:
    structure = verify_pytree_structure({"o": np.zeros(1), "x": [1, 2]}, {"o": np.zeros(1), "lse": np.zeros(1)})
    assert structure.matches is False
    assert len(structure.problems) == 1
    assert "missing keys ['lse']" in structure.problems[0] and "unexpected keys ['x']" in structure.problems[0]
    # shared leaves are still aligned so the caller can describe what was comparable
    assert [path for path, _, _ in structure.leaves] == ["o"]


def test_verify_pytree_structure_root_leaf_and_none() -> None:
    root = verify_pytree_structure(np.zeros(2), np.zeros(3))  # shape is a leaf-level concern, not structure
    assert root.matches is True and [p for p, _, _ in root.leaves] == ["<root>"]
    none = verify_pytree_structure(None, None)
    assert none.matches is True and none.leaves == []


# ------------------------------------------------------------------------------ correctness report artifact
def test_correctness_report_bytes_is_strict_json_with_rule_and_maxima() -> None:
    cases = [
        {"case": "uniform", "passed": True, "max_abs_error": 0.0, "max_rel_error": 0.0},
        {"case": "large", "passed": False, "max_abs_error": 0.5, "max_rel_error": 0.25},
        {"case": "errored", "passed": False, "max_abs_error": None, "max_rel_error": None},
    ]
    data = correctness_report_bytes(
        status="fail",
        cases=cases,
        tolerances={"atol": "1e-6", "rtol": "1e-6"},
        nonfinite_policy="reject_unexpected",
        reference_id="demo:reference",
        entrypoint="demo:candidate",
    )
    doc = strict_json(data)
    assert doc["kind"] == "correctness_report"
    assert doc["status"] == "fail"
    assert doc["rule"] == ACCEPTANCE_RULE
    assert doc["tolerances"] == {"atol": "1e-6", "rtol": "1e-6"}
    assert doc["nonfinite_policy"] == "reject_unexpected"
    assert doc["rel_error_epsilon"] == REL_ERROR_EPSILON
    assert doc["reference_id"] == "demo:reference"
    assert doc["cases_total"] == 3 and doc["cases_passed"] == 1
    assert doc["max_abs_error"] == 0.5 and doc["max_rel_error"] == 0.25
    assert doc["cases"] == cases
    assert doc["entrypoint"] == "demo:candidate"
    assert doc["is_fixture"] is False


def test_correctness_report_uncollected_maxima_are_null_not_zero() -> None:
    doc = strict_json(
        correctness_report_bytes(status="error", cases=[{"case": "c", "passed": False}], tolerances={}, nonfinite_policy="reject_all", reference_id="r")
    )
    assert doc["max_abs_error"] is None and doc["max_rel_error"] is None
    assert doc["cases_total"] == 1 and doc["cases_passed"] == 0


def test_correctness_report_writes_non_finite_numbers_as_strings() -> None:
    cases = [{"case": "c", "passed": False, "max_abs_error": INF, "max_rel_error": 0.0, "value": NAN, "arr": np.array([1.0, INF])}]
    doc = strict_json(correctness_report_bytes(status="fail", cases=cases, tolerances={}, nonfinite_policy="reject_all", reference_id="r"))
    case = doc["cases"][0]
    assert case["max_abs_error"] == "inf" and case["value"] == "nan"
    assert case["arr"] == [1.0, "inf"]
    assert doc["max_abs_error"] == "inf"  # the maximum is reported, then made strict-JSON safe


def test_correctness_report_numpy_scalars_become_plain_json() -> None:
    cases = [{"case": "c", "passed": np.bool_(True), "max_abs_error": np.float32(0.5), "count": np.int64(3)}]
    doc = strict_json(correctness_report_bytes(status="pass", cases=cases, tolerances={}, nonfinite_policy="reject_all", reference_id="r"))
    assert doc["cases"][0] == {"case": "c", "passed": True, "max_abs_error": 0.5, "count": 3}
    assert doc["cases_passed"] == 1


@pytest.mark.parametrize("status", ["ok", "PASS", "", None])
def test_correctness_report_rejects_invalid_status(status: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        correctness_report_bytes(status=status, cases=[], tolerances={}, nonfinite_policy="reject_all", reference_id="r")
    assert excinfo.value.code == "INVALID_STATUS"


@pytest.mark.parametrize("key", ["kind", "is_fixture"])
def test_correctness_report_reserved_metadata_keys_are_refused(key: str) -> None:
    with pytest.raises(InputError) as excinfo:
        correctness_report_bytes(status="pass", cases=[], tolerances={}, nonfinite_policy="reject_all", reference_id="r", **{key: "x"})
    assert excinfo.value.code == "RESERVED_KEY"


def test_correctness_report_cannot_be_relabelled_as_fixture() -> None:
    with pytest.raises(InputError):
        correctness_report_bytes(status="pass", cases=[], tolerances={}, nonfinite_policy="reject_all", reference_id="r", is_fixture=True)


def test_module_exports_are_pure_and_documented() -> None:
    assert set(verify.__all__) >= {"elementwise_close", "verify_pytree_structure", "parse_tolerance", "correctness_report_bytes"}
    assert math.isfinite(REL_ERROR_EPSILON) and REL_ERROR_EPSILON > 0
