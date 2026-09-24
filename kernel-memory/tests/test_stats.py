"""Derived statistics over raw latency samples (spec 11.2). Nothing else may invent these numbers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from conftest import record_dict

from kernel_memory.domain import stats
from kernel_memory.domain.errors import InputError

BASELINE_SAMPLES = [100, 101, 100, 99, 100]
CANDIDATE_SAMPLES = [90, 91, 90, 89, 90]


# --------------------------------------------------------------------------------------
# Fixture arithmetic (fictional medians exist only to exercise the formulas)
# --------------------------------------------------------------------------------------
def test_baseline_samples_median_and_p90() -> None:
    assert stats.median(BASELINE_SAMPLES) == 100
    assert stats.p90(BASELINE_SAMPLES) == pytest.approx(100.6)


def test_candidate_samples_median_and_p90() -> None:
    assert stats.median(CANDIDATE_SAMPLES) == 90
    assert stats.p90(CANDIDATE_SAMPLES) == pytest.approx(90.6)


def test_fixture_artifacts_agree_with_recorded_summaries(bundle_dicts: list[dict], artifact_root: Path) -> None:
    for run_id in ("run-demo-baseline", "run-demo-a", "run-demo-c"):
        payload = record_dict(bundle_dicts, run_id)["payload"]
        ref = next(a for a in payload["artifacts"] if a["artifact_id"] == payload["timing"]["samples_artifact_ref"])
        doc = json.loads((artifact_root / ref["uri"]).read_bytes())
        assert stats.summaries_agree(doc["samples"], doc["unit"], payload["timing"]["median_us"], payload["timing"]["p90_us"])
        assert stats.summarize(doc["samples"], doc["unit"]).sample_count == payload["timing"]["sample_count"]


def test_p90_uses_sorted_linear_interpolation_at_h_equals_n_minus_1_times_p() -> None:
    # sorted baseline: [99, 100, 100, 100, 101]; h = 4 * 0.9 = 3.6 -> 100 + 0.6 * (101 - 100) = 100.6
    assert stats.quantile_linear(BASELINE_SAMPLES, 0.9) == pytest.approx(100.6)
    # Order of the input must not matter.
    assert stats.p90(sorted(BASELINE_SAMPLES, reverse=True)) == pytest.approx(100.6)


# --------------------------------------------------------------------------------------
# speedup and latency reduction: 1.2x is NOT 20%
# --------------------------------------------------------------------------------------
def test_speedup_and_latency_reduction_for_fixture_medians() -> None:
    assert stats.speedup(100, 90) == 100 / 90
    assert stats.speedup(100, 90) == pytest.approx(1.1111111111)
    assert stats.latency_reduction_pct(100, 90) == 10.0


def test_speedup_1_2_corresponds_to_16_67_percent_reduction_not_20() -> None:
    baseline, candidate = 120.0, 100.0
    assert stats.speedup(baseline, candidate) == pytest.approx(1.2)
    reduction = stats.latency_reduction_pct(baseline, candidate)
    assert reduction == pytest.approx(16.666666666666668)
    assert reduction != pytest.approx(20.0)
    assert abs(reduction - 20.0) > 3.0


def test_latency_reduction_is_exact_when_medians_divide_cleanly() -> None:
    assert stats.latency_reduction_pct(100, 88) == 12.0
    assert stats.latency_reduction_pct(200, 100) == 50.0
    assert stats.latency_reduction_pct(100, 100) == 0.0


def test_speedup_below_one_is_negative_reduction() -> None:
    assert stats.speedup(90, 100) == pytest.approx(0.9)
    assert stats.latency_reduction_pct(90, 100) == pytest.approx(-11.111111111)


def test_speedup_and_reduction_are_consistent() -> None:
    for baseline, candidate in ((100, 90), (120, 100), (88, 100), (1.5, 0.5)):
        s = stats.speedup(baseline, candidate)
        r = stats.latency_reduction_pct(baseline, candidate)
        assert r == pytest.approx(100.0 * (1.0 - 1.0 / s))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "100", None])
def test_speedup_rejects_nonpositive_or_nonnumeric_medians(bad: object) -> None:
    with pytest.raises(InputError) as info:
        stats.speedup(bad, 90)  # type: ignore[arg-type]
    assert info.value.code == "INVALID_SUMMARY"
    with pytest.raises(InputError):
        stats.speedup(100, bad)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        stats.latency_reduction_pct(bad, 90)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        stats.latency_reduction_pct(100, bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# median / quantile_linear / normalized_iqr
# --------------------------------------------------------------------------------------
def test_median_odd_and_even_counts() -> None:
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 3, 2]) == 2.5
    assert stats.median([7]) == 7
    assert stats.median([1, 1000]) == 500.5


def test_quantile_linear_even_count() -> None:
    samples = [1, 2, 3, 4]
    assert stats.quantile_linear(samples, 0.0) == 1
    assert stats.quantile_linear(samples, 0.5) == 2.5
    assert stats.quantile_linear(samples, 0.9) == pytest.approx(3.7)
    assert stats.quantile_linear(samples, 1.0) == 4


def test_quantile_linear_odd_count() -> None:
    samples = [5, 1, 4, 2, 3]
    assert stats.quantile_linear(samples, 0.5) == 3
    assert stats.quantile_linear(samples, 0.25) == 2
    assert stats.quantile_linear(samples, 0.75) == 4
    assert stats.quantile_linear(samples, 0.9) == pytest.approx(4.6)


def test_quantile_linear_single_sample_is_that_sample() -> None:
    for p in (0.0, 0.5, 0.9, 1.0):
        assert stats.quantile_linear([42], p) == 42


def test_quantile_linear_median_agrees_with_median() -> None:
    for samples in ([1, 2, 3, 4], [5, 1, 4, 2, 3], BASELINE_SAMPLES, [0.5, 0.25]):
        assert stats.quantile_linear(samples, 0.5) == stats.median(samples)


@pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
def test_quantile_linear_rejects_probability_outside_unit_interval(p: float) -> None:
    with pytest.raises(InputError) as info:
        stats.quantile_linear([1, 2, 3], p)
    assert info.value.code == "INVALID_QUANTILE"


def test_normalized_iqr() -> None:
    # sorted [1,2,3,4,5]: Q75 = 4, Q25 = 2, median = 3 -> (4-2)/3
    assert stats.normalized_iqr([5, 1, 4, 2, 3]) == pytest.approx(2 / 3)
    assert stats.normalized_iqr(BASELINE_SAMPLES) == 0.0  # Q25 == Q75 == 100
    # sorted [10, 20, 30, 40]: Q25 = 17.5, Q75 = 32.5, median = 25 -> 15/25
    assert stats.normalized_iqr([40, 10, 30, 20]) == pytest.approx(0.6)
    assert stats.normalized_iqr([7]) == 0.0


def test_normalized_iqr_is_scale_invariant() -> None:
    samples = [5, 1, 4, 2, 3]
    assert stats.normalized_iqr([s * 1000 for s in samples]) == pytest.approx(stats.normalized_iqr(samples))


# --------------------------------------------------------------------------------------
# validate_samples rejections
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "samples",
    [
        [],
        [100, -1],
        [0, 100],
        [100, float("nan")],
        [float("inf"), 100],
        [float("-inf")],
        [True, 100],
        [False],
        [100, "101"],
        [None],
        "100",
        100,
        None,
    ],
)
def test_invalid_samples_are_rejected(samples: object) -> None:
    with pytest.raises(InputError) as info:
        stats.validate_samples(samples)  # type: ignore[arg-type]
    assert info.value.code == "INVALID_SAMPLES"
    assert info.value.exit_code == 2
    for fn in (stats.median, stats.p90, stats.normalized_iqr):
        with pytest.raises(InputError):
            fn(samples)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        stats.summarize(samples, "microseconds")  # type: ignore[arg-type]


def test_validate_samples_returns_floats_and_accepts_tuple() -> None:
    cleaned = stats.validate_samples((1, 2.5, 3))
    assert cleaned == [1.0, 2.5, 3.0]
    assert all(isinstance(v, float) for v in cleaned)


# --------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------
def test_to_microseconds_conversions() -> None:
    assert stats.to_microseconds([1, 2], "milliseconds") == [1000.0, 2000.0]
    assert stats.to_microseconds([1000, 500], "nanoseconds") == [1.0, 0.5]
    assert stats.to_microseconds([1, 0.001], "seconds") == [1_000_000.0, 1000.0]
    assert stats.to_microseconds([100, 101], "microseconds") == [100.0, 101.0]
    assert stats.to_microseconds([], "seconds") == []


@pytest.mark.parametrize("unit", ["us", "ms", "cycles", "", "Microseconds", None])
def test_unknown_unit_is_rejected(unit: object) -> None:
    with pytest.raises(InputError) as info:
        stats.to_microseconds([1], unit)  # type: ignore[arg-type]
    assert info.value.code == "UNKNOWN_UNIT"
    with pytest.raises(InputError) as info2:
        stats.summarize([1, 2, 3], unit)  # type: ignore[arg-type]
    assert info2.value.code == "UNKNOWN_UNIT"


def test_supported_units_are_exactly_the_four_si_units() -> None:
    assert stats.SUPPORTED_UNITS == {"nanoseconds", "microseconds", "milliseconds", "seconds"}


# --------------------------------------------------------------------------------------
# summarize / summaries_agree
# --------------------------------------------------------------------------------------
def test_summarize_fields_for_microsecond_samples() -> None:
    summary = stats.summarize(BASELINE_SAMPLES, "microseconds")
    assert isinstance(summary, stats.SampleSummary)
    assert summary.unit == "microseconds"
    assert summary.sample_count == 5
    assert summary.median == 100.0
    assert summary.p90 == pytest.approx(100.6)
    assert summary.normalized_iqr == 0.0
    assert summary.median_us == 100.0
    assert summary.p90_us == pytest.approx(100.6)


def test_summarize_converts_to_microseconds_but_keeps_native_summary() -> None:
    summary = stats.summarize([1, 2, 3, 4, 5], "milliseconds")
    assert summary.median == 3.0 and summary.p90 == pytest.approx(4.6)
    assert summary.median_us == 3000.0 and summary.p90_us == pytest.approx(4600.0)
    assert summary.normalized_iqr == pytest.approx(2 / 3)
    nanos = stats.summarize([1000, 2000, 3000], "nanoseconds")
    assert nanos.median == 2000.0 and nanos.median_us == 2.0


def test_summary_is_frozen() -> None:
    summary = stats.summarize([1, 2, 3], "microseconds")
    with pytest.raises(Exception):
        summary.median = 0.0  # type: ignore[misc]


def test_summaries_agree_true_for_recorded_fixture_values() -> None:
    assert stats.summaries_agree(BASELINE_SAMPLES, "microseconds", 100, 100.6) is True
    assert stats.summaries_agree(CANDIDATE_SAMPLES, "microseconds", 90, 90.6) is True
    # Unit conversion is honoured: same samples expressed in nanoseconds.
    assert stats.summaries_agree([s * 1000 for s in BASELINE_SAMPLES], "nanoseconds", 100, 100.6) is True


def test_t20_summaries_agree_false_when_recorded_numbers_disagree() -> None:
    assert stats.summaries_agree(BASELINE_SAMPLES, "microseconds", 101, 100.6) is False
    assert stats.summaries_agree(BASELINE_SAMPLES, "microseconds", 100, 101) is False
    assert stats.summaries_agree(BASELINE_SAMPLES, "milliseconds", 100, 100.6) is False  # wrong unit
    assert stats.summaries_agree(CANDIDATE_SAMPLES, "microseconds", 100, 100.6) is False  # wrong samples


def test_summaries_agree_tolerates_only_floating_point_noise() -> None:
    assert stats.summaries_agree(BASELINE_SAMPLES, "microseconds", 100 + 1e-12, 100.6) is True
    assert stats.summaries_agree(BASELINE_SAMPLES, "microseconds", 100 + 1e-6, 100.6) is False


def test_summaries_agree_rejects_invalid_samples_or_unit() -> None:
    with pytest.raises(InputError):
        stats.summaries_agree([], "microseconds", 0, 0)
    with pytest.raises(InputError):
        stats.summaries_agree(BASELINE_SAMPLES, "us", 100, 100.6)


def test_stats_never_uses_wall_clock() -> None:
    """All functions are pure over their inputs: calling twice yields identical values."""
    a = stats.summarize(BASELINE_SAMPLES, "microseconds")
    b = stats.summarize(list(BASELINE_SAMPLES), "microseconds")
    assert a == b
    assert math.isfinite(a.p90_us)
