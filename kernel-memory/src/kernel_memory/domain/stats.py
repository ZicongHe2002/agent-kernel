"""Pure statistics over raw latency samples (specification section 11.2).

All derived numbers (median, p90, speedup, latency reduction, normalised IQR)
are computed from raw samples by these functions. Nothing else may invent them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import InputError

SUPPORTED_UNITS = {"microseconds", "nanoseconds", "milliseconds", "seconds"}
_TO_MICROSECONDS = {"nanoseconds": 1e-3, "microseconds": 1.0, "milliseconds": 1e3, "seconds": 1e6}


def validate_samples(samples: Sequence[float]) -> list[float]:
    if not isinstance(samples, (list, tuple)) or len(samples) == 0:
        raise InputError("samples must be a non-empty list", code="INVALID_SAMPLES")
    cleaned: list[float] = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(f"sample is not a number: {value!r}", code="INVALID_SAMPLES")
        if not math.isfinite(value) or value <= 0:
            raise InputError(f"latency samples must be positive and finite, got {value!r}", code="INVALID_SAMPLES")
        cleaned.append(float(value))
    return cleaned


def median(samples: Sequence[float]) -> float:
    values = sorted(validate_samples(samples))
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def quantile_linear(samples: Sequence[float], probability: float) -> float:
    """Linear interpolation at h = (n - 1) * p over the sorted samples."""
    if not (0.0 <= probability <= 1.0):
        raise InputError("probability must be within [0, 1]", code="INVALID_QUANTILE")
    values = sorted(validate_samples(samples))
    position = (len(values) - 1) * probability
    lo = int(math.floor(position))
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def p90(samples: Sequence[float]) -> float:
    return quantile_linear(samples, 0.9)


def normalized_iqr(samples: Sequence[float]) -> float:
    q75 = quantile_linear(samples, 0.75)
    q25 = quantile_linear(samples, 0.25)
    return (q75 - q25) / median(samples)


def speedup(baseline_median: float, candidate_median: float) -> float:
    _check_positive(baseline_median, "baseline median")
    _check_positive(candidate_median, "candidate median")
    return baseline_median / candidate_median


def latency_reduction_pct(baseline_median: float, candidate_median: float) -> float:
    """``100 * (1 - T_candidate / T_baseline)`` (specification 11.2).

    Evaluated as ``100 * (T_baseline - T_candidate) / T_baseline``: algebraically identical but
    exact when the medians divide cleanly (100 us -> 90 us is exactly 10.0, not 9.999999999999998).
    """
    _check_positive(baseline_median, "baseline median")
    _check_positive(candidate_median, "candidate median")
    return 100.0 * (float(baseline_median) - float(candidate_median)) / float(baseline_median)


def _check_positive(value: float, what: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise InputError(f"{what} must be positive and finite, got {value!r}", code="INVALID_SUMMARY")


def to_microseconds(samples: Iterable[float], unit: str) -> list[float]:
    if unit not in _TO_MICROSECONDS:
        raise InputError(f"unknown sample unit: {unit!r}", code="UNKNOWN_UNIT")
    factor = _TO_MICROSECONDS[unit]
    return [float(s) * factor for s in samples]


@dataclass(frozen=True)
class SampleSummary:
    unit: str
    sample_count: int
    median: float
    p90: float
    normalized_iqr: float
    median_us: float
    p90_us: float


def summarize(samples: Sequence[float], unit: str) -> SampleSummary:
    values = validate_samples(samples)
    if unit not in SUPPORTED_UNITS:
        raise InputError(f"unknown sample unit: {unit!r}", code="UNKNOWN_UNIT")
    med = median(values)
    q90 = p90(values)
    micro = to_microseconds(values, unit)
    return SampleSummary(
        unit=unit,
        sample_count=len(values),
        median=med,
        p90=q90,
        normalized_iqr=normalized_iqr(values),
        median_us=median(micro),
        p90_us=p90(micro),
    )


def summaries_agree(samples: Sequence[float], unit: str, median_us: float, p90_us: float, *, rel_tol: float = 1e-9) -> bool:
    """Check recorded median/p90 (in microseconds) against raw samples."""
    summary = summarize(samples, unit)
    return math.isclose(summary.median_us, median_us, rel_tol=rel_tol, abs_tol=1e-12) and math.isclose(
        summary.p90_us, p90_us, rel_tol=rel_tol, abs_tol=1e-12
    )
