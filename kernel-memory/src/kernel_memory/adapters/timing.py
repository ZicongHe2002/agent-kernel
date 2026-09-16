"""Host-synchronized latency sampling shared by every kernel adapter (specification section 11).

Public API
----------
``collect_samples_ns(call, *, warmup, repetitions, block_until_ready=None, prepare=None,
time_prepare=False, max_wall_seconds=None) -> list[int]``
    Real measurement loop. The first call (compile / first execution) and ``warmup`` further
    calls happen outside the measured region; then ``repetitions`` samples are taken with
    ``time.perf_counter_ns`` around ``call`` plus ``block_until_ready(output)``. When
    ``prepare`` is given it runs once per iteration and its return value is passed to
    ``call`` as the single positional argument; it is inside the timed region only when
    ``time_prepare`` is true (the adapter fixes this explicitly, see section 11.1).
    Invalid counts raise ``InputError``; a non-positive elapsed time raises
    ``ExecutionInfrastructureError`` (code ``NON_POSITIVE_ELAPSED``); exceeding
    ``max_wall_seconds`` between iterations raises ``ExecutionInfrastructureError`` with
    code ``TIMEOUT``. Samples are returned exactly as measured (integers, nanoseconds);
    no filtering happens here.

``host_timer_environment() -> dict``
    Facts about the host timer that belong in ``Environment.host_timer_environment``.

``samples_artifact_bytes(samples, unit, **meta) -> bytes``
    JSON bytes for a ``latency_samples`` artifact: ``{"unit", "samples", "count", ...meta,
    "is_fixture": false}``. Raw samples are retained unmodified.

Guarantees: nothing in this module invents measurements; every sample comes from a real
``perf_counter_ns`` difference. ``wall_clock`` and ``perf_counter_ns`` are module-level
indirections so tests can inject clocks without touching the standard library.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Callable, Sequence

from ..domain.errors import ExecutionInfrastructureError, InputError
from ..domain.jsonio import dumps_readable
from ..domain.stats import SUPPORTED_UNITS, validate_samples

TIMER_NAME = "time.perf_counter_ns"
RESERVED_SAMPLE_KEYS = frozenset({"unit", "samples", "count", "is_fixture"})


def perf_counter_ns() -> int:
    """Monotonic high-resolution counter in nanoseconds (indirection for tests)."""
    return time.perf_counter_ns()


def wall_clock() -> float:
    """Monotonic wall clock in seconds used for deadline checks (indirection for tests)."""
    return time.monotonic()


def _validate_count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{name} must be an integer, got {value!r}", code="INVALID_SAMPLING_COUNTS")
    if value < minimum:
        raise InputError(f"{name} must be >= {minimum}, got {value}", code="INVALID_SAMPLING_COUNTS")
    return value


def collect_samples_ns(
    call: Callable[..., Any],
    *,
    warmup: int,
    repetitions: int,
    block_until_ready: Callable[[Any], Any] | None = None,
    prepare: Callable[[], Any] | None = None,
    time_prepare: bool = False,
    max_wall_seconds: float | None = None,
) -> list[int]:
    warmup = _validate_count(warmup, "warmup", minimum=0)
    repetitions = _validate_count(repetitions, "repetitions", minimum=1)
    if not callable(call):
        raise InputError("call must be callable", code="INVALID_CALL")
    if block_until_ready is not None and not callable(block_until_ready):
        raise InputError("block_until_ready must be callable or None", code="INVALID_CALL")
    if prepare is not None and not callable(prepare):
        raise InputError("prepare must be callable or None", code="INVALID_CALL")
    if max_wall_seconds is not None:
        if isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float)) or not max_wall_seconds > 0:
            raise InputError(f"max_wall_seconds must be a positive number or None, got {max_wall_seconds!r}", code="INVALID_DEADLINE")

    started_wall = wall_clock()
    samples: list[int] = []

    def check_deadline(phase: str) -> None:
        if max_wall_seconds is None:
            return
        elapsed = wall_clock() - started_wall
        if elapsed > max_wall_seconds:
            raise ExecutionInfrastructureError(
                f"benchmark exceeded max_wall_seconds={max_wall_seconds} during {phase} after {elapsed:.3f}s",
                code="TIMEOUT",
                details={
                    "phase": phase,
                    "elapsed_seconds": elapsed,
                    "max_wall_seconds": max_wall_seconds,
                    "samples_collected": len(samples),
                },
            )

    def invoke_untimed() -> None:
        if prepare is not None:
            output = call(prepare())
        else:
            output = call()
        if block_until_ready is not None:
            block_until_ready(output)

    # First execution (may include compilation / first dispatch) is never measured.
    invoke_untimed()
    check_deadline("first_call")
    for _ in range(warmup):
        invoke_untimed()
        check_deadline("warmup")

    for _ in range(repetitions):
        if prepare is not None and not time_prepare:
            state = prepare()
            start = perf_counter_ns()
            output = call(state)
        elif prepare is not None:
            start = perf_counter_ns()
            output = call(prepare())
        else:
            start = perf_counter_ns()
            output = call()
        if block_until_ready is not None:
            block_until_ready(output)
        elapsed = perf_counter_ns() - start
        if elapsed <= 0:
            raise ExecutionInfrastructureError(
                f"non-positive elapsed time {elapsed} ns; the host timer is unusable for this measurement",
                code="NON_POSITIVE_ELAPSED",
                details={"elapsed_ns": int(elapsed), "samples_collected": len(samples)},
            )
        samples.append(int(elapsed))
        check_deadline("repetitions")
    return samples


def host_timer_environment() -> dict[str, Any]:
    info = time.get_clock_info("perf_counter")
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "perf_counter_resolution_ns": float(info.resolution) * 1e9,
        "cpu_count": os.cpu_count(),
        "process_priority_hint": "unknown",
        "timer": TIMER_NAME,
    }


def samples_artifact_bytes(samples: Sequence[int | float], unit: str, **meta: Any) -> bytes:
    if unit not in SUPPORTED_UNITS:
        raise InputError(f"unknown sample unit: {unit!r}", code="UNKNOWN_UNIT")
    validate_samples(list(samples))
    clash = RESERVED_SAMPLE_KEYS.intersection(meta)
    if clash:
        raise InputError(f"metadata keys {sorted(clash)} are reserved in a samples artifact", code="RESERVED_KEY")
    document: dict[str, Any] = {"unit": unit, "samples": list(samples), "count": len(samples)}
    document.update(meta)
    document["is_fixture"] = False
    try:
        return dumps_readable(document).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputError(f"samples artifact metadata is not JSON-serialisable: {exc}", code="INVALID_ARTIFACT_META") from exc


__all__ = [
    "TIMER_NAME",
    "collect_samples_ns",
    "host_timer_environment",
    "perf_counter_ns",
    "samples_artifact_bytes",
    "wall_clock",
]
