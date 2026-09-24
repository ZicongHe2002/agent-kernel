"""Host-synchronized sampling primitives (specification section 11; adapters/timing.py).

Every measured number comes from a real ``perf_counter_ns`` difference; tests inject
deterministic counters through the module-level indirections (``timing.perf_counter_ns``,
``timing.wall_clock``) and never assert on wall-clock speed.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pytest

from kernel_memory.adapters import timing
from kernel_memory.domain.errors import ExecutionInfrastructureError, InputError, KernelMemoryError


class FakeCounter:
    """Deterministic nanosecond counter: every read returns the current value; ``advance`` moves it."""

    def __init__(self, start: int = 1_000_000) -> None:
        self.now = start
        self.reads = 0

    def __call__(self) -> int:
        self.reads += 1
        return self.now

    def advance(self, ns: int) -> None:
        self.now += ns


class FakeWall:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _work() -> int:
    total = 0
    for i in range(50):
        total += i * i
    return total


# ------------------------------------------------------------------------------ real measurements
def test_collect_samples_returns_exactly_repetitions_positive_ints() -> None:
    samples = timing.collect_samples_ns(_work, warmup=2, repetitions=7)
    assert isinstance(samples, list)
    assert len(samples) == 7
    assert all(type(s) is int for s in samples)
    assert all(s > 0 for s in samples)


def test_collect_samples_with_zero_warmup_and_one_repetition() -> None:
    samples = timing.collect_samples_ns(_work, warmup=0, repetitions=1)
    assert len(samples) == 1 and samples[0] > 0


def test_samples_are_returned_exactly_as_measured_unfiltered(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)
    increments = iter([1, 10, 20, 30, 5, 40])  # the leading 1 is consumed by the untimed first call

    def call() -> str:
        counter.advance(next(increments))
        return "out"

    samples = timing.collect_samples_ns(call, warmup=0, repetitions=5)
    # No filtering, no sorting, no invented values: the raw per-iteration differences in order.
    assert samples == [10, 20, 30, 5, 40]


# ------------------------------------------------------------------------------ input validation
@pytest.mark.parametrize("warmup", [-1, True, 1.5, "2", None])
def test_invalid_warmup_is_input_error(warmup: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns(_work, warmup=warmup, repetitions=3)
    assert excinfo.value.code == "INVALID_SAMPLING_COUNTS"
    assert excinfo.value.exit_code == 2


@pytest.mark.parametrize("repetitions", [0, -3, True, 2.0, "3", None])
def test_invalid_repetitions_is_input_error(repetitions: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns(_work, warmup=1, repetitions=repetitions)
    assert excinfo.value.code == "INVALID_SAMPLING_COUNTS"
    assert excinfo.value.exit_code == 2


def test_invalid_counts_do_not_execute_the_call() -> None:
    calls = {"n": 0}

    def call() -> None:
        calls["n"] += 1

    with pytest.raises(InputError):
        timing.collect_samples_ns(call, warmup=1, repetitions=0)
    assert calls["n"] == 0


def test_non_callable_arguments_are_input_errors() -> None:
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns("not-callable", warmup=0, repetitions=1)  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_CALL"
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns(_work, warmup=0, repetitions=1, block_until_ready=5)  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_CALL"
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns(_work, warmup=0, repetitions=1, prepare="x")  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_CALL"


@pytest.mark.parametrize("deadline", [0, -1.0, True, "10"])
def test_invalid_max_wall_seconds_is_input_error(deadline: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        timing.collect_samples_ns(_work, warmup=0, repetitions=1, max_wall_seconds=deadline)
    assert excinfo.value.code == "INVALID_DEADLINE"


# ------------------------------------------------------------------------------ prepare hook
def test_prepare_hook_called_once_per_iteration_and_result_passed_to_call() -> None:
    prepared: list[int] = []
    received: list[int] = []
    state = {"n": 0}

    def prepare() -> int:
        state["n"] += 1
        prepared.append(state["n"])
        return state["n"]

    def call(arg: int) -> int:
        received.append(arg)
        return arg * 2

    warmup, repetitions = 3, 4
    samples = timing.collect_samples_ns(call, warmup=warmup, repetitions=repetitions, prepare=prepare)
    assert len(samples) == repetitions
    # first (untimed) call + warmups + measured repetitions, each with its own fresh state
    assert prepared == list(range(1, 1 + warmup + repetitions + 1))
    assert received == prepared


def test_prepare_excluded_from_timing_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)

    def prepare() -> str:
        counter.advance(1000)  # expensive preparation
        return "state"

    def call(state: str) -> str:
        assert state == "state"
        counter.advance(10)
        return "out"

    samples = timing.collect_samples_ns(call, warmup=1, repetitions=4, prepare=prepare)
    assert samples == [10, 10, 10, 10]


def test_prepare_included_when_time_prepare_true(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)

    def prepare() -> str:
        counter.advance(1000)
        return "state"

    def call(state: str) -> str:
        counter.advance(10)
        return "out"

    samples = timing.collect_samples_ns(call, warmup=1, repetitions=3, prepare=prepare, time_prepare=True)
    assert samples == [1010, 1010, 1010]


def test_time_prepare_without_prepare_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)

    def call() -> None:
        counter.advance(7)

    assert timing.collect_samples_ns(call, warmup=0, repetitions=2, time_prepare=True) == [7, 7]


# ------------------------------------------------------------------------------ block_until_ready
def test_block_until_ready_called_on_every_output_inside_timed_region(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)
    outputs: list[object] = []
    produced: list[object] = []

    def call() -> object:
        out = object()
        produced.append(out)
        counter.advance(5)
        return out

    def block(out: object) -> None:
        outputs.append(out)
        counter.advance(100)  # asynchronous completion is part of the measurement

    warmup, repetitions = 2, 3
    samples = timing.collect_samples_ns(call, warmup=warmup, repetitions=repetitions, block_until_ready=block)
    assert len(outputs) == 1 + warmup + repetitions
    assert outputs == produced  # exactly the objects the call returned, in order
    assert samples == [105, 105, 105]  # synchronisation time is inside the sample


def test_without_block_until_ready_only_the_call_is_timed(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)

    def call() -> int:
        counter.advance(42)
        return 1

    assert timing.collect_samples_ns(call, warmup=0, repetitions=2) == [42, 42]


# ------------------------------------------------------------------------------ unusable timer
def test_non_advancing_perf_counter_ns_is_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 123_456_789)
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        timing.collect_samples_ns(_work, warmup=1, repetitions=3)
    err = excinfo.value
    assert err.code == "NON_POSITIVE_ELAPSED"
    assert err.exit_code == 6
    assert err.details["elapsed_ns"] == 0
    assert err.details["samples_collected"] == 0


def test_timer_going_backwards_is_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = FakeCounter()
    monkeypatch.setattr(timing, "perf_counter_ns", counter)
    steps = iter([1, 10, 10, -3])  # first (untimed) call, two good samples, then the clock goes backwards

    def call() -> None:
        counter.advance(next(steps))

    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        timing.collect_samples_ns(call, warmup=0, repetitions=5)
    assert excinfo.value.code == "NON_POSITIVE_ELAPSED"
    assert excinfo.value.details == {"elapsed_ns": -3, "samples_collected": 2}


# ------------------------------------------------------------------------------ deadline
def test_exceeding_max_wall_seconds_between_iterations_is_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    wall = FakeWall()
    monkeypatch.setattr(timing, "wall_clock", wall)
    calls = {"n": 0}

    def call() -> None:
        calls["n"] += 1
        wall.advance(0.4)

    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        timing.collect_samples_ns(call, warmup=1, repetitions=10, max_wall_seconds=1.0)
    err = excinfo.value
    assert err.code == "TIMEOUT" and err.exit_code == 6
    assert err.details["max_wall_seconds"] == 1.0
    assert err.details["phase"] == "repetitions"
    assert err.details["samples_collected"] == 1
    assert err.details["elapsed_seconds"] == pytest.approx(1.2)
    assert calls["n"] == 3  # first call, one warmup, one measured repetition; then refused


def test_deadline_during_warmup_reports_warmup_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    wall = FakeWall()
    monkeypatch.setattr(timing, "wall_clock", wall)

    def call() -> None:
        wall.advance(3.0)

    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        timing.collect_samples_ns(call, warmup=2, repetitions=2, max_wall_seconds=2.0)
    assert excinfo.value.code == "TIMEOUT"
    assert excinfo.value.details["phase"] == "first_call"
    assert excinfo.value.details["samples_collected"] == 0


def test_deadline_not_exceeded_returns_all_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    wall = FakeWall()
    monkeypatch.setattr(timing, "wall_clock", wall)

    def call() -> None:
        wall.advance(0.01)

    samples = timing.collect_samples_ns(call, warmup=1, repetitions=5, max_wall_seconds=10.0)
    assert len(samples) == 5


def test_no_deadline_means_no_wall_clock_check(monkeypatch: pytest.MonkeyPatch) -> None:
    wall = FakeWall()
    monkeypatch.setattr(timing, "wall_clock", wall)

    def call() -> None:
        wall.advance(1e6)

    assert len(timing.collect_samples_ns(call, warmup=0, repetitions=3, max_wall_seconds=None)) == 3


# ------------------------------------------------------------------------------ host timer environment
def test_host_timer_environment_names_timer_and_resolution() -> None:
    env = timing.host_timer_environment()
    assert env["timer"] == timing.TIMER_NAME == "time.perf_counter_ns"
    assert isinstance(env["perf_counter_resolution_ns"], float)
    assert env["perf_counter_resolution_ns"] > 0
    assert env["perf_counter_resolution_ns"] == pytest.approx(time.get_clock_info("perf_counter").resolution * 1e9)
    assert isinstance(env["platform"], str) and env["platform"]
    assert isinstance(env["python_version"], str) and env["python_version"]
    assert "cpu_count" in env and "process_priority_hint" in env
    json.dumps(env)  # must be serialisable into the Environment snapshot


def test_host_timer_environment_is_stable_across_calls() -> None:
    assert timing.host_timer_environment() == timing.host_timer_environment()


# ------------------------------------------------------------------------------ samples artifact
def test_samples_artifact_bytes_has_unit_samples_count_and_is_fixture_false() -> None:
    samples = [1200, 1100, 1300, 1000, 1150]
    data = timing.samples_artifact_bytes(samples, "nanoseconds", timing_method="host_synchronized", protocol_id="p1")
    assert isinstance(data, bytes)
    doc = json.loads(data.decode("utf-8"))
    assert doc["unit"] == "nanoseconds"
    assert doc["samples"] == samples  # retained unmodified, unsorted
    assert doc["count"] == 5
    assert doc["is_fixture"] is False
    assert doc["timing_method"] == "host_synchronized"
    assert doc["protocol_id"] == "p1"


def test_samples_artifact_accepts_every_supported_unit() -> None:
    for unit in ("nanoseconds", "microseconds", "milliseconds", "seconds"):
        doc = json.loads(timing.samples_artifact_bytes([1, 2.5], unit))
        assert doc["unit"] == unit and doc["count"] == 2


def test_samples_artifact_rejects_unknown_unit() -> None:
    with pytest.raises(InputError) as excinfo:
        timing.samples_artifact_bytes([1, 2, 3], "cycles")
    assert excinfo.value.code == "UNKNOWN_UNIT"


@pytest.mark.parametrize("bad", [[], [0], [-5], [1, float("nan")], [1, float("inf")], [1, "2"], [True]])
def test_samples_artifact_rejects_invalid_samples(bad: list[Any]) -> None:
    with pytest.raises(InputError) as excinfo:
        timing.samples_artifact_bytes(bad, "nanoseconds")
    assert excinfo.value.code == "INVALID_SAMPLES"


@pytest.mark.parametrize("key", ["count", "is_fixture"])
def test_samples_artifact_reserved_metadata_keys_are_refused(key: str) -> None:
    # "unit" and "samples" are positional parameters, so Python itself refuses a duplicate
    # keyword (TypeError) before the metadata check; the reachable reserved keys are these two.
    with pytest.raises(InputError) as excinfo:
        timing.samples_artifact_bytes([1, 2], "nanoseconds", **{key: "override"})
    assert excinfo.value.code == "RESERVED_KEY"


def test_samples_artifact_cannot_be_relabelled_as_fixture() -> None:
    with pytest.raises(InputError):
        timing.samples_artifact_bytes([1, 2], "nanoseconds", is_fixture=True)


def test_samples_artifact_non_serialisable_metadata_is_input_error() -> None:
    with pytest.raises(InputError) as excinfo:
        timing.samples_artifact_bytes([1, 2], "nanoseconds", handle=object())
    assert excinfo.value.code == "INVALID_ARTIFACT_META"
    assert isinstance(excinfo.value, KernelMemoryError)
