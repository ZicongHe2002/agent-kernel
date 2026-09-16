"""Budget ledger: reserve-before-execute accounting that survives restarts (T30)."""
from __future__ import annotations

import json

import pytest

from kernel_memory.domain.errors import BudgetExhausted, InputError
from kernel_memory.execution.budget import STOP_REASONS, BudgetLedger, replay_usage, stop_reason_for
from kernel_memory.execution.types import Budget, BudgetUsage
from kernel_memory.storage import MemoryStore


def test_t30_usage_replays_identically_in_a_fresh_instance(store: MemoryStore) -> None:
    budget = Budget(max_candidates=5, max_execution_attempts=6, max_model_calls=4, max_wall_time_seconds=100.0)
    ledger = BudgetLedger(store, "job-1", budget)
    ledger.record_candidate()
    ledger.record_candidate()
    ledger.reserve_execution()
    ledger.record_execution_result(False)
    ledger.reserve_execution()
    ledger.record_execution_result(True)
    ledger.record_model_call()
    ledger.add_wall_time(12.5)
    ledger.record_round(False)
    ledger.record_round(False)
    first = ledger.usage()
    assert first == BudgetUsage(
        candidates=2,
        execution_attempts=2,
        model_calls=1,
        wall_time_seconds=12.5,
        consecutive_execution_failures=0,
        rounds_without_improvement=2,
    )
    # Restart: a new instance for the same job restores exactly the same counters from the files.
    restored = BudgetLedger(store, "job-1", budget)
    assert restored.usage() == first
    assert restored.stop_reason() is None
    assert len(restored.entries()) == 10
    files = [f for f in store.list_facts("requests/_jobs") if "/usage/" in f]
    assert len(files) == 10
    assert files[0].endswith("/usage/000001.json")
    entry = json.loads(store.read_fact(files[0]))
    assert entry["kind"] == "candidate" and entry["delta"] == 1 and entry["success"] is None and "at" in entry
    # Continuing after the restart appends, never double counts.
    restored.record_candidate()
    assert restored.usage().candidates == 3
    assert BudgetLedger(store, "job-1", budget).usage().candidates == 3


def test_job_spec_is_written_once(store: MemoryStore) -> None:
    budget = Budget(max_candidates=2)
    ledger = BudgetLedger(store, "job-1", budget)
    spec = ledger.job_spec()
    assert spec is not None and spec["job_id"] == "job-1" and spec["budget"]["max_candidates"] == 2
    # A second instance with a different budget object does not rewrite the fact file.
    again = BudgetLedger(store, "job-1", Budget(max_candidates=9))
    assert again.job_spec() == spec
    assert ledger.job_dir() == "requests/_jobs/job-1"


def test_reserve_execution_beyond_max_raises_before_execution(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-1", Budget(max_execution_attempts=2))
    ledger.reserve_execution()
    ledger.reserve_execution()
    with pytest.raises(BudgetExhausted) as excinfo:
        ledger.reserve_execution()
    err = excinfo.value
    assert err.code == "BUDGET_EXHAUSTED"
    assert err.exit_code == 0
    assert err.details["limit"] == "max_execution_attempts"
    assert err.details["used"] == 2 and err.details["max"] == 2
    assert err.details["stop_reason"] == "BUDGET_EXECUTIONS_EXHAUSTED"
    # The refused reservation is not counted.
    assert ledger.usage().execution_attempts == 2
    assert ledger.stop_reason() == "BUDGET_EXECUTIONS_EXHAUSTED"
    with pytest.raises(BudgetExhausted):
        ledger.check()


def test_consecutive_execution_failures_stop_and_reset_on_success(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-1", Budget(max_consecutive_execution_failures=2))
    ledger.record_execution_result(False)
    assert ledger.stop_reason() is None
    ledger.record_execution_result(True)
    assert ledger.usage().consecutive_execution_failures == 0
    ledger.record_execution_result(False)
    ledger.record_execution_result(False)
    assert ledger.usage().consecutive_execution_failures == 2
    assert ledger.stop_reason() == "CONSECUTIVE_EXECUTION_FAILURES"
    with pytest.raises(BudgetExhausted) as excinfo:
        ledger.check()
    assert excinfo.value.details["stop_reason"] == "CONSECUTIVE_EXECUTION_FAILURES"
    assert excinfo.value.details["usage"]["consecutive_execution_failures"] == 2


def test_plateau_stops_and_improvement_resets(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-1", Budget(plateau_rounds=2))
    ledger.record_round(False)
    assert ledger.stop_reason() is None
    ledger.record_round(True)
    assert ledger.usage().rounds_without_improvement == 0
    ledger.record_round(False)
    ledger.record_round(False)
    assert ledger.stop_reason() == "PLATEAU"
    with pytest.raises(BudgetExhausted):
        ledger.check()


def test_wall_time_budget(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-1", Budget(max_wall_time_seconds=10.0))
    assert ledger.add_wall_time(0) is None  # nothing to record
    ledger.add_wall_time(4.0)
    ledger.add_wall_time(5.5)
    assert ledger.usage().wall_time_seconds == pytest.approx(9.5)
    assert ledger.stop_reason() is None
    ledger.add_wall_time(0.5)
    assert ledger.stop_reason() == "BUDGET_WALL_TIME_EXHAUSTED"
    with pytest.raises(InputError):
        ledger.add_wall_time(-1)
    with pytest.raises(InputError):
        ledger.add_wall_time("3")  # type: ignore[arg-type]


def test_candidate_and_model_call_budgets(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-1", Budget(max_candidates=1, max_model_calls=1))
    assert ledger.stop_reason() is None
    ledger.record_model_call()
    assert ledger.stop_reason() == "BUDGET_MODEL_CALLS_EXHAUSTED"
    ledger.record_candidate()
    # Candidate exhaustion is reported first (deterministic order of STOP_REASONS).
    assert ledger.stop_reason() == "BUDGET_CANDIDATES_EXHAUSTED"
    assert ledger.stop_reason() in STOP_REASONS
    summary = ledger.to_dict()
    assert summary["usage"]["candidates"] == 1 and summary["stop_reason"] == "BUDGET_CANDIDATES_EXHAUSTED"


def test_dry_run_ledger_keeps_usage_in_memory_only(store: MemoryStore) -> None:
    ledger = BudgetLedger(store, "job-dry", Budget(max_execution_attempts=1), persist=False)
    ledger.record_candidate()
    ledger.reserve_execution()
    ledger.record_round(False)
    assert ledger.usage().candidates == 1 and ledger.usage().execution_attempts == 1
    with pytest.raises(BudgetExhausted):
        ledger.reserve_execution()
    assert store.list_facts("requests") == []
    assert ledger.job_spec() is None
    # A fresh persistent instance for the same id sees nothing: dry runs leave no trace.
    assert BudgetLedger(store, "job-dry", Budget()).usage() == BudgetUsage()


def test_constructor_validates_inputs(store: MemoryStore) -> None:
    with pytest.raises(InputError):
        BudgetLedger(store, "", Budget())
    with pytest.raises(InputError):
        BudgetLedger(store, "job", {"max_candidates": 1})  # type: ignore[arg-type]
    with pytest.raises(InputError):
        BudgetLedger(store, "job", Budget(max_candidates=-1))


def test_replay_rejects_corrupt_entries() -> None:
    with pytest.raises(InputError) as excinfo:
        replay_usage([{"kind": "candidate", "delta": "1"}])
    assert excinfo.value.code == "LEDGER_CORRUPT"
    with pytest.raises(InputError):
        replay_usage([{"kind": "unknown", "delta": 1}])
    assert stop_reason_for(Budget(), BudgetUsage()) is None
