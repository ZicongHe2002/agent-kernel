"""Budget ledger: reserve-before-execute accounting that survives restarts (spec 18, T30).

Public API
----------
``BudgetLedger(store, job_id, budget, *, persist=True)``
    ``usage() -> BudgetUsage``            replayed from the persisted usage files on every call
    ``reserve_execution() -> dict``       raises ``BudgetExhausted`` *before* execution when
                                          ``execution_attempts >= max_execution_attempts``
    ``record_execution_result(success)``  resets/increments consecutive failures; never raises
    ``record_candidate()``, ``record_model_call()``, ``add_wall_time(seconds)``,
    ``record_round(improved)``
    ``stop_reason() -> str | None``       one of ``STOP_REASONS`` or None
    ``check()``                           raises ``BudgetExhausted`` when ``stop_reason()`` is set
    ``entries() -> list[dict]``, ``job_spec() -> dict | None``, ``job_dir() -> str``

Persistence
-----------
* ``requests/_jobs/<job-slug>/job.json``  written once (budget snapshot); never rewritten.
* ``requests/_jobs/<job-slug>/usage/<seq:06d>.json``
  ``{"kind": candidate|execution_reserved|execution_result|model_call|wall_time|round,
     "delta": number, "success": bool|null, "at": timestamp}``
  All writes go through ``MemoryStore.write_fact`` under the store lock (append-only).
* ``persist=False`` keeps the same accounting in memory only (used by dry runs, which
  must not write anything to the store).

Budgets are configurable defaults, not promises; the ledger never invents usage.
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import BudgetExhausted, InputError
from ..domain.ids import utc_now_iso
from ..domain.jsonio import dumps_readable, loads_strict
from ..storage.store import MemoryStore
from .ledger import job_dir as _job_dir
from .types import Budget, BudgetUsage

USAGE_KINDS: tuple[str, ...] = ("candidate", "execution_reserved", "execution_result", "model_call", "wall_time", "round")
STOP_REASONS: tuple[str, ...] = (
    "BUDGET_CANDIDATES_EXHAUSTED",
    "BUDGET_EXECUTIONS_EXHAUSTED",
    "BUDGET_MODEL_CALLS_EXHAUSTED",
    "BUDGET_WALL_TIME_EXHAUSTED",
    "CONSECUTIVE_EXECUTION_FAILURES",
    "PLATEAU",
)
LEDGER_VERSION = 1


def replay_usage(entries: list[dict[str, Any]]) -> BudgetUsage:
    """Pure replay of usage entries into counters (order matters for streak counters)."""
    usage = BudgetUsage()
    for entry in entries:
        kind = entry.get("kind")
        delta = entry.get("delta", 0)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise InputError(f"usage entry has a non-numeric delta: {entry!r}", code="LEDGER_CORRUPT")
        if kind == "candidate":
            usage.candidates += int(delta)
        elif kind == "execution_reserved":
            usage.execution_attempts += int(delta)
        elif kind == "execution_result":
            if entry.get("success") is True:
                usage.consecutive_execution_failures = 0
            else:
                usage.consecutive_execution_failures += 1
        elif kind == "model_call":
            usage.model_calls += int(delta)
        elif kind == "wall_time":
            usage.wall_time_seconds += float(delta)
        elif kind == "round":
            if entry.get("success") is True:
                usage.rounds_without_improvement = 0
            else:
                usage.rounds_without_improvement += 1
        else:
            raise InputError(f"unknown usage entry kind {kind!r}", code="LEDGER_CORRUPT")
    return usage


def stop_reason_for(budget: Budget, usage: BudgetUsage) -> str | None:
    if usage.candidates >= budget.max_candidates:
        return "BUDGET_CANDIDATES_EXHAUSTED"
    if usage.execution_attempts >= budget.max_execution_attempts:
        return "BUDGET_EXECUTIONS_EXHAUSTED"
    if usage.model_calls >= budget.max_model_calls:
        return "BUDGET_MODEL_CALLS_EXHAUSTED"
    if usage.wall_time_seconds >= budget.max_wall_time_seconds:
        return "BUDGET_WALL_TIME_EXHAUSTED"
    if usage.consecutive_execution_failures >= budget.max_consecutive_execution_failures:
        return "CONSECUTIVE_EXECUTION_FAILURES"
    if usage.rounds_without_improvement >= budget.plateau_rounds:
        return "PLATEAU"
    return None


class BudgetLedger:
    def __init__(self, store: MemoryStore, job_id: str, budget: Budget, *, persist: bool = True) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise InputError("job_id must be a non-empty string", code="INVALID_JOB_ID")
        if not isinstance(budget, Budget):
            raise InputError("budget must be a Budget instance", code="INVALID_BUDGET")
        budget.validate()
        self._store = store
        self._job_id = job_id
        self._budget = budget
        self._persist = persist
        self._memory: list[dict[str, Any]] = []
        if persist:
            self._ensure_job_spec()

    # ------------------------------------------------------------------ properties / paths
    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def persistent(self) -> bool:
        return self._persist

    def job_dir(self) -> str:
        return _job_dir(self._job_id)

    def _usage_dir(self) -> str:
        return f"{self.job_dir()}/usage"

    def _job_path(self) -> str:
        return f"{self.job_dir()}/job.json"

    def _ensure_job_spec(self) -> None:
        with self._store.lock():
            if self._store.read_fact(self._job_path()) is None:
                data = {
                    "ledger_version": LEDGER_VERSION,
                    "job_id": self._job_id,
                    "budget": self._budget.to_dict(),
                    "created_at": utc_now_iso(),
                }
                self._store.write_fact(self._job_path(), dumps_readable(data).encode("utf-8"))

    def job_spec(self) -> dict[str, Any] | None:
        raw = self._store.read_fact(self._job_path())
        if raw is None:
            return None
        data = loads_strict(raw)
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------ entries
    def entries(self) -> list[dict[str, Any]]:
        if not self._persist:
            return list(self._memory)
        out: list[tuple[int, dict[str, Any]]] = []
        for rel in self._store.list_facts(self._usage_dir()):
            if not rel.endswith(".json"):
                continue
            raw = self._store.read_fact(rel)
            if raw is None:
                continue
            data = loads_strict(raw)
            if not isinstance(data, dict):
                raise InputError(f"usage file {rel} is not an object", code="LEDGER_CORRUPT", details={"path": rel})
            stem = rel.rsplit("/", 1)[-1][: -len(".json")]
            try:
                seq = int(stem)
            except ValueError as exc:
                raise InputError(f"usage file {rel} has a non-numeric sequence", code="LEDGER_CORRUPT", details={"path": rel}) from exc
            out.append((seq, data))
        out.sort(key=lambda item: item[0])
        return [data for _, data in out]

    def _append(self, kind: str, delta: float | int, success: bool | None) -> dict[str, Any]:
        if kind not in USAGE_KINDS:
            raise InputError(f"unknown usage kind {kind!r}", code="INVALID_USAGE_KIND")
        entry = {"kind": kind, "delta": delta, "success": success, "at": utc_now_iso()}
        if not self._persist:
            self._memory.append(entry)
            return entry
        with self._store.lock():
            sequence = len(self.entries()) + 1
            self._store.write_fact(f"{self._usage_dir()}/{sequence:06d}.json", dumps_readable(entry).encode("utf-8"))
        return entry

    # ------------------------------------------------------------------ accounting
    def usage(self) -> BudgetUsage:
        return replay_usage(self.entries())

    def reserve_execution(self) -> dict[str, Any]:
        with self._store.lock():
            usage = self.usage()
            if usage.execution_attempts >= self._budget.max_execution_attempts:
                raise BudgetExhausted(
                    f"execution budget exhausted: {usage.execution_attempts} of {self._budget.max_execution_attempts} attempts used",
                    details={
                        "limit": "max_execution_attempts",
                        "used": usage.execution_attempts,
                        "max": self._budget.max_execution_attempts,
                        "stop_reason": "BUDGET_EXECUTIONS_EXHAUSTED",
                        "usage": usage.to_dict(),
                    },
                )
            return self._append("execution_reserved", 1, None)

    def record_execution_result(self, success: bool) -> dict[str, Any]:
        return self._append("execution_result", 1, bool(success))

    def record_candidate(self) -> dict[str, Any]:
        return self._append("candidate", 1, None)

    def record_model_call(self) -> dict[str, Any]:
        return self._append("model_call", 1, None)

    def add_wall_time(self, seconds: float) -> dict[str, Any] | None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise InputError("wall time must be a number of seconds", code="INVALID_WALL_TIME")
        if seconds < 0:
            raise InputError("wall time cannot be negative", code="INVALID_WALL_TIME", details={"seconds": seconds})
        if seconds == 0:
            return None
        return self._append("wall_time", float(seconds), None)

    def record_round(self, improved: bool) -> dict[str, Any]:
        return self._append("round", 1, bool(improved))

    # ------------------------------------------------------------------ stop conditions
    def stop_reason(self) -> str | None:
        return stop_reason_for(self._budget, self.usage())

    def check(self) -> None:
        usage = self.usage()
        reason = stop_reason_for(self._budget, usage)
        if reason is not None:
            raise BudgetExhausted(
                f"optimization loop must stop: {reason}",
                details={"stop_reason": reason, "usage": usage.to_dict(), "budget": self._budget.to_dict()},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self._job_id,
            "persistent": self._persist,
            "budget": self._budget.to_dict(),
            "usage": self.usage().to_dict(),
            "stop_reason": self.stop_reason(),
        }


__all__ = ["BudgetLedger", "USAGE_KINDS", "STOP_REASONS", "replay_usage", "stop_reason_for"]
