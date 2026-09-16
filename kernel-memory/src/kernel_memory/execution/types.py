"""Execution and orchestration types (specification section 18).

Budgets are configurable defaults, not promises. Proposals carry predictions,
never results. Planners propose; only the Runner produces evidence and only the
decision service derives selections from evidence and a fixed policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..domain.errors import InputError


@dataclass(frozen=True)
class Budget:
    max_candidates: int = 8
    max_execution_attempts: int = 24
    max_model_calls: int = 12
    max_wall_time_seconds: float = 1800.0
    max_concurrent_runners: int = 1
    max_consecutive_execution_failures: int = 3
    plateau_rounds: int = 4  # rounds without a confirmable improvement before stopping

    def validate(self) -> None:
        for name in ("max_candidates", "max_execution_attempts", "max_model_calls", "max_concurrent_runners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InputError(f"budget.{name} must be a non-negative integer")
        if self.max_wall_time_seconds <= 0:
            raise InputError("budget.max_wall_time_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_execution_attempts": self.max_execution_attempts,
            "max_model_calls": self.max_model_calls,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_concurrent_runners": self.max_concurrent_runners,
            "max_consecutive_execution_failures": self.max_consecutive_execution_failures,
            "plateau_rounds": self.plateau_rounds,
        }


@dataclass
class BudgetUsage:
    """Counters restored from the persisted ledger after restart."""

    candidates: int = 0
    execution_attempts: int = 0
    model_calls: int = 0
    wall_time_seconds: float = 0.0
    consecutive_execution_failures: int = 0
    rounds_without_improvement: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "execution_attempts": self.execution_attempts,
            "model_calls": self.model_calls,
            "wall_time_seconds": self.wall_time_seconds,
            "consecutive_execution_failures": self.consecutive_execution_failures,
            "rounds_without_improvement": self.rounds_without_improvement,
        }


@dataclass(frozen=True)
class Proposal:
    """A planner's suggestion. ``predicted_speedup`` is a prediction, never ``result.latency``."""

    proposal_id: str
    parent_ref: str  # commit/baseline the candidate derives from (optimization origin)
    target_component: str
    hypothesis: str
    planned_changes: list[dict]  # Change-like dicts: component, key, before, after, rationale
    risks_to_test: list[str] = field(default_factory=list)
    file_allowlist: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    predicted_speedup: float | None = None
    implementation_overrides: dict = field(default_factory=dict)
    planner_id: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "parent_ref": self.parent_ref,
            "target_component": self.target_component,
            "hypothesis": self.hypothesis,
            "planned_changes": list(self.planned_changes),
            "risks_to_test": list(self.risks_to_test),
            "file_allowlist": list(self.file_allowlist),
            "stop_conditions": list(self.stop_conditions),
            "predicted_speedup": self.predicted_speedup,
            "implementation_overrides": dict(self.implementation_overrides),
            "planner_id": self.planner_id,
        }


@dataclass
class MemoryContext:
    """Exported Memory context handed to a planner (data, not instructions)."""

    config_ref: str
    config_hash: str
    context: dict  # output of services.context.export_context
    round_no: int = 0


@runtime_checkable
class Planner(Protocol):
    planner_id: str
    requires_model_api: bool

    def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
        """Return a proposal or None to stop (plateau / nothing left to try)."""
        ...
