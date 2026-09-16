"""Planners: propose candidates; never produce evidence (specification section 18).

Public API
----------
``MockPlanner(planner_id="mock-v1", requires_model_api=False, *, overrides_sequence=None)``
    Proposes runtime-override variants (``implementation_overrides``) for the CPU demo
    from a fixed sequence, default ``[{"chunk": 8}, {"chunk": 64}, {"chunk": 512}]``.
    ``propose(context, budget, usage)`` returns ``None`` when the sequence is exhausted
    or ``usage.rounds_without_improvement >= budget.plateau_rounds``. The proposal index
    is ``context.round_no - 1`` when the coordinator supplies a positive round number
    (so a resumed job continues where it stopped) and an internal cursor otherwise.
    ``predicted_speedup`` is always ``None``: the mock never fabricates a prediction.
    The parent reference comes from ``context.context["current_baselines"][0]`` when
    present, else ``context.context["default_parent_ref"]``.
    MockPlanner exercises orchestration only; its results are not evidence of
    optimization effectiveness.
``UnavailableModelPlanner(planner_id="model-unavailable", requires_model_api=True)``
    ``propose`` raises ``PrerequisiteMissingError`` with code
    ``MODEL_PROVIDER_NOT_CONFIGURED`` naming the provider/credential environment
    variables and the ``allow_model_api_calls`` authorization that would be required.
``MODEL_PROVIDER_ENV``, ``MODEL_API_KEY_ENV``  environment variable names a future
    provider integration reads credentials from (never Memory).
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import InputError, PrerequisiteMissingError
from ..domain.hashing import jcs_digest
from .types import Budget, BudgetUsage, MemoryContext, Proposal

DEFAULT_OVERRIDES_SEQUENCE: tuple[dict[str, Any], ...] = ({"chunk": 8}, {"chunk": 64}, {"chunk": 512})
MODEL_PROVIDER_ENV = "KMEM_MODEL_PROVIDER"
MODEL_API_KEY_ENV = "KMEM_MODEL_API_KEY"
OVERRIDE_COMPONENT = "implementation_overrides"


def parent_ref_from_context(context: MemoryContext) -> str | None:
    data = context.context if isinstance(context.context, dict) else {}
    baselines = data.get("current_baselines")
    if isinstance(baselines, list) and baselines:
        first = baselines[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            for key in ("baseline_ref", "record_id", "subject_ref", "ref"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
    default = data.get("default_parent_ref")
    if isinstance(default, str) and default:
        return default
    return None


class MockPlanner:
    def __init__(
        self,
        planner_id: str = "mock-v1",
        requires_model_api: bool = False,
        *,
        overrides_sequence: list[dict[str, Any]] | None = None,
    ) -> None:
        sequence = list(overrides_sequence) if overrides_sequence is not None else [dict(o) for o in DEFAULT_OVERRIDES_SEQUENCE]
        for item in sequence:
            if not isinstance(item, dict) or not item:
                raise InputError("overrides_sequence entries must be non-empty objects", code="INVALID_OVERRIDES")
            for key in item:
                if not isinstance(key, str) or not key:
                    raise InputError("override keys must be non-empty strings", code="INVALID_OVERRIDES")
        self.planner_id = planner_id
        self.requires_model_api = bool(requires_model_api)
        self._sequence = sequence
        self._cursor = 0

    @property
    def sequence(self) -> list[dict[str, Any]]:
        return [dict(o) for o in self._sequence]

    def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
        if usage.rounds_without_improvement >= budget.plateau_rounds:
            return None
        index = context.round_no - 1 if isinstance(context.round_no, int) and context.round_no >= 1 else self._cursor
        if index >= len(self._sequence):
            return None
        self._cursor = index + 1
        overrides = dict(self._sequence[index])
        parent_ref = parent_ref_from_context(context)
        if parent_ref is None:
            raise InputError(
                "planner context provides neither current_baselines nor default_parent_ref",
                code="PLANNER_CONTEXT_INCOMPLETE",
                details={"config_ref": context.config_ref, "round_no": context.round_no},
            )
        changes = [
            {
                "component": OVERRIDE_COMPONENT,
                "key": key,
                "before": None,
                "after": value,
                "rationale": "mock planner sweep",
            }
            for key, value in overrides.items()
        ]
        summary = ", ".join(f"{k}={v!r}" for k, v in overrides.items())
        proposal_id = "proposal-" + jcs_digest({"planner_id": self.planner_id, "parent_ref": parent_ref, "overrides": overrides})[7:23]
        return Proposal(
            proposal_id=proposal_id,
            parent_ref=parent_ref,
            target_component=OVERRIDE_COMPONENT,
            hypothesis=(
                f"Runtime override {summary} on {parent_ref} may change loop granularity; "
                "the effect is unknown until measured under the fixed protocol."
            ),
            planned_changes=changes,
            risks_to_test=["correctness under the override", "timing variability across sessions"],
            file_allowlist=[],
            stop_conditions=["budget exhausted", "plateau without confirmable improvement", "correctness failure", "user cancellation"],
            predicted_speedup=None,
            implementation_overrides=overrides,
            planner_id=self.planner_id,
        )


class UnavailableModelPlanner:
    def __init__(self, planner_id: str = "model-unavailable", requires_model_api: bool = True) -> None:
        self.planner_id = planner_id
        self.requires_model_api = bool(requires_model_api)

    def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
        raise PrerequisiteMissingError(
            "no model provider is configured; model-backed planning is unexecuted",
            code="MODEL_PROVIDER_NOT_CONFIGURED",
            details={
                "planner_id": self.planner_id,
                "needs": ["a model provider selected via environment", "provider credentials via environment", "authorization"],
                "environment_variables": [MODEL_PROVIDER_ENV, MODEL_API_KEY_ENV],
                "authorization": "allow_model_api_calls",
                "config_ref": context.config_ref,
                "round_no": context.round_no,
            },
        )


__all__ = [
    "MockPlanner",
    "UnavailableModelPlanner",
    "DEFAULT_OVERRIDES_SEQUENCE",
    "MODEL_PROVIDER_ENV",
    "MODEL_API_KEY_ENV",
    "OVERRIDE_COMPONENT",
    "parent_ref_from_context",
]
