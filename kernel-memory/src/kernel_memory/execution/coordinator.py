"""Optimization coordinator: the budgeted propose -> execute -> compare -> decide loop.

Specification section 18 and acceptance scenario T30. The coordinator never alters
the verifier or protocol between rounds (they are fixed by ``request_factory`` for the
whole job), reserves budget before every execution, records every refusal, persists
each round so a restart continues from the next round with restored accounting, and
stops finitely on exhausted budgets, consecutive failures, plateau, planner
exhaustion, a missing backend, or user cancellation.

Public API
----------
``OptimizationCoordinator(store, *, runner, planner, budget, policy, job_id, config_ref,
                          permissions, request_factory, baseline_run_ref,
                          comparator=None, decider=None, context_exporter=None,
                          annotator=None, subject_ref=None, clock=time.monotonic,
                          dry_run=False)``
    ``run() -> JobReport``  status ``completed | stopped | cancelled | dry_run``.
    Injected collaborators (all optional; production defaults lazily import the
    service modules by path at call time so wiring works once they exist):
      * ``comparator(store, candidate_run_id, baseline_run_id) -> dict-like``
        default: ``kernel_memory.services.compare.compare_in_store``
      * ``decider(store, *, config_ref, candidate_subject_ref, candidate_run_refs,
                  baseline_run_refs, policy, comparison_key) -> Record | str | dict``
        default: ``kernel_memory.services.decide.evaluate_candidate`` + ``append_decision``
      * ``context_exporter(store, config_ref) -> dict-like | None``
        default: ``kernel_memory.services.context.export_context``; a missing module
        yields a minimal local context (baselines of the config, run count).
      * ``annotator(store, *, target_ref, category, text, author_kind, evidence_refs,
                    confidence) -> Record | str``  default: a program-authored
        ``annotation`` record (category ``note``, author ``program``, confidence
        ``unverified``) built with ``services.common.new_record`` and published directly.
      * ``request_factory(proposal, round_no, subject_ref) -> RunRequestSpec`` (required).
    Collaborator failures never crash the loop: an unavailable comparator/decider/annotator
    is recorded in the round (``error`` / ``notes``) and the job continues.
    Budget accounting per round: ``record_candidate`` after validation,
    ``reserve_execution`` before the runner is invoked, ``record_execution_result`` for
    every executed round (success = ``succeeded`` + correctness ``pass``), ``add_wall_time``
    for runner occupancy, and ``record_round(improved)`` where ``improved`` is an
    ``accepted`` decision outcome.
    Proposal validation: only ``implementation_overrides`` changes with an empty
    ``file_allowlist`` may execute; anything else is recorded as a blocked round with
    ``AuthorizationError`` code ``CANDIDATE_CODE_WRITE_NOT_AUTHORIZED`` (or, when the
    permission is granted, ``CANDIDATE_CODE_WRITE_UNSUPPORTED`` because this P0
    coordinator has no isolated workspace) and the loop continues.
``cancel_job(store, job_id, reason=None) -> str``  writes ``.runtime/jobs/<job-slug>/cancel``
``clear_cancel(store, job_id) -> str``
``load_job_rounds(store, job_id) -> list[dict]``  persisted rounds (``requests/_jobs/<job-slug>/rounds/``)
``resolve_policy(policy) -> dict``  default policy from ``services.policy.default_policy`` when
    available, else the golden ``promotion_policy`` vector from the contracts.
``JobReport``, ``RoundReport``  dataclasses with ``to_dict()``.

Dry runs (``dry_run=True``) write nothing to the store: budget accounting stays in memory
and rounds carry ``request_id=None``.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..adapters.base import RunRequestSpec
from ..domain.errors import (
    AuthorizationError,
    BackendUnavailable,
    BudgetExhausted,
    InputError,
    KernelMemoryError,
    LeaseLostError,
    PrerequisiteMissingError,
)
from ..domain.ids import slug_for_id, utc_now_iso
from ..domain.jsonio import dumps_readable, loads_strict
from ..domain.models import Record, to_json
from ..domain.schema import golden_hash_vectors
from ..services.common import baselines_for_config, new_record, runs_for_subject
from ..storage import layout
from ..storage.store import MemoryStore
from .budget import BudgetLedger
from .ledger import job_dir
from .planner import OVERRIDE_COMPONENT
from .runner import LocalRunner
from .types import Budget, MemoryContext, Proposal

RequestFactory = Callable[[Proposal, int, str], RunRequestSpec]
ALLOWED_OVERRIDE_COMPONENTS: frozenset[str] = frozenset({OVERRIDE_COMPONENT})
JOB_STATUSES: tuple[str, ...] = ("completed", "stopped", "cancelled", "dry_run")


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------
@dataclass
class RoundReport:
    round_no: int
    proposal: dict
    request_id: str | None
    run_ref: str | None
    comparison: dict | None
    decision_ref: str | None
    annotation_ref: str | None
    outcome_note: str
    improved: bool = False
    error: dict | None = None
    execution_status: str | None = None
    notes: list[str] = field(default_factory=list)
    at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "proposal": dict(self.proposal),
            "request_id": self.request_id,
            "run_ref": self.run_ref,
            "comparison": dict(self.comparison) if isinstance(self.comparison, dict) else self.comparison,
            "decision_ref": self.decision_ref,
            "annotation_ref": self.annotation_ref,
            "outcome_note": self.outcome_note,
            "improved": self.improved,
            "error": self.error,
            "execution_status": self.execution_status,
            "notes": list(self.notes),
            "at": self.at,
        }


@dataclass
class JobReport:
    job_id: str
    config_ref: str
    status: str
    stop_reason: str | None
    rounds: list[RoundReport]
    usage: dict
    budget: dict
    persisted_state_path: str | None
    planner_id: str
    dry_run: bool
    resumed_rounds: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "config_ref": self.config_ref,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "rounds": [r.to_dict() for r in self.rounds],
            "usage": dict(self.usage),
            "budget": dict(self.budget),
            "persisted_state_path": self.persisted_state_path,
            "planner_id": self.planner_id,
            "dry_run": self.dry_run,
            "resumed_rounds": self.resumed_rounds,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------
# Job-level helpers (cancel flag, persisted rounds, policy)
# --------------------------------------------------------------------------------------
def _runtime_job_dir(job_id: str) -> str:
    return f"{layout.RUNTIME_DIR}/jobs/{slug_for_id(job_id)}"


def cancel_flag_path(job_id: str) -> str:
    return f"{_runtime_job_dir(job_id)}/cancel"


def state_path(job_id: str) -> str:
    return f"{_runtime_job_dir(job_id)}/state.json"


def rounds_dir(job_id: str) -> str:
    return f"{job_dir(job_id)}/rounds"


def cancel_job(store: MemoryStore, job_id: str, reason: str | None = None) -> str:
    path = cancel_flag_path(job_id)
    data = {"job_id": job_id, "cancelled": True, "reason": reason, "at": utc_now_iso()}
    store.write_runtime_state(path, dumps_readable(data).encode("utf-8"))
    return path


def clear_cancel(store: MemoryStore, job_id: str) -> str:
    path = cancel_flag_path(job_id)
    data = {"job_id": job_id, "cancelled": False, "reason": None, "at": utc_now_iso()}
    store.write_runtime_state(path, dumps_readable(data).encode("utf-8"))
    return path


def cancel_requested(store: MemoryStore, job_id: str) -> bool:
    raw = store.read_fact(cancel_flag_path(job_id))
    if raw is None:
        return False
    try:
        data = loads_strict(raw)
    except KernelMemoryError:
        return True  # an unreadable flag is treated as a cancellation request, never ignored
    if isinstance(data, dict):
        return bool(data.get("cancelled", True))
    return True


def load_job_rounds(store: MemoryStore, job_id: str) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for rel in store.list_facts(rounds_dir(job_id)):
        if not rel.endswith(".json"):
            continue
        raw = store.read_fact(rel)
        if raw is None:
            continue
        data = loads_strict(raw)
        if isinstance(data, dict) and isinstance(data.get("round_no"), int):
            rounds.append(data)
    rounds.sort(key=lambda r: r["round_no"])
    return rounds


def golden_policy() -> dict[str, Any]:
    for vector in golden_hash_vectors().get("vectors", []):
        if vector.get("name") == "promotion_policy":
            return dict(vector["payload"])
    raise PrerequisiteMissingError("golden promotion_policy vector is missing from the contracts", code="POLICY_UNAVAILABLE")


def resolve_policy(policy: dict[str, Any] | Any | None) -> dict[str, Any]:
    if policy is None:
        try:
            from ..services import policy as policy_module  # lazily: owned by another implementer
        except ImportError:
            return golden_policy()
        default = getattr(policy_module, "default_policy", None)
        if default is None:
            return golden_policy()
        try:
            return _as_dict(default())
        except TypeError:
            return golden_policy()
    return _as_dict(policy)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        if isinstance(result, dict):
            return result
    converted = to_json(value)
    if isinstance(converted, dict):
        return converted
    return {"value": converted}


def _min_confirm_pairs(policy: dict[str, Any]) -> int:
    value = policy.get("min_confirm_pairs", 3)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError("policy.min_confirm_pairs must be a positive integer", code="INVALID_POLICY")
    return value


# --------------------------------------------------------------------------------------
# Default collaborators (lazy imports by module path)
# --------------------------------------------------------------------------------------
def _default_comparator(store: MemoryStore, candidate_run_id: str, baseline_run_id: str) -> dict[str, Any]:
    try:
        from ..services import compare as compare_module
    except ImportError as exc:
        raise PrerequisiteMissingError("comparison service is unavailable", code="COMPARATOR_UNAVAILABLE") from exc
    fn = getattr(compare_module, "compare_in_store", None)
    if fn is None:
        raise PrerequisiteMissingError("comparison service lacks compare_in_store", code="COMPARATOR_UNAVAILABLE")
    return _as_dict(fn(store, candidate_run_id, baseline_run_id))


def _default_decider(store: MemoryStore, **kwargs: Any) -> Any:
    try:
        from ..services import decide as decide_module
    except ImportError as exc:
        raise PrerequisiteMissingError("decision service is unavailable", code="DECIDER_UNAVAILABLE") from exc
    evaluate = getattr(decide_module, "evaluate_candidate", None)
    append = getattr(decide_module, "append_decision", None)
    if evaluate is None or append is None:
        raise PrerequisiteMissingError("decision service lacks evaluate_candidate/append_decision", code="DECIDER_UNAVAILABLE")
    try:
        evaluated = evaluate(store, **kwargs)
        return append(store, evaluated)
    except TypeError as exc:
        raise PrerequisiteMissingError(
            f"decision service signature mismatch; inject a decider callable ({exc})", code="DECIDER_UNAVAILABLE"
        ) from exc


def _default_context_exporter(store: MemoryStore, config_ref: str) -> dict[str, Any] | None:
    try:
        from ..services import context as context_module
    except ImportError:
        return None
    fn = getattr(context_module, "export_context", None)
    if fn is None:
        return None
    return _as_dict(fn(store, config_ref))


def _default_annotator(store: MemoryStore, **kwargs: Any) -> Any:
    """Program-authored annotation: recorded data about a run, never a judgement of improvement."""
    record = new_record(
        "annotation",
        f"annotation-{uuid.uuid4().hex[:16]}",
        {
            "target_ref": kwargs["target_ref"],
            "category": kwargs["category"],
            "text": kwargs["text"],
            "author_kind": kwargs["author_kind"],
            "evidence_refs": list(kwargs["evidence_refs"]),
            "confidence": kwargs["confidence"],
            "supersedes_ref": None,
        },
    )
    store.publish(record, label="program annotation")
    return record


def _ref_of(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Record):
        return value.record_id
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("record_id", "decision_ref", "annotation_ref", "ref"):
            if isinstance(value.get(key), str):
                return value[key]
    return getattr(value, "record_id", None)


# --------------------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------------------
class OptimizationCoordinator:
    def __init__(
        self,
        store: MemoryStore,
        *,
        runner: LocalRunner,
        planner: Any,
        budget: Budget,
        policy: dict[str, Any] | None,
        job_id: str,
        config_ref: str,
        permissions: dict[str, Any],
        request_factory: RequestFactory,
        baseline_run_ref: str | None,
        comparator: Callable[..., Any] | None = None,
        decider: Callable[..., Any] | None = None,
        context_exporter: Callable[..., Any] | None = None,
        annotator: Callable[..., Any] | None = None,
        subject_ref: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        dry_run: bool = False,
    ) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise InputError("job_id must be a non-empty string", code="INVALID_JOB_ID")
        if not hasattr(planner, "propose"):
            raise InputError("planner must implement propose(context, budget, usage)", code="INVALID_PLANNER")
        budget.validate()
        self._store = store
        self._runner = runner
        self._planner = planner
        self._budget = budget
        self._policy = resolve_policy(policy)
        self._job_id = job_id
        self._config_ref = config_ref
        self._permissions = dict(permissions or {})
        self._request_factory = request_factory
        self._baseline_run_ref = baseline_run_ref
        self._comparator = comparator or _default_comparator
        self._decider = decider or _default_decider
        self._context_exporter = context_exporter or _default_context_exporter
        self._annotator = annotator or _default_annotator
        self._explicit_subject = subject_ref
        self._clock = clock
        self._dry_run = bool(dry_run)

    # ------------------------------------------------------------------ properties
    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self._policy)

    # ------------------------------------------------------------------ main loop
    def run(self) -> JobReport:
        config = self._store.require(self._config_ref, "config")
        baseline_run: Record | None = None
        if self._baseline_run_ref is not None:
            baseline_run = self._store.require(self._baseline_run_ref, "run")
        subject_ref = self._resolve_subject(baseline_run)
        ledger = BudgetLedger(self._store, self._job_id, self._budget, persist=not self._dry_run)
        persisted = [] if self._dry_run else load_job_rounds(self._store, self._job_id)
        round_no = (max(r["round_no"] for r in persisted) + 1) if persisted else 1
        planner_id = str(getattr(self._planner, "planner_id", "unknown"))
        report = JobReport(
            job_id=self._job_id,
            config_ref=self._config_ref,
            status="dry_run" if self._dry_run else "completed",
            stop_reason=None,
            rounds=[],
            usage=ledger.usage().to_dict(),
            budget=self._budget.to_dict(),
            persisted_state_path=None if self._dry_run else state_path(self._job_id),
            planner_id=planner_id,
            dry_run=self._dry_run,
            resumed_rounds=len(persisted),
        )
        min_pairs = _min_confirm_pairs(self._policy)

        while True:
            if not self._dry_run and cancel_requested(self._store, self._job_id):
                report.status = "cancelled"
                report.stop_reason = "USER_CANCELLED"
                break
            try:
                ledger.check()
            except BudgetExhausted as exc:
                report.status = "stopped" if not self._dry_run else "dry_run"
                report.stop_reason = str(exc.details.get("stop_reason", "BUDGET_EXHAUSTED"))
                break

            context = self._export_context(config, subject_ref, round_no)
            if getattr(self._planner, "requires_model_api", False):
                if not self._permissions.get("allow_model_api_calls", False):
                    report.status = "stopped"
                    report.stop_reason = "MODEL_API_NOT_AUTHORIZED"
                    report.notes.append("planner requires a model API but allow_model_api_calls is false")
                    break
                ledger.record_model_call()
            try:
                proposal = self._planner.propose(context, self._budget, ledger.usage())
            except PrerequisiteMissingError as exc:
                report.status = "stopped"
                report.stop_reason = exc.code
                report.notes.append(f"planner unavailable: {exc.message}")
                break
            if proposal is None:
                report.status = "dry_run" if self._dry_run else "completed"
                report.stop_reason = "PLANNER_EXHAUSTED"
                break

            round_report = RoundReport(
                round_no=round_no,
                proposal=proposal.to_dict(),
                request_id=None,
                run_ref=None,
                comparison=None,
                decision_ref=None,
                annotation_ref=None,
                outcome_note="",
            )
            blocked = self._validate_proposal(proposal)
            if blocked is not None:
                round_report.outcome_note = f"blocked: {blocked.message}"
                round_report.error = blocked.to_dict()
                ledger.record_round(False)
                self._finish_round(report, round_report, ledger)
                round_no += 1
                continue

            ledger.record_candidate()
            if self._dry_run:
                round_report.outcome_note = "dry run: proposal validated, not executed"
                ledger.record_round(False)
                self._finish_round(report, round_report, ledger)
                round_no += 1
                continue

            spec = self._request_factory(proposal, round_no, subject_ref)
            if not isinstance(spec, RunRequestSpec):
                raise InputError("request_factory must return a RunRequestSpec", code="INVALID_REQUEST_FACTORY")
            round_report.request_id = spec.request_id
            try:
                ledger.reserve_execution()
            except BudgetExhausted as exc:
                report.status = "stopped"
                report.stop_reason = str(exc.details.get("stop_reason", "BUDGET_EXECUTIONS_EXHAUSTED"))
                round_report.outcome_note = "not executed: execution budget exhausted before reservation"
                ledger.record_round(False)
                self._finish_round(report, round_report, ledger)
                break

            started = float(self._clock())
            run: Record | None = None
            stop_after_round: str | None = None
            try:
                outcome, run = self._runner.submit_and_execute(spec)
                if not outcome.created and run is not None:
                    round_report.outcome_note = "replayed: request already finished; existing run reused"
            except BackendUnavailable as exc:
                round_report.error = exc.to_dict()
                round_report.outcome_note = f"backend unavailable: {exc.message}"
                stop_after_round = "BACKEND_UNAVAILABLE"
            except LeaseLostError as exc:
                round_report.error = exc.to_dict()
                round_report.outcome_note = f"lease lost; late result quarantined: {exc.message}"
            except KernelMemoryError as exc:
                round_report.error = exc.to_dict()
                round_report.outcome_note = f"execution failed: {exc.code}: {exc.message}"
                ledger.record_execution_result(False)
            finally:
                ledger.add_wall_time(max(0.0, float(self._clock()) - started))

            improved = False
            if run is not None:
                improved = self._evaluate_run(run, baseline_run, min_pairs, round_report, ledger)
            elif stop_after_round is None and round_report.error is None:
                round_report.outcome_note = round_report.outcome_note or "request finished without a run"
                ledger.record_execution_result(False)
            ledger.record_round(improved)
            self._finish_round(report, round_report, ledger)
            round_no += 1
            if stop_after_round is not None:
                report.status = "stopped"
                report.stop_reason = stop_after_round
                break

        report.usage = ledger.usage().to_dict()
        if not self._dry_run:
            self._write_state(report, ledger)
        return report

    # ------------------------------------------------------------------ round pieces
    def _resolve_subject(self, baseline_run: Record | None) -> str:
        if self._explicit_subject is not None:
            self._store.require(self._explicit_subject, "commit", "baseline")
            return self._explicit_subject
        if baseline_run is not None:
            return baseline_run.payload.subject_ref
        baselines = baselines_for_config(self._store, self._config_ref)
        if not baselines:
            raise InputError(
                f"config {self._config_ref!r} has no baseline and no subject_ref/baseline_run_ref was given",
                code="SUBJECT_UNRESOLVED",
                details={"config_ref": self._config_ref},
            )
        return sorted(baselines, key=lambda r: r.record_id)[0].record_id

    def _export_context(self, config: Record, subject_ref: str, round_no: int) -> MemoryContext:
        try:
            exported = self._context_exporter(self._store, self._config_ref)
        except (TypeError, PrerequisiteMissingError):
            exported = None  # signature mismatch or unavailable service: fall back to the minimal local context
        if exported is None:
            baselines = baselines_for_config(self._store, self._config_ref)
            exported = {
                "config_ref": self._config_ref,
                "config_hash": config.payload.config_hash,
                "current_baselines": [{"baseline_ref": b.record_id, "baseline_id": b.payload.baseline_id} for b in baselines],
                "run_count": len(runs_for_subject(self._store, subject_ref)),
                "note": "minimal local context; the context service was unavailable",
            }
        data = dict(_as_dict(exported))
        data["default_parent_ref"] = subject_ref
        data.setdefault("baseline_run_ref", self._baseline_run_ref)
        return MemoryContext(config_ref=self._config_ref, config_hash=config.payload.config_hash, context=data, round_no=round_no)

    def _validate_proposal(self, proposal: Proposal) -> KernelMemoryError | None:
        changes = proposal.planned_changes if isinstance(proposal.planned_changes, list) else []
        components: set[str] = set()
        for change in changes:
            if not isinstance(change, dict) or not isinstance(change.get("component"), str):
                return InputError("planned_changes entries must be objects with a string component", code="PROPOSAL_INVALID")
            components.add(change["component"])
        if not components and not proposal.implementation_overrides:
            return InputError("proposal plans no changes", code="PROPOSAL_EMPTY")
        code_components = sorted(c for c in components if c not in ALLOWED_OVERRIDE_COMPONENTS)
        if code_components:
            if not self._permissions.get("allow_candidate_code_write", False):
                return AuthorizationError(
                    "candidate code write not authorized",
                    code="CANDIDATE_CODE_WRITE_NOT_AUTHORIZED",
                    details={"components": code_components, "required": ["permissions.allow_candidate_code_write"]},
                )
            return PrerequisiteMissingError(
                "candidate code writes need an isolated workspace, which this coordinator does not provide",
                code="CANDIDATE_CODE_WRITE_UNSUPPORTED",
                details={"components": code_components},
            )
        if proposal.file_allowlist:
            return InputError(
                "override-only proposals must not request file access",
                code="PROPOSAL_INVALID",
                details={"file_allowlist": list(proposal.file_allowlist)},
            )
        overrides = proposal.implementation_overrides
        if not isinstance(overrides, dict) or not overrides:
            return InputError("override proposal carries no implementation_overrides", code="PROPOSAL_INVALID")
        for change in changes:
            key = change.get("key")
            if not isinstance(key, str) or key not in overrides or overrides[key] != change.get("after"):
                return InputError(
                    "planned_changes disagree with implementation_overrides",
                    code="PROPOSAL_INCONSISTENT",
                    details={"change": change, "implementation_overrides": dict(overrides)},
                )
        return None

    def _evaluate_run(
        self,
        run: Record,
        baseline_run: Record | None,
        min_pairs: int,
        round_report: RoundReport,
        ledger: BudgetLedger,
    ) -> bool:
        payload = run.payload
        round_report.run_ref = run.record_id
        round_report.execution_status = payload.execution_status
        success = payload.execution_status == "succeeded" and payload.correctness.status == "pass"
        ledger_note = f"execution_status={payload.execution_status}; correctness={payload.correctness.status}; timing={payload.timing.status}"
        ledger.record_execution_result(success)
        comparison: dict[str, Any] | None = None
        if baseline_run is not None:
            try:
                comparison = _as_dict(self._comparator(self._store, run.record_id, baseline_run.record_id))
            except (PrerequisiteMissingError, TypeError) as exc:
                comparison = {"result": "UNAVAILABLE", "error": exc.to_dict() if isinstance(exc, KernelMemoryError) else {"error": type(exc).__name__, "message": str(exc)}}
            round_report.comparison = comparison
        try:
            annotation = self._annotator(
                self._store,
                target_ref=run.record_id,
                category="note",
                text=self._annotation_text(run, comparison),
                author_kind="program",
                evidence_refs=[r for r in (run.record_id, baseline_run.record_id if baseline_run else None) if r],
                confidence="unverified",
            )
            round_report.annotation_ref = _ref_of(annotation)
        except (KernelMemoryError, TypeError) as exc:
            round_report.notes.append(f"annotation not recorded: {type(exc).__name__}: {exc}")
        improved = False
        if success and baseline_run is not None:
            pairs = self._eligible_pairs(run, baseline_run)
            if len(pairs) >= min_pairs:
                improved = self._decide(run, baseline_run, pairs, round_report)
            else:
                ledger_note += f"; pairs={len(pairs)}/{min_pairs} (no decision yet)"
        round_report.outcome_note = (round_report.outcome_note + "; " if round_report.outcome_note else "") + ledger_note
        return improved

    def _eligible_pairs(self, run: Record, baseline_run: Record) -> list[tuple[str, str]]:
        variant = run.payload.source.variant_digest
        pairs: list[tuple[str, str]] = []
        for candidate in runs_for_subject(self._store, run.payload.subject_ref):
            p = candidate.payload
            if (
                p.source.variant_digest == variant
                and p.execution_status == "succeeded"
                and p.correctness.status == "pass"
                and p.timing.status == "recorded"
                and p.provenance == "trusted_worker"
            ):
                pairs.append((candidate.record_id, baseline_run.record_id))
        return pairs

    def _decide(self, run: Record, baseline_run: Record, pairs: list[tuple[str, str]], round_report: RoundReport) -> bool:
        candidate_refs = sorted({c for c, _ in pairs})
        baseline_refs = sorted({b for _, b in pairs})
        try:
            decision = self._decider(
                self._store,
                config_ref=self._config_ref,
                candidate_subject_ref=run.payload.subject_ref,
                candidate_run_refs=candidate_refs,
                baseline_run_refs=baseline_refs,
                policy=dict(self._policy),
                comparison_key=run.payload.comparison_key,
            )
        except (PrerequisiteMissingError, TypeError) as exc:
            message = exc.message if isinstance(exc, KernelMemoryError) else f"{type(exc).__name__}: {exc}"
            round_report.error = exc.to_dict() if isinstance(exc, KernelMemoryError) else {"error": "DECIDER_UNAVAILABLE", "message": message}
            round_report.notes.append("decision unavailable: " + message)
            return False
        ref = _ref_of(decision)
        round_report.decision_ref = ref
        outcome: Any = None
        if isinstance(decision, Record) and decision.record_type == "decision":
            outcome = decision.payload.outcome
        elif isinstance(decision, dict):
            outcome = decision.get("outcome")
            payload = decision.get("payload")
            if outcome is None and isinstance(payload, dict):
                outcome = payload.get("outcome")
        elif ref is not None and self._store.exists(ref):
            stored = self._store.get(ref)
            if stored is not None and stored.record_type == "decision":
                outcome = stored.payload.outcome
        return outcome == "accepted"

    @staticmethod
    def _annotation_text(run: Record, comparison: dict[str, Any] | None) -> str:
        p = run.payload
        speedup = None
        result = None
        if isinstance(comparison, dict):
            result = comparison.get("result") or comparison.get("status")
            speedup = comparison.get("speedup")
        parts = [
            f"Program-authored note for run {run.record_id}",
            f"execution_status={p.execution_status}",
            f"correctness={p.correctness.status}",
            f"timing={p.timing.status}",
            f"comparison={result if result is not None else 'not compared'}",
            f"speedup={speedup if speedup is not None else 'n/a'}",
            "This is recorded data, not a confirmed improvement.",
        ]
        return "; ".join(str(x) for x in parts)

    def _finish_round(self, report: JobReport, round_report: RoundReport, ledger: BudgetLedger) -> None:
        report.rounds.append(round_report)
        report.usage = ledger.usage().to_dict()
        if self._dry_run:
            return
        data = {"job_id": self._job_id, **round_report.to_dict()}
        self._store.write_fact(f"{rounds_dir(self._job_id)}/{round_report.round_no:04d}.json", dumps_readable(data).encode("utf-8"))
        self._write_state(report, ledger)

    def _write_state(self, report: JobReport, ledger: BudgetLedger) -> None:
        state = {
            "job_id": self._job_id,
            "config_ref": self._config_ref,
            "status": report.status,
            "stop_reason": report.stop_reason,
            "last_round": report.rounds[-1].round_no if report.rounds else None,
            "usage": ledger.usage().to_dict(),
            "budget": self._budget.to_dict(),
            "updated_at": utc_now_iso(),
        }
        self._store.write_runtime_state(state_path(self._job_id), dumps_readable(state).encode("utf-8"))


__all__ = [
    "OptimizationCoordinator",
    "JobReport",
    "RoundReport",
    "RequestFactory",
    "ALLOWED_OVERRIDE_COMPONENTS",
    "JOB_STATUSES",
    "cancel_job",
    "clear_cancel",
    "cancel_requested",
    "cancel_flag_path",
    "state_path",
    "rounds_dir",
    "load_job_rounds",
    "resolve_policy",
    "golden_policy",
]
