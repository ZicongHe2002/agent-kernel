"""Promotion decisions: evaluate candidate/baseline pairs against a policy and append Decision records.

Specification 12.2 and 13, DESIGN section 6, acceptance scenarios T14-T23. A decision is derived
only from stored evidence (runs, their sample blobs, analysis metrics) and a fixed, hashed policy.
Nothing here executes code, invents numbers (all statistics come from ``domain.stats`` through
``services.compare``) or treats a missing metric as zero.

Public API
----------
``evaluate_candidate(store, candidate_subject_ref, pairs, policy=None, *, evaluated_by="program",
                     config_ref=None, candidate_run_refs=None, baseline_run_refs=None,
                     comparison_key=None) -> DecisionDraft``
    ``pairs`` is ``[{"candidate_run": run_ref, "baseline_run": run_ref}, ...]`` (``"candidate"`` /
    ``"baseline"`` keys and 2-tuples are accepted too). Alternatively pass ``candidate_run_refs`` and
    ``baseline_run_refs`` (equal length, or a single baseline shared by every candidate) - the keyword
    form used by the execution coordinator. ``policy`` defaults to ``services.policy.default_policy()``
    and is validated; ``policy_hash`` is computed from it.

    Gates, in order, each contributing reason codes (sorted, unique in the draft):
      * ``CANDIDATE_SUBJECT_MISMATCH``  a candidate run's ``subject_ref`` is not the candidate subject (blocked)
      * ``NOT_COMPARABLE``  runs do not share one comparison key / config (blocked; the notes summarise the
        field-level differences)
      * per pair ``services.compare.compare_in_store``: ``EXECUTION_NOT_SUCCEEDED``, ``CORRECTNESS_NOT_PASSED``,
        ``MISSING_EVIDENCE`` (timing missing, summaries inconsistent, samples unavailable, unknown required
        environment fields) - blocked
      * provenance: ``FIXTURE_NOT_ELIGIBLE`` unless ``policy.allow_fixture``; ``UNVERIFIED_PROVENANCE`` when
        ``policy.require_trusted_worker``; ``DIRTY_SOURCE`` - blocked
      * ``AMBIGUOUS_VARIANT``  candidate runs with different ``source.variant_digest`` (blocked)
      * ``INSUFFICIENT_CONFIRMATION_PAIRS``  fewer than ``min_confirm_pairs`` *distinct* pairs; a pair is
        identified by (candidate session_id, candidate pair_id, baseline session_id, baseline pair_id); repeats
        count once (note ``REPEATED_PAIR_NOT_INDEPENDENT``) and pairs without ids are not counted (inconclusive)
      * ``PAIR_SPEEDUP_BELOW_THRESHOLD``  any pair with speedup < ``min_pair_speedup`` (rejected only when the
        pairs are sufficient and no inconclusive gate fired; otherwise listed and the outcome stays inconclusive)
      * ``EXCESSIVE_VARIABILITY``  any run's normalized IQR > ``max_normalized_iqr`` (inconclusive)
      * ``BASELINE_DRIFT``  max/min of the baseline medians across distinct pairs > ``max_baseline_drift_ratio``
        (inconclusive)
      * ``hard_resource_constraints`` against the candidate runs' ``analysis_metrics`` (name, and scope/kind
        when given): ``RESOURCE_CONSTRAINT_VIOLATED`` (rejected); metric absent, not ``observed`` or ``null``
        -> ``RESOURCE_CONSTRAINT_UNVERIFIABLE`` (blocked) - null is never treated as 0
    Outcome: ``blocked`` if any blocking code; else ``rejected`` for a constraint violation; else
    ``inconclusive`` if any inconclusive code; else ``rejected`` for a speedup below threshold; else
    ``accepted``. ``is_production`` is true only for an accepted decision whose runs are all
    ``trusted_worker`` under a policy with ``require_trusted_worker``.
``append_decision(store, draft, *, supersedes_decision_ref=None, created_at=None) -> Record``
    Record id ``decision-<candidate_subject>-<sha256(policy_hash, sorted run refs, created_at)[:12]>``
    (DESIGN section 4); validated and published (idempotent for identical content).
``best_known(store, config_ref, *, comparison_key=None, policy_hash=None) -> list[dict]``
    Latest accepted production decision per ``(comparison_key, policy_hash)`` group, excluding decisions
    superseded by a later decision; never a context-free "best" flag.
``PairEvaluation``, ``DecisionDraft`` dataclasses with ``to_dict()`` (``DecisionDraft.to_payload()`` yields
the decision record payload).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable

from ..domain import stats
from ..domain.errors import InputError, InvariantViolation
from ..domain.hashing import jcs_digest
from ..domain.ids import utc_now_iso
from ..domain.models import Record
from ..storage.store import MemoryStore
from .common import decisions_for_config, new_record
from .compare import (
    BLOCKER_DIRTY,
    BLOCKER_FIXTURE,
    BLOCKER_MISSING_PAIR_IDS,
    BLOCKER_MISSING_SAMPLES,
    BLOCKER_UNKNOWN_ENVIRONMENT,
    BLOCKER_UNVERIFIED,
    COMPARABLE,
    INSUFFICIENT_EVIDENCE,
    NOT_COMPARABLE,
    NOT_ELIGIBLE,
    ComparisonResult,
    compare_in_store,
)
from .policy import PolicyThresholds, ResourceConstraint, default_policy
from .policy import policy_hash as compute_policy_hash
from .policy import thresholds as policy_thresholds
from .policy import validate_policy

# Reason codes (DESIGN section 6) grouped by their effect on the outcome.
CANDIDATE_SUBJECT_MISMATCH = "CANDIDATE_SUBJECT_MISMATCH"
EXECUTION_NOT_SUCCEEDED = "EXECUTION_NOT_SUCCEEDED"
CORRECTNESS_NOT_PASSED = "CORRECTNESS_NOT_PASSED"
MISSING_EVIDENCE = "MISSING_EVIDENCE"
FIXTURE_NOT_ELIGIBLE = "FIXTURE_NOT_ELIGIBLE"
UNVERIFIED_PROVENANCE = "UNVERIFIED_PROVENANCE"
DIRTY_SOURCE = "DIRTY_SOURCE"
AMBIGUOUS_VARIANT = "AMBIGUOUS_VARIANT"
RESOURCE_CONSTRAINT_UNVERIFIABLE = "RESOURCE_CONSTRAINT_UNVERIFIABLE"
RESOURCE_CONSTRAINT_VIOLATED = "RESOURCE_CONSTRAINT_VIOLATED"
PAIR_SPEEDUP_BELOW_THRESHOLD = "PAIR_SPEEDUP_BELOW_THRESHOLD"
INSUFFICIENT_CONFIRMATION_PAIRS = "INSUFFICIENT_CONFIRMATION_PAIRS"
EXCESSIVE_VARIABILITY = "EXCESSIVE_VARIABILITY"
BASELINE_DRIFT = "BASELINE_DRIFT"

BLOCKING_CODES: frozenset[str] = frozenset(
    {
        CANDIDATE_SUBJECT_MISMATCH,
        NOT_COMPARABLE,
        EXECUTION_NOT_SUCCEEDED,
        CORRECTNESS_NOT_PASSED,
        MISSING_EVIDENCE,
        FIXTURE_NOT_ELIGIBLE,
        UNVERIFIED_PROVENANCE,
        DIRTY_SOURCE,
        AMBIGUOUS_VARIANT,
        RESOURCE_CONSTRAINT_UNVERIFIABLE,
    }
)
REJECTING_CODES: frozenset[str] = frozenset({PAIR_SPEEDUP_BELOW_THRESHOLD, RESOURCE_CONSTRAINT_VIOLATED})
INCONCLUSIVE_CODES: frozenset[str] = frozenset({INSUFFICIENT_CONFIRMATION_PAIRS, EXCESSIVE_VARIABILITY, BASELINE_DRIFT})
OUTCOMES: tuple[str, ...] = ("accepted", "rejected", "inconclusive", "blocked")
EVALUATORS: tuple[str, ...] = ("program", "human_override")


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------
@dataclass
class PairEvaluation:
    pair_id: str
    candidate_run_ref: str
    baseline_run_ref: str
    comparison: ComparisonResult
    pair_speedup: float | None
    candidate_iqr: float | None
    baseline_iqr: float | None
    passed: bool
    reasons: list[str] = field(default_factory=list)
    pair_key: tuple[str, str, str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "candidate_run_ref": self.candidate_run_ref,
            "baseline_run_ref": self.baseline_run_ref,
            "comparison": self.comparison.to_dict(),
            "pair_speedup": self.pair_speedup,
            "candidate_iqr": self.candidate_iqr,
            "baseline_iqr": self.baseline_iqr,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "pair_key": list(self.pair_key) if self.pair_key is not None else None,
        }


@dataclass
class DecisionDraft:
    config_ref: str
    comparison_key: str | None
    candidate_subject_ref: str
    candidate_run_refs: list[str]
    baseline_run_refs: list[str]
    policy: dict[str, Any]
    policy_hash: str
    outcome: str
    reason_codes: list[str]
    is_production: bool
    evaluated_by: str
    pair_evaluations: list[PairEvaluation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        if self.comparison_key is None:
            raise InvariantViolation("decision draft has no comparison key", code="DECISION_INCOMPLETE")
        return {
            "config_ref": self.config_ref,
            "comparison_key": self.comparison_key,
            "candidate_subject_ref": self.candidate_subject_ref,
            "candidate_run_refs": sorted(set(self.candidate_run_refs)),
            "baseline_run_refs": sorted(set(self.baseline_run_refs)),
            "policy": dict(self.policy),
            "policy_hash": self.policy_hash,
            "outcome": self.outcome,
            "reason_codes": sorted(set(self.reason_codes)),
            "is_production": bool(self.is_production),
            "evaluated_by": self.evaluated_by,
            "supersedes_decision_ref": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_ref": self.config_ref,
            "comparison_key": self.comparison_key,
            "candidate_subject_ref": self.candidate_subject_ref,
            "candidate_run_refs": list(self.candidate_run_refs),
            "baseline_run_refs": list(self.baseline_run_refs),
            "policy": dict(self.policy),
            "policy_hash": self.policy_hash,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "is_production": self.is_production,
            "evaluated_by": self.evaluated_by,
            "pair_evaluations": [p.to_dict() for p in self.pair_evaluations],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _normalize_pairs(
    pairs: Iterable[Any] | None, candidate_run_refs: Iterable[str] | None, baseline_run_refs: Iterable[str] | None
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if pairs is not None:
        for n, item in enumerate(pairs):
            if isinstance(item, dict):
                cand = item.get("candidate_run", item.get("candidate"))
                base = item.get("baseline_run", item.get("baseline"))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                cand, base = item
            else:
                raise InputError(f"pairs[{n}] must be a mapping with candidate_run/baseline_run", code="INVALID_PAIRS")
            if not isinstance(cand, str) or not isinstance(base, str) or not cand or not base:
                raise InputError(f"pairs[{n}] needs candidate_run and baseline_run references", code="INVALID_PAIRS")
            out.append((cand, base))
        if candidate_run_refs is not None or baseline_run_refs is not None:
            raise InputError("pass either pairs or candidate_run_refs/baseline_run_refs, not both", code="INVALID_PAIRS")
        return out
    cands = list(candidate_run_refs or [])
    bases = list(baseline_run_refs or [])
    if not cands or not bases:
        return []
    if len(bases) == 1:
        bases = bases * len(cands)
    if len(cands) != len(bases):
        raise InputError(
            "candidate_run_refs and baseline_run_refs must pair up (equal length or one shared baseline)",
            code="INVALID_PAIRS",
            details={"candidates": len(cands), "baselines": len(bases)},
        )
    for cand, base in zip(cands, bases):
        if not isinstance(cand, str) or not isinstance(base, str):
            raise InputError("run references must be strings", code="INVALID_PAIRS")
        out.append((cand, base))
    return out


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _metric_matches(metric: Any, constraint: ResourceConstraint) -> bool:
    if metric.name != constraint.metric:
        return False
    if constraint.scope is not None and metric.scope != constraint.scope:
        return False
    if constraint.kind is not None and metric.kind != constraint.kind:
        return False
    return True


def _evaluate_constraints(
    candidates: list[Record], constraints: tuple[ResourceConstraint, ...]
) -> tuple[set[str], list[str]]:
    codes: set[str] = set()
    notes: list[str] = []
    for constraint in constraints:
        for run in candidates:
            matches = [m for m in run.payload.analysis_metrics if _metric_matches(m, constraint)]
            observed = [m for m in matches if m.status == "observed" and m.value is not None and not isinstance(m.value, bool)]
            if not observed:
                codes.add(RESOURCE_CONSTRAINT_UNVERIFIABLE)
                statuses = sorted({m.status for m in matches}) or ["absent"]
                notes.append(
                    f"{RESOURCE_CONSTRAINT_UNVERIFIABLE}: {constraint.metric} on {run.record_id} is {','.join(statuses)} "
                    "(null is not zero)"
                )
                continue
            for metric in observed:
                value = metric.value
                if constraint.max_value is not None and value > constraint.max_value:
                    codes.add(RESOURCE_CONSTRAINT_VIOLATED)
                    notes.append(
                        f"{RESOURCE_CONSTRAINT_VIOLATED}: {constraint.metric} on {run.record_id} = {value} {metric.unit} "
                        f"> max {constraint.max_value}"
                    )
                if constraint.min_value is not None and value < constraint.min_value:
                    codes.add(RESOURCE_CONSTRAINT_VIOLATED)
                    notes.append(
                        f"{RESOURCE_CONSTRAINT_VIOLATED}: {constraint.metric} on {run.record_id} = {value} {metric.unit} "
                        f"< min {constraint.min_value}"
                    )
    return codes, notes


def _provenance_codes(run: Record, th: PolicyThresholds) -> set[str]:
    codes: set[str] = set()
    p = run.payload
    if p.provenance == "fixture" and not th.allow_fixture:
        codes.add(FIXTURE_NOT_ELIGIBLE)
    elif p.provenance == "imported_unverified" and th.require_trusted_worker:
        codes.add(UNVERIFIED_PROVENANCE)
    elif p.provenance not in ("fixture", "trusted_worker", "imported_unverified"):
        codes.add(UNVERIFIED_PROVENANCE)
    if p.source.dirty:
        codes.add(DIRTY_SOURCE)
    return codes


def _blocker_codes(blockers: Iterable[str], th: PolicyThresholds) -> set[str]:
    codes: set[str] = set()
    for blocker in blockers:
        if blocker == BLOCKER_FIXTURE:
            if not th.allow_fixture:
                codes.add(FIXTURE_NOT_ELIGIBLE)
        elif blocker == BLOCKER_UNVERIFIED:
            if th.require_trusted_worker:
                codes.add(UNVERIFIED_PROVENANCE)
        elif blocker == BLOCKER_DIRTY:
            codes.add(DIRTY_SOURCE)
        elif blocker in (BLOCKER_MISSING_SAMPLES, BLOCKER_UNKNOWN_ENVIRONMENT):
            codes.add(MISSING_EVIDENCE)
        elif blocker == BLOCKER_MISSING_PAIR_IDS:
            continue  # handled by distinct-pair counting
        elif blocker in (NOT_COMPARABLE, NOT_ELIGIBLE, INSUFFICIENT_EVIDENCE):
            continue  # status codes are mapped from the comparison reasons
    return codes


def _comparison_codes(comparison: ComparisonResult) -> set[str]:
    codes: set[str] = set()
    if comparison.status == NOT_COMPARABLE:
        codes.add(NOT_COMPARABLE)
    elif comparison.status == NOT_ELIGIBLE:
        for reason in comparison.reasons:
            if reason.startswith(EXECUTION_NOT_SUCCEEDED):
                codes.add(EXECUTION_NOT_SUCCEEDED)
            elif reason.startswith(CORRECTNESS_NOT_PASSED):
                codes.add(CORRECTNESS_NOT_PASSED)
            else:
                codes.add(MISSING_EVIDENCE)
        if not codes:
            codes.add(MISSING_EVIDENCE)
    elif comparison.status == INSUFFICIENT_EVIDENCE:
        codes.add(MISSING_EVIDENCE)
    return codes


def _outcome(codes: set[str]) -> str:
    if codes & BLOCKING_CODES:
        return "blocked"
    if RESOURCE_CONSTRAINT_VIOLATED in codes:
        return "rejected"
    if codes & INCONCLUSIVE_CODES:
        return "inconclusive"
    if PAIR_SPEEDUP_BELOW_THRESHOLD in codes:
        return "rejected"
    return "accepted"


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def evaluate_candidate(
    store: MemoryStore,
    candidate_subject_ref: str | None = None,
    pairs: list[dict[str, str]] | None = None,
    policy: dict[str, Any] | None = None,
    *,
    evaluated_by: str = "program",
    config_ref: str | None = None,
    candidate_run_refs: list[str] | None = None,
    baseline_run_refs: list[str] | None = None,
    comparison_key: str | None = None,
) -> DecisionDraft:
    """Evaluate candidate/baseline pairs under a policy; returns an unpublished ``DecisionDraft``."""
    if evaluated_by not in EVALUATORS:
        raise InputError(f"evaluated_by must be one of {list(EVALUATORS)}", code="INVALID_EVALUATOR")
    if not isinstance(candidate_subject_ref, str) or not candidate_subject_ref:
        raise InputError("candidate_subject_ref is required", code="MISSING_SUBJECT")
    policy_dict = validate_policy(policy) if policy is not None else default_policy()
    th = policy_thresholds(policy_dict)
    phash = compute_policy_hash(policy_dict)
    pair_specs = _normalize_pairs(pairs, candidate_run_refs, baseline_run_refs)
    if not pair_specs:
        raise InputError("at least one candidate/baseline run pair is required", code="NO_PAIRS")

    store.require(candidate_subject_ref, "commit", "baseline")
    loaded: list[tuple[Record, Record]] = [(store.require(c, "run"), store.require(b, "run")) for c, b in pair_specs]
    all_runs: list[Record] = []
    for cand, base in loaded:
        all_runs.extend((cand, base))
    codes: set[str] = set()
    notes: list[str] = []

    # Gate 1: subject.
    for cand, _ in loaded:
        if cand.payload.subject_ref != candidate_subject_ref:
            codes.add(CANDIDATE_SUBJECT_MISMATCH)
            notes.append(f"{CANDIDATE_SUBJECT_MISMATCH}: {cand.record_id} has subject {cand.payload.subject_ref}, expected {candidate_subject_ref}")

    # Gate 2: one config, one comparison key.
    candidate_config_refs = sorted({cand.payload.config_ref for cand, _ in loaded})
    run_config_refs = sorted({r.payload.config_ref for r in all_runs})
    if config_ref is None:
        config_ref = candidate_config_refs[0]
    elif config_ref not in run_config_refs:
        raise InputError(
            f"config_ref {config_ref!r} does not match the runs' configs {run_config_refs}",
            code="CONFIG_MISMATCH",
            details={"config_ref": config_ref, "run_config_refs": run_config_refs},
        )
    store.require(config_ref, "config")
    if len(run_config_refs) > 1:
        codes.add(NOT_COMPARABLE)
        notes.append(f"{NOT_COMPARABLE}: runs span configs {run_config_refs}")
    candidate_keys = sorted({cand.payload.comparison_key for cand, _ in loaded})
    all_keys = sorted({r.payload.comparison_key for r in all_runs})
    decision_key = comparison_key if comparison_key is not None else candidate_keys[0]
    if len(all_keys) > 1 or (comparison_key is not None and all_keys != [comparison_key]):
        codes.add(NOT_COMPARABLE)
        notes.append(f"{NOT_COMPARABLE}: runs span comparison keys {all_keys}" + (f" (expected {comparison_key})" if comparison_key else ""))
    if len(candidate_keys) > 1:
        notes.append(f"decision comparison_key {decision_key} taken from the first candidate run; candidates disagree")

    # Gate 3: provenance and dirty state of every participating run.
    for run in all_runs:
        run_codes = _provenance_codes(run, th)
        codes |= run_codes
        for code in sorted(run_codes):
            notes.append(f"{code}: {run.record_id} provenance={run.payload.provenance} dirty={run.payload.source.dirty}")

    # Gate 4: unambiguous candidate variant.
    variants = sorted({cand.payload.source.variant_digest for cand, _ in loaded})
    if len(variants) > 1:
        codes.add(AMBIGUOUS_VARIANT)
        notes.append(f"{AMBIGUOUS_VARIANT}: candidate runs span variants {variants}")

    # Gate 5: per-pair comparison and thresholds.
    evaluations: list[PairEvaluation] = []
    distinct: dict[tuple[str, str, str, str], PairEvaluation] = {}
    uncounted = 0
    for n, (cand, base) in enumerate(loaded, start=1):
        comparison = compare_in_store(store, cand.record_id, base.record_id)
        pair_codes = _comparison_codes(comparison)
        pair_codes |= _blocker_codes(comparison.confirmation_blockers, th)
        if comparison.status == NOT_COMPARABLE:
            summary = "; ".join(f"{d.group}.{d.field}: {d.candidate!r} vs {d.baseline!r}" for d in comparison.identity_differences())
            notes.append(f"{NOT_COMPARABLE}: {cand.record_id} vs {base.record_id}: {summary}")
        for reason in comparison.reasons:
            notes.append(f"{cand.record_id} vs {base.record_id}: {reason}")
        for warning in comparison.warnings:
            if warning.startswith((BLOCKER_MISSING_SAMPLES, "CORRUPT_SAMPLES", "INVALID_SAMPLES_BLOB", "SAMPLES_ARTIFACT")):
                notes.append(f"{MISSING_EVIDENCE}: {cand.record_id} vs {base.record_id}: {warning}")
        speedup: float | None = None
        cand_iqr: float | None = None
        base_iqr: float | None = None
        if comparison.derived is not None:
            speedup = comparison.derived.get("speedup")
            cand_iqr = comparison.derived.get("candidate_normalized_iqr")
            base_iqr = comparison.derived.get("baseline_normalized_iqr")
            if speedup is not None and _decimal(speedup) < th.min_pair_speedup:
                pair_codes.add(PAIR_SPEEDUP_BELOW_THRESHOLD)
                notes.append(f"{PAIR_SPEEDUP_BELOW_THRESHOLD}: {cand.record_id} vs {base.record_id} speedup {speedup!r} < {th.min_pair_speedup}")
            for label, iqr, run in (("candidate", cand_iqr, cand), ("baseline", base_iqr, base)):
                if iqr is not None and _decimal(iqr) > th.max_normalized_iqr:
                    pair_codes.add(EXCESSIVE_VARIABILITY)
                    notes.append(f"{EXCESSIVE_VARIABILITY}: {label} {run.record_id} normalized IQR {iqr!r} > {th.max_normalized_iqr}")
        pair_key: tuple[str, str, str, str] | None = None
        ids = (cand.payload.session_id, cand.payload.pair_id, base.payload.session_id, base.payload.pair_id)
        if all(isinstance(x, str) and x for x in ids):
            pair_key = ids  # type: ignore[assignment]
        pair_id = cand.payload.pair_id or f"pair-{n}"
        evaluation = PairEvaluation(
            pair_id=pair_id,
            candidate_run_ref=cand.record_id,
            baseline_run_ref=base.record_id,
            comparison=comparison,
            pair_speedup=speedup,
            candidate_iqr=cand_iqr,
            baseline_iqr=base_iqr,
            passed=comparison.status == COMPARABLE and not (pair_codes & (BLOCKING_CODES | REJECTING_CODES | INCONCLUSIVE_CODES)),
            reasons=sorted(pair_codes),
            pair_key=pair_key,
        )
        evaluations.append(evaluation)
        codes |= pair_codes
        if comparison.derived is None:
            continue
        if pair_key is None:
            uncounted += 1
            notes.append(f"PAIR_IDS_MISSING: {cand.record_id} vs {base.record_id} is not counted as a distinct confirmation pair")
        elif pair_key in distinct:
            notes.append(
                f"REPEATED_PAIR_NOT_INDEPENDENT: {cand.record_id} vs {base.record_id} repeats pair {pair_key[0]}/{pair_key[1]}"
            )
        else:
            distinct[pair_key] = evaluation

    # Gate 6: distinct pairs and baseline drift.
    if len(distinct) < th.min_confirm_pairs:
        codes.add(INSUFFICIENT_CONFIRMATION_PAIRS)
        notes.append(f"{INSUFFICIENT_CONFIRMATION_PAIRS}: {len(distinct)} distinct pair(s) of {len(loaded)} supplied; policy requires {th.min_confirm_pairs}")
    baseline_medians = [
        e.comparison.derived["baseline_median_us"]
        for e in distinct.values()
        if e.comparison.derived is not None and e.comparison.derived.get("baseline_median_us") is not None
    ]
    if len(baseline_medians) >= 2:
        try:
            drift = stats.speedup(max(baseline_medians), min(baseline_medians))  # max/min ratio of two medians
        except InputError:
            drift = None
        if drift is not None and _decimal(drift) > th.max_baseline_drift_ratio:
            codes.add(BASELINE_DRIFT)
            notes.append(f"{BASELINE_DRIFT}: baseline medians {sorted(baseline_medians)} max/min {drift!r} > {th.max_baseline_drift_ratio}")

    # Gate 7: hard resource constraints on the candidate runs.
    unique_candidates: dict[str, Record] = {cand.record_id: cand for cand, _ in loaded}
    constraint_codes, constraint_notes = _evaluate_constraints(list(unique_candidates.values()), th.hard_resource_constraints)
    codes |= constraint_codes
    notes.extend(constraint_notes)

    outcome = _outcome(codes)
    all_trusted = all(r.payload.provenance == "trusted_worker" for r in all_runs)
    is_production = outcome == "accepted" and all_trusted and th.require_trusted_worker
    if outcome == "accepted" and not is_production:
        notes.append("accepted but not production: provenance is not trusted_worker for every run or the policy does not require trusted workers")
    return DecisionDraft(
        config_ref=config_ref,
        comparison_key=decision_key,
        candidate_subject_ref=candidate_subject_ref,
        candidate_run_refs=sorted({cand.record_id for cand, _ in loaded}),
        baseline_run_refs=sorted({base.record_id for _, base in loaded}),
        policy=policy_dict,
        policy_hash=phash,
        outcome=outcome,
        reason_codes=sorted(codes),
        is_production=is_production,
        evaluated_by=evaluated_by,
        pair_evaluations=evaluations,
        notes=notes,
    )


def decision_record_id(candidate_subject_ref: str, policy_hash: str, run_refs: Iterable[str], created_at: str) -> str:
    digest = jcs_digest({"policy_hash": policy_hash, "run_refs": sorted(set(run_refs)), "created_at": created_at})
    return f"decision-{candidate_subject_ref}-{digest[len('sha256:'):][:12]}"


def append_decision(
    store: MemoryStore,
    draft: DecisionDraft,
    *,
    supersedes_decision_ref: str | None = None,
    created_at: str | None = None,
) -> Record:
    """Publish a decision record built from ``draft`` (immutable, evidence-referencing)."""
    if not isinstance(draft, DecisionDraft):
        raise InputError("append_decision expects a DecisionDraft", code="INVALID_DRAFT")
    if draft.outcome not in OUTCOMES:
        raise InputError(f"unknown decision outcome {draft.outcome!r}", code="INVALID_OUTCOME")
    if supersedes_decision_ref is not None:
        previous = store.require(supersedes_decision_ref, "decision")
        if previous.payload.config_ref != draft.config_ref:
            raise InvariantViolation(
                f"decision {supersedes_decision_ref!r} belongs to config {previous.payload.config_ref!r}, not {draft.config_ref!r}",
                code="SUPERSEDES_OTHER_CONFIG",
            )
    payload = draft.to_payload()
    payload["supersedes_decision_ref"] = supersedes_decision_ref
    stamp = created_at or utc_now_iso()
    record_id = decision_record_id(
        draft.candidate_subject_ref, draft.policy_hash, payload["candidate_run_refs"] + payload["baseline_run_refs"], stamp
    )
    record = new_record("decision", record_id, payload, created_at=stamp)
    store.publish(record, label=f"decision {draft.outcome}")
    return record


def supersedes_chain(store: MemoryStore, decision: Record) -> list[str]:
    """Decision refs superseded (transitively) by ``decision``, nearest first; cycles are cut."""
    chain: list[str] = []
    seen = {decision.record_id}
    current = decision.payload.supersedes_decision_ref
    while current and current not in seen:
        chain.append(current)
        seen.add(current)
        previous = store.get(current)
        if previous is None or previous.record_type != "decision":
            break
        current = previous.payload.supersedes_decision_ref
    return chain


def best_known(
    store: MemoryStore, config_ref: str, *, comparison_key: str | None = None, policy_hash: str | None = None
) -> list[dict[str, Any]]:
    """Latest accepted production decision per (comparison_key, policy_hash), honouring supersession."""
    config = store.require(config_ref, "config")
    decisions = decisions_for_config(store, config_ref)
    superseded = {d.payload.supersedes_decision_ref for d in decisions if d.payload.supersedes_decision_ref}
    groups: dict[tuple[str, str], Record] = {}
    for decision in decisions:
        p = decision.payload
        if decision.record_id in superseded or p.outcome != "accepted" or not p.is_production:
            continue
        if comparison_key is not None and p.comparison_key != comparison_key:
            continue
        if policy_hash is not None and p.policy_hash != policy_hash:
            continue
        key = (p.comparison_key, p.policy_hash)
        current = groups.get(key)
        if current is None or (decision.created_at, decision.record_id) > (current.created_at, current.record_id):
            groups[key] = decision
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        decision = groups[key]
        p = decision.payload
        out.append(
            {
                "config_ref": config_ref,
                "config_hash": config.payload.config_hash,
                "comparison_key": p.comparison_key,
                "policy_hash": p.policy_hash,
                "policy_id": p.policy.policy_id,
                "decision_ref": decision.record_id,
                "created_at": decision.created_at,
                "candidate_subject_ref": p.candidate_subject_ref,
                "candidate_run_refs": list(p.candidate_run_refs),
                "baseline_run_refs": list(p.baseline_run_refs),
                "outcome": p.outcome,
                "is_production": p.is_production,
                "evaluated_by": p.evaluated_by,
                "supersedes": supersedes_chain(store, decision),
            }
        )
    return out


__all__ = [
    "AMBIGUOUS_VARIANT",
    "BASELINE_DRIFT",
    "BLOCKING_CODES",
    "CANDIDATE_SUBJECT_MISMATCH",
    "CORRECTNESS_NOT_PASSED",
    "DIRTY_SOURCE",
    "EXCESSIVE_VARIABILITY",
    "EXECUTION_NOT_SUCCEEDED",
    "FIXTURE_NOT_ELIGIBLE",
    "INCONCLUSIVE_CODES",
    "INSUFFICIENT_CONFIRMATION_PAIRS",
    "MISSING_EVIDENCE",
    "NOT_COMPARABLE",
    "OUTCOMES",
    "PAIR_SPEEDUP_BELOW_THRESHOLD",
    "REJECTING_CODES",
    "RESOURCE_CONSTRAINT_UNVERIFIABLE",
    "RESOURCE_CONSTRAINT_VIOLATED",
    "UNVERIFIED_PROVENANCE",
    "DecisionDraft",
    "PairEvaluation",
    "append_decision",
    "best_known",
    "decision_record_id",
    "evaluate_candidate",
    "supersedes_chain",
]
