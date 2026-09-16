"""Pairwise run comparison (specification 11.2 and 12.1, DESIGN section 6).

``compare_runs`` is a *pure* function over two run payloads plus optional raw samples. It never
touches the store, never invents a number (every derived value comes from ``domain.stats``) and
never ranks runs from different comparability groups.

Evaluation order (first hit wins for the status):

1. ``NOT_COMPARABLE`` - the recorded ``comparison_key`` differs or an identity field differs:
   config hash, any environment / protocol / verifier snapshot field (their own hash fields are
   excluded from the field diff and only reported when nothing else explains the change), or
   ``source.checkout_mode``. ``differences`` names every differing field (dotted paths).
   Variant differences (``source.variant_digest``, ``source.entrypoint``, ``source.source_digest``,
   ``source.implementation_overrides.*``) are always reported as *informational* differences with
   group ``variant``: two variants of one comparability group stay comparable, but the result names
   both variants.
2. ``NOT_ELIGIBLE`` - either run has ``execution_status != succeeded``
   (``EXECUTION_NOT_SUCCEEDED(candidate|baseline)``) or ``correctness.status != pass``
   (``CORRECTNESS_NOT_PASSED(candidate|baseline)``). ``derived`` is ``None``.
3. ``INSUFFICIENT_EVIDENCE`` - timing not recorded (``TIMING_NOT_RECORDED(side)``), missing recorded
   summaries (``SUMMARY_MISSING(side)``), invalid samples / unknown unit (``INVALID_SAMPLES(side)``,
   ``UNKNOWN_UNIT(side)``), or recomputed median/p90/count disagreeing with the recorded summary
   (``SUMMARY_MISMATCH(side)``, ``SAMPLE_COUNT_MISMATCH(side)``).
4. ``COMPARABLE`` - ``derived`` holds ``candidate_median_us``, ``baseline_median_us``, ``speedup``,
   ``latency_reduction_pct``, ``candidate_p90_us``, ``baseline_p90_us``, ``candidate_normalized_iqr``,
   ``baseline_normalized_iqr`` (``None`` unless recomputed from samples), ``candidate_sample_count``,
   ``baseline_sample_count``, ``recomputed_from_samples`` and ``unit`` (always ``"microseconds"``).
   Without raw samples the recorded summaries are used with warning
   ``DERIVED_FROM_RECORDED_SUMMARIES(side)`` and confirmation blocker ``MISSING_SAMPLES``.

``confirmation_eligible`` is true only for a ``COMPARABLE`` result with no blockers. Blockers are
plain codes: ``FIXTURE_NOT_ELIGIBLE``, ``UNVERIFIED_PROVENANCE``, ``DIRTY_SOURCE``, ``MISSING_SAMPLES``,
``MISSING_PAIR_IDS`` (session or pair id absent), ``UNKNOWN_ENVIRONMENT_FIELDS`` (required environment
fields unknown, specification 12.1), plus the non-comparable status itself. Blockers never turn a
comparable pair into ``NOT_COMPARABLE``; policy application lives in ``services.decide``.

Public API
----------
``SnapshotDifference(group, field, candidate, baseline)`` with ``to_dict()``
``ComparisonResult(...)`` with ``to_dict()`` (adds a top-level ``speedup`` convenience key)
``compare_runs(candidate, baseline, *, candidate_config_hash=None, baseline_config_hash=None,
               candidate_samples=None, baseline_samples=None) -> ComparisonResult``
    ``candidate``/``baseline`` are ``Record`` instances of type ``run``, full record dicts, or bare run
    payload dicts. Samples are ``(list_of_numbers, unit)`` tuples.
``compare_in_store(store, candidate_run_ref, baseline_run_ref) -> ComparisonResult``
    Loads both runs (``MissingReferenceError`` when absent), the config hashes, and the sample blobs from
    the content-addressed store via ``timing.samples_artifact_ref -> run.artifacts -> sha256``
    (JSON ``{"samples": [...], "unit": "..."}``). A missing, corrupt or malformed blob degrades to the
    recorded summaries with a warning (``MISSING_SAMPLES(side)``, ``CORRUPT_SAMPLES(side)``,
    ``INVALID_SAMPLES_BLOB(side)``) and the ``MISSING_SAMPLES`` blocker; it never raises for evidence gaps.
``diff_snapshots(a, b, group, *, exclude=(), prefix="") -> list[SnapshotDifference]``
    Recursive field-level diff of two JSON objects with dotted paths; absent keys show as ``"<absent>"``.
``load_run_samples(store, run) -> tuple[tuple[list[float], str] | None, list[str]]``
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..domain import hashing, stats
from ..domain.errors import InputError, KernelMemoryError
from ..domain.jsonio import loads_strict
from ..domain.models import Record, to_json
from ..storage.store import MemoryStore

COMPARABLE = "COMPARABLE"
NOT_COMPARABLE = "NOT_COMPARABLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATUSES: tuple[str, ...] = (COMPARABLE, NOT_COMPARABLE, NOT_ELIGIBLE, INSUFFICIENT_EVIDENCE)

GROUPS: tuple[str, ...] = ("config", "environment", "protocol", "verifier", "checkout_mode", "variant")
ABSENT = "<absent>"
DERIVED_UNIT = "microseconds"

# Confirmation blockers (policy-free; the decision service maps them to reason codes).
BLOCKER_FIXTURE = "FIXTURE_NOT_ELIGIBLE"
BLOCKER_UNVERIFIED = "UNVERIFIED_PROVENANCE"
BLOCKER_DIRTY = "DIRTY_SOURCE"
BLOCKER_MISSING_SAMPLES = "MISSING_SAMPLES"
BLOCKER_MISSING_PAIR_IDS = "MISSING_PAIR_IDS"
BLOCKER_UNKNOWN_ENVIRONMENT = "UNKNOWN_ENVIRONMENT_FIELDS"

_SNAPSHOT_GROUPS: tuple[tuple[str, str], ...] = (
    ("environment", "environment_hash"),
    ("protocol", "protocol_hash"),
    ("verifier", "verifier_hash"),
)


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------
@dataclass
class SnapshotDifference:
    group: str
    field: str
    candidate: Any
    baseline: Any

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, "field": self.field, "candidate": self.candidate, "baseline": self.baseline}


@dataclass
class ComparisonResult:
    status: str
    candidate_run_ref: str
    baseline_run_ref: str
    comparison_key_candidate: str | None
    comparison_key_baseline: str | None
    differences: list[SnapshotDifference] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    derived: dict[str, Any] | None = None
    confirmation_eligible: bool = False
    confirmation_blockers: list[str] = field(default_factory=list)

    @property
    def speedup(self) -> float | None:
        return None if self.derived is None else self.derived.get("speedup")

    def identity_differences(self) -> list[SnapshotDifference]:
        return [d for d in self.differences if d.group != "variant"]

    def variant_differences(self) -> list[SnapshotDifference]:
        return [d for d in self.differences if d.group == "variant"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidate_run_ref": self.candidate_run_ref,
            "baseline_run_ref": self.baseline_run_ref,
            "comparison_key_candidate": self.comparison_key_candidate,
            "comparison_key_baseline": self.comparison_key_baseline,
            "differences": [d.to_dict() for d in self.differences],
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "derived": None if self.derived is None else dict(self.derived),
            "confirmation_eligible": self.confirmation_eligible,
            "confirmation_blockers": list(self.confirmation_blockers),
            "speedup": self.speedup,
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _coerce_run(value: Any, side: str) -> tuple[str, dict[str, Any]]:
    """Accept a run Record, a full record dict, or a bare run payload dict."""
    if isinstance(value, Record):
        if value.record_type != "run":
            raise InputError(
                f"{side} must be a run record, got {value.record_type!r}", code="INVALID_RUN", details={"record_id": value.record_id}
            )
        return value.record_id, to_json(value.payload)
    if isinstance(value, dict):
        if "payload" in value or "record_type" in value:
            if value.get("record_type") != "run" or not isinstance(value.get("payload"), dict):
                raise InputError(f"{side} must be a run record dict", code="INVALID_RUN")
            ref = value.get("record_id")
            return (ref if isinstance(ref, str) and ref else f"<{side}>"), copy.deepcopy(value["payload"])
        if "execution_status" in value and "timing" in value:
            return f"<{side}>", copy.deepcopy(value)
    raise InputError(f"{side} must be a run record or run payload", code="INVALID_RUN", details={"got": type(value).__name__})


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def diff_snapshots(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    group: str,
    *,
    exclude: tuple[str, ...] | frozenset[str] = (),
    prefix: str = "",
) -> list[SnapshotDifference]:
    """Recursive field-level diff (candidate ``a`` versus baseline ``b``); dotted paths, sorted by key."""
    a = _obj(a)
    b = _obj(b)
    out: list[SnapshotDifference] = []
    for key in sorted(set(a) | set(b)):
        if not prefix and key in exclude:
            continue
        path = f"{prefix}{key}"
        in_a, in_b = key in a, key in b
        va = a.get(key, ABSENT)
        vb = b.get(key, ABSENT)
        if in_a and in_b and isinstance(va, dict) and isinstance(vb, dict):
            out.extend(diff_snapshots(va, vb, group, prefix=f"{path}."))
        elif not in_a or not in_b or _values_differ(va, vb):
            out.append(SnapshotDifference(group, path, va, vb))
    return out


def _values_differ(x: Any, y: Any) -> bool:
    """JSON-value inequality: 1 and 1.0 are equal numbers, but True and 1 (or "1") are different."""
    x_bool, y_bool = isinstance(x, bool), isinstance(y, bool)
    if x_bool != y_bool:
        return True
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not x_bool:
        return x != y
    if type(x) is not type(y):
        return True
    return x != y


def _identity_differences(
    cand: dict[str, Any], base: dict[str, Any], cand_cfg: str | None, base_cfg: str | None
) -> tuple[list[SnapshotDifference], list[SnapshotDifference], list[str]]:
    """Return (identity differences, informational differences, warnings)."""
    identity: list[SnapshotDifference] = []
    informational: list[SnapshotDifference] = []
    warnings: list[str] = []
    hashes_known = cand_cfg is not None and base_cfg is not None
    if hashes_known and cand_cfg != base_cfg:
        identity.append(SnapshotDifference("config", "config_hash", cand_cfg, base_cfg))
    if cand.get("config_ref") != base.get("config_ref"):
        diff = SnapshotDifference("config", "config_ref", cand.get("config_ref"), base.get("config_ref"))
        if hashes_known and cand_cfg == base_cfg:
            informational.append(diff)
            warnings.append("CONFIG_REF_DIFFERS_SAME_HASH")
        else:
            identity.append(diff)
    for group, hash_field in _SNAPSHOT_GROUPS:
        cs = _obj(cand.get(group))
        bs = _obj(base.get(group))
        diffs = diff_snapshots(cs, bs, group, exclude=(hash_field,))
        if diffs:
            identity.extend(diffs)
            if cs.get(hash_field) == bs.get(hash_field):
                warnings.append(f"{hash_field.upper()}_INCONSISTENT")
        elif cs.get(hash_field) != bs.get(hash_field):
            identity.append(SnapshotDifference(group, hash_field, cs.get(hash_field), bs.get(hash_field)))
    csrc = _obj(cand.get("source"))
    bsrc = _obj(base.get("source"))
    if csrc.get("checkout_mode") != bsrc.get("checkout_mode"):
        identity.append(SnapshotDifference("checkout_mode", "source.checkout_mode", csrc.get("checkout_mode"), bsrc.get("checkout_mode")))
    for name in ("variant_digest", "entrypoint", "source_digest"):
        if csrc.get(name) != bsrc.get(name):
            informational.append(SnapshotDifference("variant", f"source.{name}", csrc.get(name), bsrc.get(name)))
    informational.extend(
        diff_snapshots(
            _obj(csrc.get("implementation_overrides")),
            _obj(bsrc.get("implementation_overrides")),
            "variant",
            prefix="source.implementation_overrides.",
        )
    )
    return identity, informational, warnings


def _side_summary(
    side: str, timing: dict[str, Any], samples: tuple[Any, Any] | None
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Per-run summary in microseconds. Returns (summary|None, reasons, warnings)."""
    reasons: list[str] = []
    warnings: list[str] = []
    recorded_median = timing.get("median_us")
    recorded_p90 = timing.get("p90_us")
    recorded_count = timing.get("sample_count")
    if samples is not None:
        try:
            values, unit = samples
        except (TypeError, ValueError):
            reasons.append(f"INVALID_SAMPLES({side})")
            return None, reasons, warnings
        try:
            summary = stats.summarize(list(values) if isinstance(values, (list, tuple)) else values, unit)
        except InputError as exc:
            reasons.append(f"{exc.code}({side})")
            return None, reasons, warnings
        if recorded_median is None or recorded_p90 is None:
            reasons.append(f"SUMMARY_MISSING({side})")
        elif not stats.summaries_agree(values, unit, recorded_median, recorded_p90):
            reasons.append(f"SUMMARY_MISMATCH({side})")
        if isinstance(recorded_count, int) and not isinstance(recorded_count, bool) and recorded_count != summary.sample_count:
            reasons.append(f"SAMPLE_COUNT_MISMATCH({side})")
        if reasons:
            return None, reasons, warnings
        return (
            {
                "median_us": summary.median_us,
                "p90_us": summary.p90_us,
                "normalized_iqr": summary.normalized_iqr,
                "sample_count": summary.sample_count,
                "recomputed": True,
            },
            reasons,
            warnings,
        )
    if recorded_median is None or recorded_p90 is None:
        reasons.append(f"SUMMARY_MISSING({side})")
        return None, reasons, warnings
    for what, value in (("median_us", recorded_median), ("p90_us", recorded_p90)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            reasons.append(f"INVALID_SUMMARY({side})")
            return None, reasons, warnings
    warnings.append(f"DERIVED_FROM_RECORDED_SUMMARIES({side})")
    return (
        {
            "median_us": float(recorded_median),
            "p90_us": float(recorded_p90),
            "normalized_iqr": None,
            "sample_count": recorded_count if isinstance(recorded_count, int) and not isinstance(recorded_count, bool) else None,
            "recomputed": False,
        },
        reasons,
        warnings,
    )


def _blockers(cand: dict[str, Any], base: dict[str, Any], recomputed: bool) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    for side, payload in (("candidate", cand), ("baseline", base)):
        provenance = payload.get("provenance")
        if provenance == "fixture":
            _add(blockers, BLOCKER_FIXTURE)
        elif provenance != "trusted_worker":
            _add(blockers, BLOCKER_UNVERIFIED)
        if _obj(payload.get("source")).get("dirty") is not False:
            _add(blockers, BLOCKER_DIRTY)
        if _obj(payload.get("environment")).get("unknown_required_fields"):
            _add(blockers, BLOCKER_UNKNOWN_ENVIRONMENT)
        if payload.get("session_id") is None or payload.get("pair_id") is None:
            _add(blockers, BLOCKER_MISSING_PAIR_IDS)
    if not recomputed:
        _add(blockers, BLOCKER_MISSING_SAMPLES)
    if (
        cand.get("session_id") is not None
        and base.get("session_id") is not None
        and cand.get("pair_id") is not None
        and base.get("pair_id") is not None
        and (cand.get("session_id"), cand.get("pair_id")) != (base.get("session_id"), base.get("pair_id"))
    ):
        warnings.append("PAIR_IDS_DIFFER")
    return blockers, warnings


def _add(items: list[str], code: str) -> None:
    if code not in items:
        items.append(code)


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def compare_runs(
    candidate: Record | dict[str, Any],
    baseline: Record | dict[str, Any],
    *,
    candidate_config_hash: str | None = None,
    baseline_config_hash: str | None = None,
    candidate_samples: tuple[list[float], str] | None = None,
    baseline_samples: tuple[list[float], str] | None = None,
) -> ComparisonResult:
    """Pure comparison of a candidate run against a baseline run (see module docstring)."""
    cand_ref, cand = _coerce_run(candidate, "candidate")
    base_ref, base = _coerce_run(baseline, "baseline")
    ck_c = cand.get("comparison_key")
    ck_b = base.get("comparison_key")
    identity, informational, warnings = _identity_differences(cand, base, candidate_config_hash, baseline_config_hash)
    keys_differ = ck_c != ck_b
    if keys_differ and not identity:
        identity.append(SnapshotDifference("config", "comparison_key", ck_c, ck_b))
        warnings.append("DIFFERENCE_SOURCE_UNKNOWN")
    if identity and not keys_differ:
        warnings.append("COMPARISON_KEY_INCONSISTENT")
    if candidate_config_hash is not None and baseline_config_hash is not None:
        for side, payload, cfg in (("candidate", cand, candidate_config_hash), ("baseline", base, baseline_config_hash)):
            try:
                expected = hashing.comparison_key(
                    config_hash=cfg,
                    environment_hash=_obj(payload.get("environment")).get("environment_hash"),
                    protocol_hash=_obj(payload.get("protocol")).get("protocol_hash"),
                    verifier_hash=_obj(payload.get("verifier")).get("verifier_hash"),
                    checkout_mode=_obj(payload.get("source")).get("checkout_mode"),
                )
            except KernelMemoryError:
                expected = None
            if expected is not None and expected != payload.get("comparison_key"):
                warnings.append(f"RECORDED_COMPARISON_KEY_MISMATCH({side})")
    differences = identity + informational
    result = ComparisonResult(
        status=COMPARABLE,
        candidate_run_ref=cand_ref,
        baseline_run_ref=base_ref,
        comparison_key_candidate=ck_c,
        comparison_key_baseline=ck_b,
        differences=differences,
        warnings=warnings,
    )
    if keys_differ or identity:
        result.status = NOT_COMPARABLE
        result.reasons = [f"{NOT_COMPARABLE}({','.join(sorted({d.group for d in identity}))})"]
        result.confirmation_blockers = [NOT_COMPARABLE]
        return result

    # 2. Eligibility.
    reasons: list[str] = []
    for side, payload in (("candidate", cand), ("baseline", base)):
        if payload.get("execution_status") != "succeeded":
            reasons.append(f"EXECUTION_NOT_SUCCEEDED({side})")
        if _obj(payload.get("correctness")).get("status") != "pass":
            reasons.append(f"CORRECTNESS_NOT_PASSED({side})")
    if reasons:
        result.status = NOT_ELIGIBLE
        result.reasons = reasons
        result.confirmation_blockers = [NOT_ELIGIBLE]
        return result

    # 3. Timing evidence.
    for side, payload in (("candidate", cand), ("baseline", base)):
        if _obj(payload.get("timing")).get("status") != "recorded":
            reasons.append(f"TIMING_NOT_RECORDED({side})")
    if reasons:
        result.status = INSUFFICIENT_EVIDENCE
        result.reasons = reasons
        result.confirmation_blockers = [INSUFFICIENT_EVIDENCE]
        return result

    # 4. Derivation from raw samples (preferred) or recorded summaries.
    cand_summary, cand_reasons, cand_warnings = _side_summary("candidate", _obj(cand.get("timing")), candidate_samples)
    base_summary, base_reasons, base_warnings = _side_summary("baseline", _obj(base.get("timing")), baseline_samples)
    result.warnings.extend(cand_warnings + base_warnings)
    reasons = cand_reasons + base_reasons
    if reasons or cand_summary is None or base_summary is None:
        result.status = INSUFFICIENT_EVIDENCE
        result.reasons = reasons or ["SUMMARY_MISSING(candidate,baseline)"]
        result.confirmation_blockers = [INSUFFICIENT_EVIDENCE]
        return result
    try:
        speedup = stats.speedup(base_summary["median_us"], cand_summary["median_us"])
        reduction = stats.latency_reduction_pct(base_summary["median_us"], cand_summary["median_us"])
    except InputError as exc:
        result.status = INSUFFICIENT_EVIDENCE
        result.reasons = [f"{exc.code}(candidate,baseline)"]
        result.confirmation_blockers = [INSUFFICIENT_EVIDENCE]
        return result
    recomputed = bool(cand_summary["recomputed"] and base_summary["recomputed"])
    result.derived = {
        "candidate_median_us": cand_summary["median_us"],
        "baseline_median_us": base_summary["median_us"],
        "speedup": speedup,
        "latency_reduction_pct": reduction,
        "candidate_p90_us": cand_summary["p90_us"],
        "baseline_p90_us": base_summary["p90_us"],
        "candidate_normalized_iqr": cand_summary["normalized_iqr"],
        "baseline_normalized_iqr": base_summary["normalized_iqr"],
        "candidate_sample_count": cand_summary["sample_count"],
        "baseline_sample_count": base_summary["sample_count"],
        "recomputed_from_samples": recomputed,
        "unit": DERIVED_UNIT,
    }
    blockers, blocker_warnings = _blockers(cand, base, recomputed)
    result.warnings.extend(w for w in blocker_warnings if w not in result.warnings)
    result.confirmation_blockers = blockers
    result.confirmation_eligible = not blockers
    return result


def load_run_samples(
    store: MemoryStore, run: Record, *, side: str | None = None
) -> tuple[tuple[list[float], str] | None, list[str]]:
    """Locate and parse a run's latency sample blob. Returns ((samples, unit) | None, warnings).

    Warnings are ``CODE(side)`` or ``CODE(side:detail)`` when ``side`` is given, else ``CODE`` / ``CODE(detail)``.
    """
    warnings: list[str] = []

    def warn(code: str, detail: str | None = None) -> None:
        parts = [p for p in (side, detail) if p]
        warnings.append(f"{code}({':'.join(parts)})" if parts else code)

    payload = run.payload
    ref_id = payload.timing.samples_artifact_ref
    if ref_id is None:
        if payload.timing.status == "recorded":
            warn("SAMPLES_ARTIFACT_NOT_REFERENCED")
        return None, warnings
    artifact = payload.artifact_by_id().get(ref_id)
    if artifact is None:
        warn("SAMPLES_ARTIFACT_UNREGISTERED", ref_id)
        return None, warnings
    if artifact.availability != "present":
        warn("ARTIFACT_UNAVAILABLE", artifact.availability)
    try:
        data = store.read_artifact(artifact.sha256)
    except KernelMemoryError as exc:
        warn("CORRUPT_SAMPLES" if exc.code == "ARTIFACT_CORRUPT" else BLOCKER_MISSING_SAMPLES)
        return None, warnings
    try:
        blob = loads_strict(data)
    except InputError:
        warn("INVALID_SAMPLES_BLOB")
        return None, warnings
    if not isinstance(blob, dict) or not isinstance(blob.get("samples"), list) or not isinstance(blob.get("unit"), str):
        warn("INVALID_SAMPLES_BLOB")
        return None, warnings
    return (list(blob["samples"]), blob["unit"]), warnings


def compare_in_store(store: MemoryStore, candidate_run_ref: str, baseline_run_ref: str) -> ComparisonResult:
    """Compare two stored runs, recomputing statistics from the content-addressed sample blobs."""
    candidate = store.require(candidate_run_ref, "run")
    baseline = store.require(baseline_run_ref, "run")
    hashes: list[str | None] = []
    warnings: list[str] = []
    for side, run in (("candidate", candidate), ("baseline", baseline)):
        config = store.get(run.payload.config_ref)
        if config is None or config.record_type != "config":
            hashes.append(None)
            warnings.append(f"CONFIG_UNRESOLVED({side})")
        else:
            hashes.append(config.payload.config_hash)
    cand_samples, cand_warnings = load_run_samples(store, candidate, side="candidate")
    base_samples, base_warnings = load_run_samples(store, baseline, side="baseline")
    warnings.extend(cand_warnings)
    warnings.extend(base_warnings)
    result = compare_runs(
        candidate,
        baseline,
        candidate_config_hash=hashes[0],
        baseline_config_hash=hashes[1],
        candidate_samples=cand_samples,
        baseline_samples=base_samples,
    )
    result.warnings = warnings + [w for w in result.warnings if w not in warnings]
    return result


__all__ = [
    "ABSENT",
    "COMPARABLE",
    "GROUPS",
    "INSUFFICIENT_EVIDENCE",
    "NOT_COMPARABLE",
    "NOT_ELIGIBLE",
    "STATUSES",
    "ComparisonResult",
    "SnapshotDifference",
    "compare_in_store",
    "compare_runs",
    "diff_snapshots",
    "load_run_samples",
]
