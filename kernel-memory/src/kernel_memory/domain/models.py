"""Typed, immutable record models mirroring ``contracts/record.schema.json``.

Records are validated against the JSON Schema at every boundary (import,
persistence, export) and then materialised into frozen dataclasses. The
dataclasses add typed access, reference extraction (for dependency ordering
and deep validation), and canonical digests. They never relax the schema:
unknown fields are rejected by the schema before construction.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

from .errors import InputError, SchemaValidationError
from .hashing import jcs_digest
from .schema import SCHEMA_VERSION, validate_record_dict

JSONValue = Any


# --------------------------------------------------------------------------------------
# Generic dataclass <-> JSON conversion
# --------------------------------------------------------------------------------------
def _type_matches(tp: Any, value: Any) -> bool:
    if tp is Any:
        return True
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    if tp is dict or typing.get_origin(tp) is dict:
        return isinstance(value, dict)
    if tp is list or typing.get_origin(tp) is list:
        return isinstance(value, list)
    if tp is type(None):
        return value is None
    return False


def _from_json(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            if dataclasses.is_dataclass(arg) and isinstance(value, dict):
                return _from_json(arg, value, path)
            if _type_matches(arg, value):
                return _from_json(arg, value, path)
        raise SchemaValidationError(f"{path}: value {value!r} does not match {tp}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path}: expected object for {tp.__name__}")
        hints = typing.get_type_hints(tp)
        kwargs: dict[str, Any] = {}
        known = {f.name for f in dataclasses.fields(tp)}
        unknown = set(value) - known
        if unknown:
            raise SchemaValidationError(f"{path}: unknown fields {sorted(unknown)}")
        for f in dataclasses.fields(tp):
            if f.name not in value:
                raise SchemaValidationError(f"{path}: missing field {f.name!r}")
            kwargs[f.name] = _from_json(hints[f.name], value[f.name], f"{path}.{f.name}")
        return tp(**kwargs)
    if origin is list:
        (item_tp,) = typing.get_args(tp) or (Any,)
        if not isinstance(value, list):
            raise SchemaValidationError(f"{path}: expected array")
        return [_from_json(item_tp, item, f"{path}[{i}]") for i, item in enumerate(value)]
    if tp is dict or origin is dict:
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path}: expected object")
        return dict(value)
    if _type_matches(tp, value):
        return value
    raise SchemaValidationError(f"{path}: value {value!r} does not match {tp}")


def to_json(value: Any) -> JSONValue:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, list):
        return [to_json(v) for v in value]
    if isinstance(value, tuple):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------
# Nested contract types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GitOid:
    algorithm: str
    hex: str

    def short(self, length: int = 12) -> str:
        return self.hex[:length]

    def key(self) -> tuple[str, str]:
        return (self.algorithm, self.hex)


@dataclass(frozen=True)
class Change:
    change_id: str
    component: str
    key: str | None
    before: Any
    after: Any
    rationale: str
    extraction_source: str
    attribution: str


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    sha256: str
    size_bytes: int
    media_type: str
    retention: str
    availability: str


@dataclass(frozen=True)
class Metric:
    name: str
    status: str
    value: float | None
    unit: str
    kind: str
    scope: str
    source_artifact_ref: str | None
    parser_id: str | None
    parser_version: str | None
    definition: str


@dataclass(frozen=True)
class Environment:
    backend: str
    accelerator_model: str
    device_count: int
    topology: str
    software: dict
    execution_flags: dict
    host_timer_environment: dict
    unknown_required_fields: list[str]
    environment_hash: str


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    measurement_scope: str
    timing_method: str
    include_compile: bool
    include_transfers: bool
    warmup: int
    repetitions: int
    statistic: str
    quantile_method: str
    input_suite_hash: str
    capture_profile: bool
    protocol_hash: str


@dataclass(frozen=True)
class Verifier:
    verifier_id: str
    reference_source_hash: str
    suite_hash: str
    tolerances: dict
    nonfinite_policy: str
    verifier_hash: str


@dataclass(frozen=True)
class Timing:
    status: str
    sample_count: int
    median_us: float | None
    p90_us: float | None
    samples_artifact_ref: str | None


@dataclass(frozen=True)
class Correctness:
    status: str
    cases_total: int
    cases_passed: int
    max_abs_error: float | None
    max_rel_error: float | None
    report_artifact_ref: str | None


@dataclass(frozen=True)
class Policy:
    policy_id: str
    min_confirm_pairs: int
    min_pair_speedup: str
    max_normalized_iqr: str
    max_baseline_drift_ratio: str
    hard_resource_constraints: list[dict]
    require_trusted_worker: bool
    allow_fixture: bool


@dataclass(frozen=True)
class Source:
    repo_uid: str
    target_commit: GitOid
    tested_commit: GitOid
    tested_tree: GitOid | None
    checkout_mode: str
    merge_parent_oids: list[GitOid]
    dirty: bool
    patch_digest: str | None
    source_digest: str
    entrypoint: str
    implementation_overrides: dict
    variant_digest: str


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """An outgoing reference from a record payload field to another record or artifact."""

    field: str
    target: str
    allowed_types: tuple[str, ...]  # record types; ("artifact",) for artifact-registry references
    required: bool = True


ANY_RECORD: tuple[str, ...] = (
    "kernel",
    "config",
    "pr",
    "pr_snapshot",
    "commit",
    "baseline",
    "run",
    "relation",
    "decision",
    "annotation",
)


# --------------------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelPayload:
    kernel_id: str
    display_name: str
    adapter_id: str
    contract_notes: str

    record_type: ClassVar[str] = "kernel"

    def references(self) -> list[Reference]:
        return []


@dataclass(frozen=True)
class ConfigPayload:
    kernel_id: str
    config_id: str
    problem_schema_id: str
    problem_schema_digest: str
    problem: dict
    config_hash: str
    tags: list[str]

    record_type: ClassVar[str] = "config"

    def references(self) -> list[Reference]:
        return []


@dataclass(frozen=True)
class PrPayload:
    config_ref: str
    pr_key: str
    repo_uid: str
    provider: str
    number: int | None
    title: str
    hypothesis: str | None
    origin_ref: str | None

    record_type: ClassVar[str] = "pr"

    def references(self) -> list[Reference]:
        refs = [Reference("config_ref", self.config_ref, ("config",))]
        if self.origin_ref is not None:
            refs.append(Reference("origin_ref", self.origin_ref, ("commit", "baseline")))
        return refs

    @property
    def is_local_trial(self) -> bool:
        return self.provider == "local"


@dataclass(frozen=True)
class PrSnapshotPayload:
    pr_ref: str
    previous_snapshot_ref: str | None
    observed_head: GitOid
    observed_base: GitOid
    commit_refs: list[str]
    enumeration_status: str
    reason: str | None
    github_state: str

    record_type: ClassVar[str] = "pr_snapshot"

    def references(self) -> list[Reference]:
        refs = [Reference("pr_ref", self.pr_ref, ("pr",))]
        if self.previous_snapshot_ref is not None:
            refs.append(Reference("previous_snapshot_ref", self.previous_snapshot_ref, ("pr_snapshot",)))
        refs.extend(Reference(f"commit_refs[{i}]", c, ("commit",)) for i, c in enumerate(self.commit_refs))
        return refs


@dataclass(frozen=True)
class CommitPayload:
    pr_ref: str
    repo_uid: str
    commit_oid: GitOid
    git_parent_oids: list[GitOid]
    diff_base_oid: GitOid | None
    source_available: bool
    change_status: str
    changes: list[Change]
    summary: str
    summary_author: str
    diff_artifact_ref: str | None

    record_type: ClassVar[str] = "commit"

    def references(self) -> list[Reference]:
        refs = [Reference("pr_ref", self.pr_ref, ("pr",))]
        if self.diff_artifact_ref is not None:
            refs.append(Reference("diff_artifact_ref", self.diff_artifact_ref, ("artifact",)))
        return refs

    @property
    def source_key(self) -> tuple[str, str, str]:
        return (self.repo_uid, self.commit_oid.algorithm, self.commit_oid.hex)


@dataclass(frozen=True)
class BaselinePayload:
    config_ref: str
    baseline_id: str
    description: str
    repo_uid: str
    commit_oid: GitOid
    entrypoint: str
    role: str

    record_type: ClassVar[str] = "baseline"

    def references(self) -> list[Reference]:
        return [Reference("config_ref", self.config_ref, ("config",))]


@dataclass(frozen=True)
class RunPayload:
    subject_ref: str
    config_ref: str
    request_id: str
    attempt_no: int
    rerun_of: str | None
    stage: str
    provenance: str
    source: Source
    environment: Environment
    protocol: Protocol
    verifier: Verifier
    execution_status: str
    failure_reason: str | None
    correctness: Correctness
    timing: Timing
    analysis_metrics: list[Metric]
    analysis_conclusion: str | None
    artifacts: list[ArtifactRef]
    comparison_key: str
    session_id: str | None
    pair_id: str | None
    role_in_pair: str | None

    record_type: ClassVar[str] = "run"

    def references(self) -> list[Reference]:
        refs = [
            Reference("subject_ref", self.subject_ref, ("commit", "baseline")),
            Reference("config_ref", self.config_ref, ("config",)),
        ]
        if self.rerun_of is not None:
            refs.append(Reference("rerun_of", self.rerun_of, ("run",)))
        return refs

    def artifact_by_id(self) -> dict[str, ArtifactRef]:
        return {a.artifact_id: a for a in self.artifacts}

    @property
    def is_terminal_success(self) -> bool:
        return self.execution_status == "succeeded"


@dataclass(frozen=True)
class RelationPayload:
    config_ref: str
    kind: str
    from_ref: str
    to_ref: str
    evidence_refs: list[str]
    rationale: str

    record_type: ClassVar[str] = "relation"

    def references(self) -> list[Reference]:
        refs = [
            Reference("config_ref", self.config_ref, ("config",)),
            Reference("from_ref", self.from_ref, ANY_RECORD),
            Reference("to_ref", self.to_ref, ANY_RECORD),
        ]
        refs.extend(Reference(f"evidence_refs[{i}]", e, ANY_RECORD) for i, e in enumerate(self.evidence_refs))
        return refs


@dataclass(frozen=True)
class DecisionPayload:
    config_ref: str
    comparison_key: str
    candidate_subject_ref: str
    candidate_run_refs: list[str]
    baseline_run_refs: list[str]
    policy: Policy
    policy_hash: str
    outcome: str
    reason_codes: list[str]
    is_production: bool
    evaluated_by: str
    supersedes_decision_ref: str | None

    record_type: ClassVar[str] = "decision"

    def references(self) -> list[Reference]:
        refs = [
            Reference("config_ref", self.config_ref, ("config",)),
            Reference("candidate_subject_ref", self.candidate_subject_ref, ("commit", "baseline")),
        ]
        refs.extend(Reference(f"candidate_run_refs[{i}]", r, ("run",)) for i, r in enumerate(self.candidate_run_refs))
        refs.extend(Reference(f"baseline_run_refs[{i}]", r, ("run",)) for i, r in enumerate(self.baseline_run_refs))
        if self.supersedes_decision_ref is not None:
            refs.append(Reference("supersedes_decision_ref", self.supersedes_decision_ref, ("decision",)))
        return refs


@dataclass(frozen=True)
class AnnotationPayload:
    target_ref: str
    category: str
    text: str
    author_kind: str
    evidence_refs: list[str]
    confidence: str
    supersedes_ref: str | None

    record_type: ClassVar[str] = "annotation"

    def references(self) -> list[Reference]:
        refs = [Reference("target_ref", self.target_ref, ANY_RECORD)]
        refs.extend(Reference(f"evidence_refs[{i}]", e, ANY_RECORD) for i, e in enumerate(self.evidence_refs))
        if self.supersedes_ref is not None:
            refs.append(Reference("supersedes_ref", self.supersedes_ref, ("annotation",)))
        return refs


Payload = (
    KernelPayload
    | ConfigPayload
    | PrPayload
    | PrSnapshotPayload
    | CommitPayload
    | BaselinePayload
    | RunPayload
    | RelationPayload
    | DecisionPayload
    | AnnotationPayload
)

PAYLOAD_TYPES: dict[str, type] = {
    "kernel": KernelPayload,
    "config": ConfigPayload,
    "pr": PrPayload,
    "pr_snapshot": PrSnapshotPayload,
    "commit": CommitPayload,
    "baseline": BaselinePayload,
    "run": RunPayload,
    "relation": RelationPayload,
    "decision": DecisionPayload,
    "annotation": AnnotationPayload,
}

# Deterministic publication order used as a tie-breaker in dependency ordering.
TYPE_ORDER: dict[str, int] = {t: i for i, t in enumerate(ANY_RECORD)}


# --------------------------------------------------------------------------------------
# Record envelope
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    schema_version: str
    record_type: str
    record_id: str
    created_at: str
    payload: Any

    @classmethod
    def from_dict(cls, data: Any) -> "Record":
        record_type = validate_record_dict(data)
        payload_cls = PAYLOAD_TYPES[record_type]
        payload = _from_json(payload_cls, data["payload"], f"{record_type}.payload")
        return cls(
            schema_version=data["schema_version"],
            record_type=record_type,
            record_id=data["record_id"],
            created_at=data["created_at"],
            payload=payload,
        )

    @classmethod
    def create(cls, record_type: str, record_id: str, payload: Any, *, created_at: str) -> "Record":
        """Build a record from a typed payload and validate it against the schema."""
        if record_type not in PAYLOAD_TYPES:
            raise InputError(f"unknown record type {record_type!r}")
        if not isinstance(payload, PAYLOAD_TYPES[record_type]):
            raise InputError(f"payload type {type(payload).__name__} does not match {record_type}")
        data = {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "record_id": record_id,
            "created_at": created_at,
            "payload": to_json(payload),
        }
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "created_at": self.created_at,
            "payload": to_json(self.payload),
        }

    def canonical_digest(self) -> str:
        """Content identity of the record (RFC 8785 canonical form), independent of file formatting."""
        return jcs_digest(self.to_dict())

    def references(self) -> list[Reference]:
        return list(self.payload.references())

    def record_references(self) -> list[Reference]:
        return [r for r in self.references() if r.allowed_types != ("artifact",)]

    def artifact_references(self) -> list[Reference]:
        return [r for r in self.references() if r.allowed_types == ("artifact",)]


def records_from_iterable(items: Iterable[Any]) -> list[Record]:
    return [Record.from_dict(item) for item in items]


__all__ = [
    "GitOid",
    "Change",
    "ArtifactRef",
    "Metric",
    "Environment",
    "Protocol",
    "Verifier",
    "Timing",
    "Correctness",
    "Policy",
    "Source",
    "Reference",
    "KernelPayload",
    "ConfigPayload",
    "PrPayload",
    "PrSnapshotPayload",
    "CommitPayload",
    "BaselinePayload",
    "RunPayload",
    "RelationPayload",
    "DecisionPayload",
    "AnnotationPayload",
    "Payload",
    "PAYLOAD_TYPES",
    "TYPE_ORDER",
    "ANY_RECORD",
    "Record",
    "to_json",
    "records_from_iterable",
]
