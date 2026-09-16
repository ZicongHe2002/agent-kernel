"""Deep validation of Kernel Memory records (specification sections 6, 7, 9.2, 11.2, 13, 15).

Schema validation (``domain.schema``) proves that a record is well formed. Deep validation
proves that a *set* of records is internally consistent and that every recorded number and
hash is backed by evidence that can be recomputed. Nothing here invents measurements:
medians/p90s come from ``domain.stats`` over the raw samples artifact, every digest from
``domain.hashing``.

Public API
----------
``Issue``
    One finding: ``severity`` (``"error"`` | ``"warning"``), stable machine-readable ``code``,
    human ``message``, ``record_id`` (``None`` for set-level or store-level findings), optional
    ``field`` (payload path) and JSON-serialisable ``details``.
``ValidationReport``
    Counters (``records_checked``, ``runs_checked``, ``artifact_checks``, ``summary_checks``) plus
    ``issues``. ``.ok`` is true when no error-severity issue exists; ``.errors`` / ``.warnings``
    filter; ``.codes()`` returns the set of codes; ``.to_dict()`` is JSON-ready.
``validate_records(records, *, artifact_reader, artifact_registry=None, registry=None,
                   external_resolver=None, kernel_resolver=None, verify_artifacts=True,
                   missing_evidence_severity="error") -> ValidationReport``
    Pure function over an in-memory set of ``Record`` instances (used before publishing a
    bundle and by ``deep_validate``). References that are not in the set are looked up through
    ``external_resolver`` (e.g. ``store.get``); kernels by ``kernel_id`` through
    ``kernel_resolver`` (e.g. ``store.kernel_by_kernel_id``). ``artifact_reader(sha256_ref)``
    returns the raw bytes of a content-addressed artifact or ``None`` when it is not available;
    ``artifact_registry(artifact_id)`` returns an already registered ``ArtifactRef`` or ``None``.
    ``missing_evidence_severity="warning"`` downgrades ``MISSING_EVIDENCE`` (an importer that was
    explicitly allowed to import records whose artifact files are absent uses this).
    It never raises for invalid *content* — every problem becomes an ``Issue`` — and only raises
    ``InputError`` for programming errors (non-``Record`` inputs, bad severity).
``deep_validate(store, *, verify_artifacts=True, registry=None) -> ValidationReport``
    ``validate_records`` over every record of a ``MemoryStore`` (the store resolves kernels and
    artifact descriptors, its CAS is the artifact reader) merged with ``store.integrity_scan()``
    findings (``STORE_RECORD_MODIFIED``, ``STORE_RECORD_MISSING``, ``STORE_RECORD_CORRUPT``,
    ``STORE_RECORD_UNJOURNALED``, ``STORE_DUPLICATE_ID``, ``STORE_ARTIFACT_MISSING`` /
    ``STORE_ARTIFACT_CORRUPT`` / ``STORE_ARTIFACT_SIZE_MISMATCH``), all errors.

Issue codes (errors unless marked W = warning)
----------------------------------------------
Set level: ``DUPLICATE_RECORD_ID``, ``DUPLICATE_KERNEL_ID``, ``DUPLICATE_PR_KEY``,
``DUPLICATE_BASELINE_ID``, ``DUPLICATE_ATTEMPT``, ``MISSING_REFERENCE``, ``WRONG_REFERENCE_TYPE``,
``SELF_REFERENCE``, ``ARTIFACT_DESCRIPTOR_CONFLICT``, ``ORIGIN_CYCLE``, ``GIT_PARENT_CYCLE``,
``PARENT_OIDS_INCONSISTENT``.
config: ``CONFIG_HASH_MISMATCH``, ``CONFIG_PROBLEM_INVALID``, ``MISSING_KERNEL``.
pr: ``PROVIDER_NUMBER_MISMATCH``, W ``PR_KEY_CONVENTION``, W ``ORIGIN_CROSS_CONFIG``.
pr_snapshot: ``SNAPSHOT_CHAIN_MISMATCH``, ``SNAPSHOT_CHAIN_CYCLE``, ``CROSS_PR_MEMBERSHIP``,
``DUPLICATE_COMMIT_REF``, ``MISSING_ENUMERATION_REASON``.
commit: ``REPO_UID_MISMATCH``, ``UNEXTRACTED_CHANGES_NOT_EMPTY``, ``DUPLICATE_CHANGE_ID``,
W ``ISOLATED_ATTRIBUTION_UNSUPPORTED``, W ``DIFF_BASE_NOT_PARENT``.
relation: ``SELF_RELATION``, ``RELATION_ENDPOINT_TYPE``, W ``RELATION_CROSS_CONFIG``.
run: ``TESTED_SOURCE_MISMATCH``, ``RUN_CONFIG_MISMATCH``, ``REPO_UID_MISMATCH``,
``EXACT_COMMIT_MISMATCH``, ``INTEGRATION_MERGE_IDENTITY``, W ``INTEGRATION_MERGE_WITHOUT_PARENTS``,
``VARIANT_HASH_MISMATCH``, ``SNAPSHOT_HASH_MISMATCH``, ``COMPARISON_KEY_MISMATCH``,
``DIRTY_WITHOUT_PATCH``, W ``PATCH_WITHOUT_DIRTY``, ``DUPLICATE_ARTIFACT_ID``,
``DANGLING_IN_RUN_REF``, ``INVALID_CORRECTNESS_COUNTS``, ``INVALID_PASS``, ``INVALID_FAIL``,
``MISSING_CORRECTNESS_EVIDENCE``, ``CORRECTNESS_NOT_RUN_WITH_RESULTS``, ``SUMMARY_MISSING``,
``MISSING_TIMING_EVIDENCE``, ``MISSING_EVIDENCE`` (severity configurable), ``ARTIFACT_CORRUPT``,
``ARTIFACT_SIZE_MISMATCH``, W ``EVIDENCE_UNAVAILABLE`` (expired/missing tombstone),
``INVALID_SAMPLES_ARTIFACT``, ``UNKNOWN_UNIT``, ``INVALID_SAMPLES``, ``SAMPLE_COUNT_MISMATCH``,
``SUMMARY_MISMATCH``, ``TIMING_NOT_RUN_WITH_RESULTS``, ``FABRICATED_RESULT``,
W ``MISSING_FAILURE_REASON``, W ``FAILURE_REASON_ON_SUCCESS``,
``OBSERVED_METRIC_WITHOUT_EVIDENCE``, ``INVALID_METRIC_VALUE``, ``UNKNOWN_METRIC_NOT_NULL``,
W ``DUPLICATE_METRIC``, W ``FIXTURE_RUN``, W ``RERUN_SUBJECT_MISMATCH``.
decision: ``POLICY_HASH_MISMATCH``, ``DECISION_GROUP_MISMATCH``, ``DECISION_CONFIG_MISMATCH``,
``CANDIDATE_SUBJECT_MISMATCH``, ``PRODUCTION_WITHOUT_ACCEPTANCE``, ``PRODUCTION_WITHOUT_EVIDENCE``,
``FIXTURE_PRODUCTION_DECISION``, ``INVALID_EVALUATOR``, ``ACCEPTED_WITHOUT_EVIDENCE``,
``FIXTURE_DECISION_ACCEPTED``, ``UNVERIFIED_ACCEPTED``.
annotation: W ``DUPLICATE_EVIDENCE_REF`` (plus the generic reference checks).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, TypeVar

from ..domain import hashing, stats
from ..domain.errors import InputError, InvariantViolation, KernelMemoryError
from ..domain.jsonio import loads_strict
from ..domain.models import ArtifactRef, GitOid, Record, to_json
from ..domain.problems import ProblemRegistry, default_registry
from ..storage.store import IntegrityReport, MemoryStore

ERROR = "error"
WARNING = "warning"
SEVERITIES = (ERROR, WARNING)

ArtifactReader = Callable[[str], "bytes | None"]
ArtifactRegistry = Callable[[str], "ArtifactRef | None"]
Resolver = Callable[[str], "Record | None"]

SUBJECT_TYPES: tuple[str, ...] = ("commit", "baseline")
ORIGIN_RELATION_KINDS: tuple[str, ...] = ("optimization_origin", "rebased_from")
UNMEASURED_TIMING_STATUSES: tuple[str, ...] = ("not_run", "error")
TRUSTED_PROVENANCE = "trusted_worker"


# --------------------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------------------
@dataclass
class Issue:
    severity: str
    code: str
    message: str
    record_id: str | None
    field: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
            "field": self.field,
            "details": dict(self.details),
        }


@dataclass
class ValidationReport:
    records_checked: int = 0
    runs_checked: int = 0
    artifact_checks: int = 0
    summary_checks: int = 0
    issues: list[Issue] = dataclasses.field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not any(i.severity == ERROR for i in self.issues)

    def codes(self, severity: str | None = None) -> set[str]:
        return {i.code for i in self.issues if severity is None or i.severity == severity}

    def add(self, severity: str, code: str, message: str, record_id: str | None, field: str | None = None, **details: Any) -> Issue:
        issue = Issue(severity=severity, code=code, message=message, record_id=record_id, field=field, details=details)
        self.issues.append(issue)
        return issue

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "records_checked": self.records_checked,
            "runs_checked": self.runs_checked,
            "artifact_checks": self.artifact_checks,
            "summary_checks": self.summary_checks,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _oid_key(oid: GitOid) -> tuple[str, str]:
    return (oid.algorithm, oid.hex)


def _oid_text(oid: GitOid) -> str:
    return f"{oid.algorithm}:{oid.hex}"


T = TypeVar("T", bound=Hashable)


def find_cycle(edges: dict[T, set[T]]) -> list[T] | None:
    """Return one directed cycle (closed path) in ``edges`` or ``None``. Deterministic order."""
    white, gray, black = 0, 1, 2
    color: dict[T, int] = {}

    def ordered(node: T) -> list[T]:
        return sorted(edges.get(node, ()), key=repr)

    for start in sorted(edges, key=repr):
        if color.get(start, white) != white:
            continue
        color[start] = gray
        path: list[T] = [start]
        stack: list[tuple[T, list[T], int]] = [(start, ordered(start), 0)]
        while stack:
            node, children, pos = stack[-1]
            if pos < len(children):
                stack[-1] = (node, children, pos + 1)
                child = children[pos]
                state = color.get(child, white)
                if state == gray:
                    idx = path.index(child)
                    return path[idx:] + [child]
                if state == white:
                    color[child] = gray
                    path.append(child)
                    stack.append((child, ordered(child), 0))
            else:
                color[node] = black
                stack.pop()
                path.pop()
    return None


# --------------------------------------------------------------------------------------
# The validator
# --------------------------------------------------------------------------------------
class _Validator:
    def __init__(
        self,
        records: list[Record],
        *,
        artifact_reader: ArtifactReader,
        artifact_registry: ArtifactRegistry | None,
        registry: ProblemRegistry,
        external_resolver: Resolver | None,
        kernel_resolver: Resolver | None,
        verify_artifacts: bool,
        missing_evidence_severity: str,
    ) -> None:
        self.reader = artifact_reader
        self.artifact_registry = artifact_registry
        self.registry = registry
        self.external_resolver = external_resolver
        self.kernel_resolver = kernel_resolver
        self.verify_artifacts = verify_artifacts
        self.missing_severity = missing_evidence_severity
        self.report = ValidationReport(records_checked=len(records))

        self.by_id: dict[str, Record] = {}
        for record in records:
            rid = record.record_id
            if rid in self.by_id:
                identical = self.by_id[rid].canonical_digest() == record.canonical_digest()
                self.error(
                    "DUPLICATE_RECORD_ID",
                    f"record id {rid!r} appears more than once in the set ({'identical' if identical else 'different'} content)",
                    rid,
                    identical_content=identical,
                )
                continue
            self.by_id[rid] = record
        self.records: list[Record] = list(self.by_id.values())
        self.runs: list[Record] = [r for r in self.records if r.record_type == "run"]
        self.kernels_by_kernel_id: dict[str, Record] = {}
        self.annotations_by_target: dict[str, list[Record]] = {}
        self.relations_by_endpoint: dict[str, list[Record]] = {}
        for record in self.records:
            if record.record_type == "kernel":
                self.kernels_by_kernel_id.setdefault(record.payload.kernel_id, record)
            elif record.record_type == "annotation":
                self.annotations_by_target.setdefault(record.payload.target_ref, []).append(record)
            elif record.record_type == "relation":
                self.relations_by_endpoint.setdefault(record.payload.from_ref, []).append(record)
                self.relations_by_endpoint.setdefault(record.payload.to_ref, []).append(record)
        self.artifact_catalog: dict[str, tuple[str, ArtifactRef]] = {}
        self._artifact_bytes: dict[str, bytes | None] = {}
        self._artifact_reader_errors: dict[str, str] = {}
        self._reported_cycles: set[frozenset[str]] = set()
        self._pr_keys: dict[tuple[str, str], str] = {}
        self._baseline_ids: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------ issue helpers
    def error(self, code: str, message: str, record_id: str | None, field: str | None = None, **details: Any) -> Issue:
        return self.report.add(ERROR, code, message, record_id, field, **details)

    def warning(self, code: str, message: str, record_id: str | None, field: str | None = None, **details: Any) -> Issue:
        return self.report.add(WARNING, code, message, record_id, field, **details)

    # ------------------------------------------------------------------ resolution
    def resolve(self, record_id: str) -> Record | None:
        record = self.by_id.get(record_id)
        if record is None and self.external_resolver is not None:
            try:
                record = self.external_resolver(record_id)
            except KernelMemoryError:
                record = None
        return record

    def resolve_typed(self, record_id: str, *types: str) -> Record | None:
        record = self.resolve(record_id)
        if record is None or (types and record.record_type not in types):
            return None
        return record

    def find_kernel(self, kernel_id: str) -> Record | None:
        kernel = self.kernels_by_kernel_id.get(kernel_id)
        if kernel is not None:
            return kernel
        if self.kernel_resolver is not None:
            try:
                kernel = self.kernel_resolver(kernel_id)
            except KernelMemoryError:
                kernel = None
            if kernel is not None and kernel.record_type == "kernel":
                return kernel
        # Conventional id (DESIGN section 4) as a last resort.
        candidate = self.resolve_typed(f"kernel-{kernel_id}", "kernel")
        if candidate is not None and candidate.payload.kernel_id == kernel_id:
            return candidate
        return None

    def lookup_artifact(self, artifact_id: str) -> ArtifactRef | None:
        entry = self.artifact_catalog.get(artifact_id)
        if entry is not None:
            return entry[1]
        if self.artifact_registry is not None:
            try:
                return self.artifact_registry(artifact_id)
            except KernelMemoryError:
                return None
        return None

    def config_of_subject(self, subject: Record) -> str | None:
        if subject.record_type == "baseline":
            return subject.payload.config_ref
        if subject.record_type == "commit":
            pr = self.resolve_typed(subject.payload.pr_ref, "pr")
            return pr.payload.config_ref if pr is not None else None
        return None

    # ------------------------------------------------------------------ entry point
    def run(self) -> ValidationReport:
        self._check_kernel_ids()
        self._build_artifact_catalog()
        handlers = {
            "kernel": self._check_kernel,
            "config": self._check_config,
            "pr": self._check_pr,
            "pr_snapshot": self._check_pr_snapshot,
            "commit": self._check_commit,
            "baseline": self._check_baseline,
            "run": self._check_run,
            "relation": self._check_relation,
            "decision": self._check_decision,
            "annotation": self._check_annotation,
        }
        for record in self.records:
            self._check_references(record)
            handlers[record.record_type](record)
        self._check_run_attempts()
        self._check_origin_dag()
        self._check_git_parents()
        return self.report

    # ------------------------------------------------------------------ set-level checks
    def _check_kernel_ids(self) -> None:
        seen: dict[str, str] = {}
        for record in self.records:
            if record.record_type != "kernel":
                continue
            kid = record.payload.kernel_id
            if kid in seen:
                self.error(
                    "DUPLICATE_KERNEL_ID",
                    f"kernel_id {kid!r} is declared by both {seen[kid]!r} and {record.record_id!r}",
                    record.record_id,
                    "kernel_id",
                    other_record=seen[kid],
                )
            else:
                seen[kid] = record.record_id

    def _build_artifact_catalog(self) -> None:
        for run in self.runs:
            seen_in_run: set[str] = set()
            for i, ref in enumerate(run.payload.artifacts):
                field = f"artifacts[{i}]"
                if ref.artifact_id in seen_in_run:
                    self.error(
                        "DUPLICATE_ARTIFACT_ID",
                        f"run {run.record_id!r} declares artifact_id {ref.artifact_id!r} more than once",
                        run.record_id,
                        field,
                        artifact_id=ref.artifact_id,
                    )
                    continue
                seen_in_run.add(ref.artifact_id)
                prior = self.artifact_catalog.get(ref.artifact_id)
                if prior is not None:
                    if to_json(prior[1]) != to_json(ref):
                        self.error(
                            "ARTIFACT_DESCRIPTOR_CONFLICT",
                            f"artifact {ref.artifact_id!r} in run {run.record_id!r} differs from the descriptor declared by run {prior[0]!r}",
                            run.record_id,
                            field,
                            artifact_id=ref.artifact_id,
                            other_run=prior[0],
                            other=to_json(prior[1]),
                            this=to_json(ref),
                        )
                    continue
                registered: ArtifactRef | None = None
                if self.artifact_registry is not None:
                    try:
                        registered = self.artifact_registry(ref.artifact_id)
                    except KernelMemoryError as exc:
                        self.error(
                            "ARTIFACT_DESCRIPTOR_CONFLICT",
                            f"artifact registry entry for {ref.artifact_id!r} is unreadable: {exc}",
                            run.record_id,
                            field,
                            artifact_id=ref.artifact_id,
                        )
                if registered is not None and to_json(registered) != to_json(ref):
                    self.error(
                        "ARTIFACT_DESCRIPTOR_CONFLICT",
                        f"artifact {ref.artifact_id!r} in run {run.record_id!r} differs from the registered descriptor",
                        run.record_id,
                        field,
                        artifact_id=ref.artifact_id,
                        registered=to_json(registered),
                        this=to_json(ref),
                    )
                self.artifact_catalog[ref.artifact_id] = (run.record_id, ref)

    def _check_references(self, record: Record) -> None:
        for ref in record.references():
            if ref.allowed_types == ("artifact",):
                if self.lookup_artifact(ref.target) is None:
                    self.error(
                        "MISSING_REFERENCE",
                        f"{record.record_type} {record.record_id!r} field {ref.field} refers to unknown artifact {ref.target!r}",
                        record.record_id,
                        ref.field,
                        target=ref.target,
                        target_kind="artifact",
                    )
                continue
            target = self.resolve(ref.target)
            if target is None:
                self.error(
                    "MISSING_REFERENCE",
                    f"{record.record_type} {record.record_id!r} field {ref.field} refers to missing record {ref.target!r}",
                    record.record_id,
                    ref.field,
                    target=ref.target,
                    allowed_types=list(ref.allowed_types),
                )
            elif target.record_type not in ref.allowed_types:
                self.error(
                    "WRONG_REFERENCE_TYPE",
                    f"{record.record_type} {record.record_id!r} field {ref.field} refers to {ref.target!r} of type "
                    f"{target.record_type!r}; expected one of {list(ref.allowed_types)}",
                    record.record_id,
                    ref.field,
                    target=ref.target,
                    target_type=target.record_type,
                    allowed_types=list(ref.allowed_types),
                )

    def _check_run_attempts(self) -> None:
        seen: dict[tuple[str, int], str] = {}
        for run in self.runs:
            key = (run.payload.request_id, run.payload.attempt_no)
            if key in seen:
                self.error(
                    "DUPLICATE_ATTEMPT",
                    f"runs {seen[key]!r} and {run.record_id!r} both claim request {key[0]!r} attempt {key[1]}",
                    run.record_id,
                    "attempt_no",
                    request_id=key[0],
                    attempt_no=key[1],
                    other_run=seen[key],
                )
            else:
                seen[key] = run.record_id

    def _check_origin_dag(self) -> None:
        edges: dict[str, set[str]] = {}
        for record in self.records:
            p = record.payload
            if record.record_type == "relation" and p.kind == "optimization_origin":
                edges.setdefault(p.from_ref, set()).add(p.to_ref)
            elif record.record_type == "pr" and p.origin_ref is not None:
                edges.setdefault(p.origin_ref, set()).add(record.record_id)
            elif record.record_type == "commit":
                edges.setdefault(p.pr_ref, set()).add(record.record_id)
        cycle = find_cycle(edges)
        if cycle is not None:
            self.error(
                "ORIGIN_CYCLE",
                "optimization-origin graph (relations plus PR origin_ref edges) contains a cycle: " + " -> ".join(cycle),
                None,
                None,
                cycle=list(cycle),
            )

    def _check_git_parents(self) -> None:
        edges: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        declared: dict[tuple[str, str, str], tuple[tuple[tuple[str, str], ...], str]] = {}
        owners: dict[tuple[str, str, str], list[str]] = {}
        for record in self.records:
            if record.record_type != "commit":
                continue
            p = record.payload
            node = (p.repo_uid,) + _oid_key(p.commit_oid)
            parents = tuple(sorted(_oid_key(o) for o in p.git_parent_oids))
            owners.setdefault(node, []).append(record.record_id)
            prior = declared.get(node)
            if prior is None:
                declared[node] = (parents, record.record_id)
            elif prior[0] != parents:
                self.error(
                    "PARENT_OIDS_INCONSISTENT",
                    f"commit bindings {prior[1]!r} and {record.record_id!r} describe the same source commit "
                    f"{_oid_text(p.commit_oid)} with different git parents",
                    record.record_id,
                    "git_parent_oids",
                    other_record=prior[1],
                    repo_uid=p.repo_uid,
                )
            edges.setdefault(node, set()).update((p.repo_uid,) + _oid_key(o) for o in p.git_parent_oids)
        cycle = find_cycle(edges)
        if cycle is not None:
            involved = sorted({rid for node in cycle for rid in owners.get(node, [])})
            self.error(
                "GIT_PARENT_CYCLE",
                "git parent graph contains a cycle: " + " -> ".join(f"{n[1]}:{n[2]}" for n in cycle),
                involved[0] if involved else None,
                "git_parent_oids",
                repo_uid=cycle[0][0],
                oids=[f"{n[1]}:{n[2]}" for n in cycle],
                record_ids=involved,
            )

    # ------------------------------------------------------------------ per-type checks
    def _check_kernel(self, record: Record) -> None:
        return None

    def _check_config(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        if self.find_kernel(p.kernel_id) is None:
            self.error(
                "MISSING_KERNEL",
                f"config {rid!r} refers to kernel_id {p.kernel_id!r} but no kernel record declares it",
                rid,
                "kernel_id",
                kernel_id=p.kernel_id,
            )
        hash_ok = True
        try:
            expected = hashing.config_hash(
                kernel_id=p.kernel_id,
                problem_schema_id=p.problem_schema_id,
                problem_schema_digest=p.problem_schema_digest,
                problem=p.problem,
            )
        except KernelMemoryError as exc:
            hash_ok = False
            self.error("CONFIG_HASH_MISMATCH", f"config {rid!r}: config_hash cannot be recomputed: {exc}", rid, "config_hash")
        else:
            if expected != p.config_hash:
                hash_ok = False
                self.error(
                    "CONFIG_HASH_MISMATCH",
                    f"config {rid!r}: recorded config_hash {p.config_hash} but the identity object hashes to {expected}",
                    rid,
                    "config_hash",
                    recorded=p.config_hash,
                    computed=expected,
                )
        try:
            self.registry.verify_config_payload(to_json(p))
        except KernelMemoryError as exc:
            if not hash_ok and "computed" in exc.details:
                return  # the hash mismatch was already reported above
            extra = {k: v for k, v in exc.details.items() if k not in ("record_id", "record_type", "error_code")}
            self.error(
                "CONFIG_PROBLEM_INVALID",
                f"config {rid!r}: {exc.message}",
                rid,
                "problem",
                error_code=exc.code,
                **extra,
            )

    def _check_pr(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        if (p.provider == "github") != (p.number is not None):
            self.error(
                "PROVIDER_NUMBER_MISMATCH",
                f"pr {rid!r}: provider {p.provider!r} requires number to be "
                f"{'set' if p.provider == 'github' else 'null'} (got {p.number!r})",
                rid,
                "number",
                provider=p.provider,
                number=p.number,
            )
        key = (p.config_ref, p.pr_key)
        if key in self._pr_keys:
            self.error(
                "DUPLICATE_PR_KEY",
                f"pr_key {p.pr_key!r} under config {p.config_ref!r} is declared by both {self._pr_keys[key]!r} and {rid!r}",
                rid,
                "pr_key",
                other_record=self._pr_keys[key],
            )
        else:
            self._pr_keys[key] = rid
        if p.provider == "github" and p.number is not None and p.repo_uid.startswith("github:"):
            repo_id = p.repo_uid.rsplit(":", 1)[-1]
            expected = f"gh-{repo_id}-pr-{p.number}"
            if p.pr_key != expected:
                self.warning(
                    "PR_KEY_CONVENTION",
                    f"pr {rid!r}: pr_key {p.pr_key!r} does not follow the convention {expected!r}",
                    rid,
                    "pr_key",
                    expected=expected,
                )
        elif p.provider == "local" and not p.pr_key.startswith("local-"):
            self.warning("PR_KEY_CONVENTION", f"pr {rid!r}: local trial pr_key {p.pr_key!r} should start with 'local-'", rid, "pr_key")
        if p.origin_ref is not None:
            origin = self.resolve_typed(p.origin_ref, *SUBJECT_TYPES)
            if origin is not None:
                origin_cfg = self.config_of_subject(origin)
                if origin_cfg is not None and origin_cfg != p.config_ref:
                    self.warning(
                        "ORIGIN_CROSS_CONFIG",
                        f"pr {rid!r}: origin {p.origin_ref!r} belongs to config {origin_cfg!r}, not {p.config_ref!r}",
                        rid,
                        "origin_ref",
                        origin_config=origin_cfg,
                    )

    def _check_pr_snapshot(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        if p.previous_snapshot_ref == rid:
            self.error("SELF_REFERENCE", f"pr_snapshot {rid!r} lists itself as previous_snapshot_ref", rid, "previous_snapshot_ref")
        elif p.previous_snapshot_ref is not None:
            prev = self.resolve_typed(p.previous_snapshot_ref, "pr_snapshot")
            if prev is not None and prev.payload.pr_ref != p.pr_ref:
                self.error(
                    "SNAPSHOT_CHAIN_MISMATCH",
                    f"pr_snapshot {rid!r}: previous snapshot {p.previous_snapshot_ref!r} belongs to pr {prev.payload.pr_ref!r}, not {p.pr_ref!r}",
                    rid,
                    "previous_snapshot_ref",
                    previous_pr=prev.payload.pr_ref,
                )
            self._check_snapshot_chain(record)
        seen: set[str] = set()
        for i, cid in enumerate(p.commit_refs):
            field = f"commit_refs[{i}]"
            if cid in seen:
                self.error("DUPLICATE_COMMIT_REF", f"pr_snapshot {rid!r} lists commit {cid!r} twice", rid, field, commit=cid)
                continue
            seen.add(cid)
            commit = self.resolve_typed(cid, "commit")
            if commit is not None and commit.payload.pr_ref != p.pr_ref:
                self.error(
                    "CROSS_PR_MEMBERSHIP",
                    f"pr_snapshot {rid!r} of pr {p.pr_ref!r} lists commit binding {cid!r} that belongs to pr {commit.payload.pr_ref!r}",
                    rid,
                    field,
                    commit=cid,
                    commit_pr=commit.payload.pr_ref,
                )
        if p.enumeration_status in ("partial", "unavailable") and not p.reason:
            self.error(
                "MISSING_ENUMERATION_REASON",
                f"pr_snapshot {rid!r}: enumeration_status {p.enumeration_status!r} requires a reason",
                rid,
                "reason",
                enumeration_status=p.enumeration_status,
            )

    def _check_snapshot_chain(self, record: Record) -> None:
        path: list[str] = [record.record_id]
        seen = {record.record_id}
        current = record
        while current.payload.previous_snapshot_ref is not None:
            nxt = self.by_id.get(current.payload.previous_snapshot_ref)
            if nxt is None or nxt.record_type != "pr_snapshot":
                return
            if nxt.record_id in seen:
                idx = path.index(nxt.record_id)
                cycle = path[idx:]
                key = frozenset(cycle)
                if key not in self._reported_cycles:
                    self._reported_cycles.add(key)
                    self.error(
                        "SNAPSHOT_CHAIN_CYCLE",
                        "pr_snapshot chain contains a cycle: " + " -> ".join(cycle + [nxt.record_id]),
                        min(cycle),
                        "previous_snapshot_ref",
                        cycle=cycle,
                    )
                return
            seen.add(nxt.record_id)
            path.append(nxt.record_id)
            current = nxt

    def _check_commit(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        pr = self.resolve_typed(p.pr_ref, "pr")
        if pr is not None and pr.payload.repo_uid != p.repo_uid:
            self.error(
                "REPO_UID_MISMATCH",
                f"commit {rid!r}: repo_uid {p.repo_uid!r} differs from its pr {p.pr_ref!r} repo_uid {pr.payload.repo_uid!r}",
                rid,
                "repo_uid",
                pr_repo_uid=pr.payload.repo_uid,
            )
        if p.change_status in ("not_extracted", "no_code_change") and p.changes:
            self.error(
                "UNEXTRACTED_CHANGES_NOT_EMPTY",
                f"commit {rid!r}: change_status {p.change_status!r} requires an empty changes list (got {len(p.changes)})",
                rid,
                "changes",
                change_status=p.change_status,
            )
        if p.change_status == "recorded":
            seen: set[str] = set()
            for i, change in enumerate(p.changes):
                if change.change_id in seen:
                    self.error("DUPLICATE_CHANGE_ID", f"commit {rid!r}: change_id {change.change_id!r} is not unique", rid, f"changes[{i}].change_id")
                seen.add(change.change_id)
            isolated = [c.change_id for c in p.changes if c.attribution == "isolated"]
            if isolated and not self._has_attribution_evidence(rid):
                self.warning(
                    "ISOLATED_ATTRIBUTION_UNSUPPORTED",
                    f"commit {rid!r} claims isolated attribution for {isolated} without any annotation or relation carrying evidence",
                    rid,
                    "changes",
                    change_ids=isolated,
                )
        if p.diff_base_oid is not None:
            parents = {_oid_key(o) for o in p.git_parent_oids}
            if _oid_key(p.diff_base_oid) not in parents:
                self.warning(
                    "DIFF_BASE_NOT_PARENT",
                    f"commit {rid!r}: diff_base_oid {_oid_text(p.diff_base_oid)} is not one of the git parents",
                    rid,
                    "diff_base_oid",
                    parents=[_oid_text(o) for o in p.git_parent_oids],
                )

    def _has_attribution_evidence(self, commit_id: str) -> bool:
        for annotation in self.annotations_by_target.get(commit_id, []):
            if annotation.payload.evidence_refs:
                return True
        for relation in self.relations_by_endpoint.get(commit_id, []):
            if relation.payload.evidence_refs:
                return True
        return False

    def _check_baseline(self, record: Record) -> None:
        p = record.payload
        key = (p.config_ref, p.baseline_id)
        if key in self._baseline_ids:
            self.error(
                "DUPLICATE_BASELINE_ID",
                f"baseline_id {p.baseline_id!r} under config {p.config_ref!r} is declared by both {self._baseline_ids[key]!r} and {record.record_id!r}",
                record.record_id,
                "baseline_id",
                other_record=self._baseline_ids[key],
            )
        else:
            self._baseline_ids[key] = record.record_id

    def _check_relation(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        if p.from_ref == p.to_ref:
            self.error("SELF_RELATION", f"relation {rid!r} relates {p.from_ref!r} to itself", rid, "to_ref", target=p.from_ref)
        for field, target_id in (("from_ref", p.from_ref), ("to_ref", p.to_ref)):
            target = self.resolve(target_id)
            if target is None:
                continue  # MISSING_REFERENCE already reported
            if p.kind in ORIGIN_RELATION_KINDS and target.record_type not in SUBJECT_TYPES:
                self.error(
                    "RELATION_ENDPOINT_TYPE",
                    f"relation {rid!r} of kind {p.kind!r}: {field} {target_id!r} is a {target.record_type}, expected commit or baseline",
                    rid,
                    field,
                    kind=p.kind,
                    target_type=target.record_type,
                )
            elif target.record_type in SUBJECT_TYPES:
                cfg = self.config_of_subject(target)
                if cfg is not None and cfg != p.config_ref:
                    self.warning(
                        "RELATION_CROSS_CONFIG",
                        f"relation {rid!r}: {field} {target_id!r} belongs to config {cfg!r}, not {p.config_ref!r}",
                        rid,
                        field,
                        endpoint_config=cfg,
                    )

    def _check_annotation(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        if p.supersedes_ref == rid:
            self.error("SELF_REFERENCE", f"annotation {rid!r} supersedes itself", rid, "supersedes_ref")
        seen: set[str] = set()
        for i, evidence in enumerate(p.evidence_refs):
            if evidence in seen:
                self.warning("DUPLICATE_EVIDENCE_REF", f"annotation {rid!r} lists evidence {evidence!r} twice", rid, f"evidence_refs[{i}]")
            seen.add(evidence)

    # ------------------------------------------------------------------ runs
    def _check_run(self, record: Record) -> None:
        self.report.runs_checked += 1
        p = record.payload
        rid = record.record_id
        src = p.source

        subject = self.resolve_typed(p.subject_ref, *SUBJECT_TYPES)
        if subject is not None:
            if _oid_key(subject.payload.commit_oid) != _oid_key(src.target_commit):
                self.error(
                    "TESTED_SOURCE_MISMATCH",
                    f"run {rid!r}: source.target_commit {_oid_text(src.target_commit)} is not the subject's commit {_oid_text(subject.payload.commit_oid)}",
                    rid,
                    "source.target_commit",
                    subject=p.subject_ref,
                    subject_commit=_oid_text(subject.payload.commit_oid),
                    target_commit=_oid_text(src.target_commit),
                )
            if subject.payload.repo_uid != src.repo_uid:
                self.error(
                    "REPO_UID_MISMATCH",
                    f"run {rid!r}: source.repo_uid {src.repo_uid!r} differs from the subject's repo_uid {subject.payload.repo_uid!r}",
                    rid,
                    "source.repo_uid",
                    subject_repo_uid=subject.payload.repo_uid,
                )
            subject_cfg = self.config_of_subject(subject)
            if subject_cfg is not None and subject_cfg != p.config_ref:
                self.error(
                    "RUN_CONFIG_MISMATCH",
                    f"run {rid!r}: config_ref {p.config_ref!r} differs from the subject's config {subject_cfg!r}",
                    rid,
                    "config_ref",
                    subject_config=subject_cfg,
                )

        if src.checkout_mode == "exact_commit":
            if _oid_key(src.tested_commit) != _oid_key(src.target_commit):
                self.error(
                    "EXACT_COMMIT_MISMATCH",
                    f"run {rid!r}: checkout_mode exact_commit but tested_commit {_oid_text(src.tested_commit)} != target_commit {_oid_text(src.target_commit)}",
                    rid,
                    "source.tested_commit",
                    tested_commit=_oid_text(src.tested_commit),
                    target_commit=_oid_text(src.target_commit),
                )
            if src.merge_parent_oids:
                self.error(
                    "EXACT_COMMIT_MISMATCH",
                    f"run {rid!r}: checkout_mode exact_commit must not carry merge_parent_oids",
                    rid,
                    "source.merge_parent_oids",
                    merge_parent_oids=[_oid_text(o) for o in src.merge_parent_oids],
                )
        elif src.checkout_mode == "integration_merge":
            if _oid_key(src.tested_commit) == _oid_key(src.target_commit):
                self.error(
                    "INTEGRATION_MERGE_IDENTITY",
                    f"run {rid!r}: checkout_mode integration_merge but tested_commit equals target_commit {_oid_text(src.target_commit)}",
                    rid,
                    "source.tested_commit",
                )
            if not src.merge_parent_oids:
                self.warning("INTEGRATION_MERGE_WITHOUT_PARENTS", f"run {rid!r}: integration_merge without merge_parent_oids", rid, "source.merge_parent_oids")

        try:
            expected_variant = hashing.variant_digest(
                source_digest=src.source_digest,
                entrypoint=src.entrypoint,
                implementation_overrides=src.implementation_overrides,
                checkout_mode=src.checkout_mode,
            )
        except KernelMemoryError as exc:
            self.error("VARIANT_HASH_MISMATCH", f"run {rid!r}: variant_digest cannot be recomputed: {exc}", rid, "source.variant_digest")
        else:
            if expected_variant != src.variant_digest:
                self.error(
                    "VARIANT_HASH_MISMATCH",
                    f"run {rid!r}: recorded variant_digest {src.variant_digest} but the variant identity hashes to {expected_variant}",
                    rid,
                    "source.variant_digest",
                    recorded=src.variant_digest,
                    computed=expected_variant,
                )

        for name, fn in (("environment", hashing.environment_hash), ("protocol", hashing.protocol_hash), ("verifier", hashing.verifier_hash)):
            snapshot = to_json(getattr(p, name))
            recorded = snapshot[f"{name}_hash"]
            try:
                computed = fn(snapshot)
            except KernelMemoryError as exc:
                self.error("SNAPSHOT_HASH_MISMATCH", f"run {rid!r}: {name}_hash cannot be recomputed: {exc}", rid, name)
                continue
            if computed != recorded:
                self.error(
                    "SNAPSHOT_HASH_MISMATCH",
                    f"run {rid!r}: recorded {name}_hash {recorded} but the {name} snapshot hashes to {computed}",
                    rid,
                    f"{name}.{name}_hash",
                    snapshot=name,
                    recorded=recorded,
                    computed=computed,
                )

        config = self.resolve_typed(p.config_ref, "config")
        if config is not None:
            expected_key = hashing.comparison_key(
                config_hash=config.payload.config_hash,
                environment_hash=p.environment.environment_hash,
                protocol_hash=p.protocol.protocol_hash,
                verifier_hash=p.verifier.verifier_hash,
                checkout_mode=src.checkout_mode,
            )
            if expected_key != p.comparison_key:
                self.error(
                    "COMPARISON_KEY_MISMATCH",
                    f"run {rid!r}: recorded comparison_key {p.comparison_key} but the comparability identity hashes to {expected_key}",
                    rid,
                    "comparison_key",
                    recorded=p.comparison_key,
                    computed=expected_key,
                )

        if src.dirty and src.patch_digest is None:
            self.error("DIRTY_WITHOUT_PATCH", f"run {rid!r}: dirty source requires a patch_digest", rid, "source.patch_digest")
        elif not src.dirty and src.patch_digest is not None:
            self.warning("PATCH_WITHOUT_DIRTY", f"run {rid!r}: patch_digest recorded although dirty=false", rid, "source.patch_digest")

        local = {a.artifact_id: a for a in p.artifacts}
        loaded = self._load_run_artifacts(record)

        def in_run(value: str | None, field: str) -> bool:
            if value is None:
                return False
            if value not in local:
                self.error(
                    "DANGLING_IN_RUN_REF",
                    f"run {rid!r}: {field} refers to artifact {value!r} that is not in run.artifacts",
                    rid,
                    field,
                    artifact_id=value,
                )
                return False
            return True

        # correctness ------------------------------------------------------------
        c = p.correctness
        if c.cases_passed > c.cases_total:
            self.error(
                "INVALID_CORRECTNESS_COUNTS",
                f"run {rid!r}: cases_passed {c.cases_passed} exceeds cases_total {c.cases_total}",
                rid,
                "correctness.cases_passed",
            )
        if c.status == "pass":
            if c.cases_total <= 0 or c.cases_passed != c.cases_total:
                self.error(
                    "INVALID_PASS",
                    f"run {rid!r}: correctness pass requires cases_total > 0 and cases_passed == cases_total (got {c.cases_passed}/{c.cases_total})",
                    rid,
                    "correctness",
                    cases_total=c.cases_total,
                    cases_passed=c.cases_passed,
                )
            if c.report_artifact_ref is None:
                self.error("MISSING_CORRECTNESS_EVIDENCE", f"run {rid!r}: correctness pass without report_artifact_ref", rid, "correctness.report_artifact_ref")
            else:
                in_run(c.report_artifact_ref, "correctness.report_artifact_ref")
        elif c.status == "fail":
            if c.cases_total <= 0 or c.cases_passed >= c.cases_total:
                self.error(
                    "INVALID_FAIL",
                    f"run {rid!r}: correctness fail requires cases_passed < cases_total with cases_total > 0 (got {c.cases_passed}/{c.cases_total})",
                    rid,
                    "correctness",
                    cases_total=c.cases_total,
                    cases_passed=c.cases_passed,
                )
            in_run(c.report_artifact_ref, "correctness.report_artifact_ref")
        elif c.status == "not_run":
            if c.cases_total != 0 or c.cases_passed != 0 or c.max_abs_error is not None or c.max_rel_error is not None or c.report_artifact_ref is not None:
                self.error(
                    "CORRECTNESS_NOT_RUN_WITH_RESULTS",
                    f"run {rid!r}: correctness not_run must have zero cases, null errors and no report",
                    rid,
                    "correctness",
                )
        else:
            in_run(c.report_artifact_ref, "correctness.report_artifact_ref")

        # timing -----------------------------------------------------------------
        t = p.timing
        if t.status == "recorded":
            if t.median_us is None or t.p90_us is None:
                self.error("SUMMARY_MISSING", f"run {rid!r}: recorded timing requires median_us and p90_us", rid, "timing")
            if t.sample_count != p.protocol.repetitions:
                self.error(
                    "SAMPLE_COUNT_MISMATCH",
                    f"run {rid!r}: timing.sample_count {t.sample_count} != protocol.repetitions {p.protocol.repetitions}",
                    rid,
                    "timing.sample_count",
                    sample_count=t.sample_count,
                    repetitions=p.protocol.repetitions,
                )
            if t.samples_artifact_ref is None:
                self.error("MISSING_TIMING_EVIDENCE", f"run {rid!r}: recorded timing without samples_artifact_ref", rid, "timing.samples_artifact_ref")
            elif in_run(t.samples_artifact_ref, "timing.samples_artifact_ref"):
                self._check_samples(record, local[t.samples_artifact_ref], loaded.get(t.samples_artifact_ref))
        elif t.status in UNMEASURED_TIMING_STATUSES:
            if t.sample_count != 0 or t.median_us is not None or t.p90_us is not None or t.samples_artifact_ref is not None:
                self.error(
                    "TIMING_NOT_RUN_WITH_RESULTS",
                    f"run {rid!r}: timing {t.status} must have sample_count 0, null summaries and no samples artifact",
                    rid,
                    "timing",
                    status=t.status,
                )

        # execution status -------------------------------------------------------
        if p.execution_status != "succeeded":
            if c.status != "not_run" or t.status != "not_run":
                self.error(
                    "FABRICATED_RESULT",
                    f"run {rid!r}: execution_status {p.execution_status!r} but correctness {c.status!r} / timing {t.status!r} carry results",
                    rid,
                    "execution_status",
                    execution_status=p.execution_status,
                    correctness_status=c.status,
                    timing_status=t.status,
                )
            if p.failure_reason is None:
                self.warning("MISSING_FAILURE_REASON", f"run {rid!r}: {p.execution_status} without failure_reason", rid, "failure_reason")
        elif p.failure_reason is not None:
            self.warning("FAILURE_REASON_ON_SUCCESS", f"run {rid!r}: failure_reason recorded on a succeeded run", rid, "failure_reason")

        # metrics ----------------------------------------------------------------
        seen_metrics: set[tuple[str, str]] = set()
        for i, m in enumerate(p.analysis_metrics):
            field = f"analysis_metrics[{i}]"
            key = (m.name, m.scope)
            if key in seen_metrics:
                self.warning("DUPLICATE_METRIC", f"run {rid!r}: metric {m.name!r} with scope {m.scope!r} appears twice", rid, field)
            seen_metrics.add(key)
            if m.status == "observed":
                if m.value is None or m.source_artifact_ref is None:
                    self.error(
                        "OBSERVED_METRIC_WITHOUT_EVIDENCE",
                        f"run {rid!r}: observed metric {m.name!r} requires a value and a source_artifact_ref",
                        rid,
                        field,
                        metric=m.name,
                        value=m.value,
                        source_artifact_ref=m.source_artifact_ref,
                    )
                    continue
                if isinstance(m.value, float) and not math.isfinite(m.value):
                    self.error("INVALID_METRIC_VALUE", f"run {rid!r}: observed metric {m.name!r} has a non-finite value", rid, f"{field}.value", metric=m.name)
                if in_run(m.source_artifact_ref, f"{field}.source_artifact_ref"):
                    ref = local[m.source_artifact_ref]
                    if ref.availability != "present":
                        self.warning(
                            "EVIDENCE_UNAVAILABLE",
                            f"run {rid!r}: metric {m.name!r} evidence artifact {ref.artifact_id!r} is {ref.availability}",
                            rid,
                            f"{field}.source_artifact_ref",
                            artifact_id=ref.artifact_id,
                            availability=ref.availability,
                        )
            elif m.value is not None:
                self.error(
                    "UNKNOWN_METRIC_NOT_NULL",
                    f"run {rid!r}: metric {m.name!r} has status {m.status!r} but value {m.value!r}; uncollected values must be null, never 0",
                    rid,
                    f"{field}.value",
                    metric=m.name,
                    status=m.status,
                    value=m.value,
                )

        if p.provenance == "fixture":
            self.warning("FIXTURE_RUN", f"run {rid!r} is a fixture (synthetic; never production evidence)", rid, "provenance")

        if p.rerun_of is not None:
            if p.rerun_of == rid:
                self.error("SELF_REFERENCE", f"run {rid!r} is a rerun of itself", rid, "rerun_of")
            else:
                other = self.resolve_typed(p.rerun_of, "run")
                if other is not None and other.payload.subject_ref != p.subject_ref:
                    self.warning(
                        "RERUN_SUBJECT_MISMATCH",
                        f"run {rid!r}: rerun_of {p.rerun_of!r} tested subject {other.payload.subject_ref!r}, not {p.subject_ref!r}",
                        rid,
                        "rerun_of",
                    )

    def _read_artifact(self, sha256_ref: str) -> bytes | None:
        if sha256_ref in self._artifact_bytes:
            return self._artifact_bytes[sha256_ref]
        try:
            data = self.reader(sha256_ref)
        except (KernelMemoryError, OSError) as exc:
            self._artifact_reader_errors[sha256_ref] = str(exc)
            data = None
        if data is not None and not isinstance(data, (bytes, bytearray)):
            self._artifact_reader_errors[sha256_ref] = f"reader returned {type(data).__name__}, expected bytes"
            data = None
        self._artifact_bytes[sha256_ref] = bytes(data) if data is not None else None
        return self._artifact_bytes[sha256_ref]

    def _load_run_artifacts(self, record: Record) -> dict[str, bytes | None]:
        """Verify presence/digest/size of every 'present' artifact; return id -> verified bytes."""
        out: dict[str, bytes | None] = {}
        rid = record.record_id
        for i, ref in enumerate(record.payload.artifacts):
            if ref.artifact_id in out:
                continue
            out[ref.artifact_id] = None
            if ref.availability != "present" or not self.verify_artifacts:
                continue
            field = f"artifacts[{i}]"
            self.report.artifact_checks += 1
            data = self._read_artifact(ref.sha256)
            if data is None:
                details: dict[str, Any] = {"artifact_id": ref.artifact_id, "sha256": ref.sha256, "uri": ref.uri}
                if ref.sha256 in self._artifact_reader_errors:
                    details["reader_error"] = self._artifact_reader_errors[ref.sha256]
                self.report.add(
                    self.missing_severity,
                    "MISSING_EVIDENCE",
                    f"run {rid!r}: artifact {ref.artifact_id!r} ({ref.sha256}) is declared present but its bytes are not available",
                    rid,
                    field,
                    **details,
                )
                continue
            digest = hashing.artifact_digest(data)
            if digest != ref.sha256:
                self.error(
                    "ARTIFACT_CORRUPT",
                    f"run {rid!r}: artifact {ref.artifact_id!r} bytes hash to {digest}, not the declared {ref.sha256}",
                    rid,
                    field,
                    artifact_id=ref.artifact_id,
                    declared=ref.sha256,
                    actual=digest,
                )
                continue
            if len(data) != ref.size_bytes:
                self.error(
                    "ARTIFACT_SIZE_MISMATCH",
                    f"run {rid!r}: artifact {ref.artifact_id!r} has {len(data)} bytes, declared {ref.size_bytes}",
                    rid,
                    field,
                    artifact_id=ref.artifact_id,
                    declared=ref.size_bytes,
                    actual=len(data),
                )
            out[ref.artifact_id] = data
        return out

    def _check_samples(self, record: Record, ref: ArtifactRef, data: bytes | None) -> None:
        rid = record.record_id
        t = record.payload.timing
        field = "timing.samples_artifact_ref"
        if ref.availability != "present":
            self.warning(
                "EVIDENCE_UNAVAILABLE",
                f"run {rid!r}: samples artifact {ref.artifact_id!r} is {ref.availability}; recorded summaries cannot be verified",
                rid,
                field,
                artifact_id=ref.artifact_id,
                availability=ref.availability,
            )
            return
        if not self.verify_artifacts or data is None:
            return  # skipped, or MISSING_EVIDENCE / ARTIFACT_CORRUPT already reported
        try:
            doc = loads_strict(data)
        except InputError as exc:
            self.error("INVALID_SAMPLES_ARTIFACT", f"run {rid!r}: samples artifact {ref.artifact_id!r} is not strict JSON: {exc.message}", rid, field, artifact_id=ref.artifact_id)
            return
        if not isinstance(doc, dict) or not isinstance(doc.get("samples"), list) or not isinstance(doc.get("unit"), str):
            self.error(
                "INVALID_SAMPLES_ARTIFACT",
                f"run {rid!r}: samples artifact {ref.artifact_id!r} must be an object with a 'samples' list and a 'unit' string",
                rid,
                field,
                artifact_id=ref.artifact_id,
            )
            return
        unit = doc["unit"]
        if unit not in stats.SUPPORTED_UNITS:
            self.error("UNKNOWN_UNIT", f"run {rid!r}: samples artifact declares unknown unit {unit!r}", rid, field, unit=unit, supported=sorted(stats.SUPPORTED_UNITS))
            return
        try:
            values = stats.validate_samples(doc["samples"])
        except InputError as exc:
            self.error("INVALID_SAMPLES", f"run {rid!r}: {exc.message}", rid, field, artifact_id=ref.artifact_id)
            return
        if len(values) != t.sample_count:
            self.error(
                "SAMPLE_COUNT_MISMATCH",
                f"run {rid!r}: samples artifact holds {len(values)} samples but timing.sample_count is {t.sample_count}",
                rid,
                "timing.sample_count",
                samples=len(values),
                sample_count=t.sample_count,
                repetitions=record.payload.protocol.repetitions,
            )
        if t.median_us is None or t.p90_us is None:
            return  # SUMMARY_MISSING already reported
        summary = stats.summarize(values, unit)
        self.report.summary_checks += 2
        if not stats.summaries_agree(values, unit, t.median_us, t.p90_us):
            self.error(
                "SUMMARY_MISMATCH",
                f"run {rid!r}: recorded median_us={t.median_us} p90_us={t.p90_us} but the samples give median_us={summary.median_us} p90_us={summary.p90_us}",
                rid,
                "timing",
                recorded_median_us=t.median_us,
                recorded_p90_us=t.p90_us,
                computed_median_us=summary.median_us,
                computed_p90_us=summary.p90_us,
                unit=unit,
            )

    # ------------------------------------------------------------------ decisions
    def _check_decision(self, record: Record) -> None:
        p = record.payload
        rid = record.record_id
        try:
            expected_policy = hashing.policy_hash(to_json(p.policy))
        except KernelMemoryError as exc:
            self.error("POLICY_HASH_MISMATCH", f"decision {rid!r}: policy_hash cannot be recomputed: {exc}", rid, "policy_hash")
        else:
            if expected_policy != p.policy_hash:
                self.error(
                    "POLICY_HASH_MISMATCH",
                    f"decision {rid!r}: recorded policy_hash {p.policy_hash} but the policy hashes to {expected_policy}",
                    rid,
                    "policy_hash",
                    recorded=p.policy_hash,
                    computed=expected_policy,
                )
        if p.supersedes_decision_ref == rid:
            self.error("SELF_REFERENCE", f"decision {rid!r} supersedes itself", rid, "supersedes_decision_ref")

        runs: list[Record] = []
        for list_name, refs in (("candidate_run_refs", p.candidate_run_refs), ("baseline_run_refs", p.baseline_run_refs)):
            for i, run_id in enumerate(refs):
                field = f"{list_name}[{i}]"
                run = self.resolve_typed(run_id, "run")
                if run is None:
                    continue  # MISSING_REFERENCE / WRONG_REFERENCE_TYPE already reported
                runs.append(run)
                if run.payload.comparison_key != p.comparison_key:
                    self.error(
                        "DECISION_GROUP_MISMATCH",
                        f"decision {rid!r}: run {run_id!r} has comparison_key {run.payload.comparison_key}, decision has {p.comparison_key}",
                        rid,
                        field,
                        run=run_id,
                        run_comparison_key=run.payload.comparison_key,
                    )
                if run.payload.config_ref != p.config_ref:
                    self.error(
                        "DECISION_CONFIG_MISMATCH",
                        f"decision {rid!r}: run {run_id!r} belongs to config {run.payload.config_ref!r}, decision to {p.config_ref!r}",
                        rid,
                        field,
                        run=run_id,
                    )
                if list_name == "candidate_run_refs" and run.payload.subject_ref != p.candidate_subject_ref:
                    self.error(
                        "CANDIDATE_SUBJECT_MISMATCH",
                        f"decision {rid!r}: candidate run {run_id!r} tested {run.payload.subject_ref!r}, not candidate_subject_ref {p.candidate_subject_ref!r}",
                        rid,
                        field,
                        run=run_id,
                        run_subject=run.payload.subject_ref,
                    )
        untrusted = sorted(r.record_id for r in runs if r.payload.provenance != TRUSTED_PROVENANCE)
        fixture_runs = sorted(r.record_id for r in runs if r.payload.provenance == "fixture")
        if p.is_production:
            if p.outcome != "accepted":
                self.error("PRODUCTION_WITHOUT_ACCEPTANCE", f"decision {rid!r}: is_production requires outcome accepted (got {p.outcome!r})", rid, "is_production", outcome=p.outcome)
            if untrusted:
                self.error(
                    "FIXTURE_PRODUCTION_DECISION",
                    f"decision {rid!r}: is_production but runs {untrusted} are not trusted_worker evidence",
                    rid,
                    "is_production",
                    untrusted_runs=untrusted,
                )
            if not runs:
                self.error("PRODUCTION_WITHOUT_EVIDENCE", f"decision {rid!r}: is_production without any resolvable run evidence", rid, "is_production")
        if p.outcome == "accepted":
            if p.evaluated_by not in ("program", "human_override"):
                self.error("INVALID_EVALUATOR", f"decision {rid!r}: accepted requires evaluated_by program or human_override", rid, "evaluated_by")
            if not p.candidate_run_refs:
                self.error("ACCEPTED_WITHOUT_EVIDENCE", f"decision {rid!r}: accepted without candidate runs", rid, "candidate_run_refs")
            if fixture_runs and not p.policy.allow_fixture:
                self.error(
                    "FIXTURE_DECISION_ACCEPTED",
                    f"decision {rid!r}: accepted on fixture runs {fixture_runs} although the policy does not allow fixtures",
                    rid,
                    "outcome",
                    fixture_runs=fixture_runs,
                )
            if p.policy.require_trusted_worker and untrusted and p.evaluated_by == "program":
                self.error(
                    "UNVERIFIED_ACCEPTED",
                    f"decision {rid!r}: program accepted runs {untrusted} although the policy requires trusted_worker provenance",
                    rid,
                    "outcome",
                    untrusted_runs=untrusted,
                )


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def validate_records(
    records: Iterable[Record],
    *,
    artifact_reader: ArtifactReader,
    artifact_registry: ArtifactRegistry | None = None,
    registry: ProblemRegistry | None = None,
    external_resolver: Resolver | None = None,
    kernel_resolver: Resolver | None = None,
    verify_artifacts: bool = True,
    missing_evidence_severity: str = ERROR,
) -> ValidationReport:
    """Deep-validate an in-memory record set. See the module docstring for the checks and codes."""
    items = list(records)
    for item in items:
        if not isinstance(item, Record):
            raise InputError(f"validate_records expects Record instances, got {type(item).__name__}", code="INVALID_ARGUMENT")
    if not callable(artifact_reader):
        raise InputError("artifact_reader must be callable", code="INVALID_ARGUMENT")
    if missing_evidence_severity not in SEVERITIES:
        raise InputError(f"missing_evidence_severity must be one of {SEVERITIES}", code="INVALID_ARGUMENT")
    validator = _Validator(
        items,
        artifact_reader=artifact_reader,
        artifact_registry=artifact_registry,
        registry=registry if registry is not None else default_registry(),
        external_resolver=external_resolver,
        kernel_resolver=kernel_resolver,
        verify_artifacts=verify_artifacts,
        missing_evidence_severity=missing_evidence_severity,
    )
    return validator.run()


def store_artifact_reader(store: MemoryStore) -> ArtifactReader:
    """Reader over the store's CAS that returns raw bytes (digest verification is the validator's job)."""

    def reader(sha256_ref: str) -> bytes | None:
        try:
            path = store.artifact_path(sha256_ref)
        except InputError:
            return None
        return path.read_bytes() if path.is_file() else None

    return reader


def merge_integrity_report(report: ValidationReport, integrity: IntegrityReport) -> None:
    """Append store-level integrity findings to ``report`` as errors."""
    for item in integrity.modified:
        report.add(ERROR, "STORE_RECORD_MODIFIED", f"record file {item.get('path')} was modified after publication", item.get("record_id"), None, **item)
    for item in integrity.missing:
        report.add(ERROR, "STORE_RECORD_MISSING", f"journaled record {item.get('record_id')!r} has no file at {item.get('expected_path')}", item.get("record_id"), None, **item)
    for item in integrity.corrupt:
        report.add(ERROR, "STORE_RECORD_CORRUPT", f"record file {item.get('path')} is corrupt: {item.get('error')}", None, None, **item)
    for item in integrity.unjournaled:
        report.add(ERROR, "STORE_RECORD_UNJOURNALED", f"record file {item.get('path')} is not in the publication journal", item.get("record_id"), None, **item)
    for item in integrity.duplicate_ids:
        report.add(ERROR, "STORE_DUPLICATE_ID", f"record id {item.get('record_id')!r} exists in several files", item.get("record_id"), None, **item)
    for item in integrity.artifact_problems:
        problem = str(item.get("problem", "problem")).upper()
        report.add(ERROR, f"STORE_ARTIFACT_{problem}", f"registered artifact {item.get('artifact_id')!r}: {item.get('problem')}", None, None, **item)


def deep_validate(store: MemoryStore, *, verify_artifacts: bool = True, registry: ProblemRegistry | None = None) -> ValidationReport:
    """Validate every record of ``store`` plus the store's own integrity scan."""
    integrity = store.integrity_scan(verify_artifacts=verify_artifacts)
    try:
        records = store.records()
    except InvariantViolation as exc:
        report = ValidationReport()
        report.add(ERROR, exc.code, f"store records cannot be loaded: {exc.message}", None, None, **exc.details)
        merge_integrity_report(report, integrity)
        return report
    report = validate_records(
        records,
        artifact_reader=store_artifact_reader(store),
        artifact_registry=store.get_artifact_ref,
        registry=registry,
        external_resolver=None,
        kernel_resolver=store.kernel_by_kernel_id,
        verify_artifacts=verify_artifacts,
    )
    merge_integrity_report(report, integrity)
    return report


__all__ = [
    "ERROR",
    "WARNING",
    "Issue",
    "ValidationReport",
    "validate_records",
    "deep_validate",
    "store_artifact_reader",
    "merge_integrity_report",
    "find_cycle",
]
