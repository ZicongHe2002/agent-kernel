"""Trusted local Runner: turns a ledgered request into exactly one immutable Run record.

Specification sections 9, 15, 17. The Runner is the controlled local executor: it
captures what was actually tested (from the adapter's ``PreparedExecution``), stores
every evidence blob content-addressed, derives every timing number with
``domain.stats`` from the stored samples, and publishes the Run under the store lock
only after re-verifying its lease (fencing, T26). Because this process is the
controlled executor, Runs it publishes carry ``provenance="trusted_worker"``;
imported bundles can never claim that grade (they are downgraded by the importer).

Public API
----------
``LocalRunner(store, *, adapters, problem_registry=None, analysis_adapters=None,
              worker_id="local-runner", permissions=None, clock=time.monotonic,
              lease_seconds=600)``
    ``ledger``  the ``RequestLedger`` used for claims/finishes (shares ``store``)
    ``execute(request_id) -> Record``
        Order of checks (nothing is claimed until all pass):
          1. subject/config exist; ``subject.commit_oid == spec.source.target_commit``
             (``SUBJECT_SOURCE_MISMATCH``, exit 2)
          2. authorization: ``cpu``/``mock`` need ``permissions["allow_local_cpu_tests"]``
             (default True); ``jax_tpu`` needs ``spec.authorization["allow_tpu_execution"]``
             *and* ``permissions["allow_tpu_execution"]``; other backends need
             ``permissions["allow_<backend>_execution"]``. Denials append a ``cancelled``
             ledger event and raise ``AuthorizationError`` (exit 7).
          3. adapter lookup + ``check_environment()``: ``BackendUnavailable`` appends a
             ``backend_unavailable`` event and propagates - a missing backend is not
             an execution and produces no Run.
        Then: claim -> prepare -> source policy -> compile -> verify -> benchmark ->
        build Run -> (under lock) verify lease, store artifacts, publish, finish.
        Terminal facts recorded as Runs: ``succeeded`` (with any correctness status),
        ``compile_error`` (T14), ``runtime_error``, ``timeout``, ``infrastructure_error``,
        ``source_unavailable`` (tested source differs from target, or dirty source
        without exploratory permission). Non-succeeded Runs have correctness/timing
        ``not_run`` and carry only diagnostic (compile) artifacts - no invented numbers.
        Security refusals (``SecurityPolicyError``/``AuthorizationError``) and an
        ``IncompleteProblemContract`` raised by the adapter append ``cancelled`` with the
        reason and re-raise; a ``BackendUnavailable`` after the claim appends
        ``backend_unavailable`` and releases the attempt (``lease_lost``) - no Run either way.
        A lost lease appends ``late_result_quarantined`` (with the would-be record as
        evidence) and raises ``LeaseLostError``; nothing is published.
    ``submit_and_execute(spec) -> (SubmitOutcome, Record | None)``
        Replaying a finished request returns its existing Run without executing (T11).
    ``describe_unavailable(backend, error) -> dict``  CLI-friendly description of an
        unexecuted backend.

Wall clock: ``clock`` measures elapsed execution time and stamps leases; if the
elapsed time exceeds ``spec.max_wall_seconds`` the Run is recorded as ``timeout``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..adapters.base import (
    AdapterRegistry,
    ArtifactBlob,
    CorrectnessReport,
    PreparedExecution,
    RunRequestSpec,
    SourceSnapshot,
    TimingReport,
)
from ..domain import hashing, stats
from ..domain.errors import (
    AuthorizationError,
    BackendUnavailable,
    ExecutionInfrastructureError,
    IncompleteProblemContract,
    InputError,
    InvariantViolation,
    KernelMemoryError,
    LeaseLostError,
    PrerequisiteMissingError,
    SecurityPolicyError,
)
from ..domain.ids import validate_record_id
from ..domain.jsonio import dumps_readable
from ..domain.models import ArtifactRef, Correctness, GitOid, Record, Timing, to_json
from ..domain.problems import ProblemRegistry
from ..services.common import new_record
from ..storage.store import MemoryStore
from .ledger import Lease, RequestLedger, SubmitOutcome, run_record_id

PROVENANCE = "trusted_worker"
STAGES: tuple[str, ...] = ("compile", "verify", "benchmark", "profile")
LOCAL_BACKENDS: frozenset[str] = frozenset({"cpu", "mock"})
KNOWN_METRICS: tuple[str, ...] = ("register_spill_vmem_static_bytes",)

NOT_RUN_CORRECTNESS = Correctness(status="not_run", cases_total=0, cases_passed=0, max_abs_error=None, max_rel_error=None, report_artifact_ref=None)
NOT_RUN_TIMING = Timing(status="not_run", sample_count=0, median_us=None, p90_us=None, samples_artifact_ref=None)


def artifact_id_for(run_id: str, blob_id: str) -> str:
    return f"{run_id}-{blob_id}"


def not_collected_metric(name: str, note: str, *, status: str = "not_collected") -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "value": None,
        "unit": "bytes",
        "kind": "static_estimate",
        "scope": "compiled_kernel",
        "source_artifact_ref": None,
        "parser_id": None,
        "parser_version": None,
        "definition": f"Static register-spill estimate from compiler low-level output. {note}",
    }


@dataclass
class _StageOutcome:
    execution_status: str
    failure_reason: str | None
    correctness: Correctness
    timing: Timing
    result_blobs: list[ArtifactBlob] = field(default_factory=list)  # evidence of results (dropped on failure)
    diagnostic_blobs: list[ArtifactBlob] = field(default_factory=list)  # compile logs etc. (kept on failure)

    @property
    def blobs(self) -> list[ArtifactBlob]:
        return list(self.diagnostic_blobs) + list(self.result_blobs)


def _failure(status: str, reason: str, diagnostics: list[ArtifactBlob]) -> _StageOutcome:
    return _StageOutcome(status, reason, NOT_RUN_CORRECTNESS, NOT_RUN_TIMING, [], list(diagnostics))


class LocalRunner:
    def __init__(
        self,
        store: MemoryStore,
        *,
        adapters: AdapterRegistry,
        problem_registry: ProblemRegistry | None = None,
        analysis_adapters: list[Any] | None = None,
        worker_id: str = "local-runner",
        permissions: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        lease_seconds: float = 600.0,
    ) -> None:
        if not isinstance(worker_id, str) or not worker_id:
            raise InputError("worker_id must be a non-empty string", code="INVALID_WORKER_ID")
        self._store = store
        self._adapters = adapters
        self._problems = problem_registry
        self._analysis_adapters = analysis_adapters
        self._worker_id = worker_id
        self._permissions: dict[str, Any] = dict(permissions or {})
        self._clock = clock
        self._lease_seconds = float(lease_seconds)
        self.ledger = RequestLedger(store, clock=clock)

    # ------------------------------------------------------------------ public
    @property
    def worker_id(self) -> str:
        return self._worker_id

    def submit_and_execute(self, spec: RunRequestSpec) -> tuple[SubmitOutcome, Record | None]:
        outcome = self.ledger.submit(spec)
        state = self.ledger.state(outcome.request_id)
        if state.kind == "finished":
            existing = self._store.get(state.run_refs[-1]) if state.run_refs else None
            return outcome, existing
        return outcome, self.execute(outcome.request_id)

    @staticmethod
    def describe_unavailable(backend: str, error: Exception) -> dict[str, Any]:
        if isinstance(error, KernelMemoryError):
            info = error.to_dict()
        else:
            info = {"error": type(error).__name__, "message": str(error), "exit_code": 1, "details": {}}
        return {
            "backend": backend,
            "status": "unexecuted",
            "error": info["error"],
            "message": info["message"],
            "exit_code": info["exit_code"],
            "details": info["details"],
            "note": "The backend is unavailable; nothing was executed and no Run was recorded.",
        }

    def execute(self, request_id: str) -> Record:
        spec = self.ledger.get(request_id)
        if spec.stage not in STAGES:
            raise InputError(f"unknown stage {spec.stage!r}", code="INVALID_STAGE", details={"stage": spec.stage, "known": list(STAGES)})
        subject = self._store.require(spec.subject_ref, "commit", "baseline")
        config = self._store.require(spec.config_ref, "config")
        self._check_subject(spec, subject)
        try:
            self._authorize(spec)
        except AuthorizationError as exc:
            self.ledger.cancel(request_id, exc.message, payload={"error": exc.to_dict()})
            raise
        try:
            adapter = self._adapters.kernel_adapter(spec.backend)
            probe_environment = dict(adapter.check_environment())
        except BackendUnavailable as exc:
            self.ledger.record_backend_unavailable(request_id, exc)
            raise

        lease = self.ledger.claim(request_id, self._worker_id, lease_seconds=self._lease_seconds, now=self._clock)
        run_id = run_record_id(request_id, lease.attempt_no)
        started = float(self._clock())

        # ---- prepare: capture the actual source; refusals here are not executions
        snapshot: SourceSnapshot
        environment: dict[str, Any]
        input_suite_hash: str
        outcome: _StageOutcome | None = None
        try:
            prepared = adapter.prepare(spec, config.payload.problem)
        except (SecurityPolicyError, IncompleteProblemContract) as exc:
            # Refusals are recorded, never silently skipped; they are not executions and produce no Run.
            self.ledger.cancel(request_id, exc.message, payload={"error": exc.to_dict()}, lease=lease)
            raise
        except PrerequisiteMissingError as exc:
            self.ledger.record_backend_unavailable(request_id, exc, lease=lease)
            raise
        except InvariantViolation as exc:
            if exc.code != "TESTED_SOURCE_MISMATCH":
                raise
            recovered = self._snapshot_from_error(spec, exc)
            if recovered is None:
                # The adapter could not tell us what it actually tested: terminal without a Run.
                self.ledger.finish(
                    lease,
                    run_ref=None,
                    execution_status="source_unavailable",
                    payload={"reason": exc.message, "error": exc.to_dict()},
                )
                raise
            snapshot = recovered
            environment = probe_environment
            input_suite_hash = str(exc.details.get("input_suite_hash") or hashing.jcs_digest({"input_suite": "unavailable", "request_id": request_id}))
            outcome = _failure("source_unavailable", exc.message, [])
        else:
            snapshot = prepared.source
            environment = dict(prepared.environment) if prepared.environment else probe_environment
            input_suite_hash = prepared.input_suite_hash
            refusal = self._source_refusal(spec, snapshot)
            if refusal is not None:
                outcome = _failure("source_unavailable", refusal, [])
            else:
                self.ledger.running(lease)
                try:
                    outcome = self._run_stages(adapter, prepared, spec, run_id)
                except (SecurityPolicyError, IncompleteProblemContract) as exc:
                    self.ledger.cancel(request_id, exc.message, payload={"error": exc.to_dict()}, lease=lease)
                    raise
                except PrerequisiteMissingError as exc:
                    self.ledger.record_backend_unavailable(request_id, exc, lease=lease)
                    raise

        elapsed = float(self._clock()) - started
        if elapsed > float(spec.max_wall_seconds):
            outcome = _failure(
                "timeout",
                f"wall clock {elapsed:.3f}s exceeded max_wall_seconds {spec.max_wall_seconds}",
                outcome.diagnostic_blobs,
            )

        refs, data_by_id = self._artifact_refs(run_id, outcome.blobs)
        metrics, conclusion, analysis_blobs = self._analyze(outcome.execution_status, refs, data_by_id)
        if analysis_blobs:
            extra_refs, extra_data = self._artifact_refs(run_id, analysis_blobs, existing={r.artifact_id: r for r in refs})
            refs.extend(extra_refs)
            data_by_id.update(extra_data)
            metrics = self._check_metric_sources(metrics, {r.artifact_id for r in refs})
        record = self._build_run(
            spec=spec,
            config=config,
            run_id=run_id,
            attempt_no=lease.attempt_no,
            snapshot=snapshot,
            environment=environment,
            input_suite_hash=input_suite_hash,
            outcome=outcome,
            refs=refs,
            metrics=metrics,
            conclusion=conclusion,
        )

        with self._store.lock():
            try:
                self.ledger.require_lease(lease)
            except LeaseLostError as exc:
                self.ledger.quarantine_late_result(
                    lease,
                    {
                        "quarantined_run_ref": run_id,
                        "execution_status": outcome.execution_status,
                        "record_digest": record.canonical_digest(),
                        "record": record.to_dict(),
                    },
                    cause=exc,
                )
            for data in data_by_id.values():
                self._store.put_artifact_bytes(data)
            self._store.publish(record, label=f"run {run_id}")
            self.ledger.finish(lease, run_ref=run_id, execution_status=outcome.execution_status)
        return record

    # ------------------------------------------------------------------ pre-claim checks
    @staticmethod
    def _check_subject(spec: RunRequestSpec, subject: Record) -> None:
        subject_oid: GitOid = subject.payload.commit_oid
        target = spec.source.target_commit
        if subject_oid.key() != target.key():
            raise InputError(
                f"subject {subject.record_id!r} is commit {subject_oid.algorithm}:{subject_oid.hex} but the request targets "
                f"{target.algorithm}:{target.hex}",
                code="SUBJECT_SOURCE_MISMATCH",
                details={
                    "subject_ref": subject.record_id,
                    "subject_commit": to_json(subject_oid),
                    "target_commit": to_json(target),
                },
            )

    def _authorize(self, spec: RunRequestSpec) -> None:
        backend = spec.backend
        if not isinstance(backend, str) or not backend:
            raise InputError("request backend must be a non-empty string", code="INVALID_BACKEND")
        if backend in LOCAL_BACKENDS:
            if not self._permissions.get("allow_local_cpu_tests", True):
                raise AuthorizationError(
                    f"local {backend} execution is not authorized (allow_local_cpu_tests is false)",
                    code="LOCAL_CPU_TESTS_NOT_AUTHORIZED",
                    details={"backend": backend, "required": ["permissions.allow_local_cpu_tests"]},
                )
            return
        if backend == "jax_tpu":
            requested = bool(spec.authorization.get("allow_tpu_execution", False))
            granted = bool(self._permissions.get("allow_tpu_execution", False))
            if not (requested and granted):
                raise AuthorizationError(
                    "TPU execution requires allow_tpu_execution on both the request and the runner permissions",
                    code="TPU_EXECUTION_NOT_AUTHORIZED",
                    details={
                        "backend": backend,
                        "request_authorization": requested,
                        "runner_permission": granted,
                        "required": ["request.authorization.allow_tpu_execution", "permissions.allow_tpu_execution"],
                    },
                )
            return
        flag = f"allow_{backend}_execution"
        if not self._permissions.get(flag, False):
            raise AuthorizationError(
                f"execution on backend {backend!r} is not authorized ({flag} is false)",
                code="BACKEND_EXECUTION_NOT_AUTHORIZED",
                details={"backend": backend, "required": [f"permissions.{flag}"]},
            )

    # ------------------------------------------------------------------ source policy
    @staticmethod
    def _source_refusal(spec: RunRequestSpec, snapshot: SourceSnapshot) -> str | None:
        if snapshot.target_commit.key() != spec.source.target_commit.key():
            return (
                f"adapter snapshot targets {snapshot.target_commit.algorithm}:{snapshot.target_commit.hex} but the request "
                f"targets {spec.source.target_commit.algorithm}:{spec.source.target_commit.hex}"
            )
        if snapshot.checkout_mode == "exact_commit":
            if snapshot.tested_commit.key() != snapshot.target_commit.key():
                return (
                    f"tested commit {snapshot.tested_commit.algorithm}:{snapshot.tested_commit.hex} differs from target "
                    f"{snapshot.target_commit.algorithm}:{snapshot.target_commit.hex} under exact_commit checkout"
                )
            if snapshot.merge_parent_oids:
                return "exact_commit checkout reported merge parents; the tested source is not the target commit"
        if snapshot.dirty:
            if not spec.source.allow_dirty_exploratory:
                return "dirty working tree refused: the request does not allow exploratory dirty execution"
            if not snapshot.patch_digest:
                return "dirty working tree without a patch_digest cannot be recorded honestly"
        return None

    @staticmethod
    def _snapshot_from_error(spec: RunRequestSpec, exc: InvariantViolation) -> SourceSnapshot | None:
        details = exc.details or {}
        tested = details.get("tested_commit")
        digest = details.get("source_digest")
        if not isinstance(tested, dict) or not isinstance(digest, str) or not hashing.is_sha256_ref(digest):
            return None
        try:
            tested_oid = GitOid(algorithm=str(tested["algorithm"]), hex=str(tested["hex"]))
        except KeyError:
            return None
        tree = details.get("tested_tree")
        tree_oid = GitOid(algorithm=str(tree["algorithm"]), hex=str(tree["hex"])) if isinstance(tree, dict) and "algorithm" in tree and "hex" in tree else None
        parents: list[GitOid] = []
        for parent in details.get("merge_parent_oids", []) or []:
            if isinstance(parent, dict) and "algorithm" in parent and "hex" in parent:
                parents.append(GitOid(algorithm=str(parent["algorithm"]), hex=str(parent["hex"])))
        patch = details.get("patch_digest")
        return SourceSnapshot(
            repo_uid=spec.source.repo_uid,
            target_commit=spec.source.target_commit,
            tested_commit=tested_oid,
            tested_tree=tree_oid,
            checkout_mode=spec.source.checkout_mode,
            merge_parent_oids=parents,
            dirty=bool(details.get("dirty", False)),
            patch_digest=patch if isinstance(patch, str) else None,
            source_digest=digest,
            entrypoint=spec.source.entrypoint,
            implementation_overrides=dict(spec.source.implementation_overrides),
        )

    # ------------------------------------------------------------------ stages
    @staticmethod
    def _map_exception(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, ExecutionInfrastructureError):
            if exc.code == "TIMEOUT":
                return "timeout", exc.message
            return "infrastructure_error", f"{exc.code}: {exc.message}"
        if isinstance(exc, KernelMemoryError):
            return "runtime_error", f"{exc.code}: {exc.message}"
        return "runtime_error", f"{type(exc).__name__}: {exc}"

    def _run_stages(self, adapter: Any, prepared: PreparedExecution, spec: RunRequestSpec, run_id: str) -> _StageOutcome:
        diagnostics: list[ArtifactBlob] = []
        try:
            compile_report = adapter.compile(prepared)
        except (SecurityPolicyError, PrerequisiteMissingError):
            raise
        except Exception as exc:  # adapter failure is a recorded fact, not a crash of the runner
            status, reason = self._map_exception(exc)
            return _failure(status, reason, diagnostics)
        diagnostics.extend(compile_report.artifacts)
        if compile_report.status == "compile_error":
            return _failure("compile_error", compile_report.message or "compilation failed", diagnostics)
        if compile_report.status == "unsupported":
            return _failure("infrastructure_error", f"compile unsupported: {compile_report.message or 'no details'}", diagnostics)
        if compile_report.status != "ok":
            return _failure("infrastructure_error", f"adapter returned unknown compile status {compile_report.status!r}", diagnostics)
        if spec.stage == "compile":
            return _StageOutcome("succeeded", None, NOT_RUN_CORRECTNESS, NOT_RUN_TIMING, [], diagnostics)

        try:
            verify_report = adapter.verify(prepared)
        except (SecurityPolicyError, PrerequisiteMissingError):
            raise
        except Exception as exc:
            status, reason = self._map_exception(exc)
            return _failure(status, reason, diagnostics)
        correctness, correctness_blobs = self._correctness_from_report(verify_report, run_id)
        if spec.stage == "verify":
            return _StageOutcome("succeeded", None, correctness, NOT_RUN_TIMING, correctness_blobs, diagnostics)

        try:
            timing_report = adapter.benchmark(prepared)
        except (SecurityPolicyError, PrerequisiteMissingError):
            raise
        except Exception as exc:
            status, reason = self._map_exception(exc)
            return _failure(status, reason, diagnostics)
        timing, timing_blobs, timing_reason = self._timing_from_report(timing_report, run_id)
        return _StageOutcome("succeeded", timing_reason, correctness, timing, correctness_blobs + timing_blobs, diagnostics)

    @staticmethod
    def _correctness_from_report(report: CorrectnessReport, run_id: str) -> tuple[Correctness, list[ArtifactBlob]]:
        status = report.status if report.status in ("pass", "fail", "error", "not_run") else "error"
        blobs = list(report.artifacts)
        report_ref: str | None = None
        for blob in blobs:
            if blob.kind == "correctness_report":
                report_ref = artifact_id_for(run_id, blob.artifact_id)
                break
        if report_ref is None and status != "not_run":
            payload = {
                "status": status,
                "cases_total": report.cases_total,
                "cases_passed": report.cases_passed,
                "max_abs_error": report.max_abs_error,
                "max_rel_error": report.max_rel_error,
                "message": report.message,
            }
            blobs.append(ArtifactBlob("correctness", "correctness_report", "application/json", dumps_readable(payload).encode("utf-8")))
            report_ref = artifact_id_for(run_id, "correctness")
        correctness = Correctness(
            status=status,
            cases_total=int(report.cases_total),
            cases_passed=int(report.cases_passed),
            max_abs_error=report.max_abs_error,
            max_rel_error=report.max_rel_error,
            report_artifact_ref=report_ref,
        )
        return correctness, blobs

    @staticmethod
    def _timing_from_report(report: TimingReport, run_id: str) -> tuple[Timing, list[ArtifactBlob], str | None]:
        blobs = list(report.artifacts)
        if report.status == "recorded":
            samples = list(report.samples)
            try:
                summary = stats.summarize(samples, report.unit)
            except InputError as exc:
                return Timing("error", 0, None, None, None), blobs, f"invalid timing samples: {exc.message}"
            samples_doc = {"samples": samples, "unit": report.unit}
            blobs.append(ArtifactBlob("samples", "latency_samples", "application/json", dumps_readable(samples_doc).encode("utf-8")))
            timing = Timing(
                status="recorded",
                sample_count=summary.sample_count,
                median_us=summary.median_us,
                p90_us=summary.p90_us,
                samples_artifact_ref=artifact_id_for(run_id, "samples"),
            )
            return timing, blobs, None
        if report.status == "error":
            return Timing("error", 0, None, None, None), blobs, report.message or "timing failed"
        return NOT_RUN_TIMING, blobs, report.message

    # ------------------------------------------------------------------ artifacts / analysis
    @staticmethod
    def _artifact_refs(
        run_id: str,
        blobs: list[ArtifactBlob],
        *,
        existing: dict[str, ArtifactRef] | None = None,
    ) -> tuple[list[ArtifactRef], dict[str, bytes]]:
        refs: dict[str, ArtifactRef] = {}
        data: dict[str, bytes] = {}
        known = dict(existing or {})
        for blob in blobs:
            if not isinstance(blob.data, (bytes, bytearray)):
                raise InvariantViolation(f"artifact {blob.artifact_id!r} does not carry bytes", code="ARTIFACT_NOT_BYTES")
            aid = artifact_id_for(run_id, blob.artifact_id)
            validate_record_id(aid, what="artifact_id")
            digest = hashing.artifact_digest(bytes(blob.data))
            ref = ArtifactRef(
                artifact_id=aid,
                kind=blob.kind,
                uri=f"artifact://sha256/{digest[len(hashing.SHA256_PREFIX):]}",
                sha256=digest,
                size_bytes=len(blob.data),
                media_type=blob.media_type,
                retention=blob.retention,
                availability="present",
            )
            previous = refs.get(aid) or known.get(aid)
            if previous is not None:
                if to_json(previous) != to_json(ref):
                    raise InvariantViolation(
                        f"two artifacts of run {run_id!r} share id {aid!r} with different content",
                        code="ARTIFACT_ID_COLLISION",
                        details={"artifact_id": aid},
                    )
                continue
            refs[aid] = ref
            data[aid] = bytes(blob.data)
        return list(refs.values()), data

    def _analyze(
        self,
        execution_status: str,
        refs: list[ArtifactRef],
        data_by_id: dict[str, bytes],
    ) -> tuple[list[dict[str, Any]], str | None, list[ArtifactBlob]]:
        adapters = self._analysis_adapters
        if adapters is None:
            adapters = self._adapters.analysis_adapters() if hasattr(self._adapters, "analysis_adapters") else []
        if execution_status != "succeeded":
            return [not_collected_metric(n, "Not collected: the execution did not succeed.") for n in KNOWN_METRICS], None, []
        if not adapters:
            return [not_collected_metric(n, "Not collected: no analysis adapter was configured for this run.") for n in KNOWN_METRICS], None, []
        try:
            from ..adapters import analysis as analysis_module  # lazily: another module owner may not have shipped it yet
        except ImportError:
            return [not_collected_metric(n, "Not collected: the analysis adapter module is unavailable.") for n in KNOWN_METRICS], None, []
        analyze = getattr(analysis_module, "analyze_artifacts", None)
        if analyze is None:
            return [not_collected_metric(n, "Not collected: analysis module lacks analyze_artifacts.") for n in KNOWN_METRICS], None, []
        pairs = [(ref, data_by_id[ref.artifact_id]) for ref in refs if ref.artifact_id in data_by_id]
        try:
            result = analyze(adapters, pairs)
        except Exception as exc:  # analysis must never invent a value; report the failure as a status
            note = f"Not collected: analysis failed with {type(exc).__name__}."
            return [not_collected_metric(n, note, status="parse_error") for n in KNOWN_METRICS], None, []
        metrics_raw: Any
        conclusion: str | None = None
        extra: list[ArtifactBlob] = []
        if isinstance(result, dict):
            metrics_raw = result.get("metrics", [])
            conclusion = result.get("conclusion")
            extra = list(result.get("artifacts", []) or [])
        elif isinstance(result, list):
            metrics_raw = result
        else:
            metrics_raw = getattr(result, "metrics", [])
            conclusion = getattr(result, "conclusion", None)
            extra = list(getattr(result, "artifacts", []) or [])
        metrics: list[dict[str, Any]] = []
        for metric in metrics_raw or []:
            metrics.append(dict(metric) if isinstance(metric, dict) else to_json(metric))
        if not metrics:
            metrics = [not_collected_metric(n, "Not collected: the analysis adapters produced no metrics.") for n in KNOWN_METRICS]
        return metrics, conclusion if isinstance(conclusion, str) else None, extra

    @staticmethod
    def _check_metric_sources(metrics: list[dict[str, Any]], artifact_ids: set[str]) -> list[dict[str, Any]]:
        checked: list[dict[str, Any]] = []
        for metric in metrics:
            if metric.get("status") == "observed" and metric.get("source_artifact_ref") not in artifact_ids:
                downgraded = dict(metric)
                downgraded["status"] = "parse_error"
                downgraded["value"] = None
                downgraded["definition"] = f"{metric.get('definition', '')} (observed value discarded: source artifact is not part of this run)".strip()
                checked.append(downgraded)
            else:
                checked.append(metric)
        return checked

    # ------------------------------------------------------------------ record construction
    def _build_run(
        self,
        *,
        spec: RunRequestSpec,
        config: Record,
        run_id: str,
        attempt_no: int,
        snapshot: SourceSnapshot,
        environment: dict[str, Any],
        input_suite_hash: str,
        outcome: _StageOutcome,
        refs: list[ArtifactRef],
        metrics: list[dict[str, Any]],
        conclusion: str | None,
    ) -> Record:
        env = {k: v for k, v in environment.items() if k != "environment_hash"}
        env["environment_hash"] = hashing.environment_hash(env)
        protocol = {k: v for k, v in dict(spec.protocol).items() if k != "protocol_hash"}
        protocol["input_suite_hash"] = input_suite_hash
        protocol["protocol_hash"] = hashing.protocol_hash(protocol)
        verifier = {k: v for k, v in dict(spec.verifier).items() if k != "verifier_hash"}
        verifier["verifier_hash"] = hashing.verifier_hash(verifier)
        overrides = dict(snapshot.implementation_overrides)
        source = {
            "repo_uid": snapshot.repo_uid,
            "target_commit": to_json(snapshot.target_commit),
            "tested_commit": to_json(snapshot.tested_commit),
            "tested_tree": to_json(snapshot.tested_tree) if snapshot.tested_tree is not None else None,
            "checkout_mode": snapshot.checkout_mode,
            "merge_parent_oids": [to_json(p) for p in snapshot.merge_parent_oids],
            "dirty": bool(snapshot.dirty),
            "patch_digest": snapshot.patch_digest,
            "source_digest": snapshot.source_digest,
            "entrypoint": snapshot.entrypoint,
            "implementation_overrides": overrides,
            "variant_digest": hashing.variant_digest(
                source_digest=snapshot.source_digest,
                entrypoint=snapshot.entrypoint,
                implementation_overrides=overrides,
                checkout_mode=snapshot.checkout_mode,
            ),
        }
        comparison_key = hashing.comparison_key(
            config_hash=config.payload.config_hash,
            environment_hash=env["environment_hash"],
            protocol_hash=protocol["protocol_hash"],
            verifier_hash=verifier["verifier_hash"],
            checkout_mode=snapshot.checkout_mode,
        )
        payload = {
            "subject_ref": spec.subject_ref,
            "config_ref": spec.config_ref,
            "request_id": spec.request_id,
            "attempt_no": attempt_no,
            "rerun_of": spec.rerun_of,
            "stage": spec.stage,
            "provenance": PROVENANCE,
            "source": source,
            "environment": env,
            "protocol": protocol,
            "verifier": verifier,
            "execution_status": outcome.execution_status,
            "failure_reason": outcome.failure_reason,
            "correctness": to_json(outcome.correctness),
            "timing": to_json(outcome.timing),
            "analysis_metrics": metrics,
            "analysis_conclusion": conclusion,
            "artifacts": [to_json(r) for r in refs],
            "comparison_key": comparison_key,
            "session_id": spec.session_id,
            "pair_id": spec.pair_id,
            "role_in_pair": spec.role_in_pair,
        }
        return new_record("run", run_id, payload)


__all__ = ["LocalRunner", "PROVENANCE", "STAGES", "KNOWN_METRICS", "artifact_id_for", "not_collected_metric"]
