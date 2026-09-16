"""Deterministic mock kernel adapter (specification section 17).

Public API
----------
``MockAdapter(backend="mock", adapter_id="mock-v1")`` implements ``adapters.base.KernelAdapter``
without touching any hardware. Its behaviour is fully determined by
``request.metadata.get("mock", {})``:

======================  ==========================================================================
key                     meaning
======================  ==========================================================================
``compile``             ``"ok"`` (default) or ``"compile_error"``
``correctness``         ``"pass"`` (default), ``"fail"`` or ``"error"``
``samples``             explicit list of positive integer nanosecond samples; by default the
                        pattern ``DEFAULT_SAMPLE_PATTERN`` is cycled to ``protocol["repetitions"]``
``raise``               ``None`` (default), ``"runtime_error"``, ``"timeout"`` or
                        ``"infrastructure_error"``: ``benchmark`` raises the corresponding
                        ``ExecutionInfrastructureError`` (codes ``RUNTIME_ERROR``, ``TIMEOUT``,
                        ``EXECUTION_INFRASTRUCTURE_ERROR``)
``environment_overrides``  dict merged into the environment snapshot (``backend`` may not change)
======================  ==========================================================================

Unknown keys or invalid values raise ``InputError`` (exit 2) so a mis-typed scenario never
silently becomes a "pass".

Guarantees
----------
* Every artifact is JSON and carries ``"is_mock": true``; every report message says "mock".
* The environment always reports ``backend == "mock"`` and ``accelerator_model ==
  "MOCK-NO-HARDWARE"``; an override attempting to relabel the backend is refused.
* ``TimingReport.timing_method`` is ``"mock"`` (a protocol enum value) — never
  ``host_synchronized`` — so mock samples can never be mistaken for measurements.
* Source snapshot: ``tested_commit == target_commit``, ``source_digest =
  jcs_digest({"mock_source": entrypoint})``; ``input_suite_hash = jcs_digest({"mock_suite":
  problem})``. Nothing is executed.
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import ExecutionInfrastructureError, InputError
from ..domain.hashing import jcs_digest
from ..domain.jsonio import dumps_readable
from .base import (
    ArtifactBlob,
    CompileReport,
    CorrectnessReport,
    PreparedExecution,
    RunRequestSpec,
    SourceSnapshot,
    TimingReport,
)

DEFAULT_SAMPLE_PATTERN: tuple[int, ...] = (1000, 1010, 1000, 990, 1000)
KNOWN_KEYS = frozenset({"compile", "correctness", "samples", "raise", "environment_overrides"})
COMPILE_MODES = ("ok", "compile_error")
CORRECTNESS_MODES = ("pass", "fail", "error")
RAISE_MODES = (None, "runtime_error", "timeout", "infrastructure_error")
MOCK_ENVIRONMENT: dict[str, Any] = {
    "backend": "mock",
    "accelerator_model": "MOCK-NO-HARDWARE",
    "device_count": 1,
    "topology": "single",
    "software": {"runner": "mock-v1"},
    "execution_flags": {},
    "host_timer_environment": {},
    "unknown_required_fields": [],
}


def _json_blob(artifact_id: str, kind: str, document: dict[str, Any]) -> ArtifactBlob:
    payload = dict(document)
    payload["is_mock"] = True
    return ArtifactBlob(
        artifact_id=artifact_id,
        kind=kind,
        media_type="application/json",
        data=dumps_readable(payload).encode("utf-8"),
    )


def mock_options(request: RunRequestSpec) -> dict[str, Any]:
    """Validated mock scenario options from ``request.metadata["mock"]``."""
    raw = request.metadata.get("mock", {}) if isinstance(request.metadata, dict) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InputError("request.metadata['mock'] must be an object", code="INVALID_MOCK_OPTIONS")
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if unknown:
        raise InputError(
            f"unknown mock option keys {unknown}", code="INVALID_MOCK_OPTIONS", details={"known": sorted(KNOWN_KEYS)}
        )
    compile_mode = raw.get("compile", "ok")
    if compile_mode not in COMPILE_MODES:
        raise InputError(f"mock.compile must be one of {list(COMPILE_MODES)}, got {compile_mode!r}", code="INVALID_MOCK_OPTIONS")
    correctness = raw.get("correctness", "pass")
    if correctness not in CORRECTNESS_MODES:
        raise InputError(
            f"mock.correctness must be one of {list(CORRECTNESS_MODES)}, got {correctness!r}", code="INVALID_MOCK_OPTIONS"
        )
    raise_mode = raw.get("raise")
    if raise_mode not in RAISE_MODES:
        raise InputError(f"mock.raise must be one of {list(RAISE_MODES)}, got {raise_mode!r}", code="INVALID_MOCK_OPTIONS")
    samples = raw.get("samples")
    if samples is not None:
        if not isinstance(samples, list) or not samples:
            raise InputError("mock.samples must be a non-empty list of positive integers", code="INVALID_MOCK_OPTIONS")
        for value in samples:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InputError(f"mock.samples entries must be positive integers, got {value!r}", code="INVALID_MOCK_OPTIONS")
    overrides = raw.get("environment_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise InputError("mock.environment_overrides must be an object", code="INVALID_MOCK_OPTIONS")
    if "backend" in overrides and overrides["backend"] != "mock":
        raise InputError(
            "the mock adapter never labels itself as anything but backend 'mock'",
            code="MOCK_BACKEND_IMMUTABLE",
            details={"requested_backend": overrides["backend"]},
        )
    return {
        "compile": compile_mode,
        "correctness": correctness,
        "raise": raise_mode,
        "samples": list(samples) if samples is not None else None,
        "environment_overrides": dict(overrides),
    }


def default_samples(repetitions: int) -> list[int]:
    """The deterministic default pattern cycled to ``repetitions`` entries."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise InputError(f"protocol.repetitions must be a positive integer, got {repetitions!r}", code="INVALID_PROTOCOL")
    pattern = DEFAULT_SAMPLE_PATTERN
    return [pattern[i % len(pattern)] for i in range(repetitions)]


class MockAdapter:
    """Kernel adapter that fabricates nothing about hardware: every output is flagged mock."""

    def __init__(self, *, backend: str = "mock", adapter_id: str = "mock-v1") -> None:
        if backend != "mock":
            raise InputError("MockAdapter backend must be 'mock'", code="MOCK_BACKEND_IMMUTABLE")
        self.backend = backend
        self.adapter_id = adapter_id

    # ------------------------------------------------------------------ environment
    def check_environment(self) -> dict[str, Any]:
        env = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in MOCK_ENVIRONMENT.items()}
        env["software"] = {"runner": self.adapter_id}
        return env

    def _environment_for(self, options: dict[str, Any]) -> dict[str, Any]:
        env = self.check_environment()
        for key, value in options["environment_overrides"].items():
            if key not in env:
                raise InputError(
                    f"environment override {key!r} is not an Environment field", code="INVALID_MOCK_OPTIONS", details={"known": sorted(env)}
                )
            env[key] = value
        env["backend"] = "mock"
        return env

    # ------------------------------------------------------------------ prepare
    def prepare(self, request: RunRequestSpec, problem: dict[str, Any]) -> PreparedExecution:
        if not isinstance(problem, dict):
            raise InputError("problem must be a JSON object", code="INVALID_PROBLEM")
        options = mock_options(request)
        spec = request.source
        source = SourceSnapshot(
            repo_uid=spec.repo_uid,
            target_commit=spec.target_commit,
            tested_commit=spec.target_commit,
            tested_tree=None,
            checkout_mode=spec.checkout_mode,
            merge_parent_oids=[],
            dirty=False,
            patch_digest=None,
            source_digest=jcs_digest({"mock_source": spec.entrypoint}),
            entrypoint=spec.entrypoint,
            implementation_overrides=dict(spec.implementation_overrides),
        )
        repetitions = request.protocol.get("repetitions", len(DEFAULT_SAMPLE_PATTERN)) if isinstance(request.protocol, dict) else len(
            DEFAULT_SAMPLE_PATTERN
        )
        samples = options["samples"] if options["samples"] is not None else default_samples(repetitions)
        return PreparedExecution(
            request=request,
            problem=dict(problem),
            source=source,
            environment=self._environment_for(options),
            input_suite_hash=jcs_digest({"mock_suite": problem}),
            handle={"options": options, "samples": samples, "adapter_id": self.adapter_id},
            notes=["mock adapter: nothing was executed; all reports are scripted by request.metadata['mock']"],
        )

    @staticmethod
    def _handle(prepared: PreparedExecution) -> dict[str, Any]:
        handle = prepared.handle
        if not isinstance(handle, dict) or "options" not in handle or "samples" not in handle:
            raise InputError("PreparedExecution was not produced by MockAdapter.prepare", code="INVALID_PREPARED_EXECUTION")
        return handle

    # ------------------------------------------------------------------ stages
    def compile(self, prepared: PreparedExecution) -> CompileReport:
        handle = self._handle(prepared)
        mode = handle["options"]["compile"]
        request_id = prepared.request.request_id
        if mode == "compile_error":
            message = "mock compile error (scripted by request.metadata['mock']['compile'])"
            blob = _json_blob(
                f"{request_id}-compile-log",
                "compile_log",
                {"kind": "compile_log", "status": "compile_error", "message": message, "entrypoint": prepared.source.entrypoint},
            )
            return CompileReport(status="compile_error", message=message, artifacts=[blob])
        message = "mock compile ok (nothing was compiled)"
        blob = _json_blob(
            f"{request_id}-compile-log",
            "compile_log",
            {"kind": "compile_log", "status": "ok", "message": message, "entrypoint": prepared.source.entrypoint},
        )
        return CompileReport(status="ok", message=message, artifacts=[blob])

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        handle = self._handle(prepared)
        mode = handle["options"]["correctness"]
        request_id = prepared.request.request_id
        tolerances = prepared.request.verifier.get("tolerances", {}) if isinstance(prepared.request.verifier, dict) else {}
        policy = prepared.request.verifier.get("nonfinite_policy") if isinstance(prepared.request.verifier, dict) else None
        base = {
            "kind": "correctness_report",
            "tolerances": dict(tolerances) if isinstance(tolerances, dict) else tolerances,
            "nonfinite_policy": policy,
            "reference_id": "mock:no-reference-executed",
            "cases_total": 1,
        }
        if mode == "pass":
            message = "mock correctness pass (scripted; no computation was compared)"
            blob = _json_blob(
                f"{request_id}-correctness",
                "correctness_report",
                {**base, "status": "pass", "cases_passed": 1, "max_abs_error": 0.0, "max_rel_error": 0.0, "message": message},
            )
            return CorrectnessReport(
                status="pass", cases_total=1, cases_passed=1, max_abs_error=0.0, max_rel_error=0.0, message=message, artifacts=[blob]
            )
        if mode == "fail":
            message = "mock correctness fail (scripted): candidate output declared wrong"
            blob = _json_blob(
                f"{request_id}-correctness",
                "correctness_report",
                {**base, "status": "fail", "cases_passed": 0, "max_abs_error": 1.0, "max_rel_error": 1.0, "message": message},
            )
            return CorrectnessReport(
                status="fail", cases_total=1, cases_passed=0, max_abs_error=1.0, max_rel_error=1.0, message=message, artifacts=[blob]
            )
        message = "mock correctness error (scripted): verifier could not run"
        blob = _json_blob(
            f"{request_id}-correctness",
            "correctness_report",
            {**base, "status": "error", "cases_passed": 0, "max_abs_error": None, "max_rel_error": None, "message": message},
        )
        return CorrectnessReport(
            status="error", cases_total=1, cases_passed=0, max_abs_error=None, max_rel_error=None, message=message, artifacts=[blob]
        )

    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        handle = self._handle(prepared)
        raise_mode = handle["options"]["raise"]
        request_id = prepared.request.request_id
        details = {"request_id": request_id, "scripted_by": "request.metadata['mock']['raise']", "is_mock": True}
        if raise_mode == "runtime_error":
            raise ExecutionInfrastructureError("mock runtime error during benchmark", code="RUNTIME_ERROR", details=details)
        if raise_mode == "timeout":
            raise ExecutionInfrastructureError(
                "mock benchmark exceeded max_wall_seconds",
                code="TIMEOUT",
                details={**details, "max_wall_seconds": prepared.request.max_wall_seconds},
            )
        if raise_mode == "infrastructure_error":
            raise ExecutionInfrastructureError("mock execution infrastructure failure", details=details)
        samples = [int(s) for s in handle["samples"]]
        protocol = prepared.request.protocol if isinstance(prepared.request.protocol, dict) else {}
        blob = _json_blob(
            f"{request_id}-samples",
            "latency_samples",
            {
                "unit": "nanoseconds",
                "samples": samples,
                "count": len(samples),
                "timing_method": "mock",
                "measured": False,
                "protocol_id": protocol.get("protocol_id"),
                "is_fixture": False,
            },
        )
        return TimingReport(
            status="recorded",
            unit="nanoseconds",
            samples=list(samples),
            timing_method="mock",
            message="mock samples (scripted, not measured)",
            artifacts=[blob],
        )


__all__ = [
    "DEFAULT_SAMPLE_PATTERN",
    "KNOWN_KEYS",
    "MOCK_ENVIRONMENT",
    "MockAdapter",
    "default_samples",
    "mock_options",
]
