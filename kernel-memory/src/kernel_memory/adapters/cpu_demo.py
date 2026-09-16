"""Real CPU demonstration adapter: executes and times vector addition (specification section 17).

This adapter proves *execution integration*, not optimisation benefit. It really runs the
candidate on numpy arrays, compares against an independent pure-Python reference, and records
real ``perf_counter_ns`` host samples.

Public API
----------
Candidate / reference functions (module-level, allowlisted by name):

* ``vector_add_reference(x, y)``: pure Python, independent of numpy; the trusted reference.
  It is evaluated in Python float64 arithmetic and rounded to the contract dtype (float32)
  before comparison.
* ``vector_add_numpy(x, y)``: baseline candidate, ``numpy.add`` in float32.
* ``vector_add_numpy_chunked(x, y, chunk)``: variant that adds in chunks into a preallocated
  output; ``chunk`` comes from ``implementation_overrides={"chunk": int}`` (same source,
  different variant digest — specification section 5.3).
* ``vector_add_wrong(x, y)``: **deliberately incorrect** (adds ``1e-3``). It exists only so the
  correctness gate can be tested (acceptance T15: succeeded + fail, never promotable). It is
  not an optimisation candidate.

``ALLOWED_ENTRYPOINTS`` maps ``"kernel_memory.adapters.cpu_demo:<name>"`` to those callables.
Any other entrypoint raises ``SecurityPolicyError`` (code ``ENTRYPOINT_NOT_ALLOWLISTED``,
exit 7); request strings never cause an import.

``CpuDemoAdapter(backend="cpu", adapter_id="cpu-demo-v1")`` implements ``KernelAdapter``:

* ``check_environment()`` requires numpy (else ``BackendUnavailable``) and returns the host
  environment snapshot (without hash).
* ``prepare(request, problem)`` validates the demo problem (``InputError``), resolves the
  allowlisted entrypoint, validates overrides, captures the source snapshot and generates
  the deterministic input suite (seeded ``numpy.random.default_rng``; cases
  ``uniform_unit``, ``zeros``, ``large_magnitude``, ``alternating_signs``).
  The source identity is the **content address of the code actually loaded**: a manifest of
  this module, ``verify.py`` and ``timing.py`` hashed with ``hashing.source_digest``.
  ``tested_commit = GitOid("sha256", <source_digest hex>)`` — a content address, not a Git
  commit. The request's ``target_commit`` must equal it, otherwise ``InvariantViolation`` with
  code ``TESTED_SOURCE_MISMATCH`` is raised (the runner turns this into a refusal). Use
  ``current_source_commit()`` to register baselines/local commits with the true address.
* ``compile(prepared)`` reports ``ok`` (numpy ufuncs are precompiled); if the entrypoint no
  longer resolves it reports ``compile_error`` instead of raising, so the runner records
  ``compile_error`` with no invented correctness or latency (T14).
* ``verify(prepared)`` runs the candidate on every case and applies
  ``verify.elementwise_close`` with the request verifier's tolerances and nonfinite policy.
* ``benchmark(prepared)`` measures with ``timing.collect_samples_ns``. Boundary: *host call
  of the entrypoint, inputs preallocated (case 0), output consumed (``out[0]`` and
  ``out[-1]`` folded into a Python float accumulator inside the timed region), no allocation
  of inputs inside the timed region*. ``request.max_wall_seconds`` is honoured between
  repetitions (``ExecutionInfrastructureError`` code ``TIMEOUT``).

Helpers: ``default_cpu_protocol(repetitions=100, warmup=20)`` and ``default_cpu_verifier()``
return protocol/verifier snapshots without their hash fields; ``prepare`` supplies the real
``input_suite_hash`` (the protocol default carries an all-zero placeholder).
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Callable

import kernel_memory

from ..domain.errors import (
    BackendUnavailable,
    ExecutionInfrastructureError,
    InputError,
    InvariantViolation,
    KernelMemoryError,
    SecurityPolicyError,
)
from ..domain.hashing import jcs_digest, sha256_bytes, source_digest
from ..domain.jsonio import dumps_readable
from ..domain.models import GitOid
from ..domain.schema import demo_problem_schema, validate_against
from . import timing, verify
from .base import (
    ArtifactBlob,
    CompileReport,
    CorrectnessReport,
    PreparedExecution,
    RunRequestSpec,
    SourceSnapshot,
    TimingReport,
)

try:  # numpy is a declared dependency, but the adapter must refuse honestly if it is missing.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in broken environments
    np = None  # type: ignore[assignment]

MODULE_PREFIX = "kernel_memory.adapters.cpu_demo"
REFERENCE_ENTRYPOINT = f"{MODULE_PREFIX}:vector_add_reference"
SUITE_ID = "demo-vector-add-v1"
CASE_NAMES: tuple[str, ...] = ("uniform_unit", "zeros", "large_magnitude", "alternating_signs")
LARGE_MAGNITUDE_SCALE = 1.0e6
DEFAULT_CHUNK = 4096
DEFAULT_PROTOCOL_ID = "cpu-demo-host-sync-v1"
DEFAULT_MEASUREMENT_SCOPE = "host_call_excl_input_alloc"
DEFAULT_VERIFIER_ID = "cpu-demo-verifier-v1"
INPUT_SUITE_HASH_PLACEHOLDER = "sha256:" + "0" * 64
TIMING_BOUNDARY = (
    "host call of the entrypoint, inputs preallocated (case 0), output consumed (out[0] and out[-1] folded into a "
    "Python float accumulator inside the timed region), no allocation of inputs inside the timed region"
)
SUITE_SPEC: dict[str, Any] = {
    "suite": SUITE_ID,
    "generator": "numpy.random.default_rng(seed) with seed derived from jcs_digest({'suite', 'n'})",
    "dtype": "float32",
    "cases": [
        {"name": "uniform_unit", "definition": "x, y ~ Uniform(-1, 1)"},
        {"name": "zeros", "definition": "x = y = 0"},
        {"name": "large_magnitude", "definition": f"x, y ~ Uniform(-1, 1) * {LARGE_MAGNITUDE_SCALE:g}"},
        {"name": "alternating_signs", "definition": "x = +u, y = -u with alternating sign per element, u ~ Uniform(0, 1)"},
    ],
    "reference": REFERENCE_ENTRYPOINT,
}


# --------------------------------------------------------------------------------------
# Reference and candidates
# --------------------------------------------------------------------------------------
def vector_add_reference(x: list[float], y: list[float]) -> list[float]:
    """Pure-Python elementwise addition (the trusted reference; independent of numpy)."""
    if len(x) != len(y):
        raise InputError(f"reference inputs differ in length: {len(x)} vs {len(y)}", code="INVALID_INPUTS")
    return [float(a) + float(b) for a, b in zip(x, y)]


def vector_add_numpy(x: Any, y: Any) -> Any:
    """Baseline candidate: numpy float32 addition."""
    return np.add(x, y, dtype=np.float32)


def vector_add_numpy_chunked(x: Any, y: Any, chunk: int = DEFAULT_CHUNK) -> Any:
    """Variant candidate: adds ``chunk`` elements at a time into a preallocated output."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        np.add(x[start:stop], y[start:stop], out=out[start:stop], dtype=np.float32)
    return out


def vector_add_wrong(x: Any, y: Any) -> Any:
    """DELIBERATELY INCORRECT candidate (adds 1e-3). Exists only to test the correctness gate (T15)."""
    return (np.add(x, y, dtype=np.float32) + np.float32(1e-3)).astype(np.float32, copy=False)


ALLOWED_ENTRYPOINTS: dict[str, Callable[..., Any]] = {
    f"{MODULE_PREFIX}:vector_add_numpy": vector_add_numpy,
    f"{MODULE_PREFIX}:vector_add_numpy_chunked": vector_add_numpy_chunked,
    f"{MODULE_PREFIX}:vector_add_wrong": vector_add_wrong,
}
CHUNKED_ENTRYPOINT = f"{MODULE_PREFIX}:vector_add_numpy_chunked"


def resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """Look the entrypoint up in the allowlist. Never imports anything named by the request."""
    if not isinstance(entrypoint, str) or entrypoint not in ALLOWED_ENTRYPOINTS:
        raise SecurityPolicyError(
            f"entrypoint {entrypoint!r} is not in the CPU demo allowlist",
            code="ENTRYPOINT_NOT_ALLOWLISTED",
            details={"entrypoint": entrypoint, "allowed": sorted(ALLOWED_ENTRYPOINTS)},
        )
    return ALLOWED_ENTRYPOINTS[entrypoint]


# --------------------------------------------------------------------------------------
# Source identity (content address of the code actually loaded)
# --------------------------------------------------------------------------------------
_MANIFEST_FILES: tuple[str, ...] = ("cpu_demo.py", "verify.py", "timing.py")


def source_manifest() -> list[tuple[str, str]]:
    """(relative path inside the package, sha256 of bytes) for the modules that execute the demo."""
    here = Path(__file__).resolve().parent
    manifest: list[tuple[str, str]] = []
    for name in _MANIFEST_FILES:
        path = here / name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ExecutionInfrastructureError(
                f"cannot read loaded source file {path}: {exc}", code="SOURCE_UNREADABLE", details={"path": str(path)}
            ) from exc
        manifest.append((f"kernel_memory/adapters/{name}", sha256_bytes(data)))
    return manifest


def current_source_digest() -> str:
    return source_digest(source_manifest())


def current_source_commit() -> GitOid:
    """Content address of the loaded demo source, expressed as a sha256 GitOid (not a Git commit)."""
    digest = current_source_digest()
    return GitOid(algorithm="sha256", hex=digest.split(":", 1)[1])


def _module_bytes_sha256() -> str:
    return sha256_bytes(Path(__file__).resolve().read_bytes())


# --------------------------------------------------------------------------------------
# Defaults (project proposals, not guarantees)
# --------------------------------------------------------------------------------------
def default_cpu_protocol(repetitions: int = 100, warmup: int = 20) -> dict[str, Any]:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise InputError("repetitions must be a positive integer", code="INVALID_PROTOCOL")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise InputError("warmup must be a non-negative integer", code="INVALID_PROTOCOL")
    return {
        "protocol_id": DEFAULT_PROTOCOL_ID,
        "measurement_scope": DEFAULT_MEASUREMENT_SCOPE,
        "timing_method": "host_synchronized",
        "include_compile": False,
        "include_transfers": False,
        "warmup": warmup,
        "repetitions": repetitions,
        "statistic": "median",
        "quantile_method": "linear",
        "input_suite_hash": INPUT_SUITE_HASH_PLACEHOLDER,
        "capture_profile": False,
    }


def default_cpu_verifier() -> dict[str, Any]:
    return {
        "verifier_id": DEFAULT_VERIFIER_ID,
        "reference_source_hash": _module_bytes_sha256(),
        "suite_hash": jcs_digest(SUITE_SPEC),
        "tolerances": {"atol": "1e-6", "rtol": "1e-6"},
        "nonfinite_policy": "reject_unexpected",
    }


# --------------------------------------------------------------------------------------
# Input suite
# --------------------------------------------------------------------------------------
def suite_seed(n: int) -> int:
    """Deterministic seed derived from the suite identity and the problem size."""
    digest = jcs_digest({"suite": SUITE_ID, "n": n})
    return int(digest.split(":", 1)[1][:16], 16) % (2**32)


def input_suite_hash(n: int, seed: int) -> str:
    return jcs_digest({"suite": SUITE_ID, "n": n, "seed": seed, "cases": list(CASE_NAMES)})


def generate_cases(n: int, seed: int) -> list[tuple[str, Any, Any]]:
    rng = np.random.default_rng(seed)
    uniform_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    uniform_y = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    large_x = (rng.uniform(-1.0, 1.0, size=n) * LARGE_MAGNITUDE_SCALE).astype(np.float32)
    large_y = (rng.uniform(-1.0, 1.0, size=n) * LARGE_MAGNITUDE_SCALE).astype(np.float32)
    magnitude = rng.uniform(0.0, 1.0, size=n)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    alt_x = (magnitude * signs).astype(np.float32)
    alt_y = (-magnitude * signs).astype(np.float32)
    return [
        ("uniform_unit", uniform_x, uniform_y),
        ("zeros", zeros, zeros.copy()),
        ("large_magnitude", large_x, large_y),
        ("alternating_signs", alt_x, alt_y),
    ]


# --------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------
class CpuDemoAdapter:
    backend = "cpu"
    adapter_id = "cpu-demo-v1"

    def __init__(self, *, backend: str = "cpu", adapter_id: str = "cpu-demo-v1") -> None:
        self.backend = backend
        self.adapter_id = adapter_id

    # ------------------------------------------------------------------ environment
    def check_environment(self) -> dict[str, Any]:
        if np is None:
            raise BackendUnavailable(
                "numpy is not importable; the CPU demo adapter cannot execute",
                details={"backend": self.backend, "missing": "numpy"},
            )
        return {
            "backend": self.backend,
            "accelerator_model": f"{platform.machine()}/{platform.processor() or 'unknown'}",
            "device_count": 1,
            "topology": "single_host",
            "software": {
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "platform": platform.platform(),
                "kernel_memory": str(getattr(kernel_memory, "__version__", "0.2.0")),
            },
            "execution_flags": {},
            "host_timer_environment": timing.host_timer_environment(),
            "unknown_required_fields": [],
        }

    # ------------------------------------------------------------------ prepare
    @staticmethod
    def validate_problem(problem: Any) -> dict[str, Any]:
        if not isinstance(problem, dict):
            raise InputError("problem must be a JSON object", code="INVALID_PROBLEM")
        validate_against(demo_problem_schema(), problem, what="CPU demo problem")
        if problem.get("operation") != "vector_add" or problem.get("dtype") != "float32" or problem.get("outputs") != ["y"]:
            raise InputError(
                "CPU demo adapter only executes {operation: vector_add, dtype: float32, outputs: [y]}",
                code="INVALID_PROBLEM",
                details={"problem": problem},
            )
        return dict(problem)

    @staticmethod
    def validate_overrides(entrypoint: str, overrides: Any) -> dict[str, Any]:
        if overrides is None:
            return {}
        if not isinstance(overrides, dict):
            raise InputError("implementation_overrides must be an object", code="INVALID_OVERRIDES")
        unknown = sorted(set(overrides) - {"chunk"})
        if unknown:
            raise InputError(
                f"unknown implementation_overrides {unknown} for the CPU demo", code="INVALID_OVERRIDES", details={"known": ["chunk"]}
            )
        if "chunk" in overrides:
            chunk = overrides["chunk"]
            if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
                raise InputError(f"override 'chunk' must be a positive integer, got {chunk!r}", code="INVALID_OVERRIDES")
            if entrypoint != CHUNKED_ENTRYPOINT:
                raise InputError(
                    f"override 'chunk' only applies to {CHUNKED_ENTRYPOINT}", code="INVALID_OVERRIDES", details={"entrypoint": entrypoint}
                )
        return dict(overrides)

    def prepare(self, request: RunRequestSpec, problem: dict[str, Any]) -> PreparedExecution:
        environment = self.check_environment()
        problem = self.validate_problem(problem)
        spec = request.source
        fn = resolve_entrypoint(spec.entrypoint)
        overrides = self.validate_overrides(spec.entrypoint, spec.implementation_overrides)
        if not isinstance(request.protocol, dict) or not isinstance(request.verifier, dict):
            raise InputError("request.protocol and request.verifier must be objects", code="INVALID_REQUEST")
        for key in ("warmup", "repetitions"):
            if key not in request.protocol:
                raise InputError(f"request.protocol is missing {key!r}", code="INVALID_PROTOCOL")
        tolerances = request.verifier.get("tolerances")
        if not isinstance(tolerances, dict) or "atol" not in tolerances or "rtol" not in tolerances:
            raise InputError("request.verifier.tolerances must contain 'atol' and 'rtol'", code="INVALID_VERIFIER")
        if "nonfinite_policy" not in request.verifier:
            raise InputError("request.verifier is missing 'nonfinite_policy'", code="INVALID_VERIFIER")
        # Validate tolerances/policy now so a bad verifier fails before any execution.
        verify.parse_tolerance(tolerances["atol"], "atol")
        verify.parse_tolerance(tolerances["rtol"], "rtol")
        if request.verifier["nonfinite_policy"] not in verify.NONFINITE_POLICIES:
            raise InputError(
                f"unknown nonfinite_policy {request.verifier['nonfinite_policy']!r}",
                code="INVALID_NONFINITE_POLICY",
                details={"known": list(verify.NONFINITE_POLICIES)},
            )

        digest = current_source_digest()
        tested_commit = GitOid(algorithm="sha256", hex=digest.split(":", 1)[1])
        target = spec.target_commit
        if target.algorithm != tested_commit.algorithm or target.hex != tested_commit.hex:
            raise InvariantViolation(
                "request.source.target_commit does not match the content address of the loaded CPU demo source; "
                "register the subject with cpu_demo.current_source_commit()",
                code="TESTED_SOURCE_MISMATCH",
                details={
                    "target_commit": {"algorithm": target.algorithm, "hex": target.hex},
                    "tested_commit": {"algorithm": tested_commit.algorithm, "hex": tested_commit.hex},
                    "source_digest": digest,
                },
            )
        if spec.checkout_mode != "exact_commit":
            raise InputError(
                f"CPU demo executes in-process source only; checkout_mode must be 'exact_commit', got {spec.checkout_mode!r}",
                code="INVALID_CHECKOUT_MODE",
            )
        source = SourceSnapshot(
            repo_uid=spec.repo_uid,
            target_commit=target,
            tested_commit=tested_commit,
            tested_tree=None,
            checkout_mode=spec.checkout_mode,
            merge_parent_oids=[],
            dirty=False,
            patch_digest=None,
            source_digest=digest,
            entrypoint=spec.entrypoint,
            implementation_overrides=overrides,
        )
        n = int(problem["n"])
        seed = suite_seed(n)
        cases = generate_cases(n, seed)
        notes = [f"timing boundary: {TIMING_BOUNDARY}"]
        effective_overrides = dict(overrides)
        if spec.entrypoint == CHUNKED_ENTRYPOINT and "chunk" not in effective_overrides:
            effective_overrides["chunk"] = DEFAULT_CHUNK
            notes.append(f"chunked variant without explicit 'chunk' override: using documented default {DEFAULT_CHUNK}")
        return PreparedExecution(
            request=request,
            problem=problem,
            source=source,
            environment=environment,
            input_suite_hash=input_suite_hash(n, seed),
            handle={
                "adapter_id": self.adapter_id,
                "callable": fn,
                "kwargs": effective_overrides,
                "cases": cases,
                "seed": seed,
                "n": n,
            },
            notes=notes,
        )

    @staticmethod
    def _handle(prepared: PreparedExecution) -> dict[str, Any]:
        handle = prepared.handle
        if not isinstance(handle, dict) or "cases" not in handle or "callable" not in handle:
            raise InputError("PreparedExecution was not produced by CpuDemoAdapter.prepare", code="INVALID_PREPARED_EXECUTION")
        return handle

    def _call(self, handle: dict[str, Any]) -> Callable[[Any, Any], Any]:
        fn = handle["callable"]
        kwargs = dict(handle.get("kwargs") or {})
        if kwargs:
            return lambda x, y: fn(x, y, **kwargs)
        return fn

    # ------------------------------------------------------------------ compile
    def compile(self, prepared: PreparedExecution) -> CompileReport:
        request_id = prepared.request.request_id
        entrypoint = prepared.source.entrypoint
        try:
            handle = self._handle(prepared)
            fn = resolve_entrypoint(entrypoint)
            if handle["callable"] is not fn:
                raise InvariantViolation(
                    "prepared callable does not match the allowlisted entrypoint", code="ENTRYPOINT_MISMATCH", details={"entrypoint": entrypoint}
                )
        except KernelMemoryError as exc:
            message = f"entrypoint resolution failed: {exc.message}"
            blob = self._json_blob(
                f"{request_id}-compile-log",
                "compile_log",
                {"kind": "compile_log", "status": "compile_error", "message": message, "error": exc.to_dict(), "entrypoint": entrypoint},
            )
            return CompileReport(status="compile_error", message=message, artifacts=[blob])
        message = f"numpy ufuncs are precompiled; nothing to compile for {entrypoint} (numpy {np.__version__})"
        blob = self._json_blob(
            f"{request_id}-compile-log",
            "compile_log",
            {"kind": "compile_log", "status": "ok", "message": message, "entrypoint": entrypoint, "overrides": handle.get("kwargs", {})},
        )
        return CompileReport(status="ok", message=message, artifacts=[blob])

    @staticmethod
    def _json_blob(artifact_id: str, kind: str, document: dict[str, Any]) -> ArtifactBlob:
        payload = dict(document)
        payload.setdefault("is_fixture", False)
        return ArtifactBlob(artifact_id=artifact_id, kind=kind, media_type="application/json", data=dumps_readable(payload).encode("utf-8"))

    # ------------------------------------------------------------------ verify
    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        handle = self._handle(prepared)
        request = prepared.request
        request_id = request.request_id
        tolerances = request.verifier["tolerances"]
        atol = tolerances["atol"]
        rtol = tolerances["rtol"]
        policy = request.verifier["nonfinite_policy"]
        call = self._call(handle)
        case_results: list[dict[str, Any]] = []
        max_abs: float | None = None
        max_rel: float | None = None
        passed_count = 0
        error_message: str | None = None
        for name, x, y in handle["cases"]:
            try:
                candidate = call(x, y)
                reference = np.asarray(vector_add_reference(x.tolist(), y.tolist()), dtype=np.float32)
                outcome = verify.elementwise_close(candidate, reference, atol=atol, rtol=rtol, nonfinite_policy=policy)
            except KernelMemoryError as exc:
                error_message = f"case {name}: {exc.message}"
                case_results.append({"case": name, "n": int(x.shape[0]), "passed": False, "error": exc.to_dict()})
                break
            except Exception as exc:  # candidate raised: execution error, not a verdict
                error_message = f"case {name}: candidate raised {type(exc).__name__}: {exc}"
                case_results.append({"case": name, "n": int(x.shape[0]), "passed": False, "error": {"type": type(exc).__name__, "message": str(exc)}})
                break
            case_results.append(
                {
                    "case": name,
                    "n": int(x.shape[0]),
                    "passed": outcome.passed,
                    "checked_count": outcome.checked_count,
                    "mismatch_count": outcome.mismatch_count,
                    "max_abs_error": outcome.max_abs_error,
                    "max_rel_error": outcome.max_rel_error,
                    "message": outcome.message,
                }
            )
            if outcome.passed:
                passed_count += 1
            if outcome.max_abs_error is not None:
                max_abs = outcome.max_abs_error if max_abs is None else max(max_abs, outcome.max_abs_error)
            if outcome.max_rel_error is not None:
                max_rel = outcome.max_rel_error if max_rel is None else max(max_rel, outcome.max_rel_error)
        total = len(handle["cases"])
        if error_message is not None:
            status = "error"
        elif passed_count == total:
            status = "pass"
        else:
            status = "fail"
        message = error_message
        if message is None and status == "fail":
            failing = [c["case"] for c in case_results if not c["passed"]]
            message = f"{total - passed_count} of {total} cases failed the acceptance rule: {failing}"
        blob = ArtifactBlob(
            artifact_id=f"{request_id}-correctness",
            kind="correctness_report",
            media_type="application/json",
            data=verify.correctness_report_bytes(
                status=status,
                cases=case_results,
                tolerances={"atol": str(atol), "rtol": str(rtol)},
                nonfinite_policy=policy,
                reference_id=REFERENCE_ENTRYPOINT,
                entrypoint=prepared.source.entrypoint,
                implementation_overrides=dict(handle.get("kwargs") or {}),
                suite=SUITE_ID,
                seed=handle["seed"],
                input_suite_hash=prepared.input_suite_hash,
                reference_note="pure-Python float64 addition rounded to float32 (the contract dtype) before comparison",
                message=message,
            ),
        )
        return CorrectnessReport(
            status=status,
            cases_total=total,
            cases_passed=passed_count,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            message=message,
            artifacts=[blob],
        )

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        handle = self._handle(prepared)
        request = prepared.request
        protocol = request.protocol
        warmup = protocol["warmup"]
        repetitions = protocol["repetitions"]
        call = self._call(handle)
        name, x, y = handle["cases"][0]  # inputs preallocated outside the timed region
        sink = {"acc": 0.0, "calls": 0}

        def timed_call() -> Any:
            return call(x, y)

        def consume(out: Any) -> None:
            # Reading two elements forces the result to exist; the accumulator keeps the work observable.
            sink["acc"] += float(out[0]) + float(out[-1])
            sink["calls"] += 1

        samples = timing.collect_samples_ns(
            timed_call,
            warmup=warmup,
            repetitions=repetitions,
            block_until_ready=consume,
            max_wall_seconds=request.max_wall_seconds,
        )
        artifact = ArtifactBlob(
            artifact_id=f"{request.request_id}-samples",
            kind="latency_samples",
            media_type="application/json",
            data=timing.samples_artifact_bytes(
                samples,
                "nanoseconds",
                timing_method="host_synchronized",
                protocol_id=protocol.get("protocol_id"),
                measurement_scope=protocol.get("measurement_scope", DEFAULT_MEASUREMENT_SCOPE),
                boundary=TIMING_BOUNDARY,
                warmup=warmup,
                repetitions=repetitions,
                entrypoint=prepared.source.entrypoint,
                implementation_overrides=dict(handle.get("kwargs") or {}),
                input_case=name,
                n=handle["n"],
                output_consumed_calls=sink["calls"],
                output_checksum=sink["acc"],
                timer=timing.TIMER_NAME,
            ),
        )
        return TimingReport(
            status="recorded",
            unit="nanoseconds",
            samples=[int(s) for s in samples],
            timing_method="host_synchronized",
            message=f"{len(samples)} host-synchronized samples of {prepared.source.entrypoint} on case {name!r} (n={handle['n']})",
            artifacts=[artifact],
        )


__all__ = [
    "ALLOWED_ENTRYPOINTS",
    "CASE_NAMES",
    "CpuDemoAdapter",
    "DEFAULT_CHUNK",
    "REFERENCE_ENTRYPOINT",
    "SUITE_ID",
    "SUITE_SPEC",
    "TIMING_BOUNDARY",
    "current_source_commit",
    "current_source_digest",
    "default_cpu_protocol",
    "default_cpu_verifier",
    "generate_cases",
    "input_suite_hash",
    "resolve_entrypoint",
    "source_manifest",
    "suite_seed",
    "vector_add_numpy",
    "vector_add_numpy_chunked",
    "vector_add_reference",
    "vector_add_wrong",
]
