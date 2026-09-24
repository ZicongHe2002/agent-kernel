"""Controlled JAX/TPU kernel adapter interface (specification sections 10, 11, 17, T29).

The adapter inspects the *actual* JAX backend and devices and refuses honestly when the
required platform is absent. It never falls back to CPU while labelled as TPU, never guesses
an entrypoint, and never fabricates a compiler build id.

Public API
----------
``ResolvedEntrypoint(callable, input_factory, reference_callable, required_outputs, source_files=())``
    What a configured entrypoint resolver returns. ``input_factory(problem)`` returns the
    positional inputs (a tuple, or a single pytree of arrays that is wrapped as one argument).
    ``reference_callable`` (optional) takes the same inputs and returns the trusted reference
    output pytree; without it correctness can only be ``not_run`` (section 10).
    ``required_outputs`` names the dict keys that must be present in the output (e.g.
    ``["o", "lse"]``); a missing one fails verification with ``MISSING_REQUIRED_OUTPUT``.
    ``source_files`` optionally lists the module files that constitute the tested source; by
    default they are discovered with ``inspect.getsourcefile`` on the resolved callables.

``JaxAdapter(*, required_platform="tpu", backend="jax_tpu", adapter_id="jax-tpu-v0",
entrypoint_resolver=None, jax_loader=None)``

* ``check_environment()``: imports ``jax`` (``BackendUnavailable`` "jax not installed" when
  it cannot be imported), reads ``jax.devices()`` and ``jax.default_backend()``; if the
  backend is not ``required_platform`` or no device of that platform exists it raises
  ``BackendUnavailable`` with details ``{"required", "actual_backend", "devices"}``.
  Returns the Environment snapshot (without hash): ``backend`` is the constructor label,
  ``accelerator_model`` the first device kind, ``topology`` ``"<n>x<device_kind>"``, software
  versions as strings (``"unknown"`` when not importable), ``execution_flags`` lists the
  *names* of ``XLA_FLAGS`` / ``LIBTPU_INIT_ARGS`` / ``JAX_PLATFORMS`` if set (never values),
  and ``unknown_required_fields`` contains ``"compiler_build_id"`` unless the PJRT backend
  reports a ``platform_version``.
* ``prepare(request, problem)`` — checks in this order: (1) authorization
  (``allow_tpu_execution`` when ``required_platform == "tpu"``, else ``AuthorizationError``),
  (2) incomplete problem contracts (``kernel_id == "mla_forward"`` in the problem or request
  metadata → ``IncompleteProblemContract``), (3) entrypoint resolution through the configured
  resolver (none / unresolvable → ``PrerequisiteMissingError`` code
  ``ENTRYPOINT_NOT_CONFIGURED``; an ``IncompleteProblemContract`` raised by the resolver
  propagates), (4) device checks via ``check_environment``, (5) the source snapshot:
  without ``repo_path`` the tested commit equals the target (in-process code; the source
  digest over the resolved module files is the real evidence); with ``repo_path`` git is
  consulted through subprocess argument arrays and HEAD must equal the target
  (``InvariantViolation`` code ``TESTED_SOURCE_MISMATCH``), dirty trees are refused unless
  ``allow_dirty_exploratory`` (then ``dirty=True`` with a ``patch_digest``).
* ``compile(prepared)``: ``jax.jit(fn).lower(*inputs).compile()``; any exception becomes a
  ``compile_error`` report (never a raise), with a ``compile_log`` artifact.
* ``verify(prepared)``: runs the compiled function, ``jax.block_until_ready`` on the whole
  output pytree, checks required outputs, compares every leaf with
  ``verify.elementwise_close`` under the request verifier's tolerances/policy.
* ``benchmark(prepared)``: ``timing.collect_samples_ns`` with ``block_until_ready`` on the
  complete output pytree; compilation is excluded (first call and warmups are untimed); all
  outputs are materialised and check-summed after sampling so nothing is eliminated.

``jax_loader`` is a test-only injection point (a callable returning a jax-like module); the
default registry never sets it and the real ``import jax`` is used.

Testing on CPU: ``JaxAdapter(required_platform="cpu", backend="jax_cpu_interface_test")``
exercises the pytree/timing code path on CPU JAX; its environment reports the given label and
platform ``cpu`` — never ``tpu``.
"""
from __future__ import annotations

import importlib
import inspect
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..domain.errors import (
    AuthorizationError,
    BackendUnavailable,
    ExecutionInfrastructureError,
    IncompleteProblemContract,
    InputError,
    InvariantViolation,
    KernelMemoryError,
    PrerequisiteMissingError,
)
from ..domain.hashing import artifact_digest, jcs_digest, sha256_bytes, source_digest
from ..domain.jsonio import dumps_readable
from ..domain.models import GitOid
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

EXECUTION_FLAG_ENV_VARS: tuple[str, ...] = ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS")
COMPILER_BUILD_FIELD = "compiler_build_id"
TIMING_BOUNDARY = (
    "host call of the compiled entrypoint with preallocated device inputs, jax.block_until_ready on the complete "
    "output pytree; compilation, first execution and warmups excluded; no input allocation inside the timed region"
)


@dataclass
class ResolvedEntrypoint:
    callable: Callable[..., Any]
    input_factory: Callable[[dict[str, Any]], Any]
    reference_callable: Callable[..., Any] | None = None
    required_outputs: list[str] = field(default_factory=list)
    source_files: Sequence[str] = ()


# --------------------------------------------------------------------------------------
# git helpers (argument arrays only, explicit repo path)
# --------------------------------------------------------------------------------------
def _git(repo_path: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise PrerequisiteMissingError("git executable not found", code="GIT_MISSING") from exc
    except subprocess.TimeoutExpired as exc:
        raise ExecutionInfrastructureError(f"git {' '.join(args)} timed out", code="GIT_TIMEOUT") from exc
    if completed.returncode != 0:
        raise ExecutionInfrastructureError(
            f"git {' '.join(args)} failed in {repo_path}: {completed.stderr.decode('utf-8', 'replace').strip()}",
            code="GIT_COMMAND_FAILED",
            details={"args": list(args), "returncode": completed.returncode},
        )
    return completed.stdout.decode("utf-8", "replace")


def _oid_from_hex(text: str) -> GitOid:
    value = text.strip()
    if len(value) == 40 and all(c in "0123456789abcdef" for c in value):
        return GitOid(algorithm="sha1", hex=value)
    if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return GitOid(algorithm="sha256", hex=value)
    raise ExecutionInfrastructureError(f"git returned an unrecognised object id {value!r}", code="GIT_INVALID_OID")


def git_head_oid(repo_path: Path) -> GitOid:
    return _oid_from_hex(_git(repo_path, "rev-parse", "--verify", "HEAD^{commit}"))


def git_head_tree_oid(repo_path: Path) -> GitOid:
    return _oid_from_hex(_git(repo_path, "rev-parse", "--verify", "HEAD^{tree}"))


def git_is_dirty(repo_path: Path) -> bool:
    return bool(_git(repo_path, "status", "--porcelain", "--untracked-files=no").strip())


def git_working_tree_patch(repo_path: Path) -> bytes:
    try:
        completed = subprocess.run(["git", "-C", str(repo_path), "diff", "HEAD", "--binary"], capture_output=True, check=False, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ExecutionInfrastructureError(f"git diff failed: {exc}", code="GIT_COMMAND_FAILED") from exc
    if completed.returncode != 0:
        raise ExecutionInfrastructureError("git diff HEAD failed", code="GIT_COMMAND_FAILED")
    return completed.stdout


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - any import failure means "unknown", never a guess
        try:
            from importlib import metadata

            return str(metadata.version(module_name))
        except Exception:  # noqa: BLE001
            return "unknown"
    version = getattr(module, "__version__", None)
    return str(version) if version else "unknown"


def _json_blob(artifact_id: str, kind: str, document: dict[str, Any]) -> ArtifactBlob:
    payload = dict(document)
    payload.setdefault("is_fixture", False)
    return ArtifactBlob(artifact_id=artifact_id, kind=kind, media_type="application/json", data=dumps_readable(payload).encode("utf-8"))


def _leaf_descriptions(tree: Any) -> list[dict[str, Any]]:
    structure = verify.verify_pytree_structure(tree, tree)
    out: list[dict[str, Any]] = []
    for path, leaf, _ in structure.leaves:
        shape = list(getattr(leaf, "shape", ()))
        dtype = getattr(leaf, "dtype", None)
        out.append({"path": path, "shape": [int(s) for s in shape], "dtype": str(getattr(dtype, "name", dtype)) if dtype is not None else type(leaf).__name__})
    return out


def _consume_outputs(output: Any) -> list[dict[str, Any]]:
    """Materialise every output leaf as a host checksum (proves all outputs exist)."""
    import numpy as np

    structure = verify.verify_pytree_structure(output, output)
    sums: list[dict[str, Any]] = []
    for path, leaf, _ in structure.leaves:
        arr = np.asarray(leaf)
        total = float(np.sum(arr.astype(np.float64))) if arr.dtype.kind in "biuf" else None
        sums.append({"path": path, "checksum": total if total is None or np.isfinite(total) else repr(total), "size": int(arr.size)})
    return sums


# --------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------
class JaxAdapter:
    def __init__(
        self,
        *,
        required_platform: str = "tpu",
        backend: str = "jax_tpu",
        adapter_id: str = "jax-tpu-v0",
        entrypoint_resolver: Callable[[str], ResolvedEntrypoint | None] | None = None,
        jax_loader: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(required_platform, str) or not required_platform:
            raise InputError("required_platform must be a non-empty string", code="INVALID_ADAPTER_CONFIG")
        if not isinstance(backend, str) or not backend:
            raise InputError("backend must be a non-empty string", code="INVALID_ADAPTER_CONFIG")
        if required_platform != "tpu" and backend == "jax_tpu":
            raise InputError(
                "backend label 'jax_tpu' is reserved for required_platform='tpu'; a non-TPU instance must use another label",
                code="INVALID_ADAPTER_CONFIG",
            )
        self.required_platform = required_platform
        self.backend = backend
        self.adapter_id = adapter_id
        self._resolver = entrypoint_resolver
        self._jax_loader = jax_loader

    # ------------------------------------------------------------------ jax access
    def _load_jax(self) -> Any:
        if self._jax_loader is not None:
            return self._jax_loader()
        try:
            return importlib.import_module("jax")
        except ImportError as exc:
            raise BackendUnavailable(
                "jax not installed",
                details={"required": self.required_platform, "backend": self.backend, "import_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------ environment
    def check_environment(self) -> dict[str, Any]:
        jax = self._load_jax()
        try:
            devices = list(jax.devices())
            actual = str(jax.default_backend())
        except Exception as exc:  # backend initialisation failure is "unavailable", not a crash
            raise BackendUnavailable(
                f"jax backend initialisation failed: {type(exc).__name__}: {exc}",
                details={"required": self.required_platform, "backend": self.backend},
            ) from exc
        matching = [d for d in devices if str(getattr(d, "platform", "")).lower() == self.required_platform]
        if actual != self.required_platform or not matching:
            raise BackendUnavailable(
                f"required JAX platform {self.required_platform!r} is not available (default backend is {actual!r}); "
                "refusing to fall back to another backend",
                details={
                    "required": self.required_platform,
                    "actual_backend": actual,
                    "devices": [str(d) for d in devices],
                    "backend": self.backend,
                },
            )
        first = matching[0]
        device_kind = str(getattr(first, "device_kind", self.required_platform))
        software: dict[str, str] = {
            "jax": str(getattr(jax, "__version__", "unknown") or "unknown"),
            "jaxlib": _version_of("jaxlib"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libtpu": _version_of("libtpu"),
        }
        unknown_required: list[str] = []
        build_id = self._compiler_build_id(jax)
        if build_id:
            software[COMPILER_BUILD_FIELD] = build_id
        else:
            unknown_required.append(COMPILER_BUILD_FIELD)
        execution_flags = {name: "set" for name in EXECUTION_FLAG_ENV_VARS if os.environ.get(name)}
        return {
            "backend": self.backend,
            "accelerator_model": device_kind,
            "device_count": len(matching),
            "topology": f"{len(matching)}x{device_kind}",
            "software": software,
            "execution_flags": execution_flags,
            "host_timer_environment": timing.host_timer_environment(),
            "unknown_required_fields": unknown_required,
        }

    @staticmethod
    def _compiler_build_id(jax: Any) -> str | None:
        """PJRT ``platform_version`` when the backend exposes it; otherwise None (never invented)."""
        getters = []
        extend = getattr(jax, "extend", None)
        if extend is not None and hasattr(getattr(extend, "backend", None), "get_backend"):
            getters.append(extend.backend.get_backend)
        lib = getattr(jax, "lib", None)
        bridge = getattr(lib, "xla_bridge", None) if lib is not None else None
        if bridge is not None and hasattr(bridge, "get_backend"):
            getters.append(bridge.get_backend)
        for getter in getters:
            try:
                value = getattr(getter(), "platform_version", None)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    # ------------------------------------------------------------------ prepare
    def _check_authorization(self, request: RunRequestSpec) -> None:
        if self.required_platform == "tpu":
            auth = request.authorization if isinstance(request.authorization, dict) else {}
            if auth.get("allow_tpu_execution") is not True:
                raise AuthorizationError(
                    "TPU execution requires authorization flag allow_tpu_execution=true",
                    details={"permission": "allow_tpu_execution", "backend": self.backend, "request_id": request.request_id},
                )

    @staticmethod
    def _check_problem_contract(request: RunRequestSpec, problem: Any) -> dict[str, Any]:
        if not isinstance(problem, dict):
            raise InputError("problem must be a JSON object", code="INVALID_PROBLEM")
        kernel_id = problem.get("kernel_id")
        if kernel_id is None and isinstance(request.metadata, dict):
            kernel_id = request.metadata.get("kernel_id")
        if kernel_id == "mla_forward":
            from ..domain.problems import default_registry

            # Delegates to the problem registry, which refuses with the unresolved-item list.
            default_registry().normalize("mla_forward", problem)
            raise IncompleteProblemContract("mla_forward problem contract is incomplete", details={"kernel_id": kernel_id})
        return dict(problem)

    def _resolve(self, entrypoint: str) -> ResolvedEntrypoint:
        if self._resolver is None:
            raise PrerequisiteMissingError(
                "no JAX entrypoint resolver is configured (project settings kernel_entrypoint is null); refusing to guess",
                code="ENTRYPOINT_NOT_CONFIGURED",
                details={"entrypoint": entrypoint, "backend": self.backend},
            )
        try:
            resolved = self._resolver(entrypoint)
        except IncompleteProblemContract:
            raise
        except KernelMemoryError as exc:
            raise PrerequisiteMissingError(
                f"entrypoint {entrypoint!r} could not be resolved: {exc.message}",
                code="ENTRYPOINT_NOT_CONFIGURED",
                details={"entrypoint": entrypoint, "cause": exc.to_dict()},
            ) from exc
        except (LookupError, ImportError, AttributeError, ValueError, TypeError) as exc:
            raise PrerequisiteMissingError(
                f"entrypoint {entrypoint!r} could not be resolved: {type(exc).__name__}: {exc}",
                code="ENTRYPOINT_NOT_CONFIGURED",
                details={"entrypoint": entrypoint},
            ) from exc
        if not isinstance(resolved, ResolvedEntrypoint) or not callable(resolved.callable) or not callable(resolved.input_factory):
            raise PrerequisiteMissingError(
                f"entrypoint {entrypoint!r} is not configured (resolver returned no ResolvedEntrypoint)",
                code="ENTRYPOINT_NOT_CONFIGURED",
                details={"entrypoint": entrypoint},
            )
        if resolved.reference_callable is not None and not callable(resolved.reference_callable):
            raise InputError("reference_callable must be callable or None", code="INVALID_RESOLVED_ENTRYPOINT")
        if not isinstance(resolved.required_outputs, list) or not all(isinstance(k, str) for k in resolved.required_outputs):
            raise InputError("required_outputs must be a list of strings", code="INVALID_RESOLVED_ENTRYPOINT")
        return resolved

    @staticmethod
    def _source_files(resolved: ResolvedEntrypoint) -> list[Path]:
        if resolved.source_files:
            return [Path(p) for p in resolved.source_files]
        files: list[Path] = []
        for fn in (resolved.callable, resolved.input_factory, resolved.reference_callable):
            if fn is None:
                continue
            target = inspect.unwrap(fn) if callable(fn) else fn
            try:
                path = inspect.getsourcefile(target)
            except TypeError:
                path = None
            if path is None:
                wrapped = getattr(target, "__wrapped__", None) or getattr(target, "fun", None) or getattr(target, "__func__", None)
                try:
                    path = inspect.getsourcefile(wrapped) if wrapped is not None else None
                except TypeError:
                    path = None
            if path and Path(path).is_file():
                files.append(Path(path).resolve())
        return sorted(set(files))

    def _source_snapshot(self, request: RunRequestSpec, resolved: ResolvedEntrypoint) -> tuple[SourceSnapshot, list[str]]:
        spec = request.source
        notes: list[str] = []
        files = self._source_files(resolved)
        if not files:
            raise PrerequisiteMissingError(
                "cannot determine the source files of the resolved entrypoint; supply ResolvedEntrypoint.source_files",
                code="SOURCE_MANIFEST_UNAVAILABLE",
                details={"entrypoint": spec.entrypoint},
            )
        repo_root = Path(spec.repo_path).resolve() if spec.repo_path else None
        manifest: list[tuple[str, str]] = []
        for path in files:
            if repo_root is not None:
                try:
                    rel = path.relative_to(repo_root).as_posix()
                except ValueError:
                    raise InvariantViolation(
                        f"resolved source file {path} lies outside repo_path {repo_root}",
                        code="SOURCE_OUTSIDE_REPO",
                        details={"file": str(path), "repo_path": str(repo_root)},
                    ) from None
            else:
                rel = "/".join(path.parts[-3:]) if len(path.parts) >= 3 else path.name
            try:
                content = path.read_bytes()
            except OSError as exc:
                # A declared source file that cannot be read means the tested source is unknown: refuse, never guess.
                raise PrerequisiteMissingError(
                    f"declared source file {path} cannot be read: {type(exc).__name__}: {exc}",
                    code="SOURCE_MANIFEST_UNAVAILABLE",
                    details={"entrypoint": spec.entrypoint, "file": str(path)},
                ) from exc
            manifest.append((rel, sha256_bytes(content)))
        digest = source_digest(manifest)
        if repo_root is None:
            tested = spec.target_commit
            tested_tree = None
            dirty = False
            patch_digest = None
            notes.append("in-process entrypoint without repo_path: tested_commit is taken from the request; source_digest covers the loaded module files")
        else:
            if not (repo_root / ".git").exists():
                raise InputError(f"repo_path {repo_root} is not a git repository", code="NOT_A_GIT_REPOSITORY")
            head = git_head_oid(repo_root)
            if head.key() != spec.target_commit.key():
                raise InvariantViolation(
                    f"repository HEAD {head.hex} does not match target_commit {spec.target_commit.hex}; refusing to test other source",
                    code="TESTED_SOURCE_MISMATCH",
                    details={"head": {"algorithm": head.algorithm, "hex": head.hex}, "target": {"algorithm": spec.target_commit.algorithm, "hex": spec.target_commit.hex}},
                )
            tested = head
            tested_tree = git_head_tree_oid(repo_root)
            dirty = git_is_dirty(repo_root)
            patch_digest = None
            if dirty:
                if not spec.allow_dirty_exploratory:
                    raise InvariantViolation(
                        "working tree is dirty; comparable runs require a clean checkout (set allow_dirty_exploratory for an exploratory run)",
                        code="DIRTY_SOURCE",
                        details={"repo_path": str(repo_root)},
                    )
                patch_digest = artifact_digest(git_working_tree_patch(repo_root))
                notes.append("dirty working tree accepted in exploratory mode; run is never promotable")
        snapshot = SourceSnapshot(
            repo_uid=spec.repo_uid,
            target_commit=spec.target_commit,
            tested_commit=tested,
            tested_tree=tested_tree,
            checkout_mode=spec.checkout_mode,
            merge_parent_oids=[],
            dirty=dirty,
            patch_digest=patch_digest,
            source_digest=digest,
            entrypoint=spec.entrypoint,
            implementation_overrides=dict(spec.implementation_overrides),
        )
        return snapshot, notes

    def prepare(self, request: RunRequestSpec, problem: dict[str, Any]) -> PreparedExecution:
        self._check_authorization(request)
        problem = self._check_problem_contract(request, problem)
        resolved = self._resolve(request.source.entrypoint)
        environment = self.check_environment()
        jax = self._load_jax()
        if not isinstance(request.protocol, dict) or "warmup" not in request.protocol or "repetitions" not in request.protocol:
            raise InputError("request.protocol must contain warmup and repetitions", code="INVALID_PROTOCOL")
        if not isinstance(request.verifier, dict):
            raise InputError("request.verifier must be an object", code="INVALID_VERIFIER")
        source, notes = self._source_snapshot(request, resolved)
        try:
            inputs = resolved.input_factory(problem)
        except KernelMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - factory failure is an infrastructure failure, not a verdict
            raise ExecutionInfrastructureError(
                f"input_factory failed: {type(exc).__name__}: {exc}", code="INPUT_FACTORY_FAILED", details={"entrypoint": request.source.entrypoint}
            ) from exc
        args: tuple[Any, ...] = tuple(inputs) if isinstance(inputs, tuple) else (inputs,)
        descriptions = [_leaf_descriptions(a) for a in args]
        suite_hash = jcs_digest(
            {
                "suite": "jax-entrypoint-inputs-v1",
                "entrypoint": request.source.entrypoint,
                "problem": problem,
                "input_factory": f"{getattr(resolved.input_factory, '__module__', '?')}.{getattr(resolved.input_factory, '__qualname__', '?')}",
                "inputs": descriptions,
            }
        )
        overrides = dict(request.source.implementation_overrides)
        fn = resolved.callable
        if overrides:
            base_fn = fn

            def fn(*a: Any, _base: Callable[..., Any] = base_fn, _kw: dict[str, Any] = overrides) -> Any:  # type: ignore[misc]
                return _base(*a, **_kw)

        notes.append(f"timing boundary: {TIMING_BOUNDARY}")
        return PreparedExecution(
            request=request,
            problem=problem,
            source=source,
            environment=environment,
            input_suite_hash=suite_hash,
            handle={
                "adapter_id": self.adapter_id,
                "jax": jax,
                "callable": fn,
                "reference": resolved.reference_callable,
                "required_outputs": list(resolved.required_outputs),
                "inputs": args,
                "compiled": None,
                "compile_error": None,
            },
            notes=notes,
        )

    @staticmethod
    def _handle(prepared: PreparedExecution) -> dict[str, Any]:
        handle = prepared.handle
        if not isinstance(handle, dict) or "jax" not in handle or "callable" not in handle or "inputs" not in handle:
            raise InputError("PreparedExecution was not produced by JaxAdapter.prepare", code="INVALID_PREPARED_EXECUTION")
        return handle

    # ------------------------------------------------------------------ compile
    def compile(self, prepared: PreparedExecution) -> CompileReport:
        handle = self._handle(prepared)
        jax = handle["jax"]
        request_id = prepared.request.request_id
        entrypoint = prepared.source.entrypoint
        try:
            lowered = jax.jit(handle["callable"]).lower(*handle["inputs"])
            compiled = lowered.compile()
        except Exception as exc:  # noqa: BLE001 - every compile failure is evidence, never a crash
            message = f"{type(exc).__name__}: {exc}"
            handle["compiled"] = None
            handle["compile_error"] = message
            blob = _json_blob(
                f"{request_id}-compile-log",
                "compile_log",
                {"kind": "compile_log", "status": "compile_error", "entrypoint": entrypoint, "message": message, "backend": self.backend},
            )
            return CompileReport(status="compile_error", message=message, artifacts=[blob])
        handle["compiled"] = compiled
        handle["compile_error"] = None
        message = f"compiled {entrypoint} for {prepared.environment.get('accelerator_model', self.required_platform)} ({self.backend})"
        blob = _json_blob(
            f"{request_id}-compile-log",
            "compile_log",
            {"kind": "compile_log", "status": "ok", "entrypoint": entrypoint, "message": message, "backend": self.backend},
        )
        return CompileReport(status="ok", message=message, artifacts=[blob])

    def _compiled(self, prepared: PreparedExecution) -> tuple[dict[str, Any], Any, str | None]:
        handle = self._handle(prepared)
        if handle.get("compiled") is None and handle.get("compile_error") is None:
            self.compile(prepared)
        return handle, handle.get("compiled"), handle.get("compile_error")

    # ------------------------------------------------------------------ verify
    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        handle, compiled, compile_error = self._compiled(prepared)
        request = prepared.request
        request_id = request.request_id
        tolerances = request.verifier.get("tolerances") if isinstance(request.verifier, dict) else None
        policy = request.verifier.get("nonfinite_policy") if isinstance(request.verifier, dict) else None
        if compiled is None:
            return CorrectnessReport(status="not_run", message=f"compile_error: {compile_error}")
        reference = handle["reference"]
        if reference is None:
            message = "no trusted reference callable is configured; correctness cannot be established on this backend (recorded, not passed)"
            blob = _json_blob(
                f"{request_id}-correctness",
                "correctness_report",
                {"kind": "correctness_report", "status": "not_run", "message": message, "entrypoint": prepared.source.entrypoint},
            )
            return CorrectnessReport(status="not_run", message=message, artifacts=[blob])
        if not isinstance(tolerances, dict) or "atol" not in tolerances or "rtol" not in tolerances or policy is None:
            raise InputError("request.verifier must contain tolerances.atol, tolerances.rtol and nonfinite_policy", code="INVALID_VERIFIER")
        jax = handle["jax"]
        required = handle["required_outputs"]
        try:
            output = compiled(*handle["inputs"])
            jax.block_until_ready(output)
            expected = reference(*handle["inputs"])
            jax.block_until_ready(expected)
        except Exception as exc:  # noqa: BLE001 - execution error is an error verdict with evidence
            message = f"execution raised {type(exc).__name__}: {exc}"
            blob = _json_blob(
                f"{request_id}-correctness",
                "correctness_report",
                {"kind": "correctness_report", "status": "error", "message": message, "entrypoint": prepared.source.entrypoint},
            )
            return CorrectnessReport(status="error", cases_total=1, cases_passed=0, message=message, artifacts=[blob])
        missing = [k for k in required if not (isinstance(output, dict) and k in output)]
        if missing:
            message = f"MISSING_REQUIRED_OUTPUT: candidate output lacks required outputs {missing}"
            blob = _json_blob(
                f"{request_id}-correctness",
                "correctness_report",
                {
                    "kind": "correctness_report",
                    "status": "fail",
                    "message": message,
                    "required_outputs": required,
                    "present_outputs": sorted(str(k) for k in output) if isinstance(output, dict) else None,
                    "entrypoint": prepared.source.entrypoint,
                },
            )
            return CorrectnessReport(status="fail", cases_total=1, cases_passed=0, message=message, artifacts=[blob])
        structure = verify.verify_pytree_structure(output, expected)
        leaf_results: list[dict[str, Any]] = []
        max_abs: float | None = None
        max_rel: float | None = None
        if structure.matches:
            for path, cand, ref in structure.leaves:
                outcome = verify.elementwise_close(cand, ref, atol=tolerances["atol"], rtol=tolerances["rtol"], nonfinite_policy=policy)
                leaf_results.append({"output": path, **outcome.to_dict()})
                if outcome.max_abs_error is not None:
                    max_abs = outcome.max_abs_error if max_abs is None else max(max_abs, outcome.max_abs_error)
                if outcome.max_rel_error is not None:
                    max_rel = outcome.max_rel_error if max_rel is None else max(max_rel, outcome.max_rel_error)
            passed = all(r["passed"] for r in leaf_results) and bool(leaf_results)
            message = None if passed else "; ".join(str(r["message"]) for r in leaf_results if not r["passed"])
        else:
            passed = False
            message = "; ".join(structure.problems[: verify.MAX_PROBLEMS_IN_MESSAGE])
        status = "pass" if passed else "fail"
        case = {
            "case": "input_factory",
            "passed": passed,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "leaves": leaf_results,
            "message": message,
        }
        blob = ArtifactBlob(
            artifact_id=f"{request_id}-correctness",
            kind="correctness_report",
            media_type="application/json",
            data=verify.correctness_report_bytes(
                status=status,
                cases=[case],
                tolerances={"atol": str(tolerances["atol"]), "rtol": str(tolerances["rtol"])},
                nonfinite_policy=policy,
                reference_id=f"{getattr(reference, '__module__', '?')}.{getattr(reference, '__qualname__', repr(reference))}",
                entrypoint=prepared.source.entrypoint,
                required_outputs=required,
                backend=self.backend,
                input_suite_hash=prepared.input_suite_hash,
            ),
        )
        return CorrectnessReport(
            status=status,
            cases_total=1,
            cases_passed=1 if passed else 0,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            message=message,
            artifacts=[blob],
        )

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        handle, compiled, compile_error = self._compiled(prepared)
        request = prepared.request
        if compiled is None:
            return TimingReport(status="not_run", samples=[], message=f"compile_error: {compile_error}")
        jax = handle["jax"]
        inputs = handle["inputs"]
        protocol = request.protocol
        warmup = protocol["warmup"]
        repetitions = protocol["repetitions"]

        def call() -> Any:
            return compiled(*inputs)

        samples = timing.collect_samples_ns(
            call,
            warmup=warmup,
            repetitions=repetitions,
            block_until_ready=jax.block_until_ready,
            max_wall_seconds=request.max_wall_seconds,
        )
        final = compiled(*inputs)
        jax.block_until_ready(final)
        consumed = _consume_outputs(final)
        artifact = ArtifactBlob(
            artifact_id=f"{request.request_id}-samples",
            kind="latency_samples",
            media_type="application/json",
            data=timing.samples_artifact_bytes(
                samples,
                "nanoseconds",
                timing_method="host_synchronized",
                protocol_id=protocol.get("protocol_id"),
                measurement_scope=protocol.get("measurement_scope"),
                boundary=TIMING_BOUNDARY,
                warmup=warmup,
                repetitions=repetitions,
                entrypoint=prepared.source.entrypoint,
                backend=self.backend,
                accelerator_model=prepared.environment.get("accelerator_model"),
                outputs_consumed=consumed,
                timer=timing.TIMER_NAME,
            ),
        )
        return TimingReport(
            status="recorded",
            unit="nanoseconds",
            samples=[int(s) for s in samples],
            timing_method="host_synchronized",
            message=f"{len(samples)} host-synchronized samples on {self.backend}",
            artifacts=[artifact],
        )


__all__ = [
    "COMPILER_BUILD_FIELD",
    "EXECUTION_FLAG_ENV_VARS",
    "JaxAdapter",
    "ResolvedEntrypoint",
    "TIMING_BOUNDARY",
    "git_head_oid",
    "git_head_tree_oid",
    "git_is_dirty",
]
