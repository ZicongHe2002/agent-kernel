"""Adapter contracts: what a Runner asks of a kernel adapter and what it gets back.

These are concrete dataclasses and protocols (specification section 17). The
Runner (``execution/runner.py``) turns the reports into an immutable Run record;
adapters never write Memory themselves.

Trust model: the Runner, not the adapter or candidate code, captures the actual
tested source (commit, tree, dirty state, source digest) and environment. An
adapter that cannot execute must raise ``BackendUnavailable`` (no hardware,
missing package, no authorization) or ``UnsupportedFormat`` (unknown artifact
format) — never fall back silently to another backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..domain.errors import BackendUnavailable, InputError
from ..domain.hashing import jcs_digest
from ..domain.models import ArtifactRef, GitOid, to_json


# --------------------------------------------------------------------------------------
# Request side
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSpec:
    """Where the candidate code is and how it must be checked out and invoked."""

    repo_uid: str
    target_commit: GitOid
    entrypoint: str  # e.g. "package.module:function" for Python entrypoints
    checkout_mode: str = "exact_commit"  # exact_commit | integration_merge
    implementation_overrides: dict = field(default_factory=dict)
    repo_path: str | None = None  # local repository path for a controlled checkout; None = in-process demo
    allow_dirty_exploratory: bool = False


@dataclass(frozen=True)
class RunRequestSpec:
    """One user/system intent to execute a candidate. Persisted in the request ledger."""

    request_id: str
    idempotency_key: str
    subject_ref: str
    config_ref: str
    backend: str  # "mock" | "cpu" | "jax_tpu" | ...
    stage: str  # compile | verify | benchmark | profile
    protocol: dict  # Protocol fields without protocol_hash
    verifier: dict  # Verifier fields without verifier_hash
    source: SourceSpec
    authorization: dict = field(default_factory=dict)
    session_id: str | None = None
    pair_id: str | None = None
    role_in_pair: str | None = None
    max_wall_seconds: float = 600.0
    rerun_of: str | None = None
    metadata: dict = field(default_factory=dict)

    def spec_hash(self) -> str:
        """Canonical hash of the request content (excluding request_id) for idempotency checks."""
        data = to_json(self)
        data.pop("request_id", None)
        return jcs_digest(data)

    def to_dict(self) -> dict[str, Any]:
        return to_json(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRequestSpec":
        if not isinstance(data, dict):
            raise InputError("request spec must be an object")
        src = dict(data["source"])
        src["target_commit"] = GitOid(**src["target_commit"])
        fields = dict(data)
        fields["source"] = SourceSpec(**src)
        return cls(**fields)


# --------------------------------------------------------------------------------------
# Evidence produced by the Runner/adapter
# --------------------------------------------------------------------------------------
@dataclass
class ArtifactBlob:
    """Evidence bytes produced during execution; the Runner stores them content-addressed."""

    artifact_id: str
    kind: str  # latency_samples | correctness_report | compile_log | profile | analysis_report | diff | ...
    media_type: str
    data: bytes
    retention: str = "permanent"

    def size(self) -> int:
        return len(self.data)


@dataclass
class SourceSnapshot:
    """What was actually tested, captured by the trusted Runner before execution."""

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


@dataclass
class PreparedExecution:
    request: RunRequestSpec
    problem: dict  # normalized problem of the config
    source: SourceSnapshot
    environment: dict  # Environment fields without environment_hash
    input_suite_hash: str
    handle: Any = None  # adapter-private state (callables, inputs, compiled objects)
    notes: list[str] = field(default_factory=list)


@dataclass
class CompileReport:
    status: str  # ok | compile_error | unsupported
    message: str | None = None
    artifacts: list[ArtifactBlob] = field(default_factory=list)


@dataclass
class CorrectnessReport:
    status: str  # pass | fail | error | not_run
    cases_total: int = 0
    cases_passed: int = 0
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    message: str | None = None
    artifacts: list[ArtifactBlob] = field(default_factory=list)


@dataclass
class TimingReport:
    status: str  # recorded | error | not_run
    unit: str = "nanoseconds"
    samples: list[float] = field(default_factory=list)
    timing_method: str = "host_synchronized"
    message: str | None = None
    artifacts: list[ArtifactBlob] = field(default_factory=list)


@dataclass
class AnalysisReport:
    metrics: list[dict] = field(default_factory=list)  # Metric dicts (see record schema)
    conclusion: str | None = None
    artifacts: list[ArtifactBlob] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------
@runtime_checkable
class KernelAdapter(Protocol):
    adapter_id: str
    backend: str

    def check_environment(self) -> dict:
        """Return the Environment snapshot (without hash) or raise BackendUnavailable."""
        ...

    def prepare(self, request: RunRequestSpec, problem: dict) -> PreparedExecution: ...

    def compile(self, prepared: PreparedExecution) -> CompileReport: ...

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport: ...

    def benchmark(self, prepared: PreparedExecution) -> TimingReport: ...


@runtime_checkable
class AnalysisAdapter(Protocol):
    adapter_id: str
    parser_version: str

    def accepts(self, artifact_manifest: dict) -> bool: ...

    def parse(self, artifacts: list[tuple[ArtifactRef, bytes]]) -> AnalysisReport: ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._kernel: dict[str, Any] = {}
        self._analysis: dict[str, Any] = {}

    def register_kernel_adapter(self, adapter: Any) -> None:
        backend = getattr(adapter, "backend", None)
        if not backend:
            raise InputError("kernel adapter must define a backend name")
        self._kernel[backend] = adapter

    def register_analysis_adapter(self, adapter: Any) -> None:
        adapter_id = getattr(adapter, "adapter_id", None)
        if not adapter_id:
            raise InputError("analysis adapter must define adapter_id")
        self._analysis[adapter_id] = adapter

    def kernel_adapter(self, backend: str) -> Any:
        try:
            return self._kernel[backend]
        except KeyError as exc:
            raise BackendUnavailable(
                f"no kernel adapter registered for backend {backend!r}",
                details={"backend": backend, "available": sorted(self._kernel)},
            ) from exc

    def analysis_adapters(self) -> list[Any]:
        return [self._analysis[k] for k in sorted(self._analysis)]

    def backends(self) -> list[str]:
        return sorted(self._kernel)
