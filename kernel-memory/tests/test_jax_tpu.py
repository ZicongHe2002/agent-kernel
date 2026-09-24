"""Tests for the controlled JAX/TPU adapter interface (spec sections 10, 11, 17; T29).

Two families:

* **Unit paths with an injected fake ``jax``** (``jax_loader``): the honesty gates — no TPU
  device -> ``BackendUnavailable`` (never a CPU fallback labelled ``tpu``), authorization
  before any device probe, refusal to guess an entrypoint, incomplete ``mla_forward`` contract,
  never-invented compiler build ids, and execution-flag *names* only.
* **Real-jax paths on CPU** (``JaxAdapter(required_platform="cpu", backend="jax_cpu_interface_test")``):
  the pytree / compile / verify / timing code path on the CPU backend, whose environment is
  labelled by the constructor and never ``tpu``.

Exactly one test is marked ``integration`` (T29): on this CPU-only host the TPU adapter must
refuse with ``actual_backend == "cpu"``.  It executes here; it is never skipped.
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kernel_memory.adapters import jax_tpu
from kernel_memory.adapters.base import PreparedExecution, RunRequestSpec, SourceSpec
from kernel_memory.adapters.jax_tpu import COMPILER_BUILD_FIELD, TIMING_BOUNDARY, JaxAdapter, ResolvedEntrypoint
from kernel_memory.domain.errors import (
    AuthorizationError,
    BackendUnavailable,
    ExecutionInfrastructureError,
    IncompleteProblemContract,
    InputError,
    PrerequisiteMissingError,
)
from kernel_memory.domain.models import GitOid

REPO_UID = "github:github.com:repo:900002"
ENTRYPOINT = "demo.kernels:attention_forward"
TARGET = GitOid("sha256", "cd" * 32)
PROBLEM: dict[str, Any] = {"kernel_id": "interface_test_add", "batch": 4, "dim": 16, "seed": 7}
REPETITIONS = 7
WARMUP = 1
CPU_LABEL = "jax_cpu_interface_test"
HEX64 = re.compile(r"^sha256:[0-9a-f]{64}$")  # digests are prefixed with their algorithm


# ------------------------------------------------------------------------------ request helpers
def make_spec(
    request_id: str,
    *,
    entrypoint: str = ENTRYPOINT,
    backend: str = "jax_tpu",
    authorization: dict | None = None,
    protocol: dict | None = None,
    verifier: dict | None = None,
    metadata: dict | None = None,
    overrides: dict | None = None,
    repetitions: int = REPETITIONS,
    warmup: int = WARMUP,
    max_wall_seconds: float = 600.0,
    stage: str = "benchmark",
) -> RunRequestSpec:
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=f"key-{request_id}",
        subject_ref="baseline-jax-source",
        config_ref="cfg-jax",
        backend=backend,
        stage=stage,
        # passed through uncoerced so malformed (non-dict) values reach the adapter's own checks
        protocol=protocol
        if protocol is not None
        else {"warmup": warmup, "repetitions": repetitions, "protocol_id": "jax-interface-test-v1", "measurement_scope": "kernel_only"},
        verifier=verifier if verifier is not None else {"tolerances": {"atol": "1e-4", "rtol": "1e-4"}, "nonfinite_policy": "reject_unexpected"},
        source=SourceSpec(
            repo_uid=REPO_UID,
            target_commit=TARGET,
            entrypoint=entrypoint,
            checkout_mode="exact_commit",
            implementation_overrides=dict(overrides or {}),
        ),
        authorization=dict(authorization or {}),
        max_wall_seconds=max_wall_seconds,
        metadata=dict(metadata or {}),
    )


AUTHORIZED = {"allow_tpu_execution": True}


class RecordingResolver:
    """An entrypoint resolver that records every call and returns a fixed value (or raises)."""

    def __init__(self, resolved: Any = None, *, raises: BaseException | None = None) -> None:
        self.resolved = resolved
        self.raises = raises
        self.calls: list[str] = []

    def __call__(self, entrypoint: str) -> Any:
        self.calls.append(entrypoint)
        if self.raises is not None:
            raise self.raises
        return self.resolved


def blob_json(blob: Any) -> dict[str, Any]:
    assert blob.media_type == "application/json"
    return json.loads(blob.data.decode("utf-8"))


# ------------------------------------------------------------------------------ kernels under test
def good_kernel(x: Any, y: Any) -> dict[str, Any]:
    return {"o": x + y, "lse": (x * y).sum(axis=-1)}


def kernel_missing_lse(x: Any, y: Any) -> dict[str, Any]:
    return {"o": x + y}


def wrong_kernel(x: Any, y: Any) -> dict[str, Any]:
    return {"o": x + y + 0.01, "lse": (x * y).sum(axis=-1)}


def scaled_kernel(x: Any, y: Any, scale: float = 1.0) -> dict[str, Any]:
    return {"o": (x + y) * scale, "lse": (x * y).sum(axis=-1)}


def untraceable_kernel(x: Any, y: Any) -> dict[str, Any]:
    return {"o": x @ y, "lse": (x * y).sum(axis=-1)}  # (4,16)@(4,16) cannot be traced -> compile_error


def numpy_reference(x: Any, y: Any) -> dict[str, Any]:
    xn = np.asarray(x, dtype=np.float32)
    yn = np.asarray(y, dtype=np.float32)
    return {"o": xn + yn, "lse": (xn.astype(np.float64) * yn.astype(np.float64)).sum(axis=-1).astype(np.float32)}


def numpy_inputs(problem: dict[str, Any]) -> tuple[Any, Any]:
    rng = np.random.default_rng(problem["seed"])
    shape = (problem["batch"], problem["dim"])
    return rng.standard_normal(shape, dtype=np.float32), rng.standard_normal(shape, dtype=np.float32)


def jax_inputs(problem: dict[str, Any]) -> tuple[Any, Any]:
    import jax.numpy as jnp  # the real-jax fixture has imported jax already

    x, y = numpy_inputs(problem)
    return jnp.asarray(x), jnp.asarray(y)


def resolved(fn: Any, *, reference: Any = numpy_reference, inputs: Any = numpy_inputs, required: list[str] | None = None, **kw: Any) -> ResolvedEntrypoint:
    return ResolvedEntrypoint(
        callable=fn,
        input_factory=inputs,
        reference_callable=reference,
        required_outputs=["o", "lse"] if required is None else required,
        **kw,
    )


# ------------------------------------------------------------------------------ fake jax
class FakeDevice:
    def __init__(self, platform: str, device_kind: str, index: int = 0) -> None:
        self.platform = platform
        self.device_kind = device_kind
        self.id = index

    def __str__(self) -> str:
        return f"{self.platform}:{self.id}"


class FakeJax:
    """The subset of the jax module surface that JaxAdapter touches, with probe counters."""

    __version__ = "0.0.0-fake"

    def __init__(self, backend: str = "cpu", devices: list[FakeDevice] | None = None, platform_version: str | None = None) -> None:
        self._backend = backend
        self._devices = list(devices) if devices is not None else [FakeDevice("cpu", "cpu")]
        self.devices_calls = 0
        self.default_backend_calls = 0
        self.jit_calls = 0
        if platform_version is not None:
            client = SimpleNamespace(platform_version=platform_version)
            self.extend = SimpleNamespace(backend=SimpleNamespace(get_backend=lambda: client))

    def devices(self) -> list[FakeDevice]:
        self.devices_calls += 1
        return list(self._devices)

    def default_backend(self) -> str:
        self.default_backend_calls += 1
        return self._backend

    def jit(self, fn: Any) -> Any:
        self.jit_calls += 1
        return SimpleNamespace(lower=lambda *args: SimpleNamespace(compile=lambda: fn))

    @staticmethod
    def block_until_ready(tree: Any) -> Any:
        return tree


def fake_cpu() -> FakeJax:
    return FakeJax("cpu", [FakeDevice("cpu", "cpu")])


def fake_tpu(count: int = 1, **kw: Any) -> FakeJax:
    return FakeJax("tpu", [FakeDevice("tpu", "TPU v5 lite", i) for i in range(count)], **kw)


def tpu_adapter(jax: FakeJax, resolver: Any = None) -> JaxAdapter:
    return JaxAdapter(jax_loader=lambda: jax, entrypoint_resolver=resolver)


# ------------------------------------------------------------------------------ constructor
def test_defaults_are_tpu_labelled_and_non_tpu_instances_may_not_reuse_the_label() -> None:
    adapter = JaxAdapter()
    assert (adapter.required_platform, adapter.backend, adapter.adapter_id) == ("tpu", "jax_tpu", "jax-tpu-v0")
    with pytest.raises(InputError) as excinfo:
        JaxAdapter(required_platform="cpu", backend="jax_tpu")
    assert excinfo.value.code == "INVALID_ADAPTER_CONFIG" and excinfo.value.exit_code == 2
    for bad in ({"required_platform": ""}, {"backend": ""}, {"required_platform": None}):
        with pytest.raises(InputError):
            JaxAdapter(**bad)  # type: ignore[arg-type]
    cpu = JaxAdapter(required_platform="cpu", backend=CPU_LABEL)
    assert cpu.backend == CPU_LABEL and cpu.required_platform == "cpu"


# ------------------------------------------------------------------------------ environment (fake jax)
@pytest.mark.parametrize(
    "jax",
    [
        pytest.param(fake_cpu(), id="cpu-only"),
        pytest.param(FakeJax("cpu", [FakeDevice("cpu", "cpu"), FakeDevice("tpu", "TPU v5 lite")]), id="tpu-device-but-cpu-default-backend"),
        pytest.param(FakeJax("tpu", [FakeDevice("cpu", "cpu")]), id="tpu-backend-but-no-tpu-device"),
        pytest.param(FakeJax("cpu", []), id="no-devices"),
    ],
)
def test_t29_no_tpu_is_backend_unavailable_never_a_cpu_fallback_labelled_tpu(jax: FakeJax) -> None:
    adapter = tpu_adapter(jax)
    with pytest.raises(BackendUnavailable) as excinfo:
        adapter.check_environment()
    err = excinfo.value
    assert err.exit_code == 5
    assert err.details["required"] == "tpu"
    assert err.details["actual_backend"] == jax._backend
    assert err.details["devices"] == [str(d) for d in jax._devices]
    assert err.details["backend"] == "jax_tpu"
    assert "refusing to fall back" in err.message
    assert jax.devices_calls == 1  # the real device list was consulted, not assumed


def test_backend_initialisation_failure_is_backend_unavailable_not_a_crash() -> None:
    class Exploding(FakeJax):
        def devices(self) -> list[FakeDevice]:
            raise RuntimeError("libtpu could not be initialised")

    with pytest.raises(BackendUnavailable) as excinfo:
        tpu_adapter(Exploding()).check_environment()
    assert "RuntimeError: libtpu could not be initialised" in excinfo.value.message
    assert excinfo.value.details == {"required": "tpu", "backend": "jax_tpu"}


def test_jax_not_installed_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "jax":
            raise ImportError("No module named 'jax'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(jax_tpu.importlib, "import_module", fake_import)
    with pytest.raises(BackendUnavailable) as excinfo:
        JaxAdapter().check_environment()  # no jax_loader: the real import path
    assert excinfo.value.message == "jax not installed"
    assert excinfo.value.details["required"] == "tpu" and "import_error" in excinfo.value.details


def test_fake_tpu_environment_reports_devices_and_never_invents_a_compiler_build_id() -> None:
    env = tpu_adapter(fake_tpu(count=2)).check_environment()
    assert env["backend"] == "jax_tpu"
    assert env["accelerator_model"] == "TPU v5 lite"
    assert env["device_count"] == 2 and env["topology"] == "2xTPU v5 lite"
    assert env["software"]["jax"] == "0.0.0-fake"
    assert isinstance(env["software"]["python"], str) and isinstance(env["software"]["jaxlib"], str)
    assert COMPILER_BUILD_FIELD not in env["software"]
    assert env["unknown_required_fields"] == [COMPILER_BUILD_FIELD]  # unknown is recorded, never guessed
    assert env["host_timer_environment"]["timer"]


def test_fake_tpu_environment_records_pjrt_platform_version_as_compiler_build_id() -> None:
    env = tpu_adapter(fake_tpu(platform_version="  libtpu-2026.01.01 \n")).check_environment()
    assert env["software"][COMPILER_BUILD_FIELD] == "libtpu-2026.01.01"
    assert env["unknown_required_fields"] == []


def test_execution_flags_record_names_only_never_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_tpu_secret_flag=hunter2")
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    env = tpu_adapter(fake_tpu()).check_environment()
    assert env["execution_flags"] == {"XLA_FLAGS": "set", "JAX_PLATFORMS": "set"}
    assert "hunter2" not in json.dumps(env)


# ------------------------------------------------------------------------------ prepare gates (fake jax)
@pytest.mark.parametrize("authorization", [None, {}, {"allow_tpu_execution": False}, {"allow_tpu_execution": "true"}, {"allow_tpu_execution": 1}, {"other": True}])
def test_prepare_without_allow_tpu_execution_is_authorization_error_before_any_device_probe(authorization: dict | None) -> None:
    jax = fake_tpu()
    resolver = RecordingResolver(resolved(good_kernel))
    adapter = tpu_adapter(jax, resolver)
    with pytest.raises(AuthorizationError) as excinfo:
        adapter.prepare(make_spec("req-unauth", authorization=authorization), PROBLEM)
    err = excinfo.value
    assert err.exit_code == 7
    assert err.details["permission"] == "allow_tpu_execution" and err.details["request_id"] == "req-unauth"
    assert jax.devices_calls == 0 and jax.default_backend_calls == 0  # refused before touching the device
    assert resolver.calls == []  # ...and before resolving any code


def test_prepare_authorized_but_no_tpu_is_backend_unavailable_after_the_gates() -> None:
    jax = fake_cpu()
    resolver = RecordingResolver(resolved(good_kernel))
    with pytest.raises(BackendUnavailable) as excinfo:
        tpu_adapter(jax, resolver).prepare(make_spec("req-auth-cpu", authorization=AUTHORIZED), PROBLEM)
    assert excinfo.value.details["actual_backend"] == "cpu"
    assert resolver.calls == [ENTRYPOINT]  # authorization and resolution passed; the device probe refused
    assert jax.devices_calls == 1


@pytest.mark.parametrize(
    "resolver",
    [
        pytest.param(None, id="no-resolver-configured"),
        pytest.param(RecordingResolver(None), id="resolver-returns-none"),
        pytest.param(RecordingResolver(("not", "a", "ResolvedEntrypoint")), id="resolver-returns-wrong-type"),
        pytest.param(RecordingResolver(ResolvedEntrypoint(callable="demo.kernels:attention_forward", input_factory=numpy_inputs)), id="non-callable"),  # type: ignore[arg-type]
        pytest.param(RecordingResolver(raises=LookupError("no such entrypoint")), id="resolver-raises-lookup"),
        pytest.param(RecordingResolver(raises=ImportError("demo.kernels")), id="resolver-raises-import"),
        pytest.param(RecordingResolver(raises=InputError("bad entrypoint syntax", code="INVALID_ENTRYPOINT")), id="resolver-raises-kernel-memory-error"),
    ],
)
def test_unresolvable_entrypoint_is_prerequisite_missing_entrypoint_not_configured(resolver: Any) -> None:
    jax = fake_tpu()
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        tpu_adapter(jax, resolver).prepare(make_spec("req-noep", authorization=AUTHORIZED), PROBLEM)
    err = excinfo.value
    assert type(err) is PrerequisiteMissingError  # not BackendUnavailable: the backend was never the problem
    assert err.code == "ENTRYPOINT_NOT_CONFIGURED" and err.exit_code == 5
    assert err.details["entrypoint"] == ENTRYPOINT
    assert jax.devices_calls == 0  # refusing to guess happens before any device probe
    if resolver is not None:
        assert resolver.calls == [ENTRYPOINT]


def test_resolver_raising_kernel_memory_error_keeps_the_cause() -> None:
    resolver = RecordingResolver(raises=InputError("bad entrypoint syntax", code="INVALID_ENTRYPOINT"))
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        tpu_adapter(fake_tpu(), resolver).prepare(make_spec("req-cause", authorization=AUTHORIZED), PROBLEM)
    assert excinfo.value.details["cause"]["error"] == "INVALID_ENTRYPOINT"


@pytest.mark.parametrize(
    ("problem", "metadata"),
    [
        pytest.param({"kernel_id": "mla_forward", "batch": 1, "seq_len": 128}, None, id="kernel_id-in-problem"),
        pytest.param({"batch": 1, "seq_len": 128}, {"kernel_id": "mla_forward"}, id="kernel_id-in-request-metadata"),
    ],
)
def test_mla_forward_problem_is_incomplete_problem_contract(problem: dict[str, Any], metadata: dict | None) -> None:
    jax = fake_tpu()
    resolver = RecordingResolver(resolved(good_kernel))
    with pytest.raises(IncompleteProblemContract) as excinfo:
        tpu_adapter(jax, resolver).prepare(make_spec("req-mla", authorization=AUTHORIZED, metadata=metadata), problem)
    assert excinfo.value.exit_code == 5
    assert "mla_forward" in excinfo.value.message
    assert resolver.calls == [] and jax.devices_calls == 0  # contract gate precedes resolution and devices


def test_incomplete_contract_raised_by_the_resolver_propagates_unwrapped() -> None:
    resolver = RecordingResolver(raises=IncompleteProblemContract("resolver: contract incomplete"))
    with pytest.raises(IncompleteProblemContract):
        tpu_adapter(fake_tpu(), resolver).prepare(make_spec("req-mla-resolver", authorization=AUTHORIZED), PROBLEM)


@pytest.mark.parametrize("problem", [None, [1, 2], "batch=4"])
def test_non_object_problem_is_input_error(problem: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel))).prepare(make_spec("req-notobj", authorization=AUTHORIZED), problem)
    assert excinfo.value.code == "INVALID_PROBLEM" and excinfo.value.exit_code == 2


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("protocol", {"warmup": 1}, "INVALID_PROTOCOL"),
        ("protocol", {"repetitions": 3}, "INVALID_PROTOCOL"),
        ("protocol", [], "INVALID_PROTOCOL"),
        ("verifier", [], "INVALID_VERIFIER"),
    ],
)
def test_prepare_validates_protocol_and_verifier_shapes(field: str, value: Any, code: str) -> None:
    spec = make_spec("req-badreq", authorization=AUTHORIZED, **{field: value})
    with pytest.raises(InputError) as excinfo:
        tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel))).prepare(spec, PROBLEM)
    assert excinfo.value.code == code


def test_input_factory_failure_is_infrastructure_error_not_a_verdict() -> None:
    def broken_inputs(problem: dict[str, Any]) -> Any:
        raise ValueError("cannot allocate inputs")

    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel, inputs=broken_inputs))).prepare(
            make_spec("req-factory", authorization=AUTHORIZED), PROBLEM
        )
    assert excinfo.value.code == "INPUT_FACTORY_FAILED" and excinfo.value.exit_code == 6


def test_entrypoint_without_discoverable_source_files_is_prerequisite_missing() -> None:
    # Builtins have no source file; without ResolvedEntrypoint.source_files the tested source is unknown -> refuse.
    jax = fake_tpu()
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        tpu_adapter(jax, RecordingResolver(resolved(len, inputs=dict, reference=None))).prepare(make_spec("req-nosrc", authorization=AUTHORIZED), PROBLEM)
    assert excinfo.value.code == "SOURCE_MANIFEST_UNAVAILABLE" and excinfo.value.exit_code == 5
    assert excinfo.value.details["entrypoint"] == ENTRYPOINT
    assert jax.devices_calls == 1  # the device probe precedes the source snapshot; the refusal is about source, not devices


def test_declared_source_file_that_does_not_exist_is_prerequisite_missing(tmp_path: Any) -> None:
    missing = tmp_path / "kernel_gone.py"
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel, source_files=[str(missing)]))).prepare(
            make_spec("req-src-missing", authorization=AUTHORIZED), PROBLEM
        )
    assert excinfo.value.code == "SOURCE_MANIFEST_UNAVAILABLE" and excinfo.value.exit_code == 5
    assert excinfo.value.details["file"] == str(missing)


def test_prepare_on_fake_tpu_with_authorization_captures_source_environment_and_suite() -> None:
    jax = fake_tpu()
    resolver = RecordingResolver(resolved(good_kernel))
    adapter = tpu_adapter(jax, resolver)
    prepared = adapter.prepare(make_spec("req-fake-ok", authorization=AUTHORIZED), PROBLEM)
    assert prepared.environment["backend"] == "jax_tpu" and prepared.environment["accelerator_model"] == "TPU v5 lite"
    assert prepared.source.tested_commit == TARGET and prepared.source.target_commit == TARGET
    assert prepared.source.dirty is False and prepared.source.patch_digest is None and prepared.source.tested_tree is None
    assert HEX64.match(prepared.source.source_digest)
    assert prepared.source.entrypoint == ENTRYPOINT and prepared.source.checkout_mode == "exact_commit"
    assert HEX64.match(prepared.input_suite_hash)
    assert prepared.problem == PROBLEM and prepared.problem is not PROBLEM
    assert prepared.handle["adapter_id"] == "jax-tpu-v0" and prepared.handle["required_outputs"] == ["o", "lse"]
    assert len(prepared.handle["inputs"]) == 2
    assert any("in-process entrypoint without repo_path" in n for n in prepared.notes)
    assert any(TIMING_BOUNDARY in n for n in prepared.notes)
    # the same request prepared again yields identical evidence hashes (deterministic inputs)
    again = adapter.prepare(make_spec("req-fake-ok", authorization=AUTHORIZED), PROBLEM)
    assert again.input_suite_hash == prepared.input_suite_hash and again.source.source_digest == prepared.source.source_digest
    # compile/verify through the fake jit run the numpy kernel: pass with zero error
    assert adapter.compile(prepared).status == "ok" and jax.jit_calls == 1
    report = adapter.verify(prepared)
    assert report.status == "pass" and (report.cases_total, report.cases_passed) == (1, 1)
    assert report.max_abs_error is not None and report.max_abs_error < 1e-6  # float32 sum vs float64 reference sum


def test_source_digest_changes_with_the_declared_source_files(tmp_path: Any) -> None:
    other = tmp_path / "kernel_variant.py"
    other.write_text("def attention_forward(x, y):\n    return {'o': x + y}\n")
    adapter = tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel)))
    default = adapter.prepare(make_spec("req-src-a", authorization=AUTHORIZED), PROBLEM)
    explicit_same = tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel, source_files=[__file__]))).prepare(
        make_spec("req-src-b", authorization=AUTHORIZED), PROBLEM
    )
    explicit_other = tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel, source_files=[str(other)]))).prepare(
        make_spec("req-src-c", authorization=AUTHORIZED), PROBLEM
    )
    assert inspect.getsourcefile(good_kernel) == __file__
    assert default.source.source_digest == explicit_same.source.source_digest  # discovery found this module
    assert explicit_other.source.source_digest != default.source.source_digest


def test_stages_refuse_a_prepared_execution_from_another_adapter() -> None:
    adapter = tpu_adapter(fake_tpu(), RecordingResolver(resolved(good_kernel)))
    prepared = adapter.prepare(make_spec("req-foreign", authorization=AUTHORIZED), PROBLEM)
    prepared.handle = {"adapter_id": "cpu-demo-v1", "callable": good_kernel}
    for stage in (adapter.compile, adapter.verify, adapter.benchmark):
        with pytest.raises(InputError) as excinfo:
            stage(prepared)
        assert excinfo.value.code == "INVALID_PREPARED_EXECUTION"


# ------------------------------------------------------------------------------ real jax on CPU
@pytest.fixture(scope="module")
def real_jax() -> Any:
    import jax  # imported once for the module; this host is CPU-only

    return jax


def cpu_adapter(resolver: Any) -> JaxAdapter:
    return JaxAdapter(required_platform="cpu", backend=CPU_LABEL, adapter_id="jax-cpu-interface-test", entrypoint_resolver=resolver)


def cpu_spec(request_id: str, **kw: Any) -> RunRequestSpec:
    kw.setdefault("backend", CPU_LABEL)
    return make_spec(request_id, **kw)


def test_cpu_interface_environment_is_labelled_by_the_constructor_and_never_tpu(real_jax: Any) -> None:
    env = cpu_adapter(None).check_environment()
    assert env["backend"] == CPU_LABEL
    assert env["accelerator_model"] == "cpu" and env["topology"] == f"{env['device_count']}xcpu"
    assert env["device_count"] == len([d for d in real_jax.devices() if d.platform == "cpu"]) >= 1
    for key in ("backend", "accelerator_model", "topology"):
        assert "tpu" not in str(env[key]).lower()
    assert env["software"]["jax"] == real_jax.__version__
    # the compiler build id is either the PJRT-reported string or explicitly unknown, never both/neither
    assert (COMPILER_BUILD_FIELD in env["software"]) != (COMPILER_BUILD_FIELD in env["unknown_required_fields"])


def test_cpu_interface_full_pipeline_compile_verify_benchmark(real_jax: Any) -> None:
    resolver = RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs))
    adapter = cpu_adapter(resolver)
    prepared = adapter.prepare(cpu_spec("req-cpu-pipeline"), PROBLEM)
    assert prepared.environment["backend"] == CPU_LABEL and "tpu" not in prepared.environment["accelerator_model"]
    assert prepared.source.tested_commit == TARGET and HEX64.match(prepared.source.source_digest)
    # the jitted callable unwraps to this module, so discovery and explicit declaration agree
    explicit = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs, source_files=[__file__]))).prepare(
        cpu_spec("req-cpu-pipeline-explicit"), PROBLEM
    )
    assert explicit.source.source_digest == prepared.source.source_digest

    compiled = adapter.compile(prepared)
    assert compiled.status == "ok" and CPU_LABEL in (compiled.message or "")
    log = compiled.artifacts[0]
    assert log.kind == "compile_log" and log.artifact_id == "req-cpu-pipeline-compile-log"
    assert blob_json(log) == {"kind": "compile_log", "status": "ok", "entrypoint": ENTRYPOINT, "message": compiled.message, "backend": CPU_LABEL, "is_fixture": False}

    report = adapter.verify(prepared)
    assert report.status == "pass" and (report.cases_total, report.cases_passed) == (1, 1)
    assert report.max_abs_error is not None and 0.0 <= report.max_abs_error < 1e-5
    assert report.message is None
    doc = blob_json(report.artifacts[0])
    assert report.artifacts[0].kind == "correctness_report" and report.artifacts[0].artifact_id == "req-cpu-pipeline-correctness"
    assert doc["status"] == "pass" and doc["cases_total"] == 1 and doc["cases_passed"] == 1
    assert doc["required_outputs"] == ["o", "lse"] and doc["backend"] == CPU_LABEL and doc["is_fixture"] is False
    assert doc["tolerances"] == {"atol": "1e-4", "rtol": "1e-4"} and doc["nonfinite_policy"] == "reject_unexpected"
    assert doc["input_suite_hash"] == prepared.input_suite_hash and doc["entrypoint"] == ENTRYPOINT
    assert "numpy_reference" in doc["reference_id"]
    assert {leaf["output"] for leaf in doc["cases"][0]["leaves"]} and len(doc["cases"][0]["leaves"]) == 2

    timing_report = adapter.benchmark(prepared)
    assert timing_report.status == "recorded" and timing_report.unit == "nanoseconds"
    assert len(timing_report.samples) == REPETITIONS
    assert all(isinstance(s, int) and s >= 0 for s in timing_report.samples)
    assert timing_report.timing_method == "host_synchronized"
    samples_doc = blob_json(timing_report.artifacts[0])
    assert timing_report.artifacts[0].kind == "latency_samples" and timing_report.artifacts[0].artifact_id == "req-cpu-pipeline-samples"
    assert samples_doc["samples"] == timing_report.samples and samples_doc["count"] == REPETITIONS
    assert samples_doc["unit"] == "nanoseconds" and samples_doc["backend"] == CPU_LABEL and samples_doc["accelerator_model"] == "cpu"
    assert samples_doc["warmup"] == WARMUP and samples_doc["repetitions"] == REPETITIONS
    assert samples_doc["protocol_id"] == "jax-interface-test-v1" and samples_doc["measurement_scope"] == "kernel_only"
    assert samples_doc["boundary"] == TIMING_BOUNDARY and samples_doc["is_fixture"] is False
    consumed = samples_doc["outputs_consumed"]
    assert sorted(c["size"] for c in consumed) == [PROBLEM["batch"], PROBLEM["batch"] * PROBLEM["dim"]]  # both outputs materialised
    assert all(isinstance(c["checksum"], float) for c in consumed)
    assert {c["path"] for c in consumed} == {"o", "lse"}


@pytest.mark.parametrize("repetitions", [1, 3, 12])
def test_benchmark_returns_exactly_the_requested_number_of_samples(real_jax: Any, repetitions: int) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec(f"req-cpu-reps-{repetitions}", repetitions=repetitions, warmup=0), PROBLEM)
    report = adapter.benchmark(prepared)  # benchmark compiles on demand
    assert report.status == "recorded" and len(report.samples) == repetitions
    assert prepared.handle["compiled"] is not None


def test_verify_missing_required_output_fails_naming_missing_required_output(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(kernel_missing_lse), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-missing"), PROBLEM)
    assert adapter.compile(prepared).status == "ok"  # it compiles and runs...
    report = adapter.verify(prepared)
    assert report.status == "fail"  # ...but the contract is not met: a verdict, never a pass
    assert (report.cases_total, report.cases_passed) == (1, 0)
    assert report.message is not None and report.message.startswith("MISSING_REQUIRED_OUTPUT") and "'lse'" in report.message
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "fail" and doc["required_outputs"] == ["o", "lse"] and doc["present_outputs"] == ["o"]
    assert doc["message"] == report.message and doc["is_fixture"] is False


def test_verify_wrong_kernel_is_fail_with_measured_error(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(wrong_kernel), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-wrong"), PROBLEM)
    report = adapter.verify(prepared)
    assert report.status == "fail" and report.cases_passed == 0
    assert report.max_abs_error is not None and 5e-3 < report.max_abs_error < 2e-2  # the injected 0.01 offset (float32)
    assert report.message  # names the failing leaf
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "fail" and doc["max_abs_error"] == report.max_abs_error
    leaves = {leaf["output"]: leaf["passed"] for leaf in doc["cases"][0]["leaves"]}
    assert list(leaves.values()).count(False) == 1  # only "o" is wrong, "lse" still agrees


def test_verify_honours_request_tolerances(real_jax: Any) -> None:
    verifier = {"tolerances": {"atol": "0.05", "rtol": "0"}, "nonfinite_policy": "reject_unexpected"}
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(wrong_kernel), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-loose", verifier=verifier), PROBLEM)
    report = adapter.verify(prepared)
    assert report.status == "pass" and report.max_abs_error is not None and report.max_abs_error > 5e-3
    assert blob_json(report.artifacts[0])["tolerances"] == {"atol": "0.05", "rtol": "0"}


def test_verify_without_trusted_reference_is_not_run_never_pass(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs, reference=None)))
    prepared = adapter.prepare(cpu_spec("req-cpu-noref"), PROBLEM)
    report = adapter.verify(prepared)
    assert report.status == "not_run" and (report.cases_total, report.cases_passed) == (0, 0)
    assert report.max_abs_error is None and "no trusted reference" in (report.message or "")
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "not_run" and doc["kind"] == "correctness_report"


def test_verify_with_incomplete_verifier_is_input_error(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-badverifier", verifier={"tolerances": {"atol": "1e-4"}}), PROBLEM)
    with pytest.raises(InputError) as excinfo:
        adapter.verify(prepared)
    assert excinfo.value.code == "INVALID_VERIFIER"


def test_compile_error_is_a_report_and_downstream_stages_are_not_run(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(untraceable_kernel, inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-compile-error"), PROBLEM)
    compiled = adapter.compile(prepared)
    assert compiled.status == "compile_error" and compiled.message and "TypeError" in compiled.message
    log = blob_json(compiled.artifacts[0])
    assert log["status"] == "compile_error" and log["message"] == compiled.message and log["backend"] == CPU_LABEL
    assert prepared.handle["compiled"] is None and prepared.handle["compile_error"] == compiled.message
    correctness = adapter.verify(prepared)
    assert correctness.status == "not_run" and correctness.artifacts == [] and "compile_error:" in (correctness.message or "")
    timing_report = adapter.benchmark(prepared)
    assert timing_report.status == "not_run" and timing_report.samples == [] and timing_report.artifacts == []


def test_verify_execution_exception_is_error_status_not_fail(real_jax: Any) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-boom"), PROBLEM)
    assert adapter.compile(prepared).status == "ok"

    def exploding(*args: Any) -> Any:
        raise RuntimeError("device lost")

    prepared.handle["compiled"] = exploding
    report = adapter.verify(prepared)
    assert report.status == "error" and (report.cases_total, report.cases_passed) == (1, 0)
    assert "execution raised RuntimeError: device lost" in (report.message or "")
    assert blob_json(report.artifacts[0])["status"] == "error"


@pytest.mark.parametrize(("overrides", "expected"), [({}, "pass"), ({"scale": 1.0}, "pass"), ({"scale": 2.0}, "fail")])
def test_implementation_overrides_are_applied_as_keyword_arguments(real_jax: Any, overrides: dict[str, Any], expected: str) -> None:
    adapter = cpu_adapter(RecordingResolver(resolved(scaled_kernel, inputs=jax_inputs)))
    prepared = adapter.prepare(cpu_spec("req-cpu-overrides", overrides=overrides), PROBLEM)
    assert prepared.source.implementation_overrides == overrides
    report = adapter.verify(prepared)
    assert report.status == expected


def test_cpu_interface_never_requires_tpu_authorization_but_tpu_label_is_refused(real_jax: Any) -> None:
    # No allow_tpu_execution flag is needed for a non-TPU instance...
    prepared = cpu_adapter(RecordingResolver(resolved(real_jax.jit(good_kernel), inputs=jax_inputs))).prepare(
        cpu_spec("req-cpu-noauth", authorization={}), PROBLEM
    )
    assert prepared.environment["backend"] == CPU_LABEL
    # ...and the TPU-labelled adapter on this host refuses even when authorized (see T29 below).
    with pytest.raises(BackendUnavailable):
        JaxAdapter(entrypoint_resolver=RecordingResolver(resolved(good_kernel))).prepare(make_spec("req-tpu-auth", authorization=AUTHORIZED), PROBLEM)


# ------------------------------------------------------------------------------ T29 (integration, executes here)
@pytest.mark.integration
def test_t29_tpu_adapter_refuses_honestly_on_this_cpu_only_host(real_jax: Any) -> None:
    """T29: without a TPU the adapter refuses with BackendUnavailable; it never runs on CPU labelled tpu.

    This runs against the real jax installation of this host.  It is deliberately not skipped:
    a skip would not be evidence.  If this host ever gains a TPU the assertion fails loudly and
    the positive T29 path must be exercised there instead.
    """
    if real_jax.default_backend() == "tpu":
        pytest.fail("this host reports a TPU backend; the negative T29 path is not observable here")
    adapter = JaxAdapter()  # production configuration: required_platform="tpu", backend="jax_tpu", real import of jax
    with pytest.raises(BackendUnavailable) as excinfo:
        adapter.check_environment()
    err = excinfo.value
    assert err.exit_code == 5
    assert err.details["required"] == "tpu"
    assert err.details["actual_backend"] == "cpu"
    assert err.details["devices"] and all(isinstance(d, str) and "tpu" not in d.lower() for d in err.details["devices"])
    assert err.details["backend"] == "jax_tpu"
    # The same refusal through prepare, with authorization granted and an entrypoint configured:
    with pytest.raises(BackendUnavailable) as excinfo2:
        JaxAdapter(entrypoint_resolver=RecordingResolver(resolved(good_kernel))).prepare(make_spec("req-t29", authorization=AUTHORIZED), PROBLEM)
    assert excinfo2.value.details["actual_backend"] == "cpu"
