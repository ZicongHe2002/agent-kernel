"""Real CPU demonstration adapter (specification section 17; T14, T15; adapters/cpu_demo.py).

These tests really execute numpy vector addition, compare it against the pure-Python
reference and record real ``perf_counter_ns`` samples. Only structure is asserted about the
samples (count, positivity, units, artifact agreement) -- never speed.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kernel_memory.adapters import cpu_demo, timing
from kernel_memory.adapters.base import PreparedExecution, RunRequestSpec, SourceSpec
from kernel_memory.adapters.cpu_demo import (
    ALLOWED_ENTRYPOINTS,
    CASE_NAMES,
    CHUNKED_ENTRYPOINT,
    DEFAULT_CHUNK,
    REFERENCE_ENTRYPOINT,
    SUITE_ID,
    SUITE_SPEC,
    TIMING_BOUNDARY,
    CpuDemoAdapter,
    current_source_commit,
    current_source_digest,
    default_cpu_protocol,
    default_cpu_verifier,
    generate_cases,
    resolve_entrypoint,
    source_manifest,
    suite_seed,
    vector_add_numpy,
    vector_add_numpy_chunked,
    vector_add_reference,
    vector_add_wrong,
)
from kernel_memory.domain import hashing, stats
from kernel_memory.domain.errors import (
    BackendUnavailable,
    ExecutionInfrastructureError,
    InputError,
    InvariantViolation,
    SecurityPolicyError,
)
from kernel_memory.domain.models import GitOid
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.services.register import add_baseline
from kernel_memory.storage import MemoryStore

from test_runner import registry_with

REPO_UID = "github:github.com:repo:900001"
NUMPY_EP = "kernel_memory.adapters.cpu_demo:vector_add_numpy"
WRONG_EP = "kernel_memory.adapters.cpu_demo:vector_add_wrong"
PROBLEM = {"n": 64, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}
REPETITIONS = 15
WARMUP = 2
OTHER_TARGET = GitOid("sha256", "ab" * 32)


def make_cpu_spec(
    request_id: str,
    *,
    entrypoint: str = NUMPY_EP,
    overrides: dict | None = None,
    target: GitOid | None = None,
    protocol: dict | None = None,
    verifier: dict | None = None,
    max_wall_seconds: float = 600.0,
    stage: str = "benchmark",
    subject_ref: str = "baseline-cpu-demo-source",
    config_ref: str = "cfg-demo",
    checkout_mode: str = "exact_commit",
) -> RunRequestSpec:
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=f"key-{request_id}",
        subject_ref=subject_ref,
        config_ref=config_ref,
        backend="cpu",
        stage=stage,
        protocol=dict(protocol if protocol is not None else default_cpu_protocol(repetitions=REPETITIONS, warmup=WARMUP)),
        verifier=dict(verifier if verifier is not None else default_cpu_verifier()),
        source=SourceSpec(
            repo_uid=REPO_UID,
            target_commit=target if target is not None else current_source_commit(),
            entrypoint=entrypoint,
            checkout_mode=checkout_mode,
            implementation_overrides=dict(overrides or {}),
        ),
        max_wall_seconds=max_wall_seconds,
    )


def prepared_for(request_id: str, **kwargs: Any) -> tuple[CpuDemoAdapter, PreparedExecution]:
    adapter = CpuDemoAdapter()
    return adapter, adapter.prepare(make_cpu_spec(request_id, **kwargs), PROBLEM)


def blob_json(blob: Any) -> dict[str, Any]:
    assert blob.media_type == "application/json"
    return json.loads(blob.data.decode("utf-8"))


class AdvancingWall:
    """A monotonic wall clock that jumps ``step`` seconds on every read (deterministic timeouts)."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


# ------------------------------------------------------------------------------ environment
def test_check_environment_reports_real_cpu_backend_and_numpy_version() -> None:
    env = CpuDemoAdapter().check_environment()
    assert env["backend"] == "cpu"
    assert env["software"]["numpy"] == str(np.__version__)
    assert env["software"]["python"] == platform.python_version()
    assert env["device_count"] == 1 and env["topology"] == "single_host"
    assert env["host_timer_environment"] == timing.host_timer_environment()
    assert env["host_timer_environment"]["timer"] == "time.perf_counter_ns"
    assert env["unknown_required_fields"] == []
    assert isinstance(env["accelerator_model"], str) and env["accelerator_model"]
    json.dumps(env)


def test_check_environment_without_numpy_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_demo, "np", None)
    adapter = CpuDemoAdapter()
    with pytest.raises(BackendUnavailable) as excinfo:
        adapter.check_environment()
    assert excinfo.value.exit_code == 5 and excinfo.value.details["missing"] == "numpy"
    with pytest.raises(BackendUnavailable):
        adapter.prepare(make_cpu_spec("req-nonumpy"), PROBLEM)


# ------------------------------------------------------------------------------ source identity
def test_current_source_commit_is_sha256_of_loaded_source_manifest_and_stable() -> None:
    manifest = source_manifest()
    assert [path for path, _ in manifest] == [
        "kernel_memory/adapters/cpu_demo.py",
        "kernel_memory/adapters/verify.py",
        "kernel_memory/adapters/timing.py",
    ]
    here = Path(cpu_demo.__file__).resolve().parent
    for rel, digest in manifest:
        assert digest == hashing.sha256_bytes((here / Path(rel).name).read_bytes())
    digest = hashing.source_digest(manifest)
    assert current_source_digest() == digest
    commit = current_source_commit()
    assert commit.algorithm == "sha256"
    assert len(commit.hex) == 64 and all(c in "0123456789abcdef" for c in commit.hex)
    assert digest == f"sha256:{commit.hex}"
    assert current_source_commit() == commit == current_source_commit()


# ------------------------------------------------------------------------------ prepare
def test_prepare_with_matching_target_commit_captures_real_source_and_suite() -> None:
    adapter, prepared = prepared_for("req-prep")
    source = prepared.source
    target = current_source_commit()
    assert source.tested_commit == target == source.target_commit
    assert source.source_digest == current_source_digest()
    assert source.dirty is False and source.patch_digest is None and source.tested_tree is None
    assert source.checkout_mode == "exact_commit" and source.entrypoint == NUMPY_EP
    assert source.implementation_overrides == {}
    n = PROBLEM["n"]
    seed = suite_seed(n)
    assert prepared.input_suite_hash == cpu_demo.input_suite_hash(n, seed)
    assert prepared.input_suite_hash != prepared.request.protocol["input_suite_hash"]  # placeholder replaced by real hash
    assert prepared.environment["backend"] == "cpu"
    assert [name for name, _, _ in prepared.handle["cases"]] == list(CASE_NAMES)
    assert prepared.handle["callable"] is vector_add_numpy and prepared.handle["kwargs"] == {}
    assert prepared.handle["n"] == n and prepared.handle["seed"] == seed
    assert any(TIMING_BOUNDARY in note for note in prepared.notes)


def test_prepare_with_mismatched_target_is_tested_source_mismatch() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        prepared_for("req-mismatch", target=OTHER_TARGET)
    err = excinfo.value
    assert err.code == "TESTED_SOURCE_MISMATCH" and err.exit_code == 2
    assert err.details["target_commit"] == {"algorithm": "sha256", "hex": OTHER_TARGET.hex}
    assert err.details["tested_commit"] == {"algorithm": "sha256", "hex": current_source_commit().hex}
    assert err.details["source_digest"] == current_source_digest()


def test_prepare_with_sha1_target_of_same_hex_length_mismatch_is_refused() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        prepared_for("req-algo", target=GitOid("sha1", "0" * 40))
    assert excinfo.value.code == "TESTED_SOURCE_MISMATCH"


@pytest.mark.parametrize(
    "problem",
    [
        {"n": 0, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
        {"n": 4.0, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
        {"n": True, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
        {"dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
        {"n": 8, "dtype": "float64", "operation": "vector_add", "outputs": ["y"]},
        {"n": 8, "dtype": "float32", "operation": "matmul", "outputs": ["y"]},
        {"n": 8, "dtype": "float32", "operation": "vector_add", "outputs": ["y", "lse"]},
        {"n": 8, "m": 8, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
    ],
)
def test_prepare_rejects_wrong_problem_shape(problem: dict[str, Any]) -> None:
    with pytest.raises(InputError) as excinfo:
        CpuDemoAdapter().prepare(make_cpu_spec("req-badproblem"), problem)
    assert excinfo.value.exit_code == 2


@pytest.mark.parametrize("problem", [None, [16], "n=16"])
def test_prepare_rejects_non_object_problem(problem: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        CpuDemoAdapter().prepare(make_cpu_spec("req-notobj"), problem)  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_PROBLEM"


@pytest.mark.parametrize("entrypoint", ["os:system", "kernel_memory.adapters.cpu_demo:vector_add_reference", "demo.reference:vector_add", "", None])
def test_non_allowlisted_entrypoint_is_security_policy_error_exit_7(entrypoint: Any) -> None:
    with pytest.raises(SecurityPolicyError) as excinfo:
        resolve_entrypoint(entrypoint)
    err = excinfo.value
    assert err.code == "ENTRYPOINT_NOT_ALLOWLISTED" and err.exit_code == 7
    assert err.details["allowed"] == sorted(ALLOWED_ENTRYPOINTS)
    if isinstance(entrypoint, str):
        with pytest.raises(SecurityPolicyError) as prep:
            CpuDemoAdapter().prepare(make_cpu_spec("req-sec", entrypoint=entrypoint), PROBLEM)
        assert prep.value.code == "ENTRYPOINT_NOT_ALLOWLISTED"


def test_allowlist_maps_names_to_the_module_callables_only() -> None:
    assert ALLOWED_ENTRYPOINTS == {
        NUMPY_EP: vector_add_numpy,
        CHUNKED_ENTRYPOINT: vector_add_numpy_chunked,
        WRONG_EP: vector_add_wrong,
    }
    assert REFERENCE_ENTRYPOINT not in ALLOWED_ENTRYPOINTS  # the reference is never a candidate


@pytest.mark.parametrize(
    "entrypoint, overrides",
    [
        (NUMPY_EP, {"chunk": 7}),
        (CHUNKED_ENTRYPOINT, {"chunk": 0}),
        (CHUNKED_ENTRYPOINT, {"chunk": True}),
        (CHUNKED_ENTRYPOINT, {"chunk": "7"}),
        (CHUNKED_ENTRYPOINT, {"block": 4}),
    ],
)
def test_prepare_rejects_invalid_overrides(entrypoint: str, overrides: dict[str, Any]) -> None:
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-ovr", entrypoint=entrypoint, overrides=overrides)
    assert excinfo.value.code == "INVALID_OVERRIDES"


def test_prepare_rejects_non_exact_checkout_mode() -> None:
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-merge", checkout_mode="integration_merge")
    assert excinfo.value.code == "INVALID_CHECKOUT_MODE"


def test_prepare_validates_protocol_and_verifier_before_execution() -> None:
    protocol = default_cpu_protocol()
    del protocol["warmup"]
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-proto", protocol=protocol)
    assert excinfo.value.code == "INVALID_PROTOCOL"

    verifier = default_cpu_verifier()
    verifier["tolerances"] = {"atol": "1e-6"}
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-ver1", verifier=verifier)
    assert excinfo.value.code == "INVALID_VERIFIER"

    verifier = default_cpu_verifier()
    verifier["tolerances"]["atol"] = "-1"
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-ver2", verifier=verifier)
    assert excinfo.value.code == "INVALID_TOLERANCE"

    verifier = default_cpu_verifier()
    verifier["nonfinite_policy"] = "accept_all"
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-ver3", verifier=verifier)
    assert excinfo.value.code == "INVALID_NONFINITE_POLICY"

    verifier = default_cpu_verifier()
    del verifier["nonfinite_policy"]
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-ver4", verifier=verifier)
    assert excinfo.value.code == "INVALID_VERIFIER"


# ------------------------------------------------------------------------------ defaults
def test_default_protocol_and_verifier_are_explicit_snapshots() -> None:
    protocol = default_cpu_protocol(repetitions=REPETITIONS, warmup=WARMUP)
    assert protocol["timing_method"] == "host_synchronized"
    assert protocol["warmup"] == WARMUP and protocol["repetitions"] == REPETITIONS
    assert protocol["statistic"] == "median" and protocol["quantile_method"] == "linear"
    assert protocol["include_compile"] is False and protocol["include_transfers"] is False
    assert protocol["input_suite_hash"] == "sha256:" + "0" * 64  # placeholder until prepare
    defaults = default_cpu_protocol()
    assert defaults["repetitions"] == 100 and defaults["warmup"] == 20
    for bad in ({"repetitions": 0}, {"repetitions": True}, {"warmup": -1}, {"warmup": 1.5}):
        with pytest.raises(InputError) as excinfo:
            default_cpu_protocol(**bad)
        assert excinfo.value.code == "INVALID_PROTOCOL"

    verifier = default_cpu_verifier()
    assert verifier["reference_source_hash"] == hashing.sha256_bytes(Path(cpu_demo.__file__).resolve().read_bytes())
    assert verifier["suite_hash"] == hashing.jcs_digest(SUITE_SPEC)
    assert verifier["tolerances"] == {"atol": "1e-6", "rtol": "1e-6"}
    assert verifier["nonfinite_policy"] == "reject_unexpected"
    assert SUITE_SPEC["reference"] == REFERENCE_ENTRYPOINT and SUITE_SPEC["suite"] == SUITE_ID


# ------------------------------------------------------------------------------ candidates and suite
def test_candidates_and_reference_behave_as_documented() -> None:
    x = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    y = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    expected = np.array([1.5, 3.0, -2.5], dtype=np.float32)
    assert vector_add_reference(x.tolist(), y.tolist()) == [1.5, 3.0, -2.5]
    assert np.array_equal(vector_add_numpy(x, y), expected) and vector_add_numpy(x, y).dtype == np.float32
    assert np.array_equal(vector_add_numpy_chunked(x, y, chunk=2), expected)
    wrong = vector_add_wrong(x, y)
    assert wrong.dtype == np.float32 and np.all(np.abs(wrong - expected) > 5e-4)
    with pytest.raises(InputError) as excinfo:
        vector_add_reference([1.0], [1.0, 2.0])
    assert excinfo.value.code == "INVALID_INPUTS"


def test_generate_cases_is_deterministic_and_float32() -> None:
    n = 33
    seed = suite_seed(n)
    first = generate_cases(n, seed)
    second = generate_cases(n, seed)
    assert [c[0] for c in first] == list(CASE_NAMES)
    for (name, x1, y1), (_, x2, y2) in zip(first, second):
        assert x1.dtype == np.float32 and y1.dtype == np.float32 and x1.shape == (n,)
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2), name
    cases = dict((name, (x, y)) for name, x, y in first)
    assert not cases["zeros"][0].any() and not cases["zeros"][1].any()
    assert np.array_equal(cases["alternating_signs"][0], -cases["alternating_signs"][1])
    assert suite_seed(n) != suite_seed(n + 1)
    assert generate_cases(n, suite_seed(n + 1))[0][1].tolist() != first[0][1].tolist()


# ------------------------------------------------------------------------------ compile
def test_compile_ok_records_precompiled_note_and_overrides() -> None:
    adapter, prepared = prepared_for("req-compile", entrypoint=CHUNKED_ENTRYPOINT, overrides={"chunk": 7})
    report = adapter.compile(prepared)
    assert report.status == "ok"
    assert str(np.__version__) in (report.message or "")
    blob = report.artifacts[0]
    assert blob.kind == "compile_log" and blob.artifact_id == "req-compile-compile-log"
    doc = blob_json(blob)
    assert doc["status"] == "ok" and doc["is_fixture"] is False
    assert doc["entrypoint"] == CHUNKED_ENTRYPOINT and doc["overrides"] == {"chunk": 7}


def test_compile_reports_compile_error_instead_of_raising_when_entrypoint_no_longer_resolves() -> None:
    adapter, prepared = prepared_for("req-compile-bad")
    prepared.source.entrypoint = "os:system"  # documented behaviour: a report, not a raise (T14)
    report = adapter.compile(prepared)
    assert report.status == "compile_error"
    assert "entrypoint resolution failed" in (report.message or "")
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "compile_error"
    assert doc["error"]["error"] == "ENTRYPOINT_NOT_ALLOWLISTED" and doc["error"]["exit_code"] == 7


def test_compile_reports_compile_error_when_prepared_callable_does_not_match_allowlist() -> None:
    adapter, prepared = prepared_for("req-compile-swap")
    prepared.handle["callable"] = vector_add_wrong  # tampering with the prepared callable is detected
    report = adapter.compile(prepared)
    assert report.status == "compile_error"
    assert blob_json(report.artifacts[0])["error"]["error"] == "ENTRYPOINT_MISMATCH"


def test_stages_refuse_prepared_execution_from_another_adapter() -> None:
    adapter, prepared = prepared_for("req-foreign")
    foreign = PreparedExecution(
        request=prepared.request, problem=prepared.problem, source=prepared.source, environment=prepared.environment, input_suite_hash="x", handle={"other": 1}
    )
    with pytest.raises(InputError) as excinfo:
        adapter.verify(foreign)
    assert excinfo.value.code == "INVALID_PREPARED_EXECUTION"
    with pytest.raises(InputError):
        adapter.benchmark(foreign)


# ------------------------------------------------------------------------------ verify (real execution)
def test_verify_numpy_candidate_passes_every_case_with_zero_error() -> None:
    adapter, prepared = prepared_for("req-verify-ok")
    report = adapter.verify(prepared)
    assert report.status == "pass"
    assert report.cases_total == len(CASE_NAMES) == 4
    assert report.cases_passed == report.cases_total
    assert report.max_abs_error == 0.0 and report.max_rel_error == 0.0
    assert report.message is None
    blob = report.artifacts[0]
    assert blob.kind == "correctness_report" and blob.artifact_id == "req-verify-ok-correctness"
    doc = blob_json(blob)
    assert doc["kind"] == "correctness_report" and doc["status"] == "pass"
    assert doc["is_fixture"] is False
    assert doc["reference_id"] == REFERENCE_ENTRYPOINT
    assert doc["tolerances"] == {"atol": "1e-6", "rtol": "1e-6"} and doc["nonfinite_policy"] == "reject_unexpected"
    assert doc["rel_error_epsilon"] == 1e-12
    assert [c["case"] for c in doc["cases"]] == list(CASE_NAMES)
    assert all(c["passed"] is True and c["checked_count"] == PROBLEM["n"] for c in doc["cases"])
    assert doc["cases_total"] == 4 and doc["cases_passed"] == 4
    assert doc["input_suite_hash"] == prepared.input_suite_hash and doc["entrypoint"] == NUMPY_EP


def test_t15_verify_wrong_candidate_executes_but_fails_with_about_1e_3_error() -> None:
    """T15: the wrong candidate executes, but its output is wrong -> correctness ``fail``.

    ``vector_add_wrong`` adds ``1e-3`` in float32.  On the larger-magnitude cases the float32
    sum is quantised to the nearest representable value, so the recorded max absolute error is
    a power-of-two multiple near ``2e-3`` (``0.001953125``), not exactly ``1e-3``.  The evidence
    the spec needs (executes, fails verification, never promotable) does not depend on the
    exact digit, so the assertion brackets the error instead of pinning it.
    """
    adapter, prepared = prepared_for("req-verify-wrong", entrypoint=WRONG_EP)
    assert adapter.compile(prepared).status == "ok"  # it compiles and runs...
    report = adapter.verify(prepared)
    assert report.status == "fail"  # ...but the output is wrong: a verdict, not an execution failure
    assert report.cases_total == 4 and report.cases_passed < 4
    assert report.max_abs_error is not None and 5e-4 < report.max_abs_error < 5e-3
    assert report.max_rel_error is not None and report.max_rel_error > 0
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "fail" and doc["cases_passed"] == report.cases_passed
    failing = {c["case"] for c in doc["cases"] if not c["passed"]}
    assert {"uniform_unit", "zeros", "alternating_signs"} <= failing
    assert all(str(name) in (report.message or "") for name in failing)
    assert doc["max_abs_error"] == report.max_abs_error


def test_verify_candidate_exception_is_error_status_not_fail() -> None:
    adapter, prepared = prepared_for("req-verify-boom")

    def boom(x: Any, y: Any) -> Any:
        raise RuntimeError("kernel exploded")

    prepared.handle["callable"] = boom
    report = adapter.verify(prepared)
    assert report.status == "error"
    assert report.cases_passed == 0 and report.cases_total == 4
    assert report.max_abs_error is None and report.max_rel_error is None  # nothing measured, nothing invented
    assert "candidate raised RuntimeError: kernel exploded" in (report.message or "")
    doc = blob_json(report.artifacts[0])
    assert doc["status"] == "error" and doc["cases"][0]["error"]["type"] == "RuntimeError"
    assert len(doc["cases"]) == 1  # stops at the first erroring case


def test_verify_honours_request_tolerances_rather_than_defaults() -> None:
    """The request's tolerances (not the adapter defaults) decide the verdict.

    With ``atol=0.01`` the deliberately wrong candidate passes.  The measured error is bracketed
    rather than pinned to ``1e-3``: float32 quantisation of the sum on the larger-magnitude
    inputs makes the max absolute error a power-of-two multiple near ``2e-3``.
    """
    verifier = default_cpu_verifier()
    verifier["tolerances"] = {"atol": "0.01", "rtol": "0"}  # loose enough for the deliberately wrong candidate
    adapter, prepared = prepared_for("req-verify-loose", entrypoint=WRONG_EP, verifier=verifier)
    report = adapter.verify(prepared)
    assert report.status == "pass" and report.cases_passed == 4
    assert report.max_abs_error is not None and 5e-4 < report.max_abs_error < 5e-3
    assert blob_json(report.artifacts[0])["tolerances"] == {"atol": "0.01", "rtol": "0"}


# ------------------------------------------------------------------------------ chunked variant
def test_chunked_variant_with_override_passes_and_has_distinct_variant_digest() -> None:
    adapter, chunked = prepared_for("req-chunked", entrypoint=CHUNKED_ENTRYPOINT, overrides={"chunk": 7})
    assert chunked.source.implementation_overrides == {"chunk": 7}
    assert chunked.handle["kwargs"] == {"chunk": 7}
    assert chunked.handle["callable"] is vector_add_numpy_chunked
    assert adapter.verify(chunked).status == "pass"

    _, plain = prepared_for("req-plain")
    assert chunked.source.source_digest == plain.source.source_digest  # same source, ...
    chunked_variant = hashing.variant_digest(
        source_digest=chunked.source.source_digest,
        entrypoint=chunked.source.entrypoint,
        implementation_overrides=chunked.source.implementation_overrides,
        checkout_mode=chunked.source.checkout_mode,
    )
    plain_variant = hashing.variant_digest(
        source_digest=plain.source.source_digest,
        entrypoint=plain.source.entrypoint,
        implementation_overrides=plain.source.implementation_overrides,
        checkout_mode=plain.source.checkout_mode,
    )
    assert chunked_variant != plain_variant  # ... different variant (specification section 5.3)
    other_chunk = hashing.variant_digest(
        source_digest=chunked.source.source_digest, entrypoint=CHUNKED_ENTRYPOINT, implementation_overrides={"chunk": 8}, checkout_mode="exact_commit"
    )
    assert other_chunk != chunked_variant


def test_chunked_variant_without_override_uses_documented_default_and_notes_it() -> None:
    _, prepared = prepared_for("req-chunked-default", entrypoint=CHUNKED_ENTRYPOINT)
    assert prepared.source.implementation_overrides == {}
    assert prepared.handle["kwargs"] == {"chunk": DEFAULT_CHUNK}
    assert any(str(DEFAULT_CHUNK) in note for note in prepared.notes)


# ------------------------------------------------------------------------------ benchmark (real samples)
def test_benchmark_records_exactly_repetitions_real_nanosecond_samples() -> None:
    adapter, prepared = prepared_for("req-bench")
    report = adapter.benchmark(prepared)
    assert report.status == "recorded"
    assert report.unit == "nanoseconds"
    assert report.timing_method == "host_synchronized"
    assert len(report.samples) == REPETITIONS == 15
    assert all(type(s) is int and s > 0 for s in report.samples)
    assert NUMPY_EP in (report.message or "") and "15" in report.message
    blob = report.artifacts[0]
    assert blob.kind == "latency_samples" and blob.artifact_id == "req-bench-samples"
    doc = blob_json(blob)
    assert doc["unit"] == "nanoseconds" and doc["count"] == 15
    assert doc["samples"] == report.samples
    assert doc["is_fixture"] is False
    assert doc["timing_method"] == "host_synchronized"
    assert doc["warmup"] == WARMUP and doc["repetitions"] == REPETITIONS
    assert doc["boundary"] == TIMING_BOUNDARY
    assert doc["timer"] == "time.perf_counter_ns"
    assert doc["entrypoint"] == NUMPY_EP and doc["n"] == PROBLEM["n"] and doc["input_case"] == "uniform_unit"
    assert doc["output_consumed_calls"] == 1 + WARMUP + REPETITIONS  # every output was consumed
    assert isinstance(doc["output_checksum"], float)
    summary = stats.summarize(doc["samples"], doc["unit"])
    assert summary.sample_count == 15 and summary.median_us > 0


def test_benchmark_tiny_max_wall_seconds_with_monkeypatched_clock_is_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "wall_clock", AdvancingWall(step=1.0))
    adapter, prepared = prepared_for("req-bench-timeout", max_wall_seconds=0.5)
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        adapter.benchmark(prepared)
    err = excinfo.value
    assert err.code == "TIMEOUT" and err.exit_code == 6
    assert err.details["max_wall_seconds"] == 0.5
    assert err.details["phase"] == "first_call" and err.details["samples_collected"] == 0


def test_benchmark_unusable_timer_is_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "perf_counter_ns", lambda: 42)
    adapter, prepared = prepared_for("req-bench-stuck")
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        adapter.benchmark(prepared)
    assert excinfo.value.code == "NON_POSITIVE_ELAPSED"


# ------------------------------------------------------------------------------ through the trusted runner
@pytest.fixture
def cpu_store(demo_store: MemoryStore) -> MemoryStore:
    add_baseline(
        demo_store,
        "cfg-demo",
        "cpu-demo-source",
        description="CPU demo baseline at the content address of the loaded demo source.",
        repo_uid=REPO_UID,
        commit_oid=current_source_commit(),
        entrypoint=NUMPY_EP,
    )
    return demo_store


def artifact_json(store: MemoryStore, run: Any, artifact_id: str) -> dict[str, Any]:
    ref = run.payload.artifact_by_id()[artifact_id]
    return json.loads(store.read_artifact(ref.sha256))


def test_runner_executes_cpu_demo_and_summary_agrees_with_domain_stats(cpu_store: MemoryStore) -> None:
    adapter = CpuDemoAdapter()
    runner = LocalRunner(cpu_store, adapters=registry_with(adapter))
    runner.ledger.submit(make_cpu_spec("request-cpu-ok", entrypoint=CHUNKED_ENTRYPOINT, overrides={"chunk": 7}))
    run = runner.execute("request-cpu-ok")
    p = run.payload
    assert p.provenance == "trusted_worker"
    assert p.execution_status == "succeeded" and p.failure_reason is None
    assert p.correctness.status == "pass" and p.correctness.cases_total == 4 and p.correctness.cases_passed == 4
    assert p.correctness.max_abs_error == 0.0
    assert p.timing.status == "recorded" and p.timing.sample_count == REPETITIONS
    assert p.timing.median_us is not None and p.timing.median_us > 0
    assert p.source.tested_commit == p.source.target_commit == current_source_commit()
    assert p.source.source_digest == current_source_digest()
    assert p.source.implementation_overrides == {"chunk": 7} and p.source.dirty is False
    assert p.source.variant_digest == hashing.variant_digest(
        source_digest=p.source.source_digest, entrypoint=CHUNKED_ENTRYPOINT, implementation_overrides={"chunk": 7}, checkout_mode="exact_commit"
    )
    env = run.to_dict()["payload"]["environment"]
    assert env["backend"] == "cpu" and env["software"]["numpy"] == str(np.__version__)
    assert env["environment_hash"] == hashing.environment_hash(env)

    # The runner's summary is exactly domain.stats over the stored raw samples.
    samples_doc = artifact_json(cpu_store, run, p.timing.samples_artifact_ref)
    assert samples_doc["unit"] == "nanoseconds" and len(samples_doc["samples"]) == REPETITIONS
    summary = stats.summarize(samples_doc["samples"], samples_doc["unit"])
    assert p.timing.median_us == summary.median_us and p.timing.p90_us == summary.p90_us
    assert stats.summaries_agree(samples_doc["samples"], samples_doc["unit"], p.timing.median_us, p.timing.p90_us)
    # The adapter's own artifact carries the same raw samples and the measurement metadata.
    adapter_doc = artifact_json(cpu_store, run, f"{run.record_id}-request-cpu-ok-samples")
    assert adapter_doc["samples"] == samples_doc["samples"] and adapter_doc["is_fixture"] is False
    assert adapter_doc["timing_method"] == "host_synchronized"
    # Uncollected metrics stay null (T17), never 0.
    assert all(m.value is None and m.status == "not_collected" for m in p.analysis_metrics)
    assert runner.ledger.state("request-cpu-ok").kind == "finished"


def test_t15_runner_records_wrong_candidate_as_succeeded_plus_fail_never_promotable(cpu_store: MemoryStore) -> None:
    runner = LocalRunner(cpu_store, adapters=registry_with(CpuDemoAdapter()))
    runner.ledger.submit(make_cpu_spec("request-cpu-wrong", entrypoint=WRONG_EP))
    run = runner.execute("request-cpu-wrong")
    p = run.payload
    assert p.execution_status == "succeeded"  # it ran to completion ...
    assert p.correctness.status == "fail"  # ... and produced wrong numbers: separate facts
    assert p.correctness.cases_passed < p.correctness.cases_total == 4
    assert p.correctness.max_abs_error == pytest.approx(1e-3, rel=0.05)
    assert p.timing.status == "recorded" and p.timing.sample_count == REPETITIONS
    assert not p.is_terminal_success or p.correctness.status != "pass"


def test_runner_refuses_mismatched_target_without_executing(cpu_store: MemoryStore) -> None:
    adapter = CpuDemoAdapter()
    runner = LocalRunner(cpu_store, adapters=registry_with(adapter))
    runner.ledger.submit(make_cpu_spec("request-cpu-mismatch", target=OTHER_TARGET, subject_ref="baseline-cpu-demo-source"))
    with pytest.raises(InputError) as excinfo:  # the subject's commit is the real content address
        runner.execute("request-cpu-mismatch")
    assert excinfo.value.code == "SUBJECT_SOURCE_MISMATCH"
    assert cpu_store.get("run-request-cpu-mismatch-a1") is None


def test_runner_refuses_non_allowlisted_entrypoint_as_security_denial(cpu_store: MemoryStore) -> None:
    runner = LocalRunner(cpu_store, adapters=registry_with(CpuDemoAdapter()))
    runner.ledger.submit(make_cpu_spec("request-cpu-sec", entrypoint="os:system"))
    with pytest.raises(SecurityPolicyError) as excinfo:
        runner.execute("request-cpu-sec")
    assert excinfo.value.exit_code == 7
    assert cpu_store.get("run-request-cpu-sec-a1") is None
    assert runner.ledger.state("request-cpu-sec").kind == "cancelled"


def test_runner_records_adapter_timeout_as_timeout_status(cpu_store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """A benchmark timeout is recorded as execution_status ``timeout`` with no results.

    DESIGN.md section 5: any non-succeeded execution status carries correctness ``not_run`` and
    timing ``not_run`` (the validator's FABRICATED_RESULT rule).  Even though verification
    completed before the benchmark timed out, the run as a whole did not succeed, so the runner
    must not persist a ``pass`` verdict alongside a ``timeout`` status.
    """
    monkeypatch.setattr(timing, "wall_clock", AdvancingWall(step=1.0))
    runner = LocalRunner(cpu_store, adapters=registry_with(CpuDemoAdapter()))
    runner.ledger.submit(make_cpu_spec("request-cpu-timeout", max_wall_seconds=0.5))
    run = runner.execute("request-cpu-timeout")
    p = run.payload
    assert p.execution_status == "timeout"
    assert "max_wall_seconds=0.5" in (p.failure_reason or "")
    assert p.timing.status == "not_run" and p.timing.median_us is None
    assert p.correctness.status == "not_run"  # non-succeeded status never carries a verdict (DESIGN section 5)
