"""Deterministic mock kernel adapter (specification section 17; adapters/mock.py).

The mock executes nothing: every report is scripted by ``request.metadata["mock"]``, every
artifact is flagged ``is_mock``, and its environment can never be relabelled as hardware.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from kernel_memory.adapters.base import PreparedExecution, RunRequestSpec, SourceSpec
from kernel_memory.adapters.mock import (
    DEFAULT_SAMPLE_PATTERN,
    KNOWN_KEYS,
    MOCK_ENVIRONMENT,
    MockAdapter,
    default_samples,
    mock_options,
)
from kernel_memory.domain import hashing
from kernel_memory.domain.errors import ExecutionInfrastructureError, InputError
from kernel_memory.domain.models import GitOid
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.storage import MemoryStore

from test_runner import BASELINE_OID, PROTOCOL, VERIFIER, registry_with

REPO_UID = "github:github.com:repo:900001"
ENTRYPOINT = "demo.reference:vector_add"
PROBLEM = {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}


def make_spec(
    request_id: str,
    *,
    mock: Any = "unset",
    repetitions: int = 5,
    overrides: dict | None = None,
    max_wall_seconds: float = 600.0,
    stage: str = "benchmark",
) -> RunRequestSpec:
    metadata: dict[str, Any] = {}
    if mock != "unset":
        metadata["mock"] = mock
    protocol = dict(PROTOCOL)
    protocol["repetitions"] = repetitions
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=f"key-{request_id}",
        subject_ref="baseline-demo",
        config_ref="cfg-demo",
        backend="mock",
        stage=stage,
        protocol=protocol,
        verifier=dict(VERIFIER),
        source=SourceSpec(
            repo_uid=REPO_UID,
            target_commit=BASELINE_OID,
            entrypoint=ENTRYPOINT,
            implementation_overrides=dict(overrides or {}),
        ),
        max_wall_seconds=max_wall_seconds,
        metadata=metadata,
    )


def blob_json(blob: Any) -> dict[str, Any]:
    assert blob.media_type == "application/json"
    return json.loads(blob.data.decode("utf-8"))


def prepared_for(request_id: str, **kwargs: Any) -> tuple[MockAdapter, PreparedExecution]:
    adapter = MockAdapter()
    return adapter, adapter.prepare(make_spec(request_id, **kwargs), PROBLEM)


# ------------------------------------------------------------------------------ option validation
def test_mock_options_defaults_when_no_scenario_given() -> None:
    for spec in (make_spec("r"), make_spec("r", mock=None), make_spec("r", mock={})):
        assert mock_options(spec) == {
            "compile": "ok",
            "correctness": "pass",
            "raise": None,
            "samples": None,
            "environment_overrides": {},
        }


def test_mock_options_unknown_key_is_invalid_mock_options() -> None:
    with pytest.raises(InputError) as excinfo:
        mock_options(make_spec("r", mock={"compile": "ok", "corectness": "pass"}))
    err = excinfo.value
    assert err.code == "INVALID_MOCK_OPTIONS" and err.exit_code == 2
    assert "corectness" in err.message
    assert err.details["known"] == sorted(KNOWN_KEYS)


@pytest.mark.parametrize("raw", ["ok", ["compile"], 1, True])
def test_mock_options_must_be_an_object(raw: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        mock_options(make_spec("r", mock=raw))
    assert excinfo.value.code == "INVALID_MOCK_OPTIONS"


@pytest.mark.parametrize(
    "scenario",
    [
        {"compile": "error"},
        {"compile": None},
        {"correctness": "passed"},
        {"correctness": True},
        {"raise": "oom"},
        {"raise": "RUNTIME_ERROR"},
        {"samples": []},
        {"samples": [0]},
        {"samples": [-5]},
        {"samples": [True]},
        {"samples": [1.5]},
        {"samples": "1000"},
        {"samples": [1000, "1000"]},
        {"environment_overrides": "tpu"},
        {"environment_overrides": [("backend", "mock")]},
    ],
)
def test_mock_options_invalid_values_are_invalid_mock_options(scenario: dict[str, Any]) -> None:
    with pytest.raises(InputError) as excinfo:
        mock_options(make_spec("r", mock=scenario))
    assert excinfo.value.code == "INVALID_MOCK_OPTIONS"


def test_mock_options_backend_override_is_forbidden() -> None:
    with pytest.raises(InputError) as excinfo:
        mock_options(make_spec("r", mock={"environment_overrides": {"backend": "jax_tpu"}}))
    err = excinfo.value
    assert err.code == "MOCK_BACKEND_IMMUTABLE" and err.exit_code == 2
    assert err.details == {"requested_backend": "jax_tpu"}


def test_mock_options_backend_override_equal_to_mock_is_a_no_op() -> None:
    options = mock_options(make_spec("r", mock={"environment_overrides": {"backend": "mock", "device_count": 4}}))
    assert options["environment_overrides"] == {"backend": "mock", "device_count": 4}


def test_mock_options_valid_scenario_round_trips() -> None:
    scenario = {"compile": "compile_error", "correctness": "fail", "raise": "timeout", "samples": [7, 8, 9], "environment_overrides": None}
    options = mock_options(make_spec("r", mock=scenario))
    assert options == {"compile": "compile_error", "correctness": "fail", "raise": "timeout", "samples": [7, 8, 9], "environment_overrides": {}}
    options["samples"].append(1)  # a copy, not the request's list
    assert scenario["samples"] == [7, 8, 9]


def test_default_samples_cycles_pattern_and_validates_repetitions() -> None:
    assert default_samples(5) == list(DEFAULT_SAMPLE_PATTERN)
    assert default_samples(7) == [1000, 1010, 1000, 990, 1000, 1000, 1010]
    assert default_samples(1) == [1000]
    for bad in (0, -1, True, 2.0, "3", None):
        with pytest.raises(InputError) as excinfo:
            default_samples(bad)  # type: ignore[arg-type]
        assert excinfo.value.code == "INVALID_PROTOCOL"


# ------------------------------------------------------------------------------ construction / environment
def test_adapter_backend_label_is_immutable() -> None:
    assert MockAdapter().backend == "mock"
    assert MockAdapter(adapter_id="mock-test").adapter_id == "mock-test"
    with pytest.raises(InputError) as excinfo:
        MockAdapter(backend="cpu")
    assert excinfo.value.code == "MOCK_BACKEND_IMMUTABLE"


def test_check_environment_reports_mock_without_hardware() -> None:
    adapter = MockAdapter(adapter_id="mock-env-test")
    env = adapter.check_environment()
    assert env["backend"] == "mock"
    assert env["accelerator_model"] == "MOCK-NO-HARDWARE"
    assert env["device_count"] == 1 and env["topology"] == "single"
    assert env["software"] == {"runner": "mock-env-test"}
    assert env["execution_flags"] == {} and env["host_timer_environment"] == {}
    assert env["unknown_required_fields"] == []
    assert set(env) == set(MOCK_ENVIRONMENT)
    json.dumps(env)


def test_check_environment_returns_fresh_copies() -> None:
    adapter = MockAdapter()
    env = adapter.check_environment()
    env["software"]["runner"] = "tampered"
    env["unknown_required_fields"].append("x")
    env["backend"] = "tpu"
    fresh = adapter.check_environment()
    assert fresh["backend"] == "mock" and fresh["software"] == {"runner": "mock-v1"} and fresh["unknown_required_fields"] == []
    assert MOCK_ENVIRONMENT["backend"] == "mock"


# ------------------------------------------------------------------------------ prepare
def test_prepare_returns_tested_equal_to_target_and_mock_digests() -> None:
    adapter, prepared = prepared_for("req-prep", overrides={"chunk": 8})
    source = prepared.source
    assert source.tested_commit == source.target_commit == BASELINE_OID
    assert source.tested_tree is None
    assert source.dirty is False and source.patch_digest is None
    assert source.checkout_mode == "exact_commit"
    assert source.merge_parent_oids == []
    assert source.entrypoint == ENTRYPOINT
    assert source.implementation_overrides == {"chunk": 8}
    assert source.source_digest == hashing.jcs_digest({"mock_source": ENTRYPOINT})
    assert prepared.input_suite_hash == hashing.jcs_digest({"mock_suite": PROBLEM})
    assert prepared.problem == PROBLEM and prepared.problem is not PROBLEM
    assert prepared.environment["backend"] == "mock" and prepared.environment["accelerator_model"] == "MOCK-NO-HARDWARE"
    assert any("mock" in note and "nothing was executed" in note for note in prepared.notes)
    assert prepared.handle["samples"] == list(DEFAULT_SAMPLE_PATTERN)
    assert prepared.handle["adapter_id"] == adapter.adapter_id


def test_prepare_cycles_default_samples_to_protocol_repetitions() -> None:
    _, prepared = prepared_for("req-rep", repetitions=8)
    assert prepared.handle["samples"] == default_samples(8)
    assert len(prepared.handle["samples"]) == 8


def test_prepare_uses_custom_samples_verbatim() -> None:
    _, prepared = prepared_for("req-custom", mock={"samples": [5, 4, 3]}, repetitions=10)
    assert prepared.handle["samples"] == [5, 4, 3]


def test_prepare_applies_environment_overrides_but_backend_stays_mock() -> None:
    _, prepared = prepared_for(
        "req-env", mock={"environment_overrides": {"accelerator_model": "MOCK-FAKE-CHIP", "device_count": 8, "backend": "mock"}}
    )
    assert prepared.environment["accelerator_model"] == "MOCK-FAKE-CHIP"
    assert prepared.environment["device_count"] == 8
    assert prepared.environment["backend"] == "mock"


def test_prepare_rejects_unknown_environment_field_override() -> None:
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-envbad", mock={"environment_overrides": {"gpu_model": "H100"}})
    assert excinfo.value.code == "INVALID_MOCK_OPTIONS"
    assert "gpu_model" in excinfo.value.message


def test_prepare_rejects_backend_relabel_and_bad_options_before_anything_else() -> None:
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-relabel", mock={"environment_overrides": {"backend": "tpu"}})
    assert excinfo.value.code == "MOCK_BACKEND_IMMUTABLE"
    with pytest.raises(InputError) as excinfo:
        prepared_for("req-unknown", mock={"bogus": 1})
    assert excinfo.value.code == "INVALID_MOCK_OPTIONS"


@pytest.mark.parametrize("problem", [None, [], "n=16", 16])
def test_prepare_requires_an_object_problem(problem: Any) -> None:
    with pytest.raises(InputError) as excinfo:
        MockAdapter().prepare(make_spec("req-prob"), problem)  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_PROBLEM"


def test_stages_refuse_a_prepared_execution_from_another_adapter() -> None:
    adapter, prepared = prepared_for("req-foreign")
    foreign = PreparedExecution(
        request=prepared.request, problem=prepared.problem, source=prepared.source, environment=prepared.environment, input_suite_hash="x", handle=None
    )
    for stage in (adapter.compile, adapter.verify, adapter.benchmark):
        with pytest.raises(InputError) as excinfo:
            stage(foreign)
        assert excinfo.value.code == "INVALID_PREPARED_EXECUTION"


# ------------------------------------------------------------------------------ compile
def test_compile_ok_is_scripted_and_flagged_mock() -> None:
    adapter, prepared = prepared_for("req-cok")
    report = adapter.compile(prepared)
    assert report.status == "ok"
    assert "mock" in (report.message or "")
    assert [b.kind for b in report.artifacts] == ["compile_log"]
    blob = report.artifacts[0]
    assert blob.artifact_id == "req-cok-compile-log"
    doc = blob_json(blob)
    assert doc["is_mock"] is True and doc["status"] == "ok" and doc["entrypoint"] == ENTRYPOINT


def test_compile_error_branch() -> None:
    adapter, prepared = prepared_for("req-cerr", mock={"compile": "compile_error"})
    report = adapter.compile(prepared)
    assert report.status == "compile_error"
    assert "mock" in (report.message or "") and "compile" in report.message
    doc = blob_json(report.artifacts[0])
    assert doc["is_mock"] is True and doc["status"] == "compile_error" and doc["kind"] == "compile_log"


# ------------------------------------------------------------------------------ verify
def test_verify_pass_branch() -> None:
    adapter, prepared = prepared_for("req-vpass")
    report = adapter.verify(prepared)
    assert report.status == "pass"
    assert (report.cases_total, report.cases_passed) == (1, 1)
    assert report.max_abs_error == 0.0 and report.max_rel_error == 0.0
    assert "mock" in (report.message or "")
    blob = report.artifacts[0]
    assert blob.kind == "correctness_report" and blob.artifact_id == "req-vpass-correctness"
    doc = blob_json(blob)
    assert doc["is_mock"] is True and doc["status"] == "pass"
    assert doc["reference_id"] == "mock:no-reference-executed"
    assert doc["tolerances"] == VERIFIER["tolerances"] and doc["nonfinite_policy"] == VERIFIER["nonfinite_policy"]


def test_verify_fail_branch() -> None:
    adapter, prepared = prepared_for("req-vfail", mock={"correctness": "fail"})
    report = adapter.verify(prepared)
    assert report.status == "fail"
    assert (report.cases_total, report.cases_passed) == (1, 0)
    assert report.max_abs_error == 1.0 and report.max_rel_error == 1.0
    doc = blob_json(report.artifacts[0])
    assert doc["is_mock"] is True and doc["status"] == "fail" and doc["cases_passed"] == 0


def test_verify_error_branch_reports_null_errors_not_zero() -> None:
    adapter, prepared = prepared_for("req-verr", mock={"correctness": "error"})
    report = adapter.verify(prepared)
    assert report.status == "error"
    assert (report.cases_total, report.cases_passed) == (1, 0)
    assert report.max_abs_error is None and report.max_rel_error is None
    doc = blob_json(report.artifacts[0])
    assert doc["is_mock"] is True and doc["status"] == "error"
    assert doc["max_abs_error"] is None and doc["max_rel_error"] is None


# ------------------------------------------------------------------------------ benchmark
def test_benchmark_default_samples_are_scripted_never_host_synchronized() -> None:
    adapter, prepared = prepared_for("req-bench", repetitions=6)
    report = adapter.benchmark(prepared)
    assert report.status == "recorded"
    assert report.unit == "nanoseconds"
    assert report.samples == default_samples(6) and len(report.samples) == 6
    assert report.timing_method == "mock"
    assert report.timing_method != "host_synchronized"
    assert "mock" in (report.message or "")
    blob = report.artifacts[0]
    assert blob.kind == "latency_samples" and blob.artifact_id == "req-bench-samples"
    doc = blob_json(blob)
    assert doc["is_mock"] is True
    assert doc["measured"] is False
    assert doc["timing_method"] == "mock"
    assert doc["unit"] == "nanoseconds" and doc["samples"] == report.samples and doc["count"] == 6
    assert doc["is_fixture"] is False
    assert doc["protocol_id"] == PROTOCOL["protocol_id"]


def test_benchmark_custom_samples_branch() -> None:
    adapter, prepared = prepared_for("req-bcustom", mock={"samples": [1500, 1400, 1600]})
    report = adapter.benchmark(prepared)
    assert report.samples == [1500, 1400, 1600]
    assert blob_json(report.artifacts[0])["samples"] == [1500, 1400, 1600]


def test_benchmark_raise_runtime_error() -> None:
    adapter, prepared = prepared_for("req-rt", mock={"raise": "runtime_error"})
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        adapter.benchmark(prepared)
    err = excinfo.value
    assert err.code == "RUNTIME_ERROR" and err.exit_code == 6
    assert err.details["is_mock"] is True and err.details["request_id"] == "req-rt"


def test_benchmark_raise_timeout_carries_max_wall_seconds() -> None:
    adapter, prepared = prepared_for("req-to", mock={"raise": "timeout"}, max_wall_seconds=12.5)
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        adapter.benchmark(prepared)
    err = excinfo.value
    assert err.code == "TIMEOUT" and err.exit_code == 6
    assert err.details["max_wall_seconds"] == 12.5 and err.details["is_mock"] is True


def test_benchmark_raise_infrastructure_error() -> None:
    adapter, prepared = prepared_for("req-infra", mock={"raise": "infrastructure_error"})
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        adapter.benchmark(prepared)
    err = excinfo.value
    assert err.code == "EXECUTION_INFRASTRUCTURE_ERROR" and err.exit_code == 6
    assert err.details["is_mock"] is True


def test_raise_modes_do_not_affect_compile_or_verify() -> None:
    adapter, prepared = prepared_for("req-raise-only", mock={"raise": "runtime_error"})
    assert adapter.compile(prepared).status == "ok"
    assert adapter.verify(prepared).status == "pass"


def test_every_artifact_in_every_branch_is_json_flagged_mock() -> None:
    scenarios = [
        {},
        {"compile": "compile_error"},
        {"correctness": "fail"},
        {"correctness": "error"},
        {"samples": [3, 2, 1]},
    ]
    for i, scenario in enumerate(scenarios):
        adapter, prepared = prepared_for(f"req-all-{i}", mock=scenario)
        reports = [adapter.compile(prepared), adapter.verify(prepared), adapter.benchmark(prepared)]
        blobs = [b for r in reports for b in r.artifacts]
        assert len(blobs) == 3
        for blob in blobs:
            doc = blob_json(blob)
            assert doc["is_mock"] is True
            assert blob.artifact_id.startswith(f"req-all-{i}-")
        for report in reports:
            assert "mock" in (report.message or "")


# ------------------------------------------------------------------------------ through the trusted runner
def run_json_artifacts(store: MemoryStore, run: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ref in run.payload.artifacts:
        out[ref.artifact_id] = json.loads(store.read_artifact(ref.sha256))
    return out


def test_runner_records_mock_run_with_mock_environment_and_flagged_artifacts(demo_store: MemoryStore) -> None:
    adapter = MockAdapter()
    runner = LocalRunner(demo_store, adapters=registry_with(adapter))
    runner.ledger.submit(make_spec("request-mock-ok", repetitions=5))
    run = runner.execute("request-mock-ok")
    p = run.payload
    assert p.execution_status == "succeeded"
    assert p.correctness.status == "pass" and p.correctness.cases_passed == 1
    assert p.timing.status == "recorded" and p.timing.sample_count == 5
    assert p.timing.median_us == pytest.approx(1.0)  # 1000 ns
    env = run.to_dict()["payload"]["environment"]
    assert env["backend"] == "mock" and env["accelerator_model"] == "MOCK-NO-HARDWARE"
    assert p.source.tested_commit == p.source.target_commit == BASELINE_OID
    docs = run_json_artifacts(demo_store, run)
    # adapter blobs are stored as <run_id>-<blob_id>; the runner adds its own <run_id>-samples summary blob
    adapter_ids = {f"{run.record_id}-request-mock-ok-{suffix}" for suffix in ("compile-log", "correctness", "samples")}
    assert adapter_ids <= set(docs)
    assert all(docs[k]["is_mock"] is True for k in adapter_ids)
    assert docs[f"{run.record_id}-samples"]["samples"] == list(DEFAULT_SAMPLE_PATTERN)
    # metrics that were not collected are null with a status, never 0
    assert all(m.value is None and m.status == "not_collected" for m in p.analysis_metrics)


def test_t14_runner_records_mock_compile_error_without_invented_numbers(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=registry_with(MockAdapter()))
    runner.ledger.submit(make_spec("request-mock-ce", mock={"compile": "compile_error"}))
    run = runner.execute("request-mock-ce")
    p = run.payload
    assert p.execution_status == "compile_error"
    assert p.correctness.status == "not_run" and p.correctness.max_abs_error is None
    assert p.timing.status == "not_run" and p.timing.sample_count == 0 and p.timing.median_us is None
    assert [a.kind for a in p.artifacts] == ["compile_log"]
    assert run_json_artifacts(demo_store, run)[p.artifacts[0].artifact_id]["is_mock"] is True


def test_t15_runner_records_mock_wrong_output_as_succeeded_plus_fail(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=registry_with(MockAdapter()))
    runner.ledger.submit(make_spec("request-mock-fail", mock={"correctness": "fail"}))
    run = runner.execute("request-mock-fail")
    p = run.payload
    assert p.execution_status == "succeeded"
    assert p.correctness.status == "fail" and p.correctness.cases_passed == 0 and p.correctness.max_abs_error == 1.0
    assert p.timing.status == "recorded"
    assert not p.is_terminal_success or p.correctness.status != "pass"


def test_runner_maps_mock_timeout_to_timeout_status(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=registry_with(MockAdapter()))
    runner.ledger.submit(make_spec("request-mock-to", mock={"raise": "timeout"}))
    run = runner.execute("request-mock-to")
    assert run.payload.execution_status == "timeout"
    assert run.payload.timing.status == "not_run"


def test_runner_refuses_invalid_mock_scenario_as_input_error(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=registry_with(MockAdapter()))
    runner.ledger.submit(make_spec("request-mock-bad", mock={"environment_overrides": {"backend": "tpu"}}))
    with pytest.raises(InputError) as excinfo:
        runner.execute("request-mock-bad")
    assert excinfo.value.code == "MOCK_BACKEND_IMMUTABLE"
    assert demo_store.get("run-request-mock-bad-a1") is None  # a refused request produces no Run


def test_adapter_source_snapshot_uses_request_target_commit_for_other_oids() -> None:
    other = GitOid("sha1", "0000000000000000000000000000000000000002")
    spec = make_spec("req-other")
    spec = RunRequestSpec(**{**spec.to_dict(), "source": SourceSpec(**{**spec.source.__dict__, "target_commit": other})})
    prepared = MockAdapter().prepare(spec, PROBLEM)
    assert prepared.source.tested_commit == other == prepared.source.target_commit
