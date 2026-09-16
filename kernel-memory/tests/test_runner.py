"""Trusted local Runner: request -> exactly one immutable Run (T11, T13, T14, T15, T26, T29)."""
from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from kernel_memory.adapters.base import (
    AdapterRegistry,
    ArtifactBlob,
    CompileReport,
    CorrectnessReport,
    PreparedExecution,
    RunRequestSpec,
    SourceSnapshot,
    SourceSpec,
    TimingReport,
)
from kernel_memory.domain import hashing, stats
from kernel_memory.domain.errors import (
    AuthorizationError,
    BackendUnavailable,
    ExecutionInfrastructureError,
    IncompleteProblemContract,
    InputError,
    InvariantViolation,
    LeaseLostError,
    SecurityPolicyError,
)
from kernel_memory.domain.models import GitOid, Record
from kernel_memory.execution.ledger import run_record_id
from kernel_memory.execution.runner import KNOWN_METRICS, LocalRunner
from kernel_memory.storage import MemoryStore

BASELINE_OID = GitOid("sha1", "0000000000000000000000000000000000000001")
OTHER_OID = GitOid("sha1", "0000000000000000000000000000000000000002")
REPO_UID = "github:github.com:repo:900001"
ENTRYPOINT = "demo.reference:vector_add"
PROTOCOL = {
    "protocol_id": "test-bench-v1",
    "measurement_scope": "kernel_only",
    "timing_method": "host_synchronized",
    "include_compile": False,
    "include_transfers": False,
    "warmup": 1,
    "repetitions": 5,
    "statistic": "median",
    "quantile_method": "linear",
    "capture_profile": False,
}
VERIFIER = {
    "verifier_id": "test-verifier-v1",
    "reference_source_hash": hashing.sha256_bytes(b"reference"),
    "suite_hash": hashing.sha256_bytes(b"suite"),
    "tolerances": {"atol": "0", "rtol": "0"},
    "nonfinite_policy": "reject_unexpected",
}
SAMPLES = [10.0, 11.0, 10.0, 9.0, 12.0]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubAdapter:
    """In-test KernelAdapter with configurable outcomes and call counters (never collected by pytest)."""

    __test__ = False
    adapter_id = "test-adapter-v1"
    backend = "mock"

    def __init__(
        self,
        *,
        compile_status: str = "ok",
        correctness: str = "pass",
        samples: list[float] | None = None,
        unit: str = "microseconds",
        env_error: Exception | None = None,
        prepare_error: Exception | None = None,
        verify_error: Exception | None = None,
        benchmark_error: Exception | None = None,
        tested_hex: str | None = None,
        dirty: bool = False,
        on_benchmark: Callable[[PreparedExecution], None] | None = None,
    ) -> None:
        self.compile_status = compile_status
        self.correctness = correctness
        self.samples = list(SAMPLES if samples is None else samples)
        self.unit = unit
        self.env_error = env_error
        self.prepare_error = prepare_error
        self.verify_error = verify_error
        self.benchmark_error = benchmark_error
        self.tested_hex = tested_hex
        self.dirty = dirty
        self.on_benchmark = on_benchmark
        self.calls: dict[str, int] = {"check_environment": 0, "prepare": 0, "compile": 0, "verify": 0, "benchmark": 0}

    def check_environment(self) -> dict:
        self.calls["check_environment"] += 1
        if self.env_error is not None:
            raise self.env_error
        return {
            "backend": "mock",
            "accelerator_model": "HOST-CPU-STUB",
            "device_count": 1,
            "topology": "single",
            "software": {"python": "3.11", "adapter": self.adapter_id},
            "execution_flags": {},
            "host_timer_environment": {"timer": "perf_counter_ns"},
            "unknown_required_fields": [],
        }

    def prepare(self, request: RunRequestSpec, problem: dict) -> PreparedExecution:
        self.calls["prepare"] += 1
        if self.prepare_error is not None:
            raise self.prepare_error
        target = request.source.target_commit
        tested = GitOid(target.algorithm, self.tested_hex or target.hex)
        snapshot = SourceSnapshot(
            repo_uid=request.source.repo_uid,
            target_commit=target,
            tested_commit=tested,
            tested_tree=GitOid("sha1", "0000000000000000000000000000000000000065"),
            checkout_mode=request.source.checkout_mode,
            merge_parent_oids=[],
            dirty=self.dirty,
            patch_digest=hashing.sha256_bytes(b"patch") if self.dirty else None,
            source_digest=hashing.source_digest([("demo/reference.py", hashing.sha256_bytes(b"def vector_add"))]),
            entrypoint=request.source.entrypoint,
            implementation_overrides=dict(request.source.implementation_overrides),
        )
        return PreparedExecution(
            request=request,
            problem=dict(problem),
            source=snapshot,
            environment=self.check_environment(),
            input_suite_hash=hashing.jcs_digest({"problem": problem, "seed": 1}),
            handle={"overrides": dict(request.source.implementation_overrides)},
        )

    def compile(self, prepared: PreparedExecution) -> CompileReport:
        self.calls["compile"] += 1
        if self.compile_status == "compile_error":
            log = ArtifactBlob("compile-log", "compile_log", "text/plain", b"error: synthetic compile failure\n")
            return CompileReport("compile_error", "synthetic compile failure", [log])
        return CompileReport(self.compile_status, None, [])

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        self.calls["verify"] += 1
        if self.verify_error is not None:
            raise self.verify_error
        passed = self.correctness == "pass"
        report = {"status": self.correctness, "cases_total": 1, "cases_passed": 1 if passed else 0}
        blob = ArtifactBlob("correctness", "correctness_report", "application/json", json.dumps(report).encode("utf-8"))
        return CorrectnessReport(
            status=self.correctness,
            cases_total=1,
            cases_passed=1 if passed else 0,
            max_abs_error=0.0 if passed else 0.5,
            max_rel_error=0.0 if passed else 0.25,
            artifacts=[blob],
        )

    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        self.calls["benchmark"] += 1
        if self.on_benchmark is not None:
            self.on_benchmark(prepared)
        if self.benchmark_error is not None:
            raise self.benchmark_error
        return TimingReport("recorded", self.unit, list(self.samples), "host_synchronized")


def registry_with(adapter: Any) -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register_kernel_adapter(adapter)
    return registry


def make_spec(
    request_id: str,
    *,
    key: str | None = None,
    overrides: dict | None = None,
    backend: str = "mock",
    stage: str = "benchmark",
    target: GitOid = BASELINE_OID,
    authorization: dict | None = None,
    max_wall_seconds: float = 600.0,
    rerun_of: str | None = None,
    allow_dirty: bool = False,
    subject_ref: str = "baseline-demo",
) -> RunRequestSpec:
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=key if key is not None else f"key-{request_id}",
        subject_ref=subject_ref,
        config_ref="cfg-demo",
        backend=backend,
        stage=stage,
        protocol=dict(PROTOCOL),
        verifier=dict(VERIFIER),
        source=SourceSpec(
            repo_uid=REPO_UID,
            target_commit=target,
            entrypoint=ENTRYPOINT,
            implementation_overrides=dict(overrides or {}),
            allow_dirty_exploratory=allow_dirty,
        ),
        authorization=dict(authorization or {}),
        session_id="session-test",
        pair_id="pair-test",
        role_in_pair="candidate",
        max_wall_seconds=max_wall_seconds,
        rerun_of=rerun_of,
    )


def make_runner(store: MemoryStore, adapter: Any, **kwargs: Any) -> LocalRunner:
    return LocalRunner(store, adapters=registry_with(adapter), **kwargs)


def artifact_json(store: MemoryStore, run: Record, artifact_id: str) -> dict:
    ref = run.payload.artifact_by_id()[artifact_id]
    return json.loads(store.read_artifact(ref.sha256))


def event_kinds(runner: LocalRunner, request_id: str) -> list[str]:
    return [e["kind"] for e in runner.ledger.state(request_id).events]


# ------------------------------------------------------------------------------ success path
def test_success_path_publishes_trusted_run_with_recomputable_evidence(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    runner = make_runner(demo_store, adapter)
    spec = make_spec("request-ok", overrides={"chunk": 8})
    outcome = runner.ledger.submit(spec)
    run = runner.execute(outcome.request_id)
    p = run.payload

    assert run.record_id == run_record_id("request-ok", 1) == "run-request-ok-a1"
    assert demo_store.get(run.record_id) == run
    assert p.provenance == "trusted_worker"
    assert p.execution_status == "succeeded" and p.failure_reason is None
    assert p.attempt_no == 1 and p.request_id == "request-ok" and p.rerun_of is None
    assert p.correctness.status == "pass" and p.correctness.cases_passed == 1
    assert p.correctness.report_artifact_ref == "run-request-ok-a1-correctness"
    assert (p.session_id, p.pair_id, p.role_in_pair) == ("session-test", "pair-test", "candidate")
    assert p.source.implementation_overrides == {"chunk": 8}
    assert p.source.tested_commit == p.source.target_commit == BASELINE_OID
    assert p.source.dirty is False and p.source.checkout_mode == "exact_commit"

    # Timing summary equals stats over the stored samples artifact exactly.
    assert p.timing.status == "recorded" and p.timing.samples_artifact_ref == "run-request-ok-a1-samples"
    samples_doc = artifact_json(demo_store, run, p.timing.samples_artifact_ref)
    assert samples_doc["unit"] == "microseconds" and samples_doc["samples"] == SAMPLES
    summary = stats.summarize(samples_doc["samples"], samples_doc["unit"])
    assert p.timing.sample_count == summary.sample_count == 5
    assert p.timing.median_us == summary.median_us == 10.0
    assert p.timing.p90_us == summary.p90_us
    assert stats.summaries_agree(samples_doc["samples"], samples_doc["unit"], p.timing.median_us, p.timing.p90_us)

    # Every artifact is content-addressed, present, and registered.
    assert {a.artifact_id for a in p.artifacts} == {"run-request-ok-a1-correctness", "run-request-ok-a1-samples"}
    for ref in p.artifacts:
        assert ref.uri == f"artifact://sha256/{ref.sha256[len('sha256:'):]}"
        assert ref.availability == "present" and ref.retention == "permanent"
        data = demo_store.read_artifact(ref.sha256)
        assert len(data) == ref.size_bytes and hashing.artifact_digest(data) == ref.sha256
        assert demo_store.get_artifact_ref(ref.artifact_id) == ref
    assert demo_store.verify_artifact(p.artifacts[0]) is None

    # Hashes are recomputable from the recorded snapshots (DESIGN section 5).
    env = json.loads(json.dumps(run.to_dict()["payload"]["environment"]))
    assert env["environment_hash"] == hashing.environment_hash(env)
    protocol = run.to_dict()["payload"]["protocol"]
    assert protocol["protocol_hash"] == hashing.protocol_hash(protocol)
    assert protocol["warmup"] == 1 and protocol["repetitions"] == 5
    verifier = run.to_dict()["payload"]["verifier"]
    assert verifier["verifier_hash"] == hashing.verifier_hash(verifier)
    config = demo_store.require("cfg-demo", "config")
    assert p.comparison_key == hashing.comparison_key(
        config_hash=config.payload.config_hash,
        environment_hash=env["environment_hash"],
        protocol_hash=protocol["protocol_hash"],
        verifier_hash=verifier["verifier_hash"],
        checkout_mode="exact_commit",
    )
    assert p.source.variant_digest == hashing.variant_digest(
        source_digest=p.source.source_digest,
        entrypoint=ENTRYPOINT,
        implementation_overrides={"chunk": 8},
        checkout_mode="exact_commit",
    )
    # Uncollected metric is null with a status, never 0.
    metrics = {m.name: m for m in p.analysis_metrics}
    assert set(metrics) == set(KNOWN_METRICS)
    assert metrics["register_spill_vmem_static_bytes"].status == "not_collected"
    assert metrics["register_spill_vmem_static_bytes"].value is None

    state = runner.ledger.state("request-ok")
    assert state.kind == "finished" and state.run_refs == [run.record_id]
    assert event_kinds(runner, "request-ok") == ["queued", "claimed", "running", "finished"]
    assert adapter.calls == {"check_environment": 2, "prepare": 1, "compile": 1, "verify": 1, "benchmark": 1}


def test_stage_verify_records_correctness_without_timing(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-verify", stage="verify"))
    run = runner.execute("request-verify")
    assert run.payload.stage == "verify"
    assert run.payload.execution_status == "succeeded"
    assert run.payload.correctness.status == "pass"
    assert run.payload.timing.status == "not_run" and run.payload.timing.median_us is None
    assert adapter.calls["benchmark"] == 0


def test_nanosecond_samples_are_converted_to_microseconds(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(samples=[1000.0, 2000.0, 3000.0], unit="nanoseconds")
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-ns"))
    run = runner.execute("request-ns")
    assert run.payload.timing.median_us == pytest.approx(2.0)
    doc = artifact_json(demo_store, run, run.payload.timing.samples_artifact_ref)
    assert doc["unit"] == "nanoseconds"
    assert stats.summarize(doc["samples"], doc["unit"]).p90_us == run.payload.timing.p90_us


# ------------------------------------------------------------------------------ failures are facts
def test_t14_compile_error_records_run_without_invented_numbers(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(compile_status="compile_error")
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-ce"))
    run = runner.execute("request-ce")
    p = run.payload
    assert p.execution_status == "compile_error"
    assert "synthetic compile failure" in (p.failure_reason or "")
    assert p.correctness.status == "not_run" and p.correctness.cases_total == 0
    assert p.correctness.max_abs_error is None and p.correctness.report_artifact_ref is None
    assert p.timing.status == "not_run" and p.timing.sample_count == 0
    assert p.timing.median_us is None and p.timing.p90_us is None and p.timing.samples_artifact_ref is None
    assert [a.kind for a in p.artifacts] == ["compile_log"]
    assert demo_store.read_artifact(p.artifacts[0].sha256).startswith(b"error:")
    assert all(m.value is None and m.status == "not_collected" for m in p.analysis_metrics)
    assert adapter.calls["verify"] == 0 and adapter.calls["benchmark"] == 0
    assert runner.ledger.state("request-ce").events[-1]["execution_status"] == "compile_error"


def test_t15_succeeded_with_wrong_output_is_recorded_as_fail(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(correctness="fail"))
    runner.ledger.submit(make_spec("request-wrong"))
    run = runner.execute("request-wrong")
    p = run.payload
    assert p.execution_status == "succeeded"
    assert p.correctness.status == "fail" and p.correctness.cases_passed == 0
    assert p.correctness.max_abs_error == 0.5
    assert p.timing.status == "recorded"  # execution and correctness are separate facts
    assert not p.is_terminal_success or p.correctness.status != "pass"


def test_verifier_error_status_is_recorded(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(correctness="error"))
    runner.ledger.submit(make_spec("request-verr"))
    run = runner.execute("request-verr")
    assert run.payload.execution_status == "succeeded"
    assert run.payload.correctness.status == "error"


def test_unexpected_exception_becomes_runtime_error(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(benchmark_error=RuntimeError("kernel exploded")))
    runner.ledger.submit(make_spec("request-rt"))
    run = runner.execute("request-rt")
    p = run.payload
    assert p.execution_status == "runtime_error"
    assert p.failure_reason == "RuntimeError: kernel exploded"
    assert p.correctness.status == "not_run" and p.timing.status == "not_run"
    assert p.artifacts == []  # result evidence of a failed execution is not kept as if it were valid


def test_kernel_memory_error_in_verify_becomes_runtime_error_with_code(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(verify_error=InputError("bad output shape", code="OUTPUT_SHAPE")))
    runner.ledger.submit(make_spec("request-kme"))
    run = runner.execute("request-kme")
    assert run.payload.execution_status == "runtime_error"
    assert run.payload.failure_reason == "OUTPUT_SHAPE: bad output shape"


def test_adapter_timeout_is_recorded_as_timeout(demo_store: MemoryStore) -> None:
    error = ExecutionInfrastructureError("benchmark exceeded 30s", code="TIMEOUT")
    runner = make_runner(demo_store, StubAdapter(benchmark_error=error))
    runner.ledger.submit(make_spec("request-to"))
    run = runner.execute("request-to")
    assert run.payload.execution_status == "timeout"
    assert run.payload.failure_reason == "benchmark exceeded 30s"
    assert run.payload.timing.status == "not_run"


def test_other_infrastructure_error_is_recorded(demo_store: MemoryStore) -> None:
    error = ExecutionInfrastructureError("device reset", code="DEVICE_RESET")
    runner = make_runner(demo_store, StubAdapter(benchmark_error=error))
    runner.ledger.submit(make_spec("request-infra"))
    run = runner.execute("request-infra")
    assert run.payload.execution_status == "infrastructure_error"
    assert run.payload.failure_reason == "DEVICE_RESET: device reset"


def test_wall_clock_over_max_wall_seconds_is_timeout(demo_store: MemoryStore) -> None:
    clock = FakeClock()
    adapter = StubAdapter(on_benchmark=lambda prepared: clock.advance(100))
    runner = make_runner(demo_store, adapter, clock=clock, lease_seconds=1000)
    runner.ledger.submit(make_spec("request-wall", max_wall_seconds=30))
    run = runner.execute("request-wall")
    assert run.payload.execution_status == "timeout"
    assert "max_wall_seconds" in (run.payload.failure_reason or "")
    assert run.payload.timing.status == "not_run" and run.payload.timing.median_us is None


# ------------------------------------------------------------------------------ idempotency / reruns
def test_t11_submit_and_execute_replays_without_second_execution(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    runner = make_runner(demo_store, adapter)
    spec = make_spec("request-replay", key="replay-key")
    first_outcome, first_run = runner.submit_and_execute(spec)
    assert first_outcome.created is True and first_run is not None
    runs_after_first = len(demo_store.records("run"))
    second_outcome, second_run = runner.submit_and_execute(make_spec("request-replay-2", key="replay-key"))
    assert second_outcome.created is False
    assert second_outcome.request_id == "request-replay"
    assert second_run is not None and second_run.record_id == first_run.record_id
    assert second_run.canonical_digest() == first_run.canonical_digest()
    assert adapter.calls["benchmark"] == 1 and adapter.calls["prepare"] == 1
    assert len(demo_store.records("run")) == runs_after_first
    assert event_kinds(runner, "request-replay").count("claimed") == 1
    # Executing a finished request directly is refused: a new measurement needs a new request.
    with pytest.raises(InputError) as excinfo:
        runner.execute("request-replay")
    assert excinfo.value.code == "REQUEST_FINISHED"
    assert adapter.calls["benchmark"] == 1


def test_t13_intentional_rerun_creates_new_run_and_leaves_old_unchanged(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-first"))
    first = runner.execute("request-first")
    first_digest = first.canonical_digest()
    first_path_bytes = demo_store.record_path(first.record_id).read_bytes()

    adapter.samples = [8.0, 8.5, 8.0, 9.0, 8.0]  # a genuinely new measurement
    runner.ledger.submit(make_spec("request-second", rerun_of=first.record_id))
    second = runner.execute("request-second")
    assert second.record_id == "run-request-second-a1"
    assert second.payload.rerun_of == first.record_id
    assert second.payload.timing.median_us != first.payload.timing.median_us
    demo_store.invalidate_index()
    assert demo_store.get(first.record_id).canonical_digest() == first_digest
    assert demo_store.record_path(first.record_id).read_bytes() == first_path_bytes
    assert demo_store.integrity_scan().ok
    assert adapter.calls["benchmark"] == 2


# ------------------------------------------------------------------------------ fencing (T26)
def test_t26_lease_lost_before_publish_quarantines_and_publishes_nothing(demo_store: MemoryStore) -> None:
    clock = FakeClock()
    runner_holder: dict[str, LocalRunner] = {}

    def rival_takes_over(prepared: PreparedExecution) -> None:
        clock.advance(20)  # our 10s lease expires while the benchmark is still running
        runner_holder["runner"].ledger.claim(prepared.request.request_id, "rival-worker", lease_seconds=10, now=clock)

    adapter = StubAdapter(on_benchmark=rival_takes_over)
    runner = make_runner(demo_store, adapter, clock=clock, lease_seconds=10)
    runner_holder["runner"] = runner
    runner.ledger.submit(make_spec("request-fence"))
    runs_before = len(demo_store.records("run"))
    artifacts_before = len(demo_store.artifact_registry())
    with pytest.raises(LeaseLostError) as excinfo:
        runner.execute("request-fence")
    assert excinfo.value.exit_code == 6
    assert excinfo.value.details["stale_fencing_token"] == 1
    assert excinfo.value.details["current_fencing_token"] == 2
    assert not demo_store.exists("run-request-fence-a1")
    assert len(demo_store.records("run")) == runs_before
    assert len(demo_store.artifact_registry()) == artifacts_before
    state = runner.ledger.state("request-fence")
    assert state.run_refs == []
    assert state.kind == "claimed" and state.worker_id == "rival-worker" and state.attempt_no == 2
    quarantined = [e for e in state.events if e["kind"] == "late_result_quarantined"]
    assert len(quarantined) == 1
    assert quarantined[0]["payload"]["quarantined_run_ref"] == "run-request-fence-a1"
    assert quarantined[0]["payload"]["record"]["record_id"] == "run-request-fence-a1"
    assert adapter.calls["benchmark"] == 1


# ------------------------------------------------------------------------------ unexecuted / refused
def test_backend_unavailable_records_event_and_no_run(demo_store: MemoryStore) -> None:
    error = BackendUnavailable("no TPU devices visible", details={"backend": "mock"})
    adapter = StubAdapter(env_error=error)
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-nobackend"))
    runs_before = len(demo_store.records("run"))
    with pytest.raises(BackendUnavailable) as excinfo:
        runner.execute("request-nobackend")
    assert excinfo.value.exit_code == 5
    state = runner.ledger.state("request-nobackend")
    assert event_kinds(runner, "request-nobackend") == ["queued", "backend_unavailable"]
    assert state.attempt_no == 0 and state.kind == "queued"  # never claimed: not an execution
    assert state.events[-1]["payload"]["error"] == "BACKEND_UNAVAILABLE"
    assert len(demo_store.records("run")) == runs_before
    assert adapter.calls["prepare"] == 0
    description = LocalRunner.describe_unavailable("mock", excinfo.value)
    assert description["status"] == "unexecuted" and description["error"] == "BACKEND_UNAVAILABLE"
    assert description["exit_code"] == 5


def test_unregistered_backend_is_unavailable(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=AdapterRegistry())
    runner.ledger.submit(make_spec("request-noadapter"))
    with pytest.raises(BackendUnavailable) as excinfo:
        runner.execute("request-noadapter")
    assert excinfo.value.details["backend"] == "mock"
    assert event_kinds(runner, "request-noadapter") == ["queued", "backend_unavailable"]


def test_backend_unavailable_during_prepare_releases_the_attempt(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(prepare_error=BackendUnavailable("device disappeared"))
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-late-unavailable"))
    with pytest.raises(BackendUnavailable):
        runner.execute("request-late-unavailable")
    state = runner.ledger.state("request-late-unavailable")
    assert state.kind == "lease_lost" and state.run_refs == []
    assert "backend_unavailable" in event_kinds(runner, "request-late-unavailable")
    assert not demo_store.exists("run-request-late-unavailable-a1")


def test_t29_tpu_request_without_authorization_is_denied_before_claim(demo_store: MemoryStore) -> None:
    class TpuStub(StubAdapter):
        backend = "jax_tpu"

    adapter = TpuStub()
    # Request does not ask for TPU authorization.
    runner = make_runner(demo_store, adapter, permissions={"allow_tpu_execution": True})
    runner.ledger.submit(make_spec("request-tpu-1", backend="jax_tpu"))
    with pytest.raises(AuthorizationError) as excinfo:
        runner.execute("request-tpu-1")
    assert excinfo.value.exit_code == 7 and excinfo.value.code == "TPU_EXECUTION_NOT_AUTHORIZED"
    assert runner.ledger.state("request-tpu-1").attempt_no == 0
    assert "claimed" not in event_kinds(runner, "request-tpu-1")
    assert adapter.calls["check_environment"] == 0
    # Request asks for it but the runner permissions do not grant it.
    runner2 = make_runner(demo_store, adapter, permissions={})
    runner2.ledger.submit(make_spec("request-tpu-2", backend="jax_tpu", authorization={"allow_tpu_execution": True}))
    with pytest.raises(AuthorizationError):
        runner2.execute("request-tpu-2")
    assert runner2.ledger.state("request-tpu-2").attempt_no == 0
    assert runner2.ledger.state("request-tpu-2").cancelled is True  # the denial is recorded, not skipped
    assert not any(r.payload.request_id.startswith("request-tpu") for r in demo_store.records("run"))
    # Both sides authorize: the stub executes (documenting that authorization is the only gate here).
    runner3 = make_runner(demo_store, adapter, permissions={"allow_tpu_execution": True})
    runner3.ledger.submit(make_spec("request-tpu-3", backend="jax_tpu", authorization={"allow_tpu_execution": True}))
    run = runner3.execute("request-tpu-3")
    assert run.payload.execution_status == "succeeded"


def test_local_cpu_tests_can_be_denied(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(), permissions={"allow_local_cpu_tests": False})
    runner.ledger.submit(make_spec("request-cpu-denied"))
    with pytest.raises(AuthorizationError) as excinfo:
        runner.execute("request-cpu-denied")
    assert excinfo.value.code == "LOCAL_CPU_TESTS_NOT_AUTHORIZED" and excinfo.value.exit_code == 7
    assert runner.ledger.state("request-cpu-denied").attempt_no == 0


def test_subject_target_mismatch_is_refused_before_claim(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-mismatch", target=OTHER_OID))
    with pytest.raises(InputError) as excinfo:
        runner.execute("request-mismatch")
    assert excinfo.value.code == "SUBJECT_SOURCE_MISMATCH" and excinfo.value.exit_code == 2
    assert excinfo.value.details["subject_commit"]["hex"] == BASELINE_OID.hex
    assert event_kinds(runner, "request-mismatch") == ["queued"]
    assert adapter.calls["check_environment"] == 0


def test_adapter_tested_source_mismatch_records_source_unavailable(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(tested_hex=OTHER_OID.hex)
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-tested"))
    run = runner.execute("request-tested")
    p = run.payload
    assert p.execution_status == "source_unavailable"
    assert p.source.tested_commit == OTHER_OID and p.source.target_commit == BASELINE_OID
    assert "differs from target" in (p.failure_reason or "")
    assert p.correctness.status == "not_run" and p.timing.status == "not_run"
    assert p.artifacts == []
    assert adapter.calls["compile"] == 0
    assert runner.ledger.state("request-tested").run_refs == [run.record_id]


def test_adapter_raising_tested_source_mismatch_records_source_unavailable(demo_store: MemoryStore) -> None:
    error = InvariantViolation(
        "checked-out tree is not the target commit",
        code="TESTED_SOURCE_MISMATCH",
        details={
            "tested_commit": {"algorithm": "sha1", "hex": OTHER_OID.hex},
            "source_digest": hashing.sha256_bytes(b"other tree"),
            "dirty": False,
        },
    )
    runner = make_runner(demo_store, StubAdapter(prepare_error=error))
    runner.ledger.submit(make_spec("request-raised"))
    run = runner.execute("request-raised")
    assert run.payload.execution_status == "source_unavailable"
    assert run.payload.source.tested_commit == OTHER_OID
    assert run.payload.source.source_digest == hashing.sha256_bytes(b"other tree")
    assert run.payload.timing.status == "not_run"


def test_dirty_source_without_exploratory_permission_is_source_unavailable(demo_store: MemoryStore) -> None:
    runner = make_runner(demo_store, StubAdapter(dirty=True))
    runner.ledger.submit(make_spec("request-dirty"))
    run = runner.execute("request-dirty")
    assert run.payload.execution_status == "source_unavailable"
    assert "dirty" in (run.payload.failure_reason or "")
    # With explicit exploratory permission the run is recorded honestly as dirty (never promotable).
    runner.ledger.submit(make_spec("request-dirty-ok", allow_dirty=True))
    run_ok = runner.execute("request-dirty-ok")
    assert run_ok.payload.execution_status == "succeeded"
    assert run_ok.payload.source.dirty is True and run_ok.payload.source.patch_digest is not None


def test_security_refusal_in_prepare_cancels_without_run(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(prepare_error=SecurityPolicyError("entrypoint escapes the source root", code="UNSAFE_ENTRYPOINT"))
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-sec"))
    with pytest.raises(SecurityPolicyError):
        runner.execute("request-sec")
    state = runner.ledger.state("request-sec")
    assert state.cancelled is True and state.run_refs == []
    assert state.events[-1]["payload"]["error"]["error"] == "UNSAFE_ENTRYPOINT"
    assert not demo_store.exists("run-request-sec-a1")


def test_incomplete_problem_contract_cancels_without_run(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(prepare_error=IncompleteProblemContract("mla_forward contract incomplete"))
    runner = make_runner(demo_store, adapter)
    runner.ledger.submit(make_spec("request-incomplete"))
    with pytest.raises(IncompleteProblemContract) as excinfo:
        runner.execute("request-incomplete")
    assert excinfo.value.exit_code == 5
    state = runner.ledger.state("request-incomplete")
    assert state.kind == "cancelled" and state.run_refs == []
    assert "backend_unavailable" not in event_kinds(runner, "request-incomplete")


def test_missing_subject_or_config_is_a_missing_reference(demo_store: MemoryStore) -> None:
    from kernel_memory.domain.errors import MissingReferenceError

    runner = make_runner(demo_store, StubAdapter())
    runner.ledger.submit(make_spec("request-nosubject", subject_ref="baseline-missing"))
    with pytest.raises(MissingReferenceError):
        runner.execute("request-nosubject")
    with pytest.raises(MissingReferenceError):
        runner.execute("request-never-submitted")


def test_runner_rejects_empty_worker_id(demo_store: MemoryStore) -> None:
    with pytest.raises(InputError):
        LocalRunner(demo_store, adapters=AdapterRegistry(), worker_id="")
