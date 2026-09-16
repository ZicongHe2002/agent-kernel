"""Optimization coordinator, MockPlanner, and the optimize service (T29, T30)."""
from __future__ import annotations

import json
from typing import Any

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
from kernel_memory.domain import hashing
from kernel_memory.domain.errors import BackendUnavailable, InputError, PrerequisiteMissingError
from kernel_memory.domain.models import GitOid
from kernel_memory.execution.budget import BudgetLedger
from kernel_memory.execution.coordinator import (
    OptimizationCoordinator,
    cancel_job,
    clear_cancel,
    golden_policy,
    load_job_rounds,
    resolve_policy,
    state_path,
)
from kernel_memory.execution.planner import (
    DEFAULT_OVERRIDES_SEQUENCE,
    MODEL_API_KEY_ENV,
    MODEL_PROVIDER_ENV,
    MockPlanner,
    UnavailableModelPlanner,
)
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.execution.types import Budget, BudgetUsage, MemoryContext, Proposal
from kernel_memory.services.optimize import run_optimization
from kernel_memory.storage import MemoryStore

BASELINE_OID = GitOid("sha1", "0000000000000000000000000000000000000001")
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


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubAdapter:
    """Minimal KernelAdapter for orchestration tests (not collected by pytest)."""

    __test__ = False
    adapter_id = "test-adapter-v1"
    backend = "mock"

    def __init__(self, *, compile_status: str = "ok", env_error: Exception | None = None, on_benchmark: Any = None) -> None:
        self.compile_status = compile_status
        self.env_error = env_error
        self.on_benchmark = on_benchmark
        self.calls: dict[str, int] = {"check_environment": 0, "prepare": 0, "compile": 0, "verify": 0, "benchmark": 0}
        self.overrides_seen: list[dict] = []

    def check_environment(self) -> dict:
        self.calls["check_environment"] += 1
        if self.env_error is not None:
            raise self.env_error
        return {
            "backend": "mock",
            "accelerator_model": "HOST-CPU-STUB",
            "device_count": 1,
            "topology": "single",
            "software": {"adapter": self.adapter_id},
            "execution_flags": {},
            "host_timer_environment": {},
            "unknown_required_fields": [],
        }

    def prepare(self, request: RunRequestSpec, problem: dict) -> PreparedExecution:
        self.calls["prepare"] += 1
        target = request.source.target_commit
        snapshot = SourceSnapshot(
            repo_uid=request.source.repo_uid,
            target_commit=target,
            tested_commit=target,
            tested_tree=None,
            checkout_mode=request.source.checkout_mode,
            merge_parent_oids=[],
            dirty=False,
            patch_digest=None,
            source_digest=hashing.source_digest([("demo/reference.py", hashing.sha256_bytes(b"def vector_add"))]),
            entrypoint=request.source.entrypoint,
            implementation_overrides=dict(request.source.implementation_overrides),
        )
        return PreparedExecution(request, dict(problem), snapshot, self.check_environment(), hashing.jcs_digest({"problem": problem}))

    def compile(self, prepared: PreparedExecution) -> CompileReport:
        self.calls["compile"] += 1
        if self.compile_status == "compile_error":
            return CompileReport("compile_error", "synthetic compile failure", [ArtifactBlob("compile-log", "compile_log", "text/plain", b"error\n")])
        return CompileReport("ok")

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        self.calls["verify"] += 1
        return CorrectnessReport("pass", 1, 1, 0.0, 0.0)

    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        self.calls["benchmark"] += 1
        self.overrides_seen.append(dict(prepared.source.implementation_overrides))
        if self.on_benchmark is not None:
            self.on_benchmark(prepared)
        chunk = prepared.source.implementation_overrides.get("chunk", 1)
        base = 10.0 + (chunk % 7)  # deterministic, obviously synthetic samples
        return TimingReport("recorded", "microseconds", [base, base + 1.0, base, base - 0.5, base + 0.5])


def registry_with(adapter: Any) -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register_kernel_adapter(adapter)
    return registry


def make_factory(job: str):
    def factory(proposal: Proposal, round_no: int, subject_ref: str) -> RunRequestSpec:
        overrides = dict(proposal.implementation_overrides)
        return RunRequestSpec(
            request_id=f"request-{job}-r{round_no}",
            idempotency_key=hashing.jcs_digest({"job": job, "round": round_no, "overrides": overrides}),
            subject_ref=subject_ref,
            config_ref="cfg-demo",
            backend="mock",
            stage="benchmark",
            protocol=dict(PROTOCOL),
            verifier=dict(VERIFIER),
            source=SourceSpec(repo_uid=REPO_UID, target_commit=BASELINE_OID, entrypoint=ENTRYPOINT, implementation_overrides=overrides),
            session_id=job,
            pair_id=f"{job}-r{round_no}",
            role_in_pair="candidate",
        )

    return factory


class Collab:
    """Recording stubs for compare/decide/context so tests never import the in-flux services."""

    def __init__(self, decision_outcome: str = "accepted") -> None:
        self.compare_calls: list[tuple[str, str]] = []
        self.decide_calls: list[dict] = []
        self.context_calls = 0
        self.decision_outcome = decision_outcome

    def comparator(self, store: MemoryStore, candidate: str, baseline: str) -> dict:
        self.compare_calls.append((candidate, baseline))
        return {"result": "COMPARABLE", "candidate_run_ref": candidate, "baseline_run_ref": baseline, "speedup": None}

    def decider(self, store: MemoryStore, **kwargs: Any) -> dict:
        self.decide_calls.append(kwargs)
        return {"record_id": f"decision-stub-{len(self.decide_calls)}", "outcome": self.decision_outcome}

    def context_exporter(self, store: MemoryStore, config_ref: str) -> dict:
        self.context_calls += 1
        return {"config_ref": config_ref, "current_baselines": [{"baseline_ref": "baseline-demo"}], "history": []}


def failing_decider(store: MemoryStore, **kwargs: Any) -> Any:
    raise AssertionError("decider must not be invoked without enough pairs")


def make_coordinator(store: MemoryStore, adapter: Any, *, job: str, budget: Budget | None = None, planner: Any = None, collab: Collab | None = None, **kwargs: Any) -> OptimizationCoordinator:
    collab = collab or Collab()
    runner = kwargs.pop("runner", None) or LocalRunner(store, adapters=registry_with(adapter))
    return OptimizationCoordinator(
        store,
        runner=runner,
        planner=planner or MockPlanner(),
        budget=budget or Budget(),
        policy=kwargs.pop("policy", None),
        job_id=job,
        config_ref="cfg-demo",
        permissions=kwargs.pop("permissions", {}),
        request_factory=make_factory(job),
        baseline_run_ref=kwargs.pop("baseline_run_ref", "run-demo-baseline"),
        comparator=kwargs.pop("comparator", collab.comparator),
        decider=kwargs.pop("decider", failing_decider),
        context_exporter=kwargs.pop("context_exporter", collab.context_exporter),
        **kwargs,
    )


# ------------------------------------------------------------------------------ planner
def test_mock_planner_default_sequence_and_stop_conditions() -> None:
    planner = MockPlanner()
    assert planner.planner_id == "mock-v1" and planner.requires_model_api is False
    ctx = MemoryContext("cfg-demo", "sha256:" + "0" * 64, {"current_baselines": [{"baseline_ref": "baseline-demo"}]}, round_no=1)
    proposals = []
    for round_no in (1, 2, 3):
        ctx.round_no = round_no
        proposal = planner.propose(ctx, Budget(), BudgetUsage())
        assert proposal is not None
        proposals.append(proposal)
    assert [p.implementation_overrides for p in proposals] == list(DEFAULT_OVERRIDES_SEQUENCE)
    first = proposals[0]
    assert first.parent_ref == "baseline-demo"
    assert first.target_component == "implementation_overrides"
    assert first.planned_changes == [{"component": "implementation_overrides", "key": "chunk", "before": None, "after": 8, "rationale": "mock planner sweep"}]
    assert first.predicted_speedup is None
    assert first.file_allowlist == [] and first.stop_conditions and first.hypothesis
    assert first.planner_id == "mock-v1"
    ctx.round_no = 4
    assert planner.propose(ctx, Budget(), BudgetUsage()) is None  # sequence exhausted
    ctx.round_no = 1
    assert planner.propose(ctx, Budget(plateau_rounds=2), BudgetUsage(rounds_without_improvement=2)) is None  # plateau


def test_mock_planner_parent_ref_fallback_and_missing_context() -> None:
    planner = MockPlanner(overrides_sequence=[{"chunk": 3}])
    ctx = MemoryContext("cfg-demo", "sha256:" + "0" * 64, {"default_parent_ref": "commit-demo-a"}, round_no=1)
    proposal = planner.propose(ctx, Budget(), BudgetUsage())
    assert proposal is not None and proposal.parent_ref == "commit-demo-a"
    with pytest.raises(InputError) as excinfo:
        planner.propose(MemoryContext("cfg-demo", "sha256:" + "0" * 64, {}, round_no=1), Budget(), BudgetUsage())
    assert excinfo.value.code == "PLANNER_CONTEXT_INCOMPLETE"
    with pytest.raises(InputError):
        MockPlanner(overrides_sequence=[{}])


def test_t29_unavailable_model_planner_names_missing_prerequisites() -> None:
    planner = UnavailableModelPlanner()
    assert planner.requires_model_api is True
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        planner.propose(MemoryContext("cfg-demo", "sha256:" + "0" * 64, {}, round_no=1), Budget(), BudgetUsage())
    err = excinfo.value
    assert err.code == "MODEL_PROVIDER_NOT_CONFIGURED" and err.exit_code == 5
    assert set(err.details["environment_variables"]) == {MODEL_PROVIDER_ENV, MODEL_API_KEY_ENV}
    assert err.details["authorization"] == "allow_model_api_calls"


def test_resolve_policy_defaults_and_golden_vector() -> None:
    golden = golden_policy()
    assert golden["policy_id"] == "default-confirm-v1" and golden["min_confirm_pairs"] == 3
    assert hashing.policy_hash(golden) == "sha256:4c214e453dadb52cf291a03a05079c5e9d0f4c501a67056c4f484b97cf3127a2"
    assert resolve_policy({"min_confirm_pairs": 1}) == {"min_confirm_pairs": 1}
    default = resolve_policy(None)
    assert default["min_confirm_pairs"] >= 1 and "policy_id" in default


# ------------------------------------------------------------------------------ loop
def test_mock_planner_completes_three_rounds_with_persistence(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    collab = Collab()
    coordinator = make_coordinator(demo_store, adapter, job="job-three", collab=collab)
    report = coordinator.run()
    d = report.to_dict()

    assert d["status"] == "completed" and d["stop_reason"] == "PLANNER_EXHAUSTED"
    assert d["planner_id"] == "mock-v1" and d["dry_run"] is False and d["resumed_rounds"] == 0
    assert [r["round_no"] for r in d["rounds"]] == [1, 2, 3]
    assert [r["proposal"]["implementation_overrides"] for r in d["rounds"]] == [{"chunk": 8}, {"chunk": 64}, {"chunk": 512}]
    assert [r["request_id"] for r in d["rounds"]] == ["request-job-three-r1", "request-job-three-r2", "request-job-three-r3"]
    assert [r["run_ref"] for r in d["rounds"]] == ["run-request-job-three-r1-a1", "run-request-job-three-r2-a1", "run-request-job-three-r3-a1"]
    assert all(r["execution_status"] == "succeeded" for r in d["rounds"])
    assert all(r["comparison"]["result"] == "COMPARABLE" for r in d["rounds"])
    assert all(r["decision_ref"] is None and r["improved"] is False for r in d["rounds"])  # variants differ: no pairs
    assert d["usage"]["candidates"] == 3 and d["usage"]["execution_attempts"] == 3
    assert d["usage"]["consecutive_execution_failures"] == 0 and d["usage"]["rounds_without_improvement"] == 3
    assert d["usage"]["model_calls"] == 0
    assert adapter.calls["benchmark"] == 3
    assert adapter.overrides_seen == [{"chunk": 8}, {"chunk": 64}, {"chunk": 512}]
    assert collab.compare_calls == [(r["run_ref"], "run-demo-baseline") for r in d["rounds"]]
    assert collab.context_calls == 4  # one export per proposal attempt, including the exhausting one

    # Runs are real trusted_worker records carrying the proposal's overrides.
    for r in d["rounds"]:
        run = demo_store.require(r["run_ref"], "run")
        assert run.payload.provenance == "trusted_worker"
        assert run.payload.source.implementation_overrides == r["proposal"]["implementation_overrides"]
        assert run.payload.session_id == "job-three"
        annotation = demo_store.require(r["annotation_ref"], "annotation")
        assert annotation.payload.author_kind == "program" and annotation.payload.category == "note"
        assert annotation.payload.confidence == "unverified"
        assert annotation.payload.evidence_refs == [r["run_ref"], "run-demo-baseline"]
        assert "not a confirmed improvement" in annotation.payload.text

    # Persistence: rounds under requests/_jobs, runtime state, budget usage files.
    persisted = load_job_rounds(demo_store, "job-three")
    assert [p["round_no"] for p in persisted] == [1, 2, 3]
    assert persisted[0]["job_id"] == "job-three" and persisted[0]["run_ref"] == "run-request-job-three-r1-a1"
    assert d["persisted_state_path"] == state_path("job-three")
    state = json.loads(demo_store.read_fact(state_path("job-three")))
    assert state["status"] == "completed" and state["last_round"] == 3 and state["usage"]["candidates"] == 3
    assert BudgetLedger(demo_store, "job-three", Budget()).usage().to_dict() == d["usage"]


def test_t30_execution_budget_stops_finitely(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    coordinator = make_coordinator(demo_store, adapter, job="job-budget", budget=Budget(max_execution_attempts=2))
    report = coordinator.run()
    assert report.status == "stopped"
    assert report.stop_reason == "BUDGET_EXECUTIONS_EXHAUSTED"
    assert len(report.rounds) == 2
    assert adapter.calls["benchmark"] == 2
    assert report.usage["execution_attempts"] == 2 and report.usage["candidates"] == 2
    assert len(load_job_rounds(demo_store, "job-budget")) == 2
    assert json.loads(demo_store.read_fact(state_path("job-budget")))["stop_reason"] == "BUDGET_EXECUTIONS_EXHAUSTED"


def test_t30_restart_restores_usage_and_continues_without_double_counting(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    first = make_coordinator(demo_store, adapter, job="job-resume", budget=Budget(max_execution_attempts=2)).run()
    assert first.stop_reason == "BUDGET_EXECUTIONS_EXHAUSTED" and len(first.rounds) == 2

    # Same budget: the resumed job stops immediately, accounting untouched, no execution.
    again = make_coordinator(demo_store, adapter, job="job-resume", budget=Budget(max_execution_attempts=2)).run()
    assert again.status == "stopped" and again.stop_reason == "BUDGET_EXECUTIONS_EXHAUSTED"
    assert again.rounds == [] and again.resumed_rounds == 2
    assert again.usage["execution_attempts"] == 2 and again.usage["candidates"] == 2
    assert adapter.calls["benchmark"] == 2

    # Raised budget: rounds continue from 3, usage is restored (not reset), nothing is counted twice.
    resumed = make_coordinator(demo_store, adapter, job="job-resume", budget=Budget(max_execution_attempts=24)).run()
    assert resumed.resumed_rounds == 2
    assert [r.round_no for r in resumed.rounds] == [3]
    assert resumed.rounds[0].proposal["implementation_overrides"] == {"chunk": 512}
    assert resumed.rounds[0].run_ref == "run-request-job-resume-r3-a1"
    assert resumed.status == "completed" and resumed.stop_reason == "PLANNER_EXHAUSTED"
    assert resumed.usage["execution_attempts"] == 3 and resumed.usage["candidates"] == 3
    assert adapter.calls["benchmark"] == 3
    assert [p["round_no"] for p in load_job_rounds(demo_store, "job-resume")] == [1, 2, 3]
    assert demo_store.exists("run-request-job-resume-r1-a1") and demo_store.exists("run-request-job-resume-r3-a1")


def test_cancel_flag_between_rounds_stops_with_cancelled(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()

    class CancellingPlanner(MockPlanner):
        def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
            proposal = super().propose(context, budget, usage)
            if context.round_no == 2:
                cancel_job(demo_store, "job-cancel", reason="operator stop")
            return proposal

    report = make_coordinator(demo_store, adapter, job="job-cancel", planner=CancellingPlanner()).run()
    assert report.status == "cancelled" and report.stop_reason == "USER_CANCELLED"
    assert [r.round_no for r in report.rounds] == [1, 2]  # round 2 finishes; the flag is honoured before round 3
    assert adapter.calls["benchmark"] == 2
    assert json.loads(demo_store.read_fact(state_path("job-cancel")))["status"] == "cancelled"
    # Clearing the flag and restarting resumes from round 3 with restored accounting.
    clear_cancel(demo_store, "job-cancel")
    resumed = make_coordinator(demo_store, adapter, job="job-cancel").run()
    assert [r.round_no for r in resumed.rounds] == [3]
    assert resumed.usage["candidates"] == 3 and resumed.usage["execution_attempts"] == 3
    assert adapter.calls["benchmark"] == 3


def test_code_write_proposal_is_blocked_without_permission(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()

    class CodePlanner:
        planner_id = "code-stub"
        requires_model_api = False

        def __init__(self) -> None:
            self.calls = 0

        def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
            self.calls += 1
            if self.calls > 1:
                return None
            return Proposal(
                proposal_id="proposal-code-1",
                parent_ref="baseline-demo",
                target_component="tiling",
                hypothesis="rewrite the tiling loop",
                planned_changes=[{"component": "tiling", "key": "block_q", "before": 128, "after": 256, "rationale": "stub"}],
                file_allowlist=["kernels/tiling.py"],
                planner_id="code-stub",
            )

    report = make_coordinator(demo_store, adapter, job="job-code", planner=CodePlanner()).run()
    assert report.status == "completed" and report.stop_reason == "PLANNER_EXHAUSTED"
    assert len(report.rounds) == 1
    blocked = report.rounds[0]
    assert blocked.outcome_note == "blocked: candidate code write not authorized"
    assert blocked.error["error"] == "CANDIDATE_CODE_WRITE_NOT_AUTHORIZED" and blocked.error["exit_code"] == 7
    assert blocked.request_id is None and blocked.run_ref is None
    assert adapter.calls["prepare"] == 0 and adapter.calls["benchmark"] == 0
    assert report.usage["candidates"] == 0 and report.usage["execution_attempts"] == 0
    assert report.usage["rounds_without_improvement"] == 1
    assert load_job_rounds(demo_store, "job-code")[0]["error"]["error"] == "CANDIDATE_CODE_WRITE_NOT_AUTHORIZED"
    # Granting the permission still cannot execute code writes in this coordinator (no isolated workspace).
    granted = make_coordinator(demo_store, adapter, job="job-code-2", planner=CodePlanner(), permissions={"allow_candidate_code_write": True}).run()
    assert granted.rounds[0].error["error"] == "CANDIDATE_CODE_WRITE_UNSUPPORTED"
    assert adapter.calls["prepare"] == 0


def test_override_proposal_with_file_allowlist_is_rejected(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()

    class LeakyPlanner(MockPlanner):
        def propose(self, context: MemoryContext, budget: Budget, usage: BudgetUsage) -> Proposal | None:
            proposal = super().propose(context, budget, usage)
            if proposal is None:
                return None
            import dataclasses

            return dataclasses.replace(proposal, file_allowlist=["kernels/anything.py"])

    report = make_coordinator(demo_store, adapter, job="job-leaky", planner=LeakyPlanner(overrides_sequence=[{"chunk": 8}])).run()
    assert len(report.rounds) == 1 and report.rounds[0].error["error"] == "PROPOSAL_INVALID"
    assert adapter.calls["prepare"] == 0


def test_dry_run_writes_nothing_to_the_store(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    facts_before = demo_store.list_facts("requests")
    runs_before = len(demo_store.records("run"))
    report = make_coordinator(demo_store, adapter, job="job-dry", dry_run=True).run()
    assert report.status == "dry_run" and report.stop_reason == "PLANNER_EXHAUSTED"
    assert len(report.rounds) == 3
    assert all(r.request_id is None and r.run_ref is None for r in report.rounds)
    assert [r.proposal["implementation_overrides"] for r in report.rounds] == [{"chunk": 8}, {"chunk": 64}, {"chunk": 512}]
    assert report.usage["candidates"] == 3 and report.usage["execution_attempts"] == 0
    assert report.persisted_state_path is None
    assert demo_store.list_facts("requests") == facts_before
    assert demo_store.read_fact(state_path("job-dry")) is None
    assert len(demo_store.records("run")) == runs_before
    assert adapter.calls == {"check_environment": 0, "prepare": 0, "compile": 0, "verify": 0, "benchmark": 0}
    assert BudgetLedger(demo_store, "job-dry", Budget()).usage() == BudgetUsage()


def test_decider_invoked_when_pairs_suffice(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    collab = Collab(decision_outcome="accepted")
    policy = dict(golden_policy(), min_confirm_pairs=1)
    report = make_coordinator(
        demo_store,
        adapter,
        job="job-decide",
        collab=collab,
        decider=collab.decider,
        policy=policy,
        planner=MockPlanner(overrides_sequence=[{"chunk": 8}, {"chunk": 64}]),
    ).run()
    assert len(report.rounds) == 2
    assert [r.decision_ref for r in report.rounds] == ["decision-stub-1", "decision-stub-2"]
    assert all(r.improved for r in report.rounds)
    assert report.usage["rounds_without_improvement"] == 0
    assert len(collab.decide_calls) == 2
    call = collab.decide_calls[0]
    assert call["config_ref"] == "cfg-demo" and call["candidate_subject_ref"] == "baseline-demo"
    assert call["candidate_run_refs"] == ["run-request-job-decide-r1-a1"]
    assert call["baseline_run_refs"] == ["run-demo-baseline"]
    assert call["policy"]["min_confirm_pairs"] == 1
    run = demo_store.require("run-request-job-decide-r1-a1", "run")
    assert call["comparison_key"] == run.payload.comparison_key
    # A non-accepted outcome is not an improvement.
    rejected = Collab(decision_outcome="rejected")
    report2 = make_coordinator(demo_store, adapter, job="job-decide-2", collab=rejected, decider=rejected.decider, policy=policy, planner=MockPlanner(overrides_sequence=[{"chunk": 8}])).run()
    assert report2.rounds[0].decision_ref == "decision-stub-1" and report2.rounds[0].improved is False


def test_default_policy_needs_three_pairs_so_decider_is_not_called(demo_store: MemoryStore) -> None:
    # Three rounds of the *same* variant: pairs accumulate 1, 2, 3 -> the decider runs only on the third round.
    adapter = StubAdapter()
    collab = Collab(decision_outcome="inconclusive")
    planner = MockPlanner(overrides_sequence=[{"chunk": 8}, {"chunk": 8}, {"chunk": 8}])
    report = make_coordinator(demo_store, adapter, job="job-pairs", collab=collab, decider=collab.decider, planner=planner).run()
    assert [r.decision_ref for r in report.rounds] == [None, None, "decision-stub-1"]
    assert sorted(collab.decide_calls[0]["candidate_run_refs"]) == [
        "run-request-job-pairs-r1-a1",
        "run-request-job-pairs-r2-a1",
        "run-request-job-pairs-r3-a1",
    ]
    assert "pairs=1/3" in report.rounds[0].outcome_note


def test_backend_unavailable_stops_the_job(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(env_error=BackendUnavailable("no devices", details={"backend": "mock"}))
    report = make_coordinator(demo_store, adapter, job="job-nobackend").run()
    assert report.status == "stopped" and report.stop_reason == "BACKEND_UNAVAILABLE"
    assert len(report.rounds) == 1
    assert report.rounds[0].error["error"] == "BACKEND_UNAVAILABLE" and report.rounds[0].run_ref is None
    assert report.usage["execution_attempts"] == 1  # reserved before the attempt; the refusal is recorded
    assert adapter.calls["prepare"] == 0
    assert not demo_store.exists("run-request-job-nobackend-r1-a1")


def test_consecutive_execution_failures_stop_the_job(demo_store: MemoryStore) -> None:
    adapter = StubAdapter(compile_status="compile_error")
    report = make_coordinator(demo_store, adapter, job="job-failures", budget=Budget(max_consecutive_execution_failures=2)).run()
    assert report.status == "stopped" and report.stop_reason == "CONSECUTIVE_EXECUTION_FAILURES"
    assert len(report.rounds) == 2
    assert all(r.execution_status == "compile_error" for r in report.rounds)
    assert report.usage["consecutive_execution_failures"] == 2
    assert adapter.calls["compile"] == 2 and adapter.calls["verify"] == 0


def test_lease_lost_round_is_noted_and_the_loop_continues(demo_store: MemoryStore) -> None:
    clock = FakeClock()
    holder: dict[str, LocalRunner] = {}

    def rival_once(prepared: PreparedExecution) -> None:
        if prepared.request.request_id.endswith("-r1"):
            clock.advance(20)
            holder["runner"].ledger.claim(prepared.request.request_id, "rival", lease_seconds=10, now=clock)

    adapter = StubAdapter(on_benchmark=rival_once)
    runner = LocalRunner(demo_store, adapters=registry_with(adapter), clock=clock, lease_seconds=10)
    holder["runner"] = runner
    report = make_coordinator(demo_store, adapter, job="job-lease", runner=runner).run()
    assert report.status == "completed"
    assert report.rounds[0].run_ref is None and report.rounds[0].error["error"] == "LEASE_LOST"
    assert "quarantined" in report.rounds[0].outcome_note
    assert [r.run_ref for r in report.rounds[1:]] == ["run-request-job-lease-r2-a1", "run-request-job-lease-r3-a1"]
    assert not demo_store.exists("run-request-job-lease-r1-a1")


def test_model_planner_without_authorization_stops_finitely(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    report = make_coordinator(demo_store, adapter, job="job-model", planner=UnavailableModelPlanner()).run()
    assert report.status == "stopped" and report.stop_reason == "MODEL_API_NOT_AUTHORIZED"
    assert report.rounds == [] and report.usage["model_calls"] == 0
    authorized = make_coordinator(demo_store, adapter, job="job-model-2", planner=UnavailableModelPlanner(), permissions={"allow_model_api_calls": True}).run()
    assert authorized.status == "stopped" and authorized.stop_reason == "MODEL_PROVIDER_NOT_CONFIGURED"
    assert authorized.usage["model_calls"] == 1 and authorized.rounds == []
    assert adapter.calls["prepare"] == 0


def test_context_exporter_failure_falls_back_to_minimal_context(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()

    def broken_exporter(store: MemoryStore, config_ref: str) -> dict:
        raise PrerequisiteMissingError("context service missing")

    report = make_coordinator(demo_store, adapter, job="job-ctx", context_exporter=broken_exporter, planner=MockPlanner(overrides_sequence=[{"chunk": 8}])).run()
    assert report.status == "completed" and len(report.rounds) == 1
    assert report.rounds[0].proposal["parent_ref"] == "baseline-demo"


def test_coordinator_validates_inputs(demo_store: MemoryStore) -> None:
    runner = LocalRunner(demo_store, adapters=registry_with(StubAdapter()))
    with pytest.raises(InputError):
        OptimizationCoordinator(demo_store, runner=runner, planner=object(), budget=Budget(), policy=None, job_id="j", config_ref="cfg-demo", permissions={}, request_factory=make_factory("j"), baseline_run_ref=None)
    with pytest.raises(InputError):
        OptimizationCoordinator(demo_store, runner=runner, planner=MockPlanner(), budget=Budget(), policy=None, job_id="", config_ref="cfg-demo", permissions={}, request_factory=make_factory("j"), baseline_run_ref=None)


# ------------------------------------------------------------------------------ optimize service
def test_run_optimization_service_with_mock_planner(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    collab = Collab()
    report = run_optimization(
        demo_store,
        "cfg-demo",
        planner_name="mock",
        budget=Budget(max_execution_attempts=5),
        policy=None,
        dry_run=False,
        permissions={},
        adapters=registry_with(adapter),
        job_id="job-service",
        backend="mock",
        protocol=PROTOCOL,
        verifier=VERIFIER,
        subject_ref="baseline-demo",
        baseline_run_ref="run-demo-baseline",
        repo_uid=REPO_UID,
        entrypoint=ENTRYPOINT,
        target_commit={"algorithm": "sha1", "hex": BASELINE_OID.hex},
        comparator=collab.comparator,
        decider=collab.decider,
        context_exporter=collab.context_exporter,
    )
    assert report["status"] == "completed" and report["stop_reason"] == "PLANNER_EXHAUSTED"
    assert report["note"] == "MockPlanner exercises orchestration only; not evidence of optimization effectiveness"
    assert report["backend"] == "mock" and report["policy"]["min_confirm_pairs"] >= 1
    assert [r["request_id"] for r in report["rounds"]] == ["request-job-service-r1", "request-job-service-r2", "request-job-service-r3"]
    assert adapter.calls["benchmark"] == 3
    run = demo_store.require("run-request-job-service-r1-a1", "run")
    assert run.payload.session_id == "job-service" and run.payload.role_in_pair == "candidate"
    assert run.payload.source.implementation_overrides == {"chunk": 8}
    # Protocol and verifier are fixed for the whole job.
    hashes = {(demo_store.require(r["run_ref"], "run").payload.protocol.protocol_hash, demo_store.require(r["run_ref"], "run").payload.verifier.verifier_hash) for r in report["rounds"]}
    assert len(hashes) == 1


def test_run_optimization_dry_run_and_unknown_planner(demo_store: MemoryStore) -> None:
    adapter = StubAdapter()
    facts_before = demo_store.list_facts("requests")
    report = run_optimization(
        demo_store,
        "cfg-demo",
        planner_name="mock",
        dry_run=True,
        permissions={},
        adapters=registry_with(adapter),
        backend="mock",
        protocol=PROTOCOL,
        verifier=VERIFIER,
        subject_ref="baseline-demo",
        baseline_run_ref=None,
        repo_uid=REPO_UID,
        entrypoint=ENTRYPOINT,
        target_commit=BASELINE_OID,
        context_exporter=lambda store, config_ref: None,
    )
    assert report["status"] == "dry_run" and len(report["rounds"]) == 3
    assert demo_store.list_facts("requests") == facts_before
    assert adapter.calls["prepare"] == 0
    with pytest.raises(InputError) as excinfo:
        run_optimization(
            demo_store,
            "cfg-demo",
            planner_name="oracle",
            permissions={},
            backend="mock",
            protocol=PROTOCOL,
            verifier=VERIFIER,
            subject_ref="baseline-demo",
            repo_uid=REPO_UID,
            entrypoint=ENTRYPOINT,
            target_commit=BASELINE_OID,
        )
    assert excinfo.value.code == "UNKNOWN_PLANNER"


def test_t29_run_optimization_model_planner_is_explicitly_unexecuted(demo_store: MemoryStore) -> None:
    report = run_optimization(
        demo_store,
        "cfg-demo",
        planner_name="model",
        permissions={"allow_model_api_calls": True},
        adapters=registry_with(StubAdapter()),
        job_id="job-model-service",
        backend="mock",
        protocol=PROTOCOL,
        verifier=VERIFIER,
        subject_ref="baseline-demo",
        repo_uid=REPO_UID,
        entrypoint=ENTRYPOINT,
        target_commit=BASELINE_OID,
        context_exporter=lambda store, config_ref: None,
    )
    assert report["status"] == "stopped" and report["stop_reason"] == "MODEL_PROVIDER_NOT_CONFIGURED"
    assert report["rounds"] == [] and report["usage"]["model_calls"] == 1
    assert any("planner unavailable" in note for note in report["notes"])


def test_run_optimization_without_registered_backend_reports_unavailable(demo_store: MemoryStore) -> None:
    report = run_optimization(
        demo_store,
        "cfg-demo",
        planner_name="mock",
        permissions={},
        adapters=AdapterRegistry(),
        job_id="job-noadapter",
        backend="mock",
        protocol=PROTOCOL,
        verifier=VERIFIER,
        subject_ref="baseline-demo",
        repo_uid=REPO_UID,
        entrypoint=ENTRYPOINT,
        target_commit=BASELINE_OID,
        context_exporter=lambda store, config_ref: None,
    )
    assert report["status"] == "stopped" and report["stop_reason"] == "BACKEND_UNAVAILABLE"
    assert len(report["rounds"]) == 1 and report["rounds"][0]["run_ref"] is None
