"""Analysis adapters: observations are not explanations; uncollected values are null (T16, T17)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kernel_memory.adapters import analysis
from kernel_memory.adapters.analysis import (
    KNOWN_METRICS,
    LloAnalysisAdapter,
    MockSpillAnalysisAdapter,
    analyze_artifacts,
    metric_dict,
)
from kernel_memory.adapters.base import (
    AdapterRegistry,
    AnalysisReport,
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
from kernel_memory.domain.errors import InputError, UnsupportedFormat
from kernel_memory.domain.models import ArtifactRef, GitOid
from kernel_memory.domain.schema import validate_nested
from kernel_memory.execution import runner as runner_module
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore

METRIC = "register_spill_vmem_static_bytes"
FIXTURE_BLOB_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "handoff" / "examples" / "artifacts" / "mock-spill-report.json"
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
SAMPLES = [10.0, 11.0, 10.0, 9.0, 12.0]


@pytest.fixture(scope="module")
def fixture_bytes() -> bytes:
    return FIXTURE_BLOB_PATH.read_bytes()


def make_ref(artifact_id: str, kind: str, data: bytes, media_type: str = "application/json") -> ArtifactRef:
    digest = hashing.artifact_digest(data)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=f"artifact://sha256/{digest[len('sha256:'):]}",
        sha256=digest,
        size_bytes=len(data),
        media_type=media_type,
        retention="permanent",
        availability="present",
    )


def spill_payload(**fields: Any) -> bytes:
    payload: dict[str, Any] = {"format": "mock-spill-v1"}
    payload.update(fields)
    return json.dumps(payload).encode("utf-8")


def both_adapters() -> list[Any]:
    return [MockSpillAnalysisAdapter(), LloAnalysisAdapter()]


def only_metric(report: AnalysisReport) -> dict[str, Any]:
    assert [m["name"] for m in report.metrics] == [METRIC]
    validate_nested("Metric", report.metrics[0])
    return report.metrics[0]


# ------------------------------------------------------------------------------ declarations
def test_known_metrics_match_the_runner_declaration() -> None:
    assert tuple(KNOWN_METRICS) == runner_module.KNOWN_METRICS
    spec = KNOWN_METRICS[METRIC]
    assert (spec["unit"], spec["kind"], spec["scope"]) == ("bytes", "static_estimate", "compiled_kernel")
    assert "static estimate" in spec["definition"] and "not runtime HBM traffic" in spec["definition"]


def test_metric_dict_builds_schema_valid_metrics_and_refuses_invented_values() -> None:
    observed = metric_dict(METRIC, status="observed", value=12, source_artifact_ref="run-x-a1-spill", parser_id="p", parser_version="1")
    validate_nested("Metric", observed)
    assert observed["value"] == 12 and observed["kind"] == "static_estimate"
    missing = metric_dict(METRIC, status="not_collected", note="Not collected: nothing.")
    validate_nested("Metric", missing)
    assert missing["value"] is None and missing["definition"].endswith("Not collected: nothing.")
    with pytest.raises(InputError):
        metric_dict(METRIC, status="not_collected", value=0)  # an uncollected value is never a number
    with pytest.raises(InputError):
        metric_dict(METRIC, status="observed", value=None, source_artifact_ref="run-x-a1-spill")
    with pytest.raises(InputError):
        metric_dict(METRIC, status="observed", value=3)  # observed needs evidence
    with pytest.raises(InputError):
        metric_dict(METRIC, status="observed", value=True, source_artifact_ref="run-x-a1-spill")
    with pytest.raises(InputError):
        metric_dict("made_up_metric", status="not_collected")
    with pytest.raises(InputError):
        metric_dict(METRIC, status="bogus")


# ------------------------------------------------------------------------------ T16 / T17
def test_t16_fixture_spill_report_is_observed_with_traceable_evidence(fixture_bytes: bytes) -> None:
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", fixture_bytes)
    report = analyze_artifacts(both_adapters(), [(ref, fixture_bytes)])
    metric = only_metric(report)
    assert metric["status"] == "observed" and metric["value"] == 65536
    assert metric["unit"] == "bytes" and metric["kind"] == "static_estimate" and metric["scope"] == "compiled_kernel"
    assert metric["parser_id"] == "mock-spill" and metric["parser_version"] == "1"
    assert metric["source_artifact_ref"] == ref.artifact_id
    assert report.conclusion == "spill observed; execution status is unaffected"
    assert report.artifacts == []


def test_t17_no_artifacts_yields_not_collected_null() -> None:
    metric = only_metric(analyze_artifacts(both_adapters(), []))
    assert metric["status"] == "not_collected" and metric["value"] is None
    assert metric["source_artifact_ref"] is None and metric["parser_id"] is None
    assert "no relevant artifact" in metric["definition"]


def test_unrelated_artifacts_are_not_accepted_and_yield_not_collected() -> None:
    samples = make_ref("run-req-a1-samples", "latency_samples", b'{"samples": [1, 2]}')
    metric = only_metric(analyze_artifacts(both_adapters(), [(samples, b'{"samples": [1, 2]}')]))
    assert metric["status"] == "not_collected" and metric["value"] is None


def test_llo_dump_is_unsupported_with_null_value_and_named_reason() -> None:
    llo = make_ref("run-req-a1-llo", "llo_dump", b"llo bytes", media_type="text/plain")
    report = analyze_artifacts(both_adapters(), [(llo, b"llo bytes")])
    metric = only_metric(report)
    assert metric["status"] == "unsupported" and metric["value"] is None
    assert "LLO" in metric["definition"] and "llo-unsupported" in metric["definition"]
    assert metric["parser_id"] == "llo-unsupported" and metric["parser_version"] == "0"
    assert report.conclusion is None
    with pytest.raises(UnsupportedFormat) as info:
        LloAnalysisAdapter().parse([(llo, b"llo bytes")])
    assert info.value.exit_code == 5 and info.value.details["artifact_ids"] == ["run-req-a1-llo"]


def test_malformed_json_is_a_parse_error_with_null_value() -> None:
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", b"{not json")
    metric = only_metric(analyze_artifacts(both_adapters(), [(ref, b"{not json")]))
    assert metric["status"] == "parse_error" and metric["value"] is None
    assert "invalid JSON" in metric["definition"] and metric["parser_id"] == "mock-spill"


def test_duplicate_keys_and_non_object_reports_are_parse_errors() -> None:
    dup = b'{"format": "mock-spill-v1", "register_spill_vmem_static_bytes": 1, "register_spill_vmem_static_bytes": 2}'
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", dup)
    metric = only_metric(analyze_artifacts([MockSpillAnalysisAdapter()], [(ref, dup)]))
    assert metric["status"] == "parse_error" and metric["value"] is None
    arr = b"[1, 2, 3]"
    metric = only_metric(analyze_artifacts([MockSpillAnalysisAdapter()], [(make_ref("run-req-a1-spill-report", "mock_analysis", arr), arr)]))
    assert metric["status"] == "parse_error" and metric["value"] is None


def test_wrong_format_string_is_unsupported() -> None:
    data = json.dumps({"format": "mock-spill-v2", "register_spill_vmem_static_bytes": 5}).encode("utf-8")
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", data)
    metric = only_metric(analyze_artifacts(both_adapters(), [(ref, data)]))
    assert metric["status"] == "unsupported" and metric["value"] is None
    assert "mock-spill-v2" in metric["definition"] and "mock-spill" in metric["definition"]
    with pytest.raises(UnsupportedFormat):
        MockSpillAnalysisAdapter().parse([(ref, data)])


def test_zero_is_observed_only_when_the_report_says_zero() -> None:
    data = spill_payload(register_spill_vmem_static_bytes=0)
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", data)
    metric = only_metric(analyze_artifacts(both_adapters(), [(ref, data)]))
    assert metric["status"] == "observed" and metric["value"] == 0
    # The same report without the key is not collected, never 0.
    missing = spill_payload()
    metric = only_metric(analyze_artifacts(both_adapters(), [(make_ref("run-req-a1-spill-report", "mock_analysis", missing), missing)]))
    assert metric["status"] == "not_collected" and metric["value"] is None
    assert "no 'register_spill_vmem_static_bytes' field" in metric["definition"]


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, "65536", None, [65536]])
def test_non_integer_or_negative_values_are_parse_errors(bad: Any) -> None:
    data = spill_payload(register_spill_vmem_static_bytes=bad)
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", data)
    metric = only_metric(analyze_artifacts([MockSpillAnalysisAdapter()], [(ref, data)]))
    assert metric["status"] == "parse_error" and metric["value"] is None
    assert metric["source_artifact_ref"] is None


def test_accepts_by_kind_or_by_json_spill_report_id_only() -> None:
    adapter = MockSpillAnalysisAdapter()
    assert adapter.accepts({"artifact_id": "run-x-a1-anything", "kind": "mock_analysis", "media_type": "text/plain"})
    assert adapter.accepts({"artifact_id": "run-x-a1-spill-report", "kind": "analysis_report", "media_type": "application/json"})
    assert not adapter.accepts({"artifact_id": "run-x-a1-spill-report", "kind": "analysis_report", "media_type": "text/plain"})
    assert not adapter.accepts({"artifact_id": "run-x-a1-samples", "kind": "latency_samples", "media_type": "application/json"})
    assert not adapter.accepts("not a manifest")  # type: ignore[arg-type]
    llo = LloAnalysisAdapter()
    assert llo.accepts({"artifact_id": "run-x-a1-llo", "kind": "llo_dump"})
    assert not llo.accepts({"artifact_id": "run-x-a1-llo", "kind": "mock_analysis"})


def test_observed_value_wins_over_an_unsupported_sibling_artifact(fixture_bytes: bytes) -> None:
    spill = make_ref("run-req-a1-spill-report", "mock_analysis", fixture_bytes)
    llo = make_ref("run-req-a1-llo", "llo_dump", b"llo", media_type="text/plain")
    metric = only_metric(analyze_artifacts(both_adapters(), [(llo, b"llo"), (spill, fixture_bytes)]))
    assert metric["status"] == "observed" and metric["value"] == 65536 and metric["source_artifact_ref"] == spill.artifact_id


def test_adapter_crash_becomes_parse_error_and_never_raises(fixture_bytes: bytes) -> None:
    class Crashing:
        adapter_id = "aaa-crashing"
        parser_version = "9"
        metrics = (METRIC,)

        def accepts(self, manifest: dict) -> bool:
            return True

        def parse(self, artifacts: list) -> AnalysisReport:
            raise RuntimeError("boom")

    ref = make_ref("run-req-a1-samples", "latency_samples", b"{}")
    metric = only_metric(analyze_artifacts([Crashing()], [(ref, b"{}")]))
    assert metric["status"] == "parse_error" and metric["value"] is None
    assert "RuntimeError" in metric["definition"] and metric["parser_id"] == "aaa-crashing"
    # With the real adapter present the observed value is still produced (first producer wins per name).
    spill = make_ref("run-req-a1-spill-report", "mock_analysis", fixture_bytes)
    metric = only_metric(analyze_artifacts([MockSpillAnalysisAdapter(), Crashing()], [(spill, fixture_bytes)]))
    assert metric["status"] == "observed" and metric["value"] == 65536


def test_invalid_metric_from_an_adapter_is_downgraded_not_trusted() -> None:
    class Inventing:
        adapter_id = "inventing"
        parser_version = "1"

        def accepts(self, manifest: dict) -> bool:
            return True

        def parse(self, artifacts: list) -> AnalysisReport:
            # observed without a value, plus a not_collected that smuggles a number
            return AnalysisReport(
                metrics=[
                    {**metric_dict(METRIC, status="not_collected"), "value": 0},
                ]
            )

    ref = make_ref("run-req-a1-samples", "latency_samples", b"{}")
    metric = only_metric(analyze_artifacts([Inventing()], [(ref, b"{}")]))
    assert metric["status"] == "parse_error" and metric["value"] is None


def test_invalid_pairs_are_input_errors() -> None:
    with pytest.raises(InputError):
        analyze_artifacts(both_adapters(), [("not a ref", b"{}")])  # type: ignore[list-item]
    ref = make_ref("run-req-a1-spill-report", "mock_analysis", b"{}")
    with pytest.raises(InputError):
        analyze_artifacts(both_adapters(), [(ref, "text not bytes")])  # type: ignore[list-item]


# ------------------------------------------------------------------------------ end to end
class SpillStubAdapter:
    """KernelAdapter stub whose compile step emits the fixture spill report as evidence."""

    __test__ = False
    adapter_id = "test-spill-adapter-v1"
    backend = "mock"

    def __init__(self, compile_blobs: list[ArtifactBlob]) -> None:
        self.compile_blobs = compile_blobs

    def check_environment(self) -> dict:
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
        target = request.source.target_commit
        snapshot = SourceSnapshot(
            repo_uid=request.source.repo_uid,
            target_commit=target,
            tested_commit=GitOid(target.algorithm, target.hex),
            tested_tree=GitOid("sha1", "0000000000000000000000000000000000000065"),
            checkout_mode=request.source.checkout_mode,
            merge_parent_oids=[],
            dirty=False,
            patch_digest=None,
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
            handle={},
        )

    def compile(self, prepared: PreparedExecution) -> CompileReport:
        return CompileReport("ok", None, list(self.compile_blobs))

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        report = {"status": "pass", "cases_total": 1, "cases_passed": 1}
        blob = ArtifactBlob("correctness", "correctness_report", "application/json", json.dumps(report).encode("utf-8"))
        return CorrectnessReport(status="pass", cases_total=1, cases_passed=1, max_abs_error=0.0, max_rel_error=0.0, artifacts=[blob])

    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        return TimingReport("recorded", "microseconds", list(SAMPLES), "host_synchronized")


def make_spec(request_id: str) -> RunRequestSpec:
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=f"key-{request_id}",
        subject_ref="baseline-demo",
        config_ref="cfg-demo",
        backend="mock",
        stage="benchmark",
        protocol=dict(PROTOCOL),
        verifier=dict(VERIFIER),
        source=SourceSpec(repo_uid=REPO_UID, target_commit=BASELINE_OID, entrypoint=ENTRYPOINT),
        session_id="session-analysis",
        pair_id="pair-analysis",
        role_in_pair="candidate",
    )


def run_with_compile_blobs(store: MemoryStore, request_id: str, blobs: list[ArtifactBlob]):
    registry = AdapterRegistry()
    registry.register_kernel_adapter(SpillStubAdapter(blobs))
    runner = LocalRunner(store, adapters=registry, analysis_adapters=[MockSpillAnalysisAdapter()])
    outcome = runner.ledger.submit(make_spec(request_id))
    return runner.execute(outcome.request_id)


def test_end_to_end_runner_records_observed_spill_with_evidence_in_the_run(demo_store: MemoryStore, fixture_bytes: bytes) -> None:
    blob = ArtifactBlob("spill-report", "mock_analysis", "application/json", fixture_bytes)
    run = run_with_compile_blobs(demo_store, "request-spill", [blob])
    p = run.payload
    assert p.provenance == "trusted_worker" and p.execution_status == "succeeded"
    assert p.correctness.status == "pass" and p.timing.status == "recorded"

    metrics = {m.name: m for m in p.analysis_metrics}
    assert set(metrics) == set(runner_module.KNOWN_METRICS)
    metric = metrics[METRIC]
    assert metric.status == "observed" and metric.value == 65536
    assert metric.parser_id == "mock-spill" and metric.parser_version == "1"
    artifact_ids = {a.artifact_id for a in p.artifacts}
    assert metric.source_artifact_ref == "run-request-spill-a1-spill-report"
    assert metric.source_artifact_ref in artifact_ids
    stored = demo_store.read_artifact(p.artifact_by_id()[metric.source_artifact_ref].sha256)
    assert stored == fixture_bytes
    assert p.analysis_conclusion == "spill observed; execution status is unaffected"

    # The persisted record round-trips through the schema and the whole store stays consistent.
    persisted = demo_store.get(run.record_id).to_dict()["payload"]["analysis_metrics"]
    for m in persisted:
        validate_nested("Metric", m)
    assert deep_validate(demo_store).ok


def test_end_to_end_runner_without_spill_artifact_is_not_collected(demo_store: MemoryStore) -> None:
    run = run_with_compile_blobs(demo_store, "request-nospill", [])
    metric = {m.name: m for m in run.payload.analysis_metrics}[METRIC]
    assert run.payload.execution_status == "succeeded"
    assert metric.status == "not_collected" and metric.value is None and metric.source_artifact_ref is None
    assert deep_validate(demo_store).ok


def test_end_to_end_runner_llo_dump_is_unsupported_not_zero(demo_store: MemoryStore) -> None:
    blob = ArtifactBlob("llo", "llo_dump", "text/plain", b"opaque low-level output")
    registry = AdapterRegistry()
    registry.register_kernel_adapter(SpillStubAdapter([blob]))
    runner = LocalRunner(demo_store, adapters=registry, analysis_adapters=[MockSpillAnalysisAdapter(), LloAnalysisAdapter()])
    outcome = runner.ledger.submit(make_spec("request-llo"))
    run = runner.execute(outcome.request_id)
    metric = {m.name: m for m in run.payload.analysis_metrics}[METRIC]
    assert run.payload.execution_status == "succeeded"
    assert metric.status == "unsupported" and metric.value is None
    assert "LLO" in metric.definition
    assert "run-request-llo-a1-llo" in {a.artifact_id for a in run.payload.artifacts}  # raw dump retained as evidence
    assert deep_validate(demo_store).ok
