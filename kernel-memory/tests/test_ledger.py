"""Request ledger: idempotent submission, leases, fencing (T26), reconcile (T25)."""
from __future__ import annotations

import json

import pytest
from conftest import record_dict

from kernel_memory.adapters.base import RunRequestSpec, SourceSpec
from kernel_memory.domain.errors import (
    ExecutionInfrastructureError,
    IdConflictError,
    IdempotencyConflictError,
    InputError,
    LeaseLostError,
    MissingReferenceError,
)
from kernel_memory.domain.hashing import sha256_bytes
from kernel_memory.domain.models import GitOid, Record
from kernel_memory.execution.ledger import RequestLedger, run_record_id
from kernel_memory.storage import MemoryStore

BASELINE_OID = GitOid("sha1", "0000000000000000000000000000000000000001")
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
    "reference_source_hash": sha256_bytes(b"reference"),
    "suite_hash": sha256_bytes(b"suite"),
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


def make_spec(request_id: str, *, key: str | None = None, overrides: dict | None = None, stage: str = "benchmark") -> RunRequestSpec:
    return RunRequestSpec(
        request_id=request_id,
        idempotency_key=key or f"key-{request_id}",
        subject_ref="baseline-demo",
        config_ref="cfg-demo",
        backend="mock",
        stage=stage,
        protocol=dict(PROTOCOL),
        verifier=dict(VERIFIER),
        source=SourceSpec(
            repo_uid="github:github.com:repo:900001",
            target_commit=BASELINE_OID,
            entrypoint="demo.reference:vector_add",
            implementation_overrides=dict(overrides or {}),
        ),
    )


def kinds(ledger: RequestLedger, request_id: str) -> list[str]:
    return [e["kind"] for e in ledger.state(request_id).events]


# ------------------------------------------------------------------------------ submit
def test_t11_submit_same_key_same_spec_is_idempotent(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    first = ledger.submit(make_spec("request-a", key="k1"))
    assert first.created is True and first.request_id == "request-a"
    # A replay may even carry a different request_id: the key + content identify the intent.
    second = ledger.submit(make_spec("request-b", key="k1"))
    assert second.created is False
    assert second.request_id == "request-a"
    assert second.spec_hash == first.spec_hash
    assert kinds(ledger, "request-a") == ["queued"]
    assert ledger.list_requests() == ["request-a"]
    assert not ledger.exists("request-b")


def test_t12_same_key_different_spec_conflicts(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a", key="k1"))
    with pytest.raises(IdempotencyConflictError) as excinfo:
        ledger.submit(make_spec("request-a", key="k1", overrides={"chunk": 8}))
    assert excinfo.value.exit_code == 3
    assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"
    assert excinfo.value.details["existing_request_id"] == "request-a"
    assert kinds(ledger, "request-a") == ["queued"]


def test_t12_same_request_id_different_key_conflicts(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a", key="k1"))
    with pytest.raises(IdConflictError) as excinfo:
        ledger.submit(make_spec("request-a", key="k2"))
    assert excinfo.value.exit_code == 3
    assert excinfo.value.code == "REQUEST_ID_CONFLICT"


def test_submit_persists_request_key_and_event_files(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    spec = make_spec("request-a", key="k1")
    ledger.submit(spec)
    facts = store.list_facts("requests")
    assert "requests/request-a/request.json" in facts
    assert any(f.startswith("requests/_keys/") for f in facts)
    events = [f for f in facts if f.startswith("requests/request-a/events/")]
    assert len(events) == 1 and events[0].split("/")[-1].startswith("000001-")
    envelope = json.loads(store.read_fact("requests/request-a/request.json"))
    assert envelope["ledger_version"] == 1
    assert envelope["spec_hash"] == spec.spec_hash()
    assert envelope["spec"]["source"]["target_commit"] == {"algorithm": "sha1", "hex": BASELINE_OID.hex}
    # A fresh ledger instance replays the same state from the files (nothing is cached).
    assert RequestLedger(store).get("request-a") == spec
    assert RequestLedger(store).state("request-a").kind == "queued"


def test_submit_rejects_empty_key_and_missing_request(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    import dataclasses

    with pytest.raises(InputError) as key_err:
        ledger.submit(dataclasses.replace(make_spec("request-a"), idempotency_key=""))
    assert key_err.value.code == "INVALID_IDEMPOTENCY_KEY"
    with pytest.raises(MissingReferenceError) as excinfo:
        ledger.get("request-missing")
    assert excinfo.value.code == "REQUEST_NOT_FOUND"


# ------------------------------------------------------------------------------ leases
def test_claim_increments_attempt_and_fencing_token_across_expired_attempts(store: MemoryStore) -> None:
    clock = FakeClock()
    ledger = RequestLedger(store, clock=clock)
    ledger.submit(make_spec("request-a"))
    lease1 = ledger.claim("request-a", "worker-1", lease_seconds=10)
    assert (lease1.attempt_no, lease1.fencing_token) == (1, 1)
    assert lease1.expires_at == clock.now + 10
    assert ledger.next_attempt_no("request-a") == 2
    clock.advance(11)  # the lease expires without a terminal event (crashed worker)
    lease2 = ledger.claim("request-a", "worker-2", lease_seconds=10)
    assert (lease2.attempt_no, lease2.fencing_token) == (2, 2)
    state = ledger.state("request-a")
    assert state.kind == "claimed" and state.worker_id == "worker-2"
    assert "lease_lost" in kinds(ledger, "request-a")
    # The same worker may re-claim its own live lease (restart); the token still moves forward.
    lease3 = ledger.claim("request-a", "worker-2", lease_seconds=10)
    assert (lease3.attempt_no, lease3.fencing_token) == (3, 3)
    assert ledger.verify_lease(lease3) is True
    assert ledger.verify_lease(lease2) is False
    assert ledger.verify_lease(lease1) is False


def test_live_lease_of_another_worker_is_held(store: MemoryStore) -> None:
    clock = FakeClock()
    ledger = RequestLedger(store, clock=clock)
    ledger.submit(make_spec("request-a"))
    ledger.claim("request-a", "worker-1", lease_seconds=60)
    with pytest.raises(ExecutionInfrastructureError) as excinfo:
        ledger.claim("request-a", "worker-2", lease_seconds=60)
    assert excinfo.value.code == "LEASE_HELD"
    assert excinfo.value.exit_code == 6
    assert excinfo.value.details["holder"] == "worker-1"


def test_claim_validates_arguments(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a"))
    with pytest.raises(InputError):
        ledger.claim("request-a", "worker-1", lease_seconds=0)
    with pytest.raises(InputError):
        ledger.claim("request-a", "", lease_seconds=10)


def test_heartbeat_extends_live_lease_and_rejects_stale_token(store: MemoryStore) -> None:
    clock = FakeClock()
    ledger = RequestLedger(store, clock=clock)
    ledger.submit(make_spec("request-a"))
    lease1 = ledger.claim("request-a", "worker-1", lease_seconds=10)
    clock.advance(5)
    extended = ledger.heartbeat(lease1)
    assert extended.expires_at == clock.now + 10
    assert ledger.state("request-a").lease_expires_at == extended.expires_at
    clock.advance(11)
    lease2 = ledger.claim("request-a", "worker-2", lease_seconds=10)
    with pytest.raises(LeaseLostError) as excinfo:
        ledger.heartbeat(lease1)
    assert excinfo.value.exit_code == 6
    assert excinfo.value.details["current_token"] == lease2.fencing_token
    with pytest.raises(LeaseLostError):
        ledger.running(lease1)
    assert ledger.running(lease2)["kind"] == "running"
    assert ledger.state("request-a").kind == "running"


def test_t26_finish_with_stale_token_is_quarantined_and_never_recorded(store: MemoryStore) -> None:
    clock = FakeClock()
    ledger = RequestLedger(store, clock=clock)
    ledger.submit(make_spec("request-a"))
    stale = ledger.claim("request-a", "worker-1", lease_seconds=10)
    clock.advance(30)
    current = ledger.claim("request-a", "worker-2", lease_seconds=10)
    with pytest.raises(LeaseLostError) as excinfo:
        ledger.finish(stale, run_ref="run-request-a-a1", execution_status="succeeded")
    assert excinfo.value.code == "LEASE_LOST"
    state = ledger.state("request-a")
    assert state.run_refs == []
    assert state.kind == "claimed"
    quarantined = [e for e in state.events if e["kind"] == "late_result_quarantined"]
    assert len(quarantined) == 1
    assert quarantined[0]["payload"]["quarantined_run_ref"] == "run-request-a-a1"
    assert quarantined[0]["payload"]["stale_fencing_token"] == stale.fencing_token
    assert quarantined[0]["payload"]["current_fencing_token"] == current.fencing_token
    assert quarantined[0]["fencing_token"] == stale.fencing_token
    # The current holder finishes normally and only its run is the request's result.
    ledger.finish(current, run_ref="run-request-a-a2", execution_status="succeeded")
    state = ledger.state("request-a")
    assert state.kind == "finished"
    assert state.run_refs == ["run-request-a-a2"]
    # A finished request cannot be claimed or finished twice.
    with pytest.raises(InputError) as claim_err:
        ledger.claim("request-a", "worker-3", lease_seconds=10)
    assert claim_err.value.code == "REQUEST_FINISHED"
    with pytest.raises(LeaseLostError):
        ledger.finish(current, run_ref="run-request-a-a2-dup", execution_status="succeeded")
    assert ledger.state("request-a").run_refs == ["run-request-a-a2"]


def test_cancel_prevents_claims_and_is_recorded_once(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a"))
    event = ledger.cancel("request-a", "operator abort")
    assert event is not None and event["payload"]["reason"] == "operator abort"
    assert ledger.cancel("request-a", "again") is None
    state = ledger.state("request-a")
    assert state.cancelled is True and state.kind == "cancelled"
    with pytest.raises(InputError) as excinfo:
        ledger.claim("request-a", "worker-1", lease_seconds=10)
    assert excinfo.value.code == "REQUEST_CANCELLED"
    assert excinfo.value.exit_code == 2
    assert kinds(ledger, "request-a") == ["queued", "cancelled"]


def test_cancel_of_finished_request_is_refused(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a"))
    lease = ledger.claim("request-a", "worker-1", lease_seconds=10)
    ledger.finish(lease, run_ref="run-request-a-a1", execution_status="compile_error")
    with pytest.raises(InputError) as excinfo:
        ledger.cancel("request-a", "too late")
    assert excinfo.value.code == "REQUEST_FINISHED"


def test_mark_lease_lost_requires_a_leased_request(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a"))
    with pytest.raises(InputError) as excinfo:
        ledger.mark_lease_lost("request-a", "no lease")
    assert excinfo.value.code == "REQUEST_NOT_LEASED"
    lease = ledger.claim("request-a", "worker-1", lease_seconds=10)
    ledger.mark_lease_lost("request-a", "worker crashed")
    assert ledger.state("request-a").kind == "lease_lost"
    assert ledger.verify_lease(lease) is False
    # A new attempt may be claimed afterwards.
    lease2 = ledger.claim("request-a", "worker-1", lease_seconds=10)
    assert lease2.attempt_no == 2 and lease2.fencing_token == 2


def test_record_backend_unavailable_with_lease_releases_the_attempt(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-a"))
    lease = ledger.claim("request-a", "worker-1", lease_seconds=10)
    from kernel_memory.domain.errors import BackendUnavailable

    ledger.record_backend_unavailable("request-a", BackendUnavailable("no hardware", details={"backend": "mock"}), lease=lease)
    state = ledger.state("request-a")
    assert state.kind == "lease_lost"
    assert [e["kind"] for e in state.events][-2:] == ["backend_unavailable", "lease_lost"]
    assert state.events[-2]["payload"]["error"] == "BACKEND_UNAVAILABLE"
    assert state.run_refs == []


# ------------------------------------------------------------------------------ reconcile
def test_t25_reconcile_repairs_claimed_request_whose_run_exists(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    ledger = RequestLedger(demo_store)
    ledger.submit(make_spec("request-recover"))
    lease = ledger.claim("request-recover", "worker-1", lease_seconds=600)
    run_id = run_record_id("request-recover", lease.attempt_no)
    assert run_id == "run-request-recover-a1"
    # Simulate: the Run was published, then the process died before ledger.finish().
    data = record_dict(bundle_dicts, "run-demo-baseline")
    data["record_id"] = run_id
    data["payload"]["request_id"] = "request-recover"
    data["payload"]["attempt_no"] = lease.attempt_no
    demo_store.publish(Record.from_dict(data))
    runs_before = len(demo_store.records("run"))
    events_before = len(ledger.state("request-recover").events)

    report = ledger.reconcile()
    assert report.checked == 1
    assert report.repaired == ["request-recover"]
    assert report.lease_lost == [] and report.still_leased == []
    state = ledger.state("request-recover")
    assert state.kind == "finished"
    assert state.run_refs == [run_id]
    finished = state.events[-1]
    assert finished["kind"] == "finished" and finished["run_ref"] == run_id
    assert finished["execution_status"] == "succeeded"
    assert finished["payload"]["repaired_by"] == "reconcile"
    # Nothing was re-measured: no new run, no new claim, exactly one repair event.
    assert len(demo_store.records("run")) == runs_before
    assert len(state.events) == events_before + 1
    assert "claimed" not in [e["kind"] for e in state.events[events_before:]]
    assert ledger.reconcile().checked == 0  # idempotent: nothing left to repair


def test_reconcile_marks_expired_lease_lost_and_leaves_live_lease(store: MemoryStore) -> None:
    clock = FakeClock()
    ledger = RequestLedger(store, clock=clock)
    ledger.submit(make_spec("request-expired"))
    ledger.submit(make_spec("request-live"))
    ledger.submit(make_spec("request-queued"))
    ledger.claim("request-expired", "worker-1", lease_seconds=10)
    clock.advance(5)
    ledger.claim("request-live", "worker-2", lease_seconds=100)
    clock.advance(10)
    report = ledger.reconcile()
    assert report.checked == 2
    assert report.lease_lost == ["request-expired"]
    assert report.still_leased == ["request-live"]
    assert report.repaired == []
    assert ledger.state("request-expired").kind == "lease_lost"
    assert ledger.state("request-live").kind == "claimed"
    assert ledger.state("request-queued").kind == "queued"
    assert report.to_dict()["lease_lost"] == ["request-expired"]


def test_list_requests_ignores_internal_directories(store: MemoryStore) -> None:
    ledger = RequestLedger(store)
    ledger.submit(make_spec("request-b"))
    ledger.submit(make_spec("request-a"))
    store.write_fact("requests/_jobs/job-x/job.json", b"{}")
    assert ledger.list_requests() == ["request-a", "request-b"]
