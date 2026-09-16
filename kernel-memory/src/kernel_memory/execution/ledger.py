"""Request ledger: one *Request* (intent) versus many *attempts* (executions).

Specification sections 9.1 and 15. A Request is registered once under an idempotency
key together with the canonical hash of its content. Execution attempts claim the
request under a lease with a strictly increasing fencing token; a stale token can
never record a result as the request's outcome (T26). Nothing here executes code.

Public API
----------
``RequestLedger(store, *, clock=time.time)``
    ``submit(spec) -> SubmitOutcome``
        Same key + same content returns the existing request (``created=False``) and
        appends nothing. Same key + different content raises
        ``IdempotencyConflictError`` (exit 3). A new key writes ``request.json``, the
        key file and a ``queued`` event.
    ``get(request_id) -> RunRequestSpec``
    ``request_record(request_id) -> dict``  (the persisted request envelope)
    ``state(request_id) -> RequestState``  (always replayed from the event files)
    ``claim(request_id, worker_id, *, lease_seconds, now=None) -> Lease``
        Refuses cancelled (``REQUEST_CANCELLED``) and finished (``REQUEST_FINISHED``)
        requests and live leases held by another worker (``LEASE_HELD``, exit 6).
        ``attempt_no`` and ``fencing_token`` are ``max(previous) + 1``. A superseded
        attempt (expired lease, or the same worker claiming again) gets a ``lease_lost``
        event first so the history says explicitly why it never finished.
    ``heartbeat(lease, *, now=None) -> Lease``  (stale token -> ``LeaseLostError``)
    ``running(lease, *, now=None) -> dict``
    ``verify_lease(lease) -> bool``  (``False`` when the token is stale or the attempt
        was marked lost/finished; never raises)
    ``require_lease(lease) -> None``  (raises ``LeaseLostError`` with the same checks)
    ``finish(lease, *, run_ref, execution_status, payload=None) -> dict``
        A stale token appends ``late_result_quarantined`` carrying the would-be
        result as evidence and raises ``LeaseLostError``; the run_ref is never
        recorded as the request's result.
    ``quarantine_late_result(lease, payload) -> NoReturn``
    ``cancel(request_id, reason, *, payload=None, lease=None) -> dict | None``
    ``mark_lease_lost(request_id, reason) -> dict``
    ``record_backend_unavailable(request_id, error, *, lease=None) -> dict``
    ``next_attempt_no(request_id) -> int``
    ``list_requests() -> list[str]``
    ``reconcile(now=None) -> ReconcileReport``
        For every claimed/running request: if ``run-<request_id>-a<attempt_no>``
        exists the terminal state is repaired without rerunning (T25); an expired
        lease without a run becomes ``lease_lost``.
``run_record_id(request_id, attempt_no) -> str``  (``run-<request_id>-a<attempt_no>``)
``job_dir(job_id) -> str``  (``requests/_jobs/<job-slug>``, shared with the budget ledger)

Persistence and guarantees
--------------------------
* Every write goes through ``MemoryStore.write_fact`` (absent-or-identical, append-only)
  under the store lock. Files:
  ``requests/<request-slug>/request.json``, ``requests/_keys/<key-slug>.json`` and
  ``requests/<request-slug>/events/<seq:06d>-<event-id>.json``.
* State is derived by replaying the event files on every call; nothing is cached as
  authoritative. Event kinds: ``queued | claimed | heartbeat | running | finished |
  cancelled | lease_lost | late_result_quarantined | backend_unavailable``.
* Lease timestamps are floats on the caller-supplied clock (``now``); the default is
  ``time.time`` so that leases remain meaningful across processes.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn

from ..adapters.base import RunRequestSpec
from ..domain.errors import (
    ExecutionInfrastructureError,
    IdConflictError,
    IdempotencyConflictError,
    InputError,
    KernelMemoryError,
    LeaseLostError,
    MissingReferenceError,
)
from ..domain.ids import slug_for_id, utc_now_iso, validate_record_id
from ..domain.jsonio import dumps_readable, loads_strict
from ..storage import layout
from ..storage.store import MemoryStore

LEDGER_VERSION = 1
KEYS_DIR = f"{layout.REQUESTS_DIR}/_keys"
JOBS_DIR = f"{layout.REQUESTS_DIR}/_jobs"

EVENT_KINDS: tuple[str, ...] = (
    "queued",
    "claimed",
    "heartbeat",
    "running",
    "finished",
    "cancelled",
    "lease_lost",
    "late_result_quarantined",
    "backend_unavailable",
)
# Kinds that move the request through its lifecycle; the others are diagnostics/evidence.
LIFECYCLE_KINDS: frozenset[str] = frozenset({"queued", "claimed", "running", "finished", "cancelled", "lease_lost"})
LEASE_KINDS: frozenset[str] = frozenset({"claimed", "heartbeat", "running"})
Clock = Callable[[], float]


def run_record_id(request_id: str, attempt_no: int) -> str:
    return f"run-{request_id}-a{attempt_no}"


def job_dir(job_id: str) -> str:
    return f"{JOBS_DIR}/{slug_for_id(job_id)}"


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubmitOutcome:
    request_id: str
    created: bool
    spec_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "created": self.created, "spec_hash": self.spec_hash}


@dataclass(frozen=True)
class Lease:
    request_id: str
    attempt_no: int
    fencing_token: int
    worker_id: str
    expires_at: float
    lease_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempt_no": self.attempt_no,
            "fencing_token": self.fencing_token,
            "worker_id": self.worker_id,
            "expires_at": self.expires_at,
            "lease_seconds": self.lease_seconds,
        }


@dataclass
class RequestState:
    request_id: str
    kind: str
    attempt_no: int
    fencing_token: int
    lease_expires_at: float | None
    worker_id: str | None
    run_refs: list[str]
    cancelled: bool
    events: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "attempt_no": self.attempt_no,
            "fencing_token": self.fencing_token,
            "lease_expires_at": self.lease_expires_at,
            "worker_id": self.worker_id,
            "run_refs": list(self.run_refs),
            "cancelled": self.cancelled,
            "event_count": len(self.events),
        }


@dataclass
class ReconcileReport:
    checked: int = 0
    repaired: list[str] = field(default_factory=list)
    lease_lost: list[str] = field(default_factory=list)
    still_leased: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "repaired": list(self.repaired),
            "lease_lost": list(self.lease_lost),
            "still_leased": list(self.still_leased),
        }


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------
class RequestLedger:
    def __init__(self, store: MemoryStore, *, clock: Clock = time.time) -> None:
        self._store = store
        self._clock = clock

    # ------------------------------------------------------------------ paths
    @staticmethod
    def _request_dir(request_id: str) -> str:
        return str(layout.request_dir(request_id))

    @classmethod
    def _request_path(cls, request_id: str) -> str:
        return f"{cls._request_dir(request_id)}/request.json"

    @classmethod
    def _events_dir(cls, request_id: str) -> str:
        return f"{cls._request_dir(request_id)}/events"

    @staticmethod
    def _key_path(idempotency_key: str) -> str:
        return f"{KEYS_DIR}/{slug_for_id(idempotency_key)}.json"

    def _write_json(self, relpath: str, data: dict[str, Any]) -> str:
        return self._store.write_fact(relpath, dumps_readable(data).encode("utf-8"))

    def _read_json(self, relpath: str) -> dict[str, Any] | None:
        raw = self._store.read_fact(relpath)
        if raw is None:
            return None
        data = loads_strict(raw)
        if not isinstance(data, dict):
            raise InputError(f"ledger file {relpath} is not a JSON object", code="LEDGER_CORRUPT", details={"path": relpath})
        return data

    # ------------------------------------------------------------------ request envelope
    def request_record(self, request_id: str) -> dict[str, Any]:
        validate_record_id(request_id, what="request_id")
        data = self._read_json(self._request_path(request_id))
        if data is None:
            raise MissingReferenceError(f"request {request_id!r} not found", code="REQUEST_NOT_FOUND", details={"request_id": request_id})
        return data

    def get(self, request_id: str) -> RunRequestSpec:
        return RunRequestSpec.from_dict(self.request_record(request_id)["spec"])

    def exists(self, request_id: str) -> bool:
        return self._store.read_fact(self._request_path(request_id)) is not None

    def list_requests(self) -> list[str]:
        ids: list[str] = []
        for rel in self._store.list_facts(layout.REQUESTS_DIR):
            parts = rel.split("/")
            if len(parts) != 3 or parts[2] != "request.json" or parts[1].startswith("_"):
                continue
            data = self._read_json(rel)
            if data is not None and isinstance(data.get("request_id"), str):
                ids.append(data["request_id"])
        return sorted(ids)

    # ------------------------------------------------------------------ submit
    def submit(self, spec: RunRequestSpec) -> SubmitOutcome:
        validate_record_id(spec.request_id, what="request_id")
        if not isinstance(spec.idempotency_key, str) or not spec.idempotency_key:
            raise InputError("idempotency_key must be a non-empty string", code="INVALID_IDEMPOTENCY_KEY")
        spec_hash = spec.spec_hash()
        key_rel = self._key_path(spec.idempotency_key)
        req_rel = self._request_path(spec.request_id)
        key_record = {"idempotency_key": spec.idempotency_key, "request_id": spec.request_id, "spec_hash": spec_hash}
        with self._store.lock():
            existing_key = self._read_json(key_rel)
            if existing_key is not None:
                if existing_key.get("idempotency_key") != spec.idempotency_key:
                    raise IdempotencyConflictError(
                        "idempotency key file collision: a different key maps to the same slug",
                        code="IDEMPOTENCY_KEY_COLLISION",
                        details={"idempotency_key": spec.idempotency_key, "existing_key": existing_key.get("idempotency_key")},
                    )
                if existing_key.get("spec_hash") != spec_hash:
                    raise IdempotencyConflictError(
                        f"idempotency key {spec.idempotency_key!r} was already used for a different request specification",
                        details={
                            "idempotency_key": spec.idempotency_key,
                            "existing_request_id": existing_key.get("request_id"),
                            "existing_spec_hash": existing_key.get("spec_hash"),
                            "new_spec_hash": spec_hash,
                        },
                    )
                existing_id = str(existing_key["request_id"])
                self._ensure_queued(existing_id)
                return SubmitOutcome(existing_id, False, spec_hash)
            existing_req = self._read_json(req_rel)
            if existing_req is not None:
                if existing_req.get("idempotency_key") != spec.idempotency_key or existing_req.get("spec_hash") != spec_hash:
                    raise IdConflictError(
                        f"request id {spec.request_id!r} already exists with different content",
                        code="REQUEST_ID_CONFLICT",
                        details={
                            "request_id": spec.request_id,
                            "existing_spec_hash": existing_req.get("spec_hash"),
                            "new_spec_hash": spec_hash,
                        },
                    )
                # request.json exists but the key file was never written (crash between writes): repair.
                self._write_json(key_rel, key_record)
                self._ensure_queued(spec.request_id)
                return SubmitOutcome(spec.request_id, False, spec_hash)
            self._write_json(
                req_rel,
                {
                    "ledger_version": LEDGER_VERSION,
                    "request_id": spec.request_id,
                    "idempotency_key": spec.idempotency_key,
                    "spec_hash": spec_hash,
                    "spec": spec.to_dict(),
                    "submitted_at": utc_now_iso(),
                },
            )
            self._write_json(key_rel, key_record)
            self._append_event(spec.request_id, "queued")
            return SubmitOutcome(spec.request_id, True, spec_hash)

    def _ensure_queued(self, request_id: str) -> None:
        if not self._events(request_id):
            self._append_event(request_id, "queued")

    # ------------------------------------------------------------------ events
    def _events(self, request_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for rel in self._store.list_facts(self._events_dir(request_id)):
            if not rel.endswith(".json"):
                continue
            data = self._read_json(rel)
            if data is None:
                continue
            if not isinstance(data.get("sequence"), int):
                raise InputError(f"event file {rel} lacks an integer sequence", code="LEDGER_CORRUPT", details={"path": rel})
            events.append(data)
        events.sort(key=lambda e: (e["sequence"], str(e.get("event_id", ""))))
        return events

    def events(self, request_id: str) -> list[dict[str, Any]]:
        self.request_record(request_id)
        return self._events(request_id)

    def _append_event(
        self,
        request_id: str,
        kind: str,
        *,
        attempt_no: int | None = None,
        worker_id: str | None = None,
        fencing_token: int | None = None,
        lease_expires_at: float | None = None,
        run_ref: str | None = None,
        execution_status: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in EVENT_KINDS:
            raise InputError(f"unknown ledger event kind {kind!r}", code="INVALID_EVENT_KIND")
        with self._store.lock():
            events = self._events(request_id)
            sequence = (max(e["sequence"] for e in events) + 1) if events else 1
            event_id = uuid.uuid4().hex[:16]
            event = {
                "event_id": event_id,
                "request_id": request_id,
                "sequence": sequence,
                "kind": kind,
                "attempt_no": attempt_no,
                "worker_id": worker_id,
                "fencing_token": fencing_token,
                "lease_expires_at": lease_expires_at,
                "run_ref": run_ref,
                "execution_status": execution_status,
                "payload": dict(payload or {}),
                "at": utc_now_iso(),
            }
            self._write_json(f"{self._events_dir(request_id)}/{sequence:06d}-{event_id}.json", event)
            return event

    # ------------------------------------------------------------------ state (replayed)
    def state(self, request_id: str) -> RequestState:
        self.request_record(request_id)
        events = self._events(request_id)
        kind = "queued"
        attempt_no = 0
        fencing_token = 0
        cancelled = False
        run_refs: list[str] = []
        for event in events:
            if event["kind"] in LIFECYCLE_KINDS:
                kind = event["kind"]
            if isinstance(event.get("attempt_no"), int):
                attempt_no = max(attempt_no, event["attempt_no"])
            if isinstance(event.get("fencing_token"), int):
                fencing_token = max(fencing_token, event["fencing_token"])
            if event["kind"] == "cancelled":
                cancelled = True
            if event["kind"] == "finished" and isinstance(event.get("run_ref"), str):
                run_refs.append(event["run_ref"])
        lease_expires_at: float | None = None
        worker_id: str | None = None
        for event in events:
            if event["kind"] in LEASE_KINDS and event.get("attempt_no") == attempt_no:
                if isinstance(event.get("lease_expires_at"), (int, float)):
                    lease_expires_at = float(event["lease_expires_at"])
                if isinstance(event.get("worker_id"), str):
                    worker_id = event["worker_id"]
        return RequestState(
            request_id=request_id,
            kind=kind,
            attempt_no=attempt_no,
            fencing_token=fencing_token,
            lease_expires_at=lease_expires_at,
            worker_id=worker_id,
            run_refs=run_refs,
            cancelled=cancelled,
            events=events,
        )

    def next_attempt_no(self, request_id: str) -> int:
        return self.state(request_id).attempt_no + 1

    # ------------------------------------------------------------------ leases
    def claim(self, request_id: str, worker_id: str, *, lease_seconds: float, now: Clock | None = None) -> Lease:
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise InputError("lease_seconds must be a positive number", code="INVALID_LEASE")
        if not isinstance(worker_id, str) or not worker_id:
            raise InputError("worker_id must be a non-empty string", code="INVALID_WORKER_ID")
        clock = now or self._clock
        with self._store.lock():
            state = self.state(request_id)
            if state.cancelled:
                raise InputError(
                    f"request {request_id!r} is cancelled and cannot be claimed",
                    code="REQUEST_CANCELLED",
                    details={"request_id": request_id},
                )
            if state.kind == "finished":
                raise InputError(
                    f"request {request_id!r} is finished; a new measurement needs a new request",
                    code="REQUEST_FINISHED",
                    details={"request_id": request_id, "run_refs": list(state.run_refs)},
                )
            current = float(clock())
            if (
                state.kind in ("claimed", "running")
                and state.lease_expires_at is not None
                and current < state.lease_expires_at
                and state.worker_id != worker_id
            ):
                raise ExecutionInfrastructureError(
                    f"request {request_id!r} is leased by worker {state.worker_id!r} until {state.lease_expires_at}",
                    code="LEASE_HELD",
                    details={
                        "request_id": request_id,
                        "holder": state.worker_id,
                        "lease_expires_at": state.lease_expires_at,
                        "now": current,
                    },
                )
            if state.kind in ("claimed", "running"):
                # The previous attempt never reached a terminal event; record why it is superseded.
                expired = state.lease_expires_at is None or current >= state.lease_expires_at
                self._append_event(
                    request_id,
                    "lease_lost",
                    attempt_no=state.attempt_no,
                    worker_id=state.worker_id,
                    fencing_token=state.fencing_token,
                    lease_expires_at=state.lease_expires_at,
                    payload={
                        "reason": "lease expired before a terminal event" if expired else "superseded by a new claim from the same worker",
                        "superseded_by_worker": worker_id,
                        "now": current,
                    },
                )
            attempt_no = state.attempt_no + 1
            fencing_token = state.fencing_token + 1
            expires_at = current + float(lease_seconds)
            self._append_event(
                request_id,
                "claimed",
                attempt_no=attempt_no,
                worker_id=worker_id,
                fencing_token=fencing_token,
                lease_expires_at=expires_at,
                payload={"lease_seconds": float(lease_seconds)},
            )
            return Lease(request_id, attempt_no, fencing_token, worker_id, expires_at, float(lease_seconds))

    def verify_lease(self, lease: Lease) -> bool:
        """True when ``lease`` still holds the current fencing token for its request."""
        try:
            self.require_lease(lease)
        except LeaseLostError:
            return False
        return True

    def require_lease(self, lease: Lease) -> None:
        state = self.state(lease.request_id)
        if lease.fencing_token != state.fencing_token:
            raise LeaseLostError(
                f"fencing token {lease.fencing_token} for request {lease.request_id!r} is stale (current {state.fencing_token})",
                details={
                    "request_id": lease.request_id,
                    "held_token": lease.fencing_token,
                    "current_token": state.fencing_token,
                    "held_attempt_no": lease.attempt_no,
                    "current_attempt_no": state.attempt_no,
                    "current_worker": state.worker_id,
                },
            )
        if state.kind == "lease_lost" and state.attempt_no == lease.attempt_no:
            raise LeaseLostError(
                f"lease for request {lease.request_id!r} attempt {lease.attempt_no} was marked lost",
                details={"request_id": lease.request_id, "attempt_no": lease.attempt_no},
            )
        if state.kind == "finished" and state.attempt_no == lease.attempt_no:
            raise LeaseLostError(
                f"request {lease.request_id!r} attempt {lease.attempt_no} already finished",
                code="ATTEMPT_ALREADY_FINISHED",
                details={"request_id": lease.request_id, "attempt_no": lease.attempt_no, "run_refs": list(state.run_refs)},
            )

    def heartbeat(self, lease: Lease, *, now: Clock | None = None) -> Lease:
        clock = now or self._clock
        with self._store.lock():
            self.require_lease(lease)
            expires_at = float(clock()) + lease.lease_seconds
            self._append_event(
                lease.request_id,
                "heartbeat",
                attempt_no=lease.attempt_no,
                worker_id=lease.worker_id,
                fencing_token=lease.fencing_token,
                lease_expires_at=expires_at,
            )
            return dataclasses.replace(lease, expires_at=expires_at)

    def running(self, lease: Lease, *, now: Clock | None = None) -> dict[str, Any]:
        with self._store.lock():
            self.require_lease(lease)
            return self._append_event(
                lease.request_id,
                "running",
                attempt_no=lease.attempt_no,
                worker_id=lease.worker_id,
                fencing_token=lease.fencing_token,
                lease_expires_at=lease.expires_at,
            )

    def quarantine_late_result(self, lease: Lease, payload: dict[str, Any], *, cause: KernelMemoryError | None = None) -> NoReturn:
        """Record a stale worker's result as evidence only and raise ``LeaseLostError``."""
        with self._store.lock():
            state = self.state(lease.request_id)
            evidence = {
                "reason": cause.message if cause is not None else "stale fencing token",
                "stale_fencing_token": lease.fencing_token,
                "current_fencing_token": state.fencing_token,
                "stale_attempt_no": lease.attempt_no,
                "current_attempt_no": state.attempt_no,
            }
            evidence.update(payload)
            self._append_event(
                lease.request_id,
                "late_result_quarantined",
                attempt_no=lease.attempt_no,
                worker_id=lease.worker_id,
                fencing_token=lease.fencing_token,
                payload=evidence,
            )
        error = LeaseLostError(
            f"late result from worker {lease.worker_id!r} (token {lease.fencing_token}) for request {lease.request_id!r} "
            f"was quarantined; current token is {state.fencing_token}",
            details={k: v for k, v in evidence.items() if k != "record"},
        )
        if cause is not None:
            raise error from cause
        raise error

    def finish(
        self,
        lease: Lease,
        *,
        run_ref: str | None,
        execution_status: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._store.lock():
            try:
                self.require_lease(lease)
            except LeaseLostError as exc:
                self.quarantine_late_result(
                    lease,
                    {"quarantined_run_ref": run_ref, "execution_status": execution_status, **(payload or {})},
                    cause=exc,
                )
            return self._append_event(
                lease.request_id,
                "finished",
                attempt_no=lease.attempt_no,
                worker_id=lease.worker_id,
                fencing_token=lease.fencing_token,
                lease_expires_at=lease.expires_at,
                run_ref=run_ref,
                execution_status=execution_status,
                payload=payload,
            )

    # ------------------------------------------------------------------ terminal / diagnostic events
    def cancel(
        self,
        request_id: str,
        reason: str,
        *,
        payload: dict[str, Any] | None = None,
        lease: Lease | None = None,
    ) -> dict[str, Any] | None:
        with self._store.lock():
            state = self.state(request_id)
            if state.kind == "finished":
                raise InputError(
                    f"request {request_id!r} is already finished and cannot be cancelled",
                    code="REQUEST_FINISHED",
                    details={"request_id": request_id, "run_refs": list(state.run_refs)},
                )
            if state.cancelled:
                return None
            return self._append_event(
                request_id,
                "cancelled",
                attempt_no=lease.attempt_no if lease else (state.attempt_no or None),
                worker_id=lease.worker_id if lease else state.worker_id,
                fencing_token=lease.fencing_token if lease else (state.fencing_token or None),
                payload={"reason": str(reason), **(payload or {})},
            )

    def mark_lease_lost(self, request_id: str, reason: str) -> dict[str, Any]:
        with self._store.lock():
            state = self.state(request_id)
            if state.kind not in ("claimed", "running"):
                raise InputError(
                    f"request {request_id!r} is not leased (state {state.kind!r})",
                    code="REQUEST_NOT_LEASED",
                    details={"request_id": request_id, "kind": state.kind},
                )
            return self._append_event(
                request_id,
                "lease_lost",
                attempt_no=state.attempt_no,
                worker_id=state.worker_id,
                fencing_token=state.fencing_token,
                lease_expires_at=state.lease_expires_at,
                payload={"reason": str(reason)},
            )

    def record_backend_unavailable(self, request_id: str, error: Exception, *, lease: Lease | None = None) -> dict[str, Any]:
        """Record that the backend could not execute the request (evidence only, not an execution).

        When a lease is supplied the claimed attempt is released with a ``lease_lost`` event
        so the request becomes claimable again once the backend is available; the attempt
        number stays consumed and nothing is inferred about the (never started) execution.
        """
        if isinstance(error, KernelMemoryError):
            evidence: dict[str, Any] = error.to_dict()
        else:
            evidence = {"error": type(error).__name__, "message": str(error)}
        with self._store.lock():
            event = self._append_event(
                request_id,
                "backend_unavailable",
                attempt_no=lease.attempt_no if lease else None,
                worker_id=lease.worker_id if lease else None,
                fencing_token=lease.fencing_token if lease else None,
                payload=evidence,
            )
            if lease is not None:
                state = self.state(request_id)
                if state.kind in ("claimed", "running") and state.fencing_token == lease.fencing_token:
                    self._append_event(
                        request_id,
                        "lease_lost",
                        attempt_no=lease.attempt_no,
                        worker_id=lease.worker_id,
                        fencing_token=lease.fencing_token,
                        lease_expires_at=lease.expires_at,
                        payload={"reason": "released: backend unavailable before execution started", "error": evidence.get("error")},
                    )
            return event

    # ------------------------------------------------------------------ recovery
    def reconcile(self, now: Clock | None = None) -> ReconcileReport:
        clock = now or self._clock
        report = ReconcileReport()
        with self._store.lock():
            for request_id in self.list_requests():
                state = self.state(request_id)
                if state.kind not in ("claimed", "running"):
                    continue
                report.checked += 1
                run_id = run_record_id(request_id, state.attempt_no)
                if self._store.exists(run_id):
                    run = self._store.get(run_id)
                    status = run.payload.execution_status if run is not None and run.record_type == "run" else "unknown"
                    self._append_event(
                        request_id,
                        "finished",
                        attempt_no=state.attempt_no,
                        worker_id=state.worker_id,
                        fencing_token=state.fencing_token,
                        lease_expires_at=state.lease_expires_at,
                        run_ref=run_id,
                        execution_status=status,
                        payload={"repaired_by": "reconcile", "reason": "run record found for (request_id, attempt_no)"},
                    )
                    report.repaired.append(request_id)
                elif state.lease_expires_at is None or float(clock()) >= state.lease_expires_at:
                    self._append_event(
                        request_id,
                        "lease_lost",
                        attempt_no=state.attempt_no,
                        worker_id=state.worker_id,
                        fencing_token=state.fencing_token,
                        lease_expires_at=state.lease_expires_at,
                        payload={"reason": "lease expired without a published run", "reconciled": True},
                    )
                    report.lease_lost.append(request_id)
                else:
                    report.still_leased.append(request_id)
        return report


__all__ = [
    "LEDGER_VERSION",
    "EVENT_KINDS",
    "LIFECYCLE_KINDS",
    "KEYS_DIR",
    "JOBS_DIR",
    "SubmitOutcome",
    "Lease",
    "RequestState",
    "ReconcileReport",
    "RequestLedger",
    "run_record_id",
    "job_dir",
]
