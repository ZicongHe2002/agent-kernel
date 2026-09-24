"""Synthetic *trusted* runs for the compare/decide test suites (importable: ``from synthetic_runs import ...``).

Nothing here recomputes identity hashes by hand for trusted runs: every trusted run is produced by
the real ``kernel_memory.execution.runner.LocalRunner`` driving an in-test adapter, so
``variant_digest -> environment_hash -> protocol_hash -> verifier_hash -> comparison_key`` are derived
exactly as in production and the latency samples land in the content-addressed store.

Helpers
-------
``SyntheticAdapter``            trimmed copy of ``tests/test_runner.py::StubAdapter`` with knobs
                                 ``samples``, ``unit``, ``correctness``, ``compile_status``,
                                 ``environment_overrides``, ``tested_hex``, ``dirty`` and
                                 ``implementation_overrides`` passthrough from the request.
``trusted_run(store, ...)``     one published ``run`` Record with ``provenance == trusted_worker``.
``trusted_pairs(store, ...)``   ``n`` candidate/baseline pairs with distinct ``session_id``/``pair_id``,
                                 returned as ``[{"candidate_run": ref, "baseline_run": ref}, ...]``.
``cloned_run_dict(bundle_dicts, run_id, ...)``
                                 dict clone of a fixture run (pattern ``tests/test_ledger.py``) with a fresh
                                 ``record_id``/``request_id``; nested payload fields are overridden with
                                 double-underscore keys (``environment__accelerator_model="X"``) and the
                                 affected snapshot hashes / variant digest / comparison key are recomputed
                                 with ``kernel_memory.domain.hashing`` unless ``reconcile=False``.
``reconcile_run_hashes(data, config_hash=...)``, ``set_path(data, "a.b.c", value)``,
``publish_dict(store, data)``   small building blocks used by the above.

Subjects are fixture records whose ``commit_oid`` is a fictional sha1 (``baseline-demo`` ...0001,
``commit-demo-a`` ...0002); ``trusted_run`` reads the subject's ``commit_oid``/``repo_uid`` from the
store so ``source.target_commit`` always equals the subject's commit and the adapter reports
``tested == target`` (for ``exact_commit``).
"""
from __future__ import annotations

import copy
import json
from typing import Any, Sequence

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
from kernel_memory.domain.models import GitOid, Record
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.storage import MemoryStore

from conftest import record_dict

ENTRYPOINT = "demo.reference:vector_add"
CONFIG_REF = "cfg-demo"
PROTOCOL: dict[str, Any] = {
    "protocol_id": "synthetic-bench-v1",
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
VERIFIER: dict[str, Any] = {
    "verifier_id": "synthetic-verifier-v1",
    "reference_source_hash": hashing.sha256_bytes(b"synthetic-reference"),
    "suite_hash": hashing.sha256_bytes(b"synthetic-suite"),
    "tolerances": {"atol": "0", "rtol": "0"},
    "nonfinite_policy": "reject_unexpected",
}
DEFAULT_SAMPLES: tuple[float, ...] = (10.0, 11.0, 10.0, 9.0, 12.0)
# Fictional OIDs used only when a request asks for an integration_merge checkout.
MERGE_TESTED_HEX = "00000000000000000000000000000000000000ab"
MERGE_PARENT_HEX = "0000000000000000000000000000000000000001"
SOURCE_DIGEST = hashing.source_digest([("demo/reference.py", hashing.sha256_bytes(b"def vector_add"))])


class SyntheticAdapter:
    """In-test KernelAdapter with configurable outcomes (never collected by pytest)."""

    __test__ = False
    adapter_id = "synthetic-adapter-v1"
    backend = "mock"

    def __init__(
        self,
        *,
        samples: Sequence[float] | None = None,
        unit: str = "microseconds",
        correctness: str = "pass",
        compile_status: str = "ok",
        environment_overrides: dict[str, Any] | None = None,
        tested_hex: str | None = None,
        dirty: bool = False,
    ) -> None:
        self.samples = list(DEFAULT_SAMPLES if samples is None else samples)
        self.unit = unit
        self.correctness = correctness
        self.compile_status = compile_status
        self.environment_overrides = dict(environment_overrides or {})
        self.tested_hex = tested_hex
        self.dirty = dirty
        self.calls: dict[str, int] = {"check_environment": 0, "prepare": 0, "compile": 0, "verify": 0, "benchmark": 0}

    def check_environment(self) -> dict[str, Any]:
        self.calls["check_environment"] += 1
        env: dict[str, Any] = {
            "backend": "mock",
            "accelerator_model": "HOST-CPU-SYNTHETIC",
            "device_count": 1,
            "topology": "single",
            "software": {"python": "3.11", "adapter": self.adapter_id},
            "execution_flags": {},
            "host_timer_environment": {"timer": "perf_counter_ns"},
            "unknown_required_fields": [],
        }
        env.update(copy.deepcopy(self.environment_overrides))
        return env

    def prepare(self, request: RunRequestSpec, problem: dict[str, Any]) -> PreparedExecution:
        self.calls["prepare"] += 1
        target = request.source.target_commit
        merge_parents: list[GitOid] = []
        if self.tested_hex is not None:
            tested = GitOid(target.algorithm, self.tested_hex)
        elif request.source.checkout_mode == "integration_merge":
            tested = GitOid(target.algorithm, MERGE_TESTED_HEX)
            merge_parents = [target, GitOid(target.algorithm, MERGE_PARENT_HEX)]
        else:
            tested = target
        snapshot = SourceSnapshot(
            repo_uid=request.source.repo_uid,
            target_commit=target,
            tested_commit=tested,
            tested_tree=GitOid("sha1", "0000000000000000000000000000000000000065"),
            checkout_mode=request.source.checkout_mode,
            merge_parent_oids=merge_parents,
            dirty=self.dirty,
            patch_digest=hashing.sha256_bytes(b"synthetic-patch") if self.dirty else None,
            source_digest=SOURCE_DIGEST,
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
        return TimingReport("recorded", self.unit, list(self.samples), "host_synchronized")


def _registry(adapter: Any) -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register_kernel_adapter(adapter)
    return registry


def trusted_run(
    store: MemoryStore,
    *,
    request_id: str,
    subject_ref: str,
    session_id: str | None,
    pair_id: str | None,
    role: str | None,
    samples: Sequence[float] = DEFAULT_SAMPLES,
    unit: str = "microseconds",
    overrides: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
    environment_overrides: dict[str, Any] | None = None,
    checkout_mode: str = "exact_commit",
    correctness: str = "pass",
    compile_status: str = "ok",
    dirty: bool = False,
    allow_dirty: bool | None = None,
    tested_hex: str | None = None,
    config_ref: str = CONFIG_REF,
    stage: str = "benchmark",
    idempotency_key: str | None = None,
    worker_id: str = "synthetic-runner",
) -> Record:
    """Execute one request through the real ``LocalRunner`` and return the published trusted run.

    ``protocol``/``verifier`` are merged over the module defaults (top-level keys replace; pass a full
    ``tolerances`` dict to change tolerances). ``environment_overrides`` are merged into the adapter's
    environment snapshot (so they change ``environment_hash`` and therefore ``comparison_key``).
    ``dirty=True`` implies ``allow_dirty=True`` unless given, so the runner records a dirty run instead
    of refusing it.
    """
    subject = store.require(subject_ref, "commit", "baseline")
    target: GitOid = subject.payload.commit_oid
    spec = RunRequestSpec(
        request_id=request_id,
        idempotency_key=idempotency_key if idempotency_key is not None else f"key-{request_id}",
        subject_ref=subject_ref,
        config_ref=config_ref,
        backend="mock",
        stage=stage,
        protocol={**PROTOCOL, **dict(protocol or {})},
        verifier={**VERIFIER, **dict(verifier or {})},
        source=SourceSpec(
            repo_uid=subject.payload.repo_uid,
            target_commit=target,
            entrypoint=ENTRYPOINT,
            checkout_mode=checkout_mode,
            implementation_overrides=dict(overrides or {}),
            allow_dirty_exploratory=bool(dirty if allow_dirty is None else allow_dirty),
        ),
        authorization={},
        session_id=session_id,
        pair_id=pair_id,
        role_in_pair=role,
        max_wall_seconds=600.0,
    )
    adapter = SyntheticAdapter(
        samples=samples,
        unit=unit,
        correctness=correctness,
        compile_status=compile_status,
        environment_overrides=environment_overrides,
        tested_hex=tested_hex,
        dirty=dirty,
    )
    runner = LocalRunner(store, adapters=_registry(adapter), worker_id=worker_id)
    _, run = runner.submit_and_execute(spec)
    if run is None:  # pragma: no cover - the runner always returns the run for a fresh request
        raise AssertionError(f"runner produced no run for request {request_id!r}")
    assert run.payload.provenance == "trusted_worker"
    return run


def _per_pair(samples: Any, n: int, what: str) -> list[list[float]]:
    """Accept one sample tuple (replicated) or a sequence of ``n`` per-pair sample tuples."""
    seq = list(samples)
    if seq and isinstance(seq[0], (list, tuple)):
        if len(seq) != n:
            raise ValueError(f"{what}: expected {n} per-pair sample tuples, got {len(seq)}")
        return [list(map(float, s)) for s in seq]
    return [list(map(float, seq)) for _ in range(n)]


def trusted_pairs(
    store: MemoryStore,
    *,
    n: int = 3,
    candidate_subject: str = "commit-demo-a",
    baseline_subject: str = "baseline-demo",
    candidate_samples: Any = (9.0,) * 5,
    baseline_samples: Any = (10.0,) * 5,
    prefix: str = "synthetic",
    session_ids: Sequence[str | None] | None = None,
    pair_ids: Sequence[str | None] | None = None,
    candidate_overrides: Sequence[dict[str, Any] | None] | None = None,
    candidate_kwargs: dict[str, Any] | None = None,
    baseline_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """Publish ``n`` trusted candidate/baseline pairs and return decide-style pair mappings.

    Every pair gets its own ``session_id``/``pair_id`` (shared by its two runs) unless ``session_ids`` /
    ``pair_ids`` are given (use repeats to model non-independent re-runs). ``candidate_samples`` /
    ``baseline_samples`` are either one tuple used for every pair or a sequence of ``n`` tuples.
    ``candidate_overrides`` gives per-pair ``implementation_overrides`` for the candidate runs.
    ``kwargs`` go to both sides of every pair; ``candidate_kwargs``/``baseline_kwargs`` to one side.
    Request ids are ``<prefix>-cand-<i>`` / ``<prefix>-base-<i>``; use a distinct ``prefix`` per call.
    """
    cand_samples = _per_pair(candidate_samples, n, "candidate_samples")
    base_samples = _per_pair(baseline_samples, n, "baseline_samples")
    pairs: list[dict[str, str]] = []
    for i in range(n):
        sid = session_ids[i] if session_ids is not None else f"{prefix}-session-{i + 1}"
        pid = pair_ids[i] if pair_ids is not None else f"{prefix}-pair-{i + 1}"
        cand_extra = dict(kwargs)
        cand_extra.update(candidate_kwargs or {})
        if candidate_overrides is not None and candidate_overrides[i] is not None:
            cand_extra["overrides"] = dict(candidate_overrides[i])
        base_extra = dict(kwargs)
        base_extra.update(baseline_kwargs or {})
        cand = trusted_run(
            store,
            request_id=f"{prefix}-cand-{i + 1}",
            subject_ref=candidate_subject,
            session_id=sid,
            pair_id=pid,
            role="candidate",
            samples=cand_samples[i],
            **cand_extra,
        )
        base = trusted_run(
            store,
            request_id=f"{prefix}-base-{i + 1}",
            subject_ref=baseline_subject,
            session_id=sid,
            pair_id=pid,
            role="baseline",
            samples=base_samples[i],
            **base_extra,
        )
        pairs.append({"candidate_run": cand.record_id, "baseline_run": base.record_id})
    return pairs


# --------------------------------------------------------------------------------------
# Fixture clones (dict level)
# --------------------------------------------------------------------------------------
def set_path(data: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Set ``data[a][b][c] = value`` for ``dotted == "a.b.c"``, creating intermediate dicts."""
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
    return data


def reconcile_run_hashes(data: dict[str, Any], *, config_hash: str | None) -> dict[str, Any]:
    """Recompute snapshot hashes, ``variant_digest`` and (with ``config_hash``) ``comparison_key`` in place."""
    payload = data["payload"]
    for group, fn in (
        ("environment", hashing.environment_hash),
        ("protocol", hashing.protocol_hash),
        ("verifier", hashing.verifier_hash),
    ):
        payload[group][f"{group}_hash"] = fn(payload[group])
    src = payload["source"]
    src["variant_digest"] = hashing.variant_digest(
        source_digest=src["source_digest"],
        entrypoint=src["entrypoint"],
        implementation_overrides=src["implementation_overrides"],
        checkout_mode=src["checkout_mode"],
    )
    if config_hash is not None:
        payload["comparison_key"] = hashing.comparison_key(
            config_hash=config_hash,
            environment_hash=payload["environment"]["environment_hash"],
            protocol_hash=payload["protocol"]["protocol_hash"],
            verifier_hash=payload["verifier"]["verifier_hash"],
            checkout_mode=src["checkout_mode"],
        )
    return data


def bundle_config_hash(bundle_dicts: list[dict[str, Any]], config_ref: str) -> str | None:
    for item in bundle_dicts:
        if item.get("record_type") == "config" and item.get("record_id") == config_ref:
            return item["payload"]["config_hash"]
    return None


def cloned_run_dict(
    bundle_dicts: list[dict[str, Any]],
    run_id: str,
    *,
    new_id: str | None = None,
    reconcile: bool = True,
    config_hash: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Deep-copied record dict of fixture run ``run_id`` under a new id.

    ``overrides`` keys name payload fields; ``a__b__c=value`` sets ``payload.a.b.c``. The keys
    ``record_id``/``created_at`` address the record envelope. With ``reconcile`` (default) the
    environment/protocol/verifier hashes, ``variant_digest`` and ``comparison_key`` (from the bundle's
    config hash, or ``config_hash``) are recomputed so the clone is self-consistent; pass
    ``reconcile=False`` to build a deliberately inconsistent record.
    """
    data = record_dict(bundle_dicts, run_id)
    clone_id = new_id or f"{run_id}-clone"
    data["record_id"] = clone_id
    data["payload"]["request_id"] = f"request-{clone_id}"
    for key, value in overrides.items():
        if key in ("record_id", "created_at", "schema_version"):
            data[key] = value
            continue
        set_path(data["payload"], key.replace("__", "."), copy.deepcopy(value))
    if reconcile:
        cfg_hash = config_hash if config_hash is not None else bundle_config_hash(bundle_dicts, data["payload"]["config_ref"])
        reconcile_run_hashes(data, config_hash=cfg_hash)
    return data


def publish_dict(store: MemoryStore, data: dict[str, Any], *, label: str | None = None) -> Record:
    """Validate a record dict, publish it, and return the Record."""
    record = Record.from_dict(json.loads(json.dumps(data)))
    store.publish(record, label=label)
    return record


__all__ = [
    "CONFIG_REF",
    "DEFAULT_SAMPLES",
    "ENTRYPOINT",
    "PROTOCOL",
    "VERIFIER",
    "SyntheticAdapter",
    "bundle_config_hash",
    "cloned_run_dict",
    "publish_dict",
    "reconcile_run_hashes",
    "set_path",
    "trusted_pairs",
    "trusted_run",
]
