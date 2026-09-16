"""Optimization service: wire a planner, runner, budget, and policy into one job.

Public API
----------
``run_optimization(store, config_ref, *, planner_name, budget=None, policy=None,
                   dry_run=False, permissions, adapters=None, problem_registry=None,
                   job_id=None, backend, protocol, verifier, subject_ref,
                   baseline_run_ref=None, repo_uid, entrypoint, target_commit) -> dict``
    * ``planner_name``: ``"mock"`` -> ``MockPlanner``; ``"model"`` -> ``UnavailableModelPlanner``
      (raises ``MODEL_PROVIDER_NOT_CONFIGURED`` on its first proposal; the loop stops
      finitely and reports it).
    * ``budget`` defaults to ``Budget()``; ``policy`` defaults to
      ``services.policy.default_policy()`` when available, else the golden policy vector.
    * Requests are ``request-<job_id>-r<round>`` with idempotency key
      ``jcs(job_id, round_no, implementation_overrides)``; protocol and verifier are
      fixed for the whole job (never changed between rounds).
    * Returns ``JobReport.to_dict()`` plus a ``note`` stating that MockPlanner exercises
      orchestration only and is not evidence of optimization effectiveness.
    * Optional ``comparator`` / ``decider`` / ``context_exporter`` / ``runner`` keyword
      arguments are passed through to the coordinator (tests inject stubs; production
      leaves them ``None`` so the coordinator lazily wires ``services.compare``,
      ``services.decide`` and ``services.context``).
``default_adapter_registry(backend) -> AdapterRegistry``  best-effort registration of the
    project's adapter for ``backend`` (``kernel_memory.adapters.<module>``); an empty
    registry makes the runner report ``BACKEND_UNAVAILABLE`` rather than guessing.
``PLANNERS``  names accepted by ``planner_name``.
"""
from __future__ import annotations

import importlib
import inspect
import uuid
from typing import Any, Callable

from ..adapters.base import AdapterRegistry, RunRequestSpec, SourceSpec
from ..domain.errors import InputError
from ..domain.hashing import jcs_digest
from ..domain.models import GitOid
from ..domain.problems import ProblemRegistry
from ..execution.coordinator import OptimizationCoordinator, resolve_policy
from ..execution.planner import MockPlanner, UnavailableModelPlanner
from ..execution.runner import LocalRunner
from ..execution.types import Budget, Proposal
from ..storage.store import MemoryStore

PLANNERS: tuple[str, ...] = ("mock", "model")
MOCK_PLANNER_NOTE = "MockPlanner exercises orchestration only; not evidence of optimization effectiveness"
_BACKEND_MODULES: dict[str, tuple[str, ...]] = {
    "mock": ("kernel_memory.adapters.mock",),
    "cpu": ("kernel_memory.adapters.cpu_demo",),
    "jax_tpu": ("kernel_memory.adapters.jax_tpu",),
}


def make_planner(planner_name: str) -> Any:
    if planner_name == "mock":
        return MockPlanner()
    if planner_name == "model":
        return UnavailableModelPlanner()
    raise InputError(f"unknown planner {planner_name!r}", code="UNKNOWN_PLANNER", details={"known": list(PLANNERS)})


def default_adapter_registry(backend: str) -> AdapterRegistry:
    """Register the project adapter for ``backend`` if its module exists; never substitute another backend."""
    registry = AdapterRegistry()
    for module_name in _BACKEND_MODULES.get(backend, ()):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != module.__name__ or getattr(candidate, "backend", None) != backend:
                continue
            try:
                registry.register_kernel_adapter(candidate())
            except TypeError:
                continue  # needs constructor arguments; the CLI must build it explicitly
            break
    return registry


def _as_git_oid(value: GitOid | dict[str, Any]) -> GitOid:
    if isinstance(value, GitOid):
        return value
    if isinstance(value, dict) and isinstance(value.get("algorithm"), str) and isinstance(value.get("hex"), str):
        return GitOid(algorithm=value["algorithm"], hex=value["hex"])
    raise InputError("target_commit must be a GitOid or {algorithm, hex} object", code="INVALID_GIT_OID")


def run_optimization(
    store: MemoryStore,
    config_ref: str,
    *,
    planner_name: str,
    budget: Budget | None = None,
    policy: dict[str, Any] | None = None,
    dry_run: bool = False,
    permissions: dict[str, Any],
    adapters: AdapterRegistry | None = None,
    problem_registry: ProblemRegistry | None = None,
    job_id: str | None = None,
    backend: str,
    protocol: dict[str, Any],
    verifier: dict[str, Any],
    subject_ref: str,
    baseline_run_ref: str | None = None,
    repo_uid: str,
    entrypoint: str,
    target_commit: GitOid | dict[str, Any],
    comparator: Callable[..., Any] | None = None,
    decider: Callable[..., Any] | None = None,
    context_exporter: Callable[..., Any] | None = None,
    runner: LocalRunner | None = None,
) -> dict[str, Any]:
    if not isinstance(protocol, dict) or not isinstance(verifier, dict):
        raise InputError("protocol and verifier must be objects", code="INVALID_REQUEST_TEMPLATE")
    planner = make_planner(planner_name)
    budget = budget if budget is not None else Budget()
    budget.validate()
    policy_dict = resolve_policy(policy)
    permissions = dict(permissions or {})
    job = job_id or f"job-{uuid.uuid4().hex[:12]}"
    registry = adapters if adapters is not None else default_adapter_registry(backend)
    target = _as_git_oid(target_commit)
    fixed_protocol = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    fixed_verifier = {k: v for k, v in verifier.items() if k != "verifier_hash"}
    store.require(config_ref, "config")
    store.require(subject_ref, "commit", "baseline")

    def request_factory(proposal: Proposal, round_no: int, executed_subject: str) -> RunRequestSpec:
        overrides = dict(proposal.implementation_overrides)
        return RunRequestSpec(
            request_id=f"request-{job}-r{round_no}",
            idempotency_key=jcs_digest({"job_id": job, "round_no": round_no, "implementation_overrides": overrides}),
            subject_ref=executed_subject,
            config_ref=config_ref,
            backend=backend,
            stage="benchmark",
            protocol=dict(fixed_protocol),
            verifier=dict(fixed_verifier),
            source=SourceSpec(
                repo_uid=repo_uid,
                target_commit=target,
                entrypoint=entrypoint,
                checkout_mode="exact_commit",
                implementation_overrides=overrides,
            ),
            authorization={"allow_tpu_execution": bool(permissions.get("allow_tpu_execution", False))},
            session_id=job,
            pair_id=f"{job}-r{round_no}",
            role_in_pair="candidate",
            metadata={"proposal_id": proposal.proposal_id, "planner_id": proposal.planner_id, "round_no": round_no},
        )

    if runner is None:
        runner = LocalRunner(store, adapters=registry, problem_registry=problem_registry, permissions=permissions)
    coordinator = OptimizationCoordinator(
        store,
        runner=runner,
        planner=planner,
        budget=budget,
        policy=policy_dict,
        job_id=job,
        config_ref=config_ref,
        permissions=permissions,
        request_factory=request_factory,
        baseline_run_ref=baseline_run_ref,
        comparator=comparator,
        decider=decider,
        context_exporter=context_exporter,
        subject_ref=subject_ref,
        dry_run=dry_run,
    )
    report = coordinator.run().to_dict()
    report["note"] = MOCK_PLANNER_NOTE
    report["policy"] = policy_dict
    report["backend"] = backend
    return report


__all__ = ["run_optimization", "make_planner", "default_adapter_registry", "PLANNERS", "MOCK_PLANNER_NOTE"]
