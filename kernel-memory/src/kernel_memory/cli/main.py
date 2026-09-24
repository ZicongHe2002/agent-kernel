"""``kmem`` command-line interface.

Every command is a thin wrapper over an application service. Results are JSON on
stdout; diagnostics go to stderr. Exit codes follow the specification:

    0 success · 2 input/schema · 3 reference/idempotency conflict · 4 incomparable/insufficient
    evidence · 5 unavailable backend/prerequisite · 6 execution infrastructure · 7 authorization

Commands never edit published records; they append new ones.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from ..domain.errors import InputError, KernelMemoryError, NotComparableError, PrerequisiteMissingError
from ..domain.jsonio import load_json_file, loads_strict
from ..domain.models import GitOid, to_json
from ..storage import MemoryStore

CommandResult = dict[str, Any] | tuple[dict[str, Any], int]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _root(args: argparse.Namespace) -> Path:
    if getattr(args, "root", None):
        return Path(args.root)
    settings = _settings(args)
    if settings is not None:
        return settings.memory_root
    env = os.environ.get("KMEM_ROOT")
    return Path(env) if env else Path("./memory")


_SETTINGS_CACHE: dict[str, Any] = {}


def _settings(args: argparse.Namespace):
    path = getattr(args, "settings", None) or os.environ.get("KMEM_SETTINGS")
    if not path:
        return None
    if path not in _SETTINGS_CACHE:
        from ..settings import load_settings

        _SETTINGS_CACHE[path] = load_settings(Path(path))
    return _SETTINGS_CACHE[path]


def _permissions(args: argparse.Namespace) -> dict[str, bool]:
    settings = _settings(args)
    if settings is not None:
        return dict(settings.permissions)
    from ..settings import DEFAULT_PERMISSIONS

    return dict(DEFAULT_PERMISSIONS)


def _open_store(args: argparse.Namespace) -> MemoryStore:
    return MemoryStore.open(_root(args))


def _json_arg(value: str | None, *, what: str) -> Any:
    """Parse an inline JSON value or ``@path`` to a JSON file."""
    if value is None:
        return None
    if value.startswith("@"):
        return load_json_file(Path(value[1:]))
    try:
        return loads_strict(value)
    except KernelMemoryError as exc:
        raise InputError(f"{what} must be inline JSON or @file: {exc}") from exc


def _git_oid(value: str) -> GitOid:
    """Accept ``sha1:<40hex>``, ``sha256:<64hex>``, a bare 40/64-hex string, or ``cpu-demo-source``."""
    if value == "cpu-demo-source":
        from ..adapters.cpu_demo import current_source_commit

        return current_source_commit()
    if ":" in value:
        algorithm, _, hex_value = value.partition(":")
    else:
        hex_value = value
        algorithm = {40: "sha1", 64: "sha256"}.get(len(hex_value), "")
    hex_value = hex_value.lower()
    if algorithm not in ("sha1", "sha256") or not all(c in "0123456789abcdef" for c in hex_value):
        raise InputError(f"invalid git object id {value!r}; use sha1:<40 hex> or sha256:<64 hex>")
    if (algorithm == "sha1" and len(hex_value) != 40) or (algorithm == "sha256" and len(hex_value) != 64):
        raise InputError(f"git object id {value!r} has the wrong length for {algorithm}; short SHAs are not accepted here")
    return GitOid(algorithm, hex_value)


def _to_dict(obj: Any) -> Any:
    """JSON-ready view of service results: to_dict() when offered, dataclasses via models.to_json."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_json(obj)
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def _record_summary(record: Any) -> dict[str, Any]:
    return {"record_id": record.record_id, "record_type": record.record_type, "created_at": record.created_at}


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> CommandResult:
    root = _root(args)
    existed = (root / "manifest.json").is_file()
    store = MemoryStore.init(root)
    return {"root": str(store.root), "created": not existed, "manifest": store.manifest()}


def cmd_validate(args: argparse.Namespace) -> CommandResult:
    store = _open_store(args)
    if args.deep:
        from ..services.validation import deep_validate

        report = deep_validate(store, verify_artifacts=not args.skip_artifacts)
        result = _to_dict(report)
        result["deep"] = True
        return (result, 0 if report.ok else 2)
    scan = store.integrity_scan(verify_artifacts=not args.skip_artifacts)
    result = {"deep": False, "records": len(store.index_entries()), "integrity": scan.to_dict()}
    return (result, 0 if scan.ok else 2)


def cmd_integrity(args: argparse.Namespace) -> CommandResult:
    store = _open_store(args)
    scan = store.integrity_scan()
    return (scan.to_dict(), 0 if scan.ok else 2)


def cmd_import_bundle(args: argparse.Namespace) -> CommandResult:
    from ..services.importer import import_bundle

    store = _open_store(args)
    report = import_bundle(
        store,
        Path(args.file),
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        allow_fixture=args.allow_fixture,
        allow_missing_artifacts=args.allow_missing_artifacts,
        trusted_source=False,
        dry_run=args.dry_run,
    )
    return _to_dict(report)


def cmd_register_kernel(args: argparse.Namespace) -> CommandResult:
    from ..services.register import register_kernel

    store = _open_store(args)
    record = register_kernel(store, args.kernel_id, args.display_name, args.adapter_id, args.notes)
    return {"kernel": _record_summary(record), "kernel_id": record.payload.kernel_id}


def cmd_register_config(args: argparse.Namespace) -> CommandResult:
    from ..services.register import register_config

    store = _open_store(args)
    problem = _json_arg(args.problem, what="--problem")
    if not isinstance(problem, dict):
        raise InputError("--problem must be a JSON object")
    record, created = register_config(store, args.kernel_id, problem, tags=args.tag or [])
    return {
        "config": _record_summary(record),
        "created": created,
        "config_hash": record.payload.config_hash,
        "config_id": record.payload.config_id,
        "problem": record.payload.problem,
    }


def cmd_register_local(args: argparse.Namespace) -> CommandResult:
    from ..services.register import describe_pr, register_local_trial

    store = _open_store(args)
    record = register_local_trial(store, args.config, args.title, repo_uid=args.repo_uid, hypothesis=args.hypothesis, origin_ref=args.origin)
    return {"pr": _record_summary(record), **describe_pr(record)}


def cmd_register_pr(args: argparse.Namespace) -> CommandResult:
    from ..services.register import describe_pr, register_pr_context

    store = _open_store(args)
    record = register_pr_context(
        store,
        args.config,
        repo_uid=args.repo_uid,
        provider="github",
        number=args.number,
        title=args.title,
        hypothesis=args.hypothesis,
        origin_ref=args.origin,
    )
    return {"pr": _record_summary(record), **describe_pr(record)}


def cmd_add_baseline(args: argparse.Namespace) -> CommandResult:
    from ..services.register import add_baseline

    store = _open_store(args)
    record = add_baseline(
        store,
        args.config,
        args.baseline_id,
        description=args.description,
        repo_uid=args.repo_uid,
        commit_oid=_git_oid(args.commit),
        entrypoint=args.entrypoint,
        role=args.role,
    )
    return {"baseline": _record_summary(record), "commit_oid": {"algorithm": record.payload.commit_oid.algorithm, "hex": record.payload.commit_oid.hex}}


def cmd_record_commit(args: argparse.Namespace) -> CommandResult:
    from ..adapters.git_local import LocalGitRepo
    from ..services.register import record_commit

    store = _open_store(args)
    repo = LocalGitRepo(Path(args.repo))
    changes = _json_arg(args.changes, what="--changes")
    record = record_commit(
        store,
        args.context,
        args.sha,
        repo=repo,
        changes=changes,
        summary=args.summary,
        summary_author=args.summary_author,
        store_diff=not args.no_diff,
    )
    p = record.payload
    return {
        "commit": _record_summary(record),
        "commit_oid": {"algorithm": p.commit_oid.algorithm, "hex": p.commit_oid.hex},
        "parents": [{"algorithm": o.algorithm, "hex": o.hex} for o in p.git_parent_oids],
        "change_status": p.change_status,
        "changes": len(p.changes),
        "diff_artifact_ref": p.diff_artifact_ref,
    }


def cmd_relate(args: argparse.Namespace) -> CommandResult:
    from ..services.register import add_relation

    store = _open_store(args)
    record = add_relation(store, args.config, args.kind, args.from_ref, args.to_ref, rationale=args.rationale, evidence_refs=tuple(args.evidence or ()))
    return {"relation": _record_summary(record), "kind": record.payload.kind}


def cmd_annotate(args: argparse.Namespace) -> CommandResult:
    from ..services.register import annotate

    store = _open_store(args)
    record = annotate(
        store,
        args.target,
        args.category,
        args.text,
        author_kind=args.author_kind,
        evidence_refs=tuple(args.evidence or ()),
        confidence=args.confidence,
        supersedes_ref=args.supersedes,
    )
    return {"annotation": _record_summary(record), "target_ref": record.payload.target_ref}


def cmd_collect_pr(args: argparse.Namespace) -> CommandResult:
    from ..adapters.github import FixtureTransport, GitHubClient, Response, UrllibTransport
    from ..services.collect import collect_pr

    store = _open_store(args)
    owner, _, repo = args.repo.partition("/")
    if not owner or not repo:
        raise InputError("--repo must be OWNER/REPO")
    settings = _settings(args)
    permissions = _permissions(args)
    if args.offline_fixture:
        raw = load_json_file(Path(args.offline_fixture))
        routes = {
            url: [Response(status=r["status"], headers=dict(r.get("headers", {})), body=json.dumps(r["body"]).encode("utf-8") if not isinstance(r["body"], str) else r["body"].encode("utf-8")) for r in responses]
            for url, responses in raw.items()
        }
        client = GitHubClient(FixtureTransport(routes), token=None, allow_network=False)
        transport_kind = "offline_fixture"
    else:
        token = settings.github_token() if settings is not None else os.environ.get("GITHUB_TOKEN")
        if not permissions.get("allow_network", False):
            raise PrerequisiteMissingError(
                "live GitHub collection requires permissions.allow_network=true in the project settings",
                code="NETWORK_NOT_AUTHORIZED",
                details={"missing": ["permissions.allow_network"] + ([] if token else ["GITHUB_TOKEN"])},
            )
        client = GitHubClient(UrllibTransport(), token=token, allow_network=True)
        transport_kind = "live_read_only"
    local_repo = None
    if args.local_repo:
        from ..adapters.git_local import LocalGitRepo

        local_repo = LocalGitRepo(Path(args.local_repo))
    report = collect_pr(store, client, owner=owner, repo=repo, number=args.number, config_ref=args.config, local_repo=local_repo, hypothesis=args.hypothesis, origin_ref=args.origin)
    result = _to_dict(report)
    result["transport"] = transport_kind
    return result


def _load_protocol_verifier(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.protocol:
        protocol = load_json_file(Path(args.protocol))
    else:
        from ..adapters.cpu_demo import default_cpu_protocol

        protocol = default_cpu_protocol(repetitions=args.repetitions, warmup=args.warmup)
    if args.verifier:
        verifier = load_json_file(Path(args.verifier))
    else:
        from ..adapters.cpu_demo import default_cpu_verifier

        verifier = default_cpu_verifier()
    if not isinstance(protocol, dict) or not isinstance(verifier, dict):
        raise InputError("protocol and verifier files must contain JSON objects")
    protocol.pop("protocol_hash", None)
    verifier.pop("verifier_hash", None)
    return protocol, verifier


def cmd_run(args: argparse.Namespace) -> CommandResult:
    from ..adapters.base import RunRequestSpec, SourceSpec
    from ..adapters.registry import default_adapter_registry
    from ..domain.hashing import jcs_digest
    from ..domain.problems import default_registry
    from ..execution.runner import LocalRunner
    from ..domain.errors import BackendUnavailable

    store = _open_store(args)
    subject = store.require(args.subject, "commit", "baseline")
    if subject.record_type == "commit":
        from ..services.common import config_of

        config_ref = config_of(store, subject).record_id
        repo_uid = subject.payload.repo_uid
        entrypoint = args.entrypoint
        if entrypoint is None:
            raise InputError("--entrypoint is required when the subject is a commit")
    else:
        config_ref = subject.payload.config_ref
        repo_uid = subject.payload.repo_uid
        entrypoint = args.entrypoint or subject.payload.entrypoint
    protocol, verifier = _load_protocol_verifier(args)
    overrides = _json_arg(args.overrides, what="--overrides") or {}
    if not isinstance(overrides, dict):
        raise InputError("--overrides must be a JSON object")
    request_id = args.request_id or f"request-{jcs_digest({'subject': args.subject, 'backend': args.backend, 'overrides': overrides, 'session': args.session_id, 'pair': args.pair_id, 'nonce': args.nonce})[7:23]}"
    idempotency_key = args.idempotency_key or request_id
    permissions = _permissions(args)
    spec = RunRequestSpec(
        request_id=request_id,
        idempotency_key=idempotency_key,
        subject_ref=args.subject,
        config_ref=config_ref,
        backend=args.backend,
        stage=args.stage,
        protocol=protocol,
        verifier=verifier,
        source=SourceSpec(
            repo_uid=repo_uid,
            target_commit=subject.payload.commit_oid,
            entrypoint=entrypoint,
            checkout_mode=args.checkout_mode,
            implementation_overrides=overrides,
            repo_path=args.repo_path,
        ),
        authorization={"allow_tpu_execution": bool(permissions.get("allow_tpu_execution", False))},
        session_id=args.session_id,
        pair_id=args.pair_id,
        role_in_pair=args.role,
        max_wall_seconds=args.max_wall_seconds,
        rerun_of=args.rerun_of,
    )
    runner = LocalRunner(store, adapters=default_adapter_registry(), problem_registry=default_registry(), permissions=permissions)
    try:
        outcome, run = runner.submit_and_execute(spec)
    except BackendUnavailable as exc:
        result = LocalRunner.describe_unavailable(args.backend, exc)
        result["request_id"] = request_id
        return (result, exc.exit_code)
    result: dict[str, Any] = {"request": _to_dict(outcome), "run": None}
    if run is not None:
        p = run.payload
        result["run"] = {
            "record_id": run.record_id,
            "provenance": p.provenance,
            "execution_status": p.execution_status,
            "failure_reason": p.failure_reason,
            "correctness": _to_dict(p.correctness),
            "timing": _to_dict(p.timing),
            "comparison_key": p.comparison_key,
            "variant_digest": p.source.variant_digest,
            "environment_backend": p.environment.backend,
            "artifacts": [a.artifact_id for a in p.artifacts],
        }
    return result


def cmd_compare(args: argparse.Namespace) -> CommandResult:
    from ..services.compare import compare_in_store

    store = _open_store(args)
    result = compare_in_store(store, args.candidate, args.baseline)
    data = _to_dict(result)
    return (data, 0 if getattr(result, "status", data.get("status")) == "COMPARABLE" else NotComparableError.exit_code)


def cmd_decide(args: argparse.Namespace) -> CommandResult:
    from ..services.decide import append_decision, evaluate_candidate
    from ..services.policy import default_policy, load_policy

    store = _open_store(args)
    pairs = load_json_file(Path(args.pairs))
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs", [])
    if not isinstance(pairs, list):
        raise InputError("--pairs file must contain a list of {candidate_run, baseline_run} objects")
    policy = load_policy(Path(args.policy)) if args.policy else default_policy()
    draft = evaluate_candidate(store, args.candidate, pairs, policy)
    result = _to_dict(draft)
    if not args.dry_run:
        record = append_decision(store, draft, supersedes_decision_ref=args.supersedes)
        result["decision"] = _record_summary(record)
    outcome = getattr(draft, "outcome", result.get("outcome"))
    return (result, 0 if outcome == "accepted" else 4)


def cmd_trajectory(args: argparse.Namespace) -> CommandResult:
    from ..services import trajectory as traj

    store = _open_store(args)
    if args.verify:
        return _to_dict(traj.verify_trajectory(store, args.config))
    if args.rebuild:
        result = _to_dict(traj.rebuild_trajectory(store, args.config, force=args.force))
    else:
        result = _to_dict(traj.publish_trajectory(store, args.config, force=args.force))
    if args.print_view:
        result["view"] = traj.build_trajectory(store, args.config)
    return result


def cmd_query(args: argparse.Namespace) -> CommandResult:
    from ..services.query import QueryFilters, query_memory

    store = _open_store(args)
    filters = QueryFilters(
        kernel_id=args.kernel,
        config_ref=args.config,
        config_hash=args.config_hash,
        record_type=args.record_type,
        component=args.component,
        parameter_key=args.parameter,
        subject_ref=args.subject,
        pr_ref=args.pr,
        execution_status=args.execution_status,
        correctness_status=args.correctness_status,
        provenance=args.provenance,
        comparison_key=args.comparison_key,
        decision_outcome=args.decision_outcome,
        reason_code=args.reason_code,
        run_status=args.run_status,
        include_cross_config_hints=args.cross_config_hints,
        limit=args.limit,
    )
    return _to_dict(query_memory(store, filters))


def cmd_export_context(args: argparse.Namespace) -> CommandResult:
    from ..services.context import export_context

    store = _open_store(args)
    return export_context(store, args.config, max_records=args.max_records, policy_hash=args.policy_hash)


def cmd_optimize(args: argparse.Namespace) -> CommandResult:
    from ..adapters.registry import default_adapter_registry
    from ..domain.problems import default_registry
    from ..execution.types import Budget
    from ..services.optimize import run_optimization
    from ..services.policy import default_policy, load_policy

    store = _open_store(args)
    subject = store.require(args.subject, "commit", "baseline")
    if subject.record_type == "baseline":
        entrypoint = args.entrypoint or subject.payload.entrypoint
        repo_uid = subject.payload.repo_uid
    else:
        entrypoint = args.entrypoint
        repo_uid = subject.payload.repo_uid
        if entrypoint is None:
            raise InputError("--entrypoint is required when the subject is a commit")
    protocol, verifier = _load_protocol_verifier(args)
    settings = _settings(args)
    if args.budget:
        raw = load_json_file(Path(args.budget))
        budget = Budget(**raw)
        budget.validate()
    elif settings is not None:
        budget = settings.budget
    else:
        budget = Budget()
    policy = load_policy(Path(args.policy)) if args.policy else default_policy()
    report = run_optimization(
        store,
        args.config,
        planner_name=args.planner,
        budget=budget,
        policy=policy,
        dry_run=args.dry_run,
        permissions=_permissions(args),
        adapters=default_adapter_registry(),
        problem_registry=default_registry(),
        job_id=args.job_id,
        backend=args.backend,
        protocol=protocol,
        verifier=verifier,
        subject_ref=args.subject,
        baseline_run_ref=args.baseline_run,
        repo_uid=repo_uid,
        entrypoint=entrypoint,
        target_commit=subject.payload.commit_oid,
    )
    return _to_dict(report)


def cmd_recover(args: argparse.Namespace) -> CommandResult:
    store = MemoryStore.open(_root(args), recover=False)
    report = store.recover()
    result = {"store": report.to_dict()}
    try:
        from ..execution.ledger import RequestLedger

        result["requests"] = _to_dict(RequestLedger(store).reconcile())
    except ImportError as exc:  # pragma: no cover - ledger module absent
        result["requests"] = {"skipped": str(exc)}
    return result


def cmd_reindex(args: argparse.Namespace) -> CommandResult:
    store = _open_store(args)
    path = store.rebuild_index()
    return {"index": str(path), "records": len(store.index_entries())}


def cmd_migrate_v01(args: argparse.Namespace) -> CommandResult:
    from ..domain.problems import default_registry
    from ..migrations.v01 import MappingResolver, NullResolver, migrate_v01

    # A dry run still opens an existing store (read-only use: T01 reuse of existing kernels/configs and
    # ID_CONFLICT pre-checks); a dry run against a non-existent root simply migrates without a store.
    root = _root(args)
    if args.dry_run:
        store = _open_store(args) if (root / "manifest.json").is_file() else None
    else:
        store = _open_store(args)
    repo_uid_map = _json_arg(args.repo_uid_map, what="--repo-uid-map") or {}
    if not isinstance(repo_uid_map, dict):
        raise InputError("--repo-uid-map must be a JSON object mapping v0.1 repo names to repo_uid values")
    resolver: Any = NullResolver()
    if args.resolver:
        kind, _, value = args.resolver.partition(":")
        if kind == "map":
            mapping = _json_arg(value, what="--resolver map:@file")
            if not isinstance(mapping, dict):
                raise InputError("--resolver map:@file must be a JSON object of \"repo_uid|shortsha\": full_hex")
            pairs: dict[tuple[str, str], str] = {}
            for key, full_hex in mapping.items():
                if not isinstance(key, str) or "|" not in key or not isinstance(full_hex, str):
                    raise InputError(
                        f"invalid resolver mapping entry {key!r}: keys must be \"repo_uid|shortsha\" and values full hex",
                        code="INVALID_RESOLVER_MAP",
                    )
                repo_uid, _, prefix = key.partition("|")
                pairs[(repo_uid, prefix)] = full_hex
            resolver = MappingResolver(pairs)
        elif kind == "git":
            from ..adapters.git_local import LocalGitRepo

            repo = LocalGitRepo(Path(value))

            class _GitResolver:
                def resolve(self, repo_uid: str, sha: str):
                    try:
                        return repo.rev_parse(sha)
                    except KernelMemoryError:
                        return None

                def is_ambiguous(self, repo_uid: str, sha: str) -> bool:
                    try:
                        repo.rev_parse(sha)
                    except KernelMemoryError as exc:
                        return exc.code == "AMBIGUOUS_REVISION"
                    return False

            resolver = _GitResolver()
        else:
            raise InputError("--resolver must be map:@file.json or git:<repo path>")
    report = migrate_v01(
        Path(args.path),
        store=store,
        registry=default_registry(),
        resolver=resolver,
        repo_uid_map=repo_uid_map,
        dry_run=args.dry_run,
        created_at=args.created_at,
    )
    return _to_dict(report)


def cmd_status(args: argparse.Namespace) -> CommandResult:
    from ..settings import default_settings, pending_integrations

    settings = _settings(args) or default_settings(_root(args))
    jax_installed = False
    tpu_available = False
    try:
        import jax  # type: ignore

        jax_installed = True
        tpu_available = jax.default_backend() == "tpu"
    except Exception:
        pass
    result: dict[str, Any] = {
        "root": str(_root(args)),
        "pending_integrations": pending_integrations(settings, tpu_available=tpu_available, jax_installed=jax_installed),
        "jax_installed": jax_installed,
        "tpu_available": tpu_available,
    }
    root = _root(args)
    if (root / "manifest.json").is_file():
        store = MemoryStore.open(root)
        counts: dict[str, int] = {}
        for entry in store.index_entries():
            counts[entry.record_type] = counts.get(entry.record_type, 0) + 1
        result["records"] = counts
    return result


def cmd_cpu_demo_defaults(args: argparse.Namespace) -> CommandResult:
    from ..adapters.cpu_demo import current_source_commit, default_cpu_protocol, default_cpu_verifier

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = default_cpu_protocol(repetitions=args.repetitions, warmup=args.warmup)
    verifier = default_cpu_verifier()
    (out / "cpu_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "cpu_verifier.json").write_text(json.dumps(verifier, indent=2, sort_keys=True) + "\n")
    oid = current_source_commit()
    return {
        "protocol": str(out / "cpu_protocol.json"),
        "verifier": str(out / "cpu_verifier.json"),
        "source_commit": {"algorithm": oid.algorithm, "hex": oid.hex},
        "note": "source_commit is the content address of the loaded CPU demo source, not a Git commit",
    }


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kmem", description="Kernel Memory CLI")
    parser.add_argument("--root", default=None, help="store root (default: settings memory_root, $KMEM_ROOT, or ./memory)")
    parser.add_argument("--settings", default=None, help="project settings JSON (default: $KMEM_SETTINGS)")
    parser.add_argument("--json", action="store_true", default=False, help="compact single-line JSON output")
    # The same global options are accepted after the subcommand; SUPPRESS keeps a subcommand's
    # unset option from clobbering a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS)
    common.add_argument("--settings", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable[[argparse.Namespace], CommandResult], help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, parents=[common])
        p.set_defaults(func=func)
        return p

    add("init", cmd_init, "create a store without overwriting a project")
    p = add("validate", cmd_validate, "validate schemas, references, hashes, samples, artifacts, invariants")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--skip-artifacts", action="store_true")
    add("integrity", cmd_integrity, "journal-based integrity scan")
    p = add("import-bundle", cmd_import_bundle, "idempotent bundle import")
    p.add_argument("file")
    p.add_argument("--allow-fixture", action="store_true")
    p.add_argument("--allow-missing-artifacts", action="store_true")
    p.add_argument("--artifact-root")
    p.add_argument("--dry-run", action="store_true")
    p = add("register-kernel", cmd_register_kernel, "register a kernel (name)")
    p.add_argument("--kernel-id", required=True)
    p.add_argument("--display-name", required=True)
    p.add_argument("--adapter-id", required=True)
    p.add_argument("--notes", default="")
    p = add("register-config", cmd_register_config, "register a normalized computational problem")
    p.add_argument("--kernel-id", required=True)
    p.add_argument("--problem", required=True, help="inline JSON or @file")
    p.add_argument("--tag", action="append")
    p = add("register-local", cmd_register_local, "register a local trial context (not a GitHub PR)")
    p.add_argument("--config", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--repo-uid", required=True)
    p.add_argument("--hypothesis")
    p.add_argument("--origin")
    p = add("register-pr", cmd_register_pr, "register a GitHub PR context without collecting")
    p.add_argument("--config", required=True)
    p.add_argument("--repo-uid", required=True)
    p.add_argument("--number", type=int, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--hypothesis")
    p.add_argument("--origin")
    p = add("add-baseline", cmd_add_baseline, "register a fixed reference implementation")
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-id", required=True)
    p.add_argument("--repo-uid", required=True)
    p.add_argument("--commit", required=True, help="sha1:<hex> | sha256:<hex> | cpu-demo-source")
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--role", default="both", choices=["reference", "performance_anchor", "both"])
    p.add_argument("--description", default="")
    p = add("record-commit", cmd_record_commit, "inspect a real commit and bind it to a PR context")
    p.add_argument("--context", required=True, help="PR record id")
    p.add_argument("--sha", required=True)
    p.add_argument("--repo", required=True, help="local git repository path")
    p.add_argument("--changes", help="inline JSON list or @file of Change objects")
    p.add_argument("--summary")
    p.add_argument("--summary-author", default="human", choices=["human", "agent", "collector"])
    p.add_argument("--no-diff", action="store_true")
    p = add("relate", cmd_relate, "append an explicit relation")
    p.add_argument("--config", required=True)
    p.add_argument("--kind", required=True, choices=["optimization_origin", "inspired_by", "rebased_from"])
    p.add_argument("--from", dest="from_ref", required=True)
    p.add_argument("--to", dest="to_ref", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--evidence", action="append")
    p = add("annotate", cmd_annotate, "append an interpretation without modifying observations")
    p.add_argument("--target", required=True)
    p.add_argument("--category", required=True, choices=["hypothesis", "lesson", "correction", "note"])
    p.add_argument("--text", required=True)
    p.add_argument("--author-kind", default="human", choices=["human", "agent", "program"])
    p.add_argument("--evidence", action="append")
    p.add_argument("--confidence", default="unverified", choices=["unverified", "supported", "contradicted"])
    p.add_argument("--supersedes")
    p = add("collect-pr", cmd_collect_pr, "read-only PR synchronization with coverage reporting")
    p.add_argument("--repo", required=True, help="OWNER/REPO")
    p.add_argument("--number", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--local-repo")
    p.add_argument("--hypothesis")
    p.add_argument("--origin")
    p.add_argument("--offline-fixture", help="JSON file of canned responses (url -> [ {status, headers, body} ])")
    p = add("run", cmd_run, "submit and execute a run, or report an unavailable backend")
    p.add_argument("--subject", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--protocol")
    p.add_argument("--verifier")
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--entrypoint")
    p.add_argument("--overrides", help="inline JSON object or @file")
    p.add_argument("--checkout-mode", default="exact_commit", choices=["exact_commit", "integration_merge"])
    p.add_argument("--repo-path")
    p.add_argument("--stage", default="benchmark", choices=["compile", "verify", "benchmark", "profile"])
    p.add_argument("--request-id")
    p.add_argument("--idempotency-key")
    p.add_argument("--nonce", default="")
    p.add_argument("--session-id")
    p.add_argument("--pair-id")
    p.add_argument("--role", choices=["baseline", "candidate"])
    p.add_argument("--rerun-of")
    p.add_argument("--max-wall-seconds", type=float, default=600.0)
    p = add("compare", cmd_compare, "comparability diagnostics and verified derived values")
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p = add("decide", cmd_decide, "apply a fixed policy and append a Decision")
    p.add_argument("--candidate", required=True, help="candidate subject (commit/baseline) record id")
    p.add_argument("--pairs", required=True, help="JSON file: list of {candidate_run, baseline_run}")
    p.add_argument("--policy")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--supersedes")
    p = add("trajectory", cmd_trajectory, "deterministically (re)build the trajectory view")
    p.add_argument("--config", required=True)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--print-view", action="store_true")
    p = add("query", cmd_query, "structured retrieval")
    p.add_argument("--kernel")
    p.add_argument("--config")
    p.add_argument("--config-hash")
    p.add_argument("--record-type")
    p.add_argument("--component")
    p.add_argument("--parameter")
    p.add_argument("--subject")
    p.add_argument("--pr")
    p.add_argument("--execution-status")
    p.add_argument("--correctness-status")
    p.add_argument("--provenance")
    p.add_argument("--comparison-key")
    p.add_argument("--decision-outcome")
    p.add_argument("--reason-code")
    p.add_argument("--run-status", choices=["tested", "not_run"])
    p.add_argument("--cross-config-hints", action="store_true")
    p.add_argument("--limit", type=int)
    p = add("export-context", cmd_export_context, "export agent context with original record ids")
    p.add_argument("--config", required=True)
    p.add_argument("--max-records", type=int, default=30)
    p.add_argument("--policy-hash")
    p = add("optimize", cmd_optimize, "budgeted orchestration (MockPlanner by default; no remote changes)")
    p.add_argument("--config", required=True)
    p.add_argument("--planner", default="mock", choices=["mock", "model"])
    p.add_argument("--subject", required=True, help="baseline/commit record the candidates derive from")
    p.add_argument("--baseline-run")
    p.add_argument("--backend", default="cpu")
    p.add_argument("--entrypoint")
    p.add_argument("--protocol")
    p.add_argument("--verifier")
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--budget")
    p.add_argument("--policy")
    p.add_argument("--job-id")
    p.add_argument("--dry-run", action="store_true")
    add("recover", cmd_recover, "repair pending publication/job state with an audit report")
    add("reindex", cmd_reindex, "rebuild the disposable SQLite index")
    p = add("migrate-v01", cmd_migrate_v01, "map v0.1 data (dry-run by default)")
    p.add_argument("path")
    p.add_argument("--apply", dest="dry_run", action="store_false")
    p.add_argument("--repo-uid-map", help="inline JSON or @file mapping v0.1 repo names to repo_uid")
    p.add_argument("--resolver", help="map:@file.json (\"repo_uid|shortsha\": fullhex) or git:<repo path>")
    p.add_argument("--created-at", help="fixed RFC 3339 UTC timestamp for all migrated records (re-applying with the same value is idempotent)")
    p.set_defaults(dry_run=True)
    add("status", cmd_status, "store counts and pending integrations")
    p = add("cpu-demo-defaults", cmd_cpu_demo_defaults, "write default CPU-demo protocol/verifier files and print the demo source commit")
    p.add_argument("--out", required=True)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    return parser


def _emit(result: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    else:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outcome = args.func(args)
        if isinstance(outcome, tuple):
            result, code = outcome
        else:
            result, code = outcome, 0
        _emit(result, args)
        return code
    except KernelMemoryError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), sort_keys=True, ensure_ascii=False, default=str) + "\n")
        return exc.exit_code
    except ImportError as exc:
        sys.stderr.write(json.dumps({"error": "MODULE_UNAVAILABLE", "message": str(exc), "exit_code": 5}) + "\n")
        return 5
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(json.dumps({"error": "UNEXPECTED_ERROR", "message": f"{type(exc).__name__}: {exc}", "exit_code": 1}) + "\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
