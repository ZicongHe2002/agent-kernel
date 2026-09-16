"""Registration services: kernels, configs, PR contexts, commits, baselines, relations, annotations.

Public API (every function returns a validated, published ``Record``)
-------------------------------------------------------------------
``register_kernel(store, kernel_id, display_name, adapter_id, contract_notes) -> Record``
    id ``kernel-<kernel_id>``. An existing kernel with the same ``kernel_id`` and identical
    payload is returned; a different payload raises ``IdConflictError``.
``register_config(store, kernel_id, raw_problem, *, registry=None, tags=None) -> (Record, created)``
    Normalises through the problem registry (``IncompleteProblemContract`` for
    ``mla_forward`` propagates, exit 5). A config with the same ``config_hash`` is returned
    with ``created=False`` (T01); otherwise ``cfg-<config_id_hint>-<hash hex[:12]>`` is created.
    The kernel record must exist (``MissingReferenceError`` code ``MISSING_KERNEL``).
``register_pr_context(store, config_ref, *, repo_uid, provider, number, title, hypothesis=None,
                      origin_ref=None, pr_key=None) -> Record``
    ``github`` requires ``number >= 1`` and derives ``pr_key = gh-<repo_id>-pr-<number>`` from
    ``repo_uid = github:<host>:repo:<repo_id>``; ``local`` requires ``number is None``.
    Record id ``pr-<pr_key>``; idempotent on an identical payload, conflict otherwise.
``register_local_trial(store, config_ref, title, *, repo_uid, hypothesis=None, origin_ref=None) -> Record``
    ``provider=local, number=None, pr_key=local-<slug(title)>-<short hash>``.
``describe_pr(pr) -> dict`` display helper; local trials are labelled
    "local trial, not yet a GitHub PR".
``record_commit(store, pr_ref, sha, *, repo, changes=None, summary=None, summary_author="human",
                store_diff=True, diff_base=None, evidence_refs=(), repo_uid=None) -> Record``
    Resolves ``sha`` through the repository (short/ambiguous handled there; never padded),
    reads parents/tree, stores the diff against ``diff_base`` (explicit, else the first
    parent) as a content-addressed ``diff`` artifact (``artifact://sha256/<hex>``), and
    publishes ``commit-<pr_key>-<oid hex[:12]>``. ``changes`` are validated as ``Change``
    objects; attribution defaults to ``group_only`` and ``isolated`` requires
    ``evidence_refs`` (``InputError`` code ``ISOLATED_REQUIRES_EVIDENCE``). ``change_status``
    is ``recorded`` (changes given), ``not_extracted`` (none given) or ``no_code_change``
    (empty numstat). The summary defaults to the commit's first message line (data,
    ``summary_author=collector``). Re-recording the identical commit is idempotent; the
    same source commit under another PR yields another binding (T07).
``add_baseline(store, config_ref, baseline_id, *, description, repo_uid, commit_oid, entrypoint,
               role="both") -> Record``  id ``baseline-<baseline_id>``.
``add_relation(store, config_ref, kind, from_ref, to_ref, *, rationale, evidence_refs=()) -> Record``
    id ``relation-<kind>-<sha(from,to)[:12]>``; existing identical content is returned;
    ``from == to`` and missing endpoints are rejected.
``annotate(store, target_ref, category, text, *, author_kind, evidence_refs=(), confidence="unverified",
           supersedes_ref=None) -> Record``  appends ``annotation-<uuid hex[:16]>``; the target
    record is never modified; ``text`` is stored as data.

All text supplied by callers, commit messages, and PR titles are data. Identity hashes
come from ``domain.hashing``; timestamps from ``ids.utc_now_iso``. Errors are
``kernel_memory.domain.errors`` subclasses with machine-readable codes.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

from ..adapters.git_local import LocalGitRepo
from ..domain.errors import IdConflictError, InputError, MissingReferenceError
from ..domain.ids import short_hash_id, validate_record_id
from ..domain.models import (
    AnnotationPayload,
    ArtifactRef,
    BaselinePayload,
    Change,
    CommitPayload,
    ConfigPayload,
    GitOid,
    KernelPayload,
    PrPayload,
    Record,
    RelationPayload,
    to_json,
)
from ..domain.problems import ProblemRegistry, default_registry
from ..domain.schema import validate_nested
from ..storage.store import MemoryStore
from .common import new_record

PROVIDERS = ("github", "local")
RELATION_KINDS = ("optimization_origin", "inspired_by", "rebased_from")
SUMMARY_AUTHORS = ("human", "agent", "collector")
BASELINE_ROLES = ("reference", "performance_anchor", "both")
_REPO_UID_RE = re.compile(r"^github:(?P<host>[A-Za-z0-9._-]+):repo:(?P<id>\d+)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------
def short_hash(*parts: str, length: int = 12) -> str:
    """Hex prefix of sha256 over the joined parts (via ``ids.short_hash_id``)."""
    return short_hash_id("h", *parts, length=length).split("-", 1)[1]


def slugify(text: str, *, fallback: str = "trial", max_length: int = 40) -> str:
    slug = _SLUG_RE.sub("-", str(text).lower()).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or fallback


def parse_github_repo_uid(repo_uid: str) -> tuple[str, int]:
    """``github:<host>:repo:<id>`` -> (host, id)."""
    match = _REPO_UID_RE.match(repo_uid or "")
    if not match:
        raise InputError(
            f"repo_uid {repo_uid!r} is not of the form github:<host>:repo:<numeric id>", code="INVALID_REPO_UID", details={"repo_uid": repo_uid}
        )
    return match.group("host"), int(match.group("id"))


def github_pr_key(repo_uid: str, number: int) -> str:
    _, repo_id = parse_github_repo_uid(repo_uid)
    return f"gh-{repo_id}-pr-{number}"


def local_pr_key(config_ref: str, repo_uid: str, title: str) -> str:
    return f"local-{slugify(title)}-{short_hash(config_ref, repo_uid, title, length=8)}"


def _publish_or_reuse(store: MemoryStore, record_type: str, record_id: str, payload: Any) -> tuple[Record, bool]:
    """Publish a new record or return the existing one when its payload is identical."""
    validate_record_id(record_id)
    existing = store.get(record_id)
    if existing is not None:
        if existing.record_type == record_type and to_json(existing.payload) == to_json(payload):
            return existing, False
        raise IdConflictError(
            f"{record_type} record {record_id!r} already exists with different content",
            details={"record_id": record_id, "record_type": existing.record_type},
        )
    record = new_record(record_type, record_id, payload)
    store.publish(record)
    return record, True


def _require_enum(value: Any, allowed: Sequence[str], what: str) -> str:
    if value not in allowed:
        raise InputError(f"{what} must be one of {list(allowed)}, got {value!r}", code="INVALID_ENUM", details={"field": what})
    return value


# --------------------------------------------------------------------------------------
# kernels and configs
# --------------------------------------------------------------------------------------
def register_kernel(store: MemoryStore, kernel_id: str, display_name: str, adapter_id: str, contract_notes: str) -> Record:
    validate_record_id(kernel_id, what="kernel_id")
    validate_record_id(adapter_id, what="adapter_id")
    payload = KernelPayload(kernel_id=kernel_id, display_name=str(display_name), adapter_id=adapter_id, contract_notes=str(contract_notes))
    existing = store.kernel_by_kernel_id(kernel_id)
    if existing is not None:
        if to_json(existing.payload) == to_json(payload):
            return existing
        raise IdConflictError(
            f"kernel {kernel_id!r} is already registered as {existing.record_id!r} with different content",
            details={"record_id": existing.record_id, "kernel_id": kernel_id},
        )
    record, _ = _publish_or_reuse(store, "kernel", f"kernel-{kernel_id}", payload)
    return record


def register_config(
    store: MemoryStore,
    kernel_id: str,
    raw_problem: dict[str, Any],
    *,
    registry: ProblemRegistry | None = None,
    tags: Sequence[str] | None = None,
) -> tuple[Record, bool]:
    registry = registry or default_registry()
    normalized = registry.normalize(kernel_id, raw_problem)  # IncompleteProblemContract propagates (exit 5)
    kernel = store.kernel_by_kernel_id(kernel_id)
    if kernel is None:
        raise MissingReferenceError(
            f"kernel {kernel_id!r} is not registered; register the kernel before its configs",
            code="MISSING_KERNEL",
            details={"kernel_id": kernel_id},
        )
    existing = store.configs_by_hash(normalized.config_hash)
    if existing:
        return sorted(existing, key=lambda r: r.record_id)[0], False
    clean_tags = sorted({str(t) for t in (tags or []) if str(t)})
    payload = ConfigPayload(
        kernel_id=kernel_id,
        config_id=normalized.config_id_hint,
        problem_schema_id=normalized.problem_schema_id,
        problem_schema_digest=normalized.problem_schema_digest,
        problem=dict(normalized.problem),
        config_hash=normalized.config_hash,
        tags=clean_tags,
    )
    record_id = f"cfg-{normalized.config_id_hint}-{normalized.config_hash.split(':', 1)[1][:12]}"
    record, created = _publish_or_reuse(store, "config", record_id, payload)
    return record, created


# --------------------------------------------------------------------------------------
# PR contexts
# --------------------------------------------------------------------------------------
def register_pr_context(
    store: MemoryStore,
    config_ref: str,
    *,
    repo_uid: str,
    provider: str,
    number: int | None,
    title: str,
    hypothesis: str | None = None,
    origin_ref: str | None = None,
    pr_key: str | None = None,
) -> Record:
    store.require(config_ref, "config")
    _require_enum(provider, PROVIDERS, "provider")
    validate_record_id(repo_uid, what="repo_uid")
    if provider == "github":
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise InputError(
                "a GitHub PR context requires an integer pull request number >= 1; use provider='local' for trials without a PR",
                code="PR_NUMBER_REQUIRED",
                details={"number": number},
            )
        derived = github_pr_key(repo_uid, number)
        pr_key = pr_key or derived
    else:
        if number is not None:
            raise InputError(
                "a local trial has no pull request number (number must be null); never invent a PR number",
                code="LOCAL_PR_HAS_NO_NUMBER",
                details={"number": number},
            )
        pr_key = pr_key or local_pr_key(config_ref, repo_uid, title)
    validate_record_id(pr_key, what="pr_key")
    if origin_ref is not None:
        store.require(origin_ref, "commit", "baseline")
    payload = PrPayload(
        config_ref=config_ref,
        pr_key=pr_key,
        repo_uid=repo_uid,
        provider=provider,
        number=number,
        title=str(title),
        hypothesis=None if hypothesis is None else str(hypothesis),
        origin_ref=origin_ref,
    )
    record, _ = _publish_or_reuse(store, "pr", f"pr-{pr_key}", payload)
    return record


def register_local_trial(
    store: MemoryStore,
    config_ref: str,
    title: str,
    *,
    repo_uid: str,
    hypothesis: str | None = None,
    origin_ref: str | None = None,
) -> Record:
    if not isinstance(title, str) or not title.strip():
        raise InputError("a local trial needs a non-empty title", code="TITLE_REQUIRED")
    return register_pr_context(
        store,
        config_ref,
        repo_uid=repo_uid,
        provider="local",
        number=None,
        title=title,
        hypothesis=hypothesis,
        origin_ref=origin_ref,
        pr_key=local_pr_key(config_ref, repo_uid, title),
    )


LOCAL_TRIAL_NOTE = "local trial, not yet a GitHub PR"


def describe_pr(pr: Record) -> dict[str, Any]:
    if pr.record_type != "pr":
        raise InputError(f"describe_pr expects a pr record, got {pr.record_type!r}", code="NOT_A_PR")
    p = pr.payload
    if p.provider == "local":
        label = f"{p.title} [{LOCAL_TRIAL_NOTE}]"
        note: str | None = LOCAL_TRIAL_NOTE
    else:
        label = f"PR #{p.number} ({p.repo_uid}): {p.title}"
        note = None
    return {
        "record_id": pr.record_id,
        "pr_key": p.pr_key,
        "provider": p.provider,
        "number": p.number,
        "repo_uid": p.repo_uid,
        "title": p.title,
        "is_local_trial": p.provider == "local",
        "display_label": label,
        "note": note,
    }


# --------------------------------------------------------------------------------------
# commits
# --------------------------------------------------------------------------------------
def _normalize_changes(changes: Sequence[dict[str, Any]], evidence_refs: Sequence[str], store: MemoryStore) -> list[Change]:
    out: list[Change] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(changes):
        if not isinstance(raw, dict):
            raise InputError(f"changes[{index}] must be an object", code="INVALID_CHANGE")
        data = dict(raw)
        data.setdefault("attribution", "group_only")
        validate_nested("Change", data)
        if data["attribution"] == "isolated":
            if not evidence_refs:
                raise InputError(
                    f"changes[{index}] ({data['change_id']}) claims isolated attribution without evidence_refs; "
                    "a combined result never proves an individual change effect (default is group_only)",
                    code="ISOLATED_REQUIRES_EVIDENCE",
                    details={"change_id": data["change_id"]},
                )
        if data["change_id"] in seen_ids:
            raise InputError(f"duplicate change_id {data['change_id']!r}", code="INVALID_CHANGE")
        seen_ids.add(data["change_id"])
        out.append(Change(**data))
    for ref in evidence_refs:
        store.require(ref)
    return out


def diff_artifact_id(pr_key: str, oid: GitOid) -> str:
    return f"diff-{pr_key}-{oid.short(12)}"


def store_diff_artifact(store: MemoryStore, pr_key: str, oid: GitOid, patch: bytes) -> ArtifactRef | None:
    """Store a diff as a content-addressed artifact and return its descriptor (None for an empty diff)."""
    if not patch:
        return None
    sha = store.put_artifact_bytes(patch)
    return ArtifactRef(
        artifact_id=diff_artifact_id(pr_key, oid),
        kind="diff",
        uri=f"artifact://sha256/{sha.split(':', 1)[1]}",
        sha256=sha,
        size_bytes=len(patch),
        media_type="text/x-diff",
        retention="permanent",
        availability="present",
    )


def record_commit(
    store: MemoryStore,
    pr_ref: str,
    sha: str,
    *,
    repo: LocalGitRepo,
    changes: Sequence[dict[str, Any]] | None = None,
    summary: str | None = None,
    summary_author: str = "human",
    store_diff: bool = True,
    diff_base: str | None = None,
    evidence_refs: Sequence[str] = (),
    repo_uid: str | None = None,
) -> Record:
    pr = store.require(pr_ref, "pr")
    pr_key = pr.payload.pr_key
    if repo_uid is not None and repo_uid != pr.payload.repo_uid:
        raise InputError(
            f"repo_uid {repo_uid!r} does not match the PR context's repo_uid {pr.payload.repo_uid!r}",
            code="REPO_UID_MISMATCH",
            details={"pr_ref": pr_ref, "repo_uid": repo_uid, "pr_repo_uid": pr.payload.repo_uid},
        )
    oid = repo.rev_parse(sha)
    info = repo.commit_info(oid)
    if diff_base is not None:
        base: GitOid | None = repo.rev_parse(diff_base)
    else:
        base = info.parents[0] if info.parents else None

    numstat = repo.diff_numstat(base, oid) if base is not None else None
    artifact: ArtifactRef | None = None
    if store_diff and base is not None:
        artifact = store_diff_artifact(store, pr_key, oid, repo.diff_patch(base, oid))

    if numstat is not None and len(numstat) == 0:
        if changes:
            raise InputError(
                f"commit {oid.hex} has no code change against {base.hex if base else None}; structured changes cannot be recorded for it",
                code="NO_CODE_CHANGE_WITH_CHANGES",
            )
        change_status = "no_code_change"
        change_list: list[Change] = []
    elif changes:
        change_list = _normalize_changes(changes, tuple(evidence_refs), store)
        change_status = "recorded"
    else:
        change_status = "not_extracted"
        change_list = []

    if summary is None:
        summary_text = info.subject
        author = "collector"
    else:
        summary_text = str(summary)
        author = _require_enum(summary_author, SUMMARY_AUTHORS, "summary_author")

    payload = CommitPayload(
        pr_ref=pr.record_id,
        repo_uid=pr.payload.repo_uid,
        commit_oid=oid,
        git_parent_oids=list(info.parents),
        diff_base_oid=base,
        source_available=True,
        change_status=change_status,
        changes=change_list,
        summary=summary_text,
        summary_author=author,
        diff_artifact_ref=artifact.artifact_id if artifact is not None else None,
    )
    record_id = f"commit-{pr_key}-{oid.short(12)}"
    existing = store.get(record_id)
    if existing is not None:
        if to_json(existing.payload) == to_json(payload):
            return existing
        raise IdConflictError(
            f"commit binding {record_id!r} already exists with different content (published records are immutable)",
            details={"record_id": record_id},
        )
    if artifact is not None:
        store.register_artifact_ref(artifact)
    record = new_record("commit", record_id, payload)
    store.publish(record)
    return record


# --------------------------------------------------------------------------------------
# baselines, relations, annotations
# --------------------------------------------------------------------------------------
def add_baseline(
    store: MemoryStore,
    config_ref: str,
    baseline_id: str,
    *,
    description: str,
    repo_uid: str,
    commit_oid: GitOid,
    entrypoint: str,
    role: str = "both",
) -> Record:
    store.require(config_ref, "config")
    validate_record_id(baseline_id, what="baseline_id")
    validate_record_id(repo_uid, what="repo_uid")
    _require_enum(role, BASELINE_ROLES, "role")
    if not isinstance(commit_oid, GitOid):
        raise InputError("commit_oid must be a GitOid (algorithm + full hex)", code="INVALID_OID")
    payload = BaselinePayload(
        config_ref=config_ref,
        baseline_id=baseline_id,
        description=str(description),
        repo_uid=repo_uid,
        commit_oid=commit_oid,
        entrypoint=str(entrypoint),
        role=role,
    )
    record, _ = _publish_or_reuse(store, "baseline", f"baseline-{baseline_id}", payload)
    return record


def add_relation(
    store: MemoryStore,
    config_ref: str,
    kind: str,
    from_ref: str,
    to_ref: str,
    *,
    rationale: str,
    evidence_refs: Sequence[str] = (),
) -> Record:
    store.require(config_ref, "config")
    _require_enum(kind, RELATION_KINDS, "kind")
    if from_ref == to_ref:
        raise InputError(f"a relation cannot connect {from_ref!r} to itself", code="SELF_RELATION")
    store.require(from_ref)
    store.require(to_ref)
    evidence = [str(e) for e in evidence_refs]
    for ref in evidence:
        store.require(ref)
    payload = RelationPayload(
        config_ref=config_ref, kind=kind, from_ref=from_ref, to_ref=to_ref, evidence_refs=evidence, rationale=str(rationale)
    )
    record_id = f"relation-{kind}-{short_hash(from_ref, to_ref, length=12)}"
    record, _ = _publish_or_reuse(store, "relation", record_id, payload)
    return record


def annotate(
    store: MemoryStore,
    target_ref: str,
    category: str,
    text: str,
    *,
    author_kind: str,
    evidence_refs: Sequence[str] = (),
    confidence: str = "unverified",
    supersedes_ref: str | None = None,
) -> Record:
    store.require(target_ref)
    evidence = [str(e) for e in evidence_refs]
    for ref in evidence:
        store.require(ref)
    if supersedes_ref is not None:
        store.require(supersedes_ref, "annotation")
    payload = AnnotationPayload(
        target_ref=target_ref,
        category=category,
        text=str(text),
        author_kind=author_kind,
        evidence_refs=evidence,
        confidence=confidence,
        supersedes_ref=supersedes_ref,
    )
    record = new_record("annotation", f"annotation-{uuid.uuid4().hex[:16]}", payload)
    store.publish(record)
    return record


__all__ = [
    "register_kernel",
    "register_config",
    "register_pr_context",
    "register_local_trial",
    "describe_pr",
    "record_commit",
    "add_baseline",
    "add_relation",
    "annotate",
    "github_pr_key",
    "local_pr_key",
    "parse_github_repo_uid",
    "short_hash",
    "slugify",
    "store_diff_artifact",
    "diff_artifact_id",
    "LOCAL_TRIAL_NOTE",
]
