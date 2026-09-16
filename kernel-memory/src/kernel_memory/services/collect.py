"""PR collection: read-only GitHub synchronisation into PR / commit / snapshot records (spec section 8).

Public API
----------
``collect_pr(store, client, *, owner, repo, number, config_ref, local_repo=None, hypothesis=None,
             origin_ref=None, store_diffs=True) -> CollectReport``
    1. ``client.get_repository`` -> ``repo_uid = github:<host>:repo:<id>``.
    2. ``client.get_pull`` -> PR context ``pr-gh-<repo_id>-pr-<number>`` via
       ``register.register_pr_context`` (title stored as data; ``hypothesis`` only from the
       caller). An existing PR record is reused unchanged (records are immutable; a changed
       remote title is reported in ``notes``).
    3. ``client.list_pull_commits`` enumerates the commits. When coverage is ``partial`` and a
       ``local_repo`` holds both base and head objects, ``rev_list(head, exclude=[base])``
       reconciles the membership (``complete_for_snapshot``, reason "reconciled with local
       git graph"); a shallow local repository cannot reconcile and coverage stays ``partial``.
    4. Every commit gets a binding ``commit-<pr_key>-<sha[:12]>`` (existing bindings are
       reused): parents from the API/local graph, ``diff_base_oid`` = first parent,
       ``source_available`` = ``local_repo.has_object(sha)`` (False without a local repo),
       ``change_status=not_extracted``, ``changes=[]``, summary = first message line
       (``summary_author=collector``), and a ``diff`` artifact when the local repository has
       both objects and ``store_diffs`` is true.
    5. The current observation (head, base, commit set, github_state) is compared with
       ``services.common.latest_snapshot``; when nothing changed no snapshot is appended,
       otherwise ``snapshot-<pr_key>-<seq:04d>`` is appended with ``previous_snapshot_ref``.
       ``observed_head`` is always the PR head SHA; ``merge_commit_sha`` is reported only
       (T08: an integration merge is a different tested source, never the head).
    6. Force push: the previous head SHA missing from the current commit set sets
       ``force_push_detected``; ``disappeared_commit_refs`` lists previous bindings not in
       the current set. They stay in the store (T10).
    7. ``untested_commit_refs`` = bindings without any run (policy ``record_all``;
       ``schedule_by_policy`` is reported, never executed here).
``ingest_pr_snapshot(store, pr_ref, *, observed_head, observed_base, commit_refs, enumeration_status,
                     reason, github_state) -> Record``  appends the next snapshot in the chain.
``CollectReport`` dataclass with ``to_dict()``.

Nothing here executes code or writes to GitHub; PR/commit text is data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..adapters.git_local import LocalGitRepo, oid_from_hex
from ..adapters.github import ApiCommit, CommitEnumeration, GitHubClient, PullInfo
from ..domain.errors import InputError, MissingReferenceError
from ..domain.models import ArtifactRef, CommitPayload, GitOid, PrSnapshotPayload, Record
from ..storage.store import MemoryStore
from .common import latest_snapshot, new_record, runs_for_subject, snapshots_for_pr
from .register import github_pr_key, register_pr_context, store_diff_artifact

COLLECTION_POLICY = "record_all, schedule_by_policy"
ENUMERATION_STATUSES = ("complete_for_snapshot", "partial", "unavailable")
GITHUB_STATES = ("open", "closed", "merged", "unknown")


@dataclass
class CollectReport:
    pr_ref: str
    repo_uid: str
    pr_key: str
    snapshot_ref: str | None
    snapshot_appended: bool
    previous_snapshot_ref: str | None
    coverage: str
    coverage_reason: str | None
    commits_total: int
    commits_new: list[str]
    commits_existing: list[str]
    untested_commit_refs: list[str]
    head_sha: str
    base_sha: str
    merge_commit_sha: str | None
    force_push_detected: bool
    disappeared_commit_refs: list[str]
    github_state: str
    notes: list[str] = field(default_factory=list)
    policy: str = COLLECTION_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_ref": self.pr_ref,
            "repo_uid": self.repo_uid,
            "pr_key": self.pr_key,
            "snapshot_ref": self.snapshot_ref,
            "snapshot_appended": self.snapshot_appended,
            "previous_snapshot_ref": self.previous_snapshot_ref,
            "coverage": self.coverage,
            "coverage_reason": self.coverage_reason,
            "commits_total": self.commits_total,
            "commits_new": list(self.commits_new),
            "commits_existing": list(self.commits_existing),
            "untested_commit_refs": list(self.untested_commit_refs),
            "head_sha": self.head_sha,
            "base_sha": self.base_sha,
            "merge_commit_sha": self.merge_commit_sha,
            "observed_head": self.head_sha,
            "force_push_detected": self.force_push_detected,
            "disappeared_commit_refs": list(self.disappeared_commit_refs),
            "github_state": self.github_state,
            "policy": self.policy,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _CommitFacts:
    sha: str
    parents: list[str]
    message: str


def _first_line(message: str) -> str:
    return message.split("\n", 1)[0].strip() if message else ""


def _facts_from_api(commits: Sequence[ApiCommit]) -> list[_CommitFacts]:
    return [_CommitFacts(sha=c.sha, parents=list(c.parents), message=c.message) for c in commits]


def _reconcile_with_local(
    enumeration: CommitEnumeration, pull: PullInfo, local_repo: LocalGitRepo, notes: list[str]
) -> tuple[list[_CommitFacts], str, str | None]:
    """Use the local graph to complete a partial enumeration when it can be trusted."""
    api_facts = _facts_from_api(enumeration.commits)
    if local_repo.is_shallow():
        reason = f"{enumeration.reason}; the local repository is shallow, so its graph cannot complete the enumeration"
        notes.append("local repository is shallow; reconciliation skipped")
        return api_facts, "partial", reason
    if not (local_repo.has_object(pull.base_sha) and local_repo.has_object(pull.head_sha)):
        reason = f"{enumeration.reason}; the local repository lacks the base or head object"
        notes.append("local repository lacks base or head object; reconciliation skipped")
        return api_facts, "partial", reason
    oids = local_repo.rev_list(pull.head_sha, exclude=[pull.base_sha])
    by_sha = {f.sha: f for f in api_facts}
    facts: list[_CommitFacts] = []
    for oid in oids:
        known = by_sha.get(oid.hex)
        if known is not None:
            facts.append(known)
            continue
        info = local_repo.commit_info(oid)
        facts.append(_CommitFacts(sha=oid.hex, parents=[p.hex for p in info.parents], message=info.message))
    notes.append(f"reconciled {len(facts)} commits with the local git graph (rev-list {pull.head_sha[:12]} --not {pull.base_sha[:12]})")
    return facts, "complete_for_snapshot", "reconciled with local git graph (git rev-list head --not base against pinned OIDs)"


def _build_binding(
    store: MemoryStore,
    pr: Record,
    facts: _CommitFacts,
    *,
    local_repo: LocalGitRepo | None,
    store_diffs: bool,
) -> tuple[Record, ArtifactRef | None]:
    oid = oid_from_hex(facts.sha)
    parents = [oid_from_hex(p) for p in facts.parents]
    diff_base = parents[0] if parents else None
    source_available = bool(local_repo is not None and local_repo.has_object(oid))
    artifact: ArtifactRef | None = None
    if store_diffs and local_repo is not None and diff_base is not None and source_available and local_repo.has_object(diff_base):
        artifact = store_diff_artifact(store, pr.payload.pr_key, oid, local_repo.diff_patch(diff_base, oid))
    payload = CommitPayload(
        pr_ref=pr.record_id,
        repo_uid=pr.payload.repo_uid,
        commit_oid=oid,
        git_parent_oids=parents,
        diff_base_oid=diff_base,
        source_available=source_available,
        change_status="not_extracted",
        changes=[],
        summary=_first_line(facts.message),
        summary_author="collector",
        diff_artifact_ref=artifact.artifact_id if artifact is not None else None,
    )
    record = new_record("commit", f"commit-{pr.payload.pr_key}-{oid.short(12)}", payload)
    return record, artifact


def ingest_pr_snapshot(
    store: MemoryStore,
    pr_ref: str,
    *,
    observed_head: GitOid,
    observed_base: GitOid,
    commit_refs: Sequence[str],
    enumeration_status: str,
    reason: str | None,
    github_state: str,
) -> Record:
    pr = store.require(pr_ref, "pr")
    if enumeration_status not in ENUMERATION_STATUSES:
        raise InputError(f"enumeration_status must be one of {list(ENUMERATION_STATUSES)}", code="INVALID_ENUM")
    if github_state not in GITHUB_STATES:
        raise InputError(f"github_state must be one of {list(GITHUB_STATES)}", code="INVALID_ENUM")
    if not isinstance(observed_head, GitOid) or not isinstance(observed_base, GitOid):
        raise InputError("observed_head and observed_base must be GitOid values", code="INVALID_OID")
    refs = list(commit_refs)
    if len(set(refs)) != len(refs):
        raise InputError("commit_refs contains duplicates", code="DUPLICATE_COMMIT_REF")
    for ref in refs:
        commit = store.require(ref, "commit")
        if commit.payload.pr_ref != pr.record_id:
            raise MissingReferenceError(
                f"commit {ref!r} belongs to {commit.payload.pr_ref!r}, not to {pr.record_id!r}",
                code="COMMIT_NOT_IN_PR",
                details={"commit_ref": ref, "pr_ref": pr.record_id},
            )
    existing = snapshots_for_pr(store, pr.record_id)
    previous = existing[-1].record_id if existing else None
    seq = len(existing) + 1
    while store.exists(f"snapshot-{pr.payload.pr_key}-{seq:04d}"):
        seq += 1
    payload = PrSnapshotPayload(
        pr_ref=pr.record_id,
        previous_snapshot_ref=previous,
        observed_head=observed_head,
        observed_base=observed_base,
        commit_refs=refs,
        enumeration_status=enumeration_status,
        reason=reason,
        github_state=github_state,
    )
    record = new_record("pr_snapshot", f"snapshot-{pr.payload.pr_key}-{seq:04d}", payload)
    store.publish(record)
    return record


def collect_pr(
    store: MemoryStore,
    client: GitHubClient,
    *,
    owner: str,
    repo: str,
    number: int,
    config_ref: str,
    local_repo: LocalGitRepo | None = None,
    hypothesis: str | None = None,
    origin_ref: str | None = None,
    store_diffs: bool = True,
) -> CollectReport:
    notes: list[str] = []
    store.require(config_ref, "config")
    repository = client.get_repository(owner, repo)
    repo_uid = repository.repo_uid
    pull = client.get_pull(owner, repo, number)
    github_state = pull.github_state

    # PR context (immutable; reuse when present).
    pr_key = github_pr_key(repo_uid, pull.number)
    pr = store.get(f"pr-{pr_key}")
    if pr is None:
        pr = register_pr_context(
            store,
            config_ref,
            repo_uid=repo_uid,
            provider="github",
            number=pull.number,
            title=pull.title,
            hypothesis=hypothesis,
            origin_ref=origin_ref,
        )
    else:
        if pr.payload.config_ref != config_ref:
            raise InputError(
                f"PR context {pr.record_id!r} belongs to config {pr.payload.config_ref!r}, not {config_ref!r}",
                code="PR_CONFIG_MISMATCH",
                details={"pr_ref": pr.record_id, "config_ref": config_ref},
            )
        if pr.payload.title != pull.title:
            notes.append("remote PR title differs from the stored title; the stored record is immutable and was kept")
        if hypothesis is not None and pr.payload.hypothesis != hypothesis:
            notes.append("hypothesis argument differs from the stored PR hypothesis; use an annotation to revise it")

    # Enumeration and optional reconciliation with the local graph.
    enumeration = client.list_pull_commits(owner, repo, pull.number, pull=pull)
    coverage, coverage_reason = enumeration.coverage, enumeration.reason
    facts = _facts_from_api(enumeration.commits)
    notes.extend(enumeration.notes)
    if coverage == "partial" and local_repo is not None:
        facts, coverage, coverage_reason = _reconcile_with_local(enumeration, pull, local_repo, notes)
    elif coverage == "partial":
        notes.append("coverage is partial and no local repository was supplied for reconciliation")

    # Deduplicate while keeping order.
    unique: list[_CommitFacts] = []
    seen: set[str] = set()
    for f in facts:
        if f.sha not in seen:
            seen.add(f.sha)
            unique.append(f)

    # Bindings.
    commits_new: list[str] = []
    commits_existing: list[str] = []
    new_records: list[Record] = []
    artifacts: list[ArtifactRef] = []
    commit_refs: list[str] = []
    for f in unique:
        record_id = f"commit-{pr_key}-{f.sha[:12]}"
        existing = store.get(record_id)
        if existing is not None:
            if existing.record_type != "commit" or existing.payload.pr_ref != pr.record_id:
                raise InputError(f"record {record_id!r} is not a commit binding of {pr.record_id!r}", code="BINDING_CONFLICT")
            commits_existing.append(record_id)
        else:
            record, artifact = _build_binding(store, pr, f, local_repo=local_repo, store_diffs=store_diffs)
            new_records.append(record)
            if artifact is not None:
                artifacts.append(artifact)
            commits_new.append(record_id)
        commit_refs.append(record_id)
    for artifact in artifacts:
        store.register_artifact_ref(artifact)
    if new_records:
        store.publish_bundle(new_records, label=f"collect-pr {pr_key}")

    # Snapshot comparison.
    head_oid = oid_from_hex(pull.head_sha)
    base_oid = oid_from_hex(pull.base_sha)
    previous = latest_snapshot(store, pr.record_id)
    force_push = False
    disappeared: list[str] = []
    if previous is not None:
        current_shas = {f.sha for f in unique}
        if previous.payload.observed_head.hex not in current_shas:
            force_push = True
            notes.append("previous head is no longer part of the PR commit set: force push or rebase detected; old records are retained")
        disappeared = [ref for ref in previous.payload.commit_refs if ref not in set(commit_refs)]
    unchanged = (
        previous is not None
        and previous.payload.observed_head.hex == head_oid.hex
        and previous.payload.observed_base.hex == base_oid.hex
        and set(previous.payload.commit_refs) == set(commit_refs)
        and previous.payload.github_state == github_state
        and previous.payload.enumeration_status == coverage
    )
    if unchanged and previous is not None:
        snapshot_ref = previous.record_id
        snapshot_appended = False
        previous_ref = previous.payload.previous_snapshot_ref
        notes.append("membership, head, base, and state unchanged since the latest snapshot; no snapshot appended")
    else:
        snapshot = ingest_pr_snapshot(
            store,
            pr.record_id,
            observed_head=head_oid,
            observed_base=base_oid,
            commit_refs=commit_refs,
            enumeration_status=coverage,
            reason=coverage_reason,
            github_state=github_state,
        )
        snapshot_ref = snapshot.record_id
        snapshot_appended = True
        previous_ref = snapshot.payload.previous_snapshot_ref

    if pull.merge_commit_sha and pull.merge_commit_sha != pull.head_sha:
        notes.append(
            "merge_commit_sha differs from head_sha; observed_head records the PR head. An integration merge is a "
            "separate tested source and comparison group (T08)"
        )

    untested = [ref for ref in commit_refs if not runs_for_subject(store, ref)]
    notes.append(f"policy {COLLECTION_POLICY}: {len(untested)} of {len(commit_refs)} bindings have no run; scheduling is reported, not executed")

    return CollectReport(
        pr_ref=pr.record_id,
        repo_uid=repo_uid,
        pr_key=pr_key,
        snapshot_ref=snapshot_ref,
        snapshot_appended=snapshot_appended,
        previous_snapshot_ref=previous_ref,
        coverage=coverage,
        coverage_reason=coverage_reason,
        commits_total=len(commit_refs),
        commits_new=commits_new,
        commits_existing=commits_existing,
        untested_commit_refs=untested,
        head_sha=pull.head_sha,
        base_sha=pull.base_sha,
        merge_commit_sha=pull.merge_commit_sha,
        force_push_detected=force_push,
        disappeared_commit_refs=disappeared,
        github_state=github_state,
        notes=notes,
    )


__all__ = ["CollectReport", "collect_pr", "ingest_pr_snapshot", "COLLECTION_POLICY"]
