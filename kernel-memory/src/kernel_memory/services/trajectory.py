"""Deterministic trajectory view generation (specification sections 4, 7, 14).

Public API
----------
``build_trajectory(store, config_ref) -> dict``
    Pure function of the authoritative records that belong to one config. The result is the
    ``trajectory-v1`` view: baselines, PRs (snapshots, commit bindings, runs), a typed node/edge
    graph, decisions, best-known candidates, annotations, relations, diagnostics, and counts.
    Guarantees:

    * Deterministic: every list is sorted by stable identifiers (or by stored snapshot order for
      display ordinals). The body carries no generation timestamp, run id, or path, so identical
      facts always produce an identical view and hash, regardless of import order.
    * Never crashes on a dangling reference: any reference that the store cannot resolve becomes
      a ``MISSING_REFERENCE`` error diagnostic and the affected branch is kept, not dropped.
    * The three relationships are kept apart as separate edge kinds: ``git_parent`` (source
      history, resolved to commit *bindings* in this config by object id), ``optimization_origin``
      (from ``pr.origin_ref`` and explicit relations; edges point from the origin to the derived
      record) and ``comparison_baseline`` (from a decision's candidate run to its baseline run).
      ``membership`` (pr -> commit), ``run_of`` (run -> subject) and ``snapshot_of`` (snapshot -> pr)
      are structural. Time ordering is never used as ancestry.
    * Untested commits carry ``status="not_run"`` derived from the absence of runs; no run is invented.
    * Error diagnostics (``MISSING_REFERENCE``, ``ORIGIN_CYCLE``, ``GIT_PARENT_CYCLE``,
      ``CROSS_PR_MEMBERSHIP``, ``CROSS_CONFIG_RUN``) make ``publishable`` false. ``SHARED_SOURCE``
      is informational; ``UNTESTED_COMMITS``, ``PARTIAL_COVERAGE`` and ``AMBIGUOUS_BEST_KNOWN``
      are warnings.

``trajectory_hash(view) -> str``
    ``sha256:<hex>`` over the RFC 8785 canonical form of the view (``hashing.jcs_digest``).

``publish_trajectory(store, config_ref, *, force=False) -> dict``
    Writes ``trajectory.json`` (``{"generated_at", "view_hash", "view"}``) and
    ``memory_records.jsonl`` (one compact line per record of the config, sorted by type order
    then id, no timestamps) through ``MemoryStore.write_view``. Raises ``InvariantViolation``
    with code ``TRAJECTORY_NOT_PUBLISHABLE`` when the view has error diagnostics, unless ``force``.

``verify_trajectory(store, config_ref) -> dict``
    Compares the stored ``view_hash`` with a fresh rebuild (``{"stored_hash", "rebuilt_hash",
    "matches", "stored"}``).

``rebuild_trajectory(store, config_ref, *, force=False) -> dict``
    Deletes the generated views and publishes again (acceptance scenario T24).

Shared helpers used by ``services.query`` and ``services.context``: ``collect_config_records``
(``ConfigRecords``), ``run_summary``, ``annotation_kind``, ``pr_display_label``,
``memory_records``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..domain.errors import InvariantViolation, MissingReferenceError
from ..domain.hashing import jcs_digest
from ..domain.ids import utc_now_iso
from ..domain.jsonio import dumps_compact, dumps_readable, loads_strict
from ..domain.models import TYPE_ORDER, Record, to_json
from ..storage.store import MemoryStore

VIEW_VERSION = "trajectory-v1"
TRAJECTORY_VIEW_FILE = "trajectory.json"
MEMORY_RECORDS_FILE = "memory_records.jsonl"
LOCAL_TRIAL_LABEL = "local trial, not yet a GitHub PR"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
_SEVERITY_ORDER = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}

# Edge kinds and sources (kept as constants so the CLI/tests can refer to them).
EDGE_MEMBERSHIP = "membership"
EDGE_GIT_PARENT = "git_parent"
EDGE_OPTIMIZATION_ORIGIN = "optimization_origin"
EDGE_INSPIRED_BY = "inspired_by"
EDGE_REBASED_FROM = "rebased_from"
EDGE_COMPARISON_BASELINE = "comparison_baseline"
EDGE_RUN_OF = "run_of"
EDGE_SNAPSHOT_OF = "snapshot_of"
SOURCE_STRUCTURE = "structure"
SOURCE_GIT = "git"
SOURCE_PR_ORIGIN = "pr.origin_ref"

_RELATION_EDGE_KINDS = {"optimization_origin": EDGE_OPTIMIZATION_ORIGIN, "inspired_by": EDGE_INSPIRED_BY, "rebased_from": EDGE_REBASED_FROM}
_UNVERIFIED_PROVENANCE = ("fixture", "imported_unverified")


# --------------------------------------------------------------------------------------
# Config scope collection
# --------------------------------------------------------------------------------------
@dataclass
class ConfigRecords:
    """All authoritative records that belong to one config, grouped and indexed by stable ids."""

    config: Record
    kernel: Record | None
    baselines: list[Record] = field(default_factory=list)
    prs: list[Record] = field(default_factory=list)
    snapshots: list[Record] = field(default_factory=list)
    commits: list[Record] = field(default_factory=list)
    runs: list[Record] = field(default_factory=list)
    relations: list[Record] = field(default_factory=list)
    decisions: list[Record] = field(default_factory=list)
    annotations: list[Record] = field(default_factory=list)
    foreign_runs: list[Record] = field(default_factory=list)  # runs of this config's subjects claiming another config

    @property
    def config_id(self) -> str:
        return self.config.record_id

    @property
    def subject_ids(self) -> set[str]:
        return {c.record_id for c in self.commits} | {b.record_id for b in self.baselines}

    def runs_by_subject(self) -> dict[str, list[Record]]:
        out: dict[str, list[Record]] = {}
        for run in self.runs:
            out.setdefault(run.payload.subject_ref, []).append(run)
        for runs in out.values():
            runs.sort(key=lambda r: r.record_id)
        return out

    def commits_by_pr(self) -> dict[str, list[Record]]:
        out: dict[str, list[Record]] = {}
        for commit in self.commits:
            out.setdefault(commit.payload.pr_ref, []).append(commit)
        for commits in out.values():
            commits.sort(key=lambda r: r.record_id)
        return out

    def snapshots_by_pr(self) -> dict[str, list[Record]]:
        """Snapshots per PR in chain order (previous_snapshot_ref), orphans appended by id."""
        grouped: dict[str, list[Record]] = {}
        for snap in self.snapshots:
            grouped.setdefault(snap.payload.pr_ref, []).append(snap)
        return {pr_id: _chain_order(snaps) for pr_id, snaps in grouped.items()}

    def all_records(self) -> list[Record]:
        records = [self.config, *self.baselines, *self.prs, *self.snapshots, *self.commits, *self.runs, *self.relations, *self.decisions, *self.annotations]
        return sorted(records, key=_type_id_key)


def _type_id_key(record: Record) -> tuple[int, str]:
    return (TYPE_ORDER[record.record_type], record.record_id)


def _chain_order(snaps: list[Record]) -> list[Record]:
    by_prev: dict[str | None, list[Record]] = {}
    for s in snaps:
        by_prev.setdefault(s.payload.previous_snapshot_ref, []).append(s)
    ordered: list[Record] = []
    seen: set[str] = set()
    frontier = sorted(by_prev.get(None, []), key=lambda r: r.record_id)
    while frontier:
        current = frontier.pop(0)
        if current.record_id in seen:
            continue
        seen.add(current.record_id)
        ordered.append(current)
        frontier = sorted(by_prev.get(current.record_id, []), key=lambda r: r.record_id) + frontier
    for s in sorted(snaps, key=lambda r: r.record_id):
        if s.record_id not in seen:
            ordered.append(s)
    return ordered


def collect_config_records(store: MemoryStore, config_ref: str) -> ConfigRecords:
    """Group every record owned by ``config_ref`` in one pass over the store (deterministic order)."""
    config = store.require(config_ref, "config")
    kernel = store.kernel_by_kernel_id(config.payload.kernel_id)
    scope = ConfigRecords(config=config, kernel=kernel)
    by_type: dict[str, list[Record]] = {}
    for record in store.records():
        by_type.setdefault(record.record_type, []).append(record)
    cid = config.record_id
    scope.baselines = [r for r in by_type.get("baseline", []) if r.payload.config_ref == cid]
    scope.prs = [r for r in by_type.get("pr", []) if r.payload.config_ref == cid]
    pr_ids = {p.record_id for p in scope.prs}
    scope.snapshots = [r for r in by_type.get("pr_snapshot", []) if r.payload.pr_ref in pr_ids]
    scope.commits = [r for r in by_type.get("commit", []) if r.payload.pr_ref in pr_ids]
    subjects = scope.subject_ids
    for run in by_type.get("run", []):
        if run.payload.config_ref == cid:
            scope.runs.append(run)
        elif run.payload.subject_ref in subjects:
            scope.foreign_runs.append(run)
    scope.relations = [r for r in by_type.get("relation", []) if r.payload.config_ref == cid]
    scope.decisions = [r for r in by_type.get("decision", []) if r.payload.config_ref == cid]
    owned = subjects | pr_ids | {cid} | {s.record_id for s in scope.snapshots} | {r.record_id for r in scope.runs}
    owned |= {r.record_id for r in scope.relations} | {d.record_id for d in scope.decisions}
    annotations = [a for a in by_type.get("annotation", []) if a.payload.target_ref in owned]
    # Annotations on annotations (corrections) belong to the config of their ultimate target.
    changed = True
    ann_ids = {a.record_id for a in annotations}
    while changed:
        changed = False
        for a in by_type.get("annotation", []):
            if a.record_id not in ann_ids and a.payload.target_ref in ann_ids:
                annotations.append(a)
                ann_ids.add(a.record_id)
                changed = True
    scope.annotations = sorted(annotations, key=lambda r: r.record_id)
    for name in ("baselines", "prs", "snapshots", "commits", "runs", "relations", "decisions", "foreign_runs"):
        getattr(scope, name).sort(key=lambda r: r.record_id)
    return scope


# --------------------------------------------------------------------------------------
# Summaries shared with query/context
# --------------------------------------------------------------------------------------
def run_summary(run: Record) -> dict[str, Any]:
    """Read-only summary of a run. Values are copied from the record; nothing is derived or invented."""
    p = run.payload
    return {
        "run_ref": run.record_id,
        "request_id": p.request_id,
        "attempt_no": p.attempt_no,
        "rerun_of": p.rerun_of,
        "stage": p.stage,
        "provenance": p.provenance,
        "execution_status": p.execution_status,
        "failure_reason": p.failure_reason,
        "correctness": {"status": p.correctness.status, "cases_total": p.correctness.cases_total, "cases_passed": p.correctness.cases_passed},
        "timing": {"status": p.timing.status, "sample_count": p.timing.sample_count, "median_us": p.timing.median_us, "p90_us": p.timing.p90_us},
        "comparison_key": p.comparison_key,
        "variant_digest": p.source.variant_digest,
        "checkout_mode": p.source.checkout_mode,
        "tested_commit": to_json(p.source.tested_commit),
        "dirty": p.source.dirty,
        "session_id": p.session_id,
        "pair_id": p.pair_id,
        "role_in_pair": p.role_in_pair,
        "metrics": [
            {"name": m.name, "status": m.status, "value": m.value, "unit": m.unit, "kind": m.kind, "scope": m.scope} for m in p.analysis_metrics
        ],
        "analysis_conclusion": p.analysis_conclusion,
        "artifact_refs": [a.artifact_id for a in p.artifacts],
    }


def annotation_kind(annotation: Record) -> str:
    """``fact`` only for program-authored, supported annotations; everything else is a hypothesis."""
    p = annotation.payload
    return "fact" if (p.author_kind == "program" and p.confidence == "supported") else "hypothesis"


def pr_display_label(pr: Record) -> str:
    p = pr.payload
    if p.provider == "local":
        return LOCAL_TRIAL_LABEL
    if p.number is None:
        return f"GitHub PR {p.pr_key}"
    return f"GitHub PR #{p.number}"


def change_summary(change: Any) -> dict[str, Any]:
    return {
        "change_id": change.change_id,
        "component": change.component,
        "key": change.key,
        "before": change.before,
        "after": change.after,
        "attribution": change.attribution,
    }


def _oid(oid: Any) -> dict[str, str] | None:
    return None if oid is None else {"algorithm": oid.algorithm, "hex": oid.hex}


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------
def _find_cycle(edges: Iterable[tuple[str, str]]) -> list[str] | None:
    """Return one directed cycle (as a node path) or None. Deterministic (sorted adjacency)."""
    adjacency: dict[str, list[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, [])
    for node in adjacency:
        adjacency[node] = sorted(set(adjacency[node]))
    white, grey, black = 0, 1, 2
    colour: dict[str, int] = {n: white for n in adjacency}
    for start in sorted(adjacency):
        if colour[start] != white:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = [start]
        colour[start] = grey
        while stack:
            node, idx = stack[-1]
            children = adjacency[node]
            if idx < len(children):
                stack[-1] = (node, idx + 1)
                child = children[idx]
                if colour[child] == grey:
                    cycle_start = path.index(child)
                    return path[cycle_start:] + [child]
                if colour[child] == white:
                    colour[child] = grey
                    stack.append((child, 0))
                    path.append(child)
            else:
                colour[node] = black
                stack.pop()
                path.pop()
    return None


class _Diagnostics:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._seen: set[str] = set()

    def add(self, severity: str, code: str, message: str, refs: Iterable[str]) -> None:
        entry = {"severity": severity, "code": code, "message": message, "refs": sorted(set(refs))}
        key = dumps_compact(entry)
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(entry)

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(self.items, key=lambda d: (_SEVERITY_ORDER[d["severity"]], d["code"], d["message"], d["refs"]))

    @property
    def has_errors(self) -> bool:
        return any(d["severity"] == SEVERITY_ERROR for d in self.items)


# --------------------------------------------------------------------------------------
# View construction
# --------------------------------------------------------------------------------------
def build_trajectory(store: MemoryStore, config_ref: str) -> dict[str, Any]:
    scope = collect_config_records(store, config_ref)
    config = scope.config
    diagnostics = _Diagnostics()

    def resolve(owner: str, field_name: str, target: str | None, *allowed: str) -> Record | None:
        """Resolve a reference; a missing or mistyped target becomes a MISSING_REFERENCE diagnostic."""
        if target is None:
            return None
        record = store.get(target)
        if record is None:
            diagnostics.add(SEVERITY_ERROR, "MISSING_REFERENCE", f"{owner} field {field_name} refers to missing record {target!r}", [owner, target])
            return None
        if allowed and record.record_type not in allowed:
            diagnostics.add(
                SEVERITY_ERROR,
                "MISSING_REFERENCE",
                f"{owner} field {field_name} refers to {target!r} of type {record.record_type!r}, expected {list(allowed)}",
                [owner, target],
            )
            return None
        return record

    if scope.kernel is None:
        diagnostics.add(SEVERITY_ERROR, "MISSING_REFERENCE", f"config {config.record_id} refers to unknown kernel_id {config.payload.kernel_id!r}", [config.record_id])

    runs_by_subject = scope.runs_by_subject()
    commits_by_pr = scope.commits_by_pr()
    snapshots_by_pr = scope.snapshots_by_pr()
    commit_ids = {c.record_id for c in scope.commits}
    pr_ids = {p.record_id for p in scope.prs}
    subject_ids = scope.subject_ids
    nodes: dict[str, str] = {}
    edges: set[tuple[str, str, str, str]] = set()  # (kind, from, to, source)

    def add_node(record_id: str, record_type: str) -> None:
        nodes[record_id] = record_type

    def add_edge(kind: str, src: str, dst: str, source: str) -> None:
        edges.add((kind, src, dst, source))

    # ---- runs: config consistency and per-subject attachment ---------------------------
    for run in scope.runs:
        add_node(run.record_id, "run")
        add_edge(EDGE_RUN_OF, run.record_id, run.payload.subject_ref, SOURCE_STRUCTURE)
        subject = resolve(run.record_id, "subject_ref", run.payload.subject_ref, "commit", "baseline")
        if subject is not None and subject.record_id not in subject_ids:
            diagnostics.add(
                SEVERITY_ERROR,
                "CROSS_CONFIG_RUN",
                f"run {run.record_id} has config_ref {config.record_id} but its subject {subject.record_id} belongs to another config",
                [run.record_id, subject.record_id],
            )
        if run.payload.rerun_of is not None:
            resolve(run.record_id, "rerun_of", run.payload.rerun_of, "run")
    for run in scope.foreign_runs:
        diagnostics.add(
            SEVERITY_ERROR,
            "CROSS_CONFIG_RUN",
            f"run {run.record_id} of subject {run.payload.subject_ref} has config_ref {run.payload.config_ref!r}, not {config.record_id!r}",
            [run.record_id, run.payload.subject_ref],
        )

    def run_summaries(subject_id: str) -> list[dict[str, Any]]:
        return [run_summary(r) for r in runs_by_subject.get(subject_id, [])]

    # ---- baselines ----------------------------------------------------------------------
    baselines_view: list[dict[str, Any]] = []
    for baseline in scope.baselines:
        add_node(baseline.record_id, "baseline")
        p = baseline.payload
        baselines_view.append(
            {
                "baseline_ref": baseline.record_id,
                "baseline_id": p.baseline_id,
                "role": p.role,
                "description": p.description,
                "repo_uid": p.repo_uid,
                "commit_oid": _oid(p.commit_oid),
                "entrypoint": p.entrypoint,
                "runs": run_summaries(baseline.record_id),
            }
        )

    # ---- source identity index for git_parent resolution -------------------------------
    bindings_by_source: dict[tuple[str, str, str], list[str]] = {}
    for commit in scope.commits:
        bindings_by_source.setdefault(commit.payload.source_key, []).append(commit.record_id)
    for key, bound in sorted(bindings_by_source.items()):
        bound.sort()
        if len(bound) > 1:
            diagnostics.add(
                SEVERITY_INFO,
                "SHARED_SOURCE",
                f"source commit {key[2]} of {key[0]} is bound in {len(bound)} PR memberships; results stay with each binding",
                bound,
            )

    # ---- PRs, snapshots, commits -------------------------------------------------------
    untested: list[str] = []
    prs_view: list[dict[str, Any]] = []
    git_edges: list[tuple[str, str]] = []
    origin_edges: list[tuple[str, str]] = []
    for pr in scope.prs:
        add_node(pr.record_id, "pr")
        p = pr.payload
        origin: dict[str, Any] | None = None
        if p.origin_ref is not None:
            origin_record = resolve(pr.record_id, "origin_ref", p.origin_ref, "commit", "baseline")
            origin = {"ref": p.origin_ref, "record_type": origin_record.record_type if origin_record else None}
            add_edge(EDGE_OPTIMIZATION_ORIGIN, p.origin_ref, pr.record_id, SOURCE_PR_ORIGIN)
            origin_edges.append((p.origin_ref, pr.record_id))
        snapshots_view: list[dict[str, Any]] = []
        snapshot_members: set[str] = set()
        ordered_snaps = snapshots_by_pr.get(pr.record_id, [])
        for snap in ordered_snaps:
            add_node(snap.record_id, "pr_snapshot")
            add_edge(EDGE_SNAPSHOT_OF, snap.record_id, pr.record_id, SOURCE_STRUCTURE)
            sp = snap.payload
            if sp.previous_snapshot_ref is not None:
                resolve(snap.record_id, "previous_snapshot_ref", sp.previous_snapshot_ref, "pr_snapshot")
            for i, cref in enumerate(sp.commit_refs):
                member = resolve(snap.record_id, f"commit_refs[{i}]", cref, "commit")
                if member is None:
                    continue
                if member.payload.pr_ref != pr.record_id:
                    diagnostics.add(
                        SEVERITY_ERROR,
                        "CROSS_PR_MEMBERSHIP",
                        f"snapshot {snap.record_id} of {pr.record_id} lists commit {cref} bound to {member.payload.pr_ref}",
                        [snap.record_id, cref, pr.record_id, member.payload.pr_ref],
                    )
                    continue
                snapshot_members.add(cref)
                add_edge(EDGE_MEMBERSHIP, pr.record_id, cref, SOURCE_STRUCTURE)
            snapshots_view.append(
                {
                    "snapshot_ref": snap.record_id,
                    "previous_snapshot_ref": sp.previous_snapshot_ref,
                    "observed_head": _oid(sp.observed_head),
                    "observed_base": _oid(sp.observed_base),
                    "commit_refs": list(sp.commit_refs),
                    "enumeration_status": sp.enumeration_status,
                    "reason": sp.reason,
                    "github_state": sp.github_state,
                }
            )
        latest = ordered_snaps[-1] if ordered_snaps else None
        latest_members = list(latest.payload.commit_refs) if latest else []
        coverage = latest.payload.enumeration_status if latest else "no_snapshot"
        if coverage != "complete_for_snapshot":
            diagnostics.add(
                SEVERITY_WARNING,
                "PARTIAL_COVERAGE",
                f"{pr.record_id} commit enumeration is {coverage}; untested or unknown commits may exist beyond the recorded memberships",
                [pr.record_id] + ([latest.record_id] if latest else []),
            )
        commits_view: list[dict[str, Any]] = []
        pr_commits = commits_by_pr.get(pr.record_id, [])
        ordinal_of = {cref: i + 1 for i, cref in enumerate(latest_members)}
        next_ordinal = len(latest_members) + 1
        for commit in pr_commits:
            add_node(commit.record_id, "commit")
            add_edge(EDGE_MEMBERSHIP, pr.record_id, commit.record_id, SOURCE_STRUCTURE)
            cp = commit.payload
            for parent in cp.git_parent_oids:
                parent_bindings = bindings_by_source.get((cp.repo_uid, parent.algorithm, parent.hex), [])
                for binding in parent_bindings:
                    add_edge(EDGE_GIT_PARENT, commit.record_id, binding, SOURCE_GIT)
                    git_edges.append((commit.record_id, binding))
            runs = run_summaries(commit.record_id)
            if not runs:
                untested.append(commit.record_id)
            if commit.record_id in ordinal_of:
                ordinal = ordinal_of[commit.record_id]
            else:
                ordinal = next_ordinal
                next_ordinal += 1
            commits_view.append(
                {
                    "commit_ref": commit.record_id,
                    "commit_oid": _oid(cp.commit_oid),
                    "repo_uid": cp.repo_uid,
                    "git_parent_oids": [_oid(o) for o in cp.git_parent_oids],
                    "diff_base_oid": _oid(cp.diff_base_oid),
                    "change_status": cp.change_status,
                    "changes": [change_summary(c) for c in cp.changes],
                    "summary": cp.summary,
                    "summary_author": cp.summary_author,
                    "source_available": cp.source_available,
                    "diff_artifact_ref": cp.diff_artifact_ref,
                    "in_latest_snapshot": commit.record_id in latest_members,
                    "display_ordinal": ordinal,
                    "status": "tested" if runs else "not_run",
                    "runs": runs,
                }
            )
        prs_view.append(
            {
                "pr_ref": pr.record_id,
                "pr_key": p.pr_key,
                "repo_uid": p.repo_uid,
                "provider": p.provider,
                "number": p.number,
                "display_label": pr_display_label(pr),
                "title": p.title,
                "hypothesis": p.hypothesis,
                "origin": origin,
                "snapshots": snapshots_view,
                "latest_snapshot_ref": latest.record_id if latest else None,
                "coverage": coverage,
                "commits": commits_view,
            }
        )

    # ---- relations ----------------------------------------------------------------------
    relations_view: list[dict[str, Any]] = []
    for relation in scope.relations:
        rp = relation.payload
        resolve(relation.record_id, "from_ref", rp.from_ref)
        resolve(relation.record_id, "to_ref", rp.to_ref)
        for i, ev in enumerate(rp.evidence_refs):
            resolve(relation.record_id, f"evidence_refs[{i}]", ev)
        kind = _RELATION_EDGE_KINDS.get(rp.kind, rp.kind)
        add_edge(kind, rp.from_ref, rp.to_ref, f"relation:{relation.record_id}")
        if rp.kind == "optimization_origin":
            origin_edges.append((rp.from_ref, rp.to_ref))
        relations_view.append(
            {
                "relation_ref": relation.record_id,
                "kind": rp.kind,
                "from_ref": rp.from_ref,
                "to_ref": rp.to_ref,
                "evidence_refs": list(rp.evidence_refs),
                "rationale": rp.rationale,
            }
        )

    # ---- decisions and best-known -------------------------------------------------------
    decisions_view: list[dict[str, Any]] = []
    superseded: set[str] = set()
    for decision in scope.decisions:
        add_node(decision.record_id, "decision")
        dp = decision.payload
        resolve(decision.record_id, "candidate_subject_ref", dp.candidate_subject_ref, "commit", "baseline")
        for i, ref in enumerate(dp.candidate_run_refs):
            resolve(decision.record_id, f"candidate_run_refs[{i}]", ref, "run")
        for i, ref in enumerate(dp.baseline_run_refs):
            resolve(decision.record_id, f"baseline_run_refs[{i}]", ref, "run")
        if dp.supersedes_decision_ref is not None:
            resolve(decision.record_id, "supersedes_decision_ref", dp.supersedes_decision_ref, "decision")
            superseded.add(dp.supersedes_decision_ref)
        for cand in dp.candidate_run_refs:
            for base in dp.baseline_run_refs:
                add_edge(EDGE_COMPARISON_BASELINE, cand, base, f"decision:{decision.record_id}")
        decisions_view.append(
            {
                "decision_ref": decision.record_id,
                "comparison_key": dp.comparison_key,
                "candidate_subject_ref": dp.candidate_subject_ref,
                "candidate_run_refs": list(dp.candidate_run_refs),
                "baseline_run_refs": list(dp.baseline_run_refs),
                "policy_id": dp.policy.policy_id,
                "policy_hash": dp.policy_hash,
                "outcome": dp.outcome,
                "reason_codes": list(dp.reason_codes),
                "is_production": dp.is_production,
                "evaluated_by": dp.evaluated_by,
                "supersedes_decision_ref": dp.supersedes_decision_ref,
            }
        )
    best_known: list[dict[str, Any]] = []
    for decision in scope.decisions:
        dp = decision.payload
        if dp.outcome == "accepted" and dp.is_production and decision.record_id not in superseded:
            best_known.append(
                {
                    "comparison_key": dp.comparison_key,
                    "policy_hash": dp.policy_hash,
                    "candidate_subject_ref": dp.candidate_subject_ref,
                    "decision_ref": decision.record_id,
                }
            )
    best_known.sort(key=lambda b: (b["comparison_key"], b["policy_hash"], b["candidate_subject_ref"], b["decision_ref"]))
    groups: dict[tuple[str, str], list[str]] = {}
    for entry in best_known:
        groups.setdefault((entry["comparison_key"], entry["policy_hash"]), []).append(entry["decision_ref"])
    for (ckey, phash), refs in sorted(groups.items()):
        if len(refs) > 1:
            diagnostics.add(
                SEVERITY_WARNING,
                "AMBIGUOUS_BEST_KNOWN",
                f"{len(refs)} accepted production decisions for comparison_key {ckey} under policy {phash} without supersession",
                refs,
            )

    # ---- annotations --------------------------------------------------------------------
    annotations_view: list[dict[str, Any]] = []
    for annotation in scope.annotations:
        ap = annotation.payload
        resolve(annotation.record_id, "target_ref", ap.target_ref)
        for i, ev in enumerate(ap.evidence_refs):
            resolve(annotation.record_id, f"evidence_refs[{i}]", ev)
        if ap.supersedes_ref is not None:
            resolve(annotation.record_id, "supersedes_ref", ap.supersedes_ref, "annotation")
        annotations_view.append(
            {
                "annotation_ref": annotation.record_id,
                "target_ref": ap.target_ref,
                "category": ap.category,
                "text": ap.text,
                "author_kind": ap.author_kind,
                "confidence": ap.confidence,
                "evidence_refs": list(ap.evidence_refs),
                "supersedes_ref": ap.supersedes_ref,
                "kind": annotation_kind(annotation),
            }
        )

    # ---- cycles -------------------------------------------------------------------------
    origin_cycle = _find_cycle(origin_edges)
    if origin_cycle is not None:
        diagnostics.add(
            SEVERITY_ERROR,
            "ORIGIN_CYCLE",
            "optimization-origin graph contains a cycle: " + " -> ".join(origin_cycle),
            origin_cycle,
        )
    git_cycle = _find_cycle(git_edges)
    if git_cycle is not None:
        diagnostics.add(
            SEVERITY_ERROR,
            "GIT_PARENT_CYCLE",
            "git-parent graph over commit bindings contains a cycle: " + " -> ".join(git_cycle),
            git_cycle,
        )
    if untested:
        diagnostics.add(
            SEVERITY_WARNING,
            "UNTESTED_COMMITS",
            f"{len(untested)} commit binding(s) have no recorded run (status not_run is derived from absence)",
            untested,
        )

    node_list = sorted(({"id": rid, "type": t} for rid, t in nodes.items()), key=lambda n: (TYPE_ORDER[n["type"]], n["id"]))
    edge_list = [
        {"kind": kind, "from": src, "to": dst, "source": source}
        for kind, src, dst, source in sorted(edges)
    ]
    diagnostics_list = diagnostics.sorted()
    view: dict[str, Any] = {
        "view_version": VIEW_VERSION,
        "config_ref": config.record_id,
        "config_hash": config.payload.config_hash,
        "kernel_id": config.payload.kernel_id,
        "kernel_ref": scope.kernel.record_id if scope.kernel else None,
        "baselines": baselines_view,
        "prs": prs_view,
        "nodes": node_list,
        "edges": edge_list,
        "decisions": decisions_view,
        "best_known": best_known,
        "annotations": annotations_view,
        "relations": relations_view,
        "diagnostics": diagnostics_list,
        "publishable": not diagnostics.has_errors,
        "counts": {
            "baselines": len(scope.baselines),
            "prs": len(scope.prs),
            "snapshots": len(scope.snapshots),
            "commits": len(scope.commits),
            "runs": len(scope.runs),
            "relations": len(scope.relations),
            "decisions": len(scope.decisions),
            "annotations": len(scope.annotations),
            "nodes": len(node_list),
            "edges": len(edge_list),
            "untested_commits": len(untested),
            "diagnostics": len(diagnostics_list),
            "errors": sum(1 for d in diagnostics_list if d["severity"] == SEVERITY_ERROR),
        },
    }
    return view


def trajectory_hash(view: dict[str, Any]) -> str:
    return jcs_digest(view)


# --------------------------------------------------------------------------------------
# Compact memory records (memory_records.jsonl)
# --------------------------------------------------------------------------------------
def _record_summary(record: Record, tested: dict[str, list[str]], pr_labels: dict[str, str]) -> tuple[str, str, list[str], str | None]:
    """Template summary text, kind, evidence refs, provenance for one record (data, not instructions)."""
    t = record.record_type
    p = record.payload
    if t == "config":
        return (
            f"Config {p.config_id} for kernel {p.kernel_id}; config_hash {p.config_hash}; tags {', '.join(p.tags) if p.tags else 'none'}",
            "fact",
            [],
            None,
        )
    if t == "baseline":
        return (
            f"Baseline {p.baseline_id} (role {p.role}) at {p.repo_uid}@{p.commit_oid.hex}: {p.description}",
            "fact",
            [],
            None,
        )
    if t == "pr":
        text = f"{pr_display_label(record)} [{p.pr_key}] in {p.config_ref}: {p.title}"
        if p.hypothesis:
            text += f"; hypothesis: {p.hypothesis}"
        text += f"; optimization origin {p.origin_ref if p.origin_ref else 'not recorded'}"
        return text, "fact", [p.origin_ref] if p.origin_ref else [], None
    if t == "pr_snapshot":
        return (
            f"Snapshot of {p.pr_ref}: head {p.observed_head.hex}, base {p.observed_base.hex}, {len(p.commit_refs)} commit binding(s), "
            f"enumeration {p.enumeration_status}, github state {p.github_state}",
            "fact",
            list(p.commit_refs),
            None,
        )
    if t == "commit":
        runs = tested.get(record.record_id, [])
        status = f"tested by {len(runs)} run(s)" if runs else "not_run (no run recorded)"
        components = ", ".join(sorted({c.component for c in p.changes})) or "none extracted"
        return (
            f"Commit binding {p.commit_oid.hex} in {p.pr_ref} ({pr_labels.get(p.pr_ref, 'unknown PR')}): {p.summary} "
            f"[change_status {p.change_status}; {len(p.changes)} change(s); components {components}; attribution group_only unless isolated; {status}]",
            "fact",
            [p.pr_ref] + runs,
            None,
        )
    if t == "run":
        timing = p.timing.status
        if p.timing.status == "recorded" and p.timing.median_us is not None:
            timing = f"recorded median {p.timing.median_us} us, p90 {p.timing.p90_us} us over {p.timing.sample_count} samples"
        text = (
            f"Run of {p.subject_ref} (stage {p.stage}, provenance {p.provenance}, checkout {p.source.checkout_mode}): "
            f"execution {p.execution_status}, correctness {p.correctness.status} ({p.correctness.cases_passed}/{p.correctness.cases_total}), timing {timing}"
        )
        if p.failure_reason:
            text += f"; failure_reason: {p.failure_reason}"
        if p.provenance in _UNVERIFIED_PROVENANCE:
            text += "; not production evidence"
        return text, "fact", [p.subject_ref] + [a.artifact_id for a in p.artifacts], p.provenance
    if t == "relation":
        return f"Relation {p.kind}: {p.from_ref} -> {p.to_ref}: {p.rationale}", "fact", list(p.evidence_refs), None
    if t == "decision":
        return (
            f"Decision {p.outcome} for {p.candidate_subject_ref} (policy {p.policy.policy_id} {p.policy_hash}, production {str(p.is_production).lower()}, "
            f"evaluated_by {p.evaluated_by}): reasons {', '.join(p.reason_codes) if p.reason_codes else 'none'}",
            "fact",
            list(p.candidate_run_refs) + list(p.baseline_run_refs),
            None,
        )
    if t == "annotation":
        return (
            f"Annotation ({p.category}, author {p.author_kind}, confidence {p.confidence}) on {p.target_ref}: {p.text}",
            annotation_kind(record),
            list(p.evidence_refs),
            None,
        )
    return f"{t} {record.record_id}", "fact", [], None


def memory_records(store: MemoryStore, config_ref: str, scope: ConfigRecords | None = None) -> list[dict[str, Any]]:
    """Compact records for one config, sorted by (type order, id). No timestamps; identical facts -> identical lines."""
    scope = scope or collect_config_records(store, config_ref)
    tested = {sid: [r.record_id for r in runs] for sid, runs in scope.runs_by_subject().items()}
    pr_labels = {pr.record_id: pr_display_label(pr) for pr in scope.prs}
    lines: list[dict[str, Any]] = []
    for record in scope.all_records():
        summary, kind, evidence, provenance = _record_summary(record, tested, pr_labels)
        lines.append(
            {
                "record_ref": record.record_id,
                "record_type": record.record_type,
                "summary": summary,
                "kind": kind,
                "evidence_refs": sorted(set(evidence)),
                "provenance": provenance,
                "config_ref": scope.config_id,
            }
        )
    return lines


# --------------------------------------------------------------------------------------
# Publication, verification, rebuild
# --------------------------------------------------------------------------------------
def publish_trajectory(store: MemoryStore, config_ref: str, *, force: bool = False) -> dict[str, Any]:
    view = build_trajectory(store, config_ref)
    errors = [d for d in view["diagnostics"] if d["severity"] == SEVERITY_ERROR]
    if errors and not force:
        raise InvariantViolation(
            f"trajectory for {view['config_ref']} has {len(errors)} error diagnostic(s); refusing to publish without --force",
            code="TRAJECTORY_NOT_PUBLISHABLE",
            details={"config_ref": view["config_ref"], "errors": errors},
        )
    view_hash = trajectory_hash(view)
    document = {"generated_at": utc_now_iso(), "view_hash": view_hash, "view": view, "forced": bool(errors)}
    scope = collect_config_records(store, config_ref)
    lines = memory_records(store, config_ref, scope)
    jsonl = "".join(dumps_compact(line) + "\n" for line in lines).encode("utf-8")
    with store.lock():
        path = store.write_view(view["config_ref"], TRAJECTORY_VIEW_FILE, dumps_readable(document).encode("utf-8"))
        records_path = store.write_view(view["config_ref"], MEMORY_RECORDS_FILE, jsonl)
    return {
        "config_ref": view["config_ref"],
        "path": str(path),
        "view_hash": view_hash,
        "memory_records_path": str(records_path),
        "record_count": len(lines),
        "publishable": view["publishable"],
        "forced": bool(errors),
        "diagnostics": view["diagnostics"],
    }


def read_stored_trajectory(store: MemoryStore, config_ref: str) -> dict[str, Any] | None:
    raw = store.read_view(store.require(config_ref, "config").record_id, TRAJECTORY_VIEW_FILE)
    if raw is None:
        return None
    document = loads_strict(raw)
    if not isinstance(document, dict) or "view" not in document or "view_hash" not in document:
        raise InvariantViolation("stored trajectory.json is not a trajectory document", code="TRAJECTORY_CORRUPT")
    return document


def verify_trajectory(store: MemoryStore, config_ref: str) -> dict[str, Any]:
    rebuilt = trajectory_hash(build_trajectory(store, config_ref))
    stored = read_stored_trajectory(store, config_ref)
    if stored is None:
        return {"stored": False, "stored_hash": None, "rebuilt_hash": rebuilt, "matches": False, "stored_view_consistent": None}
    stored_hash = stored["view_hash"]
    consistent = trajectory_hash(stored["view"]) == stored_hash
    return {
        "stored": True,
        "stored_hash": stored_hash,
        "rebuilt_hash": rebuilt,
        "matches": bool(consistent and stored_hash == rebuilt),
        "stored_view_consistent": consistent,
    }


def rebuild_trajectory(store: MemoryStore, config_ref: str, *, force: bool = False) -> dict[str, Any]:
    config = store.require(config_ref, "config")
    with store.lock():
        removed = store.delete_views(config.record_id)
        result = publish_trajectory(store, config.record_id, force=force)
    result["removed_views"] = removed
    return result


def view_path(store: MemoryStore, config_ref: str) -> Path:
    return store.config_dir(store.require(config_ref, "config").record_id) / TRAJECTORY_VIEW_FILE


__all__ = [
    "VIEW_VERSION",
    "ConfigRecords",
    "collect_config_records",
    "run_summary",
    "annotation_kind",
    "pr_display_label",
    "change_summary",
    "build_trajectory",
    "trajectory_hash",
    "memory_records",
    "publish_trajectory",
    "read_stored_trajectory",
    "verify_trajectory",
    "rebuild_trajectory",
    "view_path",
    "MissingReferenceError",
]
