"""Structured Memory retrieval (specification section 14).

Retrieval starts from structured filters, never from free text: kernel, config (by record id or
by ``config_hash``), record type, changed component / parameter key, subject, PR, execution and
correctness status, provenance, comparison context, decision outcome and reason codes, and the
derived ``tested`` / ``not_run`` status of commits. Every result item carries the original record
id (``record_ref``) so the caller can go back to the authoritative JSON record; the compact
fields are read-only copies, nothing is derived or invented here.

Public API
----------
``QueryFilters``
    Frozen dataclass with exactly the seventeen filter fields the ``kmem query`` command exposes.
    All default to ``None`` (``include_cross_config_hints=False``). ``validate()`` raises
    ``InputError`` (exit code 2) for an unknown ``record_type``, a ``run_status`` outside
    ``("tested", "not_run")`` or a ``limit`` below 1.

``QueryResult``
    ``items`` (compact dicts sorted by ``(TYPE_ORDER[record_type], record_id)``),
    ``cross_config_hints`` (same shape, each carrying ``"cross_config_hint": true``, never mixed
    into ``items``), ``total`` (matching items *before* ``limit``), ``truncated``, ``index_used``
    (whether the disposable SQLite cache narrowed the candidate commits), ``filters`` (the filters
    as a dict) and ``configs`` (the config record ids that formed the scope). ``to_dict()`` is what
    the CLI prints.

``query_memory(store, filters) -> QueryResult``
    Scope resolution: ``config_ref`` -> that one config (an unknown or non-config id raises
    ``MissingReferenceError``, exit code 3); else ``config_hash`` -> ``store.configs_by_hash``; else
    ``kernel_id`` -> every config of that kernel; else every config in the store. A config that is
    named by one scope filter must still satisfy the other scope filters that are given.
    ``ConfigRecords`` (``services.trajectory.collect_config_records``) is built once per config and
    every record in it is tested by the single predicate ``_matches``; the SQLite fast path only
    narrows the candidate commits when a ``component`` filter is given and the cache exists and is
    fresh, so its results are identical to the scan path.

    Cross-config hints are produced only when ``include_cross_config_hints`` is true *and* the scope
    was named by ``config_ref`` / ``config_hash`` / ``kernel_id``: the same predicate runs over the
    other configs of the same kernel(s) and those items are returned in ``cross_config_hints`` (not
    limited by ``limit``, which applies to ``items`` only).

``untested_commits(store, config_ref) -> list[dict]``
    Commits of the config that have no run at all (``reason="no_run_recorded"``), derived at query
    time from the absence of runs; no fake run is ever stored (T05).

Filter semantics (``_matches``)
-------------------------------
A filter that is set must be satisfied by the record; a filter that does not apply to the
record's type excludes that record. ``record_type`` filters by type. ``component`` and
``parameter_key`` apply to commit changes (``component`` equality; ``parameter_key`` equality on
``change.key``). ``run_status`` applies to commits only (``tested`` when the config has at least
one run whose subject is the commit, else ``not_run``). ``subject_ref`` matches runs by
``subject_ref``, decisions by ``candidate_subject_ref`` and annotations by ``target_ref``.
``pr_ref`` matches commits and PR snapshots by ``pr_ref`` and runs through their subject commit's
``pr_ref``. ``execution_status``, ``correctness_status`` and ``provenance`` apply to runs;
``comparison_key`` applies to runs and decisions; ``decision_outcome`` and ``reason_code`` apply to
decisions. Kernel records are listed once per kernel in scope (``config_ref`` null) and never as
cross-config hints.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from ..domain.errors import InputError, MissingReferenceError
from ..domain.models import PAYLOAD_TYPES, TYPE_ORDER, Record, to_json
from ..storage.index import SqliteIndex
from ..storage.store import MemoryStore
from .common import configs_for_kernel
from .trajectory import (
    ConfigRecords,
    annotation_kind,
    change_summary,
    collect_config_records,
    pr_display_label,
    run_summary,
)

RUN_STATUS_TESTED = "tested"
RUN_STATUS_NOT_RUN = "not_run"
RUN_STATUSES: tuple[str, ...] = (RUN_STATUS_TESTED, RUN_STATUS_NOT_RUN)
REASON_NO_RUN_RECORDED = "no_run_recorded"
CROSS_CONFIG_HINT_FIELD = "cross_config_hint"


# --------------------------------------------------------------------------------------
# Filters and result
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QueryFilters:
    """Structured retrieval filters. ``None`` means "not filtered on"."""

    kernel_id: str | None = None
    config_ref: str | None = None
    config_hash: str | None = None
    record_type: str | None = None
    component: str | None = None
    parameter_key: str | None = None
    subject_ref: str | None = None
    pr_ref: str | None = None
    execution_status: str | None = None
    correctness_status: str | None = None
    provenance: str | None = None
    comparison_key: str | None = None
    decision_outcome: str | None = None
    reason_code: str | None = None
    run_status: str | None = None
    include_cross_config_hints: bool = False
    limit: int | None = None

    def validate(self) -> None:
        """Raise ``InputError`` (exit code 2) for values the query cannot interpret."""
        if self.record_type is not None and self.record_type not in PAYLOAD_TYPES:
            raise InputError(
                f"unknown record_type {self.record_type!r}; expected one of {sorted(PAYLOAD_TYPES)}",
                code="INVALID_QUERY",
                details={"field": "record_type", "value": self.record_type, "allowed": sorted(PAYLOAD_TYPES)},
            )
        if self.run_status is not None and self.run_status not in RUN_STATUSES:
            raise InputError(
                f"unknown run_status {self.run_status!r}; expected one of {list(RUN_STATUSES)}",
                code="INVALID_QUERY",
                details={"field": "run_status", "value": self.run_status, "allowed": list(RUN_STATUSES)},
            )
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 1:
                raise InputError(
                    f"limit must be an integer >= 1, got {self.limit!r}",
                    code="INVALID_QUERY",
                    details={"field": "limit", "value": self.limit},
                )

    @property
    def names_scope(self) -> bool:
        """True when the scope is named explicitly (a prerequisite for cross-config hints)."""
        return self.config_ref is not None or self.config_hash is not None or self.kernel_id is not None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class QueryResult:
    items: list[dict[str, Any]]
    cross_config_hints: list[dict[str, Any]]
    total: int
    truncated: bool
    index_used: bool
    filters: dict[str, Any]
    configs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [dict(item) for item in self.items],
            "cross_config_hints": [dict(item) for item in self.cross_config_hints],
            "total": self.total,
            "truncated": self.truncated,
            "index_used": self.index_used,
            "filters": dict(self.filters),
            "configs": list(self.configs),
        }


# --------------------------------------------------------------------------------------
# Per-config scope (ConfigRecords plus the lookups the predicate needs)
# --------------------------------------------------------------------------------------
class _Scope:
    """One config's authoritative records and the derived lookups shared by predicate and items."""

    def __init__(self, records: ConfigRecords) -> None:
        self.records = records
        self.config_id = records.config_id
        self.run_refs_by_subject: dict[str, list[str]] = {
            subject: [run.record_id for run in runs] for subject, runs in records.runs_by_subject().items()
        }
        self.commits_by_id: dict[str, Record] = {commit.record_id: commit for commit in records.commits}

    def commit_status(self, commit_id: str) -> str:
        """``tested`` when at least one run of this config has the commit as subject, else ``not_run``."""
        return RUN_STATUS_TESTED if self.run_refs_by_subject.get(commit_id) else RUN_STATUS_NOT_RUN

    def run_pr_ref(self, run: Record) -> str | None:
        """The PR of a run's subject commit; ``None`` for baseline subjects or unresolved commits."""
        commit = self.commits_by_id.get(run.payload.subject_ref)
        return commit.payload.pr_ref if commit is not None else None

    def records_in_order(self, candidate_commit_ids: set[str] | None) -> list[Record]:
        r = self.records
        commits = r.commits if candidate_commit_ids is None else [c for c in r.commits if c.record_id in candidate_commit_ids]
        return [r.config, *r.baselines, *r.prs, *r.snapshots, *commits, *r.runs, *r.relations, *r.decisions, *r.annotations]


# --------------------------------------------------------------------------------------
# The single predicate
# --------------------------------------------------------------------------------------
def _matches(record: Record, scope: _Scope, filters: QueryFilters) -> bool:
    """Sole source of truth for whether ``record`` satisfies ``filters`` within ``scope``."""
    t = record.record_type
    p = record.payload
    if filters.record_type is not None and t != filters.record_type:
        return False
    if filters.component is not None:
        if t != "commit" or not any(change.component == filters.component for change in p.changes):
            return False
    if filters.parameter_key is not None:
        if t != "commit" or not any(change.key == filters.parameter_key for change in p.changes):
            return False
    if filters.run_status is not None:
        if t != "commit" or scope.commit_status(record.record_id) != filters.run_status:
            return False
    if filters.subject_ref is not None:
        if t == "run":
            matched = p.subject_ref == filters.subject_ref
        elif t == "decision":
            matched = p.candidate_subject_ref == filters.subject_ref
        elif t == "annotation":
            matched = p.target_ref == filters.subject_ref
        else:
            matched = False
        if not matched:
            return False
    if filters.pr_ref is not None:
        if t in ("commit", "pr_snapshot"):
            matched = p.pr_ref == filters.pr_ref
        elif t == "run":
            matched = scope.run_pr_ref(record) == filters.pr_ref
        else:
            matched = False
        if not matched:
            return False
    if filters.execution_status is not None:
        if t != "run" or p.execution_status != filters.execution_status:
            return False
    if filters.correctness_status is not None:
        if t != "run" or p.correctness.status != filters.correctness_status:
            return False
    if filters.provenance is not None:
        if t != "run" or p.provenance != filters.provenance:
            return False
    if filters.comparison_key is not None:
        if t not in ("run", "decision") or p.comparison_key != filters.comparison_key:
            return False
    if filters.decision_outcome is not None:
        if t != "decision" or p.outcome != filters.decision_outcome:
            return False
    if filters.reason_code is not None:
        if t != "decision" or filters.reason_code not in p.reason_codes:
            return False
    return True


# --------------------------------------------------------------------------------------
# Compact items
# --------------------------------------------------------------------------------------
def _type_fields(record: Record, scope: _Scope) -> dict[str, Any]:
    t = record.record_type
    p = record.payload
    if t == "commit":
        return {
            "commit_oid": p.commit_oid.hex,
            "pr_ref": p.pr_ref,
            "change_status": p.change_status,
            "changes": [change_summary(change) for change in p.changes],
            "summary": p.summary,
            "status": scope.commit_status(record.record_id),
            "run_refs": list(scope.run_refs_by_subject.get(record.record_id, [])),
        }
    if t == "run":
        fields = run_summary(record)
        fields["subject_ref"] = p.subject_ref
        fields["pr_ref"] = scope.run_pr_ref(record)
        return fields
    if t == "pr":
        return {
            "pr_key": p.pr_key,
            "provider": p.provider,
            "number": p.number,
            "title": p.title,
            "display_label": pr_display_label(record),
        }
    if t == "decision":
        return {
            "outcome": p.outcome,
            "reason_codes": list(p.reason_codes),
            "is_production": p.is_production,
            "comparison_key": p.comparison_key,
            "candidate_subject_ref": p.candidate_subject_ref,
        }
    if t == "annotation":
        return {
            "category": p.category,
            "kind": annotation_kind(record),
            "confidence": p.confidence,
            "target_ref": p.target_ref,
            "text": p.text,
        }
    if t == "baseline":
        return {"baseline_id": p.baseline_id, "role": p.role, "entrypoint": p.entrypoint}
    if t == "pr_snapshot":
        return {
            "pr_ref": p.pr_ref,
            "enumeration_status": p.enumeration_status,
            "observed_head": to_json(p.observed_head),
            "commit_refs": list(p.commit_refs),
        }
    if t == "relation":
        return {"kind": p.kind, "from_ref": p.from_ref, "to_ref": p.to_ref}
    if t == "config":
        return {
            "kernel_id": p.kernel_id,
            "config_id": p.config_id,
            "config_hash": p.config_hash,
            "problem_schema_id": p.problem_schema_id,
            "tags": list(p.tags),
        }
    if t == "kernel":
        return {"kernel_id": p.kernel_id, "display_name": p.display_name, "adapter_id": p.adapter_id}
    return {}


def _item(record: Record, scope: _Scope) -> dict[str, Any]:
    item: dict[str, Any] = {
        "record_ref": record.record_id,
        "record_type": record.record_type,
        "config_ref": None if record.record_type == "kernel" else scope.config_id,
    }
    item.update(_type_fields(record, scope))
    return item


def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (TYPE_ORDER[item["record_type"]], item["record_ref"])


def _scope_items(
    scope: _Scope,
    filters: QueryFilters,
    candidate_commit_ids: set[str] | None,
    seen_kernels: set[str] | None,
) -> list[dict[str, Any]]:
    """Items of one config scope. ``seen_kernels`` (None to skip kernels) dedupes kernel records across scopes."""
    out: list[dict[str, Any]] = []
    kernel = scope.records.kernel
    if seen_kernels is not None and kernel is not None and kernel.record_id not in seen_kernels:
        seen_kernels.add(kernel.record_id)
        if _matches(kernel, scope, filters):
            out.append(_item(kernel, scope))
    for record in scope.records_in_order(candidate_commit_ids):
        if _matches(record, scope, filters):
            out.append(_item(record, scope))
    return out


# --------------------------------------------------------------------------------------
# Scope resolution and the SQLite fast path
# --------------------------------------------------------------------------------------
def _by_id(records: Iterable[Record]) -> list[Record]:
    return sorted(records, key=lambda r: r.record_id)


def _resolve_scope(store: MemoryStore, filters: QueryFilters) -> list[Record]:
    if filters.config_ref is not None:
        configs = [store.require(filters.config_ref, "config")]
    elif filters.config_hash is not None:
        configs = _by_id(store.configs_by_hash(filters.config_hash))
    elif filters.kernel_id is not None:
        configs = _by_id(configs_for_kernel(store, filters.kernel_id))
    else:
        configs = _by_id(store.records("config"))
    # A named config must still satisfy every other scope filter that was given.
    return [
        c
        for c in configs
        if (filters.config_hash is None or c.payload.config_hash == filters.config_hash)
        and (filters.kernel_id is None or c.payload.kernel_id == filters.kernel_id)
    ]


def _other_configs_of_same_kernels(store: MemoryStore, scope_configs: list[Record]) -> list[Record]:
    kernel_ids = {c.payload.kernel_id for c in scope_configs}
    in_scope = {c.record_id for c in scope_configs}
    return _by_id(c for c in store.records("config") if c.payload.kernel_id in kernel_ids and c.record_id not in in_scope)


def _index_candidate_commits(store: MemoryStore, filters: QueryFilters) -> tuple[set[str] | None, bool]:
    """Commit ids the fresh SQLite cache reports for ``component`` (a superset of the matches), or None."""
    if filters.component is None:
        return None, False
    index = SqliteIndex(SqliteIndex.default_path(store))
    try:
        if not index.exists() or not index.is_fresh(store):
            return None, False
        return set(index.commits_by_component(filters.component)), True
    except sqlite3.Error:
        # A damaged cache never holds exclusive facts: fall back to the scan path.
        return None, False


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def query_memory(store: MemoryStore, filters: QueryFilters) -> QueryResult:
    """Structured retrieval over the authoritative records (see module docstring)."""
    filters.validate()
    configs = _resolve_scope(store, filters)
    candidate_commit_ids, index_used = _index_candidate_commits(store, filters)

    items: list[dict[str, Any]] = []
    seen_kernels: set[str] = set()
    for config in configs:
        scope = _Scope(collect_config_records(store, config.record_id))
        items.extend(_scope_items(scope, filters, candidate_commit_ids, seen_kernels))
    items.sort(key=_sort_key)
    total = len(items)
    truncated = False
    if filters.limit is not None and total > filters.limit:
        items = items[: filters.limit]
        truncated = True

    hints: list[dict[str, Any]] = []
    if filters.include_cross_config_hints and filters.names_scope:
        for other in _other_configs_of_same_kernels(store, configs):
            scope = _Scope(collect_config_records(store, other.record_id))
            for item in _scope_items(scope, filters, candidate_commit_ids, None):
                item[CROSS_CONFIG_HINT_FIELD] = True
                hints.append(item)
        hints.sort(key=_sort_key)

    return QueryResult(
        items=items,
        cross_config_hints=hints,
        total=total,
        truncated=truncated,
        index_used=index_used,
        filters=filters.to_dict(),
        configs=[c.record_id for c in configs],
    )


def untested_commits(store: MemoryStore, config_ref: str) -> list[dict[str, Any]]:
    """Commits of ``config_ref`` without any run, derived from absence (T05). Sorted by commit id."""
    records = collect_config_records(store, config_ref)
    tested = records.runs_by_subject()
    return [
        {
            "commit_ref": commit.record_id,
            "pr_ref": commit.payload.pr_ref,
            "commit_oid": commit.payload.commit_oid.hex,
            "reason": REASON_NO_RUN_RECORDED,
        }
        for commit in records.commits
        if commit.record_id not in tested
    ]


__all__ = [
    "RUN_STATUS_TESTED",
    "RUN_STATUS_NOT_RUN",
    "RUN_STATUSES",
    "REASON_NO_RUN_RECORDED",
    "CROSS_CONFIG_HINT_FIELD",
    "QueryFilters",
    "QueryResult",
    "query_memory",
    "untested_commits",
    "MissingReferenceError",
]
