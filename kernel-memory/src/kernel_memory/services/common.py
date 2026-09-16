"""Shared accessors and constructors used by all services (append-only module)."""
from __future__ import annotations

from typing import Any, Iterable

from ..domain.errors import MissingReferenceError
from ..domain.ids import utc_now_iso
from ..domain.models import PAYLOAD_TYPES, Record, to_json
from ..storage.store import MemoryStore


def now() -> str:
    return utc_now_iso()


def new_record(record_type: str, record_id: str, payload: Any, *, created_at: str | None = None) -> Record:
    """Build a validated record from a typed payload dataclass (or a payload dict)."""
    if isinstance(payload, dict):
        data = {
            "schema_version": "0.2.0",
            "record_type": record_type,
            "record_id": record_id,
            "created_at": created_at or now(),
            "payload": payload,
        }
        return Record.from_dict(data)
    if not isinstance(payload, PAYLOAD_TYPES[record_type]):
        raise TypeError(f"payload for {record_type} must be {PAYLOAD_TYPES[record_type].__name__}")
    return Record.create(record_type, record_id, payload, created_at=created_at or now())


def config_of(store: MemoryStore, record: Record) -> Record:
    """Resolve the config record that owns any record (kernel records raise)."""
    t = record.record_type
    p = record.payload
    if t == "config":
        return record
    if t in ("pr", "baseline", "run", "relation", "decision"):
        return store.require(p.config_ref, "config")
    if t in ("commit", "pr_snapshot"):
        return config_of(store, store.require(p.pr_ref, "pr"))
    if t == "annotation":
        return config_of(store, store.require(p.target_ref))
    raise MissingReferenceError(f"{t} record {record.record_id!r} has no owning config")


def kernel_of(store: MemoryStore, config: Record) -> Record:
    kernel = store.kernel_by_kernel_id(config.payload.kernel_id)
    if kernel is None:
        raise MissingReferenceError(f"kernel {config.payload.kernel_id!r} not found")
    return kernel


def pr_of(store: MemoryStore, commit: Record) -> Record:
    return store.require(commit.payload.pr_ref, "pr")


def subject_of(store: MemoryStore, run: Record) -> Record:
    return store.require(run.payload.subject_ref, "commit", "baseline")


def configs_for_kernel(store: MemoryStore, kernel_id: str) -> list[Record]:
    return [c for c in store.records("config") if c.payload.kernel_id == kernel_id]


def prs_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    return [p for p in store.records("pr") if p.payload.config_ref == config_id]


def baselines_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    return [b for b in store.records("baseline") if b.payload.config_ref == config_id]


def commits_for_pr(store: MemoryStore, pr_id: str) -> list[Record]:
    return [c for c in store.records("commit") if c.payload.pr_ref == pr_id]


def snapshots_for_pr(store: MemoryStore, pr_id: str) -> list[Record]:
    """Snapshots in chain order (following previous_snapshot_ref), then any orphans by id."""
    snaps = [s for s in store.records("pr_snapshot") if s.payload.pr_ref == pr_id]
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


def latest_snapshot(store: MemoryStore, pr_id: str) -> Record | None:
    snaps = snapshots_for_pr(store, pr_id)
    return snaps[-1] if snaps else None


def runs_for_subject(store: MemoryStore, subject_id: str) -> list[Record]:
    return [r for r in store.records("run") if r.payload.subject_ref == subject_id]


def runs_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    return [r for r in store.records("run") if r.payload.config_ref == config_id]


def decisions_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    return [d for d in store.records("decision") if d.payload.config_ref == config_id]


def relations_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    return [r for r in store.records("relation") if r.payload.config_ref == config_id]


def annotations_for_target(store: MemoryStore, target_id: str) -> list[Record]:
    return [a for a in store.records("annotation") if a.payload.target_ref == target_id]


def annotations_for_config(store: MemoryStore, config_id: str) -> list[Record]:
    out: list[Record] = []
    for a in store.records("annotation"):
        try:
            if config_of(store, a).record_id == config_id:
                out.append(a)
        except MissingReferenceError:
            continue
    return out


def payload_dict(record: Record) -> dict[str, Any]:
    return to_json(record.payload)


def sorted_by_id(records: Iterable[Record]) -> list[Record]:
    return sorted(records, key=lambda r: r.record_id)
