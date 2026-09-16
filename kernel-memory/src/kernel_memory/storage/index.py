"""Disposable SQLite index over the authoritative JSON records.

The index never holds exclusive facts. Deleting ``.cache/index.sqlite`` and calling
``SqliteIndex.rebuild`` reproduces it exactly from the JSON files. Queries fall back to
the in-memory scan when the cache is absent or stale (``is_fresh`` compares against the
store's current record digests).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ..domain.jsonio import dumps_compact
from . import layout

if TYPE_CHECKING:  # pragma: no cover
    from .store import MemoryStore

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    relpath TEXT NOT NULL,
    digest TEXT NOT NULL,
    config_ref TEXT,
    subject_ref TEXT,
    pr_ref TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type);
CREATE INDEX IF NOT EXISTS idx_records_config ON records(config_ref);
CREATE INDEX IF NOT EXISTS idx_records_subject ON records(subject_ref);
CREATE TABLE IF NOT EXISTS changes (
    record_id TEXT NOT NULL,
    change_id TEXT NOT NULL,
    component TEXT NOT NULL,
    key TEXT,
    before_json TEXT,
    after_json TEXT,
    attribution TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_changes_component ON changes(component);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def _direct_config_ref(record: Any) -> str | None:
    payload = record.payload
    return getattr(payload, "config_ref", None)


class SqliteIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def default_path(cls, store: "MemoryStore") -> Path:
        return store.root / layout.CACHE_DIR / "index.sqlite"

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> Path:
        path = cls.default_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with store.lock():
            conn = sqlite3.connect(str(tmp))
            try:
                conn.executescript(SCHEMA_SQL)
                fingerprint_parts: list[str] = []
                for entry in store.index_entries():
                    record = store.get(entry.record_id)
                    if record is None:
                        continue
                    payload = record.payload
                    conn.execute(
                        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
                        (
                            record.record_id,
                            record.record_type,
                            entry.relpath,
                            entry.digest,
                            _direct_config_ref(record),
                            getattr(payload, "subject_ref", None),
                            getattr(payload, "pr_ref", None),
                            record.created_at,
                        ),
                    )
                    fingerprint_parts.append(f"{record.record_id}:{entry.digest}")
                    if record.record_type == "commit":
                        for change in payload.changes:
                            conn.execute(
                                "INSERT INTO changes VALUES (?,?,?,?,?,?,?)",
                                (
                                    record.record_id,
                                    change.change_id,
                                    change.component,
                                    change.key,
                                    dumps_compact(change.before),
                                    dumps_compact(change.after),
                                    change.attribution,
                                ),
                            )
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('fingerprint', ?)", ("\n".join(fingerprint_parts),))
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('record_count', ?)", (str(len(fingerprint_parts)),))
                conn.commit()
            finally:
                conn.close()
            tmp.replace(path)
        return path

    def exists(self) -> bool:
        return self.path.is_file()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def fingerprint(self) -> str | None:
        if not self.exists():
            return None
        conn = self._connect()
        try:
            row = conn.execute("SELECT v FROM meta WHERE k='fingerprint'").fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def is_fresh(self, store: "MemoryStore") -> bool:
        expected = "\n".join(f"{e.record_id}:{e.digest}" for e in store.index_entries())
        return self.fingerprint() == expected

    def record_ids(self, *, record_type: str | None = None, config_ref: str | None = None, subject_ref: str | None = None) -> list[str]:
        clauses: list[str] = []
        params: list[Any] = []
        if record_type is not None:
            clauses.append("record_type = ?")
            params.append(record_type)
        if config_ref is not None:
            clauses.append("config_ref = ?")
            params.append(config_ref)
        if subject_ref is not None:
            clauses.append("subject_ref = ?")
            params.append(subject_ref)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(f"SELECT record_id FROM records{where} ORDER BY record_id", params).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def commits_by_component(self, component: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT DISTINCT record_id FROM changes WHERE component = ? ORDER BY record_id", (component,)).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def rows(self) -> Iterator[tuple[Any, ...]]:
        conn = self._connect()
        try:
            yield from conn.execute("SELECT record_id, record_type, relpath, digest FROM records ORDER BY record_id")
        finally:
            conn.close()
