"""Shared pytest fixtures. Tests never touch the network or the handoff directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Fallback for environments where the editable-install .pth is ignored (macOS can mark it
# UF_HIDDEN and Python 3.11.4+ skips hidden .pth files). The editable install remains the
# primary mechanism; this only guarantees tests import the in-tree package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kernel_memory.domain.jsonio import load_json_file  # noqa: E402
from kernel_memory.domain.models import Record
from kernel_memory.storage import MemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = PROJECT_ROOT / "fixtures" / "handoff"
BUNDLE_PATH = FIXTURES_ROOT / "examples" / "demo_bundle.json"
ARTIFACT_ROOT = FIXTURES_ROOT  # bundle artifact URIs are relative to this directory


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_root() -> Path:
    return FIXTURES_ROOT


@pytest.fixture(scope="session")
def bundle_path() -> Path:
    return BUNDLE_PATH


@pytest.fixture(scope="session")
def artifact_root() -> Path:
    return ARTIFACT_ROOT


@pytest.fixture(scope="session")
def bundle_dicts() -> list[dict]:
    return json.loads(json.dumps(load_json_file(BUNDLE_PATH)["records"]))


@pytest.fixture
def bundle_records(bundle_dicts: list[dict]) -> list[Record]:
    return [Record.from_dict(json.loads(json.dumps(d))) for d in bundle_dicts]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore.init(tmp_path / "memory")


def import_demo_bundle(store: MemoryStore, records: list[Record], artifact_root: Path) -> None:
    """Publish the fixture bundle and store its artifacts (raw store path, bypassing the importer service)."""
    store.publish_bundle(records, label="demo-fixture")
    for record in records:
        if record.record_type != "run":
            continue
        for ref in record.payload.artifacts:
            path = artifact_root / ref.uri
            store.import_artifact_file(path, expected_sha256=ref.sha256, expected_size=ref.size_bytes)


@pytest.fixture
def demo_store(tmp_path: Path, bundle_records: list[Record]) -> MemoryStore:
    store = MemoryStore.init(tmp_path / "memory")
    import_demo_bundle(store, bundle_records, ARTIFACT_ROOT)
    return store


def record_dict(bundle_dicts: list[dict], record_id: str) -> dict:
    return json.loads(json.dumps(next(r for r in bundle_dicts if r["record_id"] == record_id)))
