"""End-to-end tests for the ``kmem`` command line (specification section 16; DESIGN section 9).

The CLI is driven in-process through ``kernel_memory.cli.main.main(argv)``. Every command is a
thin wrapper over a service, so these tests assert the *contract* of the wrapper: exit codes,
a single JSON document on stdout for results, a single JSON error object on stderr for
failures, and the binding rules visible through the CLI (fixtures never promote, uncollected
metrics stay null, different protocols are NOT_COMPARABLE, replays execute nothing, unknown
backends are denied before any adapter is consulted, TPU execution is refused honestly).

Fixture data: ``fixtures/handoff/examples/demo_bundle.json`` (18 verbatim v0.2 synthetic records,
upgraded on import to 19 with the placeholder algorithm ``algorithm-demo_vector_add-unspecified``;
the 100/90 us medians are fictional and exist only to exercise the arithmetic).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from kernel_memory.adapters.base import AdapterRegistry
from kernel_memory.adapters.mock import MockAdapter
from kernel_memory.cli import main as cli
from kernel_memory.domain import stats
from kernel_memory.execution.ledger import RequestLedger
from kernel_memory.services.optimize import MOCK_PLANNER_NOTE
from kernel_memory.storage import MemoryStore

BUNDLE_REL = Path("examples") / "demo_bundle.json"
FIXTURE_RECORD_COUNT = 19  # 18 verbatim records + the placeholder algorithm synthesized by the v0.2 -> v0.3 upgrade
PLACEHOLDER_ALGORITHM = "algorithm-demo_vector_add-unspecified"
UNTESTED_COMMITS = {"commit-demo-a-in-102", "commit-demo-b"}
TESTED_COMMITS = {"commit-demo-a", "commit-demo-c"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
class Invocation:
    """Result of one in-process ``kmem`` call with the stdout/stderr contract already checked."""

    def __init__(self, argv: list[str], code: int, out: str, err: str) -> None:
        self.argv = argv
        self.code = code
        self.out = out
        self.err = err

    def _one_json_document(self, text: str, stream: str) -> Any:
        assert text.endswith("\n"), f"{stream} of {self.argv} must end with a newline: {text!r}"
        assert text.count("\n") == 1, f"{stream} of {self.argv} must be exactly one JSON line: {text!r}"
        return json.loads(text)

    @property
    def result(self) -> dict[str, Any]:
        """Success contract: exit 0, exactly one JSON document on stdout, empty stderr."""
        assert self.code == 0, f"{self.argv} exited {self.code}; stderr={self.err!r}"
        assert self.err == "", f"{self.argv} wrote to stderr on success: {self.err!r}"
        data = self._one_json_document(self.out, "stdout")
        assert isinstance(data, dict)
        return data

    def error(self, code: int) -> dict[str, Any]:
        """Error contract: the given exit code, empty stdout, one JSON error object on stderr."""
        assert self.code == code, f"{self.argv} exited {self.code} (expected {code}); stdout={self.out!r} stderr={self.err!r}"
        assert self.out == "", f"{self.argv} wrote to stdout on error: {self.out!r}"
        data = self._one_json_document(self.err, "stderr")
        assert isinstance(data, dict)
        assert set(data) >= {"error", "message", "exit_code"}, data
        assert data["exit_code"] == code
        assert isinstance(data["error"], str) and data["error"]
        assert isinstance(data["message"], str) and data["message"]
        return data

    def reported(self, code: int) -> dict[str, Any]:
        """Non-zero *result* contract (compare/decide/unavailable backend): JSON on stdout, empty stderr."""
        assert self.code == code, f"{self.argv} exited {self.code} (expected {code}); stdout={self.out!r} stderr={self.err!r}"
        assert self.err == "", f"{self.argv} wrote to stderr: {self.err!r}"
        data = self._one_json_document(self.out, "stdout")
        assert isinstance(data, dict)
        return data


def invoke(capsys: pytest.CaptureFixture[str], *argv: str, json_flag: bool = True) -> Invocation:
    capsys.readouterr()  # start from clean buffers
    args = [*argv, "--json"] if json_flag else list(argv)
    code = cli.main(args)
    out, err = capsys.readouterr()
    return Invocation(args, code, out, err)


def store_ids(root: Path, record_type: str | None = None) -> list[str]:
    return [r.record_id for r in MemoryStore.open(root).records(record_type)]


def file_listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def write_bundle_copy(bundle_path: Path, target: Path, mutate) -> Path:
    bundle = json.loads(bundle_path.read_text())
    mutate(bundle)
    target.write_text(json.dumps(bundle))
    return target


def find_record(bundle: dict, record_id: str) -> dict:
    return next(r for r in bundle["records"] if r["record_id"] == record_id)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_cli_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KMEM_ROOT", raising=False)
    monkeypatch.delenv("KMEM_SETTINGS", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_SETTINGS_CACHE", {})


@pytest.fixture
def cli_root(tmp_path: Path, capsys: pytest.CaptureFixture[str], bundle_path: Path, fixtures_root: Path) -> Path:
    """A store initialised through the CLI with the fixture bundle imported (artifacts included)."""
    root = tmp_path / "memory"
    assert invoke(capsys, "init", "--root", str(root)).result["created"] is True
    report = invoke(
        capsys, "import-bundle", str(bundle_path), "--root", str(root), "--artifact-root", str(fixtures_root), "--allow-fixture"
    ).result
    assert report["records_published"] == FIXTURE_RECORD_COUNT
    return root


@pytest.fixture
def pairs_file(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([{"candidate_run": "run-demo-a", "baseline_run": "run-demo-baseline"}]))
    return path


@pytest.fixture
def v01_dir(tmp_path: Path) -> Path:
    """A tiny SYNTHETIC v0.1 export (fictional object ids), one file per top-level key."""
    root = tmp_path / "v01"
    root.mkdir()
    (root / "kernels.json").write_text(
        json.dumps([{"kernel_id": "demo_vector_add", "display_name": "SYNTHETIC v0.1 vector add", "adapter_id": "legacy-cpu"}])
    )
    (root / "configs.json").write_text(json.dumps([{"config_id": "cfg-1", "kernel_id": "demo_vector_add", "problem": {"n": 16, "dtype": "f32"}}]))
    (root / "attempts.json").write_text(
        json.dumps(
            [{"attempt_id": "att-1", "kernel_id": "demo_vector_add", "config_id": "cfg-1", "pr_number": 7, "repo": "org/repo", "title": "SYNTHETIC attempt"}]
        )
    )
    (root / "revisions.json").write_text(
        json.dumps(
            [{"revision_id": "rev-1", "attempt_id": "att-1", "commit_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", "summary": "SYNTHETIC revision"}]
        )
    )
    return root


# --------------------------------------------------------------------------------------
# root resolution and output contract
# --------------------------------------------------------------------------------------
def test_init_creates_store_and_second_init_is_noop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "memory"
    first = invoke(capsys, "init", "--root", str(root)).result
    assert first["created"] is True
    assert (root / "manifest.json").is_file()
    assert first["manifest"]["schema_version"] == "0.3.0"
    assert first["manifest"]["layout_version"] == 2
    second = invoke(capsys, "init", "--root", str(root)).result
    assert second["created"] is False
    assert second["manifest"]["store_id"] == first["manifest"]["store_id"], "re-init must not overwrite the store identity"


def test_root_before_and_after_subcommand_are_equivalent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "memory"
    before = invoke(capsys, "--root", str(root), "init").result
    assert before["created"] is True
    after = invoke(capsys, "init", "--root", str(root)).result
    assert after["created"] is False
    assert Path(after["root"]) == Path(before["root"])
    # the global --json before the subcommand is honoured as well
    capsys.readouterr()
    code = cli.main(["--json", "--root", str(root), "integrity"])
    out, err = capsys.readouterr()
    assert code == 0 and err == "" and out.count("\n") == 1 and json.loads(out)["ok"] is True


def test_missing_root_defaults_to_memory_in_cwd(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = invoke(capsys, "init").result
    assert result["created"] is True
    assert (tmp_path / "memory" / "manifest.json").is_file()
    assert Path(result["root"]).resolve() == (tmp_path / "memory").resolve()


def test_kmem_root_environment_variable_is_used_when_no_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "env-memory"
    monkeypatch.setenv("KMEM_ROOT", str(root))
    assert invoke(capsys, "init").result["created"] is True
    assert (root / "manifest.json").is_file()
    assert Path(invoke(capsys, "status").result["root"]) == root


def test_human_readable_output_without_json_flag_is_still_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "memory"
    inv = invoke(capsys, "init", "--root", str(root), json_flag=False)
    assert inv.code == 0 and inv.err == ""
    assert inv.out.count("\n") > 1, "human-readable mode is indented multi-line JSON"
    assert json.loads(inv.out)["created"] is True


def test_validate_on_non_store_root_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "validate", "--root", str(tmp_path / "not-a-store")).error(2)
    assert err["error"] == "NOT_A_STORE"
    err = invoke(capsys, "validate", "--deep", "--root", str(tmp_path / "not-a-store")).error(2)
    assert err["error"] == "NOT_A_STORE"


# --------------------------------------------------------------------------------------
# import / validate / integrity / status / recover / reindex
# --------------------------------------------------------------------------------------
def test_import_bundle_then_reimport_is_idempotent(cli_root: Path, capsys: pytest.CaptureFixture[str], bundle_path: Path, fixtures_root: Path) -> None:
    report = invoke(
        capsys, "import-bundle", str(bundle_path), "--root", str(cli_root), "--artifact-root", str(fixtures_root), "--allow-fixture"
    ).result
    assert report["ok"] is True
    assert report["is_fixture"] is True
    assert report["records_total"] == FIXTURE_RECORD_COUNT
    assert report["records_idempotent"] == FIXTURE_RECORD_COUNT
    assert report["records_published"] == 0
    assert report["artifacts_stored"] == 0 and report["artifacts_idempotent"] > 0
    assert len(store_ids(cli_root)) == FIXTURE_RECORD_COUNT
    # fixture runs stay fixture: the importer warns and never upgrades provenance
    assert {i["code"] for i in report["issues"]} == {"FIXTURE_RUN"}
    assert all(i["severity"] == "warning" for i in report["issues"])
    assert report["provenance_downgraded"] == []


def test_import_bundle_without_allow_fixture_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str], bundle_path: Path, fixtures_root: Path) -> None:
    root = tmp_path / "memory"
    invoke(capsys, "init", "--root", str(root)).result
    err = invoke(capsys, "import-bundle", str(bundle_path), "--root", str(root), "--artifact-root", str(fixtures_root)).error(2)
    assert err["error"] == "FIXTURE_REQUIRES_FLAG"
    assert err["details"]["is_fixture"] is True
    assert store_ids(root) == [], "nothing may be published when the fixture flag is missing"


def test_t12_edited_record_in_bundle_copy_conflicts(cli_root: Path, capsys: pytest.CaptureFixture[str], bundle_path: Path, fixtures_root: Path, tmp_path: Path) -> None:
    before = MemoryStore.open(cli_root).get("cfg-demo").canonical_digest()
    edited = write_bundle_copy(
        bundle_path, tmp_path / "edited_bundle.json", lambda b: find_record(b, "cfg-demo")["payload"]["tags"].append("changed-after-publication")
    )
    err = invoke(capsys, "import-bundle", str(edited), "--root", str(cli_root), "--artifact-root", str(fixtures_root), "--allow-fixture").error(3)
    assert err["error"] == "ID_CONFLICT"
    assert err["details"]["record_id"] == "cfg-demo"
    store = MemoryStore.open(cli_root)
    assert store.get("cfg-demo").canonical_digest() == before, "published records are immutable"
    assert len(store.records()) == FIXTURE_RECORD_COUNT
    # negative control: an unedited copy of the same bundle is idempotent, not a conflict
    same = write_bundle_copy(bundle_path, tmp_path / "same_bundle.json", lambda b: None)
    report = invoke(capsys, "import-bundle", str(same), "--root", str(cli_root), "--artifact-root", str(fixtures_root), "--allow-fixture").result
    assert report["records_idempotent"] == FIXTURE_RECORD_COUNT and report["records_published"] == 0


def test_import_bundle_with_traversal_artifact_uri_is_security_error(tmp_path: Path, capsys: pytest.CaptureFixture[str], bundle_path: Path, fixtures_root: Path) -> None:
    root = tmp_path / "memory"
    invoke(capsys, "init", "--root", str(root)).result
    unsafe = write_bundle_copy(
        bundle_path, tmp_path / "unsafe_bundle.json", lambda b: find_record(b, "run-demo-a")["payload"]["artifacts"][0].update({"uri": "../outside.json"})
    )
    err = invoke(capsys, "import-bundle", str(unsafe), "--root", str(root), "--artifact-root", str(fixtures_root), "--allow-fixture").error(7)
    assert err["error"] == "UNSAFE_PATH"
    assert "../outside.json" in err["message"]
    assert store_ids(root) == [], "a refused bundle publishes nothing"


def test_validate_deep_and_integrity_are_clean_after_import(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deep = invoke(capsys, "validate", "--deep", "--root", str(cli_root)).result
    assert deep["ok"] is True and deep["deep"] is True and deep["error_count"] == 0
    shallow = invoke(capsys, "validate", "--root", str(cli_root)).result
    assert shallow["deep"] is False and shallow["records"] == FIXTURE_RECORD_COUNT and shallow["integrity"]["ok"] is True
    integrity = invoke(capsys, "integrity", "--root", str(cli_root)).result
    assert integrity["ok"] is True
    assert integrity["records_checked"] == FIXTURE_RECORD_COUNT
    assert integrity["modified"] == [] and integrity["missing"] == [] and integrity["corrupt"] == []


def test_integrity_reports_tampered_record_with_exit_2(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore.open(cli_root)
    path = store.record_path("commit-demo-b")
    data = json.loads(path.read_text())
    data["payload"]["summary"] = "tampered outside the store API"
    path.write_text(json.dumps(data))
    inv = invoke(capsys, "integrity", "--root", str(cli_root))
    report = inv.reported(2)
    assert report["ok"] is False
    assert "commit-demo-b" in json.dumps(report["modified"])


def test_status_reports_counts_and_exact_missing_inputs(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "status", "--root", str(cli_root)).result
    assert Path(result["root"]) == cli_root
    assert result["records"] == {
        "algorithm": 1,
        "annotation": 1,
        "baseline": 1,
        "commit": 4,
        "config": 1,
        "decision": 1,
        "kernel": 1,
        "pr": 2,
        "pr_snapshot": 2,
        "relation": 1,
        "run": 4,
    }
    kernel = result["kernels"]["demo_vector_add"]
    assert kernel["kernel_ref"] == "kernel-demo" and kernel["shapes"] == 1
    assert kernel["algorithms"] == {"unspecified": {"algorithm_ref": PLACEHOLDER_ALGORITHM, "shapes": 1, "is_placeholder": True}}
    assert kernel["trajectory_dir"] == "kernels/demo_vector_add/trajectory"
    assert result["store"] == {"store_version": "0.3.0", "schema_version": "0.3.0", "layout_version": 2}
    pending = {item["integration"]: item for item in result["pending_integrations"]}
    github = pending["live GitHub collection"]
    assert github["status"] == "unexecuted"
    assert github["missing"] == ["github_repository", "env GITHUB_TOKEN", "permissions.allow_network"]
    tpu = pending["TPU execution"]
    assert tpu["status"] == "unexecuted"
    assert "permissions.allow_tpu_execution" in tpu["missing"]
    assert {"kernel_entrypoint", "problem_schema_path", "trusted_reference_entrypoint", "approved_verifier_path"} <= set(tpu["missing"])
    if not result["tpu_available"]:
        assert "TPU device" in tpu["missing"]
    planner = pending["model-driven planner"]
    assert planner["missing"] == ["model_provider", "model_api_key_env_name + value", "permissions.allow_model_api_calls"]
    assert pending["LLO analysis"]["status"] == "unsupported"
    assert pending["remote PR write actions"]["status"] == "disabled"


def test_status_without_store_omits_record_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "status", "--root", str(tmp_path / "nothing-here")).result
    assert "records" not in result
    assert result["pending_integrations"]


def test_recover_and_reindex(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recover = invoke(capsys, "recover", "--root", str(cli_root)).result
    assert recover["store"]["problems"] == []
    assert recover["requests"] == {"checked": 0, "lease_lost": [], "repaired": [], "still_leased": []}
    reindex = invoke(capsys, "reindex", "--root", str(cli_root)).result
    assert reindex["records"] == FIXTURE_RECORD_COUNT
    index = Path(reindex["index"])
    assert index.is_file() and index.name == "index.sqlite" and index.parent.name == ".cache"
    assert index.resolve().is_relative_to(cli_root.resolve())


# --------------------------------------------------------------------------------------
# register-config
# --------------------------------------------------------------------------------------
def test_register_config_rejects_non_object_problem(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "register-config", "--kernel-id", "demo_vector_add", "--problem", "[1]", "--root", str(cli_root)).error(2)
    assert err["error"] == "INPUT_ERROR"
    assert "JSON object" in err["message"]
    assert store_ids(cli_root, "config") == ["cfg-demo"]


def test_register_config_rejects_zero_dimension(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--problem", '{"n": 0, "dtype": "float32"}', "--root", str(cli_root)
    ).error(2)
    assert err["error"] == "INVALID_PROBLEM"
    assert store_ids(cli_root, "config") == ["cfg-demo"]


def test_register_config_mla_forward_is_incomplete_contract(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "register-config", "--kernel-id", "mla_forward", "--problem", '{"batch": 1}', "--root", str(cli_root)).error(5)
    assert err["error"] == "INCOMPLETE_PROBLEM_CONTRACT"
    assert err["details"]["kernel_id"] == "mla_forward"
    assert err["details"]["unresolved"], "the refusal names what is still unresolved"
    assert store_ids(cli_root, "config") == ["cfg-demo"] and store_ids(cli_root, "kernel") == ["kernel-demo"]


def test_t01_register_config_reuses_existing_by_config_hash(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    same = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--problem", '{"n": 16, "dtype": "f32"}', "--root", str(cli_root)
    ).result
    assert same["created"] is False
    assert same["config"]["record_id"] == "cfg-demo"
    assert same["algorithm_ref"] == PLACEHOLDER_ALGORITHM and same["algorithm_id"] == "unspecified"
    assert same["config_hash"] == MemoryStore.open(cli_root).get("cfg-demo").payload.config_hash
    other = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--problem", '{"n": 32, "dtype": "float32"}', "--root", str(cli_root)
    ).result
    assert other["created"] is True
    assert other["config"]["record_id"].startswith("cfg-unspecified-demo-n32-f32-"), "new ids carry the algorithm"
    assert other["config_hash"] != same["config_hash"]
    assert len(store_ids(cli_root, "config")) == 2


# --------------------------------------------------------------------------------------
# register-algorithm and algorithm resolution (ADR-0004)
# --------------------------------------------------------------------------------------
def test_register_algorithm_creates_idempotently_and_conflicts_on_different_summary(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    created = invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "numpy-add",
        "--method-summary", "numpy elementwise add on preallocated arrays", "--tag", "cpu-demo", "--root", str(cli_root),
    ).result
    assert created["created"] is True and created["algorithm"]["record_id"] == "algorithm-demo_vector_add-numpy-add"
    assert created["algorithm_id"] == "numpy-add" and created["is_placeholder"] is False and created["summary_author"] == "human"
    assert created["method_summary"] == "numpy elementwise add on preallocated arrays"
    again = invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "numpy-add",
        "--method-summary", "numpy elementwise add on preallocated arrays", "--tag", "cpu-demo", "--root", str(cli_root),
    ).result
    assert again["created"] is False
    err = invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "numpy-add",
        "--method-summary", "a different description", "--root", str(cli_root),
    ).error(3)
    assert err["error"] == "ID_CONFLICT"
    store = MemoryStore.open(cli_root)
    assert store.record_path("algorithm-demo_vector_add-numpy-add").relative_to(cli_root).as_posix() == "kernels/demo_vector_add/numpy-add/algorithm.json"


def test_register_algorithm_rejects_reserved_slug_blank_summary_and_unknown_kernel(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reserved = invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "trajectory", "--method-summary", "x", "--root", str(cli_root)
    ).error(2)
    assert reserved["error"] == "RESERVED_SLUG"
    blank = invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "blank", "--method-summary", "   ", "--root", str(cli_root)
    ).error(2)
    assert blank["error"] == "METHOD_SUMMARY_REQUIRED"
    missing = invoke(
        capsys, "register-algorithm", "--kernel-id", "no_such_kernel", "--algorithm-id", "x", "--method-summary", "y", "--root", str(cli_root)
    ).error(3)
    assert missing["error"] == "MISSING_KERNEL"


def test_register_config_requires_algorithm_when_several_exist_and_same_shape_is_comparable(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invoke(
        capsys, "register-algorithm", "--kernel-id", "demo_vector_add", "--algorithm-id", "numpy-add",
        "--method-summary", "numpy elementwise add", "--root", str(cli_root),
    ).result
    ambiguous = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--problem", '{"n": 16, "dtype": "f32"}', "--root", str(cli_root)
    ).error(2)
    assert ambiguous["error"] == "ALGORITHM_REQUIRED"
    under_new = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--algorithm", "numpy-add", "--problem", '{"n": 16, "dtype": "f32"}', "--root", str(cli_root)
    ).result
    assert under_new["created"] is True
    assert under_new["config"]["record_id"].startswith("cfg-numpy-add-demo-n16-f32-")
    assert under_new["algorithm_ref"] == "algorithm-demo_vector_add-numpy-add"
    store = MemoryStore.open(cli_root)
    same_hash = store.configs_by_hash(store.get("cfg-demo").payload.config_hash)
    assert {c.record_id for c in same_hash} == {"cfg-demo", under_new["config"]["record_id"]}, "same shape, one record per algorithm"
    assert under_new["config_hash"] == store.get("cfg-demo").payload.config_hash
    unknown = invoke(
        capsys, "register-config", "--kernel-id", "demo_vector_add", "--algorithm", "nope", "--problem", '{"n": 16, "dtype": "f32"}', "--root", str(cli_root)
    ).error(3)
    assert unknown["error"] == "MISSING_ALGORITHM"


def test_query_by_algorithm_bare_id_needs_kernel_and_lists_placeholder_shape(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(
        capsys, "query", "--kernel", "demo_vector_add", "--algorithm", "unspecified", "--record-type", "config", "--root", str(cli_root)
    ).result
    assert [item["record_ref"] for item in result["items"]] == ["cfg-demo"]
    assert result["items"][0]["algorithm_ref"] == PLACEHOLDER_ALGORITHM
    err = invoke(capsys, "query", "--algorithm", "unspecified", "--root", str(cli_root)).error(2)
    assert err["error"] == "ALGORITHM_REQUIRED"
    by_ref = invoke(capsys, "query", "--algorithm", PLACEHOLDER_ALGORITHM, "--record-type", "algorithm", "--root", str(cli_root)).result
    assert [item["record_ref"] for item in by_ref["items"]] == [PLACEHOLDER_ALGORITHM]
    assert by_ref["items"][0]["is_placeholder"] is True
    assert by_ref["items"][0]["method_summary"] == "unspecified (imported from v0.2)"


# --------------------------------------------------------------------------------------
# trajectory (T24), query (T05), export-context
# --------------------------------------------------------------------------------------
def test_t24_trajectory_rebuild_then_verify_matches(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unverified = invoke(capsys, "trajectory", "--config", "cfg-demo", "--verify", "--root", str(cli_root)).result
    assert unverified["stored"] is False and unverified["matches"] is False, "nothing is stored before a publish"
    rebuilt = invoke(capsys, "trajectory", "--config", "cfg-demo", "--rebuild", "--root", str(cli_root)).result
    assert rebuilt["config_ref"] == "cfg-demo" and rebuilt["publishable"] is True and rebuilt["forced"] is False
    assert Path(rebuilt["path"]).is_file() and Path(rebuilt["memory_records_path"]).is_file()
    codes = {d["code"] for d in rebuilt["diagnostics"]}
    assert "UNTESTED_COMMITS" in codes, "not_run is derived from absence and surfaced as a diagnostic"
    verified = invoke(capsys, "trajectory", "--config", "cfg-demo", "--verify", "--root", str(cli_root)).result
    assert verified["stored"] is True and verified["matches"] is True and verified["stored_view_consistent"] is True
    assert verified["stored_hash"] == verified["rebuilt_hash"] == rebuilt["view_hash"]
    # rebuilding again is deterministic
    again = invoke(capsys, "trajectory", "--config", "cfg-demo", "--rebuild", "--root", str(cli_root)).result
    assert again["view_hash"] == rebuilt["view_hash"]
    assert again["removed_views"], "rebuild removes the previous derived views first"


def test_t24_tampered_trajectory_view_fails_verification(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rebuilt = invoke(capsys, "trajectory", "--config", "cfg-demo", "--rebuild", "--root", str(cli_root)).result
    path = Path(rebuilt["path"])
    document = json.loads(path.read_text())
    document["view"]["tampered_by_test"] = True
    path.write_text(json.dumps(document))
    verified = invoke(capsys, "trajectory", "--config", "cfg-demo", "--verify", "--root", str(cli_root)).result
    assert verified["stored"] is True
    assert verified["stored_view_consistent"] is False
    assert verified["matches"] is False


def test_trajectory_unknown_config_is_missing_reference(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "trajectory", "--config", "cfg-nope", "--rebuild", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


def test_t24_kernel_trajectory_rebuild_verify_and_directory_deletion(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rebuilt = invoke(capsys, "trajectory", "--kernel", "demo_vector_add", "--rebuild", "--root", str(cli_root)).result
    assert rebuilt["kernel_id"] == "demo_vector_add" and rebuilt["publishable"] is True and rebuilt["forced"] is False
    view_path = Path(rebuilt["path"])
    assert view_path.relative_to(cli_root).as_posix() == "kernels/demo_vector_add/trajectory/trajectory.json"
    records_path = Path(rebuilt["memory_records_path"])
    assert records_path.relative_to(cli_root).as_posix() == "kernels/demo_vector_add/trajectory/memory_records.jsonl"
    assert rebuilt["record_count"] == FIXTURE_RECORD_COUNT
    assert [s["config_ref"] for s in rebuilt["shapes"]] == ["cfg-demo"]
    assert (cli_root / "kernels/demo_vector_add/trajectory/shapes/cfg-demo/trajectory.json").is_file()
    document = json.loads(view_path.read_text())
    assert document["view"]["view_version"] == "kernel-trajectory-v1"
    algorithms = document["view"]["algorithms"]
    assert [a["algorithm_ref"] for a in algorithms] == [PLACEHOLDER_ALGORITHM]
    assert algorithms[0]["is_placeholder"] is True and algorithms[0]["method_summary"] == "unspecified (imported from v0.2)"
    assert [s["config_ref"] for s in algorithms[0]["shapes"]] == ["cfg-demo"]
    verified = invoke(capsys, "trajectory", "--kernel", "demo_vector_add", "--verify", "--root", str(cli_root)).result
    assert verified["matches"] is True and verified["stored_hash"] == rebuilt["view_hash"]
    records_before = records_path.read_bytes()
    # T24: delete the whole generated tree and the cache; the rebuild must reproduce the identical view.
    import shutil

    shutil.rmtree(cli_root / "kernels/demo_vector_add/trajectory")
    shutil.rmtree(cli_root / ".cache", ignore_errors=True)
    assert invoke(capsys, "integrity", "--root", str(cli_root)).result["ok"] is True, "deleting views never touches records"
    again = invoke(capsys, "trajectory", "--kernel", "demo_vector_add", "--rebuild", "--root", str(cli_root)).result
    assert again["view_hash"] == rebuilt["view_hash"]
    assert records_path.read_bytes() == records_before
    assert invoke(capsys, "integrity", "--root", str(cli_root)).result["ok"] is True, "views under trajectory/ are never scanned as records"


def test_trajectory_requires_exactly_one_scope(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["trajectory", "--rebuild", "--root", str(cli_root), "--json"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        cli.main(["trajectory", "--kernel", "demo_vector_add", "--config", "cfg-demo", "--root", str(cli_root), "--json"])
    assert info.value.code == 2
    err = invoke(capsys, "trajectory", "--kernel", "no_such_kernel", "--rebuild", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


# --------------------------------------------------------------------------------------
# migrate-v02 (layout 1 -> layout 2 into a new root)
# --------------------------------------------------------------------------------------
def build_legacy_store(root: Path, records: list[dict]) -> None:
    """Hand-build a layout-1 (v0.2) store holding the given verbatim 0.2.0 records, journaled."""
    from kernel_memory.domain.hashing import jcs_digest

    def legacy_relpath(record: dict, by_id: dict[str, dict]) -> str:
        t, rid, p = record["record_type"], record["record_id"], record["payload"]
        cfg = "kernels/demo_vector_add/configs/cfg-demo"
        if t == "kernel":
            return "kernels/demo_vector_add/kernel.json"
        if t == "config":
            return f"{cfg}/config.json"
        if t == "pr":
            return f"{cfg}/attempt/{rid}/pr.json"
        if t == "pr_snapshot":
            return f"{cfg}/attempt/{p['pr_ref']}/snapshots/{rid}.json"
        if t == "commit":
            return f"{cfg}/attempt/{p['pr_ref']}/commits/{rid}/commit.json"
        if t == "baseline":
            return f"{cfg}/baselines/{rid}/baseline.json"
        if t == "run":
            subject = by_id[p["subject_ref"]]
            if subject["record_type"] == "baseline":
                return f"{cfg}/baselines/{subject['record_id']}/runs/{rid}/run.json"
            return f"{cfg}/attempt/{subject['payload']['pr_ref']}/commits/{subject['record_id']}/runs/{rid}/run.json"
        return f"{cfg}/{t}s/{rid}.json"

    by_id = {r["record_id"]: r for r in records}
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"store_version": "0.2.0", "schema_version": "0.2.0", "layout_version": 1, "hash_version": "jcs-sha256-v1", "store_id": "legacy-test", "created_at": "2026-09-08T00:00:00Z", "notes": "test"})
    )
    lines = []
    for record in records:
        rel = legacy_relpath(record, by_id)
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        lines.append(json.dumps({"record_id": record["record_id"], "record_type": record["record_type"], "relpath": rel, "digest": jcs_digest(record), "txn_id": "txn-legacy", "published_at": "2026-09-08T00:00:00Z"}))
    (root / "journal").mkdir()
    (root / "journal" / "records.jsonl").write_text("\n".join(lines) + "\n")
    for sub in ("artifacts", "requests", ".runtime"):
        (root / sub).mkdir()


def test_migrate_v02_dry_run_then_apply_into_new_root(tmp_path: Path, capsys: pytest.CaptureFixture[str], legacy_bundle_dicts: list[dict]) -> None:
    old = tmp_path / "old-store"
    # Runs are left out (their artifacts would have to be copied too), so records that cite runs
    # (the decision and the annotation) are left out as well: the subset must be reference-closed.
    subset = [r for r in legacy_bundle_dicts if r["record_type"] in ("kernel", "config", "baseline", "pr", "commit", "pr_snapshot", "relation")]
    build_legacy_store(old, subset)
    before = file_listing(old)
    refused = invoke(capsys, "status", "--root", str(old)).error(2)
    assert refused["error"] == "UNSUPPORTED_STORE" and "migrate-v02" in refused["message"]
    new = tmp_path / "new-store"
    dry = invoke(capsys, "migrate-v02", "--source", str(old), "--root", str(new)).result
    assert dry["dry_run"] is True and dry["records_read"] == len(subset)
    assert dry["records_synthesized"] == [PLACEHOLDER_ALGORITHM] and dry["records_linked"] == 1
    assert not new.exists(), "a dry run writes nothing"
    applied = invoke(capsys, "migrate-v02", "--source", str(old), "--root", str(new), "--apply").result
    assert applied["dry_run"] is False and applied["records_published"] == len(subset) + 1
    assert applied["provenance"] == "preserved"
    assert file_listing(old) == before, "the source store is never modified"
    status = invoke(capsys, "status", "--root", str(new)).result
    assert status["records"]["algorithm"] == 1 and status["records"]["config"] == 1 and status["records"]["kernel"] == 1
    assert status["kernels"]["demo_vector_add"]["algorithms"]["unspecified"]["is_placeholder"] is True
    assert invoke(capsys, "integrity", "--root", str(new)).result["ok"] is True
    deep = invoke(capsys, "validate", "--deep", "--root", str(new)).result
    assert deep["ok"] is True
    assert (new / "kernels/demo_vector_add/unspecified/cfg-demo/config.json").is_file()
    assert MemoryStore.open(new).get("cfg-demo").payload.algorithm_ref == PLACEHOLDER_ALGORITHM
    # idempotent re-apply into another fresh root yields identical record digests
    third = tmp_path / "third-store"
    invoke(capsys, "migrate-v02", "--source", str(old), "--root", str(third), "--apply").result
    digests = lambda root: sorted((e.record_id, e.digest) for e in MemoryStore.open(root).index_entries())  # noqa: E731
    assert digests(new) == digests(third)


def test_migrate_v02_refuses_non_legacy_and_non_empty_destination(cli_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path, legacy_bundle_dicts: list[dict]) -> None:
    not_legacy = invoke(capsys, "migrate-v02", "--source", str(cli_root), "--root", str(tmp_path / "x")).error(2)
    assert not_legacy["error"] == "NOT_A_LEGACY_STORE"
    missing = invoke(capsys, "migrate-v02", "--source", str(tmp_path / "absent"), "--root", str(tmp_path / "y")).error(2)
    assert missing["error"] == "NOT_A_STORE"
    old = tmp_path / "old-store"
    build_legacy_store(old, [r for r in legacy_bundle_dicts if r["record_type"] in ("kernel", "config")])
    occupied = invoke(capsys, "migrate-v02", "--source", str(old), "--root", str(cli_root), "--apply").error(2)
    assert occupied["error"] == "ROOT_NOT_EMPTY"


def test_t05_query_not_run_commits_are_derived_from_absence(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(
        capsys, "query", "--config", "cfg-demo", "--record-type", "commit", "--run-status", "not_run", "--root", str(cli_root)
    ).result
    items = result["items"]
    assert {item["record_ref"] for item in items} == UNTESTED_COMMITS
    assert all(item["status"] == "not_run" and item["run_refs"] == [] for item in items)
    assert result["total"] == 2
    assert store_ids(cli_root, "run") == ["run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure"], "no fake Run was stored"
    # negative control: the tested commits are the complement
    tested = invoke(capsys, "query", "--config", "cfg-demo", "--record-type", "commit", "--run-status", "tested", "--root", str(cli_root)).result
    assert {item["record_ref"] for item in tested["items"]} == TESTED_COMMITS
    assert all(item["status"] == "tested" and item["run_refs"] for item in tested["items"])


def test_query_by_component_finds_commit_with_that_change(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "query", "--component", "tiling", "--root", str(cli_root)).result
    assert [item["record_ref"] for item in result["items"]] == ["commit-demo-a"]
    (item,) = result["items"]
    assert item["record_type"] == "commit"
    assert [c["component"] for c in item["changes"]] == ["tiling"]
    assert item["changes"][0]["attribution"] == "group_only"
    none = invoke(capsys, "query", "--component", "no-such-component", "--root", str(cli_root)).result
    assert none["items"] == [] and none["total"] == 0


def test_export_context_lists_baseline_and_notice(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    context = invoke(capsys, "export-context", "--config", "cfg-demo", "--root", str(cli_root)).result
    assert context["current_baselines"][0]["baseline_ref"] == "baseline-demo"
    assert context["default_parent_ref"] == "baseline-demo"
    assert isinstance(context["notice"], str) and context["notice"]
    assert context["config"]["config_ref"] == "cfg-demo"
    assert context["best_known"] == [], "fixture runs never yield a best-known entry"
    assert context["confirmed_candidates"] == []
    assert {"cfg-demo", "baseline-demo", "kernel-demo"} <= set(context["record_refs_included"])
    blocked = context["blocked_or_rejected_decisions"]
    assert [d["decision_ref"] for d in blocked] == ["decision-demo-blocked"]
    assert "FIXTURE_NOT_ELIGIBLE" in blocked[0]["reason_codes"]
    # uncollected metrics stay null on the latest baseline run
    latest = context["current_baselines"][0]["latest_run"]
    assert all(m["value"] is None for m in latest["metrics"] if m["status"] != "observed")


def test_export_context_unknown_config_is_missing_reference(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "export-context", "--config", "cfg-nope", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


# --------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------
def test_compare_fixture_runs_is_comparable_with_derived_values(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "compare", "--candidate", "run-demo-a", "--baseline", "run-demo-baseline", "--root", str(cli_root)).result
    assert result["status"] == "COMPARABLE"
    derived = result["derived"]
    assert derived["recomputed_from_samples"] is True
    assert derived["baseline_median_us"] == 100.0 and derived["candidate_median_us"] == 90.0
    assert derived["speedup"] == pytest.approx(1.1111, abs=1e-4)
    assert derived["speedup"] == stats.speedup(100.0, 90.0)
    assert derived["latency_reduction_pct"] == 10.0 == stats.latency_reduction_pct(100.0, 90.0)
    assert result["speedup"] == derived["speedup"]
    assert result["confirmation_eligible"] is False
    assert result["confirmation_blockers"] == ["FIXTURE_NOT_ELIGIBLE"], "fixtures compare but never confirm"


def test_compare_missing_candidate_is_reference_error(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "compare", "--candidate", "nope", "--baseline", "run-demo-baseline", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"
    assert err["details"]["record_id"] == "nope"


def test_compare_wrong_record_type_is_reference_error(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "compare", "--candidate", "commit-demo-a", "--baseline", "run-demo-baseline", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


def test_compare_mock_runs_with_different_repetitions_is_not_comparable(cli_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    five = invoke(capsys, "cpu-demo-defaults", "--out", str(tmp_path / "p5"), "--repetitions", "5", "--warmup", "0").result
    seven = invoke(capsys, "cpu-demo-defaults", "--out", str(tmp_path / "p7"), "--repetitions", "7", "--warmup", "0").result
    assert json.loads(Path(five["protocol"]).read_text())["repetitions"] == 5
    assert json.loads(Path(seven["protocol"]).read_text())["repetitions"] == 7
    run_a = invoke(
        capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "req-p5", "--protocol", five["protocol"], "--root", str(cli_root)
    ).result["run"]
    run_b = invoke(
        capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "req-p7", "--protocol", seven["protocol"], "--root", str(cli_root)
    ).result["run"]
    assert run_a["execution_status"] == run_b["execution_status"] == "succeeded"
    assert run_a["timing"]["sample_count"] == 5 and run_b["timing"]["sample_count"] == 7
    assert run_a["comparison_key"] != run_b["comparison_key"]
    result = invoke(capsys, "compare", "--candidate", run_a["record_id"], "--baseline", run_b["record_id"], "--root", str(cli_root)).reported(4)
    assert result["status"] == "NOT_COMPARABLE"
    assert result["reasons"] == ["NOT_COMPARABLE(protocol)"]
    assert result["derived"] is None and result["speedup"] is None, "no derived numbers across protocols"
    assert [(d["group"], d["field"], d["candidate"], d["baseline"]) for d in result["differences"]] == [("protocol", "repetitions", 5, 7)]
    # negative control: the same protocol file twice is comparable
    run_c = invoke(
        capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "req-p5-again", "--protocol", five["protocol"], "--root", str(cli_root)
    ).result["run"]
    same = invoke(capsys, "compare", "--candidate", run_c["record_id"], "--baseline", run_a["record_id"], "--root", str(cli_root)).result
    assert same["status"] == "COMPARABLE" and same["derived"]["speedup"] == 1.0


# --------------------------------------------------------------------------------------
# decide (T19)
# --------------------------------------------------------------------------------------
def test_t19_decide_dry_run_blocks_fixture_candidate_without_writing(cli_root: Path, capsys: pytest.CaptureFixture[str], pairs_file: Path) -> None:
    result = invoke(capsys, "decide", "--candidate", "commit-demo-a", "--pairs", str(pairs_file), "--dry-run", "--root", str(cli_root)).reported(4)
    assert result["outcome"] == "blocked"
    assert {"FIXTURE_NOT_ELIGIBLE", "INSUFFICIENT_CONFIRMATION_PAIRS"} <= set(result["reason_codes"])
    assert result["is_production"] is False
    assert result["candidate_subject_ref"] == "commit-demo-a"
    assert result["candidate_run_refs"] == ["run-demo-a"] and result["baseline_run_refs"] == ["run-demo-baseline"]
    assert "decision" not in result
    assert store_ids(cli_root, "decision") == ["decision-demo-blocked"], "dry run appends no decision record"
    # the pair itself is comparable and the derived numbers are present; blocking is about eligibility, not arithmetic
    (pair,) = result["pair_evaluations"]
    assert pair["comparison"]["status"] == "COMPARABLE"
    assert pair["comparison"]["derived"]["speedup"] == pytest.approx(1.1111, abs=1e-4)


def test_t19_decide_without_dry_run_appends_blocked_decision(cli_root: Path, capsys: pytest.CaptureFixture[str], pairs_file: Path) -> None:
    result = invoke(capsys, "decide", "--candidate", "commit-demo-a", "--pairs", str(pairs_file), "--root", str(cli_root)).reported(4)
    assert result["outcome"] == "blocked"
    decision = result["decision"]
    assert decision["record_type"] == "decision" and decision["record_id"].startswith("decision-commit-demo-a-")
    store = MemoryStore.open(cli_root)
    record = store.get(decision["record_id"])
    assert record is not None and record.record_type == "decision"
    assert record.payload.outcome == "blocked" and record.payload.is_production is False
    assert "FIXTURE_NOT_ELIGIBLE" in record.payload.reason_codes
    assert len(store.records("decision")) == 2
    # the fixture decision already in the bundle is untouched (append-only)
    assert store.get("decision-demo-blocked") is not None


def test_decide_unknown_candidate_is_reference_error(cli_root: Path, capsys: pytest.CaptureFixture[str], pairs_file: Path) -> None:
    err = invoke(capsys, "decide", "--candidate", "commit-nope", "--pairs", str(pairs_file), "--dry-run", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


def test_decide_pairs_file_must_be_a_list(cli_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "bad_pairs.json"
    bad.write_text(json.dumps("not a list"))
    err = invoke(capsys, "decide", "--candidate", "commit-demo-a", "--pairs", str(bad), "--dry-run", "--root", str(cli_root)).error(2)
    assert err["error"] == "INPUT_ERROR"


# --------------------------------------------------------------------------------------
# run (T11, authorization, T29)
# --------------------------------------------------------------------------------------
def test_run_mock_backend_publishes_trusted_worker_run(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root)).result
    assert result["request"] == {"created": True, "request_id": "r1", "spec_hash": result["request"]["spec_hash"]}
    run = result["run"]
    assert run["record_id"] == "run-r1-a1"
    assert run["provenance"] == "trusted_worker"
    assert run["execution_status"] == "succeeded" and run["failure_reason"] is None
    assert run["correctness"]["status"] == "pass"
    assert run["timing"]["status"] == "recorded" and run["timing"]["sample_count"] == 100
    assert run["environment_backend"] == "mock"
    store = MemoryStore.open(cli_root)
    record = store.get("run-r1-a1")
    assert record is not None and record.payload.provenance == "trusted_worker"
    assert record.payload.timing.samples_artifact_ref in run["artifacts"]
    # metrics that were not collected stay null with a status
    assert all(m.value is None for m in record.payload.analysis_metrics if m.status != "observed")


def test_t11_replaying_a_finished_request_executes_nothing_new(cli_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root)]
    first = invoke(capsys, *argv).result
    ledger = RequestLedger(MemoryStore.open(cli_root))
    events_before = [e["kind"] for e in ledger.events("r1")]
    assert events_before == ["queued", "claimed", "running", "finished"]
    runs_before = store_ids(cli_root, "run")

    calls: list[str] = []
    for name in ("prepare", "compile", "verify", "benchmark"):
        original = getattr(MockAdapter, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(MockAdapter, name, spy)

    second = invoke(capsys, *argv).result
    assert second["request"]["created"] is False
    assert second["request"]["spec_hash"] == first["request"]["spec_hash"]
    assert second["run"]["record_id"] == first["run"]["record_id"] == "run-r1-a1"
    assert second["run"] == first["run"]
    assert calls == [], "a replay must not touch the adapter"
    assert [e["kind"] for e in RequestLedger(MemoryStore.open(cli_root)).events("r1")] == events_before
    assert store_ids(cli_root, "run") == runs_before
    # negative control: a different request id is a new execution
    third = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r2", "--root", str(cli_root)).result
    assert third["request"]["created"] is True and third["run"]["record_id"] == "run-r2-a1"
    assert calls == ["prepare", "compile", "verify", "benchmark"]


def test_t11_same_request_id_with_different_spec_conflicts(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root)).result
    err = invoke(
        capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r1", "--repetitions", "7", "--root", str(cli_root)
    ).error(3)
    assert err["error"] in {"IDEMPOTENCY_CONFLICT", "REQUEST_ID_CONFLICT"}
    assert store_ids(cli_root, "run").count("run-r1-a1") == 1


def test_run_records_session_pair_and_role(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(
        capsys,
        "run",
        "--subject",
        "baseline-demo",
        "--backend",
        "mock",
        "--request-id",
        "r-pair",
        "--session-id",
        "sess-1",
        "--pair-id",
        "pair-1",
        "--role",
        "candidate",
        "--root",
        str(cli_root),
    ).result
    record = MemoryStore.open(cli_root).get(result["run"]["record_id"])
    assert record.payload.session_id == "sess-1"
    assert record.payload.pair_id == "pair-1"
    assert record.payload.role_in_pair == "candidate"


def test_run_generates_request_id_when_not_given(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--root", str(cli_root)).result
    assert result["request"]["request_id"].startswith("request-")
    assert result["run"]["record_id"] == f"run-{result['request']['request_id']}-a1"


def test_run_unknown_subject_is_reference_error(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "run", "--subject", "baseline-nope", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root)).error(3)
    assert err["error"] == "MISSING_REFERENCE"


def test_run_commit_subject_requires_entrypoint(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "run", "--subject", "commit-demo-a", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root)).error(2)
    assert err["error"] == "INPUT_ERROR" and "--entrypoint" in err["message"]


def test_run_unknown_backend_is_denied_before_registry_lookup(cli_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    lookups: list[str] = []
    original = AdapterRegistry.kernel_adapter

    def spy(self, backend):
        lookups.append(backend)
        return original(self, backend)

    monkeypatch.setattr(AdapterRegistry, "kernel_adapter", spy)
    err = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "nope", "--request-id", "r-nope", "--root", str(cli_root)).error(7)
    assert err["error"] == "BACKEND_EXECUTION_NOT_AUTHORIZED"
    assert err["details"]["backend"] == "nope"
    assert err["details"]["required"] == ["permissions.allow_nope_execution"]
    assert lookups == [], "authorization is decided before the adapter registry is consulted"
    assert "run-r-nope-a1" not in store_ids(cli_root, "run")
    events = [e["kind"] for e in RequestLedger(MemoryStore.open(cli_root)).events("r-nope")]
    assert events == ["queued", "cancelled"], "the denial is recorded, never silently skipped"


def test_run_jax_tpu_default_permissions_denied_before_any_device_probe(
    cli_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kernel_memory.adapters.jax_tpu import JaxAdapter

    probes: list[str] = []
    monkeypatch.setattr(JaxAdapter, "check_environment", lambda self: probes.append("probe"))
    monkeypatch.setattr(JaxAdapter, "_load_jax", lambda self: probes.append("import"), raising=False)
    err = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "jax_tpu", "--request-id", "r-tpu", "--root", str(cli_root)).error(7)
    assert err["error"] == "TPU_EXECUTION_NOT_AUTHORIZED"
    assert err["details"]["request_authorization"] is False and err["details"]["runner_permission"] is False
    assert probes == [], "no device probe may happen before authorization"
    assert "run-r-tpu-a1" not in store_ids(cli_root, "run")
    assert [e["kind"] for e in RequestLedger(MemoryStore.open(cli_root)).events("r-tpu")] == ["queued", "cancelled"]


@pytest.mark.integration
def test_t29_run_jax_tpu_on_cpu_host_reports_backend_unavailable(cli_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed; the real jax import behind --backend jax_tpu cannot be exercised")
    import jax

    if jax.default_backend() == "tpu":
        pytest.skip("a TPU is present on this host; the CPU-only unavailability path cannot be exercised")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"settings_version": "0.2.0", "memory_root": str(cli_root), "permissions": {"allow_tpu_execution": True}})
    )
    runs_before = store_ids(cli_root, "run")
    result = invoke(
        capsys, "run", "--subject", "baseline-demo", "--backend", "jax_tpu", "--request-id", "r-tpu", "--root", str(cli_root), "--settings", str(settings)
    ).reported(5)
    assert result["status"] == "unexecuted"
    assert result["backend"] == "jax_tpu"
    assert result["error"] == "BACKEND_UNAVAILABLE" and result["exit_code"] == 5
    assert result["details"]["required"] == "tpu"
    assert result["details"]["actual_backend"] == "cpu"
    assert result["request_id"] == "r-tpu"
    assert "no Run was recorded" in result["note"]
    assert store_ids(cli_root, "run") == runs_before, "an unavailable backend is not an execution"
    events = [e["kind"] for e in RequestLedger(MemoryStore.open(cli_root)).events("r-tpu")]
    assert events == ["queued", "backend_unavailable"]


def test_run_settings_file_with_credential_value_is_rejected(cli_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"settings_version": "0.2.0", "github_token": "ghp_not_allowed_here"}))
    err = invoke(capsys, "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "r1", "--root", str(cli_root), "--settings", str(settings)).error(2)
    assert err["error"] == "INPUT_ERROR"


# --------------------------------------------------------------------------------------
# collect-pr
# --------------------------------------------------------------------------------------
def test_collect_pr_without_offline_fixture_needs_network_authorization(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "collect-pr", "--repo", "org/repo", "--number", "1", "--config", "cfg-demo", "--root", str(cli_root)).error(5)
    assert err["error"] == "NETWORK_NOT_AUTHORIZED"
    assert err["details"]["missing"] == ["permissions.allow_network", "GITHUB_TOKEN"]
    assert len(store_ids(cli_root)) == FIXTURE_RECORD_COUNT


def test_collect_pr_repo_must_be_owner_slash_repo(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "collect-pr", "--repo", "no-slash", "--number", "1", "--config", "cfg-demo", "--root", str(cli_root)).error(2)
    assert err["error"] == "INPUT_ERROR"


# --------------------------------------------------------------------------------------
# migrate-v01, cpu-demo-defaults, optimize
# --------------------------------------------------------------------------------------
def test_migrate_v01_dry_run_writes_nothing(cli_root: Path, capsys: pytest.CaptureFixture[str], v01_dir: Path) -> None:
    before = file_listing(cli_root)
    result = invoke(
        capsys, "migrate-v01", str(v01_dir), "--root", str(cli_root), "--repo-uid-map", '{"org/repo": "github:github.com:repo:42"}'
    ).result
    assert result["dry_run"] is True
    assert result["publish_outcome"] is None
    # The dry run opens the existing store, so the fixture's kernel (kernel-demo) and config (cfg-demo, same
    # config_hash) are reused by identity (T01) instead of minted again: only the PR and commit are new records.
    assert result["record_counts"] == {"commit": 1, "pr": 1}
    # A v0.1 kernel maps to two v0.2/v0.3 records: the kernel and its placeholder algorithm (reused here).
    mapped = {(m["v01_kind"], m["v01_id"], m["v02_record_type"]): m for m in result["identity_map"]}
    assert mapped[("kernel", "demo_vector_add", "kernel")]["v02_record_id"] == "kernel-demo"
    assert mapped[("kernel", "demo_vector_add", "kernel")]["status"] == "mapped"
    assert mapped[("kernel", "demo_vector_add", "algorithm")]["v02_record_id"] == PLACEHOLDER_ALGORITHM
    mapped = {(kind, v01_id): m for (kind, v01_id, v02_type), m in mapped.items() if v02_type != "algorithm"}
    assert mapped[("config", "cfg-1")]["v02_record_id"] == "cfg-demo"
    assert mapped[("attempt", "att-1")]["v02_record_id"] == "pr-gh-42-pr-7"
    assert mapped[("revision", "rev-1")]["v02_record_id"] == "commit-gh-42-pr-7-a1b2c3d4e5f6"
    assert any("nothing was written" in note for note in result["notes"])
    assert file_listing(cli_root) == before, "dry run must not touch the store"
    assert len(store_ids(cli_root)) == FIXTURE_RECORD_COUNT


def test_migrate_v01_dry_run_does_not_need_a_store(tmp_path: Path, capsys: pytest.CaptureFixture[str], v01_dir: Path) -> None:
    result = invoke(capsys, "migrate-v01", str(v01_dir), "--root", str(tmp_path / "absent-store")).result
    assert result["dry_run"] is True
    assert not (tmp_path / "absent-store").exists()


def test_migrate_v01_rejects_bad_resolver_spec(cli_root: Path, capsys: pytest.CaptureFixture[str], v01_dir: Path) -> None:
    err = invoke(capsys, "migrate-v01", str(v01_dir), "--root", str(cli_root), "--resolver", "bogus:thing").error(2)
    assert err["error"] == "INPUT_ERROR"


def test_cpu_demo_defaults_writes_files_and_prints_source_commit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from kernel_memory.adapters.cpu_demo import current_source_commit

    out = tmp_path / "defaults"
    result = invoke(capsys, "cpu-demo-defaults", "--out", str(out), "--repetitions", "9", "--warmup", "2").result
    protocol = json.loads(Path(result["protocol"]).read_text())
    verifier = json.loads(Path(result["verifier"]).read_text())
    assert Path(result["protocol"]) == out / "cpu_protocol.json" and Path(result["verifier"]) == out / "cpu_verifier.json"
    assert protocol["repetitions"] == 9 and protocol["warmup"] == 2
    assert protocol["statistic"] == "median" and protocol["quantile_method"] == "linear"
    assert verifier["nonfinite_policy"] == "reject_unexpected" and verifier["tolerances"]
    oid = current_source_commit()
    assert result["source_commit"] == {"algorithm": oid.algorithm, "hex": oid.hex}
    assert result["source_commit"]["algorithm"] == "sha256" and len(result["source_commit"]["hex"]) == 64
    assert "not a Git commit" in result["note"]


def test_cpu_demo_defaults_rejects_zero_repetitions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(capsys, "cpu-demo-defaults", "--out", str(tmp_path / "defaults"), "--repetitions", "0").error(2)
    assert err["error"] == "INVALID_PROTOCOL"
    assert not (tmp_path / "defaults" / "cpu_protocol.json").exists()


def test_optimize_dry_run_with_mock_planner(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = store_ids(cli_root)
    result = invoke(
        capsys,
        "optimize",
        "--config",
        "cfg-demo",
        "--planner",
        "mock",
        "--backend",
        "mock",
        "--subject",
        "baseline-demo",
        "--dry-run",
        "--root",
        str(cli_root),
    ).result
    assert result["status"] == "dry_run" and result["dry_run"] is True
    assert result["note"] == MOCK_PLANNER_NOTE
    assert result["planner_id"] == "mock-v1"
    assert result["persisted_state_path"] is None
    assert result["rounds"], "the mock planner proposes at least one candidate"
    for round_report in result["rounds"]:
        proposal = round_report["proposal"]
        assert proposal["planner_id"] == "mock-v1" and proposal["parent_ref"] == "baseline-demo"
        assert proposal["implementation_overrides"] and proposal["file_allowlist"] == []
        assert proposal["predicted_speedup"] is None, "no invented numbers"
        assert round_report["run_ref"] is None and round_report["request_id"] is None
        assert round_report["improved"] is False
    assert store_ids(cli_root) == before, "a dry run publishes nothing"


def test_optimize_unknown_subject_is_reference_error(cli_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = invoke(
        capsys, "optimize", "--config", "cfg-demo", "--planner", "mock", "--backend", "mock", "--subject", "baseline-nope", "--dry-run", "--root", str(cli_root)
    ).error(3)
    assert err["error"] == "MISSING_REFERENCE"
