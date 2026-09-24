"""PR collection over offline GitHub fixtures (specification section 8; T05, T07, T08, T09, T10).

``collect_pr`` is driven through ``FixtureTransport`` with the canned responses in
``tests/fixtures/github``. Repository 900001 is the demo bundle's repository, so the demo
config ``cfg-demo`` owns every PR context created here. The collector never executes
anything: the single Run used by T05/T07 is produced by ``LocalRunner`` with a stub adapter.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kernel_memory.adapters.base import (
    AdapterRegistry,
    ArtifactBlob,
    CompileReport,
    CorrectnessReport,
    PreparedExecution,
    RunRequestSpec,
    SourceSnapshot,
    SourceSpec,
    TimingReport,
)
from kernel_memory.adapters.git_local import LocalGitRepo
from kernel_memory.adapters.github import FixtureTransport, GitHubClient, Response, fixture_response
from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, MissingReferenceError
from kernel_memory.domain.jsonio import load_json_file
from kernel_memory.domain.models import GitOid, PrSnapshotPayload, Record
from kernel_memory.execution.runner import LocalRunner
from kernel_memory.services.collect import COLLECTION_POLICY, CollectReport, collect_pr, ingest_pr_snapshot
from kernel_memory.services.common import new_record, runs_for_subject, snapshots_for_pr
from kernel_memory.services.validation import validate_records
from kernel_memory.storage import MemoryStore

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "github"
OWNER, REPO = "acme", "kernels"
REPO_PATH = f"/repos/{OWNER}/{REPO}"
REPO_UID = "github:github.com:repo:900001"
CONFIG_REF = "cfg-demo"
PR_KEY_101, PR_KEY_102, PR_KEY_103 = "gh-900001-pr-101", "gh-900001-pr-102", "gh-900001-pr-103"
PR_REF_101, PR_REF_102 = f"pr-{PR_KEY_101}", f"pr-{PR_KEY_102}"

BASE_SHA = "ba5e" * 10
C1, C2, C3 = "c1" * 20, "c2" * 20, "c3" * 20
MERGE_101 = "3e" * 20
D2 = "d2" * 20
E1 = "e1" * 20
A1, A2, A3 = "a1" * 20, "a2" * 20, "a3" * 20

DEMO_RUN_COUNT = 4  # runs in the fixture bundle


# --------------------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------------------
def canned_envelope(name: str) -> dict[str, Any]:
    return load_json_file(FIXTURE_DIR / f"{name}.json")


def canned(name: str) -> Response:
    data = canned_envelope(name)
    return fixture_response(int(data["status"]), data.get("body"), dict(data.get("headers") or {}))


def canned_body(name: str) -> Any:
    return json.loads(json.dumps(canned_envelope(name)["body"]))


def binding_id(pr_key: str, sha: str) -> str:
    return f"commit-{pr_key}-{sha[:12]}"


def snapshot_id(pr_key: str, seq: int) -> str:
    return f"snapshot-{pr_key}-{seq:04d}"


BINDINGS_101 = [binding_id(PR_KEY_101, sha) for sha in (C1, C2, C3)]


def pr101_routes() -> dict[str, list[Response]]:
    commits = f"{REPO_PATH}/pulls/101/commits"
    return {
        REPO_PATH: [canned("repository_900001")],
        f"{REPO_PATH}/pulls/101": [canned("pull_101")],
        f"{commits}?per_page=1": [canned("pull_101_commits_page_1")],
        f"{commits}?per_page=1&page=2": [canned("pull_101_commits_page_2")],
        f"{commits}?per_page=1&page=3": [canned("pull_101_commits_page_3")],
    }


def pr101_force_push_routes() -> dict[str, list[Response]]:
    return {
        REPO_PATH: [canned("repository_900001")],
        f"{REPO_PATH}/pulls/101": [canned("pull_101_after_force_push")],
        f"{REPO_PATH}/pulls/101/commits": [canned("pull_101_after_force_push_commits")],
    }


def pr102_routes(*, merged: bool = True) -> dict[str, list[Response]]:
    body = canned_body("pull_102")
    body["merged"] = merged
    return {
        REPO_PATH: [canned("repository_900001")],
        f"{REPO_PATH}/pulls/102": [fixture_response(200, body)],
        f"{REPO_PATH}/pulls/102/commits": [canned("pull_102_commits")],
    }


def pr103_routes() -> dict[str, list[Response]]:
    return {
        REPO_PATH: [canned("repository_900001")],
        f"{REPO_PATH}/pulls/103": [canned("pull_103_large")],
        f"{REPO_PATH}/pulls/103/commits": [canned("pull_103_commits")],
    }


def dynamic_pr_routes(number: int, *, head: str, base: str, commits_count: int, commits: list[dict[str, Any]], **pull_fields: Any) -> dict[str, list[Response]]:
    body = canned_body("pull_101")
    body["number"] = number
    body["head"]["sha"], body["base"]["sha"], body["commits"] = head, base, commits_count
    body.update(pull_fields)
    return {
        REPO_PATH: [canned("repository_900001")],
        f"{REPO_PATH}/pulls/{number}": [fixture_response(200, body)],
        f"{REPO_PATH}/pulls/{number}/commits": [fixture_response(200, commits)],
    }


def api_commit(sha: str, parent: str, message: str) -> dict[str, Any]:
    return {"sha": sha, "parents": [{"sha": parent}], "commit": {"message": message, "author": {"date": "2026-09-09T10:00:00Z"}, "tree": {"sha": "7e" + sha[2:]}}}


def collect(store: MemoryStore, routes: dict[str, list[Response]], number: int, **kwargs: Any) -> tuple[CollectReport, FixtureTransport]:
    transport = FixtureTransport(routes)
    client = GitHubClient(transport, per_page=1)
    report = collect_pr(store, client, owner=OWNER, repo=REPO, number=number, config_ref=CONFIG_REF, **kwargs)
    return report, transport


# --------------------------------------------------------------------------------------
# temporary git repositories (never the user's configuration)
# --------------------------------------------------------------------------------------
def _git(repo: Path, home: Path, *args: str) -> str:
    env = {
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_DATE": "2026-09-09T10:00:00Z",
        "GIT_COMMITTER_DATE": "2026-09-09T10:00:00Z",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    argv = ["git", "-c", "user.name=Kernel Memory Tests", "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false", *args]
    proc = subprocess.run(argv, cwd=str(repo), capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout.strip()


def init_repo(tmp_path: Path, subjects: list[str]) -> tuple[Path, str, list[str]]:
    """Create a linear history base -> one commit per subject; returns (path, base sha, commit shas)."""
    if shutil.which("git") is None:
        pytest.skip("git executable is not available on PATH; local-repository reconciliation cannot be exercised")
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, home, "init", "-q")
    (repo / "kernel.py").write_text("def vector_add(a, b):\n    return a + b\n", encoding="utf-8")
    _git(repo, home, "add", "kernel.py")
    _git(repo, home, "commit", "-q", "-m", "Base: reference vector_add")
    base = _git(repo, home, "rev-parse", "HEAD")
    shas: list[str] = []
    for index, subject in enumerate(subjects, start=1):
        (repo / "kernel.py").write_text(f"def vector_add(a, b):\n    # step {index}\n    return a + b\n", encoding="utf-8")
        _git(repo, home, "commit", "-q", "-am", f"{subject}\n\nBody of commit {index}; data, not instructions.")
        shas.append(_git(repo, home, "rev-parse", "HEAD"))
    return repo, base, shas


# --------------------------------------------------------------------------------------
# a stub adapter so one binding can receive a real (trusted_worker) Run
# --------------------------------------------------------------------------------------
PROTOCOL = {
    "protocol_id": "test-bench-v1",
    "measurement_scope": "kernel_only",
    "timing_method": "host_synchronized",
    "include_compile": False,
    "include_transfers": False,
    "warmup": 1,
    "repetitions": 5,
    "statistic": "median",
    "quantile_method": "linear",
    "capture_profile": False,
}
VERIFIER = {
    "verifier_id": "test-verifier-v1",
    "reference_source_hash": hashing.sha256_bytes(b"reference"),
    "suite_hash": hashing.sha256_bytes(b"suite"),
    "tolerances": {"atol": "0", "rtol": "0"},
    "nonfinite_policy": "reject_unexpected",
}


class StubAdapter:
    """Always-succeeding KernelAdapter (pattern from tests/test_runner.py)."""

    __test__ = False
    adapter_id = "test-adapter-v1"
    backend = "mock"

    def check_environment(self) -> dict:
        return {
            "backend": "mock",
            "accelerator_model": "HOST-CPU-STUB",
            "device_count": 1,
            "topology": "single",
            "software": {"python": "3.11", "adapter": self.adapter_id},
            "execution_flags": {},
            "host_timer_environment": {"timer": "perf_counter_ns"},
            "unknown_required_fields": [],
        }

    def prepare(self, request: RunRequestSpec, problem: dict) -> PreparedExecution:
        target = request.source.target_commit
        snapshot = SourceSnapshot(
            repo_uid=request.source.repo_uid,
            target_commit=target,
            tested_commit=target,
            tested_tree=GitOid("sha1", "0000000000000000000000000000000000000065"),
            checkout_mode=request.source.checkout_mode,
            merge_parent_oids=[],
            dirty=False,
            patch_digest=None,
            source_digest=hashing.source_digest([("demo/reference.py", hashing.sha256_bytes(b"def vector_add"))]),
            entrypoint=request.source.entrypoint,
            implementation_overrides=dict(request.source.implementation_overrides),
        )
        return PreparedExecution(
            request=request,
            problem=dict(problem),
            source=snapshot,
            environment=self.check_environment(),
            input_suite_hash=hashing.jcs_digest({"problem": problem, "seed": 1}),
            handle={},
        )

    def compile(self, prepared: PreparedExecution) -> CompileReport:
        return CompileReport("ok", None, [])

    def verify(self, prepared: PreparedExecution) -> CorrectnessReport:
        report = {"status": "pass", "cases_total": 1, "cases_passed": 1}
        blob = ArtifactBlob("correctness", "correctness_report", "application/json", json.dumps(report).encode("utf-8"))
        return CorrectnessReport(status="pass", cases_total=1, cases_passed=1, max_abs_error=0.0, max_rel_error=0.0, artifacts=[blob])

    def benchmark(self, prepared: PreparedExecution) -> TimingReport:
        return TimingReport("recorded", "microseconds", [10.0, 11.0, 10.0, 9.0, 12.0], "host_synchronized")


def publish_run_for(store: MemoryStore, binding_ref: str, request_id: str) -> Record:
    binding = store.require(binding_ref, "commit")
    registry = AdapterRegistry()
    registry.register_kernel_adapter(StubAdapter())
    runner = LocalRunner(store, adapters=registry)
    spec = RunRequestSpec(
        request_id=request_id,
        idempotency_key=f"key-{request_id}",
        subject_ref=binding_ref,
        config_ref=CONFIG_REF,
        backend="mock",
        stage="benchmark",
        protocol=dict(PROTOCOL),
        verifier=dict(VERIFIER),
        source=SourceSpec(repo_uid=REPO_UID, target_commit=binding.payload.commit_oid, entrypoint="demo.reference:vector_add"),
        session_id="session-collect",
        pair_id=f"pair-{request_id}",
        role_in_pair="candidate",
    )
    outcome = runner.ledger.submit(spec)
    return runner.execute(outcome.request_id)


# --------------------------------------------------------------------------------------
# T05: collect three commits; execute one; the others remain untested
# --------------------------------------------------------------------------------------
def test_t05_collect_three_paginated_commits_creates_bindings_snapshot_and_all_untested(demo_store: MemoryStore) -> None:
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT
    report, transport = collect(demo_store, pr101_routes(), 101)

    assert (report.pr_ref, report.repo_uid, report.pr_key) == (PR_REF_101, REPO_UID, PR_KEY_101)
    pr = demo_store.require(PR_REF_101, "pr")
    assert pr.payload.title == canned_body("pull_101")["title"]  # stored as data
    assert (pr.payload.provider, pr.payload.number, pr.payload.config_ref, pr.payload.hypothesis) == ("github", 101, CONFIG_REF, None)
    assert pr.payload.repo_uid == REPO_UID

    assert report.commits_new == BINDINGS_101 and report.commits_existing == [] and report.commits_total == 3
    expected_parents = {C1: BASE_SHA, C2: C1, C3: C2}
    for sha, ref in zip((C1, C2, C3), BINDINGS_101):
        record = demo_store.require(ref, "commit")
        p = record.payload
        assert p.pr_ref == PR_REF_101 and p.repo_uid == REPO_UID
        assert p.commit_oid == GitOid("sha1", sha)
        assert [o.hex for o in p.git_parent_oids] == [expected_parents[sha]]
        assert p.diff_base_oid is not None and p.diff_base_oid.hex == expected_parents[sha]
        assert p.change_status == "not_extracted" and p.changes == []
        assert p.summary_author == "collector" and "\n" not in p.summary and p.summary
        assert p.source_available is False and p.diff_artifact_ref is None
    assert demo_store.require(BINDINGS_101[0], "commit").payload.summary == "Introduce tiled loop skeleton"

    assert report.snapshot_ref == snapshot_id(PR_KEY_101, 1) and report.snapshot_appended is True
    assert report.previous_snapshot_ref is None
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    s = snapshot.payload
    assert s.pr_ref == PR_REF_101 and s.previous_snapshot_ref is None
    assert s.commit_refs == BINDINGS_101
    assert s.enumeration_status == "complete_for_snapshot" and s.reason is None
    assert s.observed_head.hex == C3 and s.observed_base.hex == BASE_SHA and s.github_state == "open"

    assert report.coverage == "complete_for_snapshot" and report.coverage_reason is None
    assert report.untested_commit_refs == BINDINGS_101
    assert (report.head_sha, report.base_sha, report.github_state) == (C3, BASE_SHA, "open")
    assert report.force_push_detected is False and report.disappeared_commit_refs == []
    assert report.policy == COLLECTION_POLICY == "record_all, schedule_by_policy"
    as_dict = report.to_dict()
    assert as_dict["observed_head"] == C3 and as_dict["untested_commit_refs"] == BINDINGS_101
    assert not {"execution_status", "correctness", "timing", "run_refs"} & set(as_dict)

    # Nothing was executed: no Run appeared, and every request was a read.
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT
    assert all(runs_for_subject(demo_store, ref) == [] for ref in BINDINGS_101)
    assert {r["method"] for r in transport.requests} == {"GET"}


def test_t05_after_executing_one_binding_the_other_two_remain_untested(demo_store: MemoryStore) -> None:
    first, _ = collect(demo_store, pr101_routes(), 101)
    digests_before = {ref: demo_store.require(ref).canonical_digest() for ref in BINDINGS_101}

    run = publish_run_for(demo_store, BINDINGS_101[0], "request-collect-c1")
    assert run.payload.subject_ref == BINDINGS_101[0] and run.payload.provenance == "trusted_worker"
    assert run.payload.source.target_commit.hex == C1 and run.payload.execution_status == "succeeded"
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT + 1

    second, _ = collect(demo_store, pr101_routes(), 101)
    assert second.untested_commit_refs == BINDINGS_101[1:]
    assert second.commits_existing == BINDINGS_101 and second.commits_new == []
    assert second.snapshot_appended is False and second.snapshot_ref == first.snapshot_ref == snapshot_id(PR_KEY_101, 1)
    assert len(snapshots_for_pr(demo_store, PR_REF_101)) == 1
    # The collector attached nothing to the bindings: records are immutable and unchanged, and
    # the untested bindings still have no Run (a result is never propagated to a sibling commit).
    assert {ref: demo_store.require(ref).canonical_digest() for ref in BINDINGS_101} == digests_before
    assert [r.record_id for r in runs_for_subject(demo_store, BINDINGS_101[0])] == [run.record_id]
    assert all(runs_for_subject(demo_store, ref) == [] for ref in BINDINGS_101[1:])
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT + 1
    assert any("2 of 3 bindings have no run" in note for note in second.notes)


# --------------------------------------------------------------------------------------
# T08: the integration merge commit is reported, never treated as the head
# --------------------------------------------------------------------------------------
def test_t08_merge_commit_sha_is_reported_but_observed_head_is_the_pr_head(demo_store: MemoryStore) -> None:
    report, _ = collect(demo_store, pr101_routes(), 101)
    assert report.merge_commit_sha == MERGE_101 and report.head_sha == C3 and MERGE_101 != C3
    assert report.to_dict()["observed_head"] == C3 and report.to_dict()["merge_commit_sha"] == MERGE_101
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert snapshot.payload.observed_head.hex == C3
    assert demo_store.get(binding_id(PR_KEY_101, MERGE_101)) is None
    assert not any(c.payload.commit_oid.hex == MERGE_101 for c in demo_store.records("commit"))
    assert any("merge_commit_sha differs from head_sha" in note and "T08" in note for note in report.notes)


def test_t08_no_merge_note_when_merge_sha_is_absent_or_equals_head(demo_store: MemoryStore) -> None:
    routes = pr101_routes()
    body = canned_body("pull_101")
    body["merge_commit_sha"] = None
    routes[f"{REPO_PATH}/pulls/101"] = [fixture_response(200, body)]
    report, _ = collect(demo_store, routes, 101)
    assert report.merge_commit_sha is None
    assert not any("merge_commit_sha differs" in note for note in report.notes)


def test_second_identical_collect_appends_no_snapshot(demo_store: MemoryStore) -> None:
    first, _ = collect(demo_store, pr101_routes(), 101)
    second, _ = collect(demo_store, pr101_routes(), 101)
    assert first.snapshot_appended is True and second.snapshot_appended is False
    assert second.snapshot_ref == first.snapshot_ref == snapshot_id(PR_KEY_101, 1)
    assert second.previous_snapshot_ref is None
    assert [s.record_id for s in snapshots_for_pr(demo_store, PR_REF_101)] == [snapshot_id(PR_KEY_101, 1)]
    assert any("no snapshot appended" in note for note in second.notes)
    assert second.force_push_detected is False and second.disappeared_commit_refs == []


# --------------------------------------------------------------------------------------
# T09: more than 250 commits, shallow history, incomplete pagination
# --------------------------------------------------------------------------------------
def test_t09_pull_reporting_300_commits_without_local_repo_stays_partial(demo_store: MemoryStore) -> None:
    report, _ = collect(demo_store, pr103_routes(), 103)
    assert report.coverage == "partial"
    assert report.coverage_reason is not None and "250" in report.coverage_reason and "300" in report.coverage_reason
    assert report.commits_total == 3 and len(report.untested_commit_refs) == 3
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert snapshot.payload.enumeration_status == "partial" and snapshot.payload.reason == report.coverage_reason
    assert any("no local repository" in note for note in report.notes)
    assert report.snapshot_ref == snapshot_id(PR_KEY_103, 1)


def test_t09_partial_enumeration_is_reconciled_with_a_complete_local_git_graph(demo_store: MemoryStore, tmp_path: Path) -> None:
    repo, base, (k1, k2, k3) = init_repo(tmp_path, ["Step one", "Step two (missing from the API)", "Step three"])
    # The API "forgets" k2 and claims 300 commits; the pinned base/head OIDs are in the local graph.
    routes = dynamic_pr_routes(101, head=k3, base=base, commits_count=300, commits=[api_commit(k1, base, "Step one"), api_commit(k3, k2, "Step three")])
    report, _ = collect(demo_store, routes, 101, local_repo=LocalGitRepo(repo))

    assert report.coverage == "complete_for_snapshot"
    assert report.coverage_reason is not None and report.coverage_reason.startswith("reconciled with local git graph")
    expected = [binding_id(PR_KEY_101, sha) for sha in (k1, k2, k3)]
    assert report.commits_new == expected and report.commits_total == 3
    assert any("reconciled 3 commits with the local git graph" in note for note in report.notes)
    k2_binding = demo_store.require(expected[1], "commit")
    assert k2_binding.payload.summary == "Step two (missing from the API)" and k2_binding.payload.summary_author == "collector"
    assert [o.hex for o in k2_binding.payload.git_parent_oids] == [k1]
    for ref in expected:
        p = demo_store.require(ref, "commit").payload
        assert p.source_available is True and p.change_status == "not_extracted"
        assert p.diff_artifact_ref is not None
        artifact = demo_store.get_artifact_ref(p.diff_artifact_ref)
        assert artifact is not None and artifact.kind == "diff" and demo_store.has_artifact(artifact.sha256)
        assert demo_store.read_artifact(artifact.sha256).startswith(b"diff --git")
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert snapshot.payload.enumeration_status == "complete_for_snapshot" and snapshot.payload.commit_refs == expected
    assert snapshot.payload.observed_head.hex == k3 and snapshot.payload.observed_base.hex == base
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT


def test_t09_shallow_local_repo_cannot_reconcile_and_stays_partial(demo_store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, base, (k1, k2, k3) = init_repo(tmp_path, ["Step one", "Step two", "Step three"])
    monkeypatch.setattr(LocalGitRepo, "is_shallow", lambda self: True)
    routes = dynamic_pr_routes(101, head=k3, base=base, commits_count=300, commits=[api_commit(k1, base, "Step one"), api_commit(k3, k2, "Step three")])
    report, _ = collect(demo_store, routes, 101, local_repo=LocalGitRepo(repo))

    assert report.coverage == "partial"
    assert report.coverage_reason is not None and "shallow" in report.coverage_reason and "250" in report.coverage_reason
    assert any("shallow" in note for note in report.notes)
    assert report.commits_total == 2 and demo_store.get(binding_id(PR_KEY_101, k2)) is None
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert snapshot.payload.enumeration_status == "partial" and snapshot.payload.reason == report.coverage_reason


def test_t09_local_repo_lacking_the_head_object_cannot_reconcile(demo_store: MemoryStore, tmp_path: Path) -> None:
    repo, base, (k1, _k2, _k3) = init_repo(tmp_path, ["Step one", "Step two", "Step three"])
    routes = dynamic_pr_routes(101, head=C3, base=base, commits_count=300, commits=[api_commit(k1, base, "Step one"), api_commit(C3, k1, "Unknown head")])
    report, _ = collect(demo_store, routes, 101, local_repo=LocalGitRepo(repo))
    assert report.coverage == "partial"
    assert report.coverage_reason is not None and "lacks the base or head object" in report.coverage_reason
    assert demo_store.require(binding_id(PR_KEY_101, k1), "commit").payload.source_available is True
    assert demo_store.require(binding_id(PR_KEY_101, C3), "commit").payload.source_available is False


def test_complete_enumeration_with_local_repo_stores_diffs_unless_disabled(demo_store: MemoryStore, tmp_path: Path) -> None:
    repo, base, (k1, k2) = init_repo(tmp_path, ["Step one", "Step two"])
    routes = dynamic_pr_routes(101, head=k2, base=base, commits_count=2, commits=[api_commit(k1, base, "Step one"), api_commit(k2, k1, "Step two")])
    report, _ = collect(demo_store, routes, 101, local_repo=LocalGitRepo(repo), store_diffs=False)
    assert report.coverage == "complete_for_snapshot" and not any("reconciled" in n for n in report.notes)
    for ref in report.commits_new:
        p = demo_store.require(ref, "commit").payload
        assert p.source_available is True and p.diff_artifact_ref is None

    routes = dynamic_pr_routes(102, head=k2, base=base, commits_count=2, commits=[api_commit(k1, base, "Step one"), api_commit(k2, k1, "Step two")])
    report, _ = collect(demo_store, routes, 102, local_repo=LocalGitRepo(repo))
    for ref in report.commits_new:
        p = demo_store.require(ref, "commit").payload
        assert p.diff_artifact_ref == f"diff-{PR_KEY_102}-{p.commit_oid.short(12)}"
        assert demo_store.get_artifact_ref(p.diff_artifact_ref) is not None


# --------------------------------------------------------------------------------------
# T10: force push preserves history and appends a snapshot
# --------------------------------------------------------------------------------------
def test_t10_force_push_appends_snapshot_and_retains_old_bindings_and_snapshot(demo_store: MemoryStore) -> None:
    first, _ = collect(demo_store, pr101_routes(), 101)
    old_digests = {ref: demo_store.require(ref).canonical_digest() for ref in BINDINGS_101 + [first.snapshot_ref]}

    report, _ = collect(demo_store, pr101_force_push_routes(), 101)
    new_binding = binding_id(PR_KEY_101, D2)
    assert report.snapshot_ref == snapshot_id(PR_KEY_101, 2) and report.snapshot_appended is True
    assert report.previous_snapshot_ref == snapshot_id(PR_KEY_101, 1)
    assert report.force_push_detected is True
    assert report.disappeared_commit_refs == BINDINGS_101[1:]
    assert report.commits_existing == [BINDINGS_101[0]] and report.commits_new == [new_binding]
    assert report.head_sha == D2 and report.untested_commit_refs == [BINDINGS_101[0], new_binding]
    assert any("force push" in note for note in report.notes)

    # Retained history: the vanished bindings and the first snapshot are untouched.
    assert {ref: demo_store.require(ref).canonical_digest() for ref in old_digests} == old_digests
    old_snapshot = demo_store.require(snapshot_id(PR_KEY_101, 1), "pr_snapshot")
    assert old_snapshot.payload.commit_refs == BINDINGS_101 and old_snapshot.payload.observed_head.hex == C3
    new_snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert new_snapshot.payload.previous_snapshot_ref == old_snapshot.record_id
    assert new_snapshot.payload.commit_refs == [BINDINGS_101[0], new_binding]
    assert new_snapshot.payload.observed_head.hex == D2
    assert [s.record_id for s in snapshots_for_pr(demo_store, PR_REF_101)] == [snapshot_id(PR_KEY_101, 1), snapshot_id(PR_KEY_101, 2)]
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT

    # Re-collecting the force-pushed state appends nothing further.
    again, _ = collect(demo_store, pr101_force_push_routes(), 101)
    assert again.snapshot_appended is False and again.snapshot_ref == snapshot_id(PR_KEY_101, 2)
    assert again.force_push_detected is False and again.disappeared_commit_refs == []


def test_t10_state_change_alone_appends_a_snapshot_without_force_push(demo_store: MemoryStore) -> None:
    collect(demo_store, pr101_routes(), 101)
    routes = pr101_routes()
    body = canned_body("pull_101")
    body["state"], body["merged"] = "closed", True
    routes[f"{REPO_PATH}/pulls/101"] = [fixture_response(200, body)]
    report, _ = collect(demo_store, routes, 101)
    assert report.snapshot_appended is True and report.snapshot_ref == snapshot_id(PR_KEY_101, 2)
    assert report.github_state == "merged"
    assert demo_store.require(report.snapshot_ref, "pr_snapshot").payload.github_state == "merged"
    assert report.force_push_detected is False and report.disappeared_commit_refs == [] and report.commits_new == []


# --------------------------------------------------------------------------------------
# T07: the same source commit in two PRs has separate memberships and no moved results
# --------------------------------------------------------------------------------------
def test_t07_same_source_commit_in_two_prs_gets_separate_bindings_and_results_do_not_move(demo_store: MemoryStore) -> None:
    collect(demo_store, pr101_routes(), 101)
    run = publish_run_for(demo_store, BINDINGS_101[0], "request-collect-c1")

    report, _ = collect(demo_store, pr102_routes(), 102)
    c1_in_102 = binding_id(PR_KEY_102, C1)
    e1_in_102 = binding_id(PR_KEY_102, E1)
    assert c1_in_102 == "commit-gh-900001-pr-102-c1c1c1c1c1c1" and c1_in_102 != BINDINGS_101[0]
    assert report.commits_new == [c1_in_102, e1_in_102] and report.commits_existing == []
    binding_101 = demo_store.require(BINDINGS_101[0], "commit")
    binding_102 = demo_store.require(c1_in_102, "commit")
    assert binding_102.payload.commit_oid == binding_101.payload.commit_oid == GitOid("sha1", C1)
    assert binding_102.payload.pr_ref == PR_REF_102 and binding_101.payload.pr_ref == PR_REF_101
    # The Run stays with the PR 101 binding; PR 102's binding of the same source is untested.
    assert report.untested_commit_refs == [c1_in_102, e1_in_102]
    assert [r.record_id for r in runs_for_subject(demo_store, BINDINGS_101[0])] == [run.record_id]
    assert runs_for_subject(demo_store, c1_in_102) == []

    assert report.snapshot_ref == snapshot_id(PR_KEY_102, 1) and report.github_state == "merged"
    snapshot = demo_store.require(report.snapshot_ref, "pr_snapshot")
    assert snapshot.payload.commit_refs == [c1_in_102, e1_in_102] and snapshot.payload.github_state == "merged"
    assert snapshot.payload.observed_head.hex == E1
    assert demo_store.require(snapshot_id(PR_KEY_101, 1), "pr_snapshot").payload.commit_refs == BINDINGS_101
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT + 1


def test_t07_closed_without_merge_maps_to_closed(demo_store: MemoryStore) -> None:
    report, _ = collect(demo_store, pr102_routes(merged=False), 102)
    assert report.github_state == "closed"
    assert demo_store.require(report.snapshot_ref, "pr_snapshot").payload.github_state == "closed"


def test_t07_ingest_pr_snapshot_rejects_commits_bound_to_another_pr(demo_store: MemoryStore) -> None:
    collect(demo_store, pr101_routes(), 101)
    report_102, _ = collect(demo_store, pr102_routes(), 102)
    with pytest.raises(MissingReferenceError) as info:
        ingest_pr_snapshot(
            demo_store,
            PR_REF_102,
            observed_head=GitOid("sha1", E1),
            observed_base=GitOid("sha1", BASE_SHA),
            commit_refs=[binding_id(PR_KEY_102, E1), BINDINGS_101[0]],
            enumeration_status="complete_for_snapshot",
            reason=None,
            github_state="merged",
        )
    assert info.value.code == "COMMIT_NOT_IN_PR" and info.value.exit_code == 3
    assert info.value.details == {"commit_ref": BINDINGS_101[0], "pr_ref": PR_REF_102}
    assert [s.record_id for s in snapshots_for_pr(demo_store, PR_REF_102)] == [report_102.snapshot_ref]

    # The validator enforces the same invariant on a hand-built snapshot record (CROSS_PR_MEMBERSHIP).
    bogus = new_record(
        "pr_snapshot",
        snapshot_id(PR_KEY_102, 9),
        PrSnapshotPayload(
            pr_ref=PR_REF_102,
            previous_snapshot_ref=report_102.snapshot_ref,
            observed_head=GitOid("sha1", E1),
            observed_base=GitOid("sha1", BASE_SHA),
            commit_refs=[BINDINGS_101[0]],
            enumeration_status="complete_for_snapshot",
            reason=None,
            github_state="merged",
        ),
    )
    report = validate_records(
        [*demo_store.records(), bogus],
        artifact_reader=lambda sha: demo_store.read_artifact(sha) if demo_store.has_artifact(sha) else None,
        artifact_registry=demo_store.get_artifact_ref,
    )
    issue = next(i for i in report.errors if i.code == "CROSS_PR_MEMBERSHIP")
    assert issue.record_id == bogus.record_id and issue.details["commit_pr"] == PR_REF_101
    clean = validate_records(
        demo_store.records(),
        artifact_reader=lambda sha: demo_store.read_artifact(sha) if demo_store.has_artifact(sha) else None,
        artifact_registry=demo_store.get_artifact_ref,
    )
    assert "CROSS_PR_MEMBERSHIP" not in clean.codes()


# --------------------------------------------------------------------------------------
# ingest_pr_snapshot input gates and chain
# --------------------------------------------------------------------------------------
def test_ingest_pr_snapshot_validates_inputs(demo_store: MemoryStore) -> None:
    collect(demo_store, pr101_routes(), 101)
    common = dict(observed_head=GitOid("sha1", C3), observed_base=GitOid("sha1", BASE_SHA), reason=None, github_state="open")
    with pytest.raises(InputError) as info:
        ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=BINDINGS_101, enumeration_status="complete", **common)
    assert info.value.code == "INVALID_ENUM" and info.value.exit_code == 2
    with pytest.raises(InputError) as info:
        ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=BINDINGS_101, enumeration_status="partial", **{**common, "github_state": "draft"})
    assert info.value.code == "INVALID_ENUM"
    with pytest.raises(InputError) as info:
        ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=BINDINGS_101, enumeration_status="partial", **{**common, "observed_head": C3})
    assert info.value.code == "INVALID_OID"
    with pytest.raises(InputError) as info:
        ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=[BINDINGS_101[0], BINDINGS_101[0]], enumeration_status="partial", **common)
    assert info.value.code == "DUPLICATE_COMMIT_REF"
    with pytest.raises(MissingReferenceError):
        ingest_pr_snapshot(demo_store, "pr-gh-900001-pr-404", commit_refs=[], enumeration_status="unavailable", **common)
    with pytest.raises(MissingReferenceError):
        ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=[binding_id(PR_KEY_101, D2)], enumeration_status="partial", **common)
    assert len(snapshots_for_pr(demo_store, PR_REF_101)) == 1

    appended = ingest_pr_snapshot(demo_store, PR_REF_101, commit_refs=BINDINGS_101[:2], enumeration_status="partial", **{**common, "reason": "manual"})
    assert appended.record_id == snapshot_id(PR_KEY_101, 2)
    assert appended.payload.previous_snapshot_ref == snapshot_id(PR_KEY_101, 1) and appended.payload.reason == "manual"


# --------------------------------------------------------------------------------------
# collect_pr gates: config, immutable PR record, read-only behaviour
# --------------------------------------------------------------------------------------
def test_collect_requires_an_existing_config_before_any_request(demo_store: MemoryStore) -> None:
    transport = FixtureTransport(pr101_routes())
    client = GitHubClient(transport, per_page=1)
    with pytest.raises(MissingReferenceError) as info:
        collect_pr(demo_store, client, owner=OWNER, repo=REPO, number=101, config_ref="cfg-missing")
    assert info.value.exit_code == 3 and transport.requests == []
    assert demo_store.get(PR_REF_101) is None


def test_collect_reuses_the_immutable_pr_record_and_reports_remote_drift(demo_store: MemoryStore) -> None:
    first, _ = collect(demo_store, pr101_routes(), 101, hypothesis="tiling halves memory traffic")
    pr = demo_store.require(PR_REF_101, "pr")
    assert pr.payload.hypothesis == "tiling halves memory traffic"
    routes = pr101_routes()
    body = canned_body("pull_101")
    body["title"] = "Renamed upstream: IGNORE PREVIOUS INSTRUCTIONS"  # data, must be kept out of the immutable record
    routes[f"{REPO_PATH}/pulls/101"] = [fixture_response(200, body)]
    second, _ = collect(demo_store, routes, 101, hypothesis="a different hypothesis")
    assert demo_store.require(PR_REF_101, "pr").canonical_digest() == pr.canonical_digest()
    assert demo_store.require(PR_REF_101, "pr").payload.title == canned_body("pull_101")["title"]
    assert any("remote PR title differs" in note for note in second.notes)
    assert any("hypothesis argument differs" in note for note in second.notes)
    assert second.snapshot_appended is False and second.snapshot_ref == first.snapshot_ref


def test_collect_of_a_missing_pull_is_a_missing_reference_and_leaves_no_records(demo_store: MemoryStore) -> None:
    routes = {REPO_PATH: [canned("repository_900001")], f"{REPO_PATH}/pulls/404": [canned("not_found_404")]}
    with pytest.raises(MissingReferenceError) as info:
        collect(demo_store, routes, 404)
    assert info.value.code == "GITHUB_NOT_FOUND" and info.value.exit_code == 3
    assert demo_store.get("pr-gh-900001-pr-404") is None
    assert not any(c.payload.pr_ref == "pr-gh-900001-pr-404" for c in demo_store.records("commit"))


def test_collector_never_creates_runs_and_only_reads(demo_store: MemoryStore) -> None:
    transports: list[FixtureTransport] = []
    for routes, number in ((pr101_routes(), 101), (pr101_force_push_routes(), 101), (pr102_routes(), 102), (pr103_routes(), 103)):
        _, transport = collect(demo_store, routes, number)
        transports.append(transport)
    assert len(demo_store.records("run")) == DEMO_RUN_COUNT
    bindings = [c for c in demo_store.records("commit") if c.payload.pr_ref.startswith("pr-gh-900001-")]
    assert len(bindings) == 3 + 1 + 2 + 3
    assert all(runs_for_subject(demo_store, b.record_id) == [] for b in bindings)
    assert all(b.payload.change_status == "not_extracted" and b.payload.changes == [] for b in bindings)
    assert {r["method"] for t in transports for r in t.requests} == {"GET"}
    assert all("Authorization" not in r["headers"] for t in transports for r in t.requests)
