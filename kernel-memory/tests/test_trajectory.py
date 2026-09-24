"""Trajectory view tests (specification sections 4, 7, 14; acceptance T05, T06, T07, T24).

Every assertion is against the documented view shape produced by ``services.trajectory``:
``view_version``, ``config_ref``, ``config_hash``, ``kernel_id``, ``kernel_ref``, ``baselines``,
``prs``, ``nodes``, ``edges``, ``decisions``, ``best_known``, ``annotations``, ``relations``,
``diagnostics``, ``publishable`` and ``counts``. Synthetic records are built through
``services.common.new_record`` (schema-validated) and published through the raw store, so the
tests exercise the view generator, not the importer.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from conftest import ARTIFACT_ROOT, import_demo_bundle, record_dict
from kernel_memory.domain.errors import InvariantViolation, MissingReferenceError
from kernel_memory.domain.hashing import jcs_digest
from kernel_memory.domain.jsonio import dumps_compact, loads_strict
from kernel_memory.domain.models import TYPE_ORDER, CommitPayload, Record
from kernel_memory.services import trajectory as traj
from kernel_memory.services.common import new_record
from kernel_memory.storage import MemoryStore
from kernel_memory.storage import layout

CFG = "cfg-demo"
OID_1 = "0" * 39 + "1"  # the baseline's commit; no commit binding exists for it
OID_2 = "0" * 39 + "2"  # commit A, bound twice (commit-demo-a, commit-demo-a-in-102)
OID_3 = "0" * 39 + "3"  # commit B
OID_4 = "0" * 39 + "4"  # commit C

ALL_CONFIG_RECORDS = {
    "cfg-demo",
    "baseline-demo",
    "pr-demo-101",
    "pr-demo-102",
    "snapshot-demo-101",
    "snapshot-demo-102",
    "commit-demo-a",
    "commit-demo-b",
    "commit-demo-c",
    "commit-demo-a-in-102",
    "run-demo-baseline",
    "run-demo-a",
    "run-demo-c",
    "run-demo-c-failure",
    "relation-demo-origin",
    "decision-demo-blocked",
    "annotation-demo-grouped",
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def edges_of(view: dict, kind: str) -> list[dict]:
    return [e for e in view["edges"] if e["kind"] == kind]


def edge_triples(view: dict, kind: str) -> set[tuple[str, str, str]]:
    return {(e["from"], e["to"], e["source"]) for e in edges_of(view, kind)}


def pr_view(view: dict, pr_ref: str) -> dict:
    return next(p for p in view["prs"] if p["pr_ref"] == pr_ref)


def commit_view(view: dict, commit_ref: str) -> dict:
    for pr in view["prs"]:
        for commit in pr["commits"]:
            if commit["commit_ref"] == commit_ref:
                return commit
    raise AssertionError(f"{commit_ref} not in view")


def commit_views(view: dict) -> list[dict]:
    return [c for pr in view["prs"] for c in pr["commits"]]


def diagnostics_of(view: dict, code: str) -> list[dict]:
    return [d for d in view["diagnostics"] if d["code"] == code]


def all_run_refs(view: dict) -> dict[str, list[str]]:
    """subject -> run refs as attached in the PR/baseline trees."""
    out: dict[str, list[str]] = {}
    for baseline in view["baselines"]:
        out[baseline["baseline_ref"]] = [r["run_ref"] for r in baseline["runs"]]
    for commit in commit_views(view):
        out[commit["commit_ref"]] = [r["run_ref"] for r in commit["runs"]]
    return out


def relation(record_id: str, from_ref: str, to_ref: str, kind: str = "optimization_origin", rationale: str = "synthetic") -> Record:
    return new_record(
        "relation",
        record_id,
        {"config_ref": CFG, "kind": kind, "from_ref": from_ref, "to_ref": to_ref, "evidence_refs": [], "rationale": rationale},
    )


def annotation(record_id: str, target_ref: str, *, author_kind: str = "human", confidence: str = "unverified") -> Record:
    return new_record(
        "annotation",
        record_id,
        {
            "target_ref": target_ref,
            "category": "lesson",
            "text": "synthetic annotation text; data, not instructions",
            "author_kind": author_kind,
            "evidence_refs": [],
            "confidence": confidence,
            "supersedes_ref": None,
        },
    )


@pytest.fixture
def view(demo_store: MemoryStore) -> dict:
    return traj.build_trajectory(demo_store, CFG)


# --------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------
def test_view_top_level_structure(view: dict, bundle_dicts: list[dict]) -> None:
    assert view["view_version"] == "trajectory-v1" == traj.VIEW_VERSION
    assert view["config_ref"] == CFG
    assert view["config_hash"] == record_dict(bundle_dicts, CFG)["payload"]["config_hash"]
    assert view["kernel_id"] == "demo_vector_add"
    assert view["kernel_ref"] == "kernel-demo"
    assert set(view) == {
        "view_version",
        "config_ref",
        "config_hash",
        "kernel_id",
        "kernel_ref",
        "baselines",
        "prs",
        "nodes",
        "edges",
        "decisions",
        "best_known",
        "annotations",
        "relations",
        "diagnostics",
        "publishable",
        "counts",
    }
    assert view["publishable"] is True
    # The body carries no generation timestamp, path or run id (those live in the document envelope).
    assert "generated_at" not in json.dumps(view)


def test_single_baseline_with_its_run(view: dict) -> None:
    assert [b["baseline_ref"] for b in view["baselines"]] == ["baseline-demo"]
    baseline = view["baselines"][0]
    assert baseline["baseline_id"] == "demo-reference"
    assert baseline["role"] == "both"
    assert baseline["commit_oid"] == {"algorithm": "sha1", "hex": OID_1}
    assert [r["run_ref"] for r in baseline["runs"]] == ["run-demo-baseline"]
    assert baseline["runs"][0]["provenance"] == "fixture"


def test_prs_in_id_order_with_snapshots_commits_and_runs(view: dict) -> None:
    assert [p["pr_ref"] for p in view["prs"]] == ["pr-demo-101", "pr-demo-102"]
    for pr in view["prs"]:
        assert pr["snapshots"], pr["pr_ref"]
        assert pr["latest_snapshot_ref"] == pr["snapshots"][-1]["snapshot_ref"]
        assert pr["coverage"] == "complete_for_snapshot"
        assert pr["display_label"] == f"GitHub PR #{pr['number']}"
        for commit in pr["commits"]:
            assert commit["status"] in ("tested", "not_run")
            assert isinstance(commit["runs"], list)
            assert (commit["status"] == "tested") == bool(commit["runs"])
    pr101 = pr_view(view, "pr-demo-101")
    assert [s["snapshot_ref"] for s in pr101["snapshots"]] == ["snapshot-demo-101"]
    assert pr101["snapshots"][0]["commit_refs"] == ["commit-demo-a", "commit-demo-b"]
    assert [c["commit_ref"] for c in pr101["commits"]] == ["commit-demo-a", "commit-demo-b"]
    pr102 = pr_view(view, "pr-demo-102")
    assert [c["commit_ref"] for c in pr102["commits"]] == ["commit-demo-a-in-102", "commit-demo-c"]
    # display ordinals follow the stored snapshot order, and are display-only
    assert [c["display_ordinal"] for c in pr102["commits"]] == [1, 2]
    assert all(c["in_latest_snapshot"] for c in pr102["commits"])


def test_counts_match_scope(view: dict) -> None:
    counts = view["counts"]
    assert counts["baselines"] == 1
    assert counts["prs"] == 2
    assert counts["snapshots"] == 2
    assert counts["commits"] == 4
    assert counts["runs"] == 4
    assert counts["relations"] == 1
    assert counts["decisions"] == 1
    assert counts["annotations"] == 1
    assert counts["nodes"] == len(view["nodes"])
    assert counts["edges"] == len(view["edges"])
    assert counts["diagnostics"] == len(view["diagnostics"])
    assert counts["errors"] == 0


def test_nodes_are_typed_unique_and_sorted(view: dict) -> None:
    ids = [n["id"] for n in view["nodes"]]
    assert len(ids) == len(set(ids))
    assert {n["type"] for n in view["nodes"]} == {"pr", "pr_snapshot", "commit", "baseline", "run", "decision"}
    keys = [(TYPE_ORDER[n["type"]], n["id"]) for n in view["nodes"]]
    assert keys == sorted(keys)
    by_id = {n["id"]: n["type"] for n in view["nodes"]}
    assert by_id["commit-demo-a"] == "commit"
    assert by_id["commit-demo-a-in-102"] == "commit"
    assert by_id["run-demo-c-failure"] == "run"
    assert by_id["decision-demo-blocked"] == "decision"


# --------------------------------------------------------------------------------------
# T05: collect three commits, execute one; the others remain untested
# --------------------------------------------------------------------------------------
def test_t05_untested_commit_has_status_not_run_and_no_run_anywhere(view: dict) -> None:
    commit_b = commit_view(view, "commit-demo-b")
    assert commit_b["status"] == "not_run"
    assert commit_b["runs"] == []
    # no run is attached to, or derived for, commit-demo-b anywhere in the view
    assert all("commit-demo-b" != run["to"] for run in edges_of(view, "run_of"))
    for subject, runs in all_run_refs(view).items():
        if subject == "commit-demo-b":
            assert runs == []
    run_nodes = {n["id"] for n in view["nodes"] if n["type"] == "run"}
    assert run_nodes == {"run-demo-baseline", "run-demo-a", "run-demo-c", "run-demo-c-failure"}
    assert all(d["candidate_subject_ref"] != "commit-demo-b" for d in view["decisions"])


def test_t05_tested_commit_lists_only_its_run(view: dict) -> None:
    commit_a = commit_view(view, "commit-demo-a")
    assert commit_a["status"] == "tested"
    assert [r["run_ref"] for r in commit_a["runs"]] == ["run-demo-a"]
    assert commit_a["runs"][0]["provenance"] == "fixture"
    assert ("run-demo-a", "commit-demo-a", "structure") in edge_triples(view, "run_of")


def test_t05_untested_count_and_warning(view: dict) -> None:
    assert view["counts"]["untested_commits"] == 2
    assert {c["commit_ref"] for c in commit_views(view) if c["status"] == "not_run"} == {"commit-demo-b", "commit-demo-a-in-102"}
    warnings = diagnostics_of(view, "UNTESTED_COMMITS")
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["refs"] == ["commit-demo-a-in-102", "commit-demo-b"]
    assert "no recorded run" in warnings[0]["message"]
    # a warning never blocks publication
    assert view["publishable"] is True


def test_t05_status_is_derived_from_runs_not_stored(bundle_dicts: list[dict], bundle_records: list[Record], tmp_path: Path) -> None:
    for commit_id in ("commit-demo-a", "commit-demo-b", "commit-demo-c", "commit-demo-a-in-102"):
        assert "status" not in record_dict(bundle_dicts, commit_id)["payload"]
    assert "status" not in {f.name for f in fields(CommitPayload)}
    # Same commit records, no runs published: both commits are not_run at query time.
    store = MemoryStore.init(tmp_path / "memory")
    keep = {"kernel-demo", "cfg-demo", "baseline-demo", "pr-demo-101", "snapshot-demo-101", "commit-demo-a", "commit-demo-b"}
    store.publish_bundle([r for r in bundle_records if r.record_id in keep])
    before = traj.build_trajectory(store, CFG)
    assert [(c["commit_ref"], c["status"]) for c in commit_views(before)] == [("commit-demo-a", "not_run"), ("commit-demo-b", "not_run")]
    assert before["counts"]["untested_commits"] == 2
    # Publishing the run (an authoritative fact) flips the derived status; nothing else changes.
    store.publish(next(r for r in bundle_records if r.record_id == "run-demo-a"))
    after = traj.build_trajectory(store, CFG)
    assert [(c["commit_ref"], c["status"]) for c in commit_views(after)] == [("commit-demo-a", "tested"), ("commit-demo-b", "not_run")]
    assert after["counts"]["untested_commits"] == 1
    assert [r["run_ref"] for r in commit_view(after, "commit-demo-a")["runs"]] == ["run-demo-a"]


# --------------------------------------------------------------------------------------
# T06: PR2 branches from a non-head commit of PR1
# --------------------------------------------------------------------------------------
def test_t06_pr102_origin_is_commit_a_not_pr101_head(view: dict) -> None:
    assert pr_view(view, "pr-demo-102")["origin"] == {"ref": "commit-demo-a", "record_type": "commit"}
    assert pr_view(view, "pr-demo-101")["origin"] == {"ref": "baseline-demo", "record_type": "baseline"}
    # PR 101's head is commit B; no origin edge ever points from B
    assert all(e["from"] != "commit-demo-b" for e in edges_of(view, "optimization_origin"))


def test_t06_optimization_origin_edges_from_pr_origin_and_relation(view: dict) -> None:
    assert edge_triples(view, "optimization_origin") == {
        ("baseline-demo", "pr-demo-101", traj.SOURCE_PR_ORIGIN),
        ("commit-demo-a", "pr-demo-102", traj.SOURCE_PR_ORIGIN),
        ("commit-demo-a", "commit-demo-c", "relation:relation-demo-origin"),
    }
    assert traj.SOURCE_PR_ORIGIN == "pr.origin_ref"
    assert view["relations"] == [
        {
            "relation_ref": "relation-demo-origin",
            "kind": "optimization_origin",
            "from_ref": "commit-demo-a",
            "to_ref": "commit-demo-c",
            "evidence_refs": [],
            "rationale": "An explicit optimization-origin edge, separate from Git parents and benchmark baseline.",
        }
    ]


# --------------------------------------------------------------------------------------
# T07: same source commit in multiple PRs -> separate memberships, no moved results
# --------------------------------------------------------------------------------------
def test_t07_shared_source_commit_is_two_bindings_and_runs_do_not_move(view: dict) -> None:
    a = commit_view(view, "commit-demo-a")
    a2 = commit_view(view, "commit-demo-a-in-102")
    assert a["commit_oid"] == a2["commit_oid"] == {"algorithm": "sha1", "hex": OID_2}
    assert a["commit_ref"] != a2["commit_ref"]
    assert [r["run_ref"] for r in a["runs"]] == ["run-demo-a"]
    assert a2["runs"] == [] and a2["status"] == "not_run"
    run_of = edge_triples(view, "run_of")
    assert ("run-demo-a", "commit-demo-a", "structure") in run_of
    assert all(dst != "commit-demo-a-in-102" for _, dst, _ in run_of)
    memberships = edge_triples(view, "membership")
    assert ("pr-demo-101", "commit-demo-a", "structure") in memberships
    assert ("pr-demo-102", "commit-demo-a-in-102", "structure") in memberships
    assert ("pr-demo-102", "commit-demo-a", "structure") not in memberships
    assert ("pr-demo-101", "commit-demo-a-in-102", "structure") not in memberships


def test_t07_shared_source_diagnostic_is_informational(view: dict) -> None:
    shared = diagnostics_of(view, "SHARED_SOURCE")
    assert len(shared) == 1
    assert shared[0]["severity"] in ("info", "warning")
    assert shared[0]["severity"] != "error"
    assert shared[0]["refs"] == ["commit-demo-a", "commit-demo-a-in-102"]
    assert OID_2 in shared[0]["message"]
    assert view["publishable"] is True
    assert view["counts"]["errors"] == 0


def test_t07_git_parent_edges_resolve_to_every_binding_of_the_parent_oid(view: dict) -> None:
    git = edge_triples(view, "git_parent")
    assert all(source == traj.SOURCE_GIT == "git" for _, _, source in git)
    # B's parent is oid ...0002, bound twice -> both bindings are targets; likewise for C.
    assert {dst for src, dst, _ in git if src == "commit-demo-b"} == {"commit-demo-a", "commit-demo-a-in-102"}
    assert {dst for src, dst, _ in git if src == "commit-demo-c"} == {"commit-demo-a", "commit-demo-a-in-102"}
    # oid ...0001 is the baseline's commit: no commit binding exists for it, so no git_parent edge.
    assert all(src not in ("commit-demo-a", "commit-demo-a-in-102") for src, _, _ in git)
    assert all(dst != "baseline-demo" for _, dst, _ in git)
    assert len(git) == 4
    assert not diagnostics_of(view, "GIT_PARENT_CYCLE")


# --------------------------------------------------------------------------------------
# edge kinds are kept apart
# --------------------------------------------------------------------------------------
def test_edge_kinds_are_separate_and_not_conflated(view: dict) -> None:
    kinds = {e["kind"] for e in view["edges"]}
    assert kinds == {"membership", "git_parent", "optimization_origin", "comparison_baseline", "run_of", "snapshot_of"}
    for edge in view["edges"]:
        assert set(edge) == {"kind", "from", "to", "source"}
    baseline_edges = edge_triples(view, "comparison_baseline")
    assert baseline_edges == {("run-demo-a", "run-demo-baseline", "decision:decision-demo-blocked")}
    pairs = {kind: {(e["from"], e["to"]) for e in edges_of(view, kind)} for kind in kinds}
    # no (from, to) pair appears under two different relationship kinds
    for k1 in pairs:
        for k2 in pairs:
            if k1 < k2:
                assert not (pairs[k1] & pairs[k2]), (k1, k2)
    assert pairs["membership"] == {
        ("pr-demo-101", "commit-demo-a"),
        ("pr-demo-101", "commit-demo-b"),
        ("pr-demo-102", "commit-demo-a-in-102"),
        ("pr-demo-102", "commit-demo-c"),
    }
    assert pairs["snapshot_of"] == {("snapshot-demo-101", "pr-demo-101"), ("snapshot-demo-102", "pr-demo-102")}
    assert pairs["run_of"] == {
        ("run-demo-baseline", "baseline-demo"),
        ("run-demo-a", "commit-demo-a"),
        ("run-demo-c", "commit-demo-c"),
        ("run-demo-c-failure", "commit-demo-c"),
    }
    # edges are sorted deterministically
    keys = [(e["kind"], e["from"], e["to"], e["source"]) for e in view["edges"]]
    assert keys == sorted(keys)


def test_time_ordering_is_never_used_as_ancestry(view: dict) -> None:
    # commit-demo-c (oid 0004) was recorded after commit-demo-b (0003) but its git parent is 0002:
    # the graph must not link c -> b just because b is "earlier".
    git = {(src, dst) for src, dst, _ in edge_triples(view, "git_parent")}
    assert ("commit-demo-c", "commit-demo-b") not in git
    assert ("commit-demo-b", "commit-demo-c") not in git


# --------------------------------------------------------------------------------------
# decisions, best-known, annotations
# --------------------------------------------------------------------------------------
def test_decision_listed_and_fixture_decision_is_not_best_known(view: dict) -> None:
    assert [d["decision_ref"] for d in view["decisions"]] == ["decision-demo-blocked"]
    decision = view["decisions"][0]
    assert decision["outcome"] == "blocked"
    assert decision["reason_codes"] == ["FIXTURE_NOT_ELIGIBLE", "INSUFFICIENT_CONFIRMATION_PAIRS"]
    assert decision["is_production"] is False
    assert decision["candidate_subject_ref"] == "commit-demo-a"
    assert decision["candidate_run_refs"] == ["run-demo-a"]
    assert decision["baseline_run_refs"] == ["run-demo-baseline"]
    assert decision["policy_id"] == "default-confirm-v1"
    assert view["best_known"] == []


def test_best_known_requires_accepted_production_and_honours_supersession(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    base = record_dict(bundle_dicts, "decision-demo-blocked")
    nonprod = copy.deepcopy(base)
    nonprod["record_id"] = "decision-test-accepted-nonprod"
    nonprod["payload"]["outcome"] = "accepted"
    nonprod["payload"]["reason_codes"] = []
    demo_store.publish(Record.from_dict(nonprod))
    view = traj.build_trajectory(demo_store, CFG)
    assert view["best_known"] == []  # accepted but not production -> never best-known
    prod = copy.deepcopy(nonprod)
    prod["record_id"] = "decision-test-accepted-prod"
    prod["payload"]["is_production"] = True
    demo_store.publish(Record.from_dict(prod))
    view = traj.build_trajectory(demo_store, CFG)
    assert view["best_known"] == [
        {
            "comparison_key": base["payload"]["comparison_key"],
            "policy_hash": base["payload"]["policy_hash"],
            "candidate_subject_ref": "commit-demo-a",
            "decision_ref": "decision-test-accepted-prod",
        }
    ]
    assert not diagnostics_of(view, "AMBIGUOUS_BEST_KNOWN")
    # a second accepted production decision without supersession is ambiguous (warning, still publishable)
    other = copy.deepcopy(prod)
    other["record_id"] = "decision-test-accepted-prod-other"
    demo_store.publish(Record.from_dict(other))
    view = traj.build_trajectory(demo_store, CFG)
    ambiguous = diagnostics_of(view, "AMBIGUOUS_BEST_KNOWN")
    assert len(ambiguous) == 1 and ambiguous[0]["severity"] == "warning"
    assert ambiguous[0]["refs"] == ["decision-test-accepted-prod", "decision-test-accepted-prod-other"]
    assert view["publishable"] is True
    # supersession resolves it
    newer = copy.deepcopy(prod)
    newer["record_id"] = "decision-test-accepted-prod-newer"
    newer["payload"]["supersedes_decision_ref"] = "decision-test-accepted-prod"
    demo_store.publish(Record.from_dict(newer))
    view = traj.build_trajectory(demo_store, CFG)
    assert {b["decision_ref"] for b in view["best_known"]} == {"decision-test-accepted-prod-other", "decision-test-accepted-prod-newer"}
    assert "decision-test-accepted-prod" not in {b["decision_ref"] for b in view["best_known"]}


def test_annotation_listed_as_hypothesis(view: dict) -> None:
    assert len(view["annotations"]) == 1
    ann = view["annotations"][0]
    assert ann["annotation_ref"] == "annotation-demo-grouped"
    assert ann["kind"] == "hypothesis"
    assert ann["target_ref"] == "commit-demo-c"
    assert ann["category"] == "lesson"
    assert ann["author_kind"] == "agent"
    assert ann["confidence"] == "unverified"
    assert ann["evidence_refs"] == ["run-demo-c"]


def test_annotation_kind_fact_only_for_supported_program_annotations() -> None:
    assert traj.annotation_kind(annotation("annotation-test-1", "commit-demo-c", author_kind="program", confidence="supported")) == "fact"
    assert traj.annotation_kind(annotation("annotation-test-2", "commit-demo-c", author_kind="program", confidence="unverified")) == "hypothesis"
    assert traj.annotation_kind(annotation("annotation-test-3", "commit-demo-c", author_kind="agent", confidence="supported")) == "hypothesis"
    assert traj.annotation_kind(annotation("annotation-test-4", "commit-demo-c", author_kind="human", confidence="contradicted")) == "hypothesis"


def test_run_summary_copies_values_and_keeps_uncollected_metrics_null(demo_store: MemoryStore) -> None:
    failure = traj.run_summary(demo_store.require("run-demo-c-failure", "run"))
    assert failure["execution_status"] == "compile_error"
    assert failure["failure_reason"] == "Synthetic compile failure."
    assert failure["correctness"]["status"] == "not_run"
    assert failure["timing"] == {"status": "not_run", "sample_count": 0, "median_us": None, "p90_us": None}
    assert failure["metrics"] == [
        {"name": "register_spill_vmem_static_bytes", "status": "not_collected", "value": None, "unit": "bytes", "kind": "static_estimate", "scope": "compiled_kernel"}
    ]
    assert failure["provenance"] == "fixture"
    assert failure["artifact_refs"] == []
    ok = traj.run_summary(demo_store.require("run-demo-c", "run"))
    assert ok["timing"]["median_us"] == 88 and ok["timing"]["status"] == "recorded"
    assert ok["metrics"][0]["status"] == "observed" and ok["metrics"][0]["value"] == 65536
    assert ok["tested_commit"] == {"algorithm": "sha1", "hex": OID_4}
    assert ok["checkout_mode"] == "exact_commit"


def test_pr_display_label(demo_store: MemoryStore) -> None:
    assert traj.pr_display_label(demo_store.require("pr-demo-101", "pr")) == "GitHub PR #101"
    local = new_record(
        "pr",
        "pr-local-trial-abc123",
        {
            "config_ref": CFG,
            "hypothesis": None,
            "number": None,
            "origin_ref": None,
            "pr_key": "local-trial-abc123",
            "provider": "local",
            "repo_uid": "local:repo",
            "title": "local trial",
        },
    )
    assert traj.pr_display_label(local) == traj.LOCAL_TRIAL_LABEL == "local trial, not yet a GitHub PR"


# --------------------------------------------------------------------------------------
# T24: determinism and rebuild
# --------------------------------------------------------------------------------------
def test_t24_build_twice_is_identical(demo_store: MemoryStore) -> None:
    first = traj.build_trajectory(demo_store, CFG)
    second = traj.build_trajectory(demo_store, CFG)
    assert first == second
    assert traj.trajectory_hash(first) == traj.trajectory_hash(second)
    assert traj.trajectory_hash(first) == jcs_digest(first)
    assert traj.trajectory_hash(first).startswith("sha256:")


def test_t24_import_order_does_not_change_the_view(demo_store: MemoryStore, store: MemoryStore, bundle_records: list[Record]) -> None:
    shuffled = list(bundle_records)
    random.Random(7).shuffle(shuffled)
    assert [r.record_id for r in shuffled] != [r.record_id for r in bundle_records]
    store.publish_bundle(shuffled, label="shuffled")
    reference = traj.build_trajectory(demo_store, CFG)
    rebuilt = traj.build_trajectory(store, CFG)
    assert rebuilt == reference
    assert traj.trajectory_hash(rebuilt) == traj.trajectory_hash(reference)
    assert traj.memory_records(store, CFG) == traj.memory_records(demo_store, CFG)


def test_t24_delete_views_and_cache_then_rebuild_identically(demo_store: MemoryStore) -> None:
    published = traj.publish_trajectory(demo_store, CFG)
    assert published["publishable"] is True and published["forced"] is False
    assert published["record_count"] == len(ALL_CONFIG_RECORDS)
    view_file = Path(published["path"])
    records_file = Path(published["memory_records_path"])
    assert view_file == traj.view_path(demo_store, CFG)
    assert view_file.is_file() and records_file.is_file()
    original_hash = published["view_hash"]
    original_lines = records_file.read_bytes()
    stored = traj.read_stored_trajectory(demo_store, CFG)
    assert stored["view_hash"] == original_hash
    assert stored["view"] == traj.build_trajectory(demo_store, CFG)
    assert traj.verify_trajectory(demo_store, CFG)["matches"] is True

    # T24: delete the derived trajectory and the disposable cache
    demo_store.rebuild_index()
    cache_dir = demo_store.root / layout.CACHE_DIR
    assert cache_dir.is_dir()
    removed = demo_store.delete_views(CFG)
    assert set(removed) == {traj.TRAJECTORY_VIEW_FILE, traj.MEMORY_RECORDS_FILE}
    shutil.rmtree(cache_dir)
    assert not view_file.exists() and not records_file.exists() and not cache_dir.exists()
    assert traj.read_stored_trajectory(demo_store, CFG) is None
    missing = traj.verify_trajectory(demo_store, CFG)
    assert missing == {"stored": False, "stored_hash": None, "rebuilt_hash": original_hash, "matches": False, "stored_view_consistent": None}

    rebuilt = traj.rebuild_trajectory(demo_store, CFG)
    assert rebuilt["view_hash"] == original_hash
    assert rebuilt["forced"] is False
    assert Path(rebuilt["memory_records_path"]).read_bytes() == original_lines
    assert traj.read_stored_trajectory(demo_store, CFG)["view"] == stored["view"]
    assert traj.verify_trajectory(demo_store, CFG)["matches"] is True
    # rebuild_trajectory removes whatever views exist before republishing
    again = traj.rebuild_trajectory(demo_store, CFG)
    assert set(again["removed_views"]) == {traj.TRAJECTORY_VIEW_FILE, traj.MEMORY_RECORDS_FILE}
    assert again["view_hash"] == original_hash


def test_stored_document_envelope_carries_timestamp_outside_the_hashed_body(demo_store: MemoryStore) -> None:
    traj.publish_trajectory(demo_store, CFG)
    raw = demo_store.read_view(CFG, traj.TRAJECTORY_VIEW_FILE)
    document = loads_strict(raw)
    assert set(document) == {"generated_at", "view_hash", "view", "forced"}
    assert document["forced"] is False
    assert document["view_hash"] == jcs_digest(document["view"])
    assert "generated_at" not in document["view"]


def test_verify_detects_stale_view_after_new_annotation(demo_store: MemoryStore) -> None:
    traj.publish_trajectory(demo_store, CFG)
    assert traj.verify_trajectory(demo_store, CFG)["matches"] is True
    demo_store.publish(annotation("annotation-test-stale", "commit-demo-b"))
    report = traj.verify_trajectory(demo_store, CFG)
    assert report["stored"] is True
    assert report["matches"] is False
    assert report["stored_view_consistent"] is True  # the file itself is intact, just stale
    assert report["stored_hash"] != report["rebuilt_hash"]
    traj.rebuild_trajectory(demo_store, CFG)
    fresh = traj.verify_trajectory(demo_store, CFG)
    assert fresh["matches"] is True
    assert fresh["stored_hash"] == report["rebuilt_hash"]
    view = traj.read_stored_trajectory(demo_store, CFG)["view"]
    assert [a["annotation_ref"] for a in view["annotations"]] == ["annotation-demo-grouped", "annotation-test-stale"]
    assert view["counts"]["annotations"] == 2


def test_verify_detects_tampered_stored_document(demo_store: MemoryStore) -> None:
    published = traj.publish_trajectory(demo_store, CFG)
    path = Path(published["path"])
    document = loads_strict(path.read_bytes())
    document["view"]["counts"]["untested_commits"] = 0
    path.write_text(json.dumps(document), encoding="utf-8")
    report = traj.verify_trajectory(demo_store, CFG)
    assert report["stored_view_consistent"] is False
    assert report["matches"] is False


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def test_origin_cycle_is_an_error_and_blocks_publication(demo_store: MemoryStore) -> None:
    demo_store.publish_bundle(
        [
            relation("relation-test-ab", "commit-demo-a", "commit-demo-b", rationale="a to b"),
            relation("relation-test-ba", "commit-demo-b", "commit-demo-a", rationale="b to a"),
        ]
    )
    view = traj.build_trajectory(demo_store, CFG)
    cycles = diagnostics_of(view, "ORIGIN_CYCLE")
    assert len(cycles) == 1
    assert cycles[0]["severity"] == "error"
    assert set(cycles[0]["refs"]) == {"commit-demo-a", "commit-demo-b"}
    assert view["publishable"] is False
    assert view["counts"]["errors"] == 1
    # both relations are still reported as data (kept, not dropped)
    assert {r["relation_ref"] for r in view["relations"]} == {"relation-demo-origin", "relation-test-ab", "relation-test-ba"}
    assert ("commit-demo-b", "commit-demo-a", "relation:relation-test-ba") in edge_triples(view, "optimization_origin")
    # errors are sorted first
    assert view["diagnostics"][0]["severity"] == "error"

    with pytest.raises(InvariantViolation) as excinfo:
        traj.publish_trajectory(demo_store, CFG)
    assert excinfo.value.code == "TRAJECTORY_NOT_PUBLISHABLE"
    assert excinfo.value.exit_code == 2
    assert excinfo.value.details["config_ref"] == CFG
    assert [e["code"] for e in excinfo.value.details["errors"]] == ["ORIGIN_CYCLE"]
    assert traj.read_stored_trajectory(demo_store, CFG) is None
    with pytest.raises(InvariantViolation):
        traj.rebuild_trajectory(demo_store, CFG)

    forced = traj.publish_trajectory(demo_store, CFG, force=True)
    assert forced["forced"] is True
    assert forced["publishable"] is False
    assert Path(forced["path"]).is_file()
    stored = traj.read_stored_trajectory(demo_store, CFG)
    assert stored["forced"] is True
    assert stored["view"]["publishable"] is False
    assert traj.verify_trajectory(demo_store, CFG)["matches"] is True


def test_dangling_reference_becomes_diagnostic_not_exception(demo_store: MemoryStore) -> None:
    demo_store.publish_bundle(
        [relation("relation-test-dangling", "commit-demo-a", "commit-does-not-exist", rationale="dangling target")],
        allow_dangling=True,
    )
    view = traj.build_trajectory(demo_store, CFG)  # must not raise
    missing = diagnostics_of(view, "MISSING_REFERENCE")
    assert len(missing) == 1
    assert missing[0]["severity"] == "error"
    assert missing[0]["refs"] == ["commit-does-not-exist", "relation-test-dangling"]
    assert "to_ref" in missing[0]["message"]
    assert view["publishable"] is False
    # the affected branch is kept, not silently removed
    assert "relation-test-dangling" in {r["relation_ref"] for r in view["relations"]}
    assert ("commit-demo-a", "commit-does-not-exist", "relation:relation-test-dangling") in edge_triples(view, "optimization_origin")
    assert not diagnostics_of(view, "ORIGIN_CYCLE")
    with pytest.raises(InvariantViolation) as excinfo:
        traj.publish_trajectory(demo_store, CFG)
    assert excinfo.value.code == "TRAJECTORY_NOT_PUBLISHABLE"


def test_partial_snapshot_gives_partial_coverage_warning(demo_store: MemoryStore) -> None:
    partial = new_record(
        "pr_snapshot",
        "snapshot-test-partial",
        {
            "pr_ref": "pr-demo-101",
            "previous_snapshot_ref": "snapshot-demo-101",
            "observed_head": {"algorithm": "sha1", "hex": OID_3},
            "observed_base": {"algorithm": "sha1", "hex": OID_1},
            "commit_refs": ["commit-demo-a", "commit-demo-b"],
            "enumeration_status": "partial",
            "reason": "provider listing truncated",
            "github_state": "open",
        },
    )
    demo_store.publish(partial)
    view = traj.build_trajectory(demo_store, CFG)
    pr = pr_view(view, "pr-demo-101")
    assert [s["snapshot_ref"] for s in pr["snapshots"]] == ["snapshot-demo-101", "snapshot-test-partial"]
    assert pr["latest_snapshot_ref"] == "snapshot-test-partial"
    assert pr["coverage"] == "partial"
    assert pr["snapshots"][-1]["enumeration_status"] == "partial"
    assert pr["snapshots"][-1]["reason"] == "provider listing truncated"
    warnings = diagnostics_of(view, "PARTIAL_COVERAGE")
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["refs"] == ["pr-demo-101", "snapshot-test-partial"]
    assert pr_view(view, "pr-demo-102")["coverage"] == "complete_for_snapshot"
    # a warning never blocks publication
    assert view["publishable"] is True
    assert traj.publish_trajectory(demo_store, CFG)["forced"] is False
    assert ("snapshot-test-partial", "pr-demo-101", "structure") in edge_triples(view, "snapshot_of")
    assert view["counts"]["snapshots"] == 3


def test_pr_without_snapshot_reports_no_snapshot_coverage(store: MemoryStore, bundle_records: list[Record]) -> None:
    keep = {"kernel-demo", "cfg-demo", "baseline-demo", "pr-demo-101", "commit-demo-a", "commit-demo-b"}
    store.publish_bundle([r for r in bundle_records if r.record_id in keep])
    view = traj.build_trajectory(store, CFG)
    pr = view["prs"][0]
    assert pr["snapshots"] == [] and pr["latest_snapshot_ref"] is None
    assert pr["coverage"] == "no_snapshot"
    warnings = diagnostics_of(view, "PARTIAL_COVERAGE")
    assert len(warnings) == 1 and warnings[0]["refs"] == ["pr-demo-101"]
    # commits are still listed via their pr_ref binding, with display ordinals assigned deterministically
    assert [(c["commit_ref"], c["in_latest_snapshot"], c["display_ordinal"]) for c in pr["commits"]] == [
        ("commit-demo-a", False, 1),
        ("commit-demo-b", False, 2),
    ]


def test_cross_pr_membership_is_an_error(demo_store: MemoryStore) -> None:
    bad = new_record(
        "pr_snapshot",
        "snapshot-test-cross",
        {
            "pr_ref": "pr-demo-102",
            "previous_snapshot_ref": "snapshot-demo-102",
            "observed_head": {"algorithm": "sha1", "hex": OID_4},
            "observed_base": {"algorithm": "sha1", "hex": OID_1},
            "commit_refs": ["commit-demo-b", "commit-demo-c"],  # commit-demo-b is bound to pr-demo-101
            "enumeration_status": "complete_for_snapshot",
            "reason": None,
            "github_state": "open",
        },
    )
    demo_store.publish(bad)
    view = traj.build_trajectory(demo_store, CFG)
    errors = diagnostics_of(view, "CROSS_PR_MEMBERSHIP")
    assert len(errors) == 1 and errors[0]["severity"] == "error"
    assert set(errors[0]["refs"]) == {"snapshot-test-cross", "commit-demo-b", "pr-demo-101", "pr-demo-102"}
    assert view["publishable"] is False
    # the foreign commit is not moved into pr-demo-102
    assert ("pr-demo-102", "commit-demo-b", "structure") not in edge_triples(view, "membership")
    assert [c["commit_ref"] for c in pr_view(view, "pr-demo-102")["commits"]] == ["commit-demo-a-in-102", "commit-demo-c"]


def test_diagnostics_are_sorted_and_deduplicated(view: dict) -> None:
    order = {"error": 0, "warning": 1, "info": 2}
    keys = [(order[d["severity"]], d["code"], d["message"], d["refs"]) for d in view["diagnostics"]]
    assert keys == sorted(keys)
    assert len(keys) == len(set(dumps_compact(d) for d in view["diagnostics"]))
    for d in view["diagnostics"]:
        assert set(d) == {"severity", "code", "message", "refs"}
        assert d["refs"] == sorted(set(d["refs"]))


# --------------------------------------------------------------------------------------
# memory_records.jsonl
# --------------------------------------------------------------------------------------
def test_memory_records_one_line_per_record_sorted_deterministically(demo_store: MemoryStore) -> None:
    lines = traj.memory_records(demo_store, CFG)
    assert [l["record_ref"] for l in lines] == sorted((l["record_ref"] for l in lines), key=lambda rid: (TYPE_ORDER[demo_store.require(rid).record_type], rid))
    assert {l["record_ref"] for l in lines} == ALL_CONFIG_RECORDS
    assert "kernel-demo" not in {l["record_ref"] for l in lines}  # the kernel is not owned by one config
    assert len(lines) == len(ALL_CONFIG_RECORDS)
    for line in lines:
        assert set(line) == {"record_ref", "record_type", "summary", "kind", "evidence_refs", "provenance", "config_ref"}
        assert line["record_type"] == demo_store.require(line["record_ref"]).record_type
        assert isinstance(line["summary"], str) and line["summary"]
        assert line["kind"] in ("fact", "hypothesis")
        assert line["evidence_refs"] == sorted(set(line["evidence_refs"]))
        assert line["config_ref"] == CFG
        if line["record_type"] == "run":
            assert line["provenance"] == "fixture"
            assert "not production evidence" in line["summary"]
        else:
            assert line["provenance"] is None
    by_ref = {l["record_ref"]: l for l in lines}
    assert by_ref["annotation-demo-grouped"]["kind"] == "hypothesis"
    assert by_ref["annotation-demo-grouped"]["evidence_refs"] == ["run-demo-c"]
    assert all(l["kind"] == "fact" for l in lines if l["record_type"] != "annotation")
    # untested commit: the summary says not_run and the evidence carries no run
    assert "not_run" in by_ref["commit-demo-b"]["summary"]
    assert not any(ref.startswith("run-") for ref in by_ref["commit-demo-b"]["evidence_refs"])
    assert "run-demo-a" in by_ref["commit-demo-a"]["evidence_refs"]
    assert "tested by 1 run(s)" in by_ref["commit-demo-a"]["summary"]
    assert by_ref["run-demo-c-failure"]["summary"].count("compile_error") == 1
    assert "failure_reason: Synthetic compile failure." in by_ref["run-demo-c-failure"]["summary"]
    assert by_ref["decision-demo-blocked"]["evidence_refs"] == ["run-demo-a", "run-demo-baseline"]
    assert by_ref["pr-demo-102"]["evidence_refs"] == ["commit-demo-a"]
    assert "not recorded" not in by_ref["pr-demo-102"]["summary"]
    # no timestamps in any line
    assert "2026-09-08" not in json.dumps(lines)


def test_memory_records_file_is_compact_jsonl_of_the_same_lines(demo_store: MemoryStore) -> None:
    published = traj.publish_trajectory(demo_store, CFG)
    raw = Path(published["memory_records_path"]).read_bytes()
    assert raw == demo_store.read_view(CFG, traj.MEMORY_RECORDS_FILE)
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    file_lines = text.splitlines()
    expected = traj.memory_records(demo_store, CFG)
    assert file_lines == [dumps_compact(line) for line in expected]
    assert [json.loads(l) for l in file_lines] == expected
    # identical facts -> identical bytes on republication
    demo_store.delete_views(CFG)
    assert Path(traj.publish_trajectory(demo_store, CFG)["memory_records_path"]).read_bytes() == raw


def test_memory_records_accepts_prebuilt_scope(demo_store: MemoryStore) -> None:
    scope = traj.collect_config_records(demo_store, CFG)
    assert scope.config_id == CFG
    assert scope.kernel is not None and scope.kernel.record_id == "kernel-demo"
    assert [r.record_id for r in scope.all_records()] == [l["record_ref"] for l in traj.memory_records(demo_store, CFG)]
    assert traj.memory_records(demo_store, CFG, scope) == traj.memory_records(demo_store, CFG)
    assert scope.subject_ids == {"baseline-demo", "commit-demo-a", "commit-demo-b", "commit-demo-c", "commit-demo-a-in-102"}
    assert scope.runs_by_subject()["commit-demo-c"] and [r.record_id for r in scope.runs_by_subject()["commit-demo-c"]] == ["run-demo-c", "run-demo-c-failure"]
    assert "commit-demo-b" not in scope.runs_by_subject()
    assert scope.foreign_runs == []


# --------------------------------------------------------------------------------------
# unknown config
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda s: traj.build_trajectory(s, "cfg-does-not-exist"),
        lambda s: traj.publish_trajectory(s, "cfg-does-not-exist"),
        lambda s: traj.verify_trajectory(s, "cfg-does-not-exist"),
        lambda s: traj.rebuild_trajectory(s, "cfg-does-not-exist"),
        lambda s: traj.memory_records(s, "cfg-does-not-exist"),
        lambda s: traj.collect_config_records(s, "cfg-does-not-exist"),
        lambda s: traj.read_stored_trajectory(s, "cfg-does-not-exist"),
        lambda s: traj.view_path(s, "cfg-does-not-exist"),
    ],
    ids=["build", "publish", "verify", "rebuild", "memory_records", "collect", "read_stored", "view_path"],
)
def test_unknown_config_raises_missing_reference_exit_3(demo_store: MemoryStore, call) -> None:
    with pytest.raises(MissingReferenceError) as excinfo:
        call(demo_store)
    assert excinfo.value.exit_code == 3
    assert excinfo.value.code == "MISSING_REFERENCE"


def test_wrong_type_config_ref_raises_missing_reference(demo_store: MemoryStore) -> None:
    with pytest.raises(MissingReferenceError) as excinfo:
        traj.build_trajectory(demo_store, "pr-demo-101")
    assert excinfo.value.exit_code == 3
