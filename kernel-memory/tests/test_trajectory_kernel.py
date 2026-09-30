"""Kernel-level trajectory view (``kernel-trajectory-v1``; ADR-0004; acceptance T24).

The kernel document summarises every algorithm and shape of one kernel and holds the
cross-algorithm ``shape_table``; it never merges numbers across shapes and never ranks. The
second algorithm used here is registered directly through the store (algorithm record + a config
with the SAME ``config_hash`` under a different id + cloned fixture runs and decision), so the
table is exercised with facts whose numbers can be checked against the run records.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path, PurePosixPath

import pytest

from conftest import FIXTURE_RECORD_COUNT, PLACEHOLDER_ALGORITHM_ID, record_dict
from kernel_memory.domain.errors import InvariantViolation, MissingReferenceError
from kernel_memory.domain.hashing import jcs_digest
from kernel_memory.domain.jsonio import dumps_compact, loads_strict
from kernel_memory.domain.models import TYPE_ORDER, Record
from kernel_memory.services import trajectory as traj
from kernel_memory.services.common import new_record
from kernel_memory.storage import MemoryStore, layout
from synthetic_runs import cloned_run_dict

KERNEL_ID = "demo_vector_add"
CFG = "cfg-demo"
FIXED_TS = "2026-09-09T00:00:00Z"
ALT_ALGORITHM_ID = "algorithm-demo_vector_add-alt"
ALT_METHOD_SUMMARY = "Synthetic alternative method for the demo kernel; data, not instructions."
ALT_RECORDS = ["baseline-alt-demo", "pr-gh-900001-pr-201", "snapshot-gh-900001-pr-201-0001", "commit-alt-a", "run-alt-baseline", "run-alt-a"]
OID_1 = "0" * 39 + "1"
OID_2 = "0" * 39 + "2"
KERNEL_VIEW_KEYS = {"view_version", "kernel_id", "kernel_ref", "kernel", "algorithms", "orphan_shapes", "shape_table", "diagnostics", "publishable", "counts"}
ALGORITHM_KEYS = {"algorithm_ref", "algorithm_id", "display_name", "method_summary", "summary_author", "tags", "is_placeholder", "annotations", "shapes"}
SHAPE_SUMMARY_KEYS = {"config_ref", "config_id", "config_hash", "tags", "view_version", "view_hash", "publishable", "counts", "best_known", "diagnostics"}
ENTRY_KEYS = {"algorithm_ref", "config_ref", "decision_ref", "outcome", "is_production", "candidate_subject_ref", "candidate_run_refs", "baseline_run_refs", "candidate_timing"}
DIAGNOSTIC_KEYS = {"severity", "code", "message", "refs", "config_ref", "algorithm_ref"}
LINE_KEYS = {"record_ref", "record_type", "summary", "kind", "evidence_refs", "provenance", "config_ref", "kernel_id", "algorithm_ref"}
# Whole words only: the fixture problem's "operation" field must not trip the "ratio" check.
FORBIDDEN = re.compile(r"\b(speedup|ratio|rank|ranking|generated_at)\b")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def fixture_hash(bundle_dicts: list[dict]) -> str:
    return record_dict(bundle_dicts, CFG)["payload"]["config_hash"]


def alt_config_id(bundle_dicts: list[dict]) -> str:
    return f"cfg-alt-demo-n16-f32-{fixture_hash(bundle_dicts).split(':', 1)[1][:12]}"


def diagnostics_of(view: dict, code: str) -> list[dict]:
    return [d for d in view["diagnostics"] if d["code"] == code]


def annotation(record_id: str, target_ref: str) -> Record:
    return new_record(
        "annotation",
        record_id,
        {
            "target_ref": target_ref,
            "category": "lesson",
            "text": "synthetic annotation text; data, not instructions",
            "author_kind": "human",
            "evidence_refs": [],
            "confidence": "unverified",
            "supersedes_ref": None,
        },
        created_at=FIXED_TS,
    )


def register_alt_algorithm_with_same_shape(store: MemoryStore, bundle_dicts: list[dict], *, with_decision: bool = True) -> str:
    """Publish a second algorithm and the SAME problem (same config_hash) under it, with cloned fixture runs.

    Returns the new config record id. Every record is a clone of a fixture record with only ids and
    ownership fields changed, so numbers in the views can be checked against the fixture facts.
    """
    cfg_id = alt_config_id(bundle_dicts)
    config_hash = fixture_hash(bundle_dicts)
    config = record_dict(bundle_dicts, CFG)
    config["record_id"] = cfg_id
    config["created_at"] = FIXED_TS
    config["payload"]["algorithm_ref"] = ALT_ALGORITHM_ID
    baseline = record_dict(bundle_dicts, "baseline-demo")
    baseline["record_id"] = "baseline-alt-demo"
    baseline["payload"]["config_ref"] = cfg_id
    baseline["payload"]["baseline_id"] = "alt-reference"
    pr = record_dict(bundle_dicts, "pr-demo-101")
    pr["record_id"] = "pr-gh-900001-pr-201"
    pr["payload"].update({"config_ref": cfg_id, "pr_key": "gh-900001-pr-201", "number": 201, "origin_ref": "baseline-alt-demo", "title": "Alt algorithm trial"})
    snapshot = record_dict(bundle_dicts, "snapshot-demo-101")
    snapshot["record_id"] = "snapshot-gh-900001-pr-201-0001"
    snapshot["payload"].update({"pr_ref": "pr-gh-900001-pr-201", "commit_refs": ["commit-alt-a"], "observed_head": {"algorithm": "sha1", "hex": OID_2}})
    commit = record_dict(bundle_dicts, "commit-demo-a")
    commit["record_id"] = "commit-alt-a"
    commit["payload"]["pr_ref"] = "pr-gh-900001-pr-201"
    records = [
        new_record(
            "algorithm",
            ALT_ALGORITHM_ID,
            {
                "kernel_id": KERNEL_ID,
                "algorithm_id": "alt",
                "display_name": "alt",
                "method_summary": ALT_METHOD_SUMMARY,
                "summary_author": "human",
                "tags": ["test"],
            },
            created_at=FIXED_TS,
        ),
        Record.from_dict(config),
        Record.from_dict(baseline),
        Record.from_dict(pr),
        Record.from_dict(snapshot),
        Record.from_dict(commit),
        Record.from_dict(cloned_run_dict(bundle_dicts, "run-demo-baseline", new_id="run-alt-baseline", subject_ref="baseline-alt-demo", config_ref=cfg_id, config_hash=config_hash)),
        Record.from_dict(cloned_run_dict(bundle_dicts, "run-demo-a", new_id="run-alt-a", subject_ref="commit-alt-a", config_ref=cfg_id, config_hash=config_hash)),
    ]
    if with_decision:
        decision = record_dict(bundle_dicts, "decision-demo-blocked")
        decision["record_id"] = "decision-alt-blocked"
        decision["payload"].update({"config_ref": cfg_id, "candidate_subject_ref": "commit-alt-a", "candidate_run_refs": ["run-alt-a"], "baseline_run_refs": ["run-alt-baseline"]})
        records.append(Record.from_dict(decision))
    store.publish_bundle(records, label="alt-algorithm")
    return cfg_id


def remove_algorithm_file(store: MemoryStore, algorithm_ref: str) -> None:
    """Out-of-band removal (the layout never lets a config with a dangling algorithm_ref be published)."""
    path = store.record_path(algorithm_ref)
    assert path.name == layout.ALGORITHM_FILE
    path.unlink()
    store.invalidate_index()
    assert store.get(algorithm_ref) is None


@pytest.fixture
def kview(demo_store: MemoryStore) -> dict:
    return traj.build_kernel_trajectory(demo_store, KERNEL_ID)


# --------------------------------------------------------------------------------------
# structure on the fixture: one placeholder algorithm, one shape
# --------------------------------------------------------------------------------------
def test_kernel_view_top_level_structure(kview: dict, demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    assert set(kview) == KERNEL_VIEW_KEYS
    assert kview["view_version"] == "kernel-trajectory-v1" == traj.KERNEL_VIEW_VERSION
    assert kview["kernel_id"] == KERNEL_ID and kview["kernel_ref"] == "kernel-demo"
    kernel = record_dict(bundle_dicts, "kernel-demo")["payload"]
    assert kview["kernel"] == {"display_name": kernel["display_name"], "adapter_id": kernel["adapter_id"], "contract_notes": kernel["contract_notes"]}
    assert kview["publishable"] is True
    assert kview["orphan_shapes"] == []
    text = json.dumps(kview)
    assert not FORBIDDEN.search(text)
    assert str(demo_store.root) not in text  # no paths in the body


def test_kernel_view_lists_the_placeholder_algorithm_and_its_shape(kview: dict, demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    assert len(kview["algorithms"]) == 1
    algorithm = kview["algorithms"][0]
    assert set(algorithm) == ALGORITHM_KEYS
    assert algorithm["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID
    assert algorithm["algorithm_id"] == "unspecified" and algorithm["display_name"] == "unspecified"
    assert algorithm["method_summary"] == "unspecified (imported from v0.2)"  # verbatim, never paraphrased
    assert algorithm["summary_author"] == "program" and algorithm["tags"] == ["imported-v02"]
    assert algorithm["is_placeholder"] is True
    assert algorithm["annotations"] == []
    assert len(algorithm["shapes"]) == 1
    shape = algorithm["shapes"][0]
    assert set(shape) == SHAPE_SUMMARY_KEYS
    assert shape["config_ref"] == CFG and shape["config_id"] == "demo-n16-f32"
    assert shape["config_hash"] == fixture_hash(bundle_dicts)
    assert shape["tags"] == record_dict(bundle_dicts, CFG)["payload"]["tags"]
    shape_view = traj.build_trajectory(demo_store, CFG)
    assert shape["view_version"] == "trajectory-v2"
    assert shape["view_hash"] == traj.trajectory_hash(shape_view)
    assert shape["publishable"] is True
    assert shape["counts"] == shape_view["counts"]
    assert shape["best_known"] == shape_view["best_known"] == []
    assert shape["diagnostics"] == shape_view["diagnostics"]
    # the full shape view is NOT embedded (it lives in shapes/<slug>/trajectory.json, linked by view_hash)
    assert "prs" not in shape and "edges" not in shape


def test_shape_table_single_algorithm_row_copies_decision_numbers(kview: dict, demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    assert len(kview["shape_table"]) == 1
    row = kview["shape_table"][0]
    assert set(row) == {"config_hash", "config_id", "problem", "shapes", "groups"}
    assert row["config_hash"] == fixture_hash(bundle_dicts)
    assert row["config_id"] == "demo-n16-f32"
    assert row["problem"] == record_dict(bundle_dicts, CFG)["payload"]["problem"]
    assert row["shapes"] == [{"algorithm_ref": PLACEHOLDER_ALGORITHM_ID, "config_ref": CFG}]
    assert len(row["groups"]) == 1
    group = row["groups"][0]
    decision = demo_store.require("decision-demo-blocked", "decision").payload
    assert set(group) == {"comparison_key", "policy_hash", "policy_id", "entries"}
    assert group["comparison_key"] == decision.comparison_key
    assert group["policy_hash"] == decision.policy_hash
    assert group["policy_id"] == "default-confirm-v1"
    (entry,) = group["entries"]
    assert set(entry) == ENTRY_KEYS
    assert entry["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID and entry["config_ref"] == CFG
    assert entry["decision_ref"] == "decision-demo-blocked" and entry["outcome"] == "blocked" and entry["is_production"] is False
    assert entry["candidate_subject_ref"] == "commit-demo-a"
    assert entry["candidate_run_refs"] == ["run-demo-a"] and entry["baseline_run_refs"] == ["run-demo-baseline"]
    timing = demo_store.require("run-demo-a", "run").payload.timing
    assert entry["candidate_timing"] == {"status": "recorded", "sample_count": 5, "median_us": 90, "p90_us": 90.6}
    assert entry["candidate_timing"] == {"status": timing.status, "sample_count": timing.sample_count, "median_us": timing.median_us, "p90_us": timing.p90_us}


def test_kernel_diagnostics_reemit_shape_diagnostics_and_add_single_algorithm_info(kview: dict, demo_store: MemoryStore) -> None:
    for d in kview["diagnostics"]:
        assert set(d) == DIAGNOSTIC_KEYS
        assert d["refs"] == sorted(set(d["refs"]))
    order = {"error": 0, "warning": 1, "info": 2}
    keys = [(order[d["severity"]], d["code"], d["config_ref"] or "", d["algorithm_ref"] or "", d["message"], d["refs"]) for d in kview["diagnostics"]]
    assert keys == sorted(keys)
    assert len(keys) == len({dumps_compact(d) for d in kview["diagnostics"]})
    shape_view = traj.build_trajectory(demo_store, CFG)
    reemitted = [{k: d[k] for k in ("severity", "code", "message", "refs")} for d in kview["diagnostics"] if d["config_ref"] == CFG and d["code"] != "SINGLE_ALGORITHM_SHAPE"]
    assert sorted(map(dumps_compact, reemitted)) == sorted(map(dumps_compact, shape_view["diagnostics"]))
    assert all(d["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID for d in kview["diagnostics"] if d["config_ref"] == CFG)
    single = diagnostics_of(kview, "SINGLE_ALGORITHM_SHAPE")
    assert len(single) == 1 and single[0]["severity"] == "info"
    assert single[0]["refs"] == [CFG] and single[0]["config_ref"] == CFG and single[0]["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID
    assert not diagnostics_of(kview, "NO_SHARED_COMPARISON_KEY")
    assert kview["counts"]["errors"] == 0


def test_kernel_counts_are_record_counts_never_measurements(kview: dict) -> None:
    counts = kview["counts"]
    assert set(counts) == {
        "algorithms",
        "shapes",
        "orphan_shapes",
        "shape_table_rows",
        "baselines",
        "prs",
        "snapshots",
        "commits",
        "runs",
        "decisions",
        "annotations",
        "untested_commits",
        "diagnostics",
        "errors",
    }
    assert counts["algorithms"] == 1 and counts["shapes"] == 1 and counts["orphan_shapes"] == 0 and counts["shape_table_rows"] == 1
    assert counts["baselines"] == 1 and counts["prs"] == 2 and counts["snapshots"] == 2 and counts["commits"] == 4 and counts["runs"] == 4
    assert counts["decisions"] == 1 and counts["annotations"] == 1 and counts["untested_commits"] == 2
    assert counts["diagnostics"] == len(kview["diagnostics"]) and counts["errors"] == 0


# --------------------------------------------------------------------------------------
# T24: determinism, publication, rebuild (kernel level)
# --------------------------------------------------------------------------------------
def test_t24_kernel_build_twice_is_identical(demo_store: MemoryStore) -> None:
    first = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    second = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    assert first == second
    assert traj.kernel_trajectory_hash(first) == traj.kernel_trajectory_hash(second) == jcs_digest(first)
    assert traj.kernel_trajectory_hash(first).startswith("sha256:")


def test_t24_kernel_import_order_does_not_change_the_view(demo_store: MemoryStore, store: MemoryStore, bundle_records: list[Record]) -> None:
    shuffled = list(bundle_records)
    random.Random(11).shuffle(shuffled)
    assert [r.record_id for r in shuffled] != [r.record_id for r in bundle_records]
    store.publish_bundle(shuffled, label="shuffled")
    assert traj.build_kernel_trajectory(store, KERNEL_ID) == traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    assert traj.kernel_memory_records(store, KERNEL_ID) == traj.kernel_memory_records(demo_store, KERNEL_ID)


def test_kernel_memory_records_lines_and_order(demo_store: MemoryStore) -> None:
    lines = traj.kernel_memory_records(demo_store, KERNEL_ID)
    assert len(lines) == FIXTURE_RECORD_COUNT == 19
    assert [l["record_ref"] for l in lines[:2]] == ["kernel-demo", PLACEHOLDER_ALGORITHM_ID]
    assert lines[0] == {
        "record_ref": "kernel-demo",
        "record_type": "kernel",
        "summary": lines[0]["summary"],
        "kind": "fact",
        "evidence_refs": [],
        "provenance": None,
        "config_ref": None,
        "kernel_id": KERNEL_ID,
        "algorithm_ref": None,
    }
    assert lines[0]["summary"].startswith("Kernel demo_vector_add (")
    assert lines[1]["record_type"] == "algorithm" and lines[1]["config_ref"] is None and lines[1]["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID
    assert lines[1]["kind"] == "fact" and lines[1]["evidence_refs"] == [] and lines[1]["provenance"] is None
    assert "unspecified (imported from v0.2)" in lines[1]["summary"]  # method_summary is data in the line
    assert lines[2:] == traj.memory_records(demo_store, CFG)
    for line in lines:
        assert set(line) == LINE_KEYS
        assert line["kernel_id"] == KERNEL_ID
    keys = [(l["algorithm_ref"] or "", l["config_ref"] or "", TYPE_ORDER[l["record_type"]], l["record_ref"]) for l in lines]
    assert keys == sorted(keys)
    assert "2026-09-08" not in json.dumps(lines)
    # a prebuilt scope yields the same lines
    scope = traj.collect_kernel_records(demo_store, KERNEL_ID)
    assert traj.kernel_memory_records(demo_store, KERNEL_ID, scope) == lines


def test_t24_kernel_delete_trajectory_and_cache_then_rebuild_identically(demo_store: MemoryStore) -> None:
    published = traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    assert set(published) == {"kernel_id", "kernel_ref", "path", "view_hash", "memory_records_path", "record_count", "shapes", "publishable", "forced", "diagnostics"}
    assert published["kernel_id"] == KERNEL_ID and published["kernel_ref"] == "kernel-demo"
    assert published["publishable"] is True and published["forced"] is False
    assert published["record_count"] == 19
    view_file = Path(published["path"])
    records_file = Path(published["memory_records_path"])
    trajectory_dir = demo_store.root / "kernels" / KERNEL_ID / "trajectory"
    assert view_file == trajectory_dir / "trajectory.json" == traj.kernel_view_path(demo_store, KERNEL_ID)
    assert records_file == trajectory_dir / "memory_records.jsonl"
    assert view_file.is_file() and records_file.is_file()
    assert published["shapes"] == [
        {
            "algorithm_ref": PLACEHOLDER_ALGORITHM_ID,
            "config_ref": CFG,
            "shape_view_hash": traj.trajectory_hash(traj.build_trajectory(demo_store, CFG)),
            "publishable": True,
            "path": str(trajectory_dir / "shapes" / "cfg-demo" / "trajectory.json"),
        }
    ]
    assert (trajectory_dir / "shapes" / "cfg-demo" / "memory_records.jsonl").is_file()
    original_hash = published["view_hash"]
    original_lines = records_file.read_bytes()
    assert original_lines.decode("utf-8").count("\n") == 19
    assert [json.loads(l) for l in original_lines.decode("utf-8").splitlines()] == traj.kernel_memory_records(demo_store, KERNEL_ID)
    stored = traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID)
    assert set(stored) == {"generated_at", "view_hash", "view", "forced"}
    assert stored["view_hash"] == original_hash == jcs_digest(stored["view"])
    assert stored["view"] == traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    report = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert report["matches"] is True and report["stored_view_consistent"] is True
    assert report["shapes"] == [{"config_ref": CFG, "stored_hash": published["shapes"][0]["shape_view_hash"], "rebuilt_hash": published["shapes"][0]["shape_view_hash"], "matches": True}]

    # T24: delete the generated kernel subtree and the disposable cache
    demo_store.rebuild_index()
    cache_dir = demo_store.root / layout.CACHE_DIR
    assert cache_dir.is_dir()
    shutil.rmtree(trajectory_dir)
    shutil.rmtree(cache_dir)
    assert not trajectory_dir.exists() and not cache_dir.exists()
    assert traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID) is None
    assert traj.read_stored_trajectory(demo_store, CFG) is None
    missing = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert missing["stored"] is False and missing["matches"] is False and missing["stored_view_consistent"] is None
    assert missing["rebuilt_hash"] == original_hash
    assert missing["shapes"][0]["matches"] is False and missing["shapes"][0]["stored_hash"] is None

    rebuilt = traj.rebuild_kernel_trajectory(demo_store, KERNEL_ID)
    assert rebuilt["view_hash"] == original_hash
    assert rebuilt["forced"] is False and rebuilt["removed_views"] == []
    assert Path(rebuilt["memory_records_path"]).read_bytes() == original_lines
    assert traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID)["view"] == stored["view"]
    assert traj.verify_kernel_trajectory(demo_store, KERNEL_ID)["matches"] is True
    assert traj.verify_trajectory(demo_store, CFG)["matches"] is True
    # a second rebuild removes every generated file first (kernel document + shape views)
    again = traj.rebuild_kernel_trajectory(demo_store, KERNEL_ID)
    assert sorted(again["removed_views"]) == [
        f"kernels/{KERNEL_ID}/trajectory/memory_records.jsonl",
        f"kernels/{KERNEL_ID}/trajectory/shapes/cfg-demo/memory_records.jsonl",
        f"kernels/{KERNEL_ID}/trajectory/shapes/cfg-demo/trajectory.json",
        f"kernels/{KERNEL_ID}/trajectory/trajectory.json",
    ]
    assert again["view_hash"] == original_hash
    # the generated subtree is never indexed as records
    demo_store.invalidate_index()
    assert len(demo_store.index_entries()) == 19


def test_verify_kernel_detects_tampering_and_staleness(demo_store: MemoryStore) -> None:
    published = traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    path = Path(published["path"])
    document = loads_strict(path.read_bytes())
    document["view"]["counts"]["shapes"] = 7
    path.write_text(json.dumps(document), encoding="utf-8")
    tampered = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert tampered["stored_view_consistent"] is False and tampered["matches"] is False
    assert tampered["shapes"][0]["matches"] is True  # the shape file itself was not touched

    traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    assert traj.verify_kernel_trajectory(demo_store, KERNEL_ID)["matches"] is True
    # a new fact on the shape makes both the shape view and the kernel document stale
    demo_store.publish(annotation("annotation-test-stale-shape", "commit-demo-b"))
    stale = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert stale["stored"] is True and stale["matches"] is False and stale["stored_view_consistent"] is True
    assert stale["stored_hash"] != stale["rebuilt_hash"]
    assert stale["shapes"][0]["matches"] is False
    traj.rebuild_kernel_trajectory(demo_store, KERNEL_ID)
    fresh = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert fresh["matches"] is True and fresh["stored_hash"] == stale["rebuilt_hash"]
    # a new fact on the algorithm makes only the kernel document stale; the shape view is unchanged
    demo_store.publish(annotation("annotation-test-on-algorithm", PLACEHOLDER_ALGORITHM_ID))
    stale_kernel = traj.verify_kernel_trajectory(demo_store, KERNEL_ID)
    assert stale_kernel["matches"] is False and stale_kernel["shapes"][0]["matches"] is True
    rebuilt = traj.rebuild_kernel_trajectory(demo_store, KERNEL_ID)
    view = traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID)["view"]
    assert [a["annotation_ref"] for a in view["algorithms"][0]["annotations"]] == ["annotation-test-on-algorithm"]
    assert view["algorithms"][0]["annotations"][0]["kind"] == "hypothesis"
    assert view["counts"]["annotations"] == 3  # 2 on the shape + 1 on the algorithm
    assert view["algorithms"][0]["shapes"][0]["counts"]["annotations"] == 2
    assert rebuilt["view_hash"] == traj.verify_kernel_trajectory(demo_store, KERNEL_ID)["stored_hash"]


# --------------------------------------------------------------------------------------
# a second algorithm on the same shape: one shape_table row, two entries, no aggregates
# --------------------------------------------------------------------------------------
def test_second_algorithm_on_the_same_shape_yields_one_row_with_two_entries(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    alt_cfg = register_alt_algorithm_with_same_shape(demo_store, bundle_dicts)
    assert demo_store.require(alt_cfg, "config").payload.config_hash == fixture_hash(bundle_dicts)  # same problem
    assert demo_store.record_path(alt_cfg) == demo_store.root / "kernels" / KERNEL_ID / "alt" / alt_cfg / "config.json"
    kview = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    assert kview["publishable"] is True
    assert kview["counts"]["algorithms"] == 2 and kview["counts"]["shapes"] == 2 and kview["counts"]["shape_table_rows"] == 1
    assert kview["counts"]["runs"] == 6 and kview["counts"]["decisions"] == 2
    assert [a["algorithm_ref"] for a in kview["algorithms"]] == [ALT_ALGORITHM_ID, PLACEHOLDER_ALGORITHM_ID]
    alt = kview["algorithms"][0]
    assert alt["is_placeholder"] is False and alt["method_summary"] == ALT_METHOD_SUMMARY and alt["summary_author"] == "human"
    assert [s["config_ref"] for s in alt["shapes"]] == [alt_cfg]
    assert alt["shapes"][0]["view_hash"] == traj.trajectory_hash(traj.build_trajectory(demo_store, alt_cfg))
    assert alt["shapes"][0]["view_hash"] != kview["algorithms"][1]["shapes"][0]["view_hash"]

    (row,) = kview["shape_table"]
    assert row["config_hash"] == fixture_hash(bundle_dicts)
    assert row["shapes"] == [{"algorithm_ref": ALT_ALGORITHM_ID, "config_ref": alt_cfg}, {"algorithm_ref": PLACEHOLDER_ALGORITHM_ID, "config_ref": CFG}]
    (group,) = row["groups"]
    assert group["comparison_key"] == demo_store.require("run-demo-a").payload.comparison_key == demo_store.require("run-alt-a").payload.comparison_key
    assert [(e["algorithm_ref"], e["config_ref"], e["decision_ref"]) for e in group["entries"]] == [
        (ALT_ALGORITHM_ID, alt_cfg, "decision-alt-blocked"),
        (PLACEHOLDER_ALGORITHM_ID, CFG, "decision-demo-blocked"),
    ]
    for entry in group["entries"]:
        assert set(entry) == ENTRY_KEYS
        run = demo_store.require(entry["candidate_run_refs"][0], "run").payload
        assert entry["candidate_timing"] == {"status": run.timing.status, "sample_count": run.timing.sample_count, "median_us": run.timing.median_us, "p90_us": run.timing.p90_us}
        assert entry["outcome"] == "blocked" and entry["is_production"] is False
    assert group["entries"][0]["candidate_run_refs"] == ["run-alt-a"] and group["entries"][0]["baseline_run_refs"] == ["run-alt-baseline"]
    # side by side is all the table does: no aggregate, ranking or speedup anywhere
    text = json.dumps(kview)
    assert not FORBIDDEN.search(text)
    assert not diagnostics_of(kview, "SINGLE_ALGORITHM_SHAPE")
    assert not diagnostics_of(kview, "NO_SHARED_COMPARISON_KEY")
    assert not diagnostics_of(kview, "DUPLICATE_SHAPE_IN_ALGORITHM")

    lines = traj.kernel_memory_records(demo_store, KERNEL_ID)
    assert len(lines) == 19 + 1 + 8  # + alt algorithm line + 8 alt shape records
    assert [l["record_ref"] for l in lines[:2]] == ["kernel-demo", ALT_ALGORITHM_ID]
    assert {l["config_ref"] for l in lines if l["algorithm_ref"] == ALT_ALGORITHM_ID} == {None, alt_cfg}
    assert {l["record_ref"] for l in lines if l["config_ref"] == alt_cfg} == {alt_cfg, "decision-alt-blocked", *ALT_RECORDS}

    published = traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    assert [(s["algorithm_ref"], s["config_ref"]) for s in published["shapes"]] == [(ALT_ALGORITHM_ID, alt_cfg), (PLACEHOLDER_ALGORITHM_ID, CFG)]
    assert all(Path(s["path"]).is_file() for s in published["shapes"])
    assert published["record_count"] == 28
    assert traj.verify_kernel_trajectory(demo_store, KERNEL_ID)["matches"] is True


def test_no_shared_comparison_key_is_informational(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    alt_cfg = register_alt_algorithm_with_same_shape(demo_store, bundle_dicts, with_decision=False)
    kview = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    (row,) = kview["shape_table"]
    assert len(row["shapes"]) == 2
    (group,) = row["groups"]
    assert [e["algorithm_ref"] for e in group["entries"]] == [PLACEHOLDER_ALGORITHM_ID]
    info = diagnostics_of(kview, "NO_SHARED_COMPARISON_KEY")
    assert len(info) == 1 and info[0]["severity"] == "info"
    assert info[0]["refs"] == sorted([alt_cfg, CFG])
    assert info[0]["config_ref"] is None and info[0]["algorithm_ref"] is None
    assert not diagnostics_of(kview, "SINGLE_ALGORITHM_SHAPE")
    assert kview["publishable"] is True


def test_duplicate_shape_in_algorithm_is_an_error(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    copy_id = "cfg-unspecified-demo-n16-f32-copy"
    config = record_dict(bundle_dicts, CFG)
    config["record_id"] = copy_id
    demo_store.publish(Record.from_dict(config))  # same problem, same algorithm, second id
    kview = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    errors = diagnostics_of(kview, "DUPLICATE_SHAPE_IN_ALGORITHM")
    assert len(errors) == 1 and errors[0]["severity"] == "error"
    assert errors[0]["refs"] == [CFG, copy_id] and errors[0]["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID
    assert kview["publishable"] is False
    assert [s["config_ref"] for s in kview["algorithms"][0]["shapes"]] == [CFG, copy_id]
    (row,) = kview["shape_table"]
    assert row["shapes"] == [{"algorithm_ref": PLACEHOLDER_ALGORITHM_ID, "config_ref": CFG}, {"algorithm_ref": PLACEHOLDER_ALGORITHM_ID, "config_ref": copy_id}]
    with pytest.raises(InvariantViolation) as excinfo:
        traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    assert excinfo.value.code == "TRAJECTORY_NOT_PUBLISHABLE"


# --------------------------------------------------------------------------------------
# orphan shapes (algorithm removed out of band): error, not publishable, force writes
# --------------------------------------------------------------------------------------
def test_orphan_shape_blocks_publication_unless_forced(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    alt_cfg = register_alt_algorithm_with_same_shape(demo_store, bundle_dicts)
    remove_algorithm_file(demo_store, ALT_ALGORITHM_ID)
    scope = traj.collect_kernel_records(demo_store, KERNEL_ID)
    assert [a.record_id for a in scope.algorithms] == [PLACEHOLDER_ALGORITHM_ID]
    assert [s.config_id for s in scope.orphan_shapes] == [alt_cfg]
    assert scope.orphan_reasons == {alt_cfg: "MISSING_ALGORITHM"}
    assert [s.config_id for s in scope.all_shapes()] == sorted([alt_cfg, CFG])

    kview = traj.build_kernel_trajectory(demo_store, KERNEL_ID)
    assert kview["orphan_shapes"] == [{"config_ref": alt_cfg, "config_hash": fixture_hash(bundle_dicts), "algorithm_ref": ALT_ALGORITHM_ID, "reason": "MISSING_ALGORITHM"}]
    errors = diagnostics_of(kview, "MISSING_ALGORITHM")
    assert len(errors) == 1 and errors[0]["severity"] == "error"
    assert errors[0]["config_ref"] == alt_cfg and errors[0]["algorithm_ref"] == ALT_ALGORITHM_ID
    assert errors[0]["refs"] == sorted([alt_cfg, ALT_ALGORITHM_ID])
    # the orphan's own view diagnostics are re-emitted too (its algorithm_ref is a missing reference)
    assert any(d["code"] == "MISSING_REFERENCE" and d["config_ref"] == alt_cfg for d in kview["diagnostics"])
    assert kview["publishable"] is False
    assert kview["counts"] == {**kview["counts"], "algorithms": 1, "shapes": 1, "orphan_shapes": 1, "shape_table_rows": 1, "runs": 6}
    assert kview["counts"]["errors"] >= 2
    (row,) = kview["shape_table"]
    assert row["shapes"] == [{"algorithm_ref": PLACEHOLDER_ALGORITHM_ID, "config_ref": CFG}]  # orphans never enter the table
    assert diagnostics_of(kview, "SINGLE_ALGORITHM_SHAPE")

    with pytest.raises(InvariantViolation) as excinfo:
        traj.publish_kernel_trajectory(demo_store, KERNEL_ID)
    assert excinfo.value.code == "TRAJECTORY_NOT_PUBLISHABLE" and excinfo.value.exit_code == 2
    assert excinfo.value.details["kernel_id"] == KERNEL_ID
    assert excinfo.value.details["config_refs"] == [alt_cfg]
    assert {e["code"] for e in excinfo.value.details["errors"]} == {"MISSING_ALGORITHM", "MISSING_REFERENCE"}
    assert traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID) is None
    with pytest.raises(InvariantViolation):
        traj.rebuild_kernel_trajectory(demo_store, KERNEL_ID)

    forced = traj.publish_kernel_trajectory(demo_store, KERNEL_ID, force=True)
    assert forced["forced"] is True and forced["publishable"] is False
    assert Path(forced["path"]).is_file()
    assert [(s["config_ref"], s["publishable"]) for s in forced["shapes"]] == [(alt_cfg, False), (CFG, True)]
    assert all(Path(s["path"]).is_file() for s in forced["shapes"])
    assert forced["record_count"] == 19 + 8  # the orphan's 8 records are still listed (with its stored algorithm_ref); no alt algorithm line
    stored = traj.read_stored_kernel_trajectory(demo_store, KERNEL_ID)
    assert stored["forced"] is True and stored["view"]["publishable"] is False
    assert traj.verify_kernel_trajectory(demo_store, KERNEL_ID)["matches"] is True


# --------------------------------------------------------------------------------------
# per-shape publication lives under the kernel's trajectory/ subtree
# --------------------------------------------------------------------------------------
def test_publish_trajectory_writes_under_trajectory_shapes_and_returns_config_ref(demo_store: MemoryStore) -> None:
    result = traj.publish_trajectory(demo_store, CFG)
    assert result["config_ref"] == CFG
    path = Path(result["path"])
    rel = path.relative_to(demo_store.root)
    assert rel.as_posix() == f"kernels/{KERNEL_ID}/trajectory/shapes/cfg-demo/trajectory.json"
    assert layout.is_view_path(PurePosixPath(rel.as_posix()))
    assert Path(result["memory_records_path"]).relative_to(demo_store.root).as_posix() == f"kernels/{KERNEL_ID}/trajectory/shapes/cfg-demo/memory_records.jsonl"
    assert not (demo_store.config_dir(CFG) / "trajectory.json").exists()
    assert traj.view_path(demo_store, CFG) == path
    demo_store.invalidate_index()
    assert len(demo_store.index_entries()) == 19  # views are never records


# --------------------------------------------------------------------------------------
# unknown kernel
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda s: traj.build_kernel_trajectory(s, "no_such_kernel"),
        lambda s: traj.publish_kernel_trajectory(s, "no_such_kernel"),
        lambda s: traj.verify_kernel_trajectory(s, "no_such_kernel"),
        lambda s: traj.rebuild_kernel_trajectory(s, "no_such_kernel"),
        lambda s: traj.kernel_memory_records(s, "no_such_kernel"),
        lambda s: traj.collect_kernel_records(s, "no_such_kernel"),
        lambda s: traj.read_stored_kernel_trajectory(s, "no_such_kernel"),
        lambda s: traj.kernel_view_path(s, "no_such_kernel"),
    ],
    ids=["build", "publish", "verify", "rebuild", "memory_records", "collect", "read_stored", "view_path"],
)
def test_unknown_kernel_raises_missing_reference_exit_3(demo_store: MemoryStore, call) -> None:
    with pytest.raises(MissingReferenceError) as excinfo:
        call(demo_store)
    assert excinfo.value.exit_code == 3
    assert excinfo.value.code == "MISSING_REFERENCE"
