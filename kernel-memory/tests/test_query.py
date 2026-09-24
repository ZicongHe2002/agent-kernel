"""Structured retrieval (specification section 14; T05 untested commits)."""
from __future__ import annotations

import json

import pytest

from conftest import record_dict
from kernel_memory.domain.errors import InputError, MissingReferenceError
from kernel_memory.domain.models import TYPE_ORDER
from kernel_memory.domain.problems import default_registry
from kernel_memory.services.common import new_record
from kernel_memory.services.query import (
    CROSS_CONFIG_HINT_FIELD,
    REASON_NO_RUN_RECORDED,
    QueryFilters,
    query_memory,
    untested_commits,
)
from kernel_memory.storage import MemoryStore

KERNEL_ID = "demo_vector_add"
CFG_DEMO = "cfg-demo"
FIXED_TS = "2026-09-09T00:00:00Z"
OTHER_OID = "1234567890abcdef1234567890abcdef12345678"


def _ids(result) -> list[str]:
    return [item["record_ref"] for item in result.items]


def _hint_ids(result) -> list[str]:
    return [item["record_ref"] for item in result.cross_config_hints]


def _publish_kernel(store: MemoryStore) -> str:
    record = new_record(
        "kernel",
        f"kernel-{KERNEL_ID}",
        {
            "kernel_id": KERNEL_ID,
            "display_name": "Synthetic vector add",
            "adapter_id": "mock-v1",
            "contract_notes": "Synthetic test kernel; nothing here is measured.",
        },
        created_at=FIXED_TS,
    )
    store.publish(record)
    return record.record_id


def _publish_second_config(store: MemoryStore, *, component: str = "tiling", key: str = "block_q") -> tuple[str, str, str]:
    """A second config of the same kernel (n=32) with one local PR and one untested commit under it.

    Identifiers follow DESIGN section 4; the config identity comes from the problem normalizer, so the
    hash and schema digest are the registry's, never typed by hand.
    """
    normalized = default_registry().normalize(KERNEL_ID, {"n": 32})
    config_id = f"cfg-{normalized.config_id_hint}-{normalized.config_hash.split(':', 1)[1][:12]}"
    config = new_record(
        "config",
        config_id,
        {
            "kernel_id": KERNEL_ID,
            "config_id": normalized.config_id_hint,
            "problem_schema_id": normalized.problem_schema_id,
            "problem_schema_digest": normalized.problem_schema_digest,
            "problem": normalized.problem,
            "config_hash": normalized.config_hash,
            "tags": ["fixture", "not-measured"],
        },
        created_at=FIXED_TS,
    )
    pr_key = "local-tiling-on-n32-0badc0de"
    pr_id = f"pr-{pr_key}"
    pr = new_record(
        "pr",
        pr_id,
        {
            "config_ref": config_id,
            "pr_key": pr_key,
            "repo_uid": "local:demo-workspace",
            "provider": "local",
            "number": None,
            "title": "Local trial: tiling on n=32",
            "hypothesis": None,
            "origin_ref": None,
        },
        created_at=FIXED_TS,
    )
    commit_id = f"commit-{pr_key}-{OTHER_OID[:12]}"
    commit = new_record(
        "commit",
        commit_id,
        {
            "pr_ref": pr_id,
            "repo_uid": "local:demo-workspace",
            "commit_oid": {"algorithm": "sha1", "hex": OTHER_OID},
            "git_parent_oids": [],
            "diff_base_oid": None,
            "source_available": False,
            "change_status": "recorded",
            "changes": [
                {
                    "change_id": "chg-1",
                    "component": component,
                    "key": key,
                    "before": 64,
                    "after": 128,
                    "rationale": "synthetic change for retrieval tests",
                    "extraction_source": "explicit",
                    "attribution": "group_only",
                }
            ],
            "summary": "Synthetic commit; no run recorded.",
            "summary_author": "collector",
            "diff_artifact_ref": None,
        },
        created_at=FIXED_TS,
    )
    store.publish_bundle([config, pr, commit], label="second-config")
    return config_id, pr_id, commit_id


# --------------------------------------------------------------------------------------
# Change-level filters
# --------------------------------------------------------------------------------------
def test_component_tiling_matches_only_commit_a(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(component="tiling"))
    assert _ids(result) == ["commit-demo-a"]
    (item,) = result.items
    assert item["record_type"] == "commit"
    assert item["config_ref"] == CFG_DEMO
    assert [c["component"] for c in item["changes"]] == ["tiling"]
    assert result.total == 1 and result.truncated is False


def test_component_pipeline_matches_only_commit_c(demo_store: MemoryStore) -> None:
    assert _ids(query_memory(demo_store, QueryFilters(component="pipeline"))) == ["commit-demo-c"]


def test_parameter_key_layout_id_matches_only_commit_c(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(parameter_key="layout_id"))
    assert _ids(result) == ["commit-demo-c"]
    assert {c["key"] for c in result.items[0]["changes"]} == {"stages", "layout_id"}


def test_component_filter_excludes_every_non_commit_record(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(component="tiling", record_type="run"))
    assert result.items == [] and result.total == 0


# --------------------------------------------------------------------------------------
# T05: untested commits are derived from absence, never stored
# --------------------------------------------------------------------------------------
def test_t05_not_run_commits_are_derived_from_absence(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(run_status="not_run"))
    assert set(_ids(result)) == {"commit-demo-b", "commit-demo-a-in-102"}
    for item in result.items:
        assert item["status"] == "not_run"
        assert item["run_refs"] == []
    # No fake run exists for them anywhere in the store.
    assert not [r for r in demo_store.records("run") if r.payload.subject_ref in {"commit-demo-b", "commit-demo-a-in-102"}]


def test_t05_untested_commits_lists_both_with_reason(demo_store: MemoryStore) -> None:
    listed = untested_commits(demo_store, CFG_DEMO)
    assert [c["commit_ref"] for c in listed] == ["commit-demo-a-in-102", "commit-demo-b"]
    assert {c["reason"] for c in listed} == {REASON_NO_RUN_RECORDED}
    by_ref = {c["commit_ref"]: c for c in listed}
    assert by_ref["commit-demo-b"]["pr_ref"] == "pr-demo-101"
    assert by_ref["commit-demo-a-in-102"]["pr_ref"] == "pr-demo-102"
    assert by_ref["commit-demo-b"]["commit_oid"] == demo_store.require("commit-demo-b").payload.commit_oid.hex


def test_t05_tested_commit_carries_its_run_refs(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(run_status="tested"))
    assert set(_ids(result)) == {"commit-demo-a", "commit-demo-c"}
    by_ref = {i["record_ref"]: i for i in result.items}
    assert by_ref["commit-demo-a"]["status"] == "tested"
    assert by_ref["commit-demo-a"]["run_refs"] == ["run-demo-a"]
    assert by_ref["commit-demo-c"]["run_refs"] == ["run-demo-c", "run-demo-c-failure"]


# --------------------------------------------------------------------------------------
# Run, decision, subject and PR filters
# --------------------------------------------------------------------------------------
def test_record_type_run_with_fixture_provenance_returns_all_four_runs(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(record_type="run", provenance="fixture"))
    assert _ids(result) == ["run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure"]
    assert all(i["provenance"] == "fixture" for i in result.items)
    # Uncollected timing stays null with a status; it is never reported as 0.
    failure = next(i for i in result.items if i["record_ref"] == "run-demo-c-failure")
    assert failure["timing"]["status"] == "not_run" and failure["timing"]["median_us"] is None
    assert failure["correctness"]["status"] == "not_run"


def test_execution_status_compile_error(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(execution_status="compile_error"))
    assert _ids(result) == ["run-demo-c-failure"]
    assert result.items[0]["subject_ref"] == "commit-demo-c"
    assert result.items[0]["pr_ref"] == "pr-demo-102"


def test_correctness_status_pass_returns_the_three_passing_runs(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(correctness_status="pass"))
    assert _ids(result) == ["run-demo-a", "run-demo-baseline", "run-demo-c"]


def test_decision_outcome_blocked(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(decision_outcome="blocked"))
    assert _ids(result) == ["decision-demo-blocked"]
    (item,) = result.items
    assert item["is_production"] is False
    assert item["candidate_subject_ref"] == "commit-demo-a"
    assert "FIXTURE_NOT_ELIGIBLE" in item["reason_codes"]


def test_reason_code_fixture_not_eligible(demo_store: MemoryStore) -> None:
    assert _ids(query_memory(demo_store, QueryFilters(reason_code="FIXTURE_NOT_ELIGIBLE"))) == ["decision-demo-blocked"]
    assert _ids(query_memory(demo_store, QueryFilters(reason_code="BASELINE_DRIFT"))) == []


def test_comparison_key_applies_to_runs_and_decisions(demo_store: MemoryStore) -> None:
    key = demo_store.require("run-demo-a").payload.comparison_key
    result = query_memory(demo_store, QueryFilters(comparison_key=key))
    assert set(_ids(result)) == {"run-demo-a", "run-demo-baseline", "run-demo-c", "run-demo-c-failure", "decision-demo-blocked"}
    assert _ids(query_memory(demo_store, QueryFilters(comparison_key="sha256:" + "0" * 64))) == []


def test_subject_ref_commit_c_returns_runs_and_annotation(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(subject_ref="commit-demo-c"))
    assert _ids(result) == ["run-demo-c", "run-demo-c-failure", "annotation-demo-grouped"]
    annotation = result.items[-1]
    assert annotation["target_ref"] == "commit-demo-c"
    assert annotation["kind"] == "hypothesis"  # unverified lesson, not a program-authored fact


def test_subject_ref_matches_decisions_by_candidate(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(subject_ref="commit-demo-a"))
    assert _ids(result) == ["run-demo-a", "decision-demo-blocked"]


def test_pr_ref_102_returns_commits_snapshot_and_runs(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(pr_ref="pr-demo-102"))
    assert _ids(result) == ["snapshot-demo-102", "commit-demo-a-in-102", "commit-demo-c", "run-demo-c", "run-demo-c-failure"]
    snapshot = result.items[0]
    assert snapshot["record_type"] == "pr_snapshot"
    assert snapshot["commit_refs"] == ["commit-demo-a-in-102", "commit-demo-c"]


def test_pr_ref_101_runs_follow_their_subject_commit(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(pr_ref="pr-demo-101", record_type="run"))
    assert _ids(result) == ["run-demo-a"]


# --------------------------------------------------------------------------------------
# Scope resolution
# --------------------------------------------------------------------------------------
def test_config_hash_scope_equals_config_ref_scope(demo_store: MemoryStore) -> None:
    config_hash = demo_store.require(CFG_DEMO, "config").payload.config_hash
    by_ref = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO)).to_dict()
    by_hash = query_memory(demo_store, QueryFilters(config_hash=config_hash)).to_dict()
    assert by_hash["configs"] == [CFG_DEMO]
    assert by_ref["filters"] != by_hash["filters"]
    by_ref.pop("filters")
    by_hash.pop("filters")
    assert by_ref == by_hash


def test_kernel_id_scope(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(kernel_id=KERNEL_ID))
    assert result.configs == [CFG_DEMO]
    assert result.total == 18
    assert _ids(result)[:2] == ["kernel-demo", CFG_DEMO]
    assert query_memory(demo_store, QueryFilters(kernel_id="no_such_kernel")).to_dict()["items"] == []


def test_conflicting_scope_filters_yield_an_empty_scope(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, kernel_id="another_kernel"))
    assert result.configs == [] and result.items == [] and result.total == 0


def test_unknown_config_ref_raises_missing_reference_exit_3(demo_store: MemoryStore) -> None:
    with pytest.raises(MissingReferenceError) as info:
        query_memory(demo_store, QueryFilters(config_ref="cfg-does-not-exist"))
    assert info.value.exit_code == 3
    # An existing id of the wrong type is a missing *config* reference too.
    with pytest.raises(MissingReferenceError) as info2:
        query_memory(demo_store, QueryFilters(config_ref="kernel-demo"))
    assert info2.value.exit_code == 3


def test_empty_store_returns_an_empty_result(store: MemoryStore) -> None:
    result = query_memory(store, QueryFilters())
    assert result.to_dict() == {
        "items": [],
        "cross_config_hints": [],
        "total": 0,
        "truncated": False,
        "index_used": False,
        "filters": QueryFilters().to_dict(),
        "configs": [],
    }


# --------------------------------------------------------------------------------------
# Ordering, limit, serialisation
# --------------------------------------------------------------------------------------
def test_items_sorted_by_type_order_then_record_id(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters())
    keys = [(TYPE_ORDER[i["record_type"]], i["record_ref"]) for i in result.items]
    assert keys == sorted(keys)
    assert len(set(_ids(result))) == len(result.items) == 18


def test_limit_truncates_items_but_not_total(demo_store: MemoryStore) -> None:
    full = query_memory(demo_store, QueryFilters())
    limited = query_memory(demo_store, QueryFilters(limit=3))
    assert limited.truncated is True
    assert limited.total == full.total == 18
    assert limited.items == full.items[:3]
    exact = query_memory(demo_store, QueryFilters(limit=full.total))
    assert exact.truncated is False and len(exact.items) == full.total


def test_result_to_dict_is_json_serialisable_and_carries_filters(demo_store: MemoryStore) -> None:
    filters = QueryFilters(config_ref=CFG_DEMO, record_type="run", limit=2)
    data = query_memory(demo_store, filters).to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["filters"] == filters.to_dict()
    assert set(data) == {"items", "cross_config_hints", "total", "truncated", "index_used", "filters", "configs"}
    assert all(set(item) >= {"record_ref", "record_type", "config_ref"} for item in data["items"])


def test_query_filters_has_exactly_the_cli_fields() -> None:
    import dataclasses

    names = [f.name for f in dataclasses.fields(QueryFilters)]
    assert names == [
        "kernel_id",
        "config_ref",
        "config_hash",
        "record_type",
        "component",
        "parameter_key",
        "subject_ref",
        "pr_ref",
        "execution_status",
        "correctness_status",
        "provenance",
        "comparison_key",
        "decision_outcome",
        "reason_code",
        "run_status",
        "include_cross_config_hints",
        "limit",
    ]
    defaults = QueryFilters()
    assert defaults.include_cross_config_hints is False
    assert all(getattr(defaults, n) is None for n in names if n != "include_cross_config_hints")
    with pytest.raises(dataclasses.FrozenInstanceError):
        defaults.limit = 5  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Validation (exit code 2)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filters",
    [
        QueryFilters(record_type="snapshot"),
        QueryFilters(record_type="runs"),
        QueryFilters(run_status="pending"),
        QueryFilters(limit=0),
        QueryFilters(limit=-1),
    ],
    ids=["bad-record-type", "plural-record-type", "bad-run-status", "limit-zero", "limit-negative"],
)
def test_validate_rejects_bad_filters_with_exit_2(demo_store: MemoryStore, filters: QueryFilters) -> None:
    with pytest.raises(InputError) as info:
        filters.validate()
    assert info.value.exit_code == 2
    with pytest.raises(InputError) as info2:
        query_memory(demo_store, filters)
    assert info2.value.exit_code == 2


def test_validate_accepts_every_record_type_and_run_status() -> None:
    for record_type in TYPE_ORDER:
        QueryFilters(record_type=record_type).validate()
    QueryFilters(run_status="tested").validate()
    QueryFilters(run_status="not_run").validate()
    QueryFilters(limit=1).validate()


# --------------------------------------------------------------------------------------
# Cross-config hints
# --------------------------------------------------------------------------------------
def test_cross_config_hints_are_labelled_and_never_mixed_into_items(demo_store: MemoryStore) -> None:
    other_cfg, other_pr, other_commit = _publish_second_config(demo_store)
    assert demo_store.require(other_cfg, "config").payload.problem["n"] == 32

    by_component = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_config_hints=True))
    assert _ids(by_component) == ["commit-demo-a"]
    assert _hint_ids(by_component) == [other_commit]
    (hint,) = by_component.cross_config_hints
    assert hint[CROSS_CONFIG_HINT_FIELD] is True
    assert hint["config_ref"] == other_cfg
    assert hint["status"] == "not_run" and hint["run_refs"] == []
    assert all(CROSS_CONFIG_HINT_FIELD not in item for item in by_component.items)
    assert by_component.total == 1  # hints never count towards the item total

    by_type = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, record_type="commit", include_cross_config_hints=True))
    assert other_commit not in _ids(by_type)
    assert set(_ids(by_type)) == {"commit-demo-a", "commit-demo-b", "commit-demo-c", "commit-demo-a-in-102"}
    assert _hint_ids(by_type) == [other_commit]

    by_pr = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, record_type="pr", include_cross_config_hints=True))
    assert _ids(by_pr) == ["pr-demo-101", "pr-demo-102"]
    assert _hint_ids(by_pr) == [other_pr]
    assert by_pr.cross_config_hints[0]["display_label"] == "local trial, not yet a GitHub PR"
    assert by_pr.cross_config_hints[0]["provider"] == "local" and by_pr.cross_config_hints[0]["number"] is None


def test_cross_config_hints_require_the_flag_and_a_named_scope(demo_store: MemoryStore) -> None:
    other_cfg, _, other_commit = _publish_second_config(demo_store)
    without_flag = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling"))
    assert without_flag.cross_config_hints == []
    # An unnamed scope already spans every config: the other commit is an ordinary item, never a hint.
    unnamed = query_memory(demo_store, QueryFilters(component="tiling", include_cross_config_hints=True))
    assert set(_ids(unnamed)) == {"commit-demo-a", other_commit}
    assert unnamed.cross_config_hints == []
    assert unnamed.configs == sorted([CFG_DEMO, other_cfg])
    # A kernel-wide scope has no "other" configs of the same kernel.
    kernel_wide = query_memory(demo_store, QueryFilters(kernel_id=KERNEL_ID, component="tiling", include_cross_config_hints=True))
    assert set(_ids(kernel_wide)) == {"commit-demo-a", other_commit}
    assert kernel_wide.cross_config_hints == []


def test_cross_config_hints_by_config_hash_scope(demo_store: MemoryStore) -> None:
    other_cfg, _, other_commit = _publish_second_config(demo_store)
    other_hash = demo_store.require(other_cfg, "config").payload.config_hash
    result = query_memory(demo_store, QueryFilters(config_hash=other_hash, record_type="commit", include_cross_config_hints=True))
    assert result.configs == [other_cfg]
    assert _ids(result) == [other_commit]
    assert set(_hint_ids(result)) == {"commit-demo-a", "commit-demo-b", "commit-demo-c", "commit-demo-a-in-102"}
    assert all(h[CROSS_CONFIG_HINT_FIELD] is True and h["config_ref"] == CFG_DEMO for h in result.cross_config_hints)


# --------------------------------------------------------------------------------------
# SQLite fast path: identical results, never exclusive facts
# --------------------------------------------------------------------------------------
def test_sqlite_index_narrowing_is_equivalent_and_detects_staleness(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    filters = QueryFilters(config_ref=CFG_DEMO, component="tiling")
    scan = query_memory(demo_store, filters)
    assert scan.index_used is False
    assert _ids(scan) == ["commit-demo-a"]

    demo_store.rebuild_index()
    indexed = query_memory(demo_store, filters)
    assert indexed.index_used is True
    scan_dict, indexed_dict = scan.to_dict(), indexed.to_dict()
    assert scan_dict.pop("index_used") is False and indexed_dict.pop("index_used") is True
    assert scan_dict == indexed_dict

    # The index is only consulted when it can narrow commits: no component filter -> scan path.
    assert query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, record_type="run")).index_used is False

    # Publishing one more record without rebuilding makes the cache stale; results stay identical.
    stale_annotation = record_dict(bundle_dicts, "annotation-demo-grouped")
    stale_annotation["record_id"] = "annotation-query-stale"
    stale_annotation["payload"]["target_ref"] = "commit-demo-b"
    demo_store.publish(new_record("annotation", "annotation-query-stale", stale_annotation["payload"], created_at=FIXED_TS))
    after = query_memory(demo_store, filters)
    assert after.index_used is False
    after_dict = after.to_dict()
    after_dict.pop("index_used")
    assert after_dict == scan_dict
    # ... and the new record is visible through the scan path, proving the stale cache was bypassed.
    assert "annotation-query-stale" in _ids(query_memory(demo_store, QueryFilters(subject_ref="commit-demo-b")))


def test_sqlite_index_narrowing_matches_scan_for_every_component(demo_store: MemoryStore) -> None:
    components = sorted({c.component for r in demo_store.records("commit") for c in r.payload.changes} | {"unknown-component"})
    scans = {c: query_memory(demo_store, QueryFilters(component=c)).to_dict() for c in components}
    demo_store.rebuild_index()
    for component in components:
        indexed = query_memory(demo_store, QueryFilters(component=component))
        assert indexed.index_used is True
        indexed_dict = indexed.to_dict()
        expected = dict(scans[component])
        indexed_dict.pop("index_used")
        expected.pop("index_used")
        assert indexed_dict == expected, component


# --------------------------------------------------------------------------------------
# Synthetic store (no fixture bundle)
# --------------------------------------------------------------------------------------
def test_synthetic_store_untested_commit_and_scope(store: MemoryStore) -> None:
    kernel_id = _publish_kernel(store)
    cfg, pr, commit = _publish_second_config(store, component="layout", key="layout_id")

    everything = query_memory(store, QueryFilters())
    assert _ids(everything) == [kernel_id, cfg, pr, commit]
    assert everything.configs == [cfg]

    not_run = query_memory(store, QueryFilters(run_status="not_run"))
    assert _ids(not_run) == [commit]
    assert untested_commits(store, cfg) == [
        {"commit_ref": commit, "pr_ref": pr, "commit_oid": OTHER_OID, "reason": REASON_NO_RUN_RECORDED}
    ]
    assert query_memory(store, QueryFilters(run_status="tested")).items == []
    assert _ids(query_memory(store, QueryFilters(parameter_key="layout_id"))) == [commit]
    assert _ids(query_memory(store, QueryFilters(component="tiling"))) == []

    # Only one config of the kernel exists: nothing can be a cross-config hint.
    hinted = query_memory(store, QueryFilters(config_ref=cfg, include_cross_config_hints=True))
    assert hinted.cross_config_hints == []
    with pytest.raises(MissingReferenceError):
        untested_commits(store, "cfg-missing")
