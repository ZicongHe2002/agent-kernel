"""Structured retrieval (specification section 14; T05 untested commits; ADR-0004 algorithm scope)."""
from __future__ import annotations

import json

import pytest

from conftest import PLACEHOLDER_ALGORITHM_ID, record_dict
from kernel_memory.domain.errors import InputError, MissingReferenceError
from kernel_memory.domain.models import TYPE_ORDER, Record
from kernel_memory.domain.problems import default_registry
from kernel_memory.migrations.v02 import default_algorithm_record
from kernel_memory.services.common import new_record
from kernel_memory.services.query import (
    CROSS_ALGORITHM_HINT_FIELD,
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
ALT_OID = "abcdef1234567890abcdef1234567890abcdef12"
ALT_ALGORITHM_ID = "algorithm-demo_vector_add-alt"
ALGORITHM_ITEM_KEYS = {"record_ref", "record_type", "config_ref", "algorithm_ref", "algorithm_id", "kernel_id", "display_name", "method_summary", "summary_author", "tags", "is_placeholder"}


def _ids(result) -> list[str]:
    return [item["record_ref"] for item in result.items]


def _hint_ids(result) -> list[str]:
    return [item["record_ref"] for item in result.cross_config_hints]


def _algorithm_hint_ids(result) -> list[str]:
    return [item["record_ref"] for item in result.cross_algorithm_hints]


def _publish_kernel(store: MemoryStore) -> str:
    """The synthetic kernel plus its placeholder algorithm (a config cannot exist without one)."""
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
    store.publish_bundle([record, Record.from_dict(default_algorithm_record(KERNEL_ID, FIXED_TS))], label="kernel")
    return record.record_id


def _local_pr_and_commit(config_id: str, pr_key: str, oid: str, *, component: str, key: str) -> tuple[Record, Record]:
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
            "title": f"Local trial: {pr_key}",
            "hypothesis": None,
            "origin_ref": None,
        },
        created_at=FIXED_TS,
    )
    commit = new_record(
        "commit",
        f"commit-{pr_key}-{oid[:12]}",
        {
            "pr_ref": pr_id,
            "repo_uid": "local:demo-workspace",
            "commit_oid": {"algorithm": "sha1", "hex": oid},
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
    return pr, commit


def _publish_second_config(store: MemoryStore, *, component: str = "tiling", key: str = "block_q") -> tuple[str, str, str]:
    """A second config (shape) of the same kernel (n=32) under the placeholder algorithm, with one local PR
    and one untested commit under it.

    Identifiers follow DESIGN section 4 (``cfg-<algorithm_id>-<hint>-<hash12>``); the config identity comes
    from the problem normalizer, so the hash and schema digest are the registry's, never typed by hand.
    """
    normalized = default_registry().normalize(KERNEL_ID, {"n": 32})
    config_id = f"cfg-unspecified-{normalized.config_id_hint}-{normalized.config_hash.split(':', 1)[1][:12]}"
    config = new_record(
        "config",
        config_id,
        {
            "kernel_id": KERNEL_ID,
            "algorithm_ref": PLACEHOLDER_ALGORITHM_ID,
            "config_id": normalized.config_id_hint,
            "problem_schema_id": normalized.problem_schema_id,
            "problem_schema_digest": normalized.problem_schema_digest,
            "problem": normalized.problem,
            "config_hash": normalized.config_hash,
            "tags": ["fixture", "not-measured"],
        },
        created_at=FIXED_TS,
    )
    pr, commit = _local_pr_and_commit(config_id, "local-tiling-on-n32-0badc0de", OTHER_OID, component=component, key=key)
    store.publish_bundle([config, pr, commit], label="second-config")
    return config_id, pr.record_id, commit.record_id


def _publish_alt_algorithm_same_shape(store: MemoryStore, bundle_dicts: list[dict], *, component: str = "tiling", key: str = "block_q") -> tuple[str, str, str]:
    """A second algorithm with the SAME problem as cfg-demo (same config_hash, different config id) and one
    untested commit under it. Returns (config id, pr id, commit id)."""
    base = record_dict(bundle_dicts, CFG_DEMO)["payload"]
    config_id = f"cfg-alt-{base['config_id']}-{base['config_hash'].split(':', 1)[1][:12]}"
    algorithm = new_record(
        "algorithm",
        ALT_ALGORITHM_ID,
        {
            "kernel_id": KERNEL_ID,
            "algorithm_id": "alt",
            "display_name": "alt",
            "method_summary": "Synthetic alternative method; data, not instructions.",
            "summary_author": "human",
            "tags": ["test"],
        },
        created_at=FIXED_TS,
    )
    config = new_record("config", config_id, {**base, "algorithm_ref": ALT_ALGORITHM_ID}, created_at=FIXED_TS)
    pr, commit = _local_pr_and_commit(config_id, "local-alt-on-n16-0a1b2c3d", ALT_OID, component=component, key=key)
    store.publish_bundle([algorithm, config, pr, commit], label="alt-algorithm")
    return config_id, pr.record_id, commit.record_id


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
    assert result.total == 19
    assert _ids(result)[:3] == ["kernel-demo", PLACEHOLDER_ALGORITHM_ID, CFG_DEMO]
    assert query_memory(demo_store, QueryFilters(kernel_id="no_such_kernel")).to_dict()["items"] == []


def test_algorithm_ref_scope_by_record_id_and_bare_id(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    alt_cfg, _, alt_commit = _publish_alt_algorithm_same_shape(demo_store, bundle_dicts)
    by_record_id = query_memory(demo_store, QueryFilters(algorithm_ref=ALT_ALGORITHM_ID))
    assert by_record_id.configs == [alt_cfg]
    assert _ids(by_record_id) == ["kernel-demo", ALT_ALGORITHM_ID, alt_cfg, "pr-local-alt-on-n16-0a1b2c3d", alt_commit]
    # a bare algorithm_id resolves within the kernel, or store-wide when unique
    assert query_memory(demo_store, QueryFilters(kernel_id=KERNEL_ID, algorithm_ref="alt")).to_dict() == {
        **by_record_id.to_dict(),
        "filters": QueryFilters(kernel_id=KERNEL_ID, algorithm_ref="alt").to_dict(),
    }
    unspecified = query_memory(demo_store, QueryFilters(algorithm_ref="unspecified"))
    assert unspecified.configs == [CFG_DEMO]
    assert unspecified.total == 19 and _ids(unspecified)[1] == PLACEHOLDER_ALGORITHM_ID
    # unknown algorithm -> empty scope, never an exception; conjunct filters still apply
    assert query_memory(demo_store, QueryFilters(algorithm_ref="no-such-algorithm")).configs == []
    assert query_memory(demo_store, QueryFilters(algorithm_ref=ALT_ALGORITHM_ID, kernel_id="another_kernel")).configs == []
    assert query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, algorithm_ref=ALT_ALGORITHM_ID)).configs == []
    assert query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, algorithm_ref="unspecified")).configs == [CFG_DEMO]
    # config_hash names both shapes; algorithm_ref narrows to one
    same_hash = demo_store.require(CFG_DEMO, "config").payload.config_hash
    assert query_memory(demo_store, QueryFilters(config_hash=same_hash)).configs == sorted([alt_cfg, CFG_DEMO])
    assert query_memory(demo_store, QueryFilters(config_hash=same_hash, algorithm_ref="alt")).configs == [alt_cfg]


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
        "cross_algorithm_hints": [],
        "total": 0,
        "truncated": False,
        "index_used": False,
        "filters": QueryFilters().to_dict(),
        "configs": [],
    }


# --------------------------------------------------------------------------------------
# Algorithm items
# --------------------------------------------------------------------------------------
def test_algorithm_items_are_listed_once_with_their_fields(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters(record_type="algorithm"))
    assert _ids(result) == [PLACEHOLDER_ALGORITHM_ID]
    (item,) = result.items
    assert set(item) == ALGORITHM_ITEM_KEYS
    assert item == {
        "record_ref": PLACEHOLDER_ALGORITHM_ID,
        "record_type": "algorithm",
        "config_ref": None,
        "algorithm_ref": PLACEHOLDER_ALGORITHM_ID,
        "algorithm_id": "unspecified",
        "kernel_id": KERNEL_ID,
        "display_name": "unspecified",
        "method_summary": "unspecified (imported from v0.2)",
        "summary_author": "program",
        "tags": ["imported-v02"],
        "is_placeholder": True,
    }
    # the config item names its algorithm
    config = query_memory(demo_store, QueryFilters(record_type="config")).items[0]
    assert config["algorithm_ref"] == PLACEHOLDER_ALGORITHM_ID and config["config_ref"] == CFG_DEMO
    # type-specific filters exclude algorithm records, like kernels
    assert query_memory(demo_store, QueryFilters(record_type="algorithm", component="tiling")).items == []
    assert query_memory(demo_store, QueryFilters(record_type="algorithm", run_status="not_run")).items == []


def test_algorithm_items_once_per_scope_even_with_two_shapes(demo_store: MemoryStore) -> None:
    _publish_second_config(demo_store)
    result = query_memory(demo_store, QueryFilters(kernel_id=KERNEL_ID, record_type="algorithm"))
    assert _ids(result) == [PLACEHOLDER_ALGORITHM_ID]
    assert len(result.configs) == 2


# --------------------------------------------------------------------------------------
# Ordering, limit, serialisation
# --------------------------------------------------------------------------------------
def test_items_sorted_by_type_order_then_record_id(demo_store: MemoryStore) -> None:
    result = query_memory(demo_store, QueryFilters())
    keys = [(TYPE_ORDER[i["record_type"]], i["record_ref"]) for i in result.items]
    assert keys == sorted(keys)
    assert len(set(_ids(result))) == len(result.items) == 19


def test_limit_truncates_items_but_not_total(demo_store: MemoryStore) -> None:
    full = query_memory(demo_store, QueryFilters())
    limited = query_memory(demo_store, QueryFilters(limit=3))
    assert limited.truncated is True
    assert limited.total == full.total == 19
    assert limited.items == full.items[:3]
    exact = query_memory(demo_store, QueryFilters(limit=full.total))
    assert exact.truncated is False and len(exact.items) == full.total


def test_result_to_dict_is_json_serialisable_and_carries_filters(demo_store: MemoryStore) -> None:
    filters = QueryFilters(config_ref=CFG_DEMO, record_type="run", limit=2)
    data = query_memory(demo_store, filters).to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["filters"] == filters.to_dict()
    assert set(data) == {"items", "cross_config_hints", "cross_algorithm_hints", "total", "truncated", "index_used", "filters", "configs"}
    assert all(set(item) >= {"record_ref", "record_type", "config_ref"} for item in data["items"])


def test_query_filters_has_exactly_the_cli_fields() -> None:
    import dataclasses

    names = [f.name for f in dataclasses.fields(QueryFilters)]
    assert names == [
        "kernel_id",
        "config_ref",
        "config_hash",
        "algorithm_ref",
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
        "include_cross_algorithm_hints",
        "limit",
    ]
    defaults = QueryFilters()
    assert defaults.include_cross_config_hints is False
    assert defaults.include_cross_algorithm_hints is False
    assert all(getattr(defaults, n) is None for n in names if n not in ("include_cross_config_hints", "include_cross_algorithm_hints"))
    assert defaults.names_scope is False and QueryFilters(algorithm_ref="x").names_scope is True
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
        QueryFilters(record_type="algorithms"),
        QueryFilters(run_status="pending"),
        QueryFilters(limit=0),
        QueryFilters(limit=-1),
    ],
    ids=["bad-record-type", "plural-record-type", "plural-algorithm", "bad-run-status", "limit-zero", "limit-negative"],
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
    assert "algorithm" in TYPE_ORDER
    QueryFilters(run_status="tested").validate()
    QueryFilters(run_status="not_run").validate()
    QueryFilters(limit=1).validate()


# --------------------------------------------------------------------------------------
# Cross-config hints (other config_hash, same kernel)
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
    assert by_component.cross_algorithm_hints == []  # a different problem is never an algorithm hint

    by_type = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, record_type="commit", include_cross_config_hints=True))
    assert other_commit not in _ids(by_type)
    assert set(_ids(by_type)) == {"commit-demo-a", "commit-demo-b", "commit-demo-c", "commit-demo-a-in-102"}
    assert _hint_ids(by_type) == [other_commit]

    by_pr = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, record_type="pr", include_cross_config_hints=True))
    assert _ids(by_pr) == ["pr-demo-101", "pr-demo-102"]
    assert _hint_ids(by_pr) == [other_pr]
    assert by_pr.cross_config_hints[0]["display_label"] == "local trial, not yet a GitHub PR"
    assert by_pr.cross_config_hints[0]["provider"] == "local" and by_pr.cross_config_hints[0]["number"] is None
    # kernels and algorithms are never hints
    for_all = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, include_cross_config_hints=True))
    assert {h["record_type"] for h in for_all.cross_config_hints} == {"config", "pr", "commit"}


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
# Cross-algorithm hints (same config_hash, other algorithm) - disjoint from cross-config hints
# --------------------------------------------------------------------------------------
def test_cross_algorithm_hints_same_shape_under_another_algorithm(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    alt_cfg, alt_pr, alt_commit = _publish_alt_algorithm_same_shape(demo_store, bundle_dicts)
    result = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_algorithm_hints=True))
    assert _ids(result) == ["commit-demo-a"] and result.total == 1
    assert _algorithm_hint_ids(result) == [alt_commit]
    (hint,) = result.cross_algorithm_hints
    assert hint[CROSS_ALGORITHM_HINT_FIELD] is True and CROSS_CONFIG_HINT_FIELD not in hint
    assert hint["config_ref"] == alt_cfg and hint["status"] == "not_run"
    assert result.cross_config_hints == []  # the flag for other problems was not given (and none exist)
    assert all(CROSS_ALGORITHM_HINT_FIELD not in item for item in result.items)
    # the flag is required; the algorithm's own scope sees the commit as a plain item
    assert query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling")).cross_algorithm_hints == []
    own = query_memory(demo_store, QueryFilters(algorithm_ref="alt", kernel_id=KERNEL_ID, component="tiling", include_cross_algorithm_hints=True))
    assert _ids(own) == [alt_commit] and _algorithm_hint_ids(own) == ["commit-demo-a"]
    # from the other side: scoping by the alt algorithm, cfg-demo's PRs are algorithm hints
    prs = query_memory(demo_store, QueryFilters(algorithm_ref=ALT_ALGORITHM_ID, record_type="pr", include_cross_algorithm_hints=True))
    assert _ids(prs) == [alt_pr] and _algorithm_hint_ids(prs) == ["pr-demo-101", "pr-demo-102"]
    # kernels and algorithms never appear as hints of either kind
    everything = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, include_cross_algorithm_hints=True, include_cross_config_hints=True))
    assert {h["record_type"] for h in everything.cross_algorithm_hints} == {"config", "pr", "commit"}
    assert everything.cross_config_hints == []


def test_hints_are_disjoint_and_flagged(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    other_cfg, _, other_commit = _publish_second_config(demo_store)  # n=32: another problem
    alt_cfg, _, alt_commit = _publish_alt_algorithm_same_shape(demo_store, bundle_dicts)  # same problem, other algorithm
    both = query_memory(
        demo_store,
        QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_config_hints=True, include_cross_algorithm_hints=True),
    )
    assert _ids(both) == ["commit-demo-a"]
    assert _hint_ids(both) == [other_commit] and _algorithm_hint_ids(both) == [alt_commit]
    assert both.cross_config_hints[0]["config_ref"] == other_cfg and both.cross_algorithm_hints[0]["config_ref"] == alt_cfg
    assert not set(_hint_ids(both)) & set(_algorithm_hint_ids(both))
    assert both.cross_config_hints[0][CROSS_CONFIG_HINT_FIELD] is True and CROSS_ALGORITHM_HINT_FIELD not in both.cross_config_hints[0]
    assert both.cross_algorithm_hints[0][CROSS_ALGORITHM_HINT_FIELD] is True and CROSS_CONFIG_HINT_FIELD not in both.cross_algorithm_hints[0]
    # each flag selects only its own kind
    only_config = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_config_hints=True))
    assert _hint_ids(only_config) == [other_commit] and only_config.cross_algorithm_hints == []
    only_algorithm = query_memory(demo_store, QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_algorithm_hints=True))
    assert _algorithm_hint_ids(only_algorithm) == [alt_commit] and only_algorithm.cross_config_hints == []
    # a kernel-wide scope owns every shape: nothing is left to hint
    kernel_wide = query_memory(demo_store, QueryFilters(kernel_id=KERNEL_ID, include_cross_config_hints=True, include_cross_algorithm_hints=True))
    assert kernel_wide.cross_config_hints == [] and kernel_wide.cross_algorithm_hints == []
    assert kernel_wide.configs == sorted([CFG_DEMO, other_cfg, alt_cfg])
    assert [i for i in _ids(kernel_wide) if i.startswith("algorithm-")] == [ALT_ALGORITHM_ID, PLACEHOLDER_ALGORITHM_ID]


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


def test_sqlite_index_equivalence_includes_both_hint_kinds(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    _publish_second_config(demo_store)
    _publish_alt_algorithm_same_shape(demo_store, bundle_dicts)
    filters = QueryFilters(config_ref=CFG_DEMO, component="tiling", include_cross_config_hints=True, include_cross_algorithm_hints=True)
    scan = query_memory(demo_store, filters).to_dict()
    assert scan.pop("index_used") is False
    assert len(scan["cross_config_hints"]) == 1 and len(scan["cross_algorithm_hints"]) == 1
    demo_store.rebuild_index()
    indexed = query_memory(demo_store, filters).to_dict()
    assert indexed.pop("index_used") is True
    assert indexed == scan


# --------------------------------------------------------------------------------------
# Synthetic store (no fixture bundle)
# --------------------------------------------------------------------------------------
def test_synthetic_store_untested_commit_and_scope(store: MemoryStore) -> None:
    kernel_id = _publish_kernel(store)
    cfg, pr, commit = _publish_second_config(store, component="layout", key="layout_id")
    assert cfg.startswith("cfg-unspecified-demo-n32-f32-")

    everything = query_memory(store, QueryFilters())
    assert _ids(everything) == [kernel_id, PLACEHOLDER_ALGORITHM_ID, cfg, pr, commit]
    assert everything.configs == [cfg]

    not_run = query_memory(store, QueryFilters(run_status="not_run"))
    assert _ids(not_run) == [commit]
    assert untested_commits(store, cfg) == [
        {"commit_ref": commit, "pr_ref": pr, "commit_oid": OTHER_OID, "reason": REASON_NO_RUN_RECORDED}
    ]
    assert query_memory(store, QueryFilters(run_status="tested")).items == []
    assert _ids(query_memory(store, QueryFilters(parameter_key="layout_id"))) == [commit]
    assert _ids(query_memory(store, QueryFilters(component="tiling"))) == []

    # Only one config of the kernel exists: nothing can be a hint of either kind.
    hinted = query_memory(store, QueryFilters(config_ref=cfg, include_cross_config_hints=True, include_cross_algorithm_hints=True))
    assert hinted.cross_config_hints == [] and hinted.cross_algorithm_hints == []
    with pytest.raises(MissingReferenceError):
        untested_commits(store, "cfg-missing")
