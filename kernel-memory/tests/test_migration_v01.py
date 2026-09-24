"""Tests for the explicit v0.1 -> v0.2 migration (specification section 20, acceptance T31).

All input data here is SYNTHETIC. It follows the v0.1 structure documented in
``docs/MIGRATION.md`` (inferred from the concepts named in specification section 20); no
real v0.1 export exists in this workspace and the historical PDF is a design example, not a
dataset. Fictional Git object ids and numbers exist only to exercise mapping rules.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kernel_memory.domain.errors import IdConflictError, InputError, UnsafePathError
from kernel_memory.domain.models import GitOid, Record
from kernel_memory.migrations.v01 import (
    EXCESSIVE_SPILL_NOTE,
    NO_ADAPTER_REASON,
    ORIGIN_UNKNOWN,
    RESULT_TEXT_PREFIX,
    SPILL_BYTES_NOTE,
    MappingResolver,
    MigrationReport,
    NullResolver,
    ShaResolver,
    migrate_v01,
    read_v01_source,
    resolve_sha,
    source_file_digests,
)
from kernel_memory.storage import MemoryStore

# --------------------------------------------------------------------------------------
# Synthetic v0.1 export (clearly fictional object ids; none contains "0000")
# --------------------------------------------------------------------------------------
REPO_NAME = "org/repo"
REPO_UID = "github:github.com:repo:42"
FULL_1 = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"  # attempt 1, revision 1 (full sha)
FULL_2 = "beef123456789abcdef123456789abcdef123456"  # resolvable from short "beef12"
SHORT_2 = "beef12"
FULL_3A = "cafe1111abcdefabcdefabcdefabcdefabcdefab"  # "cafe" is ambiguous between 3A and 3B
FULL_3B = "cafe2222abcdefabcdefabcdefabcdefabcdefab"
SHORT_3 = "cafe"
FIXTURE_CONFIG_HASH = "sha256:f93d42a53c9f6bcdab1fa855d9b3e5727e65601ff355aa96dc16142a282c1915"
CREATED_AT = "2026-09-09T00:00:00Z"
RESULT_REV1 = {"revision_id": "rev-1", "status": "ok", "latency_us": 123.4, "speedup": 1.1}
RESULT_REV2 = {
    "revision_id": "rev-2",
    "status": "failed",
    "latency_us": 130.0,
    "spill_bytes": 65536,
    "excessive_spill": True,
    "environment": {"tpu": "synthetic-v5e"},
    "notes": "SYNTHETIC v0.1 result; not a measurement",
}

for _hex in (FULL_1, FULL_2, FULL_3A, FULL_3B):
    assert len(_hex) == 40 and "0000" not in _hex


def synthetic_v01() -> dict[str, Any]:
    return {
        "kernels": [{"kernel_id": "demo_vector_add", "display_name": "SYNTHETIC vector add (v0.1)", "adapter_id": "legacy-cpu"}],
        "configs": [{"config_id": "cfg-1", "kernel_id": "demo_vector_add", "problem": {"n": 16, "dtype": "f32"}}],
        "attempts": [
            {
                "attempt_id": "att-1",
                "kernel_id": "demo_vector_add",
                "config_id": "cfg-1",
                "pr_number": 7,
                "repo": REPO_NAME,
                "title": "SYNTHETIC attempt 1: tiling",
                "hypothesis": "SYNTHETIC hypothesis",
                "selected_revision": "rev-1",
            },
            {
                "attempt_id": "att-2",
                "kernel_id": "demo_vector_add",
                "config_id": "cfg-1",
                "repo": REPO_NAME,
                "title": "SYNTHETIC attempt 2: local trial",
                "parent_attempt_id": "att-1",
            },
        ],
        "revisions": [
            {
                "revision_id": "rev-1",
                "attempt_id": "att-1",
                "commit_sha": FULL_1,
                "changes": [
                    {"change_id": "chg-1", "component": "tiling", "key": "block_q", "before": 128, "after": 256, "rationale": "synthetic", "attribution": "isolated"},
                    {"component": "layout", "key": None, "before": "a", "after": "b", "extraction_source": "explicit", "mystery": 1},
                    {"bogus": True},
                ],
                "summary": "SYNTHETIC revision 1",
            },
            {"revision_id": "rev-2", "attempt_id": "att-1", "commit_sha": SHORT_2, "parent_sha": FULL_1, "summary": "SYNTHETIC revision 2"},
            {"revision_id": "rev-3", "attempt_id": "att-2", "commit_sha": SHORT_3, "parent_sha": SHORT_2, "summary": "SYNTHETIC revision 3"},
        ],
        "results": [dict(RESULT_REV1), dict(RESULT_REV2)],
        "trajectory": {"nodes": ["rev-1", "rev-2"], "note": "old derived view; must be rebuilt"},
        "legacy_stats": {"total_runs": 2},
    }


def resolver() -> MappingResolver:
    return MappingResolver({(REPO_UID, SHORT_2): FULL_2, (REPO_UID, "cafe1111"): FULL_3A, (REPO_UID, "cafe2222"): FULL_3B})


def write_single(tmp_path: Path, data: dict[str, Any], name: str = "v01.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def run(path: Path, **kwargs: Any) -> MigrationReport:
    kwargs.setdefault("resolver", resolver())
    kwargs.setdefault("repo_uid_map", {REPO_NAME: REPO_UID})
    kwargs.setdefault("created_at", CREATED_AT)
    return migrate_v01(path, **kwargs)


def by_type(report: MigrationReport, record_type: str) -> list[dict[str, Any]]:
    return [r for r in report.records if r["record_type"] == record_type]


# --------------------------------------------------------------------------------------
# Dry run / apply
# --------------------------------------------------------------------------------------
def test_t31_dry_run_produces_records_and_writes_nothing(tmp_path: Path, store: MemoryStore) -> None:
    report = run(write_single(tmp_path, synthetic_v01()), store=store)  # dry_run defaults to True
    assert report.dry_run is True
    assert report.record_counts() == {"annotation": 3, "commit": 2, "config": 1, "kernel": 1, "pr": 2}
    assert report.publish_outcome is None
    assert store.index_entries() == []
    assert list((store.root / "kernels").rglob("*.json")) == []
    assert any("nothing was written" in note for note in report.notes)
    for record in report.records:  # every produced record is a valid v0.2 record
        Record.from_dict(json.loads(json.dumps(record)))


def test_apply_publishes_and_deep_counts_match(tmp_path: Path, store: MemoryStore) -> None:
    report = run(write_single(tmp_path, synthetic_v01()), store=store, dry_run=False)
    assert report.dry_run is False
    assert report.publish_outcome is not None
    assert sorted(report.publish_outcome["published"]) == sorted(r["record_id"] for r in report.records)
    counts = {t: len(store.records(t)) for t in ("kernel", "config", "pr", "commit", "annotation", "relation", "decision", "run")}
    assert counts["kernel"] == 1
    assert counts["config"] == 1
    assert counts["pr"] == 2
    assert counts["commit"] == 2
    assert counts["annotation"] >= 3
    assert counts["relation"] == 0  # child attempt has no concrete commit (rev-3 ambiguous) -> rule says unresolved
    assert counts["decision"] == 0
    assert counts["run"] == 0
    for record in report.records:
        stored = store.get(record["record_id"])
        assert stored is not None
        assert stored.canonical_digest() == Record.from_dict(json.loads(json.dumps(record))).canonical_digest()
    assert store.integrity_scan().ok


def test_no_store_is_a_pure_dry_run(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()), store=None, dry_run=False)
    assert report.publish_outcome is None
    assert any("no store given" in note for note in report.notes)
    assert report.record_counts()["commit"] == 2


# --------------------------------------------------------------------------------------
# T31: short SHAs
# --------------------------------------------------------------------------------------
def test_t31_short_ambiguous_sha_is_unresolved_and_never_padded(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    commits = by_type(report, "commit")
    hexes = sorted(c["payload"]["commit_oid"]["hex"] for c in commits)
    assert hexes == sorted([FULL_1, FULL_2])
    for commit in commits:
        oid = commit["payload"]["commit_oid"]
        assert len(oid["hex"]) in (40, 64) and "0000" not in oid["hex"]
        for parent in commit["payload"]["git_parent_oids"]:
            assert "0000" not in parent["hex"] and len(parent["hex"]) in (40, 64)
    # rev-3 ("cafe") matches two objects -> unresolved with a reason, no record at all
    unresolved = [u for u in report.unresolved if u["v01_kind"] == "revision" and u["v01_id"] == "rev-3"]
    assert len(unresolved) == 1
    assert "ambiguous" in unresolved[0]["reason"] and "not padded or guessed" in unresolved[0]["reason"]
    mapped = [m for m in report.identity_map if m["v01_kind"] == "revision" and m["v01_id"] == "rev-3"]
    assert mapped == [{"v01_kind": "revision", "v01_id": "rev-3", "v02_record_type": "commit", "v02_record_id": None, "status": "unresolved"}]
    assert "0000" not in json.dumps(report.identity_map + report.unresolved + report.rejections)
    assert not any(SHORT_3 + "0" in json.dumps(c) for c in commits)
    # the resolvable short sha maps to the resolver's full object id, in the commit id as well
    rev2 = next(c for c in commits if c["payload"]["commit_oid"]["hex"] == FULL_2)
    assert rev2["record_id"] == f"commit-gh-42-pr-7-{FULL_2[:12]}"
    assert rev2["payload"]["git_parent_oids"] == [{"algorithm": "sha1", "hex": FULL_1}]


def test_t31_null_resolver_leaves_every_short_sha_unresolved(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()), resolver=NullResolver())
    commits = by_type(report, "commit")
    assert [c["payload"]["commit_oid"]["hex"] for c in commits] == [FULL_1]
    unresolved_ids = sorted(u["v01_id"] for u in report.unresolved if u["v01_kind"] == "revision")
    assert unresolved_ids == ["rev-2", "rev-3"]
    # the result attached to the unresolved revision is not migrated either (never a Run, never guessed)
    result_unresolved = [u for u in report.unresolved if u["v01_kind"] == "result"]
    assert len(result_unresolved) == 1 and result_unresolved[0]["v01_id"].startswith("rev-2#result")
    assert report.record_counts().get("run", 0) == 0


def test_default_resolver_is_null(tmp_path: Path) -> None:
    report = migrate_v01(write_single(tmp_path, synthetic_v01()), repo_uid_map={REPO_NAME: REPO_UID}, created_at=CREATED_AT)
    assert [c["payload"]["commit_oid"]["hex"] for c in by_type(report, "commit")] == [FULL_1]


def test_unresolved_parent_sha_gives_empty_parents_and_note(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["revisions"][1]["parent_sha"] = "dead"  # unknown short parent
    report = run(write_single(tmp_path, data))
    rev2 = next(c for c in by_type(report, "commit") if c["payload"]["commit_oid"]["hex"] == FULL_2)
    assert rev2["payload"]["git_parent_oids"] == []
    assert rev2["payload"]["diff_base_oid"] is None
    assert any("parent_sha unresolved" in note for note in report.notes)
    assert any(u["v01_kind"] == "revision.parent_sha" and u["v01_id"] == "rev-2" for u in report.unresolved)


def test_sha256_full_oid_accepted_by_length(tmp_path: Path) -> None:
    data = synthetic_v01()
    sha256_hex = "ab12" * 16
    data["revisions"][0]["commit_sha"] = sha256_hex.upper()  # tolerant: case normalised, never truncated
    report = run(write_single(tmp_path, data))
    rev1 = next(c for c in by_type(report, "commit") if c["payload"]["commit_oid"]["algorithm"] == "sha256")
    assert rev1["payload"]["commit_oid"]["hex"] == sha256_hex


def test_non_hex_sha_is_rejected_not_unresolved(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["revisions"][0]["commit_sha"] = "not-a-sha"
    report = run(write_single(tmp_path, data))
    rejected = [r for r in report.rejections if r["v01_kind"] == "revision" and r["v01_id"] == "rev-1"]
    assert len(rejected) == 1 and "not hexadecimal" in rejected[0]["reason"]
    # rev-1 is rejected: no commit is minted under its identity and nothing is guessed in its place.  FULL_1 may still
    # legitimately appear as rev-2's git_parent_oids (the synthetic input names it as parent_sha; spec 20 retains the
    # original value), so the check is on commit identities, not on the serialized report text.
    commits = by_type(report, "commit")
    assert all(c["payload"]["commit_oid"]["hex"] != FULL_1 for c in commits)
    assert commits, "the resolvable sibling revision must still be migrated"
    for commit in commits:
        oid_hex = commit["payload"]["commit_oid"]["hex"]
        assert len(oid_hex) in (40, 64) and oid_hex != "not-a-sha"
        assert "0000" not in oid_hex  # a rejected or short sha is never zero-padded into a full oid (spec 20)


# --------------------------------------------------------------------------------------
# T31: spill / result semantics
# --------------------------------------------------------------------------------------
def test_t31_spill_semantics_preserved_verbatim_and_no_run_or_metric_created(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    assert report.record_counts().get("run", 0) == 0
    assert "analysis_metrics" not in json.dumps(report.records)
    result_annotations = [a for a in by_type(report, "annotation") if a["payload"]["text"].startswith(RESULT_TEXT_PREFIX)]
    assert len(result_annotations) == 2
    spill = next(a for a in result_annotations if '"spill_bytes"' in a["payload"]["text"])
    text = spill["payload"]["text"]
    body = json.loads(text[len(RESULT_TEXT_PREFIX) :])
    assert body["v01_result"] == RESULT_REV2  # original fields verbatim, as data
    assert body["v01_result"]["spill_bytes"] == 65536
    assert body["v01_result"]["excessive_spill"] is True
    assert body["v01_result"]["status"] == "failed"
    assert body["field_notes"]["spill_bytes"] == SPILL_BYTES_NOTE
    assert SPILL_BYTES_NOTE in text and EXCESSIVE_SPILL_NOTE in text
    assert "execution_status" not in text and "compile_error" not in text and "correctness" not in body["v01_result"]
    payload = spill["payload"]
    assert payload["category"] == "note" and payload["confidence"] == "unverified" and payload["author_kind"] == "human"
    assert payload["evidence_refs"] == [] and payload["supersedes_ref"] is None
    assert payload["target_ref"] == f"commit-gh-42-pr-7-{FULL_2[:12]}"
    assert report.annotations_for_unverified == 3
    disposition = report.retained_fields["results"]
    assert "NOT" in disposition["status"] and "NOT" in disposition["latency_us"] and SPILL_BYTES_NOTE in disposition["spill_bytes"]
    assert "no execution failure inferred" in disposition["excessive_spill"]


def test_result_for_unknown_revision_is_rejected(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["results"].append({"revision_id": "rev-missing", "status": "ok"})
    data["results"].append({"status": "ok"})
    data["results"].append("garbage")
    report = run(write_single(tmp_path, data))
    reasons = {r["v01_id"]: r["reason"] for r in report.rejections if r["v01_kind"] == "result"}
    assert any("rev-missing" in k and "not migrated" in v for k, v in reasons.items())
    assert any("revision_id missing" in v for v in reasons.values())
    assert any("not an object" in v for v in reasons.values())
    assert len([a for a in by_type(report, "annotation") if a["payload"]["text"].startswith(RESULT_TEXT_PREFIX)]) == 2


# --------------------------------------------------------------------------------------
# selected_revision and parent_attempt_id
# --------------------------------------------------------------------------------------
def test_selected_revision_becomes_note_annotation_never_a_decision(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    assert report.record_counts().get("decision", 0) == 0
    selected = [a for a in by_type(report, "annotation") if "selected_revision=" in a["payload"]["text"]]
    assert len(selected) == 1
    payload = selected[0]["payload"]
    assert payload["text"] == "v0.1 selected_revision=rev-1; no policy/evidence-backed decision migrated"
    assert payload["target_ref"] == f"commit-gh-42-pr-7-{FULL_1[:12]}"
    assert payload["category"] == "note" and payload["author_kind"] == "human" and payload["confidence"] == "unverified"
    assert selected[0]["record_id"].startswith("annotation-") and len(selected[0]["record_id"]) == len("annotation-") + 16
    entry = next(m for m in report.identity_map if m["v01_kind"] == "selected_revision")
    assert entry["v01_id"] == "att-1" and entry["status"] == "mapped" and entry["v02_record_type"] == "annotation"


def test_selected_revision_of_unresolved_commit_is_unresolved(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["attempts"][0]["selected_revision"] = "rev-2"
    report = run(write_single(tmp_path, data), resolver=NullResolver())
    unresolved = [u for u in report.unresolved if u["v01_kind"] == "selected_revision"]
    assert len(unresolved) == 1 and "no migrated commit" in unresolved[0]["reason"]
    assert not any("selected_revision=" in a["payload"]["text"] for a in by_type(report, "annotation"))


def test_parent_attempt_relation_unresolved_when_child_has_no_concrete_commit(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    assert report.record_counts().get("relation", 0) == 0
    unresolved = [u for u in report.unresolved if u["v01_kind"] == "parent_attempt_id"]
    assert len(unresolved) == 1 and unresolved[0]["v01_id"] == "att-2"
    assert ORIGIN_UNKNOWN in unresolved[0]["reason"]
    # the PR-level origin is known (parent's selected revision) and recorded on the PR only
    pr2 = next(p for p in by_type(report, "pr") if p["payload"]["provider"] == "local")
    assert pr2["payload"]["origin_ref"] == f"commit-gh-42-pr-7-{FULL_1[:12]}"


def test_parent_attempt_relation_created_when_both_endpoints_resolve(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["revisions"][2]["commit_sha"] = FULL_3A  # child now has a single resolved revision
    report = run(write_single(tmp_path, data))
    relations = by_type(report, "relation")
    assert len(relations) == 1
    payload = relations[0]["payload"]
    origin = f"commit-gh-42-pr-7-{FULL_1[:12]}"
    pr2 = next(p for p in by_type(report, "pr") if p["payload"]["provider"] == "local")
    child = f"commit-{pr2['payload']['pr_key']}-{FULL_3A[:12]}"
    assert payload["kind"] == "optimization_origin"
    assert payload["from_ref"] == origin and payload["to_ref"] == child
    assert payload["evidence_refs"] == [] and payload["config_ref"] == pr2["payload"]["config_ref"]
    assert relations[0]["record_id"].startswith("relation-optimization_origin-")
    assert pr2["payload"]["origin_ref"] == origin
    assert not any(u["v01_kind"] == "parent_attempt_id" for u in report.unresolved)


def test_parent_attempt_unresolved_when_parent_has_no_selected_revision(tmp_path: Path) -> None:
    data = synthetic_v01()
    del data["attempts"][0]["selected_revision"]  # parent has two revisions and no selection
    data["revisions"][2]["commit_sha"] = FULL_3A
    report = run(write_single(tmp_path, data))
    assert report.record_counts().get("relation", 0) == 0
    unresolved = [u for u in report.unresolved if u["v01_kind"] == "parent_attempt_id"]
    assert len(unresolved) == 1 and ORIGIN_UNKNOWN in unresolved[0]["reason"] and "no selected_revision" in unresolved[0]["reason"]
    pr2 = next(p for p in by_type(report, "pr") if p["payload"]["provider"] == "local")
    assert pr2["payload"]["origin_ref"] is None


def test_parent_attempt_unknown_is_unresolved(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["attempts"][1]["parent_attempt_id"] = "att-ghost"
    report = run(write_single(tmp_path, data))
    unresolved = [u for u in report.unresolved if u["v01_kind"] == "parent_attempt_id"]
    assert len(unresolved) == 1 and "was not migrated" in unresolved[0]["reason"] and ORIGIN_UNKNOWN in unresolved[0]["reason"]


# --------------------------------------------------------------------------------------
# Kernels / configs / PRs
# --------------------------------------------------------------------------------------
def test_config_maps_to_fixture_hash_and_kernel_defaults(tmp_path: Path) -> None:
    data = synthetic_v01()
    del data["kernels"][0]["adapter_id"]
    report = run(write_single(tmp_path, data))
    (config,) = by_type(report, "config")
    assert config["payload"]["config_hash"] == FIXTURE_CONFIG_HASH
    assert config["record_id"] == f"cfg-demo-n16-f32-{FIXTURE_CONFIG_HASH[7:19]}"
    assert config["payload"]["problem"] == {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}
    assert config["payload"]["config_id"] == "demo-n16-f32"
    (kernel,) = by_type(report, "kernel")
    assert kernel["record_id"] == "kernel-demo_vector_add"
    assert kernel["payload"]["adapter_id"] == "unknown-v01"
    assert "Migrated from v0.1" in kernel["payload"]["contract_notes"]
    assert kernel["payload"]["display_name"] == "SYNTHETIC vector add (v0.1)"


def test_t01_existing_store_kernel_and_config_are_reused_by_hash(tmp_path: Path, demo_store: MemoryStore) -> None:
    before = {t: len(demo_store.records(t)) for t in ("kernel", "config", "pr", "commit", "annotation", "run", "decision")}
    report = run(write_single(tmp_path, synthetic_v01()), store=demo_store, dry_run=False)
    assert by_type(report, "kernel") == [] and by_type(report, "config") == []
    mapped = {(m["v01_kind"], m["v01_id"]): m for m in report.identity_map}
    assert mapped[("kernel", "demo_vector_add")]["v02_record_id"] == "kernel-demo"
    assert mapped[("config", "cfg-1")]["v02_record_id"] == "cfg-demo"
    for pr in by_type(report, "pr"):
        assert pr["payload"]["config_ref"] == "cfg-demo"
    after = {t: len(demo_store.records(t)) for t in before}
    assert after["kernel"] == before["kernel"] and after["config"] == before["config"]
    assert after["pr"] == before["pr"] + 2 and after["commit"] == before["commit"] + 2
    assert after["annotation"] == before["annotation"] + 3
    assert after["run"] == before["run"] and after["decision"] == before["decision"]
    assert demo_store.integrity_scan().ok


def test_two_equivalent_v01_configs_become_one_config(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["configs"].append({"config_id": "cfg-1-alias", "kernel_id": "demo_vector_add", "problem": {"n": 16, "dtype": "float32"}})
    data["attempts"][1]["config_id"] = "cfg-1-alias"
    report = run(write_single(tmp_path, data))
    assert len(by_type(report, "config")) == 1
    assert {m["v02_record_id"] for m in report.identity_map if m["v01_kind"] == "config"} == {by_type(report, "config")[0]["record_id"]}
    assert len(by_type(report, "pr")) == 2


def test_mla_forward_config_is_rejected_with_reason_and_nothing_fabricated(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["kernels"].append({"kernel_id": "mla_forward", "display_name": "SYNTHETIC MLA"})
    data["configs"].append({"config_id": "cfg-mla", "kernel_id": "mla_forward", "problem": {"batch": 1, "heads": 8}})
    data["attempts"].append({"attempt_id": "att-mla", "config_id": "cfg-mla", "title": "SYNTHETIC mla attempt"})
    data["revisions"].append({"revision_id": "rev-mla", "attempt_id": "att-mla", "commit_sha": FULL_3B})
    data["results"].append({"revision_id": "rev-mla", "status": "ok"})
    report = run(write_single(tmp_path, data))
    rejected = {r["v01_id"]: r["reason"] for r in report.rejections}
    assert rejected["cfg-mla"].startswith(NO_ADAPTER_REASON)
    assert "not migrated" in rejected["att-mla"] and "not migrated" in rejected["rev-mla"]
    assert any(k.startswith("rev-mla#result") for k in rejected)
    assert all(c["payload"]["kernel_id"] != "mla_forward" for c in by_type(report, "config"))
    assert FULL_3B not in json.dumps(report.records)
    assert all(m["status"] == "rejected" for m in report.identity_map if m["v01_id"] in ("cfg-mla", "att-mla", "rev-mla"))


def test_kernel_without_any_adapter_has_configs_rejected(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["kernels"].append({"kernel_id": "unknown_kernel"})
    data["configs"].append({"config_id": "cfg-unknown", "kernel_id": "unknown_kernel", "problem": {"x": 1}})
    report = run(write_single(tmp_path, data))
    reason = next(r["reason"] for r in report.rejections if r["v01_id"] == "cfg-unknown")
    assert reason.startswith(NO_ADAPTER_REASON)
    assert any(k["payload"]["kernel_id"] == "unknown_kernel" for k in by_type(report, "kernel"))


def test_invalid_problem_is_rejected_by_normalizer(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["configs"][0]["problem"] = {"n": True, "dtype": "f32"}  # boolean dimension (T32 flavour)
    report = run(write_single(tmp_path, data))
    reason = next(r["reason"] for r in report.rejections if r["v01_id"] == "cfg-1")
    assert "normalizer" in reason
    assert by_type(report, "config") == [] and by_type(report, "pr") == [] and by_type(report, "commit") == []


def test_pr_keys_github_and_local(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    prs = {p["payload"]["provider"]: p for p in by_type(report, "pr")}
    gh = prs["github"]
    assert gh["record_id"] == "pr-gh-42-pr-7" and gh["payload"]["pr_key"] == "gh-42-pr-7"
    assert gh["payload"]["number"] == 7 and gh["payload"]["repo_uid"] == REPO_UID
    assert gh["payload"]["title"] == "SYNTHETIC attempt 1: tiling" and gh["payload"]["hypothesis"] == "SYNTHETIC hypothesis"
    local = prs["local"]
    assert local["payload"]["number"] is None and local["payload"]["hypothesis"] is None
    assert local["payload"]["repo_uid"] == REPO_UID
    import re

    assert re.fullmatch(r"pr-local-att-2-[0-9a-f]{12}", local["record_id"])
    assert local["payload"]["pr_key"] == local["record_id"][len("pr-") :]
    commit_prs = {c["payload"]["pr_ref"] for c in by_type(report, "commit")}
    assert commit_prs == {"pr-gh-42-pr-7"}


def test_pr_number_without_repo_mapping_falls_back_to_local(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()), repo_uid_map={})
    prs = by_type(report, "pr")
    assert {p["payload"]["provider"] for p in prs} == {"local"}
    assert all(p["payload"]["number"] is None for p in prs)
    assert all(p["payload"]["repo_uid"].startswith("local:") for p in prs)
    assert "pr_number" in report.discarded_fields["attempts"]
    assert any("no repo_uid mapping" in note for note in report.notes)


# --------------------------------------------------------------------------------------
# Changes
# --------------------------------------------------------------------------------------
def test_changes_forced_group_only_and_extraction_source_rules(tmp_path: Path) -> None:
    report = run(write_single(tmp_path, synthetic_v01()))
    rev1 = next(c for c in by_type(report, "commit") if c["payload"]["commit_oid"]["hex"] == FULL_1)
    changes = rev1["payload"]["changes"]
    assert rev1["payload"]["change_status"] == "recorded"
    assert len(changes) == 2  # the {"bogus": true} entry is dropped
    assert {c["attribution"] for c in changes} == {"group_only"}
    by_component = {c["component"]: c for c in changes}
    assert by_component["tiling"]["extraction_source"] == "unknown"  # not provided as explicit
    assert by_component["layout"]["extraction_source"] == "explicit"
    assert by_component["layout"]["change_id"].startswith("change-v01-")
    assert any("overridden to group_only" in note for note in report.notes)
    discarded = report.discarded_fields["revisions"]
    assert any(k.startswith("changes[rev-1][2]") for k in discarded)  # dropped bogus entry
    assert any(k.endswith(".mystery") for k in discarded)  # unknown change field
    assert rev1["payload"]["summary"] == "SYNTHETIC revision 1" and rev1["payload"]["summary_author"] == "human"
    assert rev1["payload"]["source_available"] is False and rev1["payload"]["diff_artifact_ref"] is None
    rev2 = next(c for c in by_type(report, "commit") if c["payload"]["commit_oid"]["hex"] == FULL_2)
    assert rev2["payload"]["change_status"] == "not_extracted" and rev2["payload"]["changes"] == []


def test_missing_summary_is_empty_and_collector_authored(tmp_path: Path) -> None:
    data = synthetic_v01()
    del data["revisions"][0]["summary"]
    report = run(write_single(tmp_path, data))
    rev1 = next(c for c in by_type(report, "commit") if c["payload"]["commit_oid"]["hex"] == FULL_1)
    assert rev1["payload"]["summary"] == "" and rev1["payload"]["summary_author"] == "collector"


# --------------------------------------------------------------------------------------
# Idempotency / conflicts (T12 flavour)
# --------------------------------------------------------------------------------------
def test_reapply_is_idempotent_and_changed_input_conflicts(tmp_path: Path, store: MemoryStore) -> None:
    path = write_single(tmp_path, synthetic_v01())
    first = run(path, store=store, dry_run=False)
    second = run(path, store=store, dry_run=False)
    assert second.publish_outcome is not None
    assert second.publish_outcome["published"] == []
    assert sorted(second.publish_outcome["idempotent"]) == sorted(r["record_id"] for r in first.records)
    assert [r["record_id"] for r in second.records] == [r["record_id"] for r in first.records]
    assert len(store.records()) == len(first.records)
    changed = synthetic_v01()
    changed["attempts"][0]["title"] = "SYNTHETIC attempt 1: edited title"
    changed_path = write_single(tmp_path, changed, name="v01-changed.json")
    with pytest.raises(IdConflictError) as excinfo:
        run(changed_path, store=store, dry_run=False)
    assert excinfo.value.exit_code == 3 and excinfo.value.code == "ID_CONFLICT"
    assert excinfo.value.details["record_id"] == "pr-gh-42-pr-7"
    with pytest.raises(IdConflictError):  # a dry run predicts the same conflict without writing
        run(changed_path, store=store, dry_run=True)
    assert len(store.records()) == len(first.records)
    assert store.integrity_scan().ok


def test_different_created_at_conflicts_with_published_records(tmp_path: Path, store: MemoryStore) -> None:
    path = write_single(tmp_path, synthetic_v01())
    run(path, store=store, dry_run=False)
    with pytest.raises(IdConflictError):
        run(path, store=store, dry_run=False, created_at="2026-09-10T00:00:00Z")


def test_store_publish_conflict_surfaces_when_precheck_is_bypassed(tmp_path: Path, store: MemoryStore) -> None:
    """The store itself is the last line of defence: publishing a differing record with a migrated id conflicts."""
    report = run(write_single(tmp_path, synthetic_v01()), store=store, dry_run=False)
    pr = next(r for r in report.records if r["record_id"] == "pr-gh-42-pr-7")
    edited = json.loads(json.dumps(pr))
    edited["payload"]["title"] = "edited"
    with pytest.raises(IdConflictError):
        store.publish(Record.from_dict(edited))


def test_bundle_internal_conflict_is_detected(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["attempts"].append(dict(data["attempts"][0], attempt_id="att-1-dup", title="different title, same PR"))
    with pytest.raises(IdConflictError) as excinfo:
        run(write_single(tmp_path, data))
    assert excinfo.value.details["record_id"] == "pr-gh-42-pr-7"


# --------------------------------------------------------------------------------------
# Input formats and report
# --------------------------------------------------------------------------------------
def test_yaml_input_variant_matches_json(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml", reason="PyYAML is required for the YAML input variant")
    data = synthetic_v01()
    json_report = run(write_single(tmp_path / "json", data) if (tmp_path / "json").mkdir() is None else None)
    yaml_path = tmp_path / "v01.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    yaml_report = run(yaml_path)
    assert yaml_report.records == json_report.records
    assert yaml_report.unresolved == json_report.unresolved and yaml_report.rejections == json_report.rejections
    assert list(yaml_report.source_file_digests) == ["v01.yaml"]
    assert yaml_report.source_file_digests != json_report.source_file_digests


def test_directory_input_merges_files_and_reports_unknown_keys(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml", reason="PyYAML is required for the mixed-format directory variant")
    data = synthetic_v01()
    src = tmp_path / "v01-export"
    src.mkdir()
    (src / "kernels.json").write_text(json.dumps(data["kernels"]), encoding="utf-8")  # bare list named after the key
    (src / "configs.json").write_text(json.dumps({"configs": data["configs"]}), encoding="utf-8")
    (src / "attempts.yaml").write_text(yaml.safe_dump({"attempts": data["attempts"]}), encoding="utf-8")
    (src / "revisions-a.json").write_text(json.dumps({"revisions": data["revisions"][:2]}), encoding="utf-8")
    (src / "revisions-b.json").write_text(json.dumps({"revisions": data["revisions"][2:]}), encoding="utf-8")
    (src / "results.json").write_text(json.dumps(data["results"]), encoding="utf-8")
    (src / "trajectory.json").write_text(json.dumps(data["trajectory"]), encoding="utf-8")
    (src / "misc.json").write_text(json.dumps({"legacy_stats": data["legacy_stats"]}), encoding="utf-8")
    (src / "README.txt").write_text("ignored", encoding="utf-8")
    merged = read_v01_source(src)
    assert merged["kernels"] == data["kernels"] and merged["revisions"] == data["revisions"]
    assert merged["trajectory"] == data["trajectory"] and merged["legacy_stats"] == data["legacy_stats"]
    report = run(src)
    single = run(write_single(tmp_path, data))
    assert report.records == single.records
    assert set(report.source_file_digests) == {
        "kernels.json", "configs.json", "attempts.yaml", "revisions-a.json", "revisions-b.json", "results.json", "trajectory.json", "misc.json",
    }
    assert report.discarded_fields["top_level"]["legacy_stats"].startswith("unknown v0.1 top-level key")
    assert "rebuilt" in report.discarded_fields["top_level"]["trajectory"]


def test_directory_conflicting_scalar_keys_are_rejected(tmp_path: Path) -> None:
    src = tmp_path / "conflict"
    src.mkdir()
    (src / "a.json").write_text(json.dumps({"kernels": [], "legacy_stats": 1}), encoding="utf-8")
    (src / "b.json").write_text(json.dumps({"legacy_stats": 2}), encoding="utf-8")
    with pytest.raises(InputError) as excinfo:
        read_v01_source(src)
    assert excinfo.value.code == "V01_SOURCE_CONFLICT"


def test_report_digests_present_and_to_dict_json_serialisable(tmp_path: Path) -> None:
    path = write_single(tmp_path, synthetic_v01())
    report = run(path)
    assert report.source_file_digests == source_file_digests(path)
    (digest,) = report.source_file_digests.values()
    assert digest.startswith("sha256:") and len(digest) == 71
    as_dict = report.to_dict()
    json.dumps(as_dict)  # serialisable for --json output
    assert as_dict["source_path"] == str(path) and as_dict["dry_run"] is True
    assert as_dict["record_counts"] == report.record_counts()
    assert as_dict["annotations_for_unverified"] == 3
    assert {m["status"] for m in as_dict["identity_map"]} <= {"mapped", "unresolved", "rejected"}
    assert all(set(m) == {"v01_kind", "v01_id", "v02_record_type", "v02_record_id", "status"} for m in as_dict["identity_map"])
    assert all({"v01_id", "reason"} <= set(u) for u in as_dict["unresolved"] + as_dict["rejections"])
    assert as_dict["discarded_fields"]["top_level"]["legacy_stats"]
    assert "trajectory" in as_dict["discarded_fields"]["top_level"]
    assert "commit_sha" in as_dict["retained_fields"]["revisions"]
    assert any("confirm it against the real export" in note for note in as_dict["notes"])


def test_missing_source_and_invalid_files_raise_input_errors(tmp_path: Path) -> None:
    with pytest.raises(InputError) as excinfo:
        migrate_v01(tmp_path / "does-not-exist.json")
    assert excinfo.value.code == "FILE_NOT_FOUND" and excinfo.value.exit_code == 2
    dup = tmp_path / "dup.json"
    dup.write_text('{"kernels": [], "kernels": []}', encoding="utf-8")
    with pytest.raises(InputError) as excinfo:
        migrate_v01(dup)
    assert excinfo.value.code == "DUPLICATE_JSON_KEY"
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(InputError) as excinfo:
        migrate_v01(scalar)
    assert excinfo.value.code == "V01_SOURCE_INVALID"
    wrong_type = write_single(tmp_path, {"attempts": "not-a-list"}, name="wrong.json")
    with pytest.raises(InputError) as excinfo:
        migrate_v01(wrong_type)
    assert excinfo.value.code == "V01_SOURCE_INVALID"
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(InputError) as excinfo:
        migrate_v01(empty_dir)
    assert excinfo.value.code == "EMPTY_SOURCE"
    with pytest.raises(InputError):
        migrate_v01(write_single(tmp_path, synthetic_v01()), created_at="yesterday")
    with pytest.raises(InputError):
        migrate_v01(write_single(tmp_path, synthetic_v01()), repo_uid_map={REPO_NAME: "bad uid with spaces"})


def test_symlinked_directory_entry_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(synthetic_v01()), encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    try:
        os.symlink(outside, src / "linked.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this filesystem")
    with pytest.raises(UnsafePathError):
        read_v01_source(src)


def test_default_created_at_is_now_and_valid(tmp_path: Path) -> None:
    report = migrate_v01(write_single(tmp_path, synthetic_v01()))
    stamps = {r["created_at"] for r in report.records}
    assert len(stamps) == 1 and next(iter(stamps)).endswith("Z")


def test_entries_keyed_by_id_are_accepted(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["kernels"] = {k["kernel_id"]: {f: v for f, v in k.items() if f != "kernel_id"} for k in data["kernels"]}
    report = run(write_single(tmp_path, data))
    assert [k["payload"]["kernel_id"] for k in by_type(report, "kernel")] == ["demo_vector_add"]
    assert any("converted to a list" in note for note in report.notes)


def test_unknown_entry_fields_are_reported_not_dropped_silently(tmp_path: Path) -> None:
    data = synthetic_v01()
    data["attempts"][0]["owner"] = "someone"
    data["results"][0]["gpu_hours"] = 3
    report = run(write_single(tmp_path, data))
    assert report.discarded_fields["attempts"]["owner"] == "unknown v0.1 field; not migrated"
    assert report.discarded_fields["results"]["gpu_hours"] == "unknown v0.1 field; not migrated"
    # unknown result fields still travel verbatim inside the annotation text (as data)
    text = next(a["payload"]["text"] for a in by_type(report, "annotation") if '"gpu_hours"' in a["payload"]["text"])
    assert json.loads(text[len(RESULT_TEXT_PREFIX) :])["v01_result"]["gpu_hours"] == 3


# --------------------------------------------------------------------------------------
# Resolvers
# --------------------------------------------------------------------------------------
def test_mapping_and_null_resolvers() -> None:
    res = resolver()
    assert isinstance(res, ShaResolver) and isinstance(NullResolver(), ShaResolver)
    assert res.resolve(REPO_UID, SHORT_2) == GitOid("sha1", FULL_2)
    assert res.resolve(REPO_UID, SHORT_2.upper()) == GitOid("sha1", FULL_2)
    assert res.resolve(REPO_UID, SHORT_3) is None and res.is_ambiguous(REPO_UID, SHORT_3)
    assert res.resolve(REPO_UID, "cafe1") == GitOid("sha1", FULL_3A) and not res.is_ambiguous(REPO_UID, "cafe1")
    assert res.resolve("github:github.com:repo:99", SHORT_2) is None  # other repository
    assert res.resolve(REPO_UID, "zz") is None and not res.is_ambiguous(REPO_UID, "zz")
    assert NullResolver().resolve(REPO_UID, SHORT_2) is None and not NullResolver().is_ambiguous(REPO_UID, SHORT_2)
    with pytest.raises(InputError):
        MappingResolver({(REPO_UID, "beef"): "beef"})  # not a full object id
    with pytest.raises(InputError):
        MappingResolver({(REPO_UID, "dead"): FULL_2})  # prefix does not match its value
    with pytest.raises(InputError):
        MappingResolver({"not-a-tuple": FULL_2})  # type: ignore[dict-item]


def test_resolve_sha_helper_rules() -> None:
    res = resolver()
    assert resolve_sha(FULL_1.upper(), REPO_UID, NullResolver()) == (GitOid("sha1", FULL_1), None)
    assert resolve_sha("ab" * 32, REPO_UID, NullResolver()) == (GitOid("sha256", "ab" * 32), None)
    oid, reason = resolve_sha(SHORT_3, REPO_UID, res)
    assert oid is None and "ambiguous" in reason
    oid, reason = resolve_sha("feed", REPO_UID, res)
    assert oid is None and "could not be resolved" in reason
    oid, reason = resolve_sha("", REPO_UID, res)
    assert oid is None and "missing" in reason
    oid, reason = resolve_sha("g" * 40, REPO_UID, res)
    assert oid is None and "not hexadecimal" in reason

    class LyingResolver:
        def resolve(self, repo_uid: str, sha: str) -> GitOid | None:
            return GitOid("sha1", FULL_1)  # does not extend the short sha

        def is_ambiguous(self, repo_uid: str, sha: str) -> bool:
            return False

    with pytest.raises(InputError) as excinfo:
        resolve_sha(SHORT_2, REPO_UID, LyingResolver())
    assert excinfo.value.code == "RESOLVER_INVALID"
