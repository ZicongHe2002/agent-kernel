"""Typed record models (spec 6): round trips, strict field validation, references, canonical digest (T32)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import record_dict

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, SchemaValidationError
from kernel_memory.domain.jsonio import load_json_file
from kernel_memory.domain.models import (
    ANY_RECORD,
    PAYLOAD_TYPES,
    TYPE_ORDER,
    AnnotationPayload,
    ArtifactRef,
    BaselinePayload,
    CommitPayload,
    ConfigPayload,
    DecisionPayload,
    GitOid,
    KernelPayload,
    PrPayload,
    PrSnapshotPayload,
    Record,
    Reference,
    RelationPayload,
    RunPayload,
    records_from_iterable,
    to_json,
)

FIXTURE_RECORD_IDS = [
    "kernel-demo",
    "cfg-demo",
    "baseline-demo",
    "pr-demo-101",
    "commit-demo-a",
    "commit-demo-b",
    "snapshot-demo-101",
    "pr-demo-102",
    "commit-demo-c",
    "commit-demo-a-in-102",
    "snapshot-demo-102",
    "relation-demo-origin",
    "run-demo-baseline",
    "run-demo-a",
    "run-demo-c",
    "run-demo-c-failure",
    "decision-demo-blocked",
    "annotation-demo-grouped",
]


# --------------------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------------------
def test_fixture_bundle_has_eighteen_records(bundle_dicts: list[dict]) -> None:
    assert [r["record_id"] for r in bundle_dicts] == FIXTURE_RECORD_IDS
    assert len(bundle_dicts) == 18


@pytest.mark.parametrize("record_id", FIXTURE_RECORD_IDS)
def test_fixture_record_round_trips_to_identical_dict(bundle_dicts: list[dict], record_id: str) -> None:
    original = record_dict(bundle_dicts, record_id)
    record = Record.from_dict(json.loads(json.dumps(original)))
    assert record.record_id == record_id
    assert record.record_type == original["record_type"]
    assert isinstance(record.payload, PAYLOAD_TYPES[record.record_type])
    assert record.to_dict() == original
    # A second round trip through the typed model is stable too.
    assert Record.from_dict(record.to_dict()).to_dict() == original


def test_records_from_iterable_matches_individual_construction(bundle_dicts: list[dict]) -> None:
    records = records_from_iterable(json.loads(json.dumps(bundle_dicts)))
    assert [r.record_id for r in records] == FIXTURE_RECORD_IDS
    assert all(isinstance(r, Record) for r in records)


def test_payload_type_table_covers_every_record_type() -> None:
    assert set(PAYLOAD_TYPES) == set(ANY_RECORD)
    for record_type, cls in PAYLOAD_TYPES.items():
        assert cls.record_type == record_type
    assert [t for t, _ in sorted(TYPE_ORDER.items(), key=lambda kv: kv[1])] == list(ANY_RECORD)


def test_typed_access_on_run_record(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-a")
    payload = run.payload
    assert isinstance(payload, RunPayload)
    assert payload.provenance == "fixture"
    assert payload.timing.median_us == 90
    assert payload.source.tested_commit == GitOid("sha1", "0000000000000000000000000000000000000002")
    assert payload.source.tested_commit.short() == "000000000000"
    assert payload.is_terminal_success
    assert set(payload.artifact_by_id()) == {"run-demo-a-samples", "run-demo-a-correctness"}
    assert isinstance(payload.artifact_by_id()["run-demo-a-samples"], ArtifactRef)


def test_t14_failed_run_carries_no_invented_numbers(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-c-failure")
    assert run.payload.execution_status == "compile_error"
    assert not run.payload.is_terminal_success
    assert run.payload.timing.median_us is None and run.payload.timing.p90_us is None
    assert run.payload.correctness.max_abs_error is None
    assert run.payload.artifacts == []


def test_t17_uncollected_metric_is_null_not_zero(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-a")
    metric = run.payload.analysis_metrics[0]
    assert metric.status == "not_collected"
    assert metric.value is None


def test_payload_dataclasses_are_frozen(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-a")
    with pytest.raises(Exception):
        run.payload.execution_status = "failed"  # type: ignore[misc]
    with pytest.raises(Exception):
        run.record_id = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Rejections (T32)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("record_id", ["kernel-demo", "cfg-demo", "run-demo-a", "decision-demo-blocked", "annotation-demo-grouped"])
def test_t32_unknown_payload_field_is_schema_invalid(bundle_dicts: list[dict], record_id: str) -> None:
    data = record_dict(bundle_dicts, record_id)
    data["payload"]["misspelled_property"] = "x"
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(data)
    assert info.value.code == "SCHEMA_INVALID"
    assert info.value.exit_code == 2
    assert "misspelled_property" in str(info.value)


def test_t32_unknown_envelope_field_is_schema_invalid(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["extra_envelope_field"] = 1
    with pytest.raises(SchemaValidationError):
        Record.from_dict(data)


def test_t32_unknown_nested_field_is_schema_invalid(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["payload"]["timing"]["mean_us"] = 90.0
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(data)
    assert "mean_us" in str(info.value)


def test_missing_required_payload_field_is_schema_invalid(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    del data["payload"]["contract_notes"]
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(data)
    assert "contract_notes" in str(info.value)


@pytest.mark.parametrize(
    ("record_id", "path"),
    [
        ("run-demo-a", ("payload", "environment", "device_count")),
        ("run-demo-a", ("payload", "timing", "sample_count")),
        ("run-demo-a", ("payload", "attempt_no")),
        ("run-demo-a", ("payload", "protocol", "repetitions")),
        ("decision-demo-blocked", ("payload", "policy", "min_confirm_pairs")),
    ],
)
def test_t32_boolean_dimension_is_rejected(bundle_dicts: list[dict], record_id: str, path: tuple[str, ...]) -> None:
    data = record_dict(bundle_dicts, record_id)
    node = data
    for key in path[:-1]:
        node = node[key]
    assert isinstance(node[path[-1]], int)
    node[path[-1]] = True
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(data)
    assert path[-1] in str(info.value)


def test_boolean_dimension_is_rejected_by_typed_conversion_even_without_schema() -> None:
    """The dataclass layer is a second line of defence: bool never satisfies int."""
    from kernel_memory.domain.models import _from_json  # noqa: PLC2701 - internal guard under test

    with pytest.raises(SchemaValidationError):
        _from_json(GitOid, {"algorithm": "sha1", "hex": True}, "x")
    with pytest.raises(SchemaValidationError):
        _from_json(ArtifactRef, {
            "artifact_id": "a", "kind": "k", "uri": "u", "sha256": "sha256:" + "0" * 64,
            "size_bytes": True, "media_type": "m", "retention": "permanent", "availability": "present",
        }, "x")


def test_naive_timestamp_is_invalid_timestamp(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = "2026-09-08T00:00:00"
    with pytest.raises(InputError) as info:
        Record.from_dict(data)
    assert info.value.code == "INVALID_TIMESTAMP"
    assert info.value.exit_code == 2


def test_offset_timestamp_is_accepted(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = "2026-09-08T02:00:00+02:00"
    assert Record.from_dict(data).created_at == "2026-09-08T02:00:00+02:00"


def test_garbage_timestamp_is_invalid_timestamp(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = "yesterday"
    with pytest.raises(InputError) as info:
        Record.from_dict(data)
    assert info.value.code == "INVALID_TIMESTAMP"


def test_mla_config_draft_is_not_a_record(fixtures_root: Path) -> None:
    draft = load_json_file(fixtures_root / "examples" / "mla_config.draft.json")
    assert draft["status"] == "draft_not_registrable"
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(draft)
    assert info.value.code == "SCHEMA_INVALID"
    assert "record_type" in str(info.value)


def test_mla_draft_dimensions_wrapped_as_config_record_are_rejected(bundle_dicts: list[dict], fixtures_root: Path) -> None:
    """Wrapping the historical dimensions into a config envelope does not make them a valid Config."""
    draft = load_json_file(fixtures_root / "examples" / "mla_config.draft.json")
    data = record_dict(bundle_dicts, "cfg-demo")
    data["payload"] = draft  # unknown fields status/known_context_only/unresolved/instruction
    with pytest.raises(SchemaValidationError):
        Record.from_dict(data)


@pytest.mark.parametrize("bad", [None, [], "kernel", 42])
def test_non_object_record_is_rejected(bad: object) -> None:
    with pytest.raises(SchemaValidationError):
        Record.from_dict(bad)


def test_unknown_record_type_is_rejected(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_type"] = "experiment"
    with pytest.raises(SchemaValidationError) as info:
        Record.from_dict(data)
    assert "experiment" in str(info.value)
    assert info.value.details["known"] == list(ANY_RECORD)


def test_wrong_schema_version_is_rejected(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["schema_version"] = "0.1.0"
    with pytest.raises(SchemaValidationError):
        Record.from_dict(data)


def test_invalid_record_id_pattern_is_rejected(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_id"] = "../escape"
    with pytest.raises(SchemaValidationError):
        Record.from_dict(data)


def test_payload_type_mismatch_is_rejected(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "cfg-demo")
    data["payload"]["tags"] = "fixture"  # must be an array
    with pytest.raises(SchemaValidationError):
        Record.from_dict(data)


# --------------------------------------------------------------------------------------
# references()
# --------------------------------------------------------------------------------------
def _refs(record: Record) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {r.field: (r.target, r.allowed_types) for r in record.references()}


def test_run_references_subject_and_config(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-a")
    refs = _refs(run)
    assert refs == {
        "subject_ref": ("commit-demo-a", ("commit", "baseline")),
        "config_ref": ("cfg-demo", ("config",)),
    }
    assert run.artifact_references() == []
    assert len(run.record_references()) == 2


def test_run_rerun_of_is_a_run_reference(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["payload"]["rerun_of"] = "run-demo-baseline"
    data["payload"]["attempt_no"] = 2
    refs = _refs(Record.from_dict(data))
    assert refs["rerun_of"] == ("run-demo-baseline", ("run",))


def test_baseline_run_references_baseline_subject(bundle_records: list[Record]) -> None:
    run = next(r for r in bundle_records if r.record_id == "run-demo-baseline")
    assert _refs(run)["subject_ref"] == ("baseline-demo", ("commit", "baseline"))


def test_snapshot_references_pr_and_commit_refs(bundle_records: list[Record]) -> None:
    snapshot = next(r for r in bundle_records if r.record_id == "snapshot-demo-101")
    refs = _refs(snapshot)
    assert refs["pr_ref"] == ("pr-demo-101", ("pr",))
    assert refs["commit_refs[0]"] == ("commit-demo-a", ("commit",))
    assert refs["commit_refs[1]"] == ("commit-demo-b", ("commit",))
    assert "previous_snapshot_ref" not in refs  # null in the fixture


def test_snapshot_previous_snapshot_reference(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "snapshot-demo-102")
    data["payload"]["previous_snapshot_ref"] = "snapshot-demo-101"
    assert _refs(Record.from_dict(data))["previous_snapshot_ref"] == ("snapshot-demo-101", ("pr_snapshot",))


def test_decision_references_runs_config_and_subject(bundle_records: list[Record]) -> None:
    decision = next(r for r in bundle_records if r.record_id == "decision-demo-blocked")
    refs = _refs(decision)
    assert refs["config_ref"] == ("cfg-demo", ("config",))
    assert refs["candidate_subject_ref"] == ("commit-demo-a", ("commit", "baseline"))
    assert refs["candidate_run_refs[0]"] == ("run-demo-a", ("run",))
    assert refs["baseline_run_refs[0]"] == ("run-demo-baseline", ("run",))
    assert "supersedes_decision_ref" not in refs


def test_decision_supersedes_reference(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "decision-demo-blocked")
    data["record_id"] = "decision-demo-2"
    data["payload"]["supersedes_decision_ref"] = "decision-demo-blocked"
    assert _refs(Record.from_dict(data))["supersedes_decision_ref"] == ("decision-demo-blocked", ("decision",))


def test_annotation_references_target_and_evidence(bundle_records: list[Record]) -> None:
    annotation = next(r for r in bundle_records if r.record_id == "annotation-demo-grouped")
    refs = _refs(annotation)
    assert refs["target_ref"] == ("commit-demo-c", ANY_RECORD)
    assert refs["evidence_refs[0]"] == ("run-demo-c", ANY_RECORD)


def test_annotation_supersedes_reference(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "annotation-demo-grouped")
    data["payload"]["supersedes_ref"] = "annotation-demo-old"
    assert _refs(Record.from_dict(data))["supersedes_ref"] == ("annotation-demo-old", ("annotation",))


def test_commit_diff_artifact_ref_is_an_artifact_reference(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "commit-demo-a")
    data["payload"]["diff_artifact_ref"] = "commit-demo-a-diff"
    commit = Record.from_dict(data)
    refs = commit.references()
    by_field = {r.field: r for r in refs}
    assert by_field["pr_ref"].target == "pr-demo-101" and by_field["pr_ref"].allowed_types == ("pr",)
    assert by_field["diff_artifact_ref"].allowed_types == ("artifact",)
    assert commit.artifact_references() == [Reference("diff_artifact_ref", "commit-demo-a-diff", ("artifact",))]
    assert [r.field for r in commit.record_references()] == ["pr_ref"]


def test_commit_without_diff_has_only_pr_reference(bundle_records: list[Record]) -> None:
    commit = next(r for r in bundle_records if r.record_id == "commit-demo-b")
    assert _refs(commit) == {"pr_ref": ("pr-demo-101", ("pr",))}
    assert commit.payload.source_key == ("github:github.com:repo:900001", "sha1", "0000000000000000000000000000000000000003")


def test_t07_same_source_commit_in_two_prs_has_distinct_records_and_memberships(bundle_records: list[Record]) -> None:
    a = next(r for r in bundle_records if r.record_id == "commit-demo-a")
    a_in_102 = next(r for r in bundle_records if r.record_id == "commit-demo-a-in-102")
    assert a.payload.source_key == a_in_102.payload.source_key
    assert a.payload.pr_ref != a_in_102.payload.pr_ref
    assert a.canonical_digest() != a_in_102.canonical_digest()


def test_pr_relation_baseline_kernel_config_references(bundle_records: list[Record]) -> None:
    by_id = {r.record_id: r for r in bundle_records}
    assert _refs(by_id["kernel-demo"]) == {}
    assert _refs(by_id["cfg-demo"]) == {}
    assert _refs(by_id["baseline-demo"]) == {"config_ref": ("cfg-demo", ("config",))}
    pr_refs = _refs(by_id["pr-demo-102"])
    assert pr_refs["config_ref"] == ("cfg-demo", ("config",))
    if by_id["pr-demo-102"].payload.origin_ref is not None:
        assert pr_refs["origin_ref"][1] == ("commit", "baseline")
    relation_refs = _refs(by_id["relation-demo-origin"])
    assert relation_refs["config_ref"] == ("cfg-demo", ("config",))
    assert relation_refs["from_ref"][1] == ANY_RECORD
    assert relation_refs["to_ref"][1] == ANY_RECORD


def test_pr_local_trial_flag(bundle_records: list[Record]) -> None:
    pr = next(r for r in bundle_records if r.record_id == "pr-demo-101")
    assert isinstance(pr.payload, PrPayload)
    assert pr.payload.is_local_trial is (pr.payload.provider == "local")


# --------------------------------------------------------------------------------------
# Record.create validates typed payloads
# --------------------------------------------------------------------------------------
def test_record_create_from_typed_payload_round_trips() -> None:
    payload = KernelPayload(kernel_id="k1", display_name="Kernel One", adapter_id="adapter-v1", contract_notes="n")
    record = Record.create("kernel", "kernel-k1", payload, created_at="2026-09-08T00:00:00Z")
    assert record.schema_version == "0.2.0"
    assert record.payload == payload
    assert record.to_dict()["payload"] == to_json(payload)
    assert Record.from_dict(record.to_dict()) == record


def test_record_create_rejects_invalid_typed_payload() -> None:
    bad = KernelPayload(kernel_id="k1", display_name="x", adapter_id="not valid!", contract_notes="n")
    with pytest.raises(SchemaValidationError):
        Record.create("kernel", "kernel-k1", bad, created_at="2026-09-08T00:00:00Z")


def test_record_create_rejects_naive_timestamp() -> None:
    payload = KernelPayload(kernel_id="k1", display_name="x", adapter_id="a", contract_notes="n")
    with pytest.raises(InputError) as info:
        Record.create("kernel", "kernel-k1", payload, created_at="2026-09-08T00:00:00")
    assert info.value.code == "INVALID_TIMESTAMP"


def test_record_create_rejects_unknown_type_and_mismatched_payload() -> None:
    payload = KernelPayload(kernel_id="k1", display_name="x", adapter_id="a", contract_notes="n")
    with pytest.raises(InputError):
        Record.create("experiment", "e1", payload, created_at="2026-09-08T00:00:00Z")
    with pytest.raises(InputError):
        Record.create("config", "cfg-x", payload, created_at="2026-09-08T00:00:00Z")


def test_record_create_config_with_bool_dimension_passes_schema_but_hash_is_distinct(bundle_dicts: list[dict]) -> None:
    """The record schema keeps `problem` opaque; the problem adapter is the gate for dimensions (see test_problems)."""
    cfg = record_dict(bundle_dicts, "cfg-demo")["payload"]
    payload = ConfigPayload(**dict(cfg, problem=dict(cfg["problem"], n=True)))
    record = Record.create("config", "cfg-bool", payload, created_at="2026-09-08T00:00:00Z")
    assert record.payload.problem["n"] is True
    assert hashing.jcs_digest(record.payload.problem) != hashing.jcs_digest(cfg["problem"])


# --------------------------------------------------------------------------------------
# canonical_digest
# --------------------------------------------------------------------------------------
def test_canonical_digest_independent_of_key_order_and_formatting(bundle_dicts: list[dict]) -> None:
    original = record_dict(bundle_dicts, "run-demo-a")
    record = Record.from_dict(json.loads(json.dumps(original)))
    # Reverse key order at every level, and re-serialise with different whitespace/indent.
    def reverse(value):
        if isinstance(value, dict):
            return {k: reverse(value[k]) for k in reversed(list(value))}
        if isinstance(value, list):
            return [reverse(v) for v in value]
        return value

    reordered = json.loads(json.dumps(reverse(original), indent=4))
    assert list(reordered) != list(original)
    assert Record.from_dict(reordered).canonical_digest() == record.canonical_digest()
    assert record.canonical_digest() == hashing.jcs_digest(original)
    assert hashing.is_sha256_ref(record.canonical_digest())


def test_canonical_digest_changes_with_content(bundle_dicts: list[dict]) -> None:
    original = record_dict(bundle_dicts, "run-demo-a")
    changed = json.loads(json.dumps(original))
    changed["payload"]["timing"]["median_us"] = 89
    assert Record.from_dict(changed).canonical_digest() != Record.from_dict(original).canonical_digest()


def test_canonical_digest_treats_integer_valued_floats_as_equal(bundle_dicts: list[dict]) -> None:
    original = record_dict(bundle_dicts, "run-demo-a")
    as_float = json.loads(json.dumps(original))
    as_float["payload"]["timing"]["median_us"] = 90.0
    assert Record.from_dict(as_float).canonical_digest() == Record.from_dict(original).canonical_digest()


def test_all_fixture_digests_are_distinct(bundle_records: list[Record]) -> None:
    digests = {r.canonical_digest() for r in bundle_records}
    assert len(digests) == len(bundle_records) == 18


def test_to_json_handles_dataclasses_tuples_and_nested_dicts() -> None:
    oid = GitOid("sha1", "0" * 40)
    assert to_json(oid) == {"algorithm": "sha1", "hex": "0" * 40}
    assert to_json((oid, [oid])) == [to_json(oid), [to_json(oid)]]
    assert to_json({"k": {"v": oid}}) == {"k": {"v": to_json(oid)}}
    assert to_json(GitOid) is GitOid  # the class itself is not serialised


def test_every_payload_type_has_references_method() -> None:
    for cls in (
        KernelPayload, ConfigPayload, PrPayload, PrSnapshotPayload, CommitPayload, BaselinePayload,
        RunPayload, RelationPayload, DecisionPayload, AnnotationPayload,
    ):
        assert callable(getattr(cls, "references"))
