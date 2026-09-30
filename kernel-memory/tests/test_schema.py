"""JSON Schema (2020-12) validation of wire records and nested contract objects (DESIGN section 3; ADR-0004).

The current contract is 0.3.0 (kernel -> algorithm -> config/shape); the verbatim 0.2.0 handoff
contract is retained under ``contracts/legacy/`` for legacy *input* only and pinned by digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from conftest import record_dict
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from kernel_memory.domain import schema as schema_mod
from kernel_memory.domain.errors import InputError, SchemaValidationError
from kernel_memory.domain.schema import (
    LEGACY_RECORD_TYPES,
    LEGACY_SCHEMA_VERSION,
    RECORD_TYPES,
    SCHEMA_VERSION,
    demo_problem_schema,
    golden_hash_vectors,
    is_valid_record_dict,
    legacy_record_schema,
    record_schema,
    validate_against,
    validate_legacy_record_dict,
    validate_nested,
    validate_record_dict,
)

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
# sha256 of the verbatim handoff record.schema.json (0.2.0), retained byte-for-byte under contracts/legacy/.
LEGACY_SCHEMA_SHA256 = "e24f4cd5bc52f6f0b8b8c40ee123f2760e3a1371bae9e15b981c16e092ecbf77"


# --------------------------------------------------------------------------------------
# Schema loading
# --------------------------------------------------------------------------------------
def test_record_schema_loads_and_is_valid_2020_12() -> None:
    schema = record_schema()
    assert isinstance(schema, dict)
    assert schema["$schema"] == DRAFT_2020_12
    Draft202012Validator.check_schema(schema)  # raises SchemaError if invalid
    assert set(RECORD_TYPES) <= set(schema["$defs"])
    # every record type is reachable from the top-level oneOf, in RECORD_TYPES order
    refs = {alt["$ref"] for alt in schema["oneOf"]}
    assert refs == {f"#/$defs/{t}" for t in RECORD_TYPES}
    assert [alt["$ref"] for alt in schema["oneOf"]] == [f"#/$defs/{t}" for t in RECORD_TYPES]
    assert len(RECORD_TYPES) == 11
    assert RECORD_TYPES[:3] == ("kernel", "algorithm", "config")
    assert schema["$id"] == "urn:kernel-memory:records:0.3.0"
    assert SCHEMA_VERSION == "0.3.0" and LEGACY_SCHEMA_VERSION == "0.2.0"


def test_record_schema_is_cached_and_pins_schema_version() -> None:
    assert record_schema() is record_schema()
    for record_type in RECORD_TYPES:
        props = record_schema()["$defs"][record_type]["properties"]
        assert props["schema_version"] == {"const": schema_mod.SCHEMA_VERSION}
        assert props["record_type"] == {"const": record_type}


def test_demo_problem_schema_loads_and_is_valid_2020_12() -> None:
    schema = demo_problem_schema()
    assert schema["$schema"] == DRAFT_2020_12
    Draft202012Validator.check_schema(schema)
    assert schema["$id"] == "urn:kernel-memory:problem:demo-vector-add:v1"
    assert set(schema["required"]) == {"n", "dtype", "operation", "outputs"}
    assert demo_problem_schema() is demo_problem_schema()


def test_check_schema_actually_rejects_invalid_schemas() -> None:
    # Guard: the assertion above is meaningful only if check_schema can fail.
    with pytest.raises(SchemaError):
        Draft202012Validator.check_schema({"type": "not-a-type"})


def test_golden_hash_vectors_has_four_vectors() -> None:
    vectors = golden_hash_vectors()
    assert isinstance(vectors, dict)
    assert len(vectors["vectors"]) == 4
    names = [v["name"] for v in vectors["vectors"]]
    assert len(set(names)) == 4
    for vector in vectors["vectors"]:
        assert set(vector) >= {"name", "payload", "digest"}
        assert isinstance(vector["digest"], str) and vector["digest"].startswith("sha256:")
        assert len(vector["digest"]) == len("sha256:") + 64
    assert golden_hash_vectors() is golden_hash_vectors()


# --------------------------------------------------------------------------------------
# validate_record_dict
# --------------------------------------------------------------------------------------
def test_validate_record_dict_accepts_every_fixture_record(bundle_dicts: list[dict]) -> None:
    seen: set[str] = set()
    for data in bundle_dicts:
        assert validate_record_dict(copy.deepcopy(data)) == data["record_type"], data["record_id"]
        seen.add(data["record_type"])
    # the fixture bundle exercises every record type
    assert seen == set(RECORD_TYPES)


def test_validate_record_dict_does_not_mutate_input(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    before = json.dumps(data, sort_keys=True)
    validate_record_dict(data)
    assert json.dumps(data, sort_keys=True) == before


@pytest.mark.parametrize("value", [None, 1, "kernel", [], [{"record_type": "kernel"}], True])
def test_validate_record_dict_rejects_non_dict(value: Any) -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(value)
    assert excinfo.value.exit_code == 2
    assert "object" in str(excinfo.value)


def test_validate_record_dict_unknown_record_type_names_known_types(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_type"] = "widget"
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "widget" in str(excinfo.value)
    assert excinfo.value.details["known"] == list(RECORD_TYPES)
    assert set(excinfo.value.details["known"]) == set(RECORD_TYPES)


def test_validate_record_dict_missing_record_type_is_unknown_type() -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict({"schema_version": SCHEMA_VERSION, "record_id": "x"})
    assert "record_type" in str(excinfo.value)
    assert excinfo.value.details["known"] == list(RECORD_TYPES)


def test_validate_record_dict_record_type_is_case_sensitive(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_type"] = "Kernel"
    with pytest.raises(SchemaValidationError):
        validate_record_dict(data)


def test_validate_record_dict_unknown_payload_property_names_path(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["payload"]["bogus_property"] = 1
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    message = str(excinfo.value)
    assert "payload" in message
    assert "bogus_property" in message
    assert "kernel-demo" in message
    assert excinfo.value.details["record_type"] == "kernel"
    assert excinfo.value.details["record_id"] == "kernel-demo"


def test_validate_record_dict_nested_error_names_deep_path(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["payload"]["artifacts"][0]["not_a_field"] = True
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    message = str(excinfo.value)
    assert "payload/artifacts/0" in message
    assert "not_a_field" in message


def test_validate_record_dict_unknown_top_level_property(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["extra"] = "nope"
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "extra" in str(excinfo.value)


@pytest.mark.parametrize("missing", ["schema_version", "record_id", "created_at", "payload"])
def test_validate_record_dict_missing_envelope_key(bundle_dicts: list[dict], missing: str) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    del data[missing]
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert missing in str(excinfo.value)


def test_validate_record_dict_missing_payload_key(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    del data["payload"]["adapter_id"]
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "adapter_id" in str(excinfo.value)
    assert "payload" in str(excinfo.value)


def test_validate_record_dict_wrong_schema_version(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["schema_version"] = "0.1.0"
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "schema_version" in str(excinfo.value)


@pytest.mark.parametrize("bad_id", ["", "-leading-dash", "has space", "a/b", ".dot", "x" * 257])
def test_validate_record_dict_rejects_bad_record_id(bundle_dicts: list[dict], bad_id: str) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_id"] = bad_id
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "record_id" in str(excinfo.value)


def test_validate_record_dict_wrong_payload_type(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["payload"]["display_name"] = 42
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "payload/display_name" in str(excinfo.value)


def test_validate_record_dict_naive_created_at_is_invalid_timestamp(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = "2026-09-08T00:00:00"
    with pytest.raises(InputError) as excinfo:
        validate_record_dict(data)
    assert excinfo.value.code == "INVALID_TIMESTAMP"
    assert excinfo.value.exit_code == 2
    assert "created_at" in str(excinfo.value)


@pytest.mark.parametrize("stamp", ["yesterday", "2026-13-45T00:00:00Z", "2026-09-08", "", "2026-09-08T00:00:00 UTC"])
def test_validate_record_dict_malformed_created_at_rejected(bundle_dicts: list[dict], stamp: str) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = stamp
    with pytest.raises(InputError) as excinfo:
        validate_record_dict(data)
    assert excinfo.value.exit_code == 2
    assert excinfo.value.code in ("INVALID_TIMESTAMP", "SCHEMA_INVALID")


def test_validate_record_dict_non_string_created_at_is_schema_error(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = 1_700_000_000
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "created_at" in str(excinfo.value)


@pytest.mark.parametrize("stamp", ["2026-09-08T00:00:00Z", "2026-09-08T00:00:00+00:00", "2026-09-08T02:00:00+02:00"])
def test_validate_record_dict_accepts_timezone_aware_created_at(bundle_dicts: list[dict], stamp: str) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["created_at"] = stamp
    assert validate_record_dict(data) == "kernel"


def test_validate_record_dict_payload_mismatch_for_record_type(bundle_dicts: list[dict]) -> None:
    # A kernel payload declared as a config record must fail against the config definition.
    data = record_dict(bundle_dicts, "kernel-demo")
    data["record_type"] = "config"
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert excinfo.value.details["record_type"] == "config"


def test_validate_record_dict_rejects_legacy_version(legacy_bundle_dicts: list[dict]) -> None:
    raw = copy.deepcopy(next(d for d in legacy_bundle_dicts if d["record_id"] == "kernel-demo"))
    assert raw["schema_version"] == LEGACY_SCHEMA_VERSION
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(raw)
    assert "schema_version" in str(excinfo.value)
    assert excinfo.value.details["contract"] == SCHEMA_VERSION
    assert is_valid_record_dict(raw) is False


def test_current_contract_rejects_config_without_algorithm_ref(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "cfg-demo")
    del data["payload"]["algorithm_ref"]
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_record_dict(data)
    assert "algorithm_ref" in str(excinfo.value)
    assert excinfo.value.details["record_id"] == "cfg-demo"
    assert excinfo.value.details["record_type"] == "config"
    assert excinfo.value.details["contract"] == SCHEMA_VERSION
    assert is_valid_record_dict(data) is False


# --------------------------------------------------------------------------------------
# 0.3.0 contract shape and the retained 0.2.0 contract (ADR-0004)
# --------------------------------------------------------------------------------------
def test_algorithm_definition_shape() -> None:
    schema = record_schema()
    definition = schema["$defs"]["algorithm"]
    assert definition["properties"]["record_type"] == {"const": "algorithm"}
    payload = definition["properties"]["payload"]
    assert set(payload["required"]) == {"kernel_id", "algorithm_id", "display_name", "method_summary", "summary_author", "tags"}
    assert set(payload["properties"]) == set(payload["required"])
    assert payload["additionalProperties"] is False
    assert payload["properties"]["summary_author"]["enum"] == ["human", "agent", "program"]
    assert payload["properties"]["method_summary"]["minLength"] == 1
    assert payload["properties"]["algorithm_id"]["pattern"].endswith("{0,63}$")  # at most 64 characters
    assert payload["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert "references" not in payload["properties"]  # would shadow Payload.references()
    config_payload = schema["$defs"]["config"]["properties"]["payload"]
    assert "algorithm_ref" in config_payload["required"]
    assert config_payload["properties"]["algorithm_ref"] == config_payload["properties"]["kernel_id"]


def test_legacy_contract_is_verbatim_and_pinned() -> None:
    path = schema_mod.LEGACY_CONTRACTS_DIR / "record.schema.v0.2.0.json"
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LEGACY_SCHEMA_SHA256
    legacy = legacy_record_schema()
    assert legacy_record_schema() is legacy  # cached
    assert legacy["$schema"] == DRAFT_2020_12
    Draft202012Validator.check_schema(legacy)
    assert legacy["$id"] == "urn:kernel-memory:records:0.2.0"
    assert "algorithm" not in legacy["$defs"]
    assert LEGACY_RECORD_TYPES == tuple(t for t in RECORD_TYPES if t != "algorithm")
    assert len(LEGACY_RECORD_TYPES) == 10
    assert [alt["$ref"] for alt in legacy["oneOf"]] == [f"#/$defs/{t}" for t in LEGACY_RECORD_TYPES]
    for record_type in LEGACY_RECORD_TYPES:
        assert legacy["$defs"][record_type]["properties"]["schema_version"] == {"const": LEGACY_SCHEMA_VERSION}
    legacy_config_payload = legacy["$defs"]["config"]["properties"]["payload"]
    assert "algorithm_ref" not in legacy_config_payload["required"]
    assert "algorithm_ref" not in legacy_config_payload["properties"]
    # The two contracts are distinct objects with distinct ids: the legacy one is never used for publication.
    assert record_schema()["$id"] != legacy["$id"]


def test_validate_legacy_record_dict_accepts_verbatim_fixture(legacy_bundle_dicts: list[dict]) -> None:
    assert len(legacy_bundle_dicts) == 18
    seen: set[str] = set()
    for data in legacy_bundle_dicts:
        before = json.dumps(data, sort_keys=True)
        assert validate_legacy_record_dict(copy.deepcopy(data)) == data["record_type"], data["record_id"]
        assert validate_legacy_record_dict(data) == data["record_type"]
        assert json.dumps(data, sort_keys=True) == before  # no mutation
        seen.add(data["record_type"])
    assert seen == set(LEGACY_RECORD_TYPES)
    # Legacy records are NOT valid under the current contract: they must be upgraded first.
    assert not any(is_valid_record_dict(d) for d in legacy_bundle_dicts)


def test_validate_legacy_record_dict_rejects_algorithm_and_current_records(bundle_dicts: list[dict]) -> None:
    algorithm = next(d for d in bundle_dicts if d["record_type"] == "algorithm")
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_legacy_record_dict(copy.deepcopy(algorithm))
    assert "algorithm" in str(excinfo.value)
    assert excinfo.value.details["known"] == list(LEGACY_RECORD_TYPES)
    assert excinfo.value.details["contract"] == LEGACY_SCHEMA_VERSION
    # An upgraded (0.3.0) record fails the legacy contract on its schema_version.
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_legacy_record_dict(record_dict(bundle_dicts, "kernel-demo"))
    assert "schema_version" in str(excinfo.value)
    assert excinfo.value.details["contract"] == LEGACY_SCHEMA_VERSION
    # A config carrying algorithm_ref is not a legacy config even with the legacy version stamp.
    data = record_dict(bundle_dicts, "cfg-demo")
    data["schema_version"] = LEGACY_SCHEMA_VERSION
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_legacy_record_dict(data)
    assert "algorithm_ref" in str(excinfo.value)
    with pytest.raises(SchemaValidationError):
        validate_legacy_record_dict(None)


# --------------------------------------------------------------------------------------
# validate_nested
# --------------------------------------------------------------------------------------
def _nested_sample(bundle_dicts: list[dict], def_name: str) -> dict:
    run = record_dict(bundle_dicts, "run-demo-a")
    if def_name == "Environment":
        return run["payload"]["environment"]
    if def_name == "Artifact":
        return run["payload"]["artifacts"][0]
    if def_name == "Metric":
        return run["payload"]["analysis_metrics"][0]
    if def_name == "Policy":
        return record_dict(bundle_dicts, "decision-demo-blocked")["payload"]["policy"]
    if def_name == "Change":
        return record_dict(bundle_dicts, "commit-demo-a")["payload"]["changes"][0]
    raise AssertionError(def_name)


NESTED_DEFS = ["Policy", "Environment", "Artifact", "Metric", "Change"]


@pytest.mark.parametrize("def_name", NESTED_DEFS)
def test_validate_nested_accepts_fixture_data(bundle_dicts: list[dict], def_name: str) -> None:
    sample = _nested_sample(bundle_dicts, def_name)
    assert validate_nested(def_name, sample) is None


def _drop(key: str) -> Callable[[dict], None]:
    def mutate(data: dict) -> None:
        del data[key]

    return mutate


def _set(key: str, value: Any) -> Callable[[dict], None]:
    def mutate(data: dict) -> None:
        data[key] = value

    return mutate


NESTED_REJECTIONS: list[tuple[str, str, Callable[[dict], None], str]] = [
    # (definition, scenario, mutation, token expected in the message)
    ("Policy", "missing key", _drop("policy_id"), "policy_id"),
    ("Policy", "wrong type", _set("min_confirm_pairs", "3"), "min_confirm_pairs"),
    ("Policy", "wrong type", _set("require_trusted_worker", "yes"), "require_trusted_worker"),
    ("Policy", "bad pattern", _set("min_pair_speedup", "fast"), "min_pair_speedup"),
    ("Policy", "below minimum", _set("min_confirm_pairs", 0), "min_confirm_pairs"),
    ("Policy", "unknown property", _set("extra", 1), "extra"),
    ("Environment", "missing key", _drop("environment_hash"), "environment_hash"),
    ("Environment", "wrong type", _set("device_count", "1"), "device_count"),
    ("Environment", "wrong type", _set("software", ["numpy"]), "software"),
    ("Environment", "bad pattern", _set("environment_hash", "deadbeef"), "environment_hash"),
    ("Environment", "unknown property", _set("gpu", "none"), "gpu"),
    ("Artifact", "missing key", _drop("sha256"), "sha256"),
    ("Artifact", "wrong enum", _set("retention", "forever"), "retention"),
    ("Artifact", "wrong enum", _set("availability", "gone"), "availability"),
    ("Artifact", "wrong type", _set("size_bytes", "10"), "size_bytes"),
    ("Artifact", "below minimum", _set("size_bytes", -1), "size_bytes"),
    ("Artifact", "unknown property", _set("path", "/tmp/x"), "path"),
    ("Metric", "missing key", _drop("definition"), "definition"),
    ("Metric", "wrong enum", _set("status", "ok"), "status"),
    ("Metric", "wrong enum", _set("kind", "guess"), "kind"),
    ("Metric", "wrong type", _set("value", "1.0"), "value"),
    ("Metric", "unknown property", _set("confidence", 0.9), "confidence"),
    ("Change", "missing key", _drop("change_id"), "change_id"),
    ("Change", "wrong enum", _set("extraction_source", "guessed"), "extraction_source"),
    ("Change", "wrong enum", _set("attribution", "solo"), "attribution"),
    ("Change", "wrong type", _set("rationale", 1), "rationale"),
    ("Change", "unknown property", _set("author", "x"), "author"),
]


@pytest.mark.parametrize(
    "def_name,scenario,mutate,token",
    NESTED_REJECTIONS,
    ids=[f"{d}-{s}-{t}" for d, s, _, t in NESTED_REJECTIONS],
)
def test_validate_nested_rejects_bad_data(
    bundle_dicts: list[dict], def_name: str, scenario: str, mutate: Callable[[dict], None], token: str
) -> None:
    sample = _nested_sample(bundle_dicts, def_name)
    mutate(sample)
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_nested(def_name, sample)
    message = str(excinfo.value)
    assert message.startswith(f"{def_name} is invalid"), message
    assert token in message, message
    assert excinfo.value.exit_code == 2


@pytest.mark.parametrize("def_name", NESTED_DEFS)
def test_validate_nested_rejects_non_object(def_name: str) -> None:
    for value in (None, [], "text", 1):
        with pytest.raises(SchemaValidationError) as excinfo:
            validate_nested(def_name, value)
        assert def_name in str(excinfo.value)


def test_validate_nested_unknown_definition_names_known() -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_nested("NoSuchThing", {})
    assert "NoSuchThing" in str(excinfo.value)
    assert set(NESTED_DEFS) <= set(excinfo.value.details["known"])


def test_validate_nested_other_definitions_are_available(bundle_dicts: list[dict]) -> None:
    run = record_dict(bundle_dicts, "run-demo-a")["payload"]
    for def_name in ("Protocol", "Verifier", "Timing", "Correctness", "Source"):
        key = def_name.lower()
        assert validate_nested(def_name, run[key]) is None
    with pytest.raises(SchemaValidationError):
        validate_nested("Protocol", {})


# --------------------------------------------------------------------------------------
# validate_against
# --------------------------------------------------------------------------------------
VALID_PROBLEM = {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}


def test_validate_against_accepts_valid_document() -> None:
    assert validate_against(demo_problem_schema(), VALID_PROBLEM, what="problem") is None
    assert validate_against({"type": "integer"}, 3) is None


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID_PROBLEM, "n": 0},
        {**VALID_PROBLEM, "dtype": "float64"},
        {**VALID_PROBLEM, "operation": "vector_sub"},
        {**VALID_PROBLEM, "outputs": ["z"]},
        {**VALID_PROBLEM, "extra": 1},
        {k: v for k, v in VALID_PROBLEM.items() if k != "outputs"},
        [],
        None,
    ],
)
def test_validate_against_message_includes_what(bad: Any) -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_against(demo_problem_schema(), bad, what="demo problem for cfg-demo")
    message = str(excinfo.value)
    assert message.startswith("demo problem for cfg-demo is invalid")
    assert excinfo.value.exit_code == 2


def test_validate_against_default_what_is_document() -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_against({"type": "integer"}, "three")
    assert str(excinfo.value).startswith("document is invalid")


def test_validate_against_fixture_config_problem(bundle_dicts: list[dict]) -> None:
    cfg = record_dict(bundle_dicts, "cfg-demo")
    assert cfg["payload"]["problem_schema_id"] == demo_problem_schema()["$id"]
    validate_against(demo_problem_schema(), cfg["payload"]["problem"], what="cfg-demo problem")


# --------------------------------------------------------------------------------------
# is_valid_record_dict
# --------------------------------------------------------------------------------------
def test_is_valid_record_dict_true_for_fixture(bundle_dicts: list[dict]) -> None:
    assert all(is_valid_record_dict(d) for d in bundle_dicts)


def test_is_valid_record_dict_false_cases(bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "kernel-demo")
    data["payload"]["bogus"] = 1
    assert is_valid_record_dict(data) is False
    naive = record_dict(bundle_dicts, "kernel-demo")
    naive["created_at"] = "2026-09-08T00:00:00"
    assert is_valid_record_dict(naive) is False
    assert is_valid_record_dict({"record_type": "widget"}) is False
    assert is_valid_record_dict(None) is False
    assert is_valid_record_dict("kernel") is False
    assert is_valid_record_dict({}) is False


# --------------------------------------------------------------------------------------
# Draft fixture
# --------------------------------------------------------------------------------------
def test_mla_config_draft_fixture_is_rejected(fixtures_root: Path) -> None:
    path = fixtures_root / "examples" / "mla_config.draft.json"
    draft = schema_mod.load_json_file(path)
    assert draft["status"] == "draft_not_registrable"
    with pytest.raises(SchemaValidationError):
        validate_record_dict(draft)
    assert is_valid_record_dict(draft) is False
    # Also not a valid demo problem: the draft is not registrable under any known contract.
    with pytest.raises(SchemaValidationError):
        validate_against(demo_problem_schema(), draft, what="mla draft")
