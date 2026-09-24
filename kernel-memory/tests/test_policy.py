"""Promotion policy: defaults, golden hash, strict validation, Decimal thresholds, file loading (spec 12.2)."""
from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

import kernel_memory
from kernel_memory.domain import hashing
from kernel_memory.domain.errors import InputError, SchemaValidationError, UnsafePathError
from kernel_memory.domain.jsonio import load_json_file
from kernel_memory.services import policy as policy_service
from kernel_memory.services.policy import (
    DEFAULT_POLICY_ID,
    POLICY_CODE,
    PolicyThresholds,
    ResourceConstraint,
    default_policy,
    load_policy,
    parse_threshold,
    policy_hash,
    thresholds,
    validate_constraint,
    validate_policy,
)

GOLDEN_POLICY_HASH = "sha256:4c214e453dadb52cf291a03a05079c5e9d0f4c501a67056c4f484b97cf3127a2"
HASH_VECTORS = Path(kernel_memory.__file__).resolve().parent / "contracts" / "hash_vectors.json"
REQUIRED_KEYS = (
    "policy_id",
    "min_confirm_pairs",
    "min_pair_speedup",
    "max_normalized_iqr",
    "max_baseline_drift_ratio",
    "hard_resource_constraints",
    "require_trusted_worker",
    "allow_fixture",
)
SPILL = "register_spill_vmem_static_bytes"


def golden_vector(name: str) -> dict:
    doc = load_json_file(HASH_VECTORS)
    return next(v for v in doc["vectors"] if v["name"] == name)


def assert_policy_error(exc_info: pytest.ExceptionInfo) -> None:
    exc = exc_info.value
    assert isinstance(exc, SchemaValidationError)
    assert isinstance(exc, InputError)
    assert exc.code == POLICY_CODE == "INVALID_POLICY"
    assert exc.exit_code == 2


# ------------------------------------------------------------------------------ defaults and golden hash
def test_default_policy_equals_promotion_policy_golden_payload() -> None:
    vector = golden_vector("promotion_policy")
    assert default_policy() == vector["payload"]
    assert default_policy()["policy_id"] == DEFAULT_POLICY_ID == "default-confirm-v1"
    assert vector["digest"] == GOLDEN_POLICY_HASH


def test_policy_hash_reproduces_golden_digest() -> None:
    assert policy_hash(default_policy()) == GOLDEN_POLICY_HASH
    # The service hashes only through domain.hashing (RFC 8785 canonical form).
    assert policy_hash(default_policy()) == hashing.policy_hash(default_policy())


def test_policy_hash_is_independent_of_key_order_and_formatting() -> None:
    reordered = {k: default_policy()[k] for k in reversed(REQUIRED_KEYS)}
    assert list(reordered) != list(default_policy())
    assert policy_hash(reordered) == GOLDEN_POLICY_HASH
    roundtrip = json.loads(json.dumps(default_policy(), indent=4))
    assert policy_hash(roundtrip) == GOLDEN_POLICY_HASH


def test_default_policy_returns_a_fresh_copy() -> None:
    first = default_policy()
    first["min_pair_speedup"] = "9.99"
    first["hard_resource_constraints"].append({"metric": SPILL, "max_value": 1})
    second = default_policy()
    assert second["min_pair_speedup"] == "1.02"
    assert second["hard_resource_constraints"] == []


def test_default_policy_matches_specification_12_2_defaults() -> None:
    th = thresholds(default_policy())
    assert isinstance(th, PolicyThresholds)
    assert th.min_confirm_pairs == 3
    assert th.min_pair_speedup == Decimal("1.02")
    assert th.max_normalized_iqr == Decimal("0.05")
    assert th.max_baseline_drift_ratio == Decimal("1.05")
    assert th.hard_resource_constraints == ()
    assert th.require_trusted_worker is True
    assert th.allow_fixture is False


# ------------------------------------------------------------------------------ T23: weakening changes the hash
@pytest.mark.parametrize(
    "field, value",
    [
        ("min_pair_speedup", "1.01"),
        ("max_normalized_iqr", "0.5"),
        ("max_baseline_drift_ratio", "1.5"),
        ("min_confirm_pairs", 1),
        ("require_trusted_worker", False),
        ("allow_fixture", True),
    ],
)
def test_t23_weakened_policy_has_a_different_hash(field: str, value: object) -> None:
    weakened = default_policy()
    weakened[field] = value
    assert validate_policy(weakened) == weakened
    assert policy_hash(weakened) != GOLDEN_POLICY_HASH


def test_t23_equal_decimal_written_differently_is_a_different_policy_hash() -> None:
    # "1.020" parses to the same Decimal but is a different wire policy: hashes never depend on float formatting.
    variant = default_policy()
    variant["min_pair_speedup"] = "1.020"
    assert thresholds(variant).min_pair_speedup == Decimal("1.02")
    assert policy_hash(variant) != GOLDEN_POLICY_HASH


# ------------------------------------------------------------------------------ validate_policy negatives
def test_validate_policy_accepts_default_and_returns_a_deep_copy() -> None:
    source = default_policy()
    source["hard_resource_constraints"].append({"metric": SPILL, "max_value": 32768})
    valid = validate_policy(source)
    assert valid == source
    valid["hard_resource_constraints"][0]["max_value"] = 1
    assert source["hard_resource_constraints"][0]["max_value"] == 32768


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_validate_policy_rejects_missing_key(key: str) -> None:
    broken = default_policy()
    del broken[key]
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


def test_validate_policy_rejects_unknown_key() -> None:
    broken = default_policy()
    broken["min_pair_speedpu"] = "1.02"  # misspelled property (T32)
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


@pytest.mark.parametrize("field", ["min_pair_speedup", "max_normalized_iqr", "max_baseline_drift_ratio"])
@pytest.mark.parametrize("value", [1.02, 1, True, None, ["1.02"], {"value": "1.02"}])
def test_validate_policy_rejects_non_string_thresholds(field: str, value: object) -> None:
    broken = default_policy()
    broken[field] = value
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


@pytest.mark.parametrize("field", ["min_pair_speedup", "max_normalized_iqr", "max_baseline_drift_ratio"])
@pytest.mark.parametrize("value", ["", "abc", "NaN", "Infinity", "-1.0", "1e3", "1.", ".5", " 1.02", "1,02"])
def test_validate_policy_rejects_malformed_threshold_strings(field: str, value: str) -> None:
    broken = default_policy()
    broken[field] = value
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


def test_validate_policy_rejects_zero_speedup_threshold() -> None:
    broken = default_policy()
    broken["min_pair_speedup"] = "0"
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


def test_validate_policy_rejects_baseline_drift_ratio_below_one() -> None:
    broken = default_policy()
    broken["max_baseline_drift_ratio"] = "0.99"
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)
    ok = default_policy()
    ok["max_baseline_drift_ratio"] = "1"
    assert thresholds(ok).max_baseline_drift_ratio == Decimal("1")


@pytest.mark.parametrize("value", [0, -1, "3", 3.5, True, None])
def test_validate_policy_rejects_invalid_min_confirm_pairs(value: object) -> None:
    broken = default_policy()
    broken["min_confirm_pairs"] = value
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


def test_validate_policy_accepts_integral_json_number_for_min_confirm_pairs() -> None:
    # JSON has one number type: 3.0 is the integer 3 (JSON Schema "integer" semantics), never a new policy meaning.
    policy = default_policy()
    policy["min_confirm_pairs"] = 3.0
    assert thresholds(policy).min_confirm_pairs == 3
    assert isinstance(thresholds(policy).min_confirm_pairs, int)


@pytest.mark.parametrize("field", ["require_trusted_worker", "allow_fixture"])
@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_validate_policy_rejects_non_boolean_flags(field: str, value: object) -> None:
    broken = default_policy()
    broken[field] = value
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


@pytest.mark.parametrize("value", ["", "bad id", "-leading", "x" * 300, 7, None])
def test_validate_policy_rejects_invalid_policy_id(value: object) -> None:
    broken = default_policy()
    broken["policy_id"] = value
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


@pytest.mark.parametrize("value", [None, [], "policy", 42, ("a", "b")])
def test_validate_policy_rejects_non_object(value: object) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(value)
    assert_policy_error(exc_info)


# ------------------------------------------------------------------------------ hard_resource_constraints
@pytest.mark.parametrize(
    "constraints",
    [
        "not-a-list",
        {"metric": SPILL, "max_value": 1},
        [None],
        ["register_spill_vmem_static_bytes"],
        [{}],
        [{"max_value": 32768}],  # metric missing
        [{"metric": "", "max_value": 32768}],
        [{"metric": "   ", "max_value": 32768}],
        [{"metric": 5, "max_value": 32768}],
        [{"metric": SPILL}],  # no bound at all
        [{"metric": SPILL, "max_value": None, "min_value": None}],
        [{"metric": SPILL, "max_value": "32768"}],  # bound as string
        [{"metric": SPILL, "max_value": True}],  # boolean is not a number (T32)
        [{"metric": SPILL, "max_value": math.nan}],
        [{"metric": SPILL, "min_value": math.inf}],
        [{"metric": SPILL, "min_value": 10, "max_value": 1}],  # min above max
        [{"metric": SPILL, "max_value": 1, "scope": ""}],
        [{"metric": SPILL, "max_value": 1, "kind": 3}],
        [{"metric": SPILL, "max_value": 1, "unit": "bytes"}],  # unknown key
        [{"metric": SPILL, "max_value": 1, "maxValue": 1}],  # misspelled key
        [{"metric": SPILL, "max_value": 1}, {"metric": SPILL}],  # second entry invalid
    ],
)
def test_validate_policy_rejects_malformed_hard_resource_constraints(constraints: object) -> None:
    broken = default_policy()
    broken["hard_resource_constraints"] = constraints
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_policy(broken)
    assert_policy_error(exc_info)


def test_validate_constraint_returns_typed_form_with_optional_scope_and_kind() -> None:
    item = {"metric": SPILL, "max_value": 32768, "scope": "compiled_kernel", "kind": "static_estimate"}
    constraint = validate_constraint(item, 0)
    assert constraint == ResourceConstraint(metric=SPILL, max_value=32768, min_value=None, scope="compiled_kernel", kind="static_estimate")
    assert constraint.to_dict() == {"metric": SPILL, "max_value": 32768, "min_value": None, "scope": "compiled_kernel", "kind": "static_estimate"}
    lower = validate_constraint({"metric": SPILL, "min_value": 0}, 1)
    assert lower.min_value == 0 and lower.max_value is None and lower.scope is None and lower.kind is None
    both = validate_constraint({"metric": SPILL, "min_value": 0, "max_value": 0}, 2)
    assert (both.min_value, both.max_value) == (0, 0)


def test_validate_constraint_error_names_the_index() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_constraint({"metric": SPILL}, 4)
    assert_policy_error(exc_info)
    assert "hard_resource_constraints[4]" in exc_info.value.message


def test_thresholds_exposes_constraints_as_tuple_of_resource_constraints() -> None:
    policy = default_policy()
    policy["hard_resource_constraints"] = [
        {"metric": SPILL, "max_value": 32768},
        {"metric": "hbm_bytes_moved", "min_value": 1, "scope": "kernel", "kind": "measurement"},
    ]
    th = thresholds(policy)
    assert isinstance(th.hard_resource_constraints, tuple)
    assert [c.metric for c in th.hard_resource_constraints] == [SPILL, "hbm_bytes_moved"]
    assert th.hard_resource_constraints[1].scope == "kernel" and th.hard_resource_constraints[1].kind == "measurement"
    assert policy_hash(policy) != GOLDEN_POLICY_HASH


# ------------------------------------------------------------------------------ Decimal parsing
def test_parse_threshold_returns_exact_decimal() -> None:
    value = parse_threshold("1.02", "min_pair_speedup")
    assert isinstance(value, Decimal)
    assert value == Decimal("1.02")
    assert value != Decimal(1.02)  # binary float 1.02 is not exactly 1.02
    assert str(value) == "1.02"
    assert parse_threshold("0", "max_normalized_iqr") == Decimal(0)


@pytest.mark.parametrize("value", [1.02, 1, None, True, b"1.02"])
def test_parse_threshold_rejects_non_strings(value: object) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_threshold(value, "min_pair_speedup")
    assert_policy_error(exc_info)
    assert "min_pair_speedup" in exc_info.value.message


@pytest.mark.parametrize("value", ["abc", "", "NaN", "Infinity", "-Infinity", "-0.5", "1.02.3"])
def test_parse_threshold_rejects_non_finite_negative_or_garbage(value: str) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_threshold(value, "max_normalized_iqr")
    assert_policy_error(exc_info)


def test_thresholds_uses_decimal_comparisons_that_floats_would_get_wrong() -> None:
    policy = default_policy()
    policy["min_pair_speedup"] = "1.1"
    th = thresholds(policy)
    # A recorded speedup of 1.1 (float) compared exactly: Decimal(repr(1.1)) == Decimal("1.1").
    assert Decimal(repr(1.1)) >= th.min_pair_speedup
    assert Decimal(repr(1.0999999)) < th.min_pair_speedup


def test_thresholds_validates_first() -> None:
    broken = default_policy()
    broken["min_pair_speedup"] = 1.02
    with pytest.raises(SchemaValidationError) as exc_info:
        thresholds(broken)
    assert_policy_error(exc_info)


# ------------------------------------------------------------------------------ load_policy
def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_policy_json_file_round_trips_default(tmp_path: Path) -> None:
    path = write(tmp_path / "policy.json", json.dumps(default_policy(), indent=2))
    loaded = load_policy(path)
    assert loaded == default_policy()
    assert policy_hash(loaded) == GOLDEN_POLICY_HASH


def test_load_policy_yaml_file_is_normalised_to_the_same_policy(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "policy_id: default-confirm-v1",
            "min_confirm_pairs: 3",
            'min_pair_speedup: "1.02"',
            'max_normalized_iqr: "0.05"',
            'max_baseline_drift_ratio: "1.05"',
            "hard_resource_constraints: []",
            "require_trusted_worker: true",
            "allow_fixture: false",
            "",
        ]
    )
    path = write(tmp_path / "policy.yaml", text)
    assert load_policy(path) == default_policy()
    assert policy_hash(load_policy(path)) == GOLDEN_POLICY_HASH


def test_load_policy_yaml_unquoted_threshold_is_a_number_and_rejected(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "policy_id: default-confirm-v1",
            "min_confirm_pairs: 3",
            "min_pair_speedup: 1.02",  # YAML float, not the required string
            'max_normalized_iqr: "0.05"',
            'max_baseline_drift_ratio: "1.05"',
            "hard_resource_constraints: []",
            "require_trusted_worker: true",
            "allow_fixture: false",
            "",
        ]
    )
    path = write(tmp_path / "policy.yml", text)
    with pytest.raises(SchemaValidationError) as exc_info:
        load_policy(path)
    assert_policy_error(exc_info)


def test_load_policy_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    body = json.dumps(default_policy())[:-1] + ', "allow_fixture": true}'
    path = write(tmp_path / "dup.json", body)
    with pytest.raises(InputError) as exc_info:
        load_policy(path)
    assert exc_info.value.code == "DUPLICATE_JSON_KEY"
    assert exc_info.value.exit_code == 2


def test_load_policy_rejects_nan_constant(tmp_path: Path) -> None:
    body = json.dumps(default_policy()).replace('"1.02"', "NaN")
    path = write(tmp_path / "nan.json", body)
    with pytest.raises(InputError) as exc_info:
        load_policy(path)
    assert exc_info.value.code == "NONFINITE_JSON"
    assert exc_info.value.exit_code == 2


def test_load_policy_rejects_invalid_json_text(tmp_path: Path) -> None:
    path = write(tmp_path / "broken.json", "{not json")
    with pytest.raises(InputError) as exc_info:
        load_policy(path)
    assert exc_info.value.exit_code == 2


def test_load_policy_rejects_schema_invalid_file(tmp_path: Path) -> None:
    broken = default_policy()
    del broken["allow_fixture"]
    path = write(tmp_path / "missing.json", json.dumps(broken))
    with pytest.raises(SchemaValidationError) as exc_info:
        load_policy(path)
    assert_policy_error(exc_info)


def test_load_policy_missing_file_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError) as exc_info:
        load_policy(tmp_path / "absent.json")
    assert exc_info.value.code == "FILE_NOT_FOUND"
    assert exc_info.value.exit_code == 2
    with pytest.raises(InputError) as yaml_info:
        load_policy(tmp_path / "absent.yaml")
    assert yaml_info.value.code == "FILE_NOT_FOUND"


def test_load_policy_with_root_resolves_relative_paths_inside_root(tmp_path: Path) -> None:
    (tmp_path / "policies").mkdir()
    write(tmp_path / "policies" / "p.json", json.dumps(default_policy()))
    assert load_policy("policies/p.json", root=tmp_path) == default_policy()


@pytest.mark.parametrize("relative", ["../outside.json", "/etc/passwd", "policies/../../outside.json", ""])
def test_t28_load_policy_with_root_refuses_traversal(tmp_path: Path, relative: str) -> None:
    write(tmp_path / "outside.json", json.dumps(default_policy()))
    with pytest.raises(UnsafePathError) as exc_info:
        load_policy(relative, root=tmp_path / "policies")
    assert exc_info.value.exit_code == 7


def test_module_public_api_is_pure_and_complete() -> None:
    for name in ("default_policy", "load_policy", "validate_policy", "policy_hash", "thresholds", "parse_threshold"):
        assert name in policy_service.__all__
