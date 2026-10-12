"""Problem schemas and normalizers (spec 5.1, 5.2): T01, T02, T32 and the mla_forward refusal path."""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from conftest import record_dict

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import (
    IncompleteProblemContract,
    InputError,
    PrerequisiteMissingError,
    SchemaValidationError,
)
from kernel_memory.domain.problems import (
    DemoVectorAddProblem,
    MlaDecodeProblem,
    MlaForwardProblem,
    NormalizedProblem,
    ProblemRegistry,
    default_registry,
)
from kernel_memory.domain.schema import demo_problem_schema, validate_against

DEMO_CONFIG_HASH = "sha256:f93d42a53c9f6bcdab1fa855d9b3e5727e65601ff355aa96dc16142a282c1915"
DEMO_SCHEMA_DIGEST = "sha256:cdd22559a450d844802382bf42c4d5a56545918f4e961c4fe87446ccc9687928"


# --------------------------------------------------------------------------------------
# Test-local attention problem adapter (T02): O-only versus O+LSE are distinct output contracts
# --------------------------------------------------------------------------------------
TEST_ATTENTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:kernel-memory:problem:test-attention:v1",
    "title": "Test-only attention problem (not an MLA contract)",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "seq_len": {"type": "integer", "minimum": 1},
        "heads": {"type": "integer", "minimum": 1},
        "dtype": {"enum": ["bfloat16", "float32"]},
        "causal": {"type": "boolean"},
        "outputs": {"enum": [["o"], ["o", "lse"]]},
    },
    "required": ["seq_len", "heads", "dtype", "causal", "outputs"],
}


class TestAttentionProblem:
    """In-test problem adapter: O-only and O+LSE normalise to different problems."""

    __test__ = False  # not a pytest test class
    kernel_id = "test_attention"
    schema_id = "urn:kernel-memory:problem:test-attention:v1"
    DTYPE_ALIASES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "f32": "float32", "float32": "float32", "fp32": "float32"}
    ALLOWED_KEYS = {"seq_len", "heads", "dtype", "causal", "outputs"}

    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(TEST_ATTENTION_SCHEMA)

    def schema_digest(self) -> str:
        return hashing.problem_schema_digest(TEST_ATTENTION_SCHEMA)

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise InputError("problem must be a JSON object", code="INVALID_PROBLEM")
        unknown = set(raw) - self.ALLOWED_KEYS
        if unknown:
            raise InputError(f"unknown problem fields: {sorted(unknown)}", code="INVALID_PROBLEM")
        for key in ("seq_len", "heads"):
            if key not in raw:
                raise InputError(f"problem field {key!r} is required", code="INVALID_PROBLEM")
            if isinstance(raw[key], bool) or not isinstance(raw[key], int):
                raise InputError(f"{key} must be an integer", code="INVALID_PROBLEM")
        if "outputs" not in raw:
            # No semantically defined default: the output contract must be stated.
            raise InputError("problem field 'outputs' is required (O-only vs O+LSE is a contract)", code="INVALID_PROBLEM")
        dtype_raw = raw.get("dtype", "bfloat16")
        if not isinstance(dtype_raw, str) or dtype_raw.lower() not in self.DTYPE_ALIASES:
            raise InputError(f"unsupported dtype {dtype_raw!r}", code="INVALID_PROBLEM")
        problem = {
            "seq_len": raw["seq_len"],
            "heads": raw["heads"],
            "dtype": self.DTYPE_ALIASES[dtype_raw.lower()],
            "causal": raw.get("causal", True),
            "outputs": list(raw["outputs"]) if isinstance(raw["outputs"], list) else raw["outputs"],
        }
        self.validate(problem)
        return problem

    def validate(self, problem: dict[str, Any]) -> None:
        validate_against(TEST_ATTENTION_SCHEMA, problem, what="test attention problem")

    def config_id_hint(self, problem: dict[str, Any]) -> str:
        return f"attn-s{problem['seq_len']}-h{problem['heads']}-{'+'.join(problem['outputs'])}"


@pytest.fixture
def attention_registry() -> ProblemRegistry:
    registry = ProblemRegistry()
    registry.register(TestAttentionProblem())
    return registry


# --------------------------------------------------------------------------------------
# Demo normalizer: aliases, defaults, fixture reproduction
# --------------------------------------------------------------------------------------
def test_demo_adapter_schema_digest_matches_fixture() -> None:
    adapter = DemoVectorAddProblem()
    assert adapter.schema_digest() == DEMO_SCHEMA_DIGEST
    assert adapter.schema() == demo_problem_schema()
    assert adapter.schema() is not demo_problem_schema()  # a copy, never the cached object


@pytest.mark.parametrize(
    "raw",
    [
        {"n": 16},
        {"n": 16, "dtype": "f32"},
        {"n": 16, "dtype": "F32"},
        {"n": 16, "dtype": "fp32"},
        {"n": 16, "dtype": "float32"},
        {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]},
        {"outputs": ["y"], "operation": "vector_add", "n": 16},
    ],
)
def test_f32_alias_and_defaults_reproduce_fixture_hash(raw: dict) -> None:
    registry = default_registry()
    normalized = registry.normalize("demo_vector_add", raw)
    assert isinstance(normalized, NormalizedProblem)
    assert normalized.problem == {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}
    assert normalized.config_hash == DEMO_CONFIG_HASH
    assert normalized.config_id_hint == "demo-n16-f32"
    assert normalized.problem_schema_digest == DEMO_SCHEMA_DIGEST
    assert normalized.problem_schema_id == "urn:kernel-memory:problem:demo-vector-add:v1"


def test_normalized_hash_matches_fixture_config_record(bundle_dicts: list[dict]) -> None:
    cfg = record_dict(bundle_dicts, "cfg-demo")["payload"]
    normalized = default_registry().normalize(cfg["kernel_id"], cfg["problem"])
    assert normalized.config_hash == cfg["config_hash"]
    assert normalized.config_id_hint == cfg["config_id"]
    assert normalized.problem == cfg["problem"]


def test_t01_normalizing_equivalent_inputs_twice_yields_one_identical_hash() -> None:
    registry = default_registry()
    first = registry.normalize("demo_vector_add", {"n": 16, "dtype": "f32"})
    second = registry.normalize("demo_vector_add", {"dtype": "FLOAT32", "outputs": ["y"], "n": 16, "operation": "vector_add"})
    assert first == second
    assert first.config_hash == second.config_hash == DEMO_CONFIG_HASH
    assert len({first.config_hash, second.config_hash}) == 1
    # Normalisation is idempotent: normalising the normalized form is a fixed point.
    third = registry.normalize("demo_vector_add", dict(first.problem))
    assert third == first


def test_different_n_is_a_different_config() -> None:
    registry = default_registry()
    a = registry.normalize("demo_vector_add", {"n": 16})
    b = registry.normalize("demo_vector_add", {"n": 32})
    assert a.config_hash != b.config_hash
    assert b.config_id_hint == "demo-n32-f32"


def test_normalize_returns_fresh_problem_dict_each_time() -> None:
    registry = default_registry()
    a = registry.normalize("demo_vector_add", {"n": 16})
    b = registry.normalize("demo_vector_add", {"n": 16})
    assert a.problem == b.problem and a.problem is not b.problem


# --------------------------------------------------------------------------------------
# Rejections (T32)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ({"n": True}, "integer"),
        ({"n": False}, "integer"),
        ({"n": 0}, ">= 1"),
        ({"n": -4}, ">= 1"),
        ({"n": 16.0}, "integer"),
        ({"n": "16"}, "integer"),
        ({"n": 16, "dtype": "bf16"}, "dtype"),
        ({"n": 16, "dtype": 32}, "dtype"),
        ({"n": 16, "extra": 1}, "extra"),
        ({"n": 16, "tile": 128}, "tile"),
        ({}, "'n' is required"),
        ({"dtype": "f32"}, "'n' is required"),
    ],
)
def test_t32_demo_normalizer_rejects_invalid_problem(raw: dict, fragment: str) -> None:
    registry = default_registry()
    with pytest.raises(InputError) as info:
        registry.normalize("demo_vector_add", raw)
    assert info.value.exit_code == 2
    assert info.value.code == "INVALID_PROBLEM"
    assert fragment in str(info.value)


def test_demo_normalizer_rejects_non_object() -> None:
    registry = default_registry()
    for bad in (None, [], "n=16", 16):
        with pytest.raises(InputError):
            registry.normalize("demo_vector_add", bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        {"n": 16, "operation": "vector_mul"},
        {"n": 16, "outputs": ["y", "lse"]},
        {"n": 16, "outputs": "y"},
        {"n": 16, "outputs": []},
    ],
)
def test_demo_normalizer_rejects_operation_and_output_contract_changes(raw: dict) -> None:
    """The demo operator has exactly one operation and one output set; nothing else is guessed."""
    with pytest.raises(SchemaValidationError) as info:
        default_registry().normalize("demo_vector_add", raw)
    assert info.value.exit_code == 2


def test_demo_validate_rejects_boolean_dimension_directly() -> None:
    adapter = DemoVectorAddProblem()
    with pytest.raises(SchemaValidationError):
        adapter.validate({"n": True, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]})
    adapter.validate({"n": 1, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]})


# --------------------------------------------------------------------------------------
# T02: O-only versus O+LSE are different Configs (no shared ranking)
# --------------------------------------------------------------------------------------
def test_t02_o_only_and_o_plus_lse_produce_different_config_hashes(attention_registry: ProblemRegistry) -> None:
    base = {"seq_len": 4096, "heads": 16, "dtype": "bf16", "causal": True}
    o_only = attention_registry.normalize("test_attention", dict(base, outputs=["o"]))
    o_lse = attention_registry.normalize("test_attention", dict(base, outputs=["o", "lse"]))
    assert o_only.problem["outputs"] == ["o"]
    assert o_lse.problem["outputs"] == ["o", "lse"]
    assert o_only.problem_schema_digest == o_lse.problem_schema_digest  # same schema ...
    assert o_only.config_hash != o_lse.config_hash  # ... different Config
    assert o_only.config_id_hint != o_lse.config_id_hint
    assert o_only.config_id_hint == "attn-s4096-h16-o"
    assert o_lse.config_id_hint == "attn-s4096-h16-o+lse"
    # Identity comes from the full hash; the two configs would never share a comparison key.
    cfg_group_a = hashing.comparison_key(
        config_hash=o_only.config_hash, environment_hash="sha256:" + "0" * 64,
        protocol_hash="sha256:" + "0" * 64, verifier_hash="sha256:" + "0" * 64, checkout_mode="exact_commit",
    )
    cfg_group_b = hashing.comparison_key(
        config_hash=o_lse.config_hash, environment_hash="sha256:" + "0" * 64,
        protocol_hash="sha256:" + "0" * 64, verifier_hash="sha256:" + "0" * 64, checkout_mode="exact_commit",
    )
    assert cfg_group_a != cfg_group_b


def test_t02_equivalent_attention_inputs_normalize_to_one_hash(attention_registry: ProblemRegistry) -> None:
    a = attention_registry.normalize("test_attention", {"seq_len": 4096, "heads": 16, "dtype": "bf16", "outputs": ["o"]})
    b = attention_registry.normalize("test_attention", {"outputs": ["o"], "causal": True, "dtype": "bfloat16", "heads": 16, "seq_len": 4096})
    assert a == b


def test_t02_output_contract_must_be_explicit(attention_registry: ProblemRegistry) -> None:
    with pytest.raises(InputError) as info:
        attention_registry.normalize("test_attention", {"seq_len": 4096, "heads": 16})
    assert "outputs" in str(info.value)


def test_t02_unknown_output_set_is_rejected(attention_registry: ProblemRegistry) -> None:
    with pytest.raises(SchemaValidationError):
        attention_registry.normalize("test_attention", {"seq_len": 4096, "heads": 16, "outputs": ["lse"]})
    with pytest.raises(SchemaValidationError):
        attention_registry.normalize("test_attention", {"seq_len": 4096, "heads": 16, "outputs": ["lse", "o"]})


def test_attention_config_hash_is_computed_by_domain_hashing(attention_registry: ProblemRegistry) -> None:
    normalized = attention_registry.normalize("test_attention", {"seq_len": 8, "heads": 1, "outputs": ["o"]})
    assert normalized.config_hash == hashing.config_hash(
        kernel_id="test_attention",
        problem_schema_id=TestAttentionProblem.schema_id,
        problem_schema_digest=hashing.problem_schema_digest(TEST_ATTENTION_SCHEMA),
        problem=normalized.problem,
    )


# --------------------------------------------------------------------------------------
# mla_forward: explicit refusal (exit 5) listing unresolved items
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        {"batch": 1, "heads": 16, "seq_len": 4096, "q_rank": 1536, "kv_rank": 512, "d_nope": 128, "d_rope": 64},
        {},
    ],
)
def test_mla_forward_normalize_raises_incomplete_contract_with_seven_unresolved_items(raw: dict) -> None:
    registry = default_registry()
    with pytest.raises(IncompleteProblemContract) as info:
        registry.normalize("mla_forward", raw)
    err = info.value
    assert err.exit_code == 5
    assert err.code == "INCOMPLETE_PROBLEM_CONTRACT"
    assert isinstance(err, PrerequisiteMissingError)
    assert err.details["kernel_id"] == "mla_forward"
    assert len(err.details["unresolved"]) == 7
    assert err.details["unresolved"] == list(MlaForwardProblem.UNRESOLVED)
    assert "O-only versus O+LSE contract" in err.details["unresolved"]
    assert err.to_dict()["exit_code"] == 5


def test_mla_forward_unresolved_items_match_draft_fixture(fixtures_root) -> None:
    from kernel_memory.domain.jsonio import load_json_file

    draft = load_json_file(fixtures_root / "examples" / "mla_config.draft.json")
    assert draft["kernel_id"] == "mla_forward"
    assert draft["unresolved"] == list(MlaForwardProblem.UNRESOLVED)


def test_mla_forward_refuses_every_operation() -> None:
    adapter = MlaForwardProblem()
    assert adapter.status == "incomplete"
    for call in (adapter.schema, adapter.schema_digest):
        with pytest.raises(IncompleteProblemContract):
            call()
    with pytest.raises(IncompleteProblemContract):
        adapter.normalize({"heads": 16})
    with pytest.raises(IncompleteProblemContract):
        adapter.validate({"heads": 16})
    with pytest.raises(IncompleteProblemContract):
        adapter.config_id_hint({"heads": 16})


def test_mla_forward_is_listed_but_never_produces_a_hash() -> None:
    registry = default_registry()
    assert registry.kernel_ids() == ["demo_vector_add", "mla_decode", "mla_forward"]
    assert isinstance(registry.get("mla_forward"), MlaForwardProblem)


def test_mla_decode_problem_normalizes_the_paged_project2_abi() -> None:
    adapter = MlaDecodeProblem()
    problem = {
        "operation": "mla_decode",
        "interface": "paged",
        "batch": 8,
        "num_q_heads": 16,
        "query_tokens_per_request": 1,
        "latent_dim": 512,
        "rope_dim": 64,
        "q_dim": 576,
        "cache_feature_dim": 640,
        "capacity": 16384,
        "timed_length": 16384,
        "dtype": "bfloat16",
        "score_dtype": "float32",
        "probability_dtype": "bfloat16",
        "block_size": 2048,
        "sm_scale": 0.07216878364870323,
        "sliding_window": 0,
        "soft_cap": 0.0,
        "cache_layout": "paged",
        "physical_cache_shape": [1024, 64, 2, 640],
        "length_semantics": "append_then_attend",
        "input_signature": {
            "q": {"dtype": "bfloat16", "shape": [8, 16, 576], "layout": "BHQ"},
            "new_kv": {"dtype": "bfloat16", "shape": [8, 576], "layout": "BQ"},
            "cache": {"dtype": "bfloat16", "shape": [1024, 64, 2, 640], "layout": "P64x2F"},
            "lengths": {"dtype": "int32", "shape": [8], "layout": "B"},
        },
        "output_signature": {
            "attention": {"dtype": "bfloat16", "shape": [8, 16, 512], "layout": "BHL"},
            "updated_cache": {"dtype": "bfloat16", "shape": [1024, 64, 2, 640], "layout": "P64x2F"},
        },
        "input_generation": {"distribution": "standard_normal", "seed": 0, "quantization": "exact_bfloat16"},
        "page_size": 128,
        "pages_per_block": 16,
        "block_kv": 2048,
        "decode_batch_size": 2,
        "transpose_cache": False,
        "two_step": True,
    }
    normalized = adapter.normalize(problem)
    assert normalized["interface"] == "paged_native"
    assert normalized["q_dim"] == normalized["latent_dim"] + normalized["rope_dim"]
    assert normalized["physical_cache_shape"] == [1024, 64, 2, 640]


def test_mla_decode_problem_rejects_a_wrong_cache_feature_shape() -> None:
    adapter = MlaDecodeProblem()
    with pytest.raises(InputError, match="physical_cache_shape"):
        # Keep the test input small while checking the fixed physical-shape rule.
        adapter.normalize({
            "operation": "mla_decode", "interface": "paged", "batch": 1, "num_q_heads": 1,
            "query_tokens_per_request": 1, "latent_dim": 2, "rope_dim": 2, "q_dim": 4,
            "cache_feature_dim": 8, "capacity": 128, "timed_length": 128, "dtype": "bfloat16",
            "score_dtype": "float32", "probability_dtype": "bfloat16", "block_size": 128,
            "sm_scale": 1.0, "sliding_window": 0, "soft_cap": 0.0, "cache_layout": "paged",
            "physical_cache_shape": [1, 64, 2, 7], "length_semantics": "append_then_attend",
            "input_signature": {"q": {"dtype": "bfloat16", "shape": [1, 1, 4], "layout": "BHQ"}, "new_kv": {"dtype": "bfloat16", "shape": [1, 4], "layout": "BQ"}, "cache": {"dtype": "bfloat16", "shape": [1, 64, 2, 7], "layout": "P64x2F"}, "lengths": {"dtype": "int32", "shape": [1], "layout": "B"}},
            "output_signature": {"attention": {"dtype": "bfloat16", "shape": [1, 1, 2], "layout": "BHL"}, "updated_cache": {"dtype": "bfloat16", "shape": [1, 64, 2, 7], "layout": "P64x2F"}},
            "input_generation": {"distribution": "standard_normal", "seed": 0, "quantization": "exact_bfloat16"},
            "page_size": 128, "pages_per_block": 1, "block_kv": 128, "decode_batch_size": 1,
            "transpose_cache": False, "two_step": False,
        })


def test_unregistered_kernel_is_incomplete_contract() -> None:
    registry = ProblemRegistry()
    with pytest.raises(IncompleteProblemContract) as info:
        registry.normalize("nonexistent_kernel", {"n": 1})
    assert info.value.exit_code == 5
    assert info.value.details["kernel_id"] == "nonexistent_kernel"


# --------------------------------------------------------------------------------------
# Registry: registration conflicts
# --------------------------------------------------------------------------------------
def test_duplicate_registration_of_different_adapter_is_adapter_conflict() -> None:
    registry = ProblemRegistry()
    registry.register(DemoVectorAddProblem())
    with pytest.raises(InputError) as info:
        registry.register(DemoVectorAddProblem())  # a different instance for the same kernel
    assert info.value.code == "ADAPTER_CONFLICT"
    assert info.value.exit_code == 2


def test_registering_the_same_adapter_instance_twice_is_idempotent() -> None:
    registry = ProblemRegistry()
    adapter = DemoVectorAddProblem()
    registry.register(adapter)
    registry.register(adapter)
    assert registry.get("demo_vector_add") is adapter
    assert registry.kernel_ids() == ["demo_vector_add"]


def test_adapter_without_kernel_id_is_rejected() -> None:
    class Nameless:
        pass

    class EmptyName:
        kernel_id = ""

    registry = ProblemRegistry()
    with pytest.raises(InputError):
        registry.register(Nameless())
    with pytest.raises(InputError):
        registry.register(EmptyName())


def test_fresh_registry_is_isolated_from_default() -> None:
    fresh = ProblemRegistry()
    assert fresh.kernel_ids() == []
    fresh.register(TestAttentionProblem())
    assert fresh.kernel_ids() == ["test_attention"]
    assert "test_attention" not in default_registry().kernel_ids()


# --------------------------------------------------------------------------------------
# verify_config_payload
# --------------------------------------------------------------------------------------
def test_verify_config_payload_accepts_fixture_config(bundle_dicts: list[dict]) -> None:
    default_registry().verify_config_payload(record_dict(bundle_dicts, "cfg-demo")["payload"])


def test_verify_config_payload_detects_wrong_schema_digest(bundle_dicts: list[dict]) -> None:
    payload = record_dict(bundle_dicts, "cfg-demo")["payload"]
    payload["problem_schema_digest"] = "sha256:" + "1" * 64
    with pytest.raises(SchemaValidationError) as info:
        default_registry().verify_config_payload(payload)
    assert "problem_schema_digest" in str(info.value)


def test_verify_config_payload_detects_wrong_schema_id(bundle_dicts: list[dict]) -> None:
    payload = record_dict(bundle_dicts, "cfg-demo")["payload"]
    payload["problem_schema_id"] = "urn:kernel-memory:problem:demo-vector-add:v9"
    with pytest.raises(SchemaValidationError) as info:
        default_registry().verify_config_payload(payload)
    assert "problem_schema_id" in str(info.value)


def test_verify_config_payload_detects_wrong_config_hash(bundle_dicts: list[dict]) -> None:
    payload = record_dict(bundle_dicts, "cfg-demo")["payload"]
    payload["config_hash"] = "sha256:" + "2" * 64
    with pytest.raises(SchemaValidationError) as info:
        default_registry().verify_config_payload(payload)
    assert info.value.details["recorded"] == payload["config_hash"]
    assert info.value.details["computed"] == DEMO_CONFIG_HASH


def test_verify_config_payload_detects_non_normalized_problem(bundle_dicts: list[dict]) -> None:
    payload = record_dict(bundle_dicts, "cfg-demo")["payload"]
    # Missing defaults / alias not canonicalised: the payload is not in normalized form.
    payload["problem"] = {"n": 16}
    payload["config_hash"] = hashing.config_hash(
        kernel_id=payload["kernel_id"],
        problem_schema_id=payload["problem_schema_id"],
        problem_schema_digest=payload["problem_schema_digest"],
        problem=payload["problem"],
    )
    with pytest.raises(SchemaValidationError):
        default_registry().verify_config_payload(payload)


def test_verify_config_payload_detects_boolean_dimension(bundle_dicts: list[dict]) -> None:
    payload = record_dict(bundle_dicts, "cfg-demo")["payload"]
    payload["problem"]["n"] = True
    payload["config_hash"] = hashing.config_hash(
        kernel_id=payload["kernel_id"],
        problem_schema_id=payload["problem_schema_id"],
        problem_schema_digest=payload["problem_schema_digest"],
        problem=payload["problem"],
    )
    with pytest.raises(SchemaValidationError):
        default_registry().verify_config_payload(payload)


def test_verify_config_payload_unknown_kernel_checks_hash_self_consistency_only() -> None:
    problem = {"shape": [4, 4]}
    payload = {
        "kernel_id": "unknown_kernel",
        "config_id": "u1",
        "problem_schema_id": "urn:test:unknown:v1",
        "problem_schema_digest": "sha256:" + "a" * 64,
        "problem": problem,
        "tags": [],
    }
    payload["config_hash"] = hashing.config_hash(
        kernel_id="unknown_kernel", problem_schema_id="urn:test:unknown:v1",
        problem_schema_digest="sha256:" + "a" * 64, problem=problem,
    )
    ProblemRegistry().verify_config_payload(payload)
    payload["config_hash"] = "sha256:" + "b" * 64
    with pytest.raises(SchemaValidationError):
        ProblemRegistry().verify_config_payload(payload)


def test_verify_config_payload_for_incomplete_adapter_checks_hash_only() -> None:
    """An mla_forward config can never be normalised; only its recorded hash self-consistency is checked."""
    problem = {"heads": 16}
    payload = {
        "kernel_id": "mla_forward",
        "config_id": "mla-x",
        "problem_schema_id": "urn:kernel-memory:problem:mla-forward:unresolved",
        "problem_schema_digest": "sha256:" + "c" * 64,
        "problem": problem,
        "tags": [],
        "config_hash": hashing.config_hash(
            kernel_id="mla_forward", problem_schema_id="urn:kernel-memory:problem:mla-forward:unresolved",
            problem_schema_digest="sha256:" + "c" * 64, problem=problem,
        ),
    }
    default_registry().verify_config_payload(payload)
    payload["config_hash"] = "sha256:" + "d" * 64
    with pytest.raises(SchemaValidationError):
        default_registry().verify_config_payload(payload)


def test_normalized_problem_is_json_compatible_and_hash_stable() -> None:
    normalized = default_registry().normalize("demo_vector_add", {"n": 16})
    round_tripped = json.loads(json.dumps(normalized.problem))
    assert hashing.jcs_digest(round_tripped) == hashing.jcs_digest(normalized.problem)
