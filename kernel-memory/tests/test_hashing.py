"""Identity hashes (spec 5.2, 5.3, 12.1): every recorded fixture hash must be reproduced by domain.hashing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import record_dict

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import CanonicalizationError
from kernel_memory.domain.schema import golden_hash_vectors

FIXTURE_RUN_IDS = ["run-demo-baseline", "run-demo-a", "run-demo-c", "run-demo-c-failure"]


def _run_payload(bundle_dicts: list[dict], run_id: str) -> dict:
    return record_dict(bundle_dicts, run_id)["payload"]


def _config_payload(bundle_dicts: list[dict]) -> dict:
    return record_dict(bundle_dicts, "cfg-demo")["payload"]


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------
def test_sha256_bytes_prefix_and_known_value() -> None:
    assert hashing.sha256_bytes(b"") == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert hashing.sha256_bytes(b"abc") == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("sha256:" + "0" * 64, True),
        ("sha256:" + "a" * 64, True),
        ("sha256:" + "A" * 64, False),  # upper-case hex is not the canonical form
        ("sha256:" + "0" * 63, False),
        ("sha256:" + "0" * 65, False),
        ("sha1:" + "0" * 64, False),
        ("0" * 64, False),
        (None, False),
        (12, False),
    ],
)
def test_is_sha256_ref(value: object, expected: bool) -> None:
    assert hashing.is_sha256_ref(value) is expected


def test_jcs_digest_reproduces_all_golden_vectors() -> None:
    for vector in golden_hash_vectors()["vectors"]:
        assert hashing.jcs_digest(vector["payload"]) == vector["digest"], vector["name"]


def test_jcs_digest_rejects_nonfinite() -> None:
    with pytest.raises(CanonicalizationError):
        hashing.jcs_digest({"x": float("nan")})


# --------------------------------------------------------------------------------------
# config_hash (spec 5.2)
# --------------------------------------------------------------------------------------
def test_config_hash_reproduces_cfg_demo(bundle_dicts: list[dict]) -> None:
    cfg = _config_payload(bundle_dicts)
    assert hashing.config_hash(
        kernel_id=cfg["kernel_id"],
        problem_schema_id=cfg["problem_schema_id"],
        problem_schema_digest=cfg["problem_schema_digest"],
        problem=cfg["problem"],
    ) == cfg["config_hash"] == "sha256:f93d42a53c9f6bcdab1fa855d9b3e5727e65601ff355aa96dc16142a282c1915"


def test_config_hash_matches_golden_config_identity_vector(bundle_dicts: list[dict]) -> None:
    vector = next(v for v in golden_hash_vectors()["vectors"] if v["name"] == "config_identity")
    payload = vector["payload"]
    assert payload["hash_version"] == hashing.HASH_VERSION
    assert hashing.config_hash(
        kernel_id=payload["kernel_id"],
        problem_schema_id=payload["problem_schema_id"],
        problem_schema_digest=payload["problem_schema_digest"],
        problem=payload["problem"],
    ) == vector["digest"]


def test_config_hash_is_independent_of_problem_key_order(bundle_dicts: list[dict]) -> None:
    cfg = _config_payload(bundle_dicts)
    reordered = {k: cfg["problem"][k] for k in reversed(list(cfg["problem"]))}
    assert list(reordered) != list(cfg["problem"])
    kwargs = dict(kernel_id=cfg["kernel_id"], problem_schema_id=cfg["problem_schema_id"], problem_schema_digest=cfg["problem_schema_digest"])
    assert hashing.config_hash(problem=reordered, **kwargs) == cfg["config_hash"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"kernel_id": "other_kernel"},
        {"problem_schema_id": "urn:kernel-memory:problem:demo-vector-add:v2"},
        {"problem_schema_digest": "sha256:" + "1" * 64},
        {"problem": {"n": 32, "dtype": "float32", "operation": "vector_add", "outputs": ["y"]}},
        {"problem": {"n": 16, "dtype": "float32", "operation": "vector_add", "outputs": ["y", "lse"]}},
    ],
)
def test_config_hash_changes_with_any_identity_component(bundle_dicts: list[dict], mutation: dict) -> None:
    cfg = _config_payload(bundle_dicts)
    kwargs = dict(
        kernel_id=cfg["kernel_id"],
        problem_schema_id=cfg["problem_schema_id"],
        problem_schema_digest=cfg["problem_schema_digest"],
        problem=cfg["problem"],
    )
    kwargs.update(mutation)
    assert hashing.config_hash(**kwargs) != cfg["config_hash"]


def test_problem_schema_digest_reproduces_recorded_digest(bundle_dicts: list[dict]) -> None:
    from kernel_memory.domain.schema import demo_problem_schema

    cfg = _config_payload(bundle_dicts)
    assert hashing.problem_schema_digest(demo_problem_schema()) == cfg["problem_schema_digest"]


# --------------------------------------------------------------------------------------
# Per-run identities (spec 5.3, 12.1)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("run_id", FIXTURE_RUN_IDS)
def test_variant_digest_reproduced_for_every_fixture_run(bundle_dicts: list[dict], run_id: str) -> None:
    source = _run_payload(bundle_dicts, run_id)["source"]
    assert hashing.variant_digest(
        source_digest=source["source_digest"],
        entrypoint=source["entrypoint"],
        implementation_overrides=source["implementation_overrides"],
        checkout_mode=source["checkout_mode"],
    ) == source["variant_digest"]


@pytest.mark.parametrize("run_id", FIXTURE_RUN_IDS)
def test_environment_hash_reproduced_for_every_fixture_run(bundle_dicts: list[dict], run_id: str) -> None:
    env = _run_payload(bundle_dicts, run_id)["environment"]
    assert hashing.environment_hash(env) == env["environment_hash"]


@pytest.mark.parametrize("run_id", FIXTURE_RUN_IDS)
def test_protocol_hash_reproduced_for_every_fixture_run(bundle_dicts: list[dict], run_id: str) -> None:
    protocol = _run_payload(bundle_dicts, run_id)["protocol"]
    assert hashing.protocol_hash(protocol) == protocol["protocol_hash"]


@pytest.mark.parametrize("run_id", FIXTURE_RUN_IDS)
def test_verifier_hash_reproduced_for_every_fixture_run(bundle_dicts: list[dict], run_id: str) -> None:
    verifier = _run_payload(bundle_dicts, run_id)["verifier"]
    assert hashing.verifier_hash(verifier) == verifier["verifier_hash"]


@pytest.mark.parametrize("run_id", FIXTURE_RUN_IDS)
def test_comparison_key_reproduced_for_every_fixture_run(bundle_dicts: list[dict], run_id: str) -> None:
    run = _run_payload(bundle_dicts, run_id)
    cfg = _config_payload(bundle_dicts)
    assert run["config_ref"] == "cfg-demo"
    assert hashing.comparison_key(
        config_hash=cfg["config_hash"],
        environment_hash=run["environment"]["environment_hash"],
        protocol_hash=run["protocol"]["protocol_hash"],
        verifier_hash=run["verifier"]["verifier_hash"],
        checkout_mode=run["source"]["checkout_mode"],
    ) == run["comparison_key"]


def test_comparison_key_matches_golden_comparison_identity_vector() -> None:
    vector = next(v for v in golden_hash_vectors()["vectors"] if v["name"] == "comparison_identity")
    assert hashing.comparison_key(**vector["payload"]) == vector["digest"]


def test_snapshot_hash_ignores_its_own_hash_field_only(bundle_dicts: list[dict]) -> None:
    env = _run_payload(bundle_dicts, "run-demo-a")["environment"]
    without = {k: v for k, v in env.items() if k != "environment_hash"}
    assert hashing.environment_hash(without) == env["environment_hash"]
    with_wrong_recorded = dict(env, environment_hash="sha256:" + "f" * 64)
    assert hashing.environment_hash(with_wrong_recorded) == env["environment_hash"]
    # Any other field participates.
    assert hashing.environment_hash(dict(env, device_count=2)) != env["environment_hash"]
    assert hashing.environment_hash(dict(env, software={"runner": "fixture-v2"})) != env["environment_hash"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment_hash", "sha256:" + "1" * 64),
        ("protocol_hash", "sha256:" + "2" * 64),
        ("verifier_hash", "sha256:" + "3" * 64),
        ("config_hash", "sha256:" + "4" * 64),
        ("checkout_mode", "merge_result"),
    ],
)
def test_t18_comparison_key_changes_when_any_component_changes(bundle_dicts: list[dict], field: str, value: str) -> None:
    run = _run_payload(bundle_dicts, "run-demo-a")
    cfg = _config_payload(bundle_dicts)
    kwargs = dict(
        config_hash=cfg["config_hash"],
        environment_hash=run["environment"]["environment_hash"],
        protocol_hash=run["protocol"]["protocol_hash"],
        verifier_hash=run["verifier"]["verifier_hash"],
        checkout_mode=run["source"]["checkout_mode"],
    )
    kwargs[field] = value
    assert hashing.comparison_key(**kwargs) != run["comparison_key"]


def test_t23_verifier_tolerance_change_gives_new_verifier_hash(bundle_dicts: list[dict]) -> None:
    verifier = _run_payload(bundle_dicts, "run-demo-a")["verifier"]
    loosened = dict(verifier, tolerances={"atol": "0.001", "rtol": "0"})
    assert hashing.verifier_hash(loosened) != verifier["verifier_hash"]


# --------------------------------------------------------------------------------------
# T03: tiling-only change => same config, new variant
# --------------------------------------------------------------------------------------
def test_t03_variant_digest_changes_with_overrides_while_config_hash_untouched(bundle_dicts: list[dict]) -> None:
    cfg = _config_payload(bundle_dicts)
    source = _run_payload(bundle_dicts, "run-demo-a")["source"]
    config_before = hashing.config_hash(
        kernel_id=cfg["kernel_id"],
        problem_schema_id=cfg["problem_schema_id"],
        problem_schema_digest=cfg["problem_schema_digest"],
        problem=cfg["problem"],
    )
    base = dict(source_digest=source["source_digest"], entrypoint=source["entrypoint"], checkout_mode=source["checkout_mode"])
    v_plain = hashing.variant_digest(implementation_overrides={}, **base)
    v_tile_128 = hashing.variant_digest(implementation_overrides={"block_q": 128}, **base)
    v_tile_256 = hashing.variant_digest(implementation_overrides={"block_q": 256}, **base)
    assert v_plain == source["variant_digest"]
    assert len({v_plain, v_tile_128, v_tile_256}) == 3
    # The config identity does not see implementation overrides at all.
    config_after = hashing.config_hash(
        kernel_id=cfg["kernel_id"],
        problem_schema_id=cfg["problem_schema_id"],
        problem_schema_digest=cfg["problem_schema_digest"],
        problem=cfg["problem"],
    )
    assert config_after == config_before == cfg["config_hash"]


def test_variant_digest_override_key_order_is_irrelevant() -> None:
    base = dict(source_digest="sha256:" + "a" * 64, entrypoint="m:f", checkout_mode="exact_commit")
    a = hashing.variant_digest(implementation_overrides={"block_q": 128, "stages": 3}, **base)
    b = hashing.variant_digest(implementation_overrides={"stages": 3, "block_q": 128}, **base)
    assert a == b


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_digest": "sha256:" + "b" * 64},
        {"entrypoint": "m:g"},
        {"checkout_mode": "merge_result"},
    ],
)
def test_variant_digest_changes_with_source_entrypoint_or_checkout_mode(mutation: dict) -> None:
    kwargs = dict(source_digest="sha256:" + "a" * 64, entrypoint="m:f", checkout_mode="exact_commit", implementation_overrides={})
    before = hashing.variant_digest(**kwargs)
    kwargs.update(mutation)
    assert hashing.variant_digest(**kwargs) != before


# --------------------------------------------------------------------------------------
# policy_hash
# --------------------------------------------------------------------------------------
def test_policy_hash_reproduces_decision_demo_blocked(bundle_dicts: list[dict]) -> None:
    decision = record_dict(bundle_dicts, "decision-demo-blocked")["payload"]
    assert hashing.policy_hash(decision["policy"]) == decision["policy_hash"]
    vector = next(v for v in golden_hash_vectors()["vectors"] if v["name"] == "promotion_policy")
    assert decision["policy_hash"] == vector["digest"]


def test_policy_hash_changes_when_threshold_changes(bundle_dicts: list[dict]) -> None:
    policy = record_dict(bundle_dicts, "decision-demo-blocked")["payload"]["policy"]
    assert hashing.policy_hash(dict(policy, min_pair_speedup="1.01")) != hashing.policy_hash(policy)
    assert hashing.policy_hash(dict(policy, allow_fixture=True)) != hashing.policy_hash(policy)


# --------------------------------------------------------------------------------------
# artifact_digest: original bytes, never re-serialised
# --------------------------------------------------------------------------------------
def _fixture_artifacts(bundle_dicts: list[dict]) -> list[dict]:
    refs = []
    for run_id in FIXTURE_RUN_IDS:
        refs.extend(_run_payload(bundle_dicts, run_id)["artifacts"])
    return refs


def test_fixture_bundle_references_seven_artifact_files(bundle_dicts: list[dict]) -> None:
    refs = _fixture_artifacts(bundle_dicts)
    assert len(refs) == 7
    assert len({r["uri"] for r in refs}) == 7


def test_artifact_digest_and_size_reproduced_for_each_fixture_artifact(bundle_dicts: list[dict], artifact_root: Path) -> None:
    for ref in _fixture_artifacts(bundle_dicts):
        data = (artifact_root / ref["uri"]).read_bytes()
        assert hashing.artifact_digest(data) == ref["sha256"], ref["artifact_id"]
        assert len(data) == ref["size_bytes"], ref["artifact_id"]


def test_artifact_digest_covers_original_bytes_not_reserialised_json(bundle_dicts: list[dict], artifact_root: Path) -> None:
    ref = _run_payload(bundle_dicts, "run-demo-a")["artifacts"][0]
    data = (artifact_root / ref["uri"]).read_bytes()
    reserialised = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert reserialised != data
    assert hashing.artifact_digest(reserialised) != ref["sha256"]
    assert hashing.artifact_digest(data) == ref["sha256"]
    # Not a JCS digest of the parsed object either.
    assert hashing.jcs_digest(json.loads(data)) != ref["sha256"]


def test_artifact_digest_detects_single_byte_corruption(bundle_dicts: list[dict], artifact_root: Path) -> None:
    ref = _run_payload(bundle_dicts, "run-demo-a")["artifacts"][0]
    data = bytearray((artifact_root / ref["uri"]).read_bytes())
    data[-1] ^= 0x01
    assert hashing.artifact_digest(bytes(data)) != ref["sha256"]


# --------------------------------------------------------------------------------------
# source_digest
# --------------------------------------------------------------------------------------
def test_source_digest_is_order_independent() -> None:
    manifest = [
        ("src/a.py", hashing.sha256_bytes(b"a")),
        ("src/b.py", hashing.sha256_bytes(b"b")),
        ("README.md", hashing.sha256_bytes(b"readme")),
    ]
    forward = hashing.source_digest(manifest)
    assert hashing.source_digest(list(reversed(manifest))) == forward
    assert hashing.source_digest(iter([manifest[1], manifest[2], manifest[0]])) == forward
    assert hashing.source_digest(tuple(manifest)) == forward
    assert hashing.is_sha256_ref(forward)


def test_source_digest_changes_with_content_path_or_file_set() -> None:
    manifest = [("src/a.py", hashing.sha256_bytes(b"a")), ("src/b.py", hashing.sha256_bytes(b"b"))]
    base = hashing.source_digest(manifest)
    assert hashing.source_digest([("src/a.py", hashing.sha256_bytes(b"A")), manifest[1]]) != base
    assert hashing.source_digest([("src/c.py", hashing.sha256_bytes(b"a")), manifest[1]]) != base
    assert hashing.source_digest(manifest[:1]) != base
    assert hashing.source_digest([]) != base
