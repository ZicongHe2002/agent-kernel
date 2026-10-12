"""Compact Memory export keeps fixed slots and never mixes shape evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from conftest import record_dict
from kernel_memory.domain.errors import InputError
from kernel_memory.domain.hashing import config_hash, jcs_digest, sha256_bytes
from kernel_memory.domain.models import Record
from kernel_memory.services.output_memory import build_output_memory, export_output_memory, verify_output_memory
from kernel_memory.storage import MemoryStore


RESULT_FIELDS = {
    "candidate", "run_ref", "subject_ref", "source_config_ref", "execution_status",
    "correctness", "latency_us", "p90_latency_us", "samples", "mfu",
    "mfu_executed_estimate", "mxu", "attention_mxu_static", "roofline_efficiency",
    "logical_bandwidth_gbps", "logical_bandwidth_utilization", "hbm_bandwidth_gbps",
    "hbm_bandwidth_utilization", "hlo", "llo", "provenance", "failure_reason",
    "comparison_key",
}


def documents_of(docs: dict, kind: str) -> list[dict]:
    return [doc for doc in docs.values() if doc["document_type"] == kind]


def test_fixed_tree_repeatable_and_no_storage_internals(demo_store: MemoryStore, tmp_path: Path) -> None:
    output = tmp_path / "readable"
    before = demo_store.root.joinpath("journal/records.jsonl").read_bytes()
    first = export_output_memory(demo_store, output)
    second = export_output_memory(demo_store, output)
    assert first == second
    assert first["schema_version"] == "2.0.0"
    assert first["counts"] == {"kernels": 1, "algorithms": 1, "shapes": 1, "runs": 4}
    assert set(p.name for p in output.rglob("*" ) if p.is_file()) == {
        "manifest.json", "kernel.json", "baseline.json", "method_summary.json", "config.json", "attempt.json", "result.json", "code.json"
    }
    assert not {"artifacts", "journal", ".cache", "kernels", "requests"} & {p.name for p in output.rglob("*")}
    assert demo_store.root.joinpath("journal/records.jsonl").read_bytes() == before
    for relative, digest in first["files"].items():
        assert sha256_bytes((output / relative).read_bytes()) == digest
    docs = build_output_memory(demo_store)
    assert all(d["schema_version"] == "2.0.0" for d in docs.values())
    kernel = documents_of(docs, "kernel")[0]
    assert kernel["contract_notes"] == demo_store.require("kernel-demo").payload.contract_notes
    config = documents_of(docs, "config")[0]
    assert config["shape_hash"] == jcs_digest({"output_shape_version": "1.0.0",
        "kernel_id": config["kernel_id"], "problem": config["problem"]})
    assert verify_output_memory(output)["ok"] is True


def test_missing_metrics_and_ir_are_explicit_not_zero(demo_store: MemoryStore) -> None:
    docs = build_output_memory(demo_store)
    result = documents_of(docs, "result")[0]
    run = next(r for r in result["runs"] if r["run_ref"] == "run-demo-a")
    assert set(run) == RESULT_FIELDS
    assert all(not isinstance(value, (dict, list)) for value in run.values())
    for key in ("mfu", "mxu", "roofline_efficiency", "hbm_bandwidth_gbps"):
        assert run[key] is None
    assert run["llo"] == "not_collected"
    assert run["provenance"] == "fixture"
    assert run["correctness"] == "pass"
    failed = next(r for r in result["runs"] if r["run_ref"] == "run-demo-c-failure")
    assert failed["latency_us"] is None
    attempts = documents_of(docs, "attempt")[0]["attempts"]
    assert next(a for a in attempts if a["subject_ref"] == "commit-demo-b")["status"] == "not_run"
    attempt = next(a for a in attempts if a["subject_ref"] == "commit-demo-a")
    assert attempt["pr"]["number"] == 101
    assert attempt["commit_oid"] == demo_store.require("commit-demo-a").to_dict()["payload"]["commit_oid"]


def test_metric_alias_does_not_turn_mfu_into_mxu(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-metrics"
    data["payload"]["request_id"] = "request-metrics"
    data["payload"]["source"]["implementation_overrides"]["candidate"] = "example-candidate"
    data["payload"]["source"]["implementation_overrides"]["compiler_evidence"] = {
        "llo": {"status": "recorded", "sha256": ["sha256:" + "a" * 64]}}
    base = data["payload"]["analysis_metrics"][0]
    data["payload"]["analysis_metrics"] = [
        {**base, "name": "effective_mfu_percent", "status": "observed", "value": 74.84321, "unit": "percent"},
        {**base, "name": "logical_bandwidth_gbps", "status": "observed", "value": 1200.0, "unit": "GB/s"},
        {**base, "name": "measured_hbm_bandwidth_gbps", "status": "unsupported", "value": None, "unit": "GB/s"},
    ]
    demo_store.publish(Record.from_dict(data))
    result = documents_of(build_output_memory(demo_store), "result")[0]
    run = next(r for r in result["runs"] if r["run_ref"] == "run-metrics")
    assert run["candidate"] == "example-candidate"
    assert run["mfu"] == 0.7484
    assert run["mxu"] is None
    assert run["logical_bandwidth_gbps"] == 1200.0
    assert run["hbm_bandwidth_gbps"] is None
    assert run["llo"] == "recorded"
    assert "compiler" not in run


def test_fraction_scalars_preserve_zero_and_over_peak_estimates(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    data = record_dict(bundle_dicts, "run-demo-a")
    data["record_id"] = "run-fraction-metrics"
    data["payload"]["request_id"] = "request-fraction-metrics"
    data["payload"]["timing"].update(median_us=12.34567, p90_us=14.98765)
    base = data["payload"]["analysis_metrics"][0]
    observations = [
        ("effective_mfu_percent", 0, "percent", "derived"),
        ("device_executed_mfu_percent_estimate", 125.678, "percent", "static_estimate"),
        ("attention_mxu_static_utilization_percent", 62.3456, "percent", "static_estimate"),
        ("boundary_roofline_percent", 102.55555, "percent", "derived"),
        ("logical_bandwidth_gbps", 1200.12345, "GB/s", "derived"),
        ("logical_bandwidth_percent", 120.12345, "percent", "derived"),
        ("measured_hbm_bandwidth_gbps", 0, "GB/s", "measurement"),
        ("measured_hbm_bandwidth_percent", 0, "percent", "measurement"),
    ]
    data["payload"]["analysis_metrics"] = [
        {**base, "name": name, "status": "observed", "value": value, "unit": unit, "kind": kind}
        for name, value, unit, kind in observations]
    demo_store.publish(Record.from_dict(data))
    run = next(r for r in documents_of(build_output_memory(demo_store), "result")[0]["runs"]
               if r["run_ref"] == "run-fraction-metrics")
    assert run["mfu"] == 0
    assert run["mfu_executed_estimate"] == 1.2568
    assert run["attention_mxu_static"] == 0.6235
    assert run["mxu"] is None
    assert run["roofline_efficiency"] == 1.0256
    assert run["logical_bandwidth_gbps"] == 1200.123
    assert run["logical_bandwidth_utilization"] == 1.2012
    assert run["hbm_bandwidth_gbps"] == 0
    assert run["hbm_bandwidth_utilization"] == 0
    assert run["latency_us"] == 12.346
    assert run["p90_latency_us"] == 14.988
    assert run["samples"] == 5


def test_baselines_remain_in_their_shape(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    config = record_dict(bundle_dicts, "cfg-demo")
    config["record_id"] = "cfg-second"
    config["payload"]["problem"]["n"] = 32
    config["payload"]["config_id"] = "demo-n32-f32"
    config["payload"]["config_hash"] = config_hash(**{k: config["payload"][k] for k in ("kernel_id", "problem_schema_id", "problem_schema_digest", "problem")})
    baseline = record_dict(bundle_dicts, "baseline-demo")
    baseline["record_id"] = "baseline-second"
    baseline["payload"].update(config_ref="cfg-second", baseline_id="second")
    demo_store.publish_bundle([Record.from_dict(config), Record.from_dict(baseline)])
    docs = build_output_memory(demo_store)
    shapes = documents_of(docs, "baseline")[0]["shapes"]
    assert len(shapes) == 2
    first = next(s for s in shapes if s["source_config_refs"] == ["cfg-demo"])
    second = next(s for s in shapes if s["source_config_refs"] == ["cfg-second"])
    assert first["baselines"][0]["runs"][0]["run_ref"] == "run-demo-baseline"
    assert set(first["baselines"][0]["runs"][0]) == RESULT_FIELDS
    assert second["baselines"][0]["runs"] == []
    assert second["shared_baselines"] == []


def test_shared_baseline_keeps_original_identity_and_compact_metrics(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    algorithm = record_dict(bundle_dicts, "algorithm-demo_vector_add-unspecified")
    algorithm["record_id"] = "algorithm-another-method"
    algorithm["payload"]["algorithm_id"] = "another-method"
    config = record_dict(bundle_dicts, "cfg-demo")
    config["record_id"] = "cfg-another-method"
    config["payload"]["algorithm_ref"] = algorithm["record_id"]
    demo_store.publish_bundle([Record.from_dict(algorithm), Record.from_dict(config)])
    docs = build_output_memory(demo_store)
    baseline = next(d for d in documents_of(docs, "baseline") if d["algorithm_ref"] == algorithm["record_id"])
    assert baseline["shapes"][0]["baselines"] == []
    shared = baseline["shapes"][0]["shared_baselines"][0]
    assert shared["source_algorithm_ref"] == "algorithm-demo_vector_add-unspecified"
    assert shared["source_config_ref"] == "cfg-demo"
    assert shared["baseline_ref"] == "baseline-demo"
    assert shared["runs"][0]["comparison_key"] == demo_store.require("run-demo-baseline").payload.comparison_key
    assert set(shared["runs"][0]) == RESULT_FIELDS
    assert sum(len(d["runs"]) for d in documents_of(docs, "result")) == 4


def test_group_decode_tuning_without_merging_different_abi(tmp_path: Path, bundle_dicts: list[dict]) -> None:
    # The source schema remains an existing record contract; the output groups
    # only the known decode implementation parameters in its own view schema.
    keep = [record_dict(bundle_dicts, rid) for rid in ("kernel-demo", "algorithm-demo_vector_add-unspecified", "cfg-demo")]
    keep[0]["payload"]["kernel_id"] = "mla_decode"
    keep[1]["payload"]["kernel_id"] = "mla_decode"
    base = keep[2]
    base["payload"]["kernel_id"] = "mla_decode"
    base["payload"]["problem"] = {"batch": 8, "num_q_heads": 16, "capacity": 4096, "dtype": "bfloat16", "physical_cache_shape": [256, 64, 2, 640], "block_kv": 1024}
    base["payload"]["config_hash"] = config_hash(**{k: base["payload"][k] for k in ("kernel_id", "problem_schema_id", "problem_schema_digest", "problem")})
    variant = copy.deepcopy(base)
    variant["record_id"] = "cfg-tuning"
    variant["payload"]["problem"]["block_kv"] = 2048
    variant["payload"]["config_hash"] = config_hash(**{k: variant["payload"][k] for k in ("kernel_id", "problem_schema_id", "problem_schema_digest", "problem")})
    other_abi = copy.deepcopy(base)
    other_abi["record_id"] = "cfg-layout"
    other_abi["payload"]["problem"]["physical_cache_shape"] = [8, 4096, 640]
    other_abi["payload"]["config_hash"] = config_hash(**{k: other_abi["payload"][k] for k in ("kernel_id", "problem_schema_id", "problem_schema_digest", "problem")})
    store = MemoryStore.init(tmp_path / "source")
    store.publish_bundle([Record.from_dict(d) for d in [*keep, variant, other_abi]])
    configs = documents_of(build_output_memory(store), "config")
    assert len(configs) == 2
    grouped = next(c for c in configs if len(c["source_config_refs"]) == 2)
    assert grouped["source_config_refs"] == ["cfg-demo", "cfg-tuning"]
    assert "block_kv" not in grouped["problem"]
    assert grouped["problem"]["physical_cache_shape"] == [256, 64, 2, 640]


def test_declared_baseline_links_existing_commit_run_once(demo_store: MemoryStore, bundle_dicts: list[dict]) -> None:
    baseline = record_dict(bundle_dicts, "baseline-demo")
    baseline["record_id"] = "baseline-control-commit"
    baseline["payload"]["baseline_id"] = "control-commit"
    run = demo_store.require("run-demo-a")
    baseline["payload"]["commit_oid"] = run.to_dict()["payload"]["source"]["target_commit"]
    baseline["payload"]["entrypoint"] = run.payload.source.entrypoint
    explicit_run = run.to_dict()
    explicit_run["record_id"] = "run-control-commit"
    explicit_run["payload"]["request_id"] = "request-control-commit"
    explicit_run["payload"]["source"]["implementation_overrides"]["is_declared_baseline"] = True
    explicit_run["payload"]["source"]["implementation_overrides"]["source_identity_kind"] = "git_commit"
    demo_store.publish_bundle([Record.from_dict(baseline), Record.from_dict(explicit_run)])
    docs = build_output_memory(demo_store)
    baselines = documents_of(docs, "baseline")[0]["shapes"][0]["baselines"]
    control = next(b for b in baselines if b["baseline_ref"] == "baseline-control-commit")
    assert [r["run_ref"] for r in control["runs"]] == ["run-control-commit"]
    assert all(a["subject_ref"] != "baseline-control-commit" for a in documents_of(docs, "attempt")[0]["attempts"])
    assert len(documents_of(docs, "result")[0]["runs"]) == 5
    attempt = next(a for a in documents_of(docs, "attempt")[0]["attempts"] if a["subject_ref"] == "commit-demo-a")
    assert attempt["source_identity_kind"] == "git_commit"


def test_refuse_unrelated_destination_or_source_overlap(demo_store: MemoryStore, tmp_path: Path) -> None:
    with pytest.raises(InputError, match="separate"):
        export_output_memory(demo_store, demo_store.root)
    output = tmp_path / "readable"
    output.mkdir()
    foreign = output / "notes.txt"
    foreign.write_text("user content")
    with pytest.raises(InputError):
        export_output_memory(demo_store, output)
    assert foreign.read_text() == "user content"


def test_refresh_refuses_foreign_files_and_removes_only_obsolete_export_files(demo_store: MemoryStore, tmp_path: Path) -> None:
    output = tmp_path / "readable"
    export_output_memory(demo_store, output)
    foreign = output / "user-notes.txt"
    foreign.write_text("keep")
    with pytest.raises(InputError) as exc:
        export_output_memory(demo_store, output)
    assert exc.value.code == "OUTPUT_HAS_FOREIGN_FILES"
    foreign.unlink()
    empty = MemoryStore.init(tmp_path / "empty-source")
    report = export_output_memory(empty, output)
    assert report["counts"]["runs"] == 0
    assert [p.name for p in output.iterdir()] == ["manifest.json"]
    assert json.loads((output / "manifest.json").read_text())["files"] == {}


def test_verifier_detects_changed_bytes_and_broken_references(demo_store: MemoryStore, tmp_path: Path) -> None:
    output = tmp_path / "readable"
    export_output_memory(demo_store, output)
    path = next(output.rglob("result.json"))
    data = json.loads(path.read_text())
    data["runs"][0]["subject_ref"] = "absent-subject"
    path.write_text(json.dumps(data))
    report = verify_output_memory(output)
    assert report["ok"] is False
    assert {"HASH_MISMATCH", "MISSING_ATTEMPT"} <= {i["code"] for i in report["issues"]}
