"""Forward import keeps timing, correctness scopes and static metrics distinct."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from kernel_memory.domain.models import Record
from kernel_memory.services.output_memory import build_output_memory
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore


@pytest.fixture(scope="module")
def converter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_project2_mla_fwd.py"
    spec = importlib.util.spec_from_file_location("project2_fwd_converter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def source(root: Path) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "created_utc": "2026-09-04T05:50:11Z", "environment": {"device_kinds": ["TPU v6 lite"], "jax": "0.8.1"},
        "source_fingerprint": {"git_head": None, "source_archive_sha256": "a" * 64,
                               "files": {"kernel.py": {"sha256": "b" * 64}}}}), encoding="utf-8")
    common = {"scope": "whole", "output_mode": "output_only", "batch": "1", "heads": "16",
              "seq_len": "4096", "q_rank": "1536", "kv_rank": "512", "d_nope": "128",
              "d_rope": "64", "d_value": "128", "sm_scale": "0.07216878364870322",
              "numeric_class": "exact", "options_json": "{}", "sample_count": "3",
              "round_medians_ms": "[1,2,3]", "median_ms": "2", "p90_ms": "2.8",
              "useful_mfu_percent": "20", "executed_mfu_percent_estimate": "24"}
    rows = [{**common, "implementation": "leader_aligned_original", "run_id": "leader", "is_leader_aligned_original": "True"},
            {**common, "implementation": "N6_diagonal_panel512", "run_id": "n6", "is_leader_aligned_original": "False"}]
    _csv(root / "device_event_performance_mfu.csv", rows)
    _csv(root / "device_event_raw_samples.csv", [
        {"run_id": name, "event_scope": "pipeline", "device_duration_ms": value}
        for name in ("leader", "n6") for value in (1, 2, 3)])
    _csv(root / "compile.csv", [{"run_id": "leader", "result": "PASS"}, {"run_id": "n6", "result": "PASS"}])
    _csv(root / "accuracy.csv", [
        {"implementation": "leader_aligned_original", "run_id": "leader", "scope": "whole", "seq_len": "4096",
         "result": "PASS", "max_abs": ".01", "max_rel": ".02", "atol": ".01", "rtol": ".01", "reference": "independent all-key dense"},
        {"implementation": "N6_diagonal_panel512", "run_id": "n6-s1024-gate", "scope": "whole", "seq_len": "1024",
         "result": "PASS", "max_abs": ".01", "max_rel": ".02", "atol": ".01", "rtol": ".01", "reference": "S1024 dense"}])
    _csv(root / "llo_static_utilization.csv", [{"run_id": "n6", "bundle_owner": "attention", "resource": "MXU",
                                               "average_static_utilization": ".55"}])
    return root


def test_forward_source_scope_and_metric_semantics(converter, tmp_path: Path) -> None:
    records = converter.build_records(source(tmp_path / "source"))
    for record in records:
        Record.from_dict(record)
    runs = {r["payload"]["source"]["implementation_overrides"]["candidate"]: r["payload"]
            for r in records if r["record_type"] == "run"}
    assert runs["leader_aligned_original"]["correctness"]["status"] == "pass"
    n6 = runs["N6_diagonal_panel512"]
    assert n6["correctness"]["status"] == "not_run"
    assert n6["source"]["tested_commit"] == {"algorithm": "sha256", "hex": "a" * 64}
    assert n6["source"]["implementation_overrides"]["source_identity_kind"] == "source_archive_content_address"
    assert n6["timing"]["median_us"] == 2000
    metrics = {m["name"]: m for m in n6["analysis_metrics"]}
    assert metrics["attention_mxu_static_utilization_percent"]["kind"] == "static_estimate"
    assert metrics["attention_mxu_static_utilization_percent"]["value"] == pytest.approx(55)
    assert metrics["hbm_bandwidth_utilization_percent"]["value"] is None
    assert metrics["hbm_bandwidth_utilization_percent"]["status"] == "not_collected"
    assert len([r for r in records if r["record_type"] == "config"]) == 2
    algorithms = {r["payload"]["algorithm_id"] for r in records if r["record_type"] == "algorithm"}
    assert algorithms == {"expanded-online-softmax", "diagonal-panel-attention"}
    assert next(r for r in records if r["record_type"] == "pr")["payload"]["provider"] == "local"
    problem = next(r for r in records if r["record_type"] == "config")["payload"]["problem"]
    assert problem["input_signature"]["q_pe"] == {"shape": [1, 16, 4096, 64], "dtype": "bfloat16"}
    assert problem["input_signature"]["w_q_nope"]["shape"] == [16, 1536, 128]
    assert set(problem["input_signature"]) == {"q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v"}


def test_forward_store_valid_and_idempotent(converter, tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    root = tmp_path / "memory"
    first = converter.import_forward(src, root)
    assert first["records_published"] == 10
    assert first["artifacts_stored"] == 0
    assert deep_validate(MemoryStore(root)).ok
    exported = build_output_memory(MemoryStore(root))
    diagonal = next(doc for path, doc in exported.items()
                    if path.startswith("mla_fwd/diagonal-panel-attention/") and path.endswith("trajectory/result.json"))
    run = diagonal["runs"][0]
    assert run["candidate"] == "N6_diagonal_panel512"
    assert run["mfu"] == 0.2
    assert run["mfu_executed_estimate"] == 0.24
    assert run["attention_mxu_static"] == 0.55
    assert run["mxu"] is None
    assert run["correctness"] == "not_run"
    assert run["latency_us"] == 2000
    assert run["hbm_bandwidth_utilization"] is None
    assert "metrics" not in run
    baseline = exported["mla_fwd/diagonal-panel-attention/baseline.json"]["shapes"][0]
    assert baseline["baselines"] == []
    assert len(baseline["shared_baselines"]) == 1
    reference = baseline["shared_baselines"][0]
    assert reference["source_algorithm_ref"] == "algorithm-mla_fwd-expanded-online-softmax"
    assert reference["runs"][0]["mfu"] == 0.2
    again = converter.import_forward(src, root)
    assert again["records_published"] == 0
    assert again["records_idempotent"] == 10


def test_forward_rejects_inconsistent_samples(converter, tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    path = src / "device_event_performance_mfu.csv"
    path.write_text(path.read_text().replace(",2,2.8,", ",4,2.8,"), encoding="utf-8")
    with pytest.raises(ValueError, match="median_ms"):
        converter.build_records(src)


def test_forward_rejects_mismatched_raw_samples(converter, tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    raw = src / "device_event_raw_samples.csv"
    raw.write_text(raw.read_text().replace("pipeline,3", "pipeline,4"), encoding="utf-8")
    with pytest.raises(ValueError, match="raw device sample"):
        converter.build_records(src)


def test_forward_rejects_bad_source_hash(converter, tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    path = src / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_fingerprint"]["files"]["kernel.py"]["sha256"] = "unrecorded"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        converter.build_records(src)


def test_forward_compiler_summary_uses_actual_files_not_collect_flag(converter, tmp_path: Path) -> None:
    src = source(tmp_path / "source")
    bundle_dir = src / "llo" / "n6" / "attention"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "123-attention-70-final_bundles.txt").write_text("recorded compiler bundles\n", encoding="utf-8")
    _csv(src / "compile.csv", [
        {"run_id": "leader", "result": "PASS", "collect_llo": "True", "llo_directory": "", "final_bundle_count": "0"},
        {"run_id": "n6", "result": "PASS", "collect_llo": "True", "llo_directory": "llo/n6", "final_bundle_count": "1"},
    ])
    records = converter.build_records(src)
    runs = {r["payload"]["source"]["implementation_overrides"]["candidate"]: r["payload"]
            for r in records if r["record_type"] == "run"}
    leader = runs["leader_aligned_original"]["source"]["implementation_overrides"]["compiler_evidence"]
    n6 = runs["N6_diagonal_panel512"]["source"]["implementation_overrides"]["compiler_evidence"]
    assert leader["llo"]["status"] == "not_collected"
    assert n6["hlo"]["status"] == "not_collected"
    assert n6["llo"]["status"] == "recorded"
    assert n6["llo"]["file_count"] == 1
    assert len(n6["llo"]["sha256"]) == 1
    assert runs["N6_diagonal_panel512"]["artifacts"] == []
