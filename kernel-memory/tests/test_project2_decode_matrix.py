"""Matrix imports preserve attempts, ABI, failures and metric meaning."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kernel_memory.domain.problems import MlaDecodeProblem, Project2MlaDecodeProblem
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore


@pytest.fixture(scope="module")
def matrix():
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_project2_decode_matrix.py"
    spec = importlib.util.spec_from_file_location("decode_matrix_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(root: Path, *, dense: bool = False, time: str = "2026-09-18T00:00:00Z") -> Path:
    root.mkdir(parents=True)
    if dense:
        names = ["A_K1024"]
        hlos = {names[0]: "HloModule dense, entry_computation_layout={(bf16[1,16,576],bf16[1,128,576],s32[1])->bf16[1,16,512]}, frontend_attributes={}"}
    else:
        names = ["G_T1_X0_F32S", "G_T0_X0", "G_T0_X1"]
        hlos = {
            name: "HloModule decode, entry_computation_layout={(bf16[1,16,576],bf16[1,576]," + ("bf16[1,576,128]" if name.endswith("X1") else "bf16[1,64,2,640]") + ",s32[1])->(bf16[1,16,512]," + ("bf16[1,576,128]" if name.endswith("X1") else "bf16[1,64,2,640]") + ")}, frontend_attributes={}"
            for name in names
        }
    candidates = []
    for i, name in enumerate(names):
        candidate = {"name": name, "config": {"block_kv": 1024}, "timing_boundary": "dense" if dense else "native", "status": ["MEASURED", "CORRECTNESS_FAIL", "COMPILE_FAIL"][i], "compile_status": "FAIL" if i == 2 else "PASS", "correctness_status": ["PASS", "FAIL", "NOT_RUN"][i], "device_samples_ms": [.010, .012, .011] if i == 0 else [], "accuracy": {"timed_workload": {"status": "PASS" if i == 0 else "FAIL"}} if i < 2 else {}, "numerical_category": "explicit_bf16_qk_and_probability" if dense else ("controlled_fp32_scores_bf16_prob" if i == 0 else "upstream_bf16_scores_prob"), "failure_reason": None if i == 0 else "synthetic historical failure", "extension_metadata": {} if dense else {"source": "tpu-inference", "commit": "a" * 40, "block_kv": 2048, "page_size": 128, "pages_per_block": 16, "decode_batch_size": 2, "transpose_cache": i == 2}}
        candidates.append(candidate)
        directory = root / name
        directory.mkdir()
        (directory / "before_opt.hlo").write_text(hlos[name], encoding="utf-8")
    results = {"configuration": {"batch": 1, "heads": 16, "seq": 128, "length": 128, "samples": 3, "warmup": 2, "boundary": "dense" if dense else "native", "seed": 0}, "timing_boundary": "dense" if dense else "native", "candidates": candidates, "baseline_name": names[0], "finished_utc": time, "environment": {"device_platform": "tpu", "device_kind": "synthetic TPU"}, "llo": "Not enabled"}
    (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (root / "performance.csv").write_text("name,device_median_us,effective_mfu_percent,logical_bandwidth_percent\n" + "\n".join(f"{n},11,3,50" if i == 0 else f"{n},,," for i, n in enumerate(names)) + "\n", encoding="utf-8")
    (root / "accuracy.csv").write_text("name,status,max_abs,relative_l2\n" + f"{names[0]},PASS,0.001,0.0001\n" + (f"{names[1]},FAIL,0.2,0.1\n" if len(names) > 1 else ""), encoding="utf-8")
    (root / "source_snapshot").mkdir()
    return root


def test_matrix_retains_failed_attempts_and_transposed_abi(matrix, tmp_path):
    root = tmp_path / "memory"
    report = matrix.import_matrix(source(tmp_path / "input"), root)
    assert report["artifacts_stored"] == 0
    store = MemoryStore.open(root)
    assert len(store.records("pr")) == len(store.records("commit")) == len(store.records("run")) == 3
    assert len(store.records("baseline")) == 1
    assert all(r.payload.provider == "local" and r.payload.number is None for r in store.records("pr"))
    runs = {r.payload.source.implementation_overrides["candidate"]: r.to_dict()["payload"] for r in store.records("run")}
    correct_failure = runs["G_T0_X0"]
    assert correct_failure["execution_status"] == "succeeded"
    assert correct_failure["correctness"]["status"] == "fail"
    assert correct_failure["timing"]["status"] == "not_run"
    compile_failure = runs["G_T0_X1"]
    assert compile_failure["execution_status"] == "compile_error"
    assert compile_failure["correctness"]["status"] == "not_run"
    config = store.require(compile_failure["config_ref"]).payload
    assert config.problem["interface"] == "paged_native"
    assert config.problem["physical_cache_shape"] == [1, 576, 128]
    assert config.problem["score_dtype"] == "bfloat16"
    assert config.problem_schema_id == Project2MlaDecodeProblem.schema_id
    assert config.problem_schema_digest != MlaDecodeProblem().schema_digest()
    measured = runs["G_T1_X0_F32S"]
    metrics = {m["name"]: m for m in measured["analysis_metrics"]}
    assert metrics["effective_mfu_percent"]["value"] == 3
    assert metrics["mxu_utilization_percent"]["status"] == "not_collected"
    assert metrics["mxu_utilization_percent"]["value"] is None
    assert metrics["hlo_recorded"]["value"] == 1
    assert measured["source"]["implementation_overrides"]["is_declared_baseline"] is True
    assert deep_validate(store).ok
    assert matrix.import_matrix(tmp_path / "input", root)["records_published"] == 0


def test_dense_local_snapshot_keeps_three_input_contract(matrix, tmp_path):
    matrix.import_matrix(source(tmp_path / "input", dense=True), tmp_path / "memory")
    store = MemoryStore.open(tmp_path / "memory")
    config = store.records("config")[0].payload
    assert set(config.problem["input_signature"]) == {"q", "kv", "lengths"}
    assert set(config.problem["output_signature"]) == {"output"}
    assert config.problem["length_semantics"] == "valid_token_count"
    commit = store.records("commit")[0].payload
    assert commit.commit_oid.algorithm == "sha256"
    assert "no Git commit was recorded" in commit.summary
    assert deep_validate(store).ok


def test_same_shape_repeated_measurements_share_config(matrix, tmp_path):
    source(tmp_path / "matrix" / "run_00", time="2026-09-18T00:00:00Z")
    source(tmp_path / "matrix" / "run_01", time="2026-09-19T00:00:00Z")
    matrix.import_matrix(tmp_path / "matrix", tmp_path / "memory")
    store = MemoryStore.open(tmp_path / "memory")
    assert len(store.records("config")) == 3
    assert len(store.records("run")) == 6
    assert deep_validate(store).ok


def test_bad_sample_summary_rejected_before_store_creation(matrix, tmp_path):
    src = source(tmp_path / "input")
    perf = src / "performance.csv"
    perf.write_text(perf.read_text().replace(",11,3,50", ",999,3,50"))
    with pytest.raises(ValueError, match="performance.csv disagrees"):
        matrix.import_matrix(src, tmp_path / "memory")
    assert not (tmp_path / "memory" / "manifest.json").exists()


def test_tampered_source_snapshot_rejected(matrix, tmp_path):
    src = source(tmp_path / "input")
    result_path = src / "results.json"
    results = json.loads(result_path.read_text())
    results["source_sha256"] = {"kernel.py": "0" * 64}
    result_path.write_text(json.dumps(results))
    (src / "source_snapshot" / "kernel.py").write_text("# tampered\n")
    with pytest.raises(ValueError, match="source snapshot SHA-256 mismatch"):
        matrix.build_matrix_bundle(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
