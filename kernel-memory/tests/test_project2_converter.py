"""Project2 imports retain the standard Memory layout without evidence copies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from kernel_memory.domain import hashing
from kernel_memory.services.trajectory import verify_kernel_trajectory
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore


@pytest.fixture(scope="module")
def converter() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_project2_mla.py"
    spec = importlib.util.spec_from_file_location("project2_converter_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(root: Path, *, batch: int = 1, timestamp: str = "2026-10-09T00:00:00Z") -> Path:
    root.mkdir(parents=True)
    entry = (
        "HloModule mla, entry_computation_layout={("
        f"bf16[{batch},16,576],bf16[{batch},576],bf16[{batch},64,2,640],s32[{batch}]"
        f")->(bf16[{batch},16,512],bf16[{batch},64,2,640])}}"
    )
    results = {
        "finished_utc": timestamp,
        "configuration": {"batch": batch, "heads": 16, "seq": 128, "scale": 0.07216878364870323,
                          "seed": 0, "warmup": 2, "samples": 3},
        "hlo_export": {"entry": entry},
        "last_candidate": "G",
        "candidates": [{"name": "G", "status": "MEASURED", "compile_status": "PASS",
                        "device_samples_ms": [0.010, 0.012, 0.011],
                        "extension_metadata": {"commit": "a" * 40, "page_size": 128,
                                               "block_kv": 2048, "pages_per_block": 16,
                                               "decode_batch_size": 2, "two_step": True}}],
        "environment": {"device_platform": "tpu", "device_kind": "synthetic TPU"},
    }
    (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (root / "performance.csv").write_text(
        "name,device_median_us,custom_call_median_us,logical_bandwidth_gbps\nG,11,9,100\n",
        encoding="utf-8",
    )
    (root / "accuracy.csv").write_text(
        "name,status,max_abs,relative_l2\nG,PASS,0.001,0.0001\n", encoding="utf-8",
    )
    evidence = {
        "candidate_status.csv": "name,status\nG,MEASURED\n",
        "mla.export.json": json.dumps({"entry": entry}),
        "mla.hlo": entry,
        "run_console.log": "synthetic measured result\n",
        "G/before_opt.hlo": entry,
        "G/before_opt.stablehlo.mlir": "module {}\n",
        "G/device_events.json": "{}",
        "G/profile_test/trace.json.gz": "synthetic profile bytes",
        "G/profile_test/device.xplane.pb": "synthetic xplane bytes",
        "source_snapshot/source_manifest.json": "{}",
        "source_snapshot/kernel.py": "def kernel(): pass\n",
        "launcher_source/runner.py": "# synthetic launcher\n",
    }
    for relative, content in evidence.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def payload(bundle: dict, record_type: str) -> dict:
    return next(record["payload"] for record in bundle["records"] if record["record_type"] == record_type)


def test_summary_has_no_copies(converter: ModuleType, tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "summary"
    bundle_path = converter.convert_rerun(source, output, summary_only=True)
    assert list(output.iterdir()) == [bundle_path]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    run = payload(bundle, "run")
    assert run["evidence_mode"] == "summary_only"
    assert run["provenance"] == "imported_unverified"
    # Unidentified legacy imports keep their existing entrypoint convention.
    assert run["source"]["entrypoint"] == "project2:PallasKernels/pallas_kernels/kernels/mla_decode.py"
    assert run["artifacts"] == []
    assert run["correctness"]["status"] == "pass"
    assert run["correctness"]["report_artifact_ref"] is None
    assert run["timing"] == {"status": "recorded", "sample_count": 3,
                             "median_us": 11, "p90_us": pytest.approx(11.8), "samples_artifact_ref": None}
    assert run["analysis_metrics"]
    assert all(metric["source_artifact_ref"] is None for metric in run["analysis_metrics"])
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


def test_full_keeps_evidence(converter: ModuleType, tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    output = tmp_path / "full"
    bundle_path = converter.convert_rerun(source, output)
    run = payload(json.loads(bundle_path.read_text(encoding="utf-8")), "run")
    artifacts = run["artifacts"]
    kinds = {artifact["kind"] for artifact in artifacts}
    assert {"mla.device_events", "mla.hlo_before_opt", "mla.stablehlo_before_opt", "mla.profile",
            "mla.source_snapshot_zip", "mla.input_manifest", "mla.output_manifest", "mla.timing_samples"} <= kinds
    assert sum(artifact["kind"] == "mla.profile" for artifact in artifacts) == 2
    assert all((output / artifact["uri"]).is_file() for artifact in artifacts)
    assert run.get("evidence_mode", "full") != "summary_only"
    assert run["timing"]["samples_artifact_ref"] is not None
    assert run["correctness"]["report_artifact_ref"] is not None
    assert all(metric["source_artifact_ref"] is not None for metric in run["analysis_metrics"])


def test_store_layout(converter: ModuleType, tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    root = tmp_path / "memory"
    result = converter.import_rerun(source, root, summary_only=True)
    assert result["records_published"] == 5
    assert result["artifacts_stored"] == 0
    assert result["summary_only"] is True
    store = MemoryStore.open(root)
    assert len(store.records()) == 5
    config = store.records("config")[0]
    baseline = store.records("baseline")[0]
    run = store.records("run")[0]
    kernel_dir = root / "kernels" / "mla_decode"
    config_dir = kernel_dir / "paged-native" / config.record_id
    baseline_dir = config_dir / "baselines" / baseline.record_id
    assert (root / "manifest.json").is_file()
    assert (kernel_dir / "kernel.json").is_file()
    assert (kernel_dir / "paged-native" / "algorithm.json").is_file()
    assert (config_dir / "config.json").is_file()
    assert (baseline_dir / "baseline.json").is_file()
    assert (baseline_dir / "runs" / run.record_id / "run.json").is_file()
    assert (root / "journal" / "records.jsonl").is_file()
    assert (root / ".cache" / "index.sqlite").is_file()
    assert (kernel_dir / "trajectory" / "trajectory.json").is_file()
    assert not list(root.rglob("bundle.json"))
    assert not any(path.is_file() for path in (root / "artifacts").rglob("*"))
    assert store.artifact_registry() == []
    report = deep_validate(store)
    assert report.ok
    assert report.errors == []
    assert report.artifact_checks == 0
    assert {issue.code for issue in report.issues} == {"SUMMARY_ONLY_RUN"}
    assert all(issue.severity == "warning" for issue in report.issues)
    verified = verify_kernel_trajectory(store, "mla_decode")
    assert verified["matches"] is True
    assert verified["stored_view_consistent"] is True


def test_multiple_shapes(converter: ModuleType, tmp_path: Path) -> None:
    first = make_source(tmp_path / "s1")
    second = make_source(tmp_path / "s2", batch=2, timestamp="2026-10-10T00:00:00Z")
    root = tmp_path / "memory"
    converter.import_rerun(first, root, summary_only=True)
    result = converter.import_rerun(second, root, summary_only=True)
    assert result["records_published"] == 3
    assert result["records_idempotent"] == 2
    store = MemoryStore.open(root)
    assert len(store.records("kernel")) == len(store.records("algorithm")) == 1
    assert len(store.records("config")) == len(store.records("baseline")) == len(store.records("run")) == 2
    assert store.records("kernel")[0].created_at == "2026-10-09T00:00:00Z"
    assert {record.created_at for record in store.records("run")} == {"2026-10-09T00:00:00Z", "2026-10-10T00:00:00Z"}
    journal = (root / "journal" / "records.jsonl").read_bytes()
    for source in (first, second):
        repeated = converter.import_rerun(source, root, summary_only=True)
        assert repeated["records_published"] == 0
        assert repeated["records_idempotent"] == 5
        assert repeated["artifacts_stored"] == 0
    assert (root / "journal" / "records.jsonl").read_bytes() == journal
    assert not list(root.rglob("bundle.json"))
    reopened = MemoryStore.open(root)
    assert len(reopened.records()) == 8
    assert deep_validate(reopened).ok
    assert verify_kernel_trajectory(reopened, "mla_decode")["matches"] is True


def test_summary_rejects_archive(converter: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot retain an archive"):
        converter.convert_rerun(tmp_path / "source", tmp_path / "out",
                                source_archive=tmp_path / "source.zip", summary_only=True)
    assert not (tmp_path / "out").exists()


def test_summary_rejects_wrong_median(converter: ModuleType, tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    (source / "performance.csv").write_text("name,device_median_us\nG,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="median does not match"):
        converter.import_rerun(source, tmp_path / "memory", summary_only=True)
    assert not (tmp_path / "memory" / "manifest.json").exists()


@pytest.mark.parametrize(
    ("candidate_name", "two_step", "block_kv"),
    [("G_T1_X0_F32S", True, 2048), ("M_T0_K4096_B2", False, 4096)],
)
def test_paged_upstream_source_identity(
    converter: ModuleType, tmp_path: Path, candidate_name: str, two_step: bool, block_kv: int,
) -> None:
    source = make_source(tmp_path / "source")
    results_path = source / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["last_candidate"] = candidate_name
    candidate = results["candidates"][0]
    candidate["name"] = candidate_name
    candidate["extension_metadata"].update({
        "source": "tpu-inference",
        "commit": "a0135382e192ae03b4c79a5b30b740a9a31f38b1",
        "two_step": two_step,
        "block_kv": block_kv,
        "pages_per_block": block_kv // 128,
        "transpose_cache": False,
        "score_dtype": "float32",
        "precision_category": "controlled_fp32_scores_bf16_prob",
    })
    results_path.write_text(json.dumps(results), encoding="utf-8")
    for filename in ("performance.csv", "accuracy.csv"):
        path = source / filename
        path.write_text(path.read_text(encoding="utf-8").replace("\nG,", f"\n{candidate_name},"), encoding="utf-8")

    root = tmp_path / "memory"
    converter.import_rerun(source, root, candidate_name=candidate_name, summary_only=True)
    store = MemoryStore.open(root)
    baseline = store.records("baseline")[0].to_dict()["payload"]
    run = store.records("run")[0].to_dict()["payload"]
    run_source = run["source"]
    expected_entrypoint = "project2:source_snapshot/upstream_adapters.py:build_upstream_native"
    assert baseline["entrypoint"] == run_source["entrypoint"] == expected_entrypoint
    assert run_source["implementation_overrides"]["candidate"] == candidate_name
    old_variant = hashing.variant_digest(
        source_digest=run_source["source_digest"],
        entrypoint="project2:PallasKernels/pallas_kernels/kernels/mla_decode.py",
        implementation_overrides=run_source["implementation_overrides"],
        checkout_mode=run_source["checkout_mode"],
    )
    assert run_source["variant_digest"] != old_variant
    # Deep validation checks the persisted variant and comparison identities,
    # including that all references agree after changing the source entrypoint.
    assert deep_validate(store).ok
    assert verify_kernel_trajectory(store, "mla_decode")["matches"] is True
