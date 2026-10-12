#!/usr/bin/env python3
"""Import Project2 decode matrices as local attempts, commits and result summaries.

The source directory is read-only. All candidates, including failed experiments,
are retained. PR numbers are left null because the result archives record local
trials, not GitHub PR membership. Upstream Git revisions and local source content
addresses are distinguished in each commit summary and implementation metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "scripts", _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import import_project2_mla as single  # noqa: E402
from kernel_memory.domain import hashing  # noqa: E402
from kernel_memory.domain.problems import Project2MlaDecodeProblem  # noqa: E402

KERNEL_ID = "mla_decode"
_METHODS = {
    "paged-attention": "基于 TPU Pallas 的分页 KV 缓存 MLA attention：将新增 token 写入已有缓存后，按 KV 分块计算 latent 与 RoPE 分数并进行 softmax，返回 latent 输出和更新缓存。G/M 候选分别记录单/双阶段调度、score 精度、缓存转置、分块、内部 batch 以及全部请求非空时的处理分支。",
    "sglang-paged-attention": "SGLang-JAX 的分页 MLA attention：使用每请求固定容量的分页缓存，追加新 token 后计算 latent attention。SGL_A/B/C 及 B1 变体比较每块页数、KV tile 大小和内部 decode batch 分组；这些调度变体分别保存在 attempt 中。",
    "dense-tiled-attention": "连续 KV 输入上的分块 MLA attention：按有效长度对 latent 与 RoPE 分数进行 masked softmax，再聚合 latent 输出。B/D/C/R2 候选比较完整扫描、空块裁剪、循环展开、精度、分块与 DMA 路径，输入准备计入 dense 测量边界。",
    "dense-aligned-dma-attention": "对齐 DMA 的连续输入 MLA attention：在计时 JIT 内将 Q/KV 特征从 576 对齐到 640，使用 BF16 QK 和概率操作数、FP32 累加，按 KV tile 计算 attention。A_K* 变体比较 tile 大小和 DMA2 调度，输出为 latent attention。",
}
_METHOD_SHORT = {"paged-attention": "pa", "sglang-paged-attention": "sgl", "dense-tiled-attention": "dt", "dense-aligned-dma-attention": "dma"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover_reruns(source: Path) -> list[Path]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"source must be a directory: {source}")
    if (source / "results.json").is_file():
        return [source]
    return sorted({p.parent for p in source.rglob("results.json") if p.is_file()}, key=lambda p: p.as_posix())


def _verify_source_snapshot(directory: Path, results: dict[str, Any]) -> None:
    snapshot = (directory / "source_snapshot").resolve()
    for relative, recorded in (results.get("source_sha256") or {}).items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError(f"source snapshot path escapes its directory: {relative}")
        if not path.is_file():
            raise ValueError(f"missing source snapshot file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(recorded).removeprefix("sha256:"):
            raise ValueError(f"source snapshot SHA-256 mismatch: {path}")


def _method(candidate: dict[str, Any]) -> str:
    name = str(candidate["name"])
    extension = candidate.get("extension_metadata") or {}
    if name.startswith("SGL_") or extension.get("source") == "sglang-jax":
        return "sglang-paged-attention"
    if name.startswith("A_"):
        return "dense-aligned-dma-attention"
    if name.startswith(("G_", "M_")) or extension.get("source") == "tpu-inference":
        return "paged-attention"
    return "dense-tiled-attention"


def _precision(candidate: dict[str, Any]) -> tuple[str, str]:
    ext = candidate.get("extension_metadata") or {}
    knobs = candidate.get("config") or {}
    category = str(candidate.get("numerical_category") or ext.get("precision_category") or "")
    if category == "upstream_bf16_scores_prob":
        return "bfloat16", "bfloat16"
    if category in {"FP32_probability", "fp32_operands_fp32_accumulation", "upstream_bf16_qk_fp32_scores_prob_pv"}:
        return "float32", "float32"
    return str(ext.get("score_dtype") or "float32"), ("bfloat16" if knobs.get("bf16_p", True) else "float32")


def _problem(results: dict[str, Any], candidate: dict[str, Any], directory: Path) -> dict[str, Any]:
    cfg = results["configuration"]
    ext = candidate.get("extension_metadata") or {}
    knobs = {**(candidate.get("config") or {}), **ext}
    batch, heads = int(cfg["batch"]), int(cfg["heads"])
    capacity, latent, rope = int(cfg["seq"]), int(cfg.get("latent_dim", 512)), int(cfg.get("rope_dim", 64))
    qdim = latent + rope
    dense = str(candidate.get("timing_boundary") or results.get("timing_boundary") or cfg.get("boundary")) == "dense"
    shapes = single._entry_shapes(results, directory, str(candidate["name"]))
    native_interface = "paged_native" if ext.get("source") in {"tpu-inference", "sglang-jax"} else single._interface_from_results(results, directory, str(candidate["name"]))
    interface = "dense_reference" if dense else native_interface
    paged = interface == "paged_native"
    page = int(knobs.get("page_size", 128)) if paged else None
    transposed = bool(knobs.get("transpose_cache", False)) and paged
    feature = qdim if dense else (int(shapes[2][-2] if transposed else shapes[2][-1]) if len(shapes) >= 3 else qdim)
    cache_shape = [batch, capacity, feature] if not paged else [batch * ((capacity + page - 1) // page), page // 2, 2, feature]
    if transposed:
        cache_shape = [batch * ((capacity + page - 1) // page), feature, page]
    if not dense and len(shapes) >= 5 and shapes[2] != cache_shape:
        raise ValueError(f"{candidate['name']}: recorded native HLO cache {shapes[2]} differs from contract {cache_shape}")
    tensor = lambda dtype, shape, layout: {"dtype": dtype, "shape": shape, "layout": layout}
    if dense:
        inputs = {"q": tensor("bfloat16", [batch, heads, qdim], "[B,H,q_dim]"), "kv": tensor("bfloat16", cache_shape, "[B,capacity,feature]"), "lengths": tensor("int32", [batch], "[B]")}
        outputs = {"output": tensor("bfloat16", [batch, heads, latent], "[B,H,latent_dim]")}
    else:
        inputs = {"q": tensor("bfloat16", [batch, heads, qdim], "[B,H,q_dim]"), "new_kv": tensor("bfloat16", [batch, qdim], "[B,q_dim]"), "cache": tensor("bfloat16", cache_shape, "paged packed cache" if paged else "[B,capacity,feature]"), "lengths": tensor("int32", [batch], "[B]")}
        outputs = {"attention": tensor("bfloat16", [batch, heads, latent], "[B,H,latent_dim]"), "updated_cache": tensor("bfloat16", cache_shape, "same as cache")}
    score, probability = _precision(candidate)
    return Project2MlaDecodeProblem().normalize({
        "operation": KERNEL_ID, "interface": interface, "batch": batch, "num_q_heads": heads,
        "capacity": capacity, "timed_length": int(cfg.get("length", capacity)), "latent_dim": latent,
        "rope_dim": rope, "q_dim": qdim, "cache_feature_dim": feature, "dtype": "bfloat16",
        "cache_layout": "dense" if dense else ("paged" if paged else "contiguous"),
        "score_dtype": score, "probability_dtype": probability, "query_tokens_per_request": 1,
        "physical_cache_shape": cache_shape, "input_signature": inputs, "output_signature": outputs,
        "input_generation": {"distribution": "standard_normal", "seed": int(cfg.get("seed", 0)), "quantization": "exact_bfloat16"},
        "block_size": int(knobs.get("block_kv", 1024)), "sm_scale": float(cfg.get("scale", 0.0)),
        "sliding_window": int(cfg.get("window", 0)), "soft_cap": float(cfg.get("soft_cap", 0.0)),
        "length_semantics": "valid_token_count" if dense else "append_then_attend", "page_size": page,
        "pages_per_block": int(knobs.get("pages_per_block", 16)) if paged else None,
        "block_kv": int(knobs.get("block_kv", 2048)) if paged else None,
        "decode_batch_size": int(knobs.get("decode_batch_size", 2)) if paged else None,
        "transpose_cache": bool(knobs.get("transpose_cache", False)), "two_step": bool(knobs.get("two_step", False)),
    })


def _metric(name: str, value: Any, unit: str = "scalar", kind: str = "derived", definition: str | None = None, status: str = "observed") -> dict[str, Any]:
    return {"name": name, "status": status, "value": value, "unit": unit, "kind": kind,
            "scope": "candidate", "source_artifact_ref": None, "parser_id": "project2-results-v2", "parser_version": "2",
            "definition": definition or f"project2 performance.csv field {name}"}


def _metrics(perf: dict[str, str], directory: Path, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    units = {"_us": "microseconds", "_ms": "milliseconds", "_gbps": "GB/s", "_percent": "percent", "_bytes": "bytes", "_per_second": "per_second", "_seconds": "seconds"}
    metrics = []
    for key, raw in perf.items():
        value = single._number(raw)
        if value is None or key == "eligible":
            continue
        unit = next((u for suffix, u in units.items() if key.endswith(suffix)), "scalar")
        kind = "measurement" if key.startswith(("device_", "host_wall_", "custom_call_", "outside_custom_call_", "compile_wall_")) else "derived"
        metrics.append(_metric(key, value, unit, kind))
    name = str(candidate["name"])
    metrics.append(_metric("hlo_recorded", int((directory / name / "before_opt.hlo").is_file()), "boolean", "measurement", "Whether this candidate has an archived HLO export; attachment omitted from summary memory."))
    metrics.append(_metric("llo_recorded", 0, "boolean", "measurement", "The project2 decode runner explicitly did not enable LLO; HLO is not LLO."))
    # Logical byte/time estimates are not hardware utilization counters.
    for metric_name, definition in (("mxu_utilization_percent", "No MXU hardware occupancy counter was recorded."), ("measured_hbm_bandwidth_percent", "No HBM hardware byte counters were recorded; logical bandwidth is a separate estimate.")):
        if not any(m["name"] == metric_name for m in metrics):
            metrics.append(_metric(metric_name, None, "percent", "measurement", definition, "not_collected"))
    return metrics


def _records(directory: Path, results: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(candidate["name"])
    cfg = results["configuration"]
    ext = candidate.get("extension_metadata") or {}
    method = _method(candidate)
    short = _METHOD_SHORT[method]
    problem = _problem(results, candidate, directory)
    adapter = Project2MlaDecodeProblem()
    schema_digest = adapter.schema_digest()
    config_hash = hashing.config_hash(kernel_id=KERNEL_ID, problem_schema_id=adapter.schema_id, problem_schema_digest=schema_digest, problem=problem)
    config_ref = f"cfg-{short}-b{problem['batch']}-h{problem['num_q_heads']}-n{problem['timed_length']}-{config_hash[7:19]}"
    algorithm_ref = f"algorithm-{KERNEL_ID}-{method}"
    raw_digest = hashlib.sha256((directory / "results.json").read_bytes()).hexdigest()
    variant = hashlib.sha256((raw_digest + name).encode()).hexdigest()[:12]
    trial_ref, commit_ref, run_ref = f"attempt-{variant}", f"commit-{variant}", f"run-{variant}"
    created = single._created_at(results)
    source_map = results.get("source_sha256") or {}
    source_digest = hashing.source_digest((str(k), str(v)) for k, v in source_map.items()) if source_map else "sha256:" + raw_digest
    upstream = str(ext.get("commit") or "")
    git_oid = bool(re.fullmatch(r"[0-9a-f]{40}", upstream))
    oid = {"algorithm": "sha1", "hex": upstream} if git_oid else {"algorithm": "sha256", "hex": source_digest[7:]}
    repo = "project2:" + str(ext.get("source") or "mla_decode")
    if method in {"paged-attention", "sglang-paged-attention"}:
        entry = "source_snapshot/upstream_adapters.py:build_upstream_" + ("dense" if problem["interface"] == "dense_reference" else "native")
    elif method == "dense-aligned-dma-attention":
        entry = "source_snapshot/aligned_kernels.py"
    elif name.startswith("R2_"):
        entry = "source_snapshot/round2_kernels.py"
    elif name.startswith("C"):
        entry = "source_snapshot/decode_extensions.py"
    else:
        entry = "source_snapshot/decode_kernels.py"
    # The candidate parameter is part of the recorded entrypoint identity. This
    # permits an exact baseline-to-attempt match when several variants share a Git SHA.
    entrypoint = f"project2:{entry}#candidate={name}"
    overrides = {"candidate": name, "interface": problem["interface"], "precision_category": candidate.get("numerical_category", ext.get("precision_category", "unknown")), "implementation_parameters": {**(candidate.get("config") or {}), **ext}, "revision_kind": "upstream_git_revision_with_local_wrapper" if git_oid else "source_snapshot_content_address", "timing_boundary": candidate.get("timing_boundary", cfg.get("boundary")), "is_declared_baseline": results.get("baseline_name") == name, "compiler_evidence": {"hlo": {"status": "recorded" if (directory / name / "before_opt.hlo").is_file() else "not_collected"}, "llo": {"status": "not_collected", "reason": str(results.get("llo", "not recorded"))}}}
    overrides["source_identity_kind"] = overrides["revision_kind"]
    source = {"repo_uid": repo, "target_commit": oid, "tested_commit": oid, "tested_tree": None, "checkout_mode": "exact_commit", "merge_parent_oids": [], "dirty": False, "patch_digest": None, "source_digest": source_digest, "entrypoint": entrypoint, "implementation_overrides": overrides, "variant_digest": hashing.variant_digest(source_digest=source_digest, entrypoint=entrypoint, implementation_overrides=overrides, checkout_mode="exact_commit")}
    env_raw = {"backend": str((results.get("environment") or {}).get("device_platform", "tpu")), "accelerator_model": str((results.get("environment") or {}).get("device_kind", "unknown TPU")), "device_count": 1, "topology": "single", "software": {k: str(v) for k, v in (results.get("environment") or {}).items() if isinstance(v, (str, int, float, bool))}, "execution_flags": {"timing_boundary": results.get("timing_boundary", cfg.get("boundary"))}, "host_timer_environment": {"timing_scope": results.get("timing_scope", "unknown")}, "unknown_required_fields": []}
    environment = {**env_raw, "environment_hash": hashing.environment_hash(env_raw)}
    samples = [float(v) * 1000 for v in (candidate.get("device_samples_ms") or [])]
    protocol_raw = {"protocol_id": "project2-mla-decode-v2", "measurement_scope": str(results.get("timing_scope", "unknown")), "timing_method": "device_profiler", "include_compile": False, "include_transfers": False, "warmup": int(cfg.get("warmup", 0)), "repetitions": len(samples) or int(cfg.get("samples", 1)), "statistic": "median", "quantile_method": "linear", "input_suite_hash": hashing.jcs_digest({"seed": cfg.get("seed", 0), "problem": problem}), "capture_profile": not bool(cfg.get("no_profile", False))}
    protocol = {**protocol_raw, "protocol_hash": hashing.protocol_hash(protocol_raw)}
    verifier_raw = {"verifier_id": "project2-mla-reference-v2", "reference_source_hash": hashing.jcs_digest(results.get("reference", {})), "suite_hash": hashing.jcs_digest({"gates": sorted((candidate.get("accuracy") or {}).keys()), "seed": cfg.get("seed", 0)}), "tolerances": {"atol": float(cfg.get("atol", 0.01)), "rtol": float(cfg.get("rtol", 0.01))}, "nonfinite_policy": "reject"}
    verifier = {**verifier_raw, "verifier_hash": hashing.verifier_hash(verifier_raw)}
    perf = next((row for row in single._read_csv(directory / "performance.csv") if row.get("name") == name), {})
    accuracy = [row for row in single._read_csv(directory / "accuracy.csv") if row.get("name") == name]
    recorded_gates = candidate.get("accuracy") or {}
    if accuracy and all(row.get("gate") for row in accuracy):
        if {row["gate"] for row in accuracy} != set(recorded_gates):
            raise ValueError(f"{directory}/{name}: accuracy gate names disagree between CSV and results.json")
        for row in accuracy:
            recorded_gate = recorded_gates[row["gate"]]
            if row.get("status") != recorded_gate.get("status"):
                raise ValueError(f"{directory}/{name}: accuracy gate status disagrees between CSV and results.json")
            for field in ("max_abs", "relative_l2", "bad_elements", "atol", "rtol"):
                csv_value, json_value = single._number(row.get(field)), single._number(recorded_gate.get(field))
                if csv_value != json_value:
                    raise ValueError(f"{directory}/{name}: accuracy {field} disagrees between CSV and results.json")
    accuracy_status = str(candidate.get("correctness_status", "NOT_RUN")).upper()
    passed = sum(row.get("status", "").upper() == "PASS" for row in accuracy)
    if accuracy_status == "PASS" and (not accuracy or passed != len(accuracy)):
        raise ValueError(f"{directory}/{name}: PASS disagrees with accuracy.csv")
    measured = str(candidate.get("status", "")).upper() == "MEASURED"
    if measured:
        if candidate.get("compile_status") != "PASS" or accuracy_status != "PASS":
            raise ValueError(f"{directory}/{name}: MEASURED requires passing compilation and accuracy")
        median = single._number(perf.get("device_median_us"))
        if not samples or median is None or not all(math.isfinite(x) and x > 0 for x in samples) or not math.isclose(single._quantile(samples, .5), float(median), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"{directory}/{name}: performance.csv disagrees with device samples")
        if len(samples) != int(cfg.get("samples", len(samples))):
            raise ValueError(f"{directory}/{name}: sample count disagrees with configured repetitions")
        for field, q in (("device_p10_ms", .1), ("device_p90_ms", .9)):
            value = single._number(perf.get(field))
            if value is not None and not math.isclose(float(value) * 1000, single._quantile(samples, q), rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"{directory}/{name}: {field} disagrees with device samples")
        timing = {"status": "recorded", "sample_count": len(samples), "median_us": median, "p90_us": single._quantile(samples, .9), "samples_artifact_ref": None}
    else:
        timing = {"status": "not_run", "sample_count": 0, "median_us": None, "p90_us": None, "samples_artifact_ref": None}
    has_accuracy = bool(accuracy)
    execution = "succeeded" if measured or has_accuracy else ("compile_error" if candidate.get("compile_status") == "FAIL" else "runtime_error")
    def max_error(field: str) -> float | None:
        vals = [single._number(row.get(field)) for row in accuracy]
        return max(float(v) for v in vals if v is not None) if any(v is not None for v in vals) else None
    correctness = {"status": ("pass" if passed == len(accuracy) else "fail") if has_accuracy else "not_run", "cases_total": len(accuracy), "cases_passed": passed, "max_abs_error": max_error("max_abs"), "max_rel_error": max_error("relative_l2"), "report_artifact_ref": None}
    conclusion = "Historical project2 result imported without attachments; not re-executed. max_rel_error stores the source relative L2 error. Logical bandwidth and roofline values are analytical byte/FLOP models, not MXU/HBM hardware counters."
    if candidate.get("failure_reason"):
        conclusion += " Original outcome: " + str(candidate["failure_reason"])
    record = lambda typ, ref, payload: single._record(typ, ref, created, payload)
    records = [
        record("kernel", "kernel-mla_decode", {"kernel_id": KERNEL_ID, "display_name": "MLA decode", "adapter_id": "project2-mla-v2", "contract_notes": "MLA single-query decode. Native input is q/new_kv/resident cache/postappend lengths; dense input is q/kv/lengths. Outputs are latent attention, and updated cache for native. Exact tensor ABI and precision are recorded per config. No final latent-to-value projection. Historical results are imported_unverified."}),
        record("algorithm", algorithm_ref, {"kernel_id": KERNEL_ID, "algorithm_id": method, "display_name": method, "method_summary": _METHODS[method], "summary_author": "agent", "tags": ["mla", "decode", "project2"]}),
        record("config", config_ref, {"kernel_id": KERNEL_ID, "algorithm_ref": algorithm_ref, "config_id": config_ref[4:], "problem_schema_id": adapter.schema_id, "problem_schema_digest": schema_digest, "problem": problem, "config_hash": config_hash, "tags": ["mla", "project2", "imported"]}),
        record("pr", trial_ref, {"config_ref": config_ref, "pr_key": "local-" + variant, "repo_uid": repo, "provider": "local", "number": None, "title": name, "hypothesis": "Imported local candidate trial; no PR association was recorded.", "origin_ref": None}),
        record("commit", commit_ref, {"pr_ref": trial_ref, "repo_uid": repo, "commit_oid": oid, "git_parent_oids": [], "diff_base_oid": None, "source_available": (directory / "source_snapshot").is_dir(), "change_status": "not_extracted", "changes": [], "summary": f"{name}: " + ("pinned upstream Git revision; local wrapper and variant recorded by source digest." if git_oid else "source snapshot SHA-256 content address; no Git commit was recorded."), "summary_author": "collector", "diff_artifact_ref": None}),
        record("run", run_ref, {"subject_ref": commit_ref, "config_ref": config_ref, "request_id": run_ref, "attempt_no": 1, "rerun_of": None, "stage": "benchmark", "provenance": "imported_unverified", "evidence_mode": "summary_only", "source": source, "environment": environment, "protocol": protocol, "verifier": verifier, "execution_status": execution, "failure_reason": str(candidate.get("failure_reason") or candidate.get("status")) if execution != "succeeded" else None, "correctness": correctness, "timing": timing, "analysis_metrics": _metrics(perf, directory, candidate), "analysis_conclusion": conclusion, "artifacts": [], "comparison_key": hashing.comparison_key(config_hash=config_hash, environment_hash=environment["environment_hash"], protocol_hash=protocol["protocol_hash"], verifier_hash=verifier["verifier_hash"], checkout_mode="exact_commit"), "session_id": "project2-" + raw_digest[:12], "pair_id": None, "role_in_pair": None}),
    ]
    if results.get("baseline_name") == name:
        records.append(record("baseline", "baseline-" + variant, {"config_ref": config_ref, "baseline_id": name.lower() + "-" + raw_digest[:12], "description": f"Source-declared baseline candidate {name}; measurement is retained on its local attempt.", "repo_uid": repo, "commit_oid": oid, "entrypoint": entrypoint, "role": "both"}))
    return records


def build_matrix_bundle(source: Path, output: Path, *, candidate_names: set[str] | None = None, summary_only: bool = True, measured_only: bool = False) -> Path:
    if not summary_only:
        raise ValueError("matrix import currently requires summary_only=True; use the single-result converter to retain attachments")
    reruns = discover_reruns(source)
    records: dict[str, dict[str, Any]] = {}
    for directory in reruns:
        results = _load(directory / "results.json")
        candidates = results.get("candidates")
        if not isinstance(candidates, list):
            continue
        _verify_source_snapshot(directory, results)
        for candidate in candidates:
            if candidate_names and candidate.get("name") not in candidate_names:
                continue
            if measured_only and candidate.get("status") != "MEASURED":
                continue
            for record in _records(directory, results, candidate):
                previous = records.get(record["record_id"])
                if previous is not None:
                    if previous["payload"] != record["payload"] or previous["record_type"] != record["record_type"]:
                        raise ValueError(f"record ID collision: {record['record_id']}")
                    continue
                records[record["record_id"]] = record
    if not records:
        raise ValueError(f"no matching candidates found below {source}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "bundle.json"
    path.write_text(json.dumps({"bundle_version": single.SCHEMA_VERSION, "is_fixture": False, "description": "Historical Project2 decode attempts and result summaries", "records": list(records.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def import_matrix(source: Path, store_root: Path, *, candidate_names: set[str] | None = None, summary_only: bool = True, measured_only: bool = False) -> dict[str, Any]:
    from kernel_memory.services.importer import import_bundle
    from kernel_memory.services.trajectory import rebuild_kernel_trajectory
    from kernel_memory.storage import MemoryStore
    with tempfile.TemporaryDirectory(prefix="kmem-project2-decode-") as tmp:
        path = build_matrix_bundle(source, Path(tmp), candidate_names=candidate_names, summary_only=summary_only, measured_only=measured_only)
        store = MemoryStore.init(store_root)
        bundle = _load(path)
        for i, record in enumerate(bundle["records"]):
            if record["record_type"] in {"kernel", "algorithm", "config"}:
                existing = store.get(record["record_id"])
                if existing is not None and existing.to_dict()["payload"] == record["payload"]:
                    bundle["records"][i] = existing.to_dict()
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = import_bundle(store, path, artifact_root=Path(tmp))
        view = rebuild_kernel_trajectory(store, KERNEL_ID)
        store.rebuild_index()
        return {"root": str(store.root), "records_published": report.records_published, "records_idempotent": report.records_idempotent, "artifacts_stored": report.artifacts_stored, "summary_only": True, "trajectory": view["path"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--summary-only", action="store_true", default=True)
    parser.add_argument("--measured-only", action="store_true")
    parser.add_argument("--store", action="store_true")
    args = parser.parse_args(argv)
    selected = set(args.candidates or []) or None
    if args.store:
        print(json.dumps(import_matrix(args.source, args.output, candidate_names=selected, measured_only=args.measured_only), sort_keys=True))
    else:
        print(build_matrix_bundle(args.source, args.output, candidate_names=selected, measured_only=args.measured_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
