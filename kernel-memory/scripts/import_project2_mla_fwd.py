#!/usr/bin/env python3
"""Import archived project2 MLA forward measurements without evidence copies.

This reads the standard CSV/manifest outputs in result_6_tpu and
baseline_unoptimized_tpu_20260907. It does not execute TPU code. The default
scope is the complete forward pipeline; projection and row-state microbenchmarks
must be requested explicitly and are not compared with complete-pipeline runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from kernel_memory.domain import hashing  # noqa: E402

SCHEMA_VERSION = "0.3.0"
KERNEL_ID = "mla_fwd"
SCHEMA_ID = "urn:kernel-memory:problem:mla-fwd-project2:v1"
REPO_UID = "project2:mla_fwd"
ENTRYPOINT = "project2:pallas_kernels/kernels/mla_fwd.py"
METHOD_SUMMARY = (
    "基于 TPU Pallas 的 expanded dense causal MLA forward。先将压缩 KV 投影为各头的 K/V，"
    "在 query 分块中投影 Q，计算包含 NoPE 与 RoPE 的 causal attention，并用 FP32 online "
    "softmax 状态合并 KV 块，返回 BF16 O，可选返回自然对数 LSE。原始实现分别计算 K 与 V "
    "投影，并在 attention 中融合 Q 投影；这是后续方法共用的历史比较基线。"
)
PROBLEM_SCHEMA = {
    "$id": SCHEMA_ID,
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "scope", "batch", "num_q_heads", "seq_len", "q_rank", "kv_rank",
                 "d_nope", "d_rope", "d_value", "dtype", "output_mode", "sm_scale"],
    "properties": {
        **{k: {"type": "integer", "minimum": 1} for k in
           ("batch", "num_q_heads", "seq_len", "q_rank", "kv_rank", "d_nope", "d_rope", "d_value")},
        "operation": {"const": KERNEL_ID}, "scope": {"enum": ["whole", "q_projection", "kv_projection", "row_state"]},
        "dtype": {"const": "bfloat16"}, "output_mode": {"enum": ["output_only", "output_lse"]},
        "sm_scale": {"type": "number", "exclusiveMinimum": 0},
        "causal": {"const": True},
        "input_signature": {"type": "object", "additionalProperties": False,
            "required": ["q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v"],
            "properties": {k: {"$ref": "#/$defs/tensor"} for k in
                           ("q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v")}},
        "output_signature": {"type": "object", "additionalProperties": False, "required": ["O"],
            "properties": {"O": {"$ref": "#/$defs/tensor"}, "LSE": {"$ref": "#/$defs/tensor"}}},
    },
    "$defs": {"tensor": {"type": "object", "additionalProperties": False, "required": ["shape", "dtype"],
        "properties": {"shape": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 1}},
                       "dtype": {"enum": ["bfloat16", "float32"]}}}},
    "allOf": [{"if": {"properties": {"scope": {"const": "whole"}}},
               "then": {"required": ["causal", "input_signature", "output_signature"]}}],
}


def _method(name: str, scope: str) -> tuple[str, str, str]:
    if scope != "whole":
        return scope.replace("_", "-"), scope, f"MLA forward 的 {scope} 独立微基准；不等同于完整 forward pipeline。"
    if name == "leader_aligned_original":
        return "expanded-online-softmax", "Expanded online-softmax MLA", METHOD_SUMMARY
    if name == "current_qk256_mask":
        return "aligned-qk-attention", "Aligned QK attention", "将 NoPE 与 RoPE 的 QK 收缩维从192补齐到256，以更规整的 MXU dot 计算分数；完整可见的历史块跳过 causal mask，维持 FP32 online softmax 和 BF16 PV 输入。"
    if name.startswith(("N13_", "N14_")):
        return "paired-query-attention", "Paired-query attention", "同一驻留 KV 块依次服务两个 Q512 子块，保留各自的 FP32 online-softmax 状态以复用 KV；N14进一步将 causal 对角分成256 panel。具体调度属于 attempt 配置。"
    if name == "E2_kv_head_group2":
        return "grouped-kv-projection-attention", "Grouped KV projection attention", "融合 K/V projection 并让两个 head 共同复用 latent KV 输入；attention 使用512对角 panel和 FP32 online softmax。此方法的 pipeline 收益来自投影分组。"
    if name == "approx_prescale_q_bf16":
        return "bf16-prescaled-query-attention", "BF16 prescaled-query attention", "在 FP32 中先对 Q 缩放，再转 BF16 进入 QK dot，配合512对角 panel attention。此方法改变 BF16 舍入位置，是近似数值变体；不作为 exact 方法的替代。"
    if name == "approx_n6_score_bf16":
        return "bf16-score-attention", "BF16 score attention", "在512对角 panel attention中将 score 降为 BF16，以改变临时值压力；这是近似数值变体，正确性和性能必须按此数值类别解读。"
    if name == "dense_jax":
        return "dense-jax-reference", "Dense JAX reference", "独立 dense JAX 矩阵运算参考；不同测量 scope 的参考数据分别保存，不与完整 pipeline 混排。"
    return "diagonal-panel-attention", "Diagonal-panel attention", "对 dense causal MLA 的对角区域进一步划分 panel，跳过确定无效的未来 token 计算；历史全可见 KV 块维持较大分块，并使用 FP32 online softmax 合并。attempt 分别记录512/256 panel、query子块、首块处理、遍历顺序和外层 buffering 等调度参数。"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{str(k).strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def _num(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite numeric result: {value!r}")
    return int(out) if out.is_integer() else out


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _record(kind: str, ident: str, when: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "record_type": kind, "record_id": ident,
            "created_at": when, "payload": payload}


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _metric(name: str, value: int | float | None, unit: str, kind: str, scope: str,
            definition: str, status: str = "observed") -> dict[str, Any]:
    return {"name": name, "status": status, "value": value, "unit": unit, "kind": kind,
            "scope": scope, "source_artifact_ref": None, "parser_id": "project2-mla-fwd-v1",
            "parser_version": "1", "definition": definition}


def _problem(row: dict[str, str]) -> dict[str, Any]:
    problem = {"operation": KERNEL_ID, "scope": row["scope"], "dtype": "bfloat16",
               "output_mode": row["output_mode"], "sm_scale": _num(row.get("sm_scale"))}
    for key in ("batch", "heads", "seq_len", "q_rank", "kv_rank", "d_nope", "d_rope", "d_value"):
        value = _num(row.get(key))
        if value is None or not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid forward shape field {key}: {row.get(key)!r}")
        problem["num_q_heads" if key == "heads" else key] = value
    if not problem["sm_scale"] or problem["sm_scale"] <= 0:
        raise ValueError("forward sm_scale must be positive")
    if row["scope"] == "whole":
        b, h, s = problem["batch"], problem["num_q_heads"], problem["seq_len"]
        rq, rkv = problem["q_rank"], problem["kv_rank"]
        dn, dr, dv = problem["d_nope"], problem["d_rope"], problem["d_value"]
        shapes = {"q_latent": [b, s, rq], "kv_latent": [b, s, rkv], "q_pe": [b, h, s, dr],
                  "k_pe": [b, s, dr], "w_q_nope": [h, rq, dn], "w_k_nope": [h, rkv, dn], "w_v": [h, rkv, dv]}
        problem["input_signature"] = {k: {"shape": v, "dtype": "bfloat16"} for k, v in shapes.items()}
        problem["output_signature"] = {"O": {"shape": [b, h, s, dv], "dtype": "bfloat16"}}
        if row["output_mode"] == "output_lse":
            problem["output_signature"]["LSE"] = {"shape": [b, h, s], "dtype": "float32"}
        problem["causal"] = True
    Draft202012Validator(PROBLEM_SCHEMA).validate(problem)
    return problem


def _source(manifest: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    fingerprint = manifest.get("source_fingerprint") or {}
    verified = manifest.get("source_verification") or {}
    git_sha = verified.get("git_commit") or fingerprint.get("git_head")
    archive_sha = fingerprint.get("source_archive_sha256") or manifest.get("embedded_source_sha256")
    files = fingerprint.get("files") or {}
    for relative, info in files.items():
        if not isinstance(info, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", ""))):
            raise ValueError(f"invalid recorded source hash for {relative}")
    source_digest = (hashing.source_digest((p, d["sha256"]) for p, d in files.items())
                     if files else "sha256:" + str(archive_sha or verified.get("kernel_sha256")))
    if isinstance(git_sha, str) and re.fullmatch(r"[0-9a-f]{40}", git_sha):
        oid = {"algorithm": "sha1", "hex": git_sha}
        identity_kind = "git_commit"
    elif isinstance(archive_sha, str) and re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        oid = {"algorithm": "sha256", "hex": archive_sha}
        identity_kind = "source_archive_content_address"
    else:
        raise ValueError("forward manifest must identify the measured source commit or archive SHA256")
    if not hashing.is_sha256_ref(source_digest):
        raise ValueError("invalid source digest in forward manifest")
    overrides = {**options, "source_identity_kind": identity_kind}
    return {"repo_uid": REPO_UID, "target_commit": oid, "tested_commit": oid, "tested_tree": None,
            "checkout_mode": "exact_commit", "merge_parent_oids": [], "dirty": False,
            "patch_digest": None, "source_digest": source_digest, "entrypoint": ENTRYPOINT,
            "implementation_overrides": overrides,
            "variant_digest": hashing.variant_digest(source_digest=source_digest, entrypoint=ENTRYPOINT,
                                                      implementation_overrides=overrides, checkout_mode="exact_commit")}


def _compiler_evidence(source_dir: Path, compile_row: dict[str, str]) -> dict[str, Any]:
    """Keep compact compiler availability without copying any IR or bundle files."""
    result = {level: {"status": "not_collected", "sha256": [], "file_count": 0}
              for level in ("hlo", "llo")}
    relative = compile_row.get("llo_directory")
    if not relative:
        return result
    directory = (source_dir / relative).resolve()
    if not directory.is_relative_to(source_dir):
        raise ValueError("compiler evidence directory must stay inside the input dataset")
    if not directory.is_dir():
        raise ValueError(f"recorded compiler evidence directory is missing: {relative}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    final_bundles = [path for path in files if path.name.endswith("-final_bundles.txt")]
    expected_bundles = int(compile_row.get("final_bundle_count") or 0)
    if len(final_bundles) != expected_bundles:
        raise ValueError(f"recorded final bundle count does not match source files: {relative}")
    for level, selected in (("hlo", [p for p in files if p.suffix in {".hlo", ".mlir"}]),
                            ("llo", final_bundles)):
        if selected:
            if any(not path.resolve().is_relative_to(source_dir) for path in selected):
                raise ValueError("compiler evidence file must stay inside the input dataset")
            result[level] = {"status": "recorded", "sha256": [_digest(p.read_bytes()) for p in selected],
                             "file_count": len(selected)}
    result["llo"]["archive_file_count"] = len(files)
    result["llo"]["bundle_owners"] = json.loads(compile_row.get("expected_bundle_owners") or "[]")
    return result


def build_records(source_dir: Path, *, candidates: list[str] | None = None,
                  scope: str = "whole") -> list[dict[str, Any]]:
    """Build validated-shape summary records for measured forward candidates."""
    source_dir = Path(source_dir).resolve()
    manifest_path = source_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    when = str(manifest["created_utc"]).replace("+00:00", "Z")
    dataset_id = _digest(manifest_path.read_bytes())[7:19]
    performance = _read_csv(source_dir / "device_event_performance_mfu.csv")
    host_rows = {r["run_id"]: r for r in _read_csv(source_dir / "performance_mfu.csv")}
    accuracy = _read_csv(source_dir / "accuracy.csv")
    static = _read_csv(source_dir / "llo_static_utilization.csv")
    pressure = _read_csv(source_dir / "llo_pressure_summary.csv")
    raw_samples = _read_csv(source_dir / "device_event_raw_samples.csv")
    compile_rows = {r["run_id"]: r for r in _read_csv(source_dir / "compile.csv")}
    selected = [r for r in performance if r.get("scope") == scope and
                (candidates is None or r.get("implementation") in candidates)]
    if not selected:
        raise ValueError(f"no measured forward candidates for scope {scope!r}")
    missing = set(candidates or []) - {r["implementation"] for r in selected}
    if missing:
        raise ValueError(f"requested candidates have no measurement: {sorted(missing)}")
    records = [
        _record("kernel", f"kernel-{KERNEL_ID}", when, {
            "kernel_id": KERNEL_ID, "display_name": "MLA forward", "adapter_id": "project2-mla-fwd-v1",
            "contract_notes": "Expanded dense causal MLA forward with seven BF16 inputs; produces O and optional natural-log LSE. Shape and timing scope are fixed per config. Imported historical TPU measurements have not been independently reproduced."}),
    ]
    algorithm_ids: set[str] = set()
    config_ids: set[str] = set()
    schema_digest = hashing.problem_schema_digest(PROBLEM_SCHEMA)
    for row in selected:
        name, original_run_id = row["implementation"], row["run_id"]
        algorithm_id, display_name, summary = _method(name, scope)
        algorithm_ref = f"algorithm-{KERNEL_ID}-{algorithm_id}"
        if algorithm_ref not in algorithm_ids:
            records.append(_record("algorithm", algorithm_ref, when, {"kernel_id": KERNEL_ID,
                "algorithm_id": algorithm_id, "display_name": display_name, "method_summary": summary,
                "summary_author": "agent", "tags": ["mla", "project2", "forward", scope]}))
            algorithm_ids.add(algorithm_ref)
        problem = _problem(row)
        config_hash = hashing.config_hash(kernel_id=KERNEL_ID, problem_schema_id=SCHEMA_ID,
                                         problem_schema_digest=schema_digest, problem=problem)
        method_key = hashlib.sha256(algorithm_id.encode()).hexdigest()[:8]
        config_ref = f"cfg-fwd-{method_key}-b{problem['batch']}-h{problem['num_q_heads']}-s{problem['seq_len']}-{config_hash[7:19]}"
        if config_ref not in config_ids:
            records.append(_record("config", config_ref, when, {"kernel_id": KERNEL_ID,
                "algorithm_ref": algorithm_ref, "config_id": config_ref[4:], "problem_schema_id": SCHEMA_ID,
                "problem_schema_digest": schema_digest, "problem": problem, "config_hash": config_hash,
                "tags": ["mla", "project2", "forward", "imported", scope]}))
            config_ids.add(config_ref)
        identity = _digest((dataset_id + "\n" + original_run_id).encode())[7:19]
        short_name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:32].strip("-")
        run_id = f"run-{short_name}-{identity}"
        options = {"candidate": name, "numeric_class": row.get("numeric_class", "unknown"),
                   "options": json.loads(row.get("options_json") or "{}"),
                   "tiling": {k: _num(row.get(k)) for k in ("block_b", "block_q", "block_kv", "block_rq", "block_rkv")},
                   "source_run_id": original_run_id,
                   "compiler_evidence": _compiler_evidence(source_dir, compile_rows.get(original_run_id, {}))}
        source = _source(manifest, options)
        if row.get("is_leader_aligned_original") == "True":
            subject_ref = f"baseline-{short_name}-{identity}"
            records.append(_record("baseline", subject_ref, when, {"config_ref": config_ref,
                "baseline_id": subject_ref[9:], "description": "Original forward implementation measured in this archived experiment.",
                "repo_uid": REPO_UID, "commit_oid": source["tested_commit"], "entrypoint": ENTRYPOINT, "role": "both"}))
        else:
            pr_ref = f"pr-local-{identity}"
            subject_ref = f"commit-{identity}"
            records.extend([
                _record("pr", pr_ref, when, {"config_ref": config_ref, "pr_key": f"local-{identity}",
                    "repo_uid": REPO_UID, "provider": "local", "number": None, "title": name,
                    "hypothesis": "Archived candidate trial; no per-candidate GitHub PR was recorded.", "origin_ref": None}),
                _record("commit", subject_ref, when, {"pr_ref": pr_ref, "repo_uid": REPO_UID,
                    "commit_oid": source["tested_commit"], "git_parent_oids": [], "diff_base_oid": None,
                    "source_available": False, "change_status": "not_extracted", "changes": [],
                    "summary": f"{name}: recorded options {json.dumps(options['options'], sort_keys=True)}. Source identity is {options.get('source_identity_kind', source['implementation_overrides']['source_identity_kind'])}; no candidate-specific Git commit or parent relationship is asserted.",
                    "summary_author": "agent", "diff_artifact_ref": None}),
            ])
        samples_ms = [float(v) for v in json.loads(row.get("round_medians_ms") or "[]")]
        if not samples_ms or any(not math.isfinite(v) or v <= 0 for v in samples_ms):
            raise ValueError(f"missing/invalid device samples for {name}")
        if int(row["sample_count"]) != len(samples_ms):
            raise ValueError(f"sample count mismatch for {name}")
        if raw_samples:
            raw_scope = "pipeline" if scope == "whole" else scope
            measured_raw = [float(r["device_duration_ms"]) for r in raw_samples
                            if r.get("run_id") == original_run_id and r.get("event_scope") == raw_scope]
            if len(measured_raw) != len(samples_ms) or any(
                    not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                    for a, b in zip(sorted(measured_raw), sorted(samples_ms))):
                raise ValueError(f"raw device sample series mismatch for {name}")
        for q, column in ((0.5, "median_ms"), (0.9, "p90_ms")):
            if not math.isclose(_quantile(samples_ms, q), float(row[column]), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"device {column} does not match samples for {name}")
        if compile_rows.get(original_run_id, {}).get("result") != "PASS":
            raise ValueError(f"missing successful compile record for {name}")
        own_accuracy = [r for r in accuracy if r.get("run_id") == original_run_id]
        correctness_note = ("same measured shape" if own_accuracy else
                            "no direct correctness row for this measured shape; separate small-shape gates are not substituted")
        passed = [r for r in own_accuracy if r.get("result") == "PASS"]
        correctness = {"status": "pass" if own_accuracy and len(passed) == len(own_accuracy) else "fail" if own_accuracy else "not_run",
            "cases_total": len(own_accuracy), "cases_passed": len(passed),
            "max_abs_error": max((float(r["max_abs"]) for r in own_accuracy if r.get("max_abs")), default=None),
            "max_rel_error": max((float(r["max_rel"]) for r in own_accuracy if r.get("max_rel")), default=None),
            "report_artifact_ref": None}
        env_raw = manifest.get("environment") or {}
        env = {"backend": "tpu", "accelerator_model": (env_raw.get("device_kinds") or ["unknown TPU"])[0],
               "device_count": int(env_raw.get("timed_chip_count", 1)), "topology": "single",
               "software": {k: str(env_raw[k]) for k in ("jax", "jaxlib", "libtpu", "python", "numpy") if k in env_raw},
               "execution_flags": {"scope": scope}, "host_timer_environment": {}, "unknown_required_fields": []}
        env["environment_hash"] = hashing.environment_hash(env)
        run_plan = manifest.get("run_plan") or {}
        protocol = {"protocol_id": "project2-mla-fwd-device-v1", "measurement_scope": scope,
                    "timing_method": "device_profiler", "include_compile": False, "include_transfers": False,
                    "warmup": int(run_plan.get("warmup", 0)), "repetitions": len(samples_ms),
                    "statistic": "median", "quantile_method": "linear",
                    "input_suite_hash": hashing.jcs_digest({"problem": problem, "dataset": dataset_id}), "capture_profile": True}
        protocol["protocol_hash"] = hashing.protocol_hash(protocol)
        verifier = {"verifier_id": "project2-mla-fwd-reference-v1",
                    "reference_source_hash": hashing.jcs_digest(sorted({r.get("reference", "") for r in own_accuracy})),
                    "suite_hash": hashing.jcs_digest(own_accuracy),
                    "tolerances": {"atol": float(own_accuracy[0].get("atol", .01)), "rtol": float(own_accuracy[0].get("rtol", .01))} if own_accuracy else {},
                    "nonfinite_policy": "reject"}
        verifier["verifier_hash"] = hashing.verifier_hash(verifier)
        metrics: list[dict[str, Any]] = []
        keys = {"median_ms": ("milliseconds", "measurement"), "p90_ms": ("milliseconds", "measurement"),
                "useful_mfu_percent": ("percent", "derived"), "executed_mfu_percent_estimate": ("percent", "derived"),
                "useful_tflops_per_second": ("TFLOP/s", "derived"), "executed_tflops_per_second_estimate": ("TFLOP/s", "derived"),
                "useful_flops": ("FLOP", "derived"), "executed_flops_estimate": ("FLOP", "static_estimate")}
        for prefix, measured_row in (("device", row), ("host", host_rows.get(original_run_id, {}))):
            for key, (unit, kind) in keys.items():
                value = _num(measured_row.get(key))
                if value is not None:
                    metrics.append(_metric(f"{prefix}_{key}", value, unit, kind, scope,
                                           f"{prefix} {key}; device events and synchronized host wall time are separate timing boundaries."))
        for item in static:
            if item.get("run_id") != original_run_id:
                continue
            owner, resource = item["bundle_owner"], item["resource"].lower().replace(":", "_")
            metrics.append(_metric(f"{owner}_{resource}_static_utilization_percent",
                float(item["average_static_utilization"]) * 100, "percent", "static_estimate", owner,
                "LLO compiler average static schedule utilization, not a runtime hardware counter or MFU."))
        for item in pressure:
            if item.get("run_id") != original_run_id:
                continue
            owner = item["bundle_owner"]
            for key in ("unique_spill_allocations", "vector_spill_fill_static_refs", "vector_spill_store_static_refs",
                        "known_unique_vector_spill_bytes", "static_vector_spill_reload_bytes_estimate", "static_vector_spill_store_bytes_estimate"):
                value = _num(item.get(key))
                if value is not None:
                    metrics.append(_metric(f"{owner}_{key}", value, "bytes" if "bytes" in key else "count",
                        "static_estimate", owner, "LLO textual/allocation estimate, not executed memory traffic."))
        for name_missing in ("roofline_efficiency_percent", "hbm_bandwidth_utilization_percent", "runtime_mxu_utilization_percent"):
            metrics.append(_metric(name_missing, None, "percent", "measurement", scope,
                "The source archive does not report this runtime metric; static FILL/SPILL is not a substitute.", "not_collected"))
        key = hashing.comparison_key(config_hash=config_hash, environment_hash=env["environment_hash"],
            protocol_hash=protocol["protocol_hash"], verifier_hash=verifier["verifier_hash"], checkout_mode="exact_commit")
        records.append(_record("run", run_id, when, {"subject_ref": subject_ref, "config_ref": config_ref,
            "request_id": run_id, "attempt_no": 1, "rerun_of": None, "stage": "benchmark", "provenance": "imported_unverified",
            "source": source, "environment": env, "protocol": protocol, "verifier": verifier,
            "execution_status": "succeeded", "failure_reason": None, "correctness": correctness,
            "timing": {"status": "recorded", "sample_count": len(samples_ms), "median_us": float(row["median_ms"]) * 1000,
                       "p90_us": float(row["p90_ms"]) * 1000, "samples_artifact_ref": None},
            "analysis_metrics": metrics, "analysis_conclusion": f"Historical project2 forward result ({correctness_note}); source accuracy references: {', '.join(sorted({r.get('reference', '') for r in own_accuracy}))}. Summary only; no new TPU execution. Static LLO metrics are not runtime utilization. Candidate numeric class: {row.get('numeric_class', 'unknown')}.",
            "artifacts": [], "comparison_key": key, "session_id": f"project2-fwd-{dataset_id}",
            "pair_id": None, "role_in_pair": None, "evidence_mode": "summary_only"}))
    return records


def convert_forward(source_dir: Path, output_dir: Path, *, candidates: list[str] | None = None,
                    scope: str = "whole") -> Path:
    records = build_records(source_dir, candidates=candidates, scope=scope)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps({"bundle_version": SCHEMA_VERSION, "is_fixture": False,
        "description": "Historical project2 MLA forward summaries", "records": records},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bundle_path


def import_forward(source_dir: Path, store_root: Path, *, candidates: list[str] | None = None,
                   scope: str = "whole") -> dict[str, Any]:
    from kernel_memory.services.importer import import_bundle
    from kernel_memory.services.trajectory import rebuild_kernel_trajectory
    from kernel_memory.storage import MemoryStore

    with tempfile.TemporaryDirectory(prefix="kmem-project2-fwd-") as staging:
        path = convert_forward(source_dir, Path(staging), candidates=candidates, scope=scope)
        store = MemoryStore.init(store_root)
        bundle = _read_json(path)
        for index, record in enumerate(bundle["records"]):
            if record["record_type"] not in {"kernel", "algorithm", "config"}:
                continue
            existing = store.get(record["record_id"])
            if existing is not None and existing.to_dict()["payload"] == record["payload"]:
                bundle["records"][index] = existing.to_dict()
        path.write_text(json.dumps(bundle, ensure_ascii=False) + "\n", encoding="utf-8")
        report = import_bundle(store, path, artifact_root=Path(staging))
        view = rebuild_kernel_trajectory(store, KERNEL_ID)
        store.rebuild_index()
        return {"root": str(store.root), "records_published": report.records_published,
                "records_idempotent": report.records_idempotent, "artifacts_stored": report.artifacts_stored,
                "summary_only": True, "trajectory": view["path"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", help="repeat to select candidate names; defaults to all in scope")
    parser.add_argument("--scope", default="whole", choices=("whole", "q_projection", "kv_projection", "row_state"))
    parser.add_argument("--store", action="store_true")
    args = parser.parse_args(argv)
    if args.store:
        print(json.dumps(import_forward(args.source, args.output, candidates=args.candidate, scope=args.scope), sort_keys=True))
    else:
        print(convert_forward(args.source, args.output, candidates=args.candidate, scope=args.scope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
