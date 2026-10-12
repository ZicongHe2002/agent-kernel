#!/usr/bin/env python3
"""Convert one project2 MLA decode result directory to a Kernel Memory bundle.

The converter only reads ``project2``.  It writes a self-contained directory with
``bundle.json`` and copied evidence files, so the resulting bundle can be imported
with ``kmem import-bundle --artifact-root <bundle-directory>``.  One bundle is
created for one measured shape/candidate; a matrix directory must be converted one
rerun directory at a time.

This importer deliberately records historical results as ``imported_unverified``.
The source files and measurements are retained as artifacts, while the typed record
contains only the fixed MLA decode contract and scalar summaries.

``--store`` publishes directly into the standard Memory directory and removes
the temporary bundle. ``--summary-only`` retains historical scalar results
without copying evidence files or leaving artifact references.
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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kernel_memory.domain import hashing  # noqa: E402
from kernel_memory.domain.problems import MlaDecodeProblem  # noqa: E402

SCHEMA_VERSION = "0.3.0"
KERNEL_ID = "mla_decode"
SCHEMA_ID = "urn:kernel-memory:problem:mla-decode:v1"
ENTRYPOINT = "project2:PallasKernels/pallas_kernels/kernels/mla_decode.py"
PAGED_UPSTREAM_ENTRYPOINT = "project2:source_snapshot/upstream_adapters.py:build_upstream_native"
REPO_UID = "project2:mla_decode"
CONTRACT_NOTES = (
    "Single-token MLA decode: q[B,H,latent_dim+rope_dim], "
    "new_kv[B,latent_dim+rope_dim], a resident native KV cache, and lengths[B] "
    "produce latent attention output[B,H,latent_dim] and updated_cache. "
    "Lengths count valid tokens after append; append at lengths-1 before attending. "
    "Empty requests produce zero output and do not append. The final latent-to-value "
    "projection is excluded. For paged caches, only valid token/feature positions "
    "are guaranteed; future slots and padding are unspecified, and historical KV "
    "values must be finite. Tensor shapes, layouts and numerical settings are "
    "specified per config. Historical project2 runs are imported_unverified."
)
METHOD_SUMMARIES = {
    "paged_native": (
        "基于 TPU Pallas 的分页 KV 缓存 MLA 单 token 解码。输入已准备好的查询、"
        "新增 KV、分页缓存和追加后的有效长度；先为每个非空请求写入一个 KV token，"
        "再按 KV 分块，利用 latent 与 RoPE 特征计算 attention 分数，经 softmax "
        "加权得到 latent 输出，同时返回更新后的缓存，不包含最终的 latent-to-value 投影。"
        "G 系列提供单阶段或双阶段 attention 调度；M 系列在相同分页缓存表示上调整 "
        "KV 分块和内部 batch 调度，并为全部请求非空的情况增加元数据处理快速路径。"
        "具体精度、分块及调度参数见 config，实际候选名称见 run。"
    ),
    "contiguous_native": (
        "使用连续 KV 缓存的 Project2 MLA 单 token 解码。接收查询、新增 KV、"
        "已有缓存和有效长度；先为每个非空请求追加一个 KV token，再计算 attention，"
        "返回 latent attention 输出和更新后的缓存。具体精度与分块参数见 config，"
        "实际候选名称见 run。"
    ),
}


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    # Keep integral counters as integers in the JSON identity and metrics.
    return int(out) if out.is_integer() else out


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]+", "_", value).strip("_") or "unknown"


def _source_entrypoint(interface: str, candidate_name: str, extension: dict[str, Any]) -> str:
    # The archived G/M candidates wrap the pinned tpu-inference kernel in
    # upstream_adapters.build_upstream_native, not the PallasKernels module.
    if (interface == "paged_native" and extension.get("source") == "tpu-inference"
            and re.fullmatch(r"G_T[01]_X[01](?:_F32S)?|M_T[01]_K[0-9]+_B[0-9]+", candidate_name)):
        return PAGED_UPSTREAM_ENTRYPOINT
    return ENTRYPOINT


def _created_at(results: dict[str, Any]) -> str:
    value = results.get("finished_utc") or results.get("started_utc")
    if not isinstance(value, str) or not value:
        return "1970-01-01T00:00:00Z"
    # project2 emits ISO timestamps with +00:00; Kernel Memory accepts either.
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "1970-01-01T00:00:00Z"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path) -> list[dict[str, str]]:
    # ``project2`` writes UTF-8 CSV with an optional BOM on some notebook
    # runtimes.  Normalise both the header and cell whitespace so matching a
    # candidate remains deterministic across Windows/Linux copies.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [{str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in rows]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _record(record_type: str, record_id: str, created_at: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "record_id": record_id,
        "created_at": created_at,
        "payload": payload,
    }


def _artifact(source: Path, artifact_root: Path, artifact_id: str, *, kind: str, media_type: str) -> dict[str, Any]:
    """Copy one source file and return its fixed Artifact descriptor."""
    data = source.read_bytes()
    # Callers may include the complete relative source name in ``artifact_id``
    # (profile traces do this so two files with different extensions remain
    # distinct).  Avoid producing names such as ``trace.json.gz.gz`` while
    # retaining the historical convenience of appending a suffix for IDs that
    # only contain a logical stem.
    destination_name = _safe_id(artifact_id)
    suffix = source.suffix
    if suffix and not destination_name.lower().endswith(suffix.lower()):
        destination_name += suffix
    destination = artifact_root / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "uri": destination.relative_to(artifact_root.parent).as_posix(),
        "sha256": _sha256(data),
        "size_bytes": len(data),
        "media_type": media_type,
        "retention": "permanent",
        "availability": "present",
    }


def _generated_artifact(data: bytes, artifact_root: Path, artifact_id: str, *, kind: str, media_type: str) -> dict[str, Any]:
    destination = artifact_root / f"{_safe_id(artifact_id)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "uri": destination.relative_to(artifact_root.parent).as_posix(),
        "sha256": _sha256(data),
        "size_bytes": len(data),
        "media_type": media_type,
        "retention": "permanent",
        "availability": "present",
    }


def _canonical_source_zip(source_dir: Path, artifact_root: Path, artifact_id: str) -> dict[str, Any] | None:
    """Create a reproducible ZIP of source_snapshot and launcher source files.

    Zip timestamps and host-specific permissions are fixed, and paths are sorted;
    this makes the evidence digest stable when the same project2 result is copied
    between Windows and Linux.
    """
    roots = [source_dir / "source_snapshot", source_dir / "launcher_source"]
    files: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files.append((path.relative_to(source_dir).as_posix(), path))
    if not files:
        return None
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            info.external_attr = 0
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return _generated_artifact(buffer.getvalue(), artifact_root, f"{artifact_id}.zip", kind="mla.source_snapshot_zip", media_type="application/zip")


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".log": "text/plain",
        ".hlo": "text/plain",
        ".mlir": "text/plain",
    }.get(path.suffix.lower(), "application/octet-stream")


def _hlo_entry_text(results: dict[str, Any], source_dir: Path | None = None, candidate_name: str | None = None) -> str:
    entry = str(results.get("hlo_export", {}).get("entry", ""))
    if entry:
        return entry
    if source_dir is not None:
        candidates = [source_dir / str(candidate_name) / "before_opt.hlo"] if candidate_name else []
        candidates.extend(sorted(source_dir.glob("*/before_opt.hlo")))
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "bf16[" in text:
                return text
    return ""


def _interface_from_results(results: dict[str, Any], source_dir: Path | None = None, candidate_name: str | None = None) -> str:
    entry = _hlo_entry_text(results, source_dir, candidate_name)
    # Rank-4 [tile,page,2,feature] cache is the native paged ABI.  A rank-3
    # [B,capacity,feature] cache is the contiguous ABI.
    return "paged_native" if re.search(r"bf16\[[0-9]+,[0-9]+,2,[0-9]+\]", entry) else "contiguous_native"


def _entry_shapes(results: dict[str, Any], source_dir: Path | None = None, candidate_name: str | None = None) -> list[list[int]]:
    """Extract the static BF16 shapes printed in the exported HLO entry layout."""
    entry = _hlo_entry_text(results, source_dir, candidate_name)
    layout = re.search(r"entry_computation_layout=\{(.*?)\},\s*(?:frontend_attributes|metadata|$)", entry, flags=re.DOTALL)
    if layout:
        entry = layout.group(1)
    shapes: list[list[int]] = []
    for match in re.finditer(r"bf16\[([0-9,]+)\]", entry):
        try:
            shapes.append([int(dim) for dim in match.group(1).split(",")])
        except ValueError:
            continue
    return shapes


def _candidate(results: dict[str, Any], name: str | None) -> dict[str, Any]:
    candidates = results.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("results.json has no candidates")
    if name is None:
        name = str(results.get("last_candidate") or candidates[0].get("name"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("name") == name:
            return candidate
    raise ValueError(f"candidate {name!r} is not present in results.json")


def convert_rerun(
    source_dir: Path,
    output_dir: Path,
    *,
    candidate_name: str | None = None,
    source_archive: Path | None = None,
    summary_only: bool = False,
) -> Path:
    """Convert one project2 ``rerun`` directory and return its bundle path."""
    if summary_only and source_archive is not None:
        raise ValueError("summary-only output cannot retain an archive")
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    results_path = source_dir / "results.json"
    performance_path = source_dir / "performance.csv"
    accuracy_path = source_dir / "accuracy.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    candidate = _candidate(results, candidate_name)
    candidate_name = str(candidate["name"])
    cfg = dict(results.get("configuration") or {})
    interface = _interface_from_results(results, source_dir, candidate_name)
    batch = int(cfg.get("batch", 0))
    heads = int(cfg.get("heads", 0))
    capacity = int(cfg.get("seq", cfg.get("capacity", 0)))
    length = int(cfg.get("length", capacity))
    latent_dim = int(cfg.get("latent_dim", 512))
    rope_dim = int(cfg.get("rope_dim", 64))
    q_dim = latent_dim + rope_dim
    entry_shapes = _entry_shapes(results, source_dir, candidate_name)
    cache_feature_dim = int(entry_shapes[2][-1]) if len(entry_shapes) >= 3 and entry_shapes[2] else int(q_dim + latent_dim)
    input_shapes: dict[str, list[int]] = {
        "q": [batch, heads, q_dim],
        "new_kv": [batch, q_dim],
        "lengths": [batch],
    }
    output_shapes: dict[str, list[int]] = {
        "attention": [batch, heads, latent_dim],
    }
    # The physical cache shape is ABI evidence.  Keep it separate from the
    # logical capacity because paged allocators commonly reserve extra slots.
    if len(entry_shapes) >= 5:
        input_shapes["cache"] = entry_shapes[2]
        output_shapes["updated_cache"] = entry_shapes[4]
    else:
        input_shapes["cache"] = [batch, capacity, cache_feature_dim]
        output_shapes["updated_cache"] = [batch, capacity, cache_feature_dim]
    problem = {
        "operation": "mla_decode",
        "interface": interface,
        "batch": batch,
        "num_q_heads": heads,
        "capacity": capacity,
        "timed_length": length,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "q_dim": q_dim,
        "cache_feature_dim": cache_feature_dim,
        "dtype": "bfloat16",
        "cache_layout": "paged" if interface == "paged_native" else "contiguous",
        "score_dtype": "float32",
        "probability_dtype": "bfloat16",
        "query_tokens_per_request": 1,
        "physical_cache_shape": input_shapes["cache"],
        "input_signature": {
            "q": {"dtype": "bfloat16", "shape": input_shapes["q"], "layout": "[B,H,q_dim]"},
            "new_kv": {"dtype": "bfloat16", "shape": input_shapes["new_kv"], "layout": "[B,q_dim]"},
            "cache": {"dtype": "bfloat16", "shape": input_shapes["cache"], "layout": "paged packed cache" if interface == "paged_native" else "[B,capacity,feature]"},
            "lengths": {"dtype": "int32", "shape": [batch], "layout": "[B]"},
        },
        "output_signature": {
            "attention": {"dtype": "bfloat16", "shape": output_shapes["attention"], "layout": "[B,H,latent_dim]"},
            "updated_cache": {"dtype": "bfloat16", "shape": output_shapes["updated_cache"], "layout": "same as cache"},
        },
        "input_generation": {"distribution": "standard_normal", "seed": int(cfg.get("seed", 0)), "quantization": "exact_bfloat16"},
        "block_size": int((candidate.get("extension_metadata") or {}).get("block_kv", 1024)),
        "sm_scale": float(cfg.get("scale", 0.0)),
        "sliding_window": int(cfg.get("window", 0)),
        "soft_cap": float(cfg.get("soft_cap", 0.0)),
        "length_semantics": "append_then_attend",
        "page_size": int((candidate.get("extension_metadata") or {}).get("page_size", 128)) if interface == "paged_native" else None,
        "pages_per_block": int((candidate.get("extension_metadata") or {}).get("pages_per_block", 16)) if interface == "paged_native" else None,
        "block_kv": int((candidate.get("extension_metadata") or {}).get("block_kv", 2048)) if interface == "paged_native" else None,
        "decode_batch_size": int((candidate.get("extension_metadata") or {}).get("decode_batch_size", 2)) if interface == "paged_native" else None,
        "transpose_cache": bool((candidate.get("extension_metadata") or {}).get("transpose_cache", False)),
        "two_step": bool((candidate.get("extension_metadata") or {}).get("two_step", False)),
    }
    # Normalize through the same adapter used by register-config.  This is a
    # hard gate: an imported record cannot silently invent a second MLA schema.
    adapter = MlaDecodeProblem()
    problem = adapter.normalize(problem)
    schema_digest = adapter.schema_digest()
    config_hash = hashing.config_hash(
        kernel_id=KERNEL_ID,
        problem_schema_id=SCHEMA_ID,
        problem_schema_digest=schema_digest,
        problem=problem,
    )
    config_ref = f"cfg-{interface}-b{batch}-h{heads}-cap{capacity}-len{length}-{config_hash.split(':', 1)[1][:12]}"
    algorithm_id = "paged-native" if interface == "paged_native" else "contiguous-native"
    algorithm_ref = f"algorithm-{KERNEL_ID}-{algorithm_id}"
    run_slug = _safe_id(candidate_name)
    run_id = f"run-project2-{interface}-b{batch}-h{heads}-cap{capacity}-len{length}-{run_slug}"
    baseline_id = f"project2-{interface}-b{batch}-h{heads}-cap{capacity}-len{length}-{run_slug}"
    baseline_ref = f"baseline-{baseline_id}"
    created_at = _created_at(results)
    if summary_only:
        # Keep native Windows paths readable and below the legacy path limit.
        # The source-result digest distinguishes separate historical runs.
        result_id = _sha256(results_path.read_bytes())[7:19]
        config_ref = f"cfg-b{batch}-h{heads}-n{length}-{config_hash.split(':')[1][:12]}"
        run_id = f"run-{run_slug.lower()}-{result_id}"
        baseline_id = f"{run_slug.lower()}-{result_id}"
        baseline_ref = f"baseline-{baseline_id}"

    extension = candidate.get("extension_metadata") if isinstance(candidate.get("extension_metadata"), dict) else {}
    entrypoint = _source_entrypoint(interface, candidate_name, extension)
    commit_hex = str(extension.get("commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit_hex):
        raise ValueError("candidate extension_metadata.commit must be a full 40-character Git SHA-1")
    oid = {"algorithm": "sha1", "hex": commit_hex}
    source_hashes = results.get("source_sha256") if isinstance(results.get("source_sha256"), dict) else {}
    if source_hashes:
        source_digest = hashing.source_digest((str(k), str(v)) for k, v in source_hashes.items())
    else:
        source_digest = _sha256(results_path.read_bytes())
    overrides = {
        "candidate": candidate_name,
        "interface": interface,
        "precision_category": extension.get("precision_category", "unknown"),
    }
    source = {
        "repo_uid": REPO_UID,
        "target_commit": oid,
        "tested_commit": oid,
        "tested_tree": None,
        "checkout_mode": "exact_commit",
        "merge_parent_oids": [],
        "dirty": False,
        "patch_digest": None,
        "source_digest": source_digest,
        "entrypoint": entrypoint,
        "implementation_overrides": overrides,
        "variant_digest": hashing.variant_digest(
            source_digest=source_digest,
            entrypoint=entrypoint,
            implementation_overrides=overrides,
            checkout_mode="exact_commit",
        ),
    }

    environment_raw = {
        "backend": str((results.get("environment") or {}).get("device_platform", "tpu")),
        "accelerator_model": str((results.get("environment") or {}).get("device_kind", "unknown TPU")),
        "device_count": 1,
        "topology": "single",
        "software": {k: str(v) for k, v in (results.get("environment") or {}).items() if isinstance(v, (str, int, float, bool))},
        "execution_flags": {"timing_boundary": results.get("timing_boundary", "native"), "candidate": candidate_name},
        "host_timer_environment": {"timing_scope": results.get("timing_scope", "unknown")},
        "unknown_required_fields": [],
    }
    environment = {**environment_raw, "environment_hash": hashing.environment_hash(environment_raw)}

    samples_ms = candidate.get("device_samples_ms")
    samples_us = [float(v) * 1000.0 for v in samples_ms] if isinstance(samples_ms, list) else []
    if not samples_us:
        # A row without raw device samples cannot be represented as a measured timing.
        samples_us = []
    repetitions = len(samples_us) or int(cfg.get("samples", 1))
    input_suite_hash = hashing.jcs_digest({"seed": cfg.get("seed", 0), "problem": problem, "source": source_digest})
    protocol_raw = {
        "protocol_id": "project2-mla-native-v1",
        "measurement_scope": str(results.get("timing_scope", "native decode")),
        "timing_method": "device_profiler" if samples_us else "host_synchronized",
        "include_compile": False,
        "include_transfers": False,
        "warmup": int(cfg.get("warmup", 0)),
        "repetitions": repetitions,
        "statistic": "median",
        "quantile_method": "linear",
        "input_suite_hash": input_suite_hash,
        "capture_profile": bool(not cfg.get("no_profile", False)),
    }
    protocol = {**protocol_raw, "protocol_hash": hashing.protocol_hash(protocol_raw)}
    accuracy_bytes = accuracy_path.read_bytes() if accuracy_path.is_file() else b""
    verifier_raw = {
        "verifier_id": "project2-mla-reference-v1",
        "reference_source_hash": _sha256(str(results.get("reference", {})).encode("utf-8")),
        "suite_hash": _sha256(accuracy_bytes),
        "tolerances": {"atol": float(cfg.get("atol", 0.01)), "rtol": float(cfg.get("rtol", 0.01))},
        "nonfinite_policy": "reject",
    }
    verifier = {**verifier_raw, "verifier_hash": hashing.verifier_hash(verifier_raw)}
    comparison_key = hashing.comparison_key(
        config_hash=config_hash,
        environment_hash=environment["environment_hash"],
        protocol_hash=protocol["protocol_hash"],
        verifier_hash=verifier["verifier_hash"],
        checkout_mode="exact_commit",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    if not summary_only:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    selected = [
        ("results.json", "mla.results_json"),
        ("performance.csv", "mla.performance_csv"),
        ("accuracy.csv", "mla.accuracy_csv"),
        ("candidate_status.csv", "mla.candidate_status_csv"),
        ("mla.export.json", "mla.hlo_export_metadata"),
        ("run_console.log", "mla.console_log"),
        ("mla.hlo", "mla.hlo_export"),
    ]
    for name, kind in ([] if summary_only else selected):
        path = source_dir / name
        if path.is_file():
            artifacts.append(_artifact(path, artifact_dir, f"{run_id}-{_safe_id(path.stem)}", kind=kind, media_type=_media_type(path)))
    # Include candidate-level device events and pre-optimisation IR when present.
    candidate_dir = source_dir / candidate_name
    for name, kind in (("device_events.json", "mla.device_events"), ("before_opt.hlo", "mla.hlo_before_opt"), ("before_opt.stablehlo.mlir", "mla.stablehlo_before_opt")):
        path = candidate_dir / name
        if not summary_only and path.is_file():
            artifacts.append(_artifact(path, artifact_dir, f"{run_id}-{_safe_id(path.stem)}", kind=kind, media_type=_media_type(path)))
    # Preserve profile traces for the selected candidate as evidence.  Profiles
    # are optional and remain external to the typed memory record; copying their
    # original bytes keeps the record reproducible without embedding binary data.
    if not summary_only and candidate_dir.is_dir():
        for path in sorted(candidate_dir.glob("profile_*/**/*")):
            if path.is_file():
                relative = path.relative_to(candidate_dir).as_posix()
                artifact_id = f"{run_id}-profile-{_safe_id(relative)}"
                artifacts.append(_artifact(path, artifact_dir, artifact_id, kind="mla.profile", media_type=_media_type(path)))
    source_manifest_path = source_dir / "source_snapshot" / "source_manifest.json"
    if not summary_only and source_manifest_path.is_file():
        artifacts.append(_artifact(source_manifest_path, artifact_dir, f"{run_id}-source-manifest", kind="mla.source_manifest", media_type="application/json"))
    zip_ref = None if summary_only else _canonical_source_zip(source_dir, artifact_dir, f"{run_id}-source-snapshot")
    if zip_ref is not None:
        artifacts.append(zip_ref)
    if source_archive is not None:
        source_archive = Path(source_archive).resolve()
        if not source_archive.is_file():
            raise FileNotFoundError(f"source archive does not exist: {source_archive}")
        artifacts.append(_artifact(source_archive, artifact_dir, f"{run_id}-result-archive", kind="mla.result_archive", media_type="application/zip"))
    # These manifests are deliberately small and describe the fixed ABI.  Raw
    # tensors stay out of JSON records; users can attach them as separate blobs.
    input_manifest = json.dumps({"interface": interface, "inputs": problem["input_signature"], "generation": problem["input_generation"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output_manifest = json.dumps({"interface": interface, "outputs": problem["output_signature"], "semantics": problem["length_semantics"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not summary_only:
        artifacts.append(_generated_artifact(input_manifest, artifact_dir, f"{run_id}-input-manifest.json", kind="mla.input_manifest", media_type="application/json"))
        artifacts.append(_generated_artifact(output_manifest, artifact_dir, f"{run_id}-output-manifest.json", kind="mla.output_manifest", media_type="application/json"))
    if samples_us and not summary_only:
        sample_bytes = json.dumps({"samples": samples_us, "unit": "microseconds"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sample_path = artifact_dir / f"{run_id}-samples.json"
        sample_path.write_bytes(sample_bytes)
        artifacts.append({
            "artifact_id": f"{run_id}-samples",
            "kind": "mla.timing_samples",
            "uri": sample_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(sample_bytes),
            "size_bytes": len(sample_bytes),
            "media_type": "application/json",
            "retention": "permanent",
            "availability": "present",
        })
    perf_ref = next((a["artifact_id"] for a in artifacts if a["kind"] == "mla.performance_csv"), None)
    acc_ref = next((a["artifact_id"] for a in artifacts if a["kind"] == "mla.accuracy_csv"), None)
    samples_ref = next((a["artifact_id"] for a in artifacts if a["kind"] == "mla.timing_samples"), None)

    accuracy_rows = _read_csv(accuracy_path) if accuracy_path.is_file() else []
    own_accuracy = [r for r in accuracy_rows if r.get("name") == candidate_name]
    passed = [r for r in own_accuracy if str(r.get("status", "")).upper() == "PASS"]
    errors_abs = [_number(r.get("max_abs")) for r in own_accuracy]
    errors_rel = [_number(r.get("relative_l2")) for r in own_accuracy]
    errors_abs = [float(v) for v in errors_abs if v is not None]
    errors_rel = [float(v) for v in errors_rel if v is not None]
    c_status = "pass" if own_accuracy and len(passed) == len(own_accuracy) else ("fail" if own_accuracy else "not_run")
    correctness = {
        "status": c_status,
        "cases_total": len(own_accuracy),
        "cases_passed": len(passed),
        "max_abs_error": max(errors_abs) if errors_abs else None,
        "max_rel_error": max(errors_rel) if errors_rel else None,
        "report_artifact_ref": acc_ref if own_accuracy else None,
    }
    perf_rows = _read_csv(performance_path) if performance_path.is_file() else []
    perf = next((row for row in perf_rows if row.get("name") == candidate_name), {})
    measured = (
        str(candidate.get("status", "")).upper() == "MEASURED"
        and str(candidate.get("compile_status", "")).upper() == "PASS"
        and bool(samples_us)
        and _number(perf.get("device_median_us")) is not None
    )
    if summary_only and measured:
        if not all(math.isfinite(value) and value > 0 for value in samples_us):
            raise ValueError("device samples must be finite and positive")
        if not math.isclose(_quantile(samples_us, 0.5), float(perf["device_median_us"]), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("performance.csv median does not match the recorded device samples")
    execution_status = "succeeded" if measured else ("compile_error" if str(candidate.get("compile_status", "")).upper() not in ("PASS", "") else "runtime_error")
    if execution_status != "succeeded":
        correctness = {"status": "not_run", "cases_total": 0, "cases_passed": 0, "max_abs_error": None, "max_rel_error": None, "report_artifact_ref": None}
    timing = {
        "status": "recorded" if measured else "not_run",
        "sample_count": len(samples_us) if measured else 0,
        "median_us": _number(perf.get("device_median_us")) if measured else None,
        "p90_us": (_quantile(samples_us, 0.9) if measured else None),
        "samples_artifact_ref": samples_ref if measured else None,
    }
    metrics: list[dict[str, Any]] = []
    unit_map = {"_us": "microseconds", "_gbps": "GB/s", "_percent": "percent", "_bytes": "bytes", "_per_second": "per_second", "_seconds": "seconds"}
    for key, raw in perf.items():
        if key in {"name", "status", "timing_boundary", "compile_status", "correctness_status", "device_status", "last_phase", "last_gate", "failure_reason", "breakdown_status", "eligible", "hbm_traffic_status"}:
            continue
        value = _number(raw)
        if value is None or (perf_ref is None and not summary_only):
            continue
        unit = next((u for suffix, u in unit_map.items() if key.endswith(suffix)), "scalar")
        kind = "measurement" if key in {"device_median_us", "device_p10_us", "device_p90_us", "custom_call_median_us", "outside_custom_call_median_us"} else "derived"
        metrics.append({"name": key, "status": "observed", "value": value, "unit": unit, "kind": kind, "scope": "candidate", "source_artifact_ref": perf_ref, "parser_id": "project2-results-v1", "parser_version": "1", "definition": f"project2 performance.csv field {key}"})

    bundle = {
        "bundle_version": SCHEMA_VERSION,
        "is_fixture": False,
        "description": f"Imported project2 MLA decode result: {source_dir.name}, candidate {candidate_name}; historical evidence",
        "records": [
            _record("kernel", f"kernel-{KERNEL_ID}", created_at, {"kernel_id": KERNEL_ID, "display_name": "MLA decode", "adapter_id": "project2-mla-v1", "contract_notes": CONTRACT_NOTES}),
            _record("algorithm", algorithm_ref, created_at, {"kernel_id": KERNEL_ID, "algorithm_id": algorithm_id, "display_name": algorithm_id, "method_summary": METHOD_SUMMARIES[interface], "summary_author": "agent", "tags": ["mla", "project2", interface]}),
            _record("config", config_ref, created_at, {"kernel_id": KERNEL_ID, "algorithm_ref": algorithm_ref, "config_id": config_ref[4:], "problem_schema_id": SCHEMA_ID, "problem_schema_digest": schema_digest, "problem": problem, "config_hash": config_hash, "tags": ["mla", "project2", interface, "imported"]}),
            _record("baseline", baseline_ref, created_at, {"config_ref": config_ref, "baseline_id": baseline_id, "description": f"Project2 measured candidate {candidate_name} retained as imported reference.", "repo_uid": REPO_UID, "commit_oid": oid, "entrypoint": entrypoint, "role": "both"}),
            _record("run", run_id, created_at, {"subject_ref": baseline_ref, "config_ref": config_ref, "request_id": run_id, "attempt_no": 1, "rerun_of": None, "stage": "benchmark", "provenance": "imported_unverified", "source": source, "environment": environment, "protocol": protocol, "verifier": verifier, "execution_status": execution_status, "failure_reason": None if measured else str(candidate.get("failure_reason") or "project2 result was not a completed measured run"), "correctness": correctness, "timing": timing, "analysis_metrics": metrics, "analysis_conclusion": "Imported from project2 results.json/performance.csv; provenance is imported_unverified until the source snapshot and execution are reproduced.", "artifacts": artifacts, "comparison_key": comparison_key, "session_id": f"project2-{_sha256(results_path.read_bytes())[7:19]}", "pair_id": None, "role_in_pair": None}),
        ],
    }
    if summary_only:
        run_payload = bundle["records"][-1]["payload"]
        run_payload["evidence_mode"] = "summary_only"
        run_payload["analysis_conclusion"] = (
            "Historical result retained without evidence attachments; values have not "
            "been independently reproduced and raw samples are not retained."
        )
    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bundle_path


def import_rerun(
    source_dir: Path,
    store_root: Path,
    *,
    candidate_name: str | None = None,
    source_archive: Path | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    """Publish a result through the normal importer, keeping only the store."""
    from kernel_memory.services.importer import import_bundle
    from kernel_memory.services.trajectory import rebuild_kernel_trajectory
    from kernel_memory.storage import MemoryStore

    with tempfile.TemporaryDirectory(prefix="kmem-project2-") as staging:
        bundle_path = convert_rerun(
            source_dir, Path(staging), candidate_name=candidate_name,
            source_archive=source_archive, summary_only=summary_only,
        )
        store = MemoryStore.init(store_root)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        # A second shape shares the kernel/algorithm but has a different
        # measurement timestamp. Preserve their first publication timestamps;
        # only identical payloads are reused, so real changes still conflict.
        for index, record in enumerate(bundle["records"]):
            if record["record_type"] not in {"kernel", "algorithm"}:
                continue
            existing = store.get(record["record_id"])
            if existing is not None and existing.to_dict()["payload"] == record["payload"]:
                bundle["records"][index] = existing.to_dict()
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        report = import_bundle(store, bundle_path, artifact_root=Path(staging))
        view = rebuild_kernel_trajectory(store, KERNEL_ID)
        store.rebuild_index()
        return {"root": str(store.root), "records_published": report.records_published,
                "records_idempotent": report.records_idempotent,
                "artifacts_stored": report.artifacts_stored,
                "summary_only": summary_only, "trajectory": view["path"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one project2 result/rerun directory containing results.json")
    parser.add_argument("output", type=Path, help="output directory (Memory root with --store; otherwise bundle directory)")
    parser.add_argument("--candidate", help="candidate name; defaults to results.json last_candidate")
    parser.add_argument("--archive", type=Path, help="optional original project2 ZIP to retain as an evidence artifact")
    parser.add_argument("--summary-only", action="store_true", help="retain historical results without evidence files or artifact references")
    parser.add_argument("--store", action="store_true", help="publish into a standard Memory directory, without retaining the intermediate bundle")
    args = parser.parse_args(argv)
    if args.summary_only and args.archive:
        parser.error("--summary-only cannot be combined with --archive")
    if args.store:
        result = import_rerun(args.source, args.output, candidate_name=args.candidate,
                             source_archive=args.archive, summary_only=args.summary_only)
        print(json.dumps(result, sort_keys=True))
    else:
        path = convert_rerun(args.source, args.output, candidate_name=args.candidate,
                             source_archive=args.archive, summary_only=args.summary_only)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
