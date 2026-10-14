"""Export the compact, readable Memory directory from an authoritative store.

This output is a separate versioned document format. It is deliberately not a
``MemoryStore``: it contains no artifact blobs, requests, journals or cache files.
All runs remain separate, and baseline measurements stay within their shape.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from ..domain.errors import InputError
from ..domain.hashing import jcs_digest, sha256_bytes
from ..domain.ids import resolve_inside, slug_for_id
from ..domain.jsonio import load_json_file
from ..domain.models import Record, to_json
from ..storage import MemoryStore

OUTPUT_FORMAT = "kernel-memory-readable"
OUTPUT_VERSION = "2.0.0"
# Presentation changes must not rename an unchanged input shape.
SHAPE_IDENTITY_VERSION = "1.0.0"
DECODE_IMPLEMENTATION_KEYS = frozenset({"block_size", "block_kv", "pages_per_block", "decode_batch_size", "two_step", "transpose_cache"})

# Output keys are stable even when a backend did not collect a metric. Aliases
# preserve historical project2 names without conflating logical I/O with HBM.
METRIC_SLOTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "mfu": ("percent", ("effective_mfu_percent", "device_useful_mfu_percent", "mfu_percent")),
    "mfu_executed_estimate": ("percent", ("device_executed_mfu_percent_estimate",)),
    "mxu": ("percent", ("runtime_mxu_utilization_percent", "mxu_utilization_percent", "mxu_utilization", "mxu_percent")),
    "attention_mxu_static": ("percent", ("attention_mxu_static_utilization_percent",)),
    "roofline_efficiency": ("percent", ("boundary_roofline_percent", "roofline_efficiency_percent", "roofline_percent")),
    "logical_bandwidth_gbps": ("GB/s", ("logical_bandwidth_gbps",)),
    "logical_bandwidth_utilization": ("percent", ("logical_bandwidth_percent",)),
    "hbm_bandwidth_gbps": ("GB/s", ("measured_hbm_bandwidth_gbps", "hbm_bandwidth_gbps")),
    "hbm_bandwidth_utilization": ("percent", ("measured_hbm_bandwidth_percent", "hbm_bandwidth_utilization_percent", "hbm_bandwidth_percent")),
}


def _document(kind: str, **values: Any) -> dict[str, Any]:
    return {"schema_version": OUTPUT_VERSION, "document_type": kind, **values}


def round_output_value(value: float | None, digits: int = 3) -> float | None:
    """Shorten display values without turning a positive measurement into zero."""
    if value is None:
        return None
    rounded = round(value, digits)
    return value if value > 0 and rounded == 0 else rounded


def _slot(run: Record, name: str) -> float | None:
    unit, aliases = METRIC_SLOTS[name]
    matches = [m for alias in aliases for m in run.payload.analysis_metrics if m.name == alias]
    # Never pick one of several scopes as if it were a unique observation.
    if len(matches) > 1:
        matches = [m for m in matches if m.name == matches[0].name]
    if len(matches) != 1:
        return None
    metric = matches[0]
    value = metric.value
    if (metric.status != "observed" or metric.unit != unit
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        return None
    # One convention for every utilization: 1 = 100%. Do not clamp estimates
    # above peak or replace unavailable values with zero.
    return round_output_value(value / 100, 4) if unit == "percent" else round_output_value(value)


def _ir_status(run: Record, level: str) -> str:
    def matches(kind: str) -> bool:
        pieces = kind.lower().replace(".", "_").replace("-", "_").split("_")
        return level in pieces or (level == "hlo" and "stablehlo" in pieces)

    refs = [a for a in run.payload.artifacts if matches(a.kind)]
    available = [a for a in refs if a.availability == "present"]
    evidence = run.payload.source.implementation_overrides.get("compiler_evidence", {})
    summary = evidence.get(level, {}) if isinstance(evidence, dict) else {}
    summarized = isinstance(summary, dict) and summary.get("status") == "recorded"
    return "recorded" if available or summarized else ("unavailable" if refs else "not_collected")


def _result(run: Record) -> dict[str, Any]:
    p = run.payload
    candidate = p.source.implementation_overrides.get("candidate")
    return {
        "candidate": candidate if isinstance(candidate, str) else None,
        "run_ref": run.record_id,
        "source_config_ref": p.config_ref,
        "subject_ref": p.subject_ref,
        "provenance": p.provenance,
        "execution_status": p.execution_status,
        "failure_reason": p.failure_reason,
        "correctness": p.correctness.status,
        "latency_us": round_output_value(p.timing.median_us),
        "p90_latency_us": round_output_value(p.timing.p90_us),
        "samples": p.timing.sample_count,
        **{name: _slot(run, name) for name in METRIC_SLOTS},
        "hlo": _ir_status(run, "hlo"),
        "llo": _ir_status(run, "llo"),
        "comparison_key": p.comparison_key,
    }


def _shape_problem(config: Record) -> dict[str, Any]:
    problem = to_json(config.payload.problem)
    if config.payload.kernel_id == "mla_decode":
        problem = {key: value for key, value in problem.items() if key not in DECODE_IMPLEMENTATION_KEYS}
    return problem


def _shape_hash(config: Record) -> str:
    return jcs_digest({"output_shape_version": SHAPE_IDENTITY_VERSION, "kernel_id": config.payload.kernel_id, "problem": _shape_problem(config)})


def _shape_name(config: Record) -> str:
    p = config.payload.problem
    digest = _shape_hash(config).split(":", 1)[-1][:12]
    if all(key in p for key in ("batch", "num_q_heads", "capacity")):
        return f"shape-b{p['batch']}-h{p['num_q_heads']}-n{p['capacity']}-{digest}"
    if all(key in p for key in ("batch", "num_q_heads", "seq_len")):
        return f"shape-b{p['batch']}-h{p['num_q_heads']}-s{p['seq_len']}-{digest}"
    return f"shape-{slug_for_id(config.payload.config_id)[:70]}-{digest}"


def _baseline_runs(baseline: Record, runs: list[Record], by_id: dict[str, Record]) -> list[Record]:
    """Link each measurement once, including explicit baseline commit bindings."""
    out: list[Record] = []
    p = baseline.payload
    for run in runs:
        if run.payload.config_ref != p.config_ref:
            continue
        subject = by_id.get(run.payload.subject_ref)
        direct = run.payload.subject_ref == baseline.record_id
        overrides = run.payload.source.implementation_overrides
        explicit = overrides.get("is_declared_baseline") is True or (
            isinstance(overrides.get("candidate"), str)
            and p.entrypoint.endswith("#candidate=" + overrides["candidate"])
        )
        matched = (subject is not None and subject.record_type == "commit"
                   and subject.payload.repo_uid == p.repo_uid
                   and subject.payload.commit_oid == p.commit_oid
                   and run.payload.source.entrypoint == p.entrypoint and explicit)
        if direct or matched:
            out.append(run)
    return out


def _pr_identity(pr: Record | None) -> dict[str, Any] | None:
    if pr is None:
        return None
    return {"pr_ref": pr.record_id, "provider": pr.payload.provider,
            "number": pr.payload.number, "pr_key": pr.payload.pr_key,
            "repo_uid": pr.payload.repo_uid}


def _source_identity_kind(runs: list[Record]) -> str | None:
    kinds = {kind for r in runs
             if isinstance(kind := r.payload.source.implementation_overrides.get("source_identity_kind"), str)}
    return next(iter(kinds)) if len(kinds) == 1 else None


def build_output_memory(store: MemoryStore) -> dict[str, dict[str, Any]]:
    """Return relative POSIX filenames mapped to deterministic output documents."""
    records = sorted(store.records(), key=lambda r: (r.created_at, r.record_id))
    by_id = {record.record_id: record for record in records}
    docs: dict[str, dict[str, Any]] = {}
    algorithms = [r for r in records if r.record_type == "algorithm"]
    all_runs = [r for r in records if r.record_type == "run"]
    all_baselines = [r for r in records if r.record_type == "baseline"]
    for kernel in (r for r in records if r.record_type == "kernel"):
        docs[f"{slug_for_id(kernel.payload.kernel_id)}/kernel.json"] = _document(
            "kernel", kernel_ref=kernel.record_id, **to_json(kernel.payload),
            algorithms=[{"algorithm_ref": r.record_id, "algorithm_id": r.payload.algorithm_id}
                        for r in algorithms if r.payload.kernel_id == kernel.payload.kernel_id],
        )
    for algorithm in algorithms:
        a = algorithm.payload
        directory = f"{slug_for_id(a.kernel_id)}/{slug_for_id(a.algorithm_id)}"
        configs = sorted((r for r in records if r.record_type == "config" and r.payload.algorithm_ref == algorithm.record_id), key=lambda r: r.record_id)
        shape_groups: dict[str, list[Record]] = {}
        for config in configs:
            shape_groups.setdefault(_shape_hash(config), []).append(config)
        groups: list[dict[str, Any]] = []
        docs[f"{directory}/method_summary.json"] = _document(
            "method_summary", kernel_id=a.kernel_id, algorithm_ref=algorithm.record_id,
            algorithm_id=a.algorithm_id, display_name=a.display_name, method_summary=a.method_summary,
            author_kind=a.summary_author, tags=sorted(a.tags),
            annotations=[r.to_dict() for r in records if r.record_type == "annotation" and r.payload.target_ref == algorithm.record_id],
        )
        for shape_hash, grouped_configs in sorted(shape_groups.items()):
            config = grouped_configs[0]
            config_ids = {r.record_id for r in grouped_configs}
            identity = {"shape_hash": shape_hash, "source_config_refs": sorted(config_ids)}
            shape = f"{directory}/{_shape_name(config)}"
            cfg_path = f"{shape}/config.json"
            if cfg_path in docs:
                raise InputError(f"two configs map to {cfg_path}", code="OUTPUT_PATH_CONFLICT")
            docs[cfg_path] = _document(
                "config", **identity,
                kernel_id=a.kernel_id, algorithm_ref=algorithm.record_id,
                source_configs=[{"config_ref": r.record_id, "config_hash": r.payload.config_hash,
                                 "problem_schema_id": r.payload.problem_schema_id,
                                 "problem_schema_digest": r.payload.problem_schema_digest} for r in grouped_configs],
                problem=_shape_problem(config), tags=sorted({tag for r in grouped_configs for tag in r.payload.tags}),
            )
            runs = [r for r in records if r.record_type == "run" and r.payload.config_ref in config_ids]
            baselines = [r for r in records if r.record_type == "baseline" and r.payload.config_ref in config_ids]
            prs = [r for r in records if r.record_type == "pr" and r.payload.config_ref in config_ids]
            pr_ids = {r.record_id for r in prs}
            commits = [r for r in records if r.record_type == "commit" and r.payload.pr_ref in pr_ids]
            subjects = sorted(baselines + commits, key=lambda r: (r.created_at, r.record_id))
            shared_baselines = [baseline for baseline in all_baselines
                                if baseline.payload.config_ref not in config_ids
                                and _shape_hash(by_id[baseline.payload.config_ref]) == shape_hash
                                and by_id[baseline.payload.config_ref].payload.algorithm_ref != algorithm.record_id]
            groups.append({
                **identity,
                "shape": _shape_name(config),
                "baselines": [{
                    "baseline_ref": baseline.record_id,
                    **to_json(baseline.payload),
                    "runs": [_result(run) for run in _baseline_runs(baseline, runs, by_id)],
                } for baseline in baselines],
                "shared_baselines": [{
                    "baseline_ref": baseline.record_id,
                    "source_algorithm_ref": by_id[baseline.payload.config_ref].payload.algorithm_ref,
                    "source_config_ref": baseline.payload.config_ref,
                    "relationship": "same_fixed_input_shape_only",
                    "comparison_note": "Shared reference to the original measurement. Equal input shapes alone do not establish comparable protocol, verifier, or environment.",
                    "runs": [_result(run) for run in _baseline_runs(baseline, all_runs, by_id)],
                } for baseline in shared_baselines],
            })
            attempts: list[dict[str, Any]] = []
            code: list[dict[str, Any]] = []
            for subject in subjects:
                own_runs = [r for r in runs if r.payload.subject_ref == subject.record_id]
                pr = by_id.get(subject.payload.pr_ref) if subject.record_type == "commit" else None
                # A declared baseline may point to an already represented local
                # commit run. It remains in baseline.json, not a duplicate attempt.
                if subject.record_type == "baseline" and not own_runs and _baseline_runs(subject, runs, by_id):
                    continue
                subject_config = pr.payload.config_ref if pr else subject.payload.config_ref
                attempts.append({
                    "subject_ref": subject.record_id, "subject_type": subject.record_type,
                    "source_config_ref": subject_config,
                    "created_at": subject.created_at,
                    "title": pr.payload.title if pr else subject.payload.description,
                    "hypothesis": pr.payload.hypothesis if pr else None,
                    "context_ref": pr.record_id if pr else None,
                    "pr": _pr_identity(pr),
                    "repo_uid": subject.payload.repo_uid,
                    "commit_oid": to_json(subject.payload.commit_oid),
                    "source_identity_kind": _source_identity_kind(own_runs),
                    "origin_ref": pr.payload.origin_ref if pr else None,
                    "status": "recorded" if own_runs else "not_run",
                    "summary": subject.payload.summary if subject.record_type == "commit" else subject.payload.description,
                    "changes": to_json(subject.payload.changes) if subject.record_type == "commit" else [],
                    "runs": [{"run_ref": r.record_id, "request_id": r.payload.request_id,
                              "attempt_no": r.payload.attempt_no, "rerun_of": r.payload.rerun_of,
                              "stage": r.payload.stage, "session_id": r.payload.session_id,
                              "pair_id": r.payload.pair_id, "role_in_pair": r.payload.role_in_pair,
                              "source_identity_kind": r.payload.source.implementation_overrides.get("source_identity_kind"),
                              "implementation_overrides": to_json(r.payload.source.implementation_overrides)} for r in own_runs],
                })
                for run in own_runs:
                    code.append({"run_ref": run.record_id, "subject_ref": subject.record_id,
                                 "source_config_ref": run.payload.config_ref,
                                 "config_implementation_parameters": {key: value for key, value in by_id[run.payload.config_ref].payload.problem.items() if a.kernel_id == "mla_decode" and key in DECODE_IMPLEMENTATION_KEYS},
                                 "source": to_json(run.payload.source), "content": None,
                                 "content_status": "not_retained_in_compact_output"})
                if not own_runs and subject.record_type == "commit" and subject.payload.source_available:
                    code.append({"run_ref": None, "subject_ref": subject.record_id,
                                 "source_config_ref": subject_config,
                                 "config_implementation_parameters": {},
                                 "source": {"repo_uid": subject.payload.repo_uid, "commit_oid": to_json(subject.payload.commit_oid)},
                                 "content": None, "content_status": "not_retained_in_compact_output"})
            # Preserve local contexts even if no commit or run was produced.
            for pr in prs:
                if not any(commit.payload.pr_ref == pr.record_id for commit in commits):
                    attempts.append({"subject_ref": None, "subject_type": "pr", "created_at": pr.created_at,
                                     "source_config_ref": pr.payload.config_ref,
                                     "title": pr.payload.title, "hypothesis": pr.payload.hypothesis,
                                     "context_ref": pr.record_id, "origin_ref": pr.payload.origin_ref,
                                     "pr": _pr_identity(pr), "repo_uid": pr.payload.repo_uid,
                                     "commit_oid": None, "source_identity_kind": None,
                                     "status": "not_run", "summary": None, "changes": [], "runs": []})
            docs[f"{shape}/trajectory/attempt.json"] = _document("attempt", **identity, attempts=attempts)
            docs[f"{shape}/trajectory/result.json"] = _document(
                "result", **identity,
                runs=[_result(run) for run in runs],
                decisions=[r.to_dict() for r in records if r.record_type == "decision" and r.payload.config_ref in config_ids],
                annotations=[r.to_dict() for r in records if r.record_type == "annotation" and r.payload.target_ref in {*config_ids, *(r.record_id for r in subjects), *(r.record_id for r in runs)}],
            )
            if code:
                docs[f"{shape}/trajectory/code.json"] = _document("code", **identity, entries=code)
        docs[f"{directory}/baseline.json"] = _document(
            "baseline", kernel_id=a.kernel_id, algorithm_ref=algorithm.record_id,
            selection="declared_baselines_per_shape", shapes=groups,
        )
    return dict(sorted(docs.items()))


def export_output_memory(store: MemoryStore, output_root: Path) -> dict[str, Any]:
    """Write the compact output, refusing unrelated destination files.

    Repeating an export produces identical bytes. An existing readable export may
    be refreshed; only its manifest-listed files are replaced or removed. The
    source store, including its artifacts and journal, is never edited.
    """
    root = Path(output_root).absolute()
    if root.is_symlink():
        raise InputError("output root must not be a symlink", code="UNSAFE_OUTPUT_PATH")
    source_root = Path(store.root).resolve()
    resolved = root.resolve()
    if resolved == source_root or resolved in source_root.parents or source_root in resolved.parents:
        raise InputError("output root must be separate from the authoritative store", code="OUTPUT_OVERLAPS_STORE")
    return export_output_documents(build_output_memory(store), root)


def export_output_documents(documents: dict[str, dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Write validated readable documents using the shared manifest and path rules."""
    root = Path(output_root).absolute()
    if root.is_symlink():
        raise InputError("output root must not be a symlink", code="UNSAFE_OUTPUT_PATH")
    for relative in documents:
        resolve_inside(root, relative)
    encoded = {path: (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for path, doc in documents.items()}
    files = {path: sha256_bytes(data) for path, data in encoded.items()}
    manifest = {
        "format": OUTPUT_FORMAT, "schema_version": OUTPUT_VERSION,
        "storage_kind": "compact_readable_export", "artifact_blobs_included": False,
        "result_conventions": {
            "ratios": "1 = 100%; rounded to 4 decimal places; estimates above 1 are preserved.",
            "ratio_fields": [name for name, (unit, _) in METRIC_SLOTS.items() if unit == "percent"],
            "latency_us": "Median latency in microseconds; rounded to 3 decimal places.",
            "p90_latency_us": "90th percentile latency in microseconds; rounded to 3 decimal places.",
            "bandwidth_gbps": "GB/s (gigabytes, not gigabits); rounded to 3 decimal places.",
            "mfu": "Useful FLOPs / (peak FLOPs per second * elapsed seconds).",
            "mfu_executed_estimate": "Estimated executed FLOPs / (peak FLOPs per second * elapsed seconds).",
            "mxu": "Measured runtime MXU utilization.",
            "attention_mxu_static": "Compiler estimate for attention MXU utilization.",
            "roofline_efficiency": "Achieved fraction of the declared boundary roofline model.",
            "logical_bandwidth": "Calculated from a logical byte model; hbm_bandwidth uses measured HBM traffic.",
            "null": "Unavailable, unsupported, invalid, or ambiguous measurement; never treated as zero.",
            "hlo_llo": "Status in source inputs; compiler files are not included in this output.",
            "small_positive_values": "Preserve original precision if rounding would turn a positive measurement into zero.",
        },
        "content_hash": jcs_digest(files), "files": files,
        "counts": {"kernels": sum(p.endswith("/kernel.json") for p in files),
                   "algorithms": sum(p.endswith("/method_summary.json") for p in files),
                   "shapes": sum(p.endswith("/config.json") for p in files),
                   "runs": sum(len(d["runs"]) for d in documents.values() if d["document_type"] == "result")},
    }
    old_files: set[str] = set()
    if root.exists() and any(root.iterdir()):
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise InputError("output destination is not a readable Memory export", code="OUTPUT_NOT_EMPTY")
        previous = load_json_file(manifest_path)
        if previous.get("format") != OUTPUT_FORMAT or not isinstance(previous.get("files"), dict):
            raise InputError("refusing to overwrite an unrelated output directory", code="OUTPUT_NOT_EMPTY")
        old_files = set(previous["files"])
        for path in old_files:
            resolve_inside(root, path)
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if actual - old_files - {"manifest.json"}:
            raise InputError("output directory contains files outside its export manifest", code="OUTPUT_HAS_FOREIGN_FILES")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise InputError("output contains a symlink", code="UNSAFE_OUTPUT_PATH")
    root.mkdir(parents=True, exist_ok=True)
    encoded["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    for relative, data in encoded.items():
        path = resolve_inside(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_bytes() == data:
            continue
        staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            staged.write_bytes(data)
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
    for relative in sorted(old_files - set(files)):
        resolve_inside(root, relative).unlink(missing_ok=True)
    # Remove only empty obsolete folders, deepest first; never recursive-delete.
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    return {"output_root": str(root), **manifest}


def verify_output_memory(output_root: Path) -> dict[str, Any]:
    """Check export bytes, counts, shape boundaries and measurement references.

    This verifies integrity of the readable export; it does not independently
    reproduce imported measurements or promote their provenance.
    """
    root = Path(output_root).absolute()
    if root.is_symlink():
        raise InputError("output root must not be a symlink", code="UNSAFE_OUTPUT_PATH")
    manifest = load_json_file(resolve_inside(root, "manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("format") != OUTPUT_FORMAT or manifest.get("schema_version") != OUTPUT_VERSION or not isinstance(manifest.get("files"), dict):
        raise InputError("not a supported readable Memory export", code="INVALID_OUTPUT_MANIFEST")
    issues: list[dict[str, str]] = []

    def issue(code: str, path: str, message: str) -> None:
        issues.append({"code": code, "path": path, "message": message})

    files = manifest["files"]
    docs: dict[str, dict[str, Any]] = {}
    for relative, expected in files.items():
        path = resolve_inside(root, relative)
        if not path.is_file():
            issue("MISSING_FILE", relative, "manifest-listed file is missing")
            continue
        if sha256_bytes(path.read_bytes()) != expected:
            issue("HASH_MISMATCH", relative, "file bytes do not match the manifest")
        try:
            doc = load_json_file(path)
        except InputError as exc:
            issue("INVALID_DOCUMENT", relative, str(exc))
            continue
        if not isinstance(doc, dict) or doc.get("schema_version") != OUTPUT_VERSION:
            issue("INVALID_DOCUMENT", relative, "unsupported document shape or version")
            continue
        docs[relative] = doc
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    for relative in sorted(actual - set(files) - {"manifest.json"}):
        issue("FOREIGN_FILE", relative, "file is not listed in the export manifest")
    if jcs_digest(files) != manifest.get("content_hash"):
        issue("MANIFEST_HASH_MISMATCH", "manifest.json", "manifest content_hash does not match its file table")
    counts = {"kernels": sum(d.get("document_type") == "kernel" for d in docs.values()),
              "algorithms": sum(d.get("document_type") == "method_summary" for d in docs.values()),
              "shapes": sum(d.get("document_type") == "config" for d in docs.values()),
              "runs": sum(len(d.get("runs", [])) for d in docs.values() if d.get("document_type") == "result")}
    if counts != manifest.get("counts"):
        issue("COUNT_MISMATCH", "manifest.json", "manifest counts do not match its documents")
    configs = {ref: d for d in docs.values() if d.get("document_type") == "config" for ref in d.get("source_config_refs", [])}
    algorithms = {d.get("algorithm_ref") for d in docs.values() if d.get("document_type") == "method_summary"}
    runs: dict[str, dict[str, Any]] = {}
    for path, doc in docs.items():
        if doc.get("document_type") == "config":
            expected = jcs_digest({"output_shape_version": SHAPE_IDENTITY_VERSION, "kernel_id": doc.get("kernel_id"), "problem": doc.get("problem")})
            if expected != doc.get("shape_hash"):
                issue("SHAPE_HASH_MISMATCH", path, "fixed input problem does not match shape_hash")
            if doc.get("algorithm_ref") not in algorithms:
                issue("MISSING_ALGORITHM", path, "config references an absent algorithm")
        if doc.get("document_type") != "result":
            continue
        attempts = docs.get(path.replace("/result.json", "/attempt.json"), {})
        subject_refs = {a.get("subject_ref") for a in attempts.get("attempts", [])}
        for run in doc.get("runs", []):
            run_ref = run.get("run_ref")
            if run_ref in runs:
                issue("DUPLICATE_RUN", path, f"run {run_ref} appears in multiple result documents")
            runs[run_ref] = run
            if run.get("source_config_ref") not in doc.get("source_config_refs", []):
                issue("CROSS_SHAPE_RUN", path, f"run {run_ref} references another shape")
            if run.get("subject_ref") not in subject_refs:
                issue("MISSING_ATTEMPT", path, f"run {run_ref} has no matching attempt")
    baselines = {b.get("baseline_ref") for d in docs.values() if d.get("document_type") == "baseline" for s in d.get("shapes", []) for b in s.get("baselines", [])}
    for path, doc in docs.items():
        if doc.get("document_type") != "baseline":
            continue
        for shape in doc.get("shapes", []):
            for baseline in shape.get("baselines", []) + shape.get("shared_baselines", []):
                if baseline.get("baseline_ref") not in baselines:
                    issue("MISSING_BASELINE", path, "shared baseline has no original declaration")
                config = configs.get(baseline.get("source_config_ref", baseline.get("config_ref")))
                if config is None or config.get("shape_hash") != shape.get("shape_hash"):
                    issue("CROSS_SHAPE_BASELINE", path, "baseline does not belong to this fixed input shape")
                for run in baseline.get("runs", []):
                    if run != runs.get(run.get("run_ref")):
                        issue("BASELINE_RESULT_MISMATCH", path, "baseline evidence differs from the original result")
    return {"ok": not issues, "issues": issues, "issue_count": len(issues),
            "files_checked": len(docs), "content_hash": manifest.get("content_hash"), "counts": counts}


__all__ = ["OUTPUT_FORMAT", "OUTPUT_VERSION", "METRIC_SLOTS", "build_output_memory", "export_output_memory", "export_output_documents", "verify_output_memory"]
