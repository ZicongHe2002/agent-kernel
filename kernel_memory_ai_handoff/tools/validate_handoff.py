#!/usr/bin/env python3
"""Validate the documentation handoff and its synthetic fixtures.

This is NOT the Kernel Memory production implementation, a TPU test, or a
complete RFC 8785/JCS implementation. Golden hash inputs deliberately use an
ASCII/integer subset for which the serializer below matches JCS.
Requires Python 3.11+ and jsonschema. Does not make network requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: jsonschema. Install it in an authorized project "
        "virtual environment, then rerun. No validation has been performed."
    ) from exc


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not permitted: {value}")


def load_json(path: Path) -> Any:
    if path.stat().st_size > 10_000_000:
        raise ValueError(f"Input exceeds handoff-checker size limit: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def fixture_canonical_bytes(value: Any) -> bytes:
    """Serialize only this package's constrained golden-vector domain."""
    def check(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str) or not key.isascii():
                    raise ValueError("Golden hash key is outside the ASCII subset")
                check(item)
        elif isinstance(node, list):
            for item in node:
                check(item)
        elif isinstance(node, str):
            if not node.isascii():
                raise ValueError("Golden hash string is outside the ASCII subset")
        elif node is None or type(node) in (bool, int):
            return
        else:
            raise ValueError(
                "Hash input is outside the fixture domain. Use a complete JCS "
                "library for production floats and Unicode."
            )
    check(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fixture_hash(value: Any) -> str:
    return digest(fixture_canonical_bytes(value))


def without(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if k != key}


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("Invalid quantile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    schema = load_json(root / "contracts/record.schema.json")
    problem_schema = load_json(root / "contracts/demo_problem.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(problem_schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    problem_validator = Draft202012Validator(problem_schema)
    bundle = load_json(root / "examples/demo_bundle.json")
    ensure(bundle["is_fixture"] is True, "Bundle must remain explicitly synthetic")
    by_id: dict[str, dict[str, Any]] = {}
    for record in bundle["records"]:
        validator.validate(record)
        rid = record["record_id"]
        ensure(rid not in by_id, f"Duplicate record ID: {rid}")
        by_id[rid] = record

    def get(rid: str, expected: str | tuple[str, ...] | None = None) -> dict[str, Any]:
        ensure(rid in by_id, f"Missing reference: {rid}")
        record = by_id[rid]
        if expected is not None:
            accepted = (expected,) if isinstance(expected, str) else expected
            ensure(record["record_type"] in accepted, f"Wrong type for {rid}: {record['record_type']}")
        return record["payload"]

    artifacts_checked = 0
    runs_checked = 0
    summary_checks = 0
    artifact_catalog: dict[str, dict[str, Any]] = {}
    for record in by_id.values():
        if record["record_type"] == "run":
            for item in record["payload"]["artifacts"]:
                aid = item["artifact_id"]
                if aid in artifact_catalog:
                    ensure(artifact_catalog[aid] == item, f"Conflicting artifact {aid}")
                artifact_catalog[aid] = item
    for record in by_id.values():
        kind, rid, p = record["record_type"], record["record_id"], record["payload"]
        if kind == "config":
            problem_validator.validate(p["problem"])
            ensure(p["problem_schema_digest"] == fixture_hash(problem_schema), "Problem schema digest mismatch")
            identity = {
                "hash_version": "jcs-sha256-v1", "kernel_id": p["kernel_id"],
                "problem_schema_id": p["problem_schema_id"],
                "problem_schema_digest": p["problem_schema_digest"], "problem": p["problem"],
            }
            ensure(p["config_hash"] == fixture_hash(identity), f"Config hash mismatch: {rid}")
        elif kind == "pr":
            get(p["config_ref"], "config")
            if p["origin_ref"] is not None:
                get(p["origin_ref"], ("commit", "baseline"))
            ensure((p["provider"] == "github") == (p["number"] is not None), "Provider/PR-number mismatch")
        elif kind == "pr_snapshot":
            get(p["pr_ref"], "pr")
            if p["previous_snapshot_ref"] is not None:
                get(p["previous_snapshot_ref"], "pr_snapshot")
            for cid in p["commit_refs"]:
                commit = get(cid, "commit")
                ensure(commit["pr_ref"] == p["pr_ref"], f"Cross-PR membership binding: {cid}")
        elif kind == "commit":
            pr = get(p["pr_ref"], "pr")
            ensure(pr["repo_uid"] == p["repo_uid"], "Commit repository mismatch")
            if p["change_status"] == "not_extracted":
                ensure(not p["changes"], "Unextracted changes must be empty")
        elif kind == "baseline":
            get(p["config_ref"], "config")
        elif kind == "relation":
            get(p["config_ref"], "config")
            get(p["from_ref"])
            get(p["to_ref"])
            ensure(p["from_ref"] != p["to_ref"], "Self origin relation")
            for eid in p["evidence_refs"]:
                get(eid)
        elif kind == "run":
            runs_checked += 1
            cfg = get(p["config_ref"], "config")
            subject = get(p["subject_ref"], ("commit", "baseline"))
            ensure(p["provenance"] == "fixture", "Handoff must not claim a real trusted execution")
            ensure(subject["commit_oid"] == p["source"]["target_commit"], "Target source does not match subject")
            if "pr_ref" in subject:
                ensure(get(subject["pr_ref"], "pr")["config_ref"] == p["config_ref"], "Run/subject config mismatch")
            else:
                ensure(subject["config_ref"] == p["config_ref"], "Baseline/config mismatch")
            src = p["source"]
            if src["checkout_mode"] == "exact_commit":
                ensure(src["target_commit"] == src["tested_commit"], "Exact-commit source mismatch")
                ensure(not src["merge_parent_oids"], "Exact-commit run should not have integration parents")
            variant = {k: src[k] for k in ("source_digest", "entrypoint", "implementation_overrides", "checkout_mode")}
            ensure(src["variant_digest"] == fixture_hash(variant), "Variant hash mismatch")
            for field, hash_key in [("environment", "environment_hash"), ("protocol", "protocol_hash"), ("verifier", "verifier_hash")]:
                snapshot = p[field]
                ensure(snapshot[hash_key] == fixture_hash(without(snapshot, hash_key)), f"{field} hash mismatch")
            identity = {
                "config_hash": cfg["config_hash"],
                "environment_hash": p["environment"]["environment_hash"],
                "protocol_hash": p["protocol"]["protocol_hash"],
                "verifier_hash": p["verifier"]["verifier_hash"],
                "checkout_mode": src["checkout_mode"],
            }
            ensure(p["comparison_key"] == fixture_hash(identity), "Comparison hash mismatch")
            local_artifacts = {a["artifact_id"]: a for a in p["artifacts"]}
            for a in p["artifacts"]:
                path = (root / a["uri"]).resolve()
                ensure(path.is_relative_to(root), f"Unsafe artifact path: {a['uri']}")
                ensure(path.is_file(), f"Missing artifact: {a['uri']}")
                raw = path.read_bytes()
                ensure(digest(raw) == a["sha256"], "Artifact checksum mismatch")
                ensure(len(raw) == a["size_bytes"], "Artifact size mismatch")
                data = load_json(path)
                ensure(data.get("is_fixture") is True, "Artifact must remain synthetic")
                artifacts_checked += 1
            timing = p["timing"]
            if timing["status"] == "recorded":
                ensure(timing["samples_artifact_ref"] in local_artifacts, "Missing samples reference")
                a = local_artifacts[timing["samples_artifact_ref"]]
                data = load_json(root / a["uri"])
                ensure(data["unit"] == "microseconds", "Unknown sample unit")
                values = data["samples"]
                ensure(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), "Invalid samples")
                ensure(len(values) == timing["sample_count"] == p["protocol"]["repetitions"], "Sample count mismatch")
                ensure(math.isclose(statistics.median(values), timing["median_us"], rel_tol=1e-12), "Median mismatch")
                ensure(math.isclose(quantile(values, .9), timing["p90_us"], rel_tol=1e-12), "p90 mismatch")
                summary_checks += 2
            else:
                ensure(timing["sample_count"] == 0 and timing["median_us"] is None and timing["p90_us"] is None, "Unmeasured timing must be absent")
            corr = p["correctness"]
            ensure(corr["cases_passed"] <= corr["cases_total"], "Invalid correctness counts")
            if corr["status"] == "pass":
                ensure(corr["cases_total"] > 0 and corr["cases_total"] == corr["cases_passed"], "Invalid pass")
                ensure(corr["report_artifact_ref"] in local_artifacts, "Missing correctness evidence")
            if p["execution_status"] == "compile_error":
                ensure(corr["status"] == "not_run" and timing["status"] == "not_run", "Compile failure has fabricated results")
            for metric in p["analysis_metrics"]:
                if metric["status"] == "observed":
                    ensure(metric["value"] is not None and metric["source_artifact_ref"] in local_artifacts, "Observed metric needs evidence")
                else:
                    ensure(metric["value"] is None, "Unknown metric must be null")
        elif kind == "decision":
            get(p["config_ref"], "config")
            get(p["candidate_subject_ref"], "commit")
            ensure(p["policy_hash"] == fixture_hash(p["policy"]), "Policy hash mismatch")
            for rr in p["candidate_run_refs"] + p["baseline_run_refs"]:
                run = get(rr, "run")
                ensure(run["comparison_key"] == p["comparison_key"], "Decision evidence group mismatch")
            ensure(not p["is_production"], "Fixture must not be a production decision")
            ensure(p["outcome"] == "blocked" and "FIXTURE_NOT_ELIGIBLE" in p["reason_codes"], "Fixture promotion gate absent")
        elif kind == "annotation":
            get(p["target_ref"])
            for er in p["evidence_refs"]:
                get(er)

    # Check the explicit optimization-origin DAG.
    edges: dict[str, list[str]] = {}
    for r in by_id.values():
        if r["record_type"] == "relation" and r["payload"]["kind"] == "optimization_origin":
            p = r["payload"]
            edges.setdefault(p["from_ref"], []).append(p["to_ref"])
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(node: str) -> None:
        ensure(node not in visiting, "Optimization-origin cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in edges.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)
    for node in edges:
        visit(node)

    vectors = load_json(root / "contracts/hash_vectors.json")
    for vector in vectors["vectors"]:
        ensure(fixture_hash(vector["payload"]) == vector["digest"], f"Bad golden vector {vector['name']}")
    candidate = get("run-demo-a", "run")["timing"]["median_us"]
    baseline = get("run-demo-baseline", "run")["timing"]["median_us"]
    speedup = baseline / candidate
    reduction = 100 * (1 - candidate / baseline)
    ensure(math.isclose(reduction, 10.0), "Latency reduction formula mismatch")
    ensure(not any(r["record_type"] == "run" and r["payload"]["subject_ref"] == "commit-demo-b" for r in by_id.values()), "Untested commit gained a Run")
    ensure(get("pr-demo-102", "pr")["origin_ref"] == "commit-demo-a", "Wrong branch origin")
    ensure(get("run-demo-c", "run")["execution_status"] == "succeeded", "Spill incorrectly classified as execution failure")
    draft = load_json(root / "examples/mla_config.draft.json")
    ensure(not validator.is_valid(draft), "Incomplete MLA draft accidentally accepted as a record")
    return {
        "status": "passed",
        "scope": "handoff schemas and synthetic fixtures only",
        "production_system_implemented": False,
        "real_cpu_benchmark_executed": False,
        "tpu_benchmark_executed": False,
        "github_live_collection_executed": False,
        "llo_real_parser_validated": False,
        "schema_records_validated": len(by_id),
        "runs_validated": runs_checked,
        "artifact_checks": artifacts_checked,
        "latency_summary_checks": summary_checks,
        "golden_hash_vectors": len(vectors["vectors"]),
        "illustrative_speedup": speedup,
        "illustrative_latency_reduction_pct": reduction,
        "invalid_mla_draft_correctly_rejected": True,
        "all_runs_are_explicit_fixtures": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    try:
        report = validate(args.root)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
