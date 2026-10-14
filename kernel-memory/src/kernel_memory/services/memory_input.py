"""Validate portable result-only input and generate the readable Memory tree.

This ingress accepts summaries authored by people or data adapters, including AI.
It never executes source paths, reads compiler dumps, or invents Git identities.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from ..domain.errors import InputError, KernelMemoryError
from ..domain.hashing import jcs_digest
from ..domain.jsonio import load_json_file
from .output_memory import (
    METRIC_SLOTS, OUTPUT_VERSION, SHAPE_IDENTITY_VERSION,
    export_output_documents, verify_output_memory, round_output_value,
)

INPUT_FORMAT = "kernel-memory-input"
INPUT_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "contracts" / "memory_input.v1.schema.json"


@lru_cache(maxsize=1)
def memory_input_schema() -> dict[str, Any]:
    schema = load_json_file(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return schema


def _invalid(path: str, message: str) -> None:
    raise InputError(f"{path}: {message}", code="INVALID_MEMORY_INPUT")


def _check_json(value: Any, path: str = "$", depth: int = 0) -> None:
    """Check unrestricted operation parameters as well as structured fields."""
    if depth > 40:
        _invalid(path, "JSON nesting exceeds 40 levels")
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > 4000:
            _invalid(path, "text exceeds 4000 characters; provide summaries, not raw files")
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            _invalid(path, "numbers must be finite")
        if isinstance(value, int) and abs(value) > 2**53 - 1:
            _invalid(path, "integer exceeds the interoperable JSON exact-integer range")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _invalid(path, "JSON keys must be strings")
            _check_json(item, f"{path}.{key}", depth + 1)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]", depth + 1)
        return
    _invalid(path, "value is not JSON-compatible")


def _unique(items: list[dict[str, Any]], key: str, path: str) -> None:
    seen: set[str] = set()
    for item in items:
        ident = item[key]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", ident):
            _invalid(path, f"invalid {key}: {ident!r}")
        if ident in seen:
            _invalid(path, f"duplicate {key}: {ident}")
        if ident in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
            _invalid(path, f"{key} is a reserved filename on Windows: {ident}")
        seen.add(ident)


def validate_memory_input(data: Any) -> None:
    """Validate shape, units, scoped IDs and baseline/run consistency before writing."""
    _check_json(data)
    error = next(Draft202012Validator(memory_input_schema(), format_checker=FormatChecker()).iter_errors(data), None)
    if error is not None:
        _invalid("$/" + "/".join(map(str, error.absolute_path)), error.message)
    _unique(data["kernels"], "kernel_id", "kernels")
    for kernel in data["kernels"]:
        kp = kernel["kernel_id"]
        _unique(kernel["algorithms"], "algorithm_id", kp)
        for algorithm in kernel["algorithms"]:
            ap = kp + "/" + algorithm["algorithm_id"]
            if not any(c.isalnum() for c in algorithm["method_summary"]):
                _invalid(ap, "method_summary must explain the method, not just punctuation")
            _unique(algorithm["shapes"], "shape_id", ap)
            problem_hashes: set[str] = set()
            for shape in algorithm["shapes"]:
                sp = ap + "/" + shape["shape_id"]
                problem_hash = jcs_digest(shape["problem"])
                if problem_hash in problem_hashes:
                    _invalid(sp, "duplicate fixed problem; combine its attempts under one shape")
                problem_hashes.add(problem_hash)
                _unique(shape["attempts"], "attempt_id", sp)
                runs = [r for attempt in shape["attempts"] for r in attempt["runs"]]
                _unique(runs, "run_id", sp)
                for run in runs:
                    rp, result = sp + "/" + run["run_id"], run["result"]
                    latency, p90 = result["latency_us"], result.get("p90_latency_us")
                    if result["execution_status"] != "succeeded":
                        if latency is not None or p90 is not None or result["correctness"] == "pass":
                            _invalid(rp, "unsuccessful/not_run execution cannot have timing or correctness pass")
                    if p90 is not None and (latency is None or p90 < latency):
                        _invalid(rp, "p90_latency_us requires a median and must be >= latency_us")
                    if result.get("samples") == 0 and latency is not None:
                        _invalid(rp, "zero samples cannot have a recorded latency")
                baseline_id = shape.get("baseline_run_id")
                if baseline_id is not None:
                    baseline = next((r for r in runs if r["run_id"] == baseline_id), None)
                    if baseline is None:
                        _invalid(sp, "baseline_run_id must reference a run in this shape")
                    result = baseline["result"]
                    if (result["execution_status"] != "succeeded" or result["correctness"] != "pass"
                            or result["latency_us"] is None):
                        _invalid(sp, "baseline must have successful execution, passing correctness and latency")


def _ref(kind: str, *parts: str) -> str:
    return kind + "-" + jcs_digest(list(parts))[7:31]


def _doc(kind: str, **data: Any) -> dict[str, Any]:
    return {"schema_version": OUTPUT_VERSION, "document_type": kind, **data}


def _result(run: dict[str, Any], candidate: str, run_ref: str, subject: str, config: str) -> dict[str, Any]:
    result = run["result"]
    row = {
        "candidate": candidate, "run_ref": run_ref, "subject_ref": subject,
        "source_config_ref": config, "execution_status": result["execution_status"],
        "correctness": result["correctness"],
        "latency_us": round_output_value(result["latency_us"]),
        "p90_latency_us": round_output_value(result.get("p90_latency_us")),
        "samples": result.get("samples"),
        **{name: round_output_value(result.get(name), 4 if unit == "percent" else 3)
           for name, (unit, _) in METRIC_SLOTS.items()},
        "hlo": result.get("hlo", "not_collected"), "llo": result.get("llo", "not_collected"),
        "provenance": "imported_unverified", "failure_reason": result.get("failure_reason"),
        # A user-supplied summary has no independently fixed environment,
        # protocol or verifier identity. Do not manufacture comparability.
        "comparison_key": None,
    }
    return row


def build_memory_input_documents(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the standard tree without fabricating internal Git/store records."""
    validate_memory_input(data)
    docs: dict[str, dict[str, Any]] = {}
    for kernel in data["kernels"]:
        kid = kernel["kernel_id"]
        docs[f"{kid}/kernel.json"] = _doc(
            "kernel", kernel_ref=_ref("kernel", kid), kernel_id=kid,
            description=kernel["description"], algorithms=[
                {"algorithm_ref": _ref("algorithm", kid, a["algorithm_id"]), "algorithm_id": a["algorithm_id"]}
                for a in kernel["algorithms"]],
        )
        for algorithm in kernel["algorithms"]:
            aid = algorithm["algorithm_id"]
            aref, directory = _ref("algorithm", kid, aid), f"{kid}/{aid}"
            docs[f"{directory}/method_summary.json"] = _doc(
                "method_summary", kernel_id=kid, algorithm_ref=aref, algorithm_id=aid,
                display_name=algorithm.get("display_name", aid), method_summary=algorithm["method_summary"],
                author_kind=algorithm["summary_author"], tags=[], annotations=[],
            )
            groups = []
            for shape in algorithm["shapes"]:
                sid, problem = shape["shape_id"], copy.deepcopy(shape["problem"])
                shape_hash = jcs_digest({"output_shape_version": SHAPE_IDENTITY_VERSION, "kernel_id": kid, "problem": problem})
                cref = _ref("config", kid, aid, sid, shape_hash)
                identity = {"shape_hash": shape_hash, "source_config_refs": [cref]}
                shape_name = f"shape-{sid}-{shape_hash[7:19]}"
                path = f"{directory}/{shape_name}"
                docs[f"{path}/config.json"] = _doc(
                    "config", **identity, kernel_id=kid, algorithm_ref=aref, shape_id=sid,
                    problem=problem, tags=[], source_configs=[{
                        "config_ref": cref, "input_shape_id": sid,
                        "config_hash": jcs_digest({"kernel_id": kid, "problem": problem}),
                        "problem_schema_id": "urn:kernel-memory:input:1.0.0#/$defs/problem",
                        "problem_schema_digest": jcs_digest(memory_input_schema()),
                    }],
                )
                attempts, code, results = [], [], []
                baseline_result = None
                for attempt in shape["attempts"]:
                    subject = _ref("attempt", kid, aid, sid, attempt["attempt_id"])
                    attempt_runs = []
                    for run in attempt["runs"]:
                        rref = _ref("run", kid, aid, sid, run["run_id"])
                        row = _result(run, attempt["candidate"], rref, subject, cref)
                        results.append(row)
                        if run["run_id"] == shape.get("baseline_run_id"):
                            baseline_result = row
                        attempt_runs.append({
                            "run_ref": rref, "run_id": run["run_id"], "recorded_at": run.get("recorded_at"),
                            "environment": copy.deepcopy(run.get("environment", {})),
                            "measurement": copy.deepcopy(run.get("measurement", {})),
                        })
                        if any((attempt.get("code") or {}).values()):
                            code.append({"run_ref": rref, "subject_ref": subject, "source_config_ref": cref,
                                         "source": copy.deepcopy(attempt["code"]), "commit": attempt.get("commit"),
                                         "parameters": copy.deepcopy(attempt.get("parameters", {})),
                                         "content": None, "content_status": "not_retained_in_compact_output"})
                    attempts.append({
                        "subject_ref": subject, "source_config_ref": cref,
                        "attempt_id": attempt["attempt_id"], "title": attempt["candidate"],
                        "summary": attempt.get("summary"), "pr_url": attempt.get("pr_url"),
                        "parameters": copy.deepcopy(attempt.get("parameters", {})),
                        "commit": attempt.get("commit"), "runs": attempt_runs,
                    })
                docs[f"{path}/trajectory/attempt.json"] = _doc("attempt", **identity, attempts=attempts)
                docs[f"{path}/trajectory/result.json"] = _doc("result", **identity, runs=results, decisions=[], annotations=[])
                if code:
                    docs[f"{path}/trajectory/code.json"] = _doc("code", **identity, entries=code)
                groups.append({**identity, "shape": shape_name, "shared_baselines": [], "baselines": [{
                    "baseline_ref": _ref("baseline", kid, aid, sid, shape["baseline_run_id"]),
                    "config_ref": cref, "baseline_id": shape["baseline_run_id"], "runs": [baseline_result],
                }] if baseline_result is not None else []})
            docs[f"{directory}/baseline.json"] = _doc(
                "baseline", kernel_id=kid, algorithm_ref=aref, selection="declared_baselines_per_shape", shapes=groups,
            )
    return dict(sorted(docs.items()))


def import_memory_input(input_path: Path, output_root: Path | None = None, *, validate_only: bool = False) -> dict[str, Any]:
    """Import one complete input snapshot; no raw files or external sources are opened."""
    data = load_json_file(input_path)
    documents = build_memory_input_documents(data)
    counts = {"kernels": sum(d["document_type"] == "kernel" for d in documents.values()),
              "algorithms": sum(d["document_type"] == "method_summary" for d in documents.values()),
              "shapes": sum(d["document_type"] == "config" for d in documents.values()),
              "runs": sum(len(d["runs"]) for d in documents.values() if d["document_type"] == "result")}
    if validate_only:
        return {"ok": True, "input_format": INPUT_FORMAT, "schema_version": INPUT_VERSION, "counts": counts, "written": False}
    if output_root is None:
        raise InputError("output directory is required unless --validate-only is used")
    source, target = Path(input_path).resolve(), Path(output_root).resolve()
    if source == target or target in source.parents:
        raise InputError("input file must be outside the output directory", code="OUTPUT_OVERLAPS_INPUT")
    report = export_output_documents(documents, output_root)
    verified = verify_output_memory(output_root)
    if not verified["ok"]:
        raise InputError("generated output failed verification", details=verified)
    return {"ok": True, "counts": counts, "output_root": report["output_root"],
            "content_hash": report["content_hash"], "output_verified": True,
            "written": True, "measurements_reexecuted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate public memory input JSON and generate a result-only Memory directory.")
    parser.add_argument("input", type=Path, help="kernel-memory-input v1 JSON file")
    parser.add_argument("output", type=Path, nargs="?", help="complete snapshot output directory (refreshed, not merged)")
    parser.add_argument("--validate-only", action="store_true", help="check input without writing output")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output is None:
        parser.error("output is required unless --validate-only is used")
    try:
        print(json.dumps(import_memory_input(args.input, args.output, validate_only=args.validate_only), ensure_ascii=False, indent=2))
    except (KernelMemoryError, OSError) as exc:
        detail = exc.to_dict() if isinstance(exc, KernelMemoryError) else {"error": "IO_ERROR", "message": str(exc)}
        print(json.dumps(detail, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code if isinstance(exc, KernelMemoryError) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
