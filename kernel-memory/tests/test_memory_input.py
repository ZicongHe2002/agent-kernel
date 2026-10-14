"""Public input validates real boundaries and writes only compact memory results."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from kernel_memory.domain.errors import InputError
from kernel_memory.services.memory_input import (
    build_memory_input_documents,
    import_memory_input,
    main,
    validate_memory_input,
)
from kernel_memory.services.output_memory import verify_output_memory


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = {
    "candidate", "run_ref", "subject_ref", "source_config_ref", "execution_status",
    "correctness", "latency_us", "p90_latency_us", "samples", "mfu",
    "mfu_executed_estimate", "mxu", "attention_mxu_static", "roofline_efficiency",
    "logical_bandwidth_gbps", "logical_bandwidth_utilization", "hbm_bandwidth_gbps",
    "hbm_bandwidth_utilization", "hlo", "llo", "provenance", "failure_reason",
    "comparison_key",
}


@pytest.fixture
def input_data() -> dict:
    return json.loads((ROOT / "examples/memory_input.json").read_text(encoding="utf-8"))


def shape(data: dict) -> dict:
    return data["kernels"][0]["algorithms"][0]["shapes"][0]


def candidate(data: dict) -> dict:
    return shape(data)["attempts"][1]["runs"][0]


def docs_of(docs: dict, kind: str) -> list[dict]:
    return [doc for doc in docs.values() if doc["document_type"] == kind]


def write_input(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_documented_sample_matches_draft_schema_and_imports(input_data: dict, tmp_path: Path) -> None:
    schema = json.loads((ROOT / "src/kernel_memory/contracts/memory_input.v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(input_data)
    validate_memory_input(input_data)
    output = tmp_path / "memory"
    import_memory_input(write_input(tmp_path, input_data), output)
    report = verify_output_memory(output)
    assert report["ok"] is True
    assert report["counts"] == {"kernels": 1, "algorithms": 1, "shapes": 1, "runs": 2}
    assert all(p.suffix == ".json" for p in output.rglob("*") if p.is_file())
    assert not list(output.rglob("code.json"))
    assert not list(output.rglob("artifacts"))
    assert len(list(output.glob("vector_add_example/elementwise-add/shape-n4096-f32-*/config.json"))) == 1


def test_scalar_units_defaults_and_compiler_results_without_raw_files(input_data: dict) -> None:
    result = candidate(input_data)["result"]
    result.update(mfu=0.7484321, latency_us=10.123456, mfu_executed_estimate=1.251234,
                  roofline_efficiency=1.071234, logical_bandwidth_utilization=1.432198,
                  logical_bandwidth_gbps=1234.56789, attention_mxu_static=0.523456,
                  hlo="recorded", llo="recorded")
    docs = build_memory_input_documents(input_data)
    run = next(r for r in docs_of(docs, "result")[0]["runs"] if r["candidate"].startswith("Optimized"))
    assert set(run) == RESULT_FIELDS
    assert all(not isinstance(value, (dict, list)) for value in run.values())
    assert run["mfu"] == 0.7484
    assert run["latency_us"] == 10.123
    assert run["mfu_executed_estimate"] == 1.2512
    assert run["roofline_efficiency"] == 1.0712
    assert run["logical_bandwidth_utilization"] == 1.4322
    assert run["logical_bandwidth_gbps"] == 1234.568
    assert run["attention_mxu_static"] == 0.5235
    assert run["mxu"] is None
    assert run["hbm_bandwidth_gbps"] is None
    assert run["p90_latency_us"] is None
    assert run["samples"] is None
    assert run["comparison_key"] is None
    assert run["provenance"] == "imported_unverified"
    assert run["hlo"] == run["llo"] == "recorded"
    assert all(path.endswith(".json") and "artifacts" not in path for path in docs)


def test_missing_source_identity_stays_null_and_context_survives(input_data: dict) -> None:
    docs = build_memory_input_documents(input_data)
    attempts = docs_of(docs, "attempt")[0]["attempts"]
    for attempt in attempts:
        assert attempt["commit"] is None
        assert attempt["pr_url"] is None
        for run in attempt["runs"]:
            assert run["run_id"] in {"baseline-example", "candidate-example"}
            assert run["environment"]["device"] == "Hypothetical device; no actual run"
            assert run["measurement"]["scope"] == "Synthetic illustrative values, not measurements"


def test_tiny_positive_measurements_never_round_to_zero(input_data: dict) -> None:
    candidate(input_data)["result"].update(
        latency_us=0.0001, p90_latency_us=0.0002, mfu=0.00001,
        logical_bandwidth_gbps=0.00001,
    )
    docs = build_memory_input_documents(input_data)
    run = next(r for r in docs_of(docs, "result")[0]["runs"] if r["candidate"].startswith("Optimized"))
    assert run["latency_us"] == 0.0001
    assert run["p90_latency_us"] == 0.0002
    assert run["mfu"] == 0.00001
    assert run["logical_bandwidth_gbps"] == 0.00001


def test_optional_code_is_location_only(input_data: dict) -> None:
    shape(input_data)["attempts"][1]["parameters"] = {"tile": 256, "prefetch": True}
    shape(input_data)["attempts"][1]["code"] = {
        "repository_url": "https://example.com/example/repo",
        "path": "kernels/add.py", "entrypoint": "vector_add",
    }
    docs = build_memory_input_documents(input_data)
    code = docs_of(docs, "code")
    assert len(code) == 1
    serialized = json.dumps(code)
    assert "kernels/add.py" in serialized
    assert "vector_add" in serialized
    attempt = next(a for a in docs_of(docs, "attempt")[0]["attempts"] if a["attempt_id"] == "optimized-example")
    assert attempt["parameters"] == {"tile": 256, "prefetch": True}
    assert code[0]["entries"][0]["parameters"] == {"tile": 256, "prefetch": True}
    shape(input_data)["attempts"][1]["code"]["content"] = "raw source text"
    with pytest.raises(InputError):
        validate_memory_input(input_data)


@pytest.mark.parametrize("code", [None, {}, {"path": None}, {
    "repository_url": None, "path": None, "entrypoint": None,
}])
def test_empty_source_location_does_not_create_code_file(input_data: dict, code: dict | None) -> None:
    shape(input_data)["attempts"][1]["code"] = code
    docs = build_memory_input_documents(input_data)
    assert not docs_of(docs, "code")
    assert not any(path.endswith("/code.json") for path in docs)


@pytest.mark.parametrize("field,value", [
    ("mfu", 74.84), ("mfu", -0.1), ("mxu", 1.1),
    ("hbm_bandwidth_utilization", 1.1), ("attention_mxu_static", 1.1),
    ("mfu", True), ("latency_us", 0), ("latency_us", -1),
    ("llo", "raw compiler IR"), ("llo_path", "large.ll"),
    ("latency_ms", 0.01),
])
def test_unknown_fields_and_wrong_units_are_rejected(input_data: dict, field: str, value: object) -> None:
    candidate(input_data)["result"][field] = value
    with pytest.raises(InputError) as exc:
        validate_memory_input(input_data)
    assert exc.value.code == "INVALID_MEMORY_INPUT"


@pytest.mark.parametrize("scope", ["kernel", "algorithm", "shape", "attempt", "run"])
def test_duplicate_ids_are_rejected_in_their_scope(input_data: dict, scope: str) -> None:
    if scope == "kernel":
        values = input_data["kernels"]
    elif scope == "algorithm":
        values = input_data["kernels"][0]["algorithms"]
    elif scope == "shape":
        values = input_data["kernels"][0]["algorithms"][0]["shapes"]
    elif scope == "attempt":
        values = shape(input_data)["attempts"]
    else:
        values = shape(input_data)["attempts"][0]["runs"]
    values.append(copy.deepcopy(values[0]))
    with pytest.raises(InputError) as exc:
        validate_memory_input(input_data)
    assert exc.value.code == "INVALID_MEMORY_INPUT"


def test_baseline_does_not_resolve_from_another_shape(input_data: dict) -> None:
    second = copy.deepcopy(shape(input_data))
    second["shape_id"] = "another-shape"
    second["problem"]["input_signature"]["a"]["shape"] = [8192]
    second["problem"]["input_signature"]["b"]["shape"] = [8192]
    second["problem"]["output_signature"]["y"]["shape"] = [8192]
    second["attempts"] = [second["attempts"][1]]
    input_data["kernels"][0]["algorithms"][0]["shapes"].append(second)
    with pytest.raises(InputError, match="baseline_run_id"):
        validate_memory_input(input_data)


@pytest.mark.parametrize("updates", [
    {"execution_status": "runtime_error"},
    {"correctness": "fail"},
    {"latency_us": None},
])
def test_baseline_must_have_success_correctness_and_timing(input_data: dict, updates: dict) -> None:
    shape(input_data)["attempts"][0]["runs"][0]["result"].update(updates)
    with pytest.raises(InputError):
        validate_memory_input(input_data)


@pytest.mark.parametrize("updates", [
    {"execution_status": "timeout", "latency_us": None},
    {"execution_status": "compile_error", "correctness": "not_run"},
    {"p90_latency_us": 9.0},
    {"latency_us": None, "p90_latency_us": 10.0},
    {"samples": 0},
])
def test_inconsistent_measurements_are_rejected(input_data: dict, updates: dict) -> None:
    candidate(input_data)["result"].update(updates)
    with pytest.raises(InputError):
        validate_memory_input(input_data)


def test_failed_run_is_retained_without_fabricated_measurements(input_data: dict) -> None:
    candidate(input_data)["result"] = {
        "execution_status": "compile_error", "correctness": "not_run",
        "latency_us": None, "failure_reason": "Illustrative compile failure",
    }
    docs = build_memory_input_documents(input_data)
    failed = next(r for r in docs_of(docs, "result")[0]["runs"] if r["execution_status"] == "compile_error")
    assert failed["correctness"] == "not_run"
    assert failed["latency_us"] is None
    assert failed["mfu"] is None
    assert failed["llo"] == "not_collected"


def test_repeatable_export_and_invalid_input_does_not_modify_existing_output(input_data: dict, tmp_path: Path) -> None:
    source = write_input(tmp_path, input_data)
    output = tmp_path / "memory"
    import_memory_input(source, output)
    first = snapshot(output)
    import_memory_input(source, output)
    assert snapshot(output) == first
    candidate(input_data)["result"]["mfu"] = 74.84
    write_input(tmp_path, input_data)
    with pytest.raises(InputError):
        import_memory_input(source, output)
    assert snapshot(output) == first


def test_refresh_replaces_previous_snapshot_instead_of_merging(input_data: dict, tmp_path: Path) -> None:
    source = write_input(tmp_path, input_data)
    output = tmp_path / "memory"
    import_memory_input(source, output)
    input_data["kernels"][0]["algorithms"][0]["algorithm_id"] = "replacement"
    write_input(tmp_path, input_data)
    import_memory_input(source, output)
    assert not list(output.glob("vector_add_example/elementwise-add/**/*.json"))
    assert list(output.glob("vector_add_example/replacement/**/*.json"))
    assert verify_output_memory(output)["ok"] is True


def test_foreign_files_and_input_inside_output_are_preserved(input_data: dict, tmp_path: Path) -> None:
    output = tmp_path / "memory"
    output.mkdir()
    foreign = output / "keep.txt"
    foreign.write_text("keep this", encoding="utf-8")
    source = write_input(tmp_path, input_data)
    with pytest.raises(InputError):
        import_memory_input(source, output)
    assert foreign.read_text() == "keep this"
    nested_source = output / "input.json"
    nested_source.write_bytes(source.read_bytes())
    before = snapshot(output)
    with pytest.raises(InputError):
        import_memory_input(nested_source, output)
    assert snapshot(output) == before


@pytest.mark.parametrize("text", [
    '{"format": "kernel-memory-input", "format": "wrong"}',
    '{"format": "kernel-memory-input", "mfu": NaN}',
    '{"format": "kernel-memory-input", "mfu": Infinity}',
    '{"format":',
])
def test_malformed_duplicate_and_nonfinite_json_are_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputError):
        import_memory_input(path, validate_only=True)


def test_cli_validates_without_creating_output(input_data: dict, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    source = write_input(tmp_path, input_data)
    before = snapshot(tmp_path)
    assert main([str(source), "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert snapshot(tmp_path) == before
    candidate(input_data)["result"]["mfu"] = 74.84
    write_input(tmp_path, input_data)
    assert main([str(source), "--validate-only"]) == 2
