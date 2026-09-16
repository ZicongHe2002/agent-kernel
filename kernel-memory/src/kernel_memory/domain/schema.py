"""JSON Schema (2020-12) validation of wire records against the companion contracts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from .errors import SchemaValidationError
from .ids import parse_utc_timestamp
from .jsonio import load_json_file

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"
RECORD_TYPES: tuple[str, ...] = (
    "kernel",
    "config",
    "pr",
    "pr_snapshot",
    "commit",
    "baseline",
    "run",
    "relation",
    "decision",
    "annotation",
)
SCHEMA_VERSION = "0.2.0"


@lru_cache(maxsize=1)
def record_schema() -> dict[str, Any]:
    schema = load_json_file(CONTRACTS_DIR / "record.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=1)
def demo_problem_schema() -> dict[str, Any]:
    schema = load_json_file(CONTRACTS_DIR / "demo_problem.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=1)
def golden_hash_vectors() -> dict[str, Any]:
    return load_json_file(CONTRACTS_DIR / "hash_vectors.json")


@lru_cache(maxsize=None)
def _type_validator(record_type: str) -> Draft202012Validator:
    schema = record_schema()
    sub = {
        "$schema": schema["$schema"],
        "$ref": f"#/$defs/{record_type}",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(sub, format_checker=FormatChecker())


@lru_cache(maxsize=None)
def _nested_validator(def_name: str) -> Draft202012Validator:
    schema = record_schema()
    sub = {"$schema": schema["$schema"], "$ref": f"#/$defs/{def_name}", "$defs": schema["$defs"]}
    return Draft202012Validator(sub, format_checker=FormatChecker())


def _format_error(err: ValidationError) -> str:
    path = "/".join(str(p) for p in err.absolute_path) or "<root>"
    return f"{path}: {err.message}"


def best_error(errors: list[ValidationError]) -> str:
    if not errors:
        return "unknown schema error"
    # Prefer the deepest, most specific error for readable diagnostics.
    errors = sorted(errors, key=lambda e: (-len(list(e.absolute_path)), e.message))
    return "; ".join(_format_error(e) for e in errors[:3])


def validate_record_dict(data: Any) -> str:
    """Validate a wire record dict. Returns the record_type. Raises SchemaValidationError."""
    if not isinstance(data, dict):
        raise SchemaValidationError("record must be a JSON object", details={"got": type(data).__name__})
    record_type = data.get("record_type")
    if record_type not in RECORD_TYPES:
        raise SchemaValidationError(
            f"unknown record_type: {record_type!r}", details={"known": list(RECORD_TYPES)}
        )
    validator = _type_validator(record_type)
    errors = list(validator.iter_errors(data))
    if errors:
        raise SchemaValidationError(
            f"{record_type} record {data.get('record_id')!r} is invalid: {best_error(errors)}",
            details={"record_id": data.get("record_id"), "record_type": record_type},
        )
    # jsonschema only enforces `format: date-time` when an optional dependency is present;
    # enforce timezone-aware timestamps explicitly.
    parse_utc_timestamp(data["created_at"])
    return record_type


def validate_nested(def_name: str, data: Any) -> None:
    """Validate a nested contract object (Environment, Protocol, Verifier, Policy, ...)."""
    errors = list(_nested_validator(def_name).iter_errors(data))
    if errors:
        raise SchemaValidationError(f"{def_name} is invalid: {best_error(errors)}")


def validate_against(schema: dict[str, Any], data: Any, *, what: str = "document") -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = list(validator.iter_errors(data))
    if errors:
        raise SchemaValidationError(f"{what} is invalid: {best_error(errors)}")


def is_valid_record_dict(data: Any) -> bool:
    try:
        validate_record_dict(data)
    except Exception:
        return False
    return True
