"""Strict JSON loading and readable JSON writing.

* Duplicate object keys are rejected (RFC 8259 leaves them undefined; we refuse).
* ``NaN``/``Infinity`` constants are rejected.
* Oversized inputs are rejected before parsing.
* Optional YAML input uses the safe loader, rejects duplicate keys and custom
  tags, and is normalised to the same JSON value (YAML is never a second truth).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import InputError, PrerequisiteMissingError

DEFAULT_MAX_BYTES = 50_000_000


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key: {key!r}", code="DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise InputError(f"non-finite JSON constant is not permitted: {value}", code="NONFINITE_JSON")


def loads_strict(text: str | bytes) -> Any:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputError(f"input is not valid UTF-8: {exc}", code="INVALID_UTF8") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON: {exc}", code="INVALID_JSON") from exc


def load_json_file(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Any:
    path = Path(path)
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise InputError(f"file not found: {path}", code="FILE_NOT_FOUND") from exc
    if size > max_bytes:
        raise InputError(
            f"input {path} is {size} bytes, above the limit of {max_bytes}", code="INPUT_TOO_LARGE"
        )
    return loads_strict(path.read_bytes())


def dumps_readable(value: Any) -> str:
    """Stable, human-readable JSON for authoritative files (not canonical form)."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def load_yaml_strict(text: str) -> Any:
    """Safe YAML load that rejects duplicate keys and custom tags; returns JSON-compatible data."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PrerequisiteMissingError("PyYAML is not installed; YAML input is unavailable") from exc

    class StrictLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise InputError("YAML mapping keys must be strings", code="INVALID_YAML")
            if key in mapping:
                raise InputError(f"duplicate YAML key: {key!r}", code="DUPLICATE_YAML_KEY")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    def reject_tag(loader: Any, tag_suffix: Any, node: Any) -> Any:
        raise InputError(f"custom YAML tag not permitted: {node.tag}", code="INVALID_YAML")

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    StrictLoader.add_multi_constructor("!", reject_tag)
    StrictLoader.add_multi_constructor("tag:yaml.org,2002:python/", reject_tag)
    try:
        data = yaml.load(text, Loader=StrictLoader)  # noqa: S506 - StrictLoader derives from SafeLoader
    except yaml.YAMLError as exc:
        raise InputError(f"invalid YAML: {exc}", code="INVALID_YAML") from exc
    # Normalise to JSON data model (dates etc. are rejected by round-tripping).
    try:
        return loads_strict(json.dumps(data, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InputError(f"YAML content is not JSON-compatible: {exc}", code="INVALID_YAML") from exc
