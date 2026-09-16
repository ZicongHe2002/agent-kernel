"""Promotion policy: defaults, loading, validation, and hashing (specification 12.2, DESIGN section 6).

A policy is a plain JSON object matching the ``Policy`` contract in
``contracts/record.schema.json``. Thresholds are *strings* on the wire on purpose: they are
parsed with ``decimal.Decimal`` so that ``"1.02"`` means exactly 1.02 and the policy hash never
depends on binary float formatting.

Public API
----------
``default_policy() -> dict``
    Deep copy of the project default ``default-confirm-v1`` (the ``promotion_policy`` golden vector):
    3 confirmation pairs, per-pair speedup >= 1.02, normalized IQR <= 0.05, baseline drift <= 1.05,
    no hard resource constraints, trusted workers required, fixtures not allowed.
``load_policy(path, *, root=None) -> dict``
    Load a JSON (or ``.yaml``/``.yml``) policy file strictly (duplicate keys and NaN rejected) and
    validate it. When ``root`` is given the path is resolved with ``ids.resolve_inside`` (traversal safe).
``validate_policy(policy) -> dict``
    Schema validation (``schema.validate_nested("Policy")``) plus threshold/constraint checks; returns a
    deep copy. Raises ``SchemaValidationError`` (exit 2, code ``INVALID_POLICY``).
``policy_hash(policy) -> str``
    ``sha256:<hex>`` of the validated policy (RFC 8785 canonical form). Reproduces the golden digest.
``parse_threshold(value, name) -> Decimal``, ``thresholds(policy) -> PolicyThresholds``
    Typed, Decimal-valued view of a policy for the decision service.
``ResourceConstraint``
    ``{"metric": str, "max_value": number|None, "min_value": number|None, "scope": str|None, "kind": str|None}``
    at least one bound is required; unknown keys are rejected.

Guarantees: pure functions, no store access, no network. Nothing here invents measurements.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..domain import hashing
from ..domain.errors import InputError, SchemaValidationError
from ..domain.ids import resolve_inside
from ..domain.jsonio import load_json_file, load_yaml_strict
from ..domain.schema import validate_nested

DEFAULT_POLICY_ID = "default-confirm-v1"
POLICY_CODE = "INVALID_POLICY"

_DEFAULT_POLICY: dict[str, Any] = {
    "policy_id": DEFAULT_POLICY_ID,
    "min_confirm_pairs": 3,
    "min_pair_speedup": "1.02",
    "max_normalized_iqr": "0.05",
    "max_baseline_drift_ratio": "1.05",
    "hard_resource_constraints": [],
    "require_trusted_worker": True,
    "allow_fixture": False,
}

CONSTRAINT_FIELDS: frozenset[str] = frozenset({"metric", "max_value", "min_value", "scope", "kind"})
THRESHOLD_FIELDS: tuple[str, ...] = ("min_pair_speedup", "max_normalized_iqr", "max_baseline_drift_ratio")


@dataclass(frozen=True)
class ResourceConstraint:
    metric: str
    max_value: float | None
    min_value: float | None
    scope: str | None
    kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "max_value": self.max_value,
            "min_value": self.min_value,
            "scope": self.scope,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class PolicyThresholds:
    policy_id: str
    min_confirm_pairs: int
    min_pair_speedup: Decimal
    max_normalized_iqr: Decimal
    max_baseline_drift_ratio: Decimal
    hard_resource_constraints: tuple[ResourceConstraint, ...]
    require_trusted_worker: bool
    allow_fixture: bool


def default_policy() -> dict[str, Any]:
    """The project default policy (equal to the ``promotion_policy`` golden vector payload)."""
    return copy.deepcopy(_DEFAULT_POLICY)


def parse_threshold(value: Any, name: str) -> Decimal:
    """Parse a policy threshold string into a Decimal; rejects numbers, NaN, infinities, negatives."""
    if not isinstance(value, str):
        raise SchemaValidationError(
            f"policy.{name} must be a decimal string (schema strings are deliberate), got {type(value).__name__}",
            code=POLICY_CODE,
            details={"field": name, "value": value},
        )
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise SchemaValidationError(f"policy.{name} is not a decimal number: {value!r}", code=POLICY_CODE) from exc
    if not parsed.is_finite() or parsed < 0:
        raise SchemaValidationError(f"policy.{name} must be finite and non-negative: {value!r}", code=POLICY_CODE)
    return parsed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_constraint(item: Any, index: int) -> ResourceConstraint:
    """Validate one ``hard_resource_constraints`` entry and return its typed form."""
    where = f"policy.hard_resource_constraints[{index}]"
    if not isinstance(item, dict):
        raise SchemaValidationError(f"{where} must be an object", code=POLICY_CODE)
    unknown = sorted(set(item) - CONSTRAINT_FIELDS)
    if unknown:
        raise SchemaValidationError(f"{where} has unknown fields {unknown}", code=POLICY_CODE, details={"unknown": unknown})
    metric = item.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise SchemaValidationError(f"{where}.metric must be a non-empty string", code=POLICY_CODE)
    bounds: dict[str, float | None] = {}
    for bound in ("max_value", "min_value"):
        value = item.get(bound)
        if value is not None and not _is_number(value):
            raise SchemaValidationError(f"{where}.{bound} must be a number or null", code=POLICY_CODE)
        if value is not None and not (float("-inf") < float(value) < float("inf")):
            raise SchemaValidationError(f"{where}.{bound} must be finite", code=POLICY_CODE)
        bounds[bound] = value
    if bounds["max_value"] is None and bounds["min_value"] is None:
        raise SchemaValidationError(f"{where} needs at least one of max_value/min_value", code=POLICY_CODE)
    if bounds["max_value"] is not None and bounds["min_value"] is not None and bounds["min_value"] > bounds["max_value"]:
        raise SchemaValidationError(f"{where}.min_value exceeds max_value", code=POLICY_CODE)
    for optional in ("scope", "kind"):
        value = item.get(optional)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SchemaValidationError(f"{where}.{optional} must be a non-empty string or null", code=POLICY_CODE)
    return ResourceConstraint(
        metric=metric,
        max_value=bounds["max_value"],
        min_value=bounds["min_value"],
        scope=item.get("scope"),
        kind=item.get("kind"),
    )


def validate_policy(policy: Any) -> dict[str, Any]:
    """Validate a policy dict against the contract and the semantic rules; return a deep copy."""
    if not isinstance(policy, dict):
        raise SchemaValidationError(
            "policy must be a JSON object", code=POLICY_CODE, details={"got": type(policy).__name__}
        )
    try:
        validate_nested("Policy", policy)
    except SchemaValidationError as exc:
        raise SchemaValidationError(exc.message, code=POLICY_CODE, details={"policy_id": policy.get("policy_id")}) from exc
    speedup = parse_threshold(policy["min_pair_speedup"], "min_pair_speedup")
    if speedup <= 0:
        raise SchemaValidationError("policy.min_pair_speedup must be > 0", code=POLICY_CODE)
    parse_threshold(policy["max_normalized_iqr"], "max_normalized_iqr")
    drift = parse_threshold(policy["max_baseline_drift_ratio"], "max_baseline_drift_ratio")
    if drift < 1:
        raise SchemaValidationError(
            "policy.max_baseline_drift_ratio must be >= 1 (it bounds max/min of baseline medians)", code=POLICY_CODE
        )
    for index, item in enumerate(policy["hard_resource_constraints"]):
        validate_constraint(item, index)
    try:
        return json.loads(json.dumps(policy, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"policy is not JSON-serialisable: {exc}", code=POLICY_CODE) from exc


def thresholds(policy: dict[str, Any]) -> PolicyThresholds:
    """Typed view of a validated policy (validates first)."""
    valid = validate_policy(policy)
    return PolicyThresholds(
        policy_id=valid["policy_id"],
        min_confirm_pairs=int(valid["min_confirm_pairs"]),
        min_pair_speedup=parse_threshold(valid["min_pair_speedup"], "min_pair_speedup"),
        max_normalized_iqr=parse_threshold(valid["max_normalized_iqr"], "max_normalized_iqr"),
        max_baseline_drift_ratio=parse_threshold(valid["max_baseline_drift_ratio"], "max_baseline_drift_ratio"),
        hard_resource_constraints=tuple(
            validate_constraint(item, i) for i, item in enumerate(valid["hard_resource_constraints"])
        ),
        require_trusted_worker=bool(valid["require_trusted_worker"]),
        allow_fixture=bool(valid["allow_fixture"]),
    )


def policy_hash(policy: dict[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical (RFC 8785) form of the validated policy."""
    return hashing.policy_hash(validate_policy(policy))


def load_policy(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    """Load and validate a policy file (JSON, or YAML by extension)."""
    if root is not None:
        target = resolve_inside(Path(root), str(path))
    else:
        target = Path(path)
    suffix = target.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            text = target.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise InputError(f"policy file not found: {target}", code="FILE_NOT_FOUND") from exc
        except UnicodeDecodeError as exc:
            raise InputError(f"policy file is not UTF-8: {target}", code="INVALID_UTF8") from exc
        data = load_yaml_strict(text)
    else:
        data = load_json_file(target)
    return validate_policy(data)


__all__ = [
    "CONSTRAINT_FIELDS",
    "DEFAULT_POLICY_ID",
    "PolicyThresholds",
    "ResourceConstraint",
    "default_policy",
    "load_policy",
    "parse_threshold",
    "policy_hash",
    "thresholds",
    "validate_constraint",
    "validate_policy",
]
