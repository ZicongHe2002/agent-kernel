"""Correctness verification primitives (specification section 10).

Public API
----------
``elementwise_close(candidate, reference, *, atol, rtol, nonfinite_policy) -> VerifyOutcome``
    Acceptance rule, applied to every element of the complete output pytree::

        abs(candidate_i - reference_i) <= atol + rtol * abs(reference_i)

    ``candidate``/``reference`` may be numpy arrays, array-likes (anything exposing
    ``__array__``, e.g. JAX arrays), Python scalars, nested numeric Python lists (treated
    as one array), or pytrees built from ``dict`` / ``list`` / ``tuple`` of those. The tree
    structure, every leaf shape, and every leaf dtype *name* must match; otherwise the
    outcome fails with ``STRUCTURE_MISMATCH`` / ``SHAPE_MISMATCH`` / ``DTYPE_MISMATCH`` in
    the message (``list`` and ``tuple`` are distinct container kinds; a nested Python list
    of numbers is an array leaf and becomes ``float64``/``int64`` like ``numpy.asarray``).

    Tolerances may be strings (parsed with ``decimal.Decimal``, then converted to float)
    or non-negative finite numbers. ``nonfinite_policy``:

    * ``reject_unexpected`` / ``match_reference``: a non-finite candidate element where the
      reference is finite fails; where the reference is non-finite the candidate must match
      exactly (NaN matches NaN, infinities must agree in sign).
    * ``reject_all``: any non-finite element on either side fails.

    ``max_abs_error`` / ``max_rel_error`` are diagnostics over the pairs where both sides
    are finite; ``max_rel_error = max(abs(c - r) / max(abs(r), eps))`` with
    ``eps = REL_ERROR_EPSILON = 1e-12`` stated in ``VerifyOutcome.rel_error_epsilon``. They
    never replace the acceptance rule.

``verify_pytree_structure(candidate, reference) -> PytreeStructure``
    Structural comparison only; returns the aligned leaf pairs and the problems found.

``parse_tolerance(value, name) -> float``
    Decimal-string or numeric tolerance parsing with ``InputError`` on invalid values.

``correctness_report_bytes(...) -> bytes``
    JSON bytes for a ``correctness_report`` artifact (rule, tolerances, policy, epsilon,
    reference identity, per-case results). Non-finite numbers are written as strings so the
    artifact stays strict JSON.

Guarantees: pure functions, no I/O, no hidden loosening of tolerances, errors are
``kernel_memory.domain.errors`` subclasses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

from ..domain.errors import InputError
from ..domain.jsonio import dumps_readable

REL_ERROR_EPSILON = 1e-12
NONFINITE_POLICIES: tuple[str, ...] = ("reject_unexpected", "match_reference", "reject_all")
ACCEPTANCE_RULE = "abs(candidate - reference) <= atol + rtol * abs(reference), elementwise over the complete output pytree"
MAX_PROBLEMS_IN_MESSAGE = 5


@dataclass(frozen=True)
class VerifyOutcome:
    passed: bool
    max_abs_error: float | None
    max_rel_error: float | None
    rel_error_epsilon: float
    mismatch_count: int
    checked_count: int
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "rel_error_epsilon": self.rel_error_epsilon,
            "mismatch_count": self.mismatch_count,
            "checked_count": self.checked_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class PytreeStructure:
    matches: bool
    leaves: list[tuple[str, Any, Any]]
    problems: list[str]


# --------------------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------------------
def parse_tolerance(value: Any, name: str = "tolerance") -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a number or decimal string, got boolean", code="INVALID_TOLERANCE")
    if isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except (InvalidOperation, ValueError) as exc:
            raise InputError(f"{name} is not a decimal number: {value!r}", code="INVALID_TOLERANCE") from exc
        if not parsed.is_finite():
            raise InputError(f"{name} must be finite: {value!r}", code="INVALID_TOLERANCE")
        result = float(parsed)
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise InputError(f"{name} must be a number or decimal string, got {type(value).__name__}", code="INVALID_TOLERANCE")
    if not math.isfinite(result):
        raise InputError(f"{name} must be finite: {value!r}", code="INVALID_TOLERANCE")
    if result < 0:
        raise InputError(f"{name} must be non-negative: {value!r}", code="INVALID_TOLERANCE")
    return result


# --------------------------------------------------------------------------------------
# Pytree handling
# --------------------------------------------------------------------------------------
def _is_numeric_nested(value: Any) -> bool:
    """A nested Python list/tuple containing only numbers (an array written as lists)."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, complex)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_numeric_nested(v) for v in value)
    return False


def _node_kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (np.ndarray, np.generic)) or hasattr(value, "__array__"):
        return "leaf"
    if isinstance(value, (bool, int, float, complex)):
        return "leaf"
    if isinstance(value, (list, tuple)):
        if _is_numeric_nested(value):
            return "leaf"
        return "list" if isinstance(value, list) else "tuple"
    return f"unsupported:{type(value).__name__}"


def _flatten_pair(candidate: Any, reference: Any, path: str, leaves: list[tuple[str, Any, Any]], problems: list[str]) -> None:
    c_kind = _node_kind(candidate)
    r_kind = _node_kind(reference)
    if c_kind != r_kind:
        problems.append(f"STRUCTURE_MISMATCH at {path}: candidate is {c_kind}, reference is {r_kind}")
        return
    if c_kind == "none":
        return
    if c_kind == "dict":
        c_keys = set(candidate)
        r_keys = set(reference)
        if c_keys != r_keys:
            missing = sorted(str(k) for k in r_keys - c_keys)
            extra = sorted(str(k) for k in c_keys - r_keys)
            problems.append(f"STRUCTURE_MISMATCH at {path}: missing keys {missing}, unexpected keys {extra}")
            shared = c_keys & r_keys
        else:
            shared = c_keys
        for key in sorted(shared, key=str):
            _flatten_pair(candidate[key], reference[key], f"{path}.{key}" if path else str(key), leaves, problems)
        return
    if c_kind in ("list", "tuple"):
        if len(candidate) != len(reference):
            problems.append(f"STRUCTURE_MISMATCH at {path}: {c_kind} length {len(candidate)} vs {len(reference)}")
            return
        for i, (c, r) in enumerate(zip(candidate, reference)):
            _flatten_pair(c, r, f"{path}[{i}]", leaves, problems)
        return
    if c_kind == "leaf":
        leaves.append((path or "<root>", candidate, reference))
        return
    problems.append(f"STRUCTURE_MISMATCH at {path}: {c_kind} is not a supported output type")


def verify_pytree_structure(candidate: Any, reference: Any) -> PytreeStructure:
    leaves: list[tuple[str, Any, Any]] = []
    problems: list[str] = []
    _flatten_pair(candidate, reference, "", leaves, problems)
    return PytreeStructure(matches=not problems, leaves=leaves, problems=problems)


def _as_array(value: Any, path: str, side: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{side} output at {path} is not array-like: {exc}", code="INVALID_ARRAY") from exc


# --------------------------------------------------------------------------------------
# Leaf comparison
# --------------------------------------------------------------------------------------
@dataclass
class _LeafResult:
    fatal: str | None = None
    checked: int = 0
    mismatches: int = 0
    max_abs: float | None = None
    max_rel: float | None = None
    first_mismatch: str | None = None


def _working_dtype(dtype: np.dtype) -> Any:
    """Comparison dtype: complex stays complex; every other numeric dtype (including
    ml_dtypes extension types such as bfloat16, which register with kind 'V') is compared
    in float64 so the tolerance rule is evaluated without extra rounding."""
    if dtype.kind == "c":
        return np.complex128
    return np.float64


def _format_index(idx: tuple[int, ...]) -> str:
    return "[" + ",".join(str(i) for i in idx) + "]" if idx else ""


def _compare_leaf(path: str, candidate: Any, reference: Any, atol: float, rtol: float, policy: str, eps: float) -> _LeafResult:
    result = _LeafResult()
    ca = _as_array(candidate, path, "candidate")
    ra = _as_array(reference, path, "reference")
    if ca.shape != ra.shape:
        result.fatal = f"SHAPE_MISMATCH at {path}: candidate {tuple(ca.shape)} vs reference {tuple(ra.shape)}"
        return result
    if ca.dtype.name != ra.dtype.name:
        result.fatal = f"DTYPE_MISMATCH at {path}: candidate {ca.dtype.name} vs reference {ra.dtype.name}"
        return result
    if ra.dtype.kind in "OSUM":
        result.fatal = f"UNSUPPORTED_DTYPE at {path}: {ra.dtype.name} is not a numeric dtype"
        return result
    work = _working_dtype(ra.dtype)
    try:
        cw = ca.astype(work)
        rw = ra.astype(work)
    except (TypeError, ValueError) as exc:
        result.fatal = f"UNSUPPORTED_DTYPE at {path}: cannot convert {ra.dtype.name} for comparison ({exc})"
        return result
    result.checked = int(rw.size)
    if rw.size == 0:
        return result
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        c_finite = np.isfinite(cw)
        r_finite = np.isfinite(rw)
        both = c_finite & r_finite
        diff = np.zeros(rw.shape, dtype=np.float64)
        diff[both] = np.abs(cw[both] - rw[both])
        threshold = atol + rtol * np.abs(rw).astype(np.float64, copy=False)
        mismatch = np.zeros(rw.shape, dtype=bool)
        mismatch[both] = ~(diff[both] <= threshold[both])
        if policy == "reject_all":
            mismatch |= ~c_finite | ~r_finite
        else:  # reject_unexpected and match_reference share the same rule
            mismatch |= r_finite & ~c_finite
            exact = (np.isnan(cw) & np.isnan(rw)) | (cw == rw)
            mismatch |= ~r_finite & ~exact
        result.mismatches = int(np.count_nonzero(mismatch))
        if both.any():
            result.max_abs = float(diff[both].max())
            denominators = np.maximum(np.abs(rw[both]).astype(np.float64, copy=False), eps)
            result.max_rel = float((diff[both] / denominators).max())
        if result.mismatches:
            idx = tuple(int(i) for i in np.argwhere(mismatch)[0])
            c_val = cw[idx] if idx else cw[()]
            r_val = rw[idx] if idx else rw[()]
            result.first_mismatch = f"{path}{_format_index(idx)}: candidate={_scalar_repr(c_val)} reference={_scalar_repr(r_val)}"
    return result


def _scalar_repr(value: Any) -> str:
    try:
        if isinstance(value, (np.complexfloating, complex)):
            return repr(complex(value))
        return repr(float(value))
    except (TypeError, ValueError):
        return repr(value)


# --------------------------------------------------------------------------------------
# Public comparison
# --------------------------------------------------------------------------------------
def elementwise_close(candidate: Any, reference: Any, *, atol: str | float, rtol: str | float, nonfinite_policy: str) -> VerifyOutcome:
    atol_f = parse_tolerance(atol, "atol")
    rtol_f = parse_tolerance(rtol, "rtol")
    if nonfinite_policy not in NONFINITE_POLICIES:
        raise InputError(
            f"unknown nonfinite_policy {nonfinite_policy!r}",
            code="INVALID_NONFINITE_POLICY",
            details={"known": list(NONFINITE_POLICIES)},
        )
    structure = verify_pytree_structure(candidate, reference)
    if not structure.matches:
        return VerifyOutcome(
            passed=False,
            max_abs_error=None,
            max_rel_error=None,
            rel_error_epsilon=REL_ERROR_EPSILON,
            mismatch_count=0,
            checked_count=0,
            message="; ".join(structure.problems[:MAX_PROBLEMS_IN_MESSAGE]),
        )
    fatal: list[str] = []
    checked = 0
    mismatches = 0
    max_abs: float | None = None
    max_rel: float | None = None
    first_mismatches: list[str] = []
    for path, c, r in structure.leaves:
        leaf = _compare_leaf(path, c, r, atol_f, rtol_f, nonfinite_policy, REL_ERROR_EPSILON)
        if leaf.fatal:
            fatal.append(leaf.fatal)
            continue
        checked += leaf.checked
        mismatches += leaf.mismatches
        if leaf.max_abs is not None:
            max_abs = leaf.max_abs if max_abs is None else max(max_abs, leaf.max_abs)
        if leaf.max_rel is not None:
            max_rel = leaf.max_rel if max_rel is None else max(max_rel, leaf.max_rel)
        if leaf.first_mismatch and len(first_mismatches) < MAX_PROBLEMS_IN_MESSAGE:
            first_mismatches.append(leaf.first_mismatch)
    if fatal:
        return VerifyOutcome(
            passed=False,
            max_abs_error=None,
            max_rel_error=None,
            rel_error_epsilon=REL_ERROR_EPSILON,
            mismatch_count=mismatches,
            checked_count=checked,
            message="; ".join(fatal[:MAX_PROBLEMS_IN_MESSAGE]),
        )
    passed = mismatches == 0
    if passed:
        message = None if checked else "no elements compared (empty outputs with matching structure)"
    else:
        message = (
            f"{mismatches} of {checked} elements violate {ACCEPTANCE_RULE} with atol={atol_f!r}, rtol={rtol_f!r}, "
            f"nonfinite_policy={nonfinite_policy}; first mismatches: " + "; ".join(first_mismatches)
        )
    return VerifyOutcome(
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        rel_error_epsilon=REL_ERROR_EPSILON,
        mismatch_count=mismatches,
        checked_count=checked,
        message=message,
    )


# --------------------------------------------------------------------------------------
# Report artifact
# --------------------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else repr(f)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _is_true(value: Any) -> bool:
    """Exactly a boolean true (Python or numpy); never a truthy number or string."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _numeric_or_none(value: Any) -> float | None:
    """A real number (Python or numpy, never a boolean) as float; anything else is None."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def correctness_report_bytes(
    *,
    status: str,
    cases: list[dict[str, Any]],
    tolerances: dict[str, Any],
    nonfinite_policy: str,
    reference_id: str,
    rel_error_epsilon: float = REL_ERROR_EPSILON,
    **meta: Any,
) -> bytes:
    if status not in ("pass", "fail", "error", "not_run"):
        raise InputError(f"invalid correctness status {status!r}", code="INVALID_STATUS")
    for key in ("kind", "status", "cases", "tolerances", "nonfinite_policy", "reference_id", "is_fixture"):
        if key in meta:
            raise InputError(f"metadata key {key!r} is reserved in a correctness report", code="RESERVED_KEY")
    # Header counts/maxima must agree with the serialised cases: numpy scalars (np.bool_,
    # np.float32, ...) count exactly like the Python values they are written as.
    abs_errors = [v for v in (_numeric_or_none(c.get("max_abs_error")) for c in cases) if v is not None]
    rel_errors = [v for v in (_numeric_or_none(c.get("max_rel_error")) for c in cases) if v is not None]
    document: dict[str, Any] = {
        "kind": "correctness_report",
        "status": status,
        "rule": ACCEPTANCE_RULE,
        "tolerances": dict(tolerances),
        "nonfinite_policy": nonfinite_policy,
        "rel_error_epsilon": rel_error_epsilon,
        "reference_id": reference_id,
        "cases_total": len(cases),
        "cases_passed": sum(1 for c in cases if _is_true(c.get("passed"))),
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "max_rel_error": max(rel_errors) if rel_errors else None,
        "cases": [_json_safe(c) for c in cases],
    }
    document.update(_json_safe(meta))
    document["is_fixture"] = False
    return dumps_readable(_json_safe(document)).encode("utf-8")


__all__ = [
    "ACCEPTANCE_RULE",
    "NONFINITE_POLICIES",
    "REL_ERROR_EPSILON",
    "PytreeStructure",
    "VerifyOutcome",
    "correctness_report_bytes",
    "elementwise_close",
    "parse_tolerance",
    "verify_pytree_structure",
]
