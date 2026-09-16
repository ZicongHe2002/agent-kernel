"""Error hierarchy shared by services, adapters, storage, and the CLI.

Every error carries a stable machine-readable ``code`` and the CLI exit code
proposed by the implementation specification (section 16):

    0 success
    2 input / schema error
    3 reference / idempotency conflict
    4 incomparable / insufficient evidence
    5 unavailable backend / missing prerequisite
    6 execution infrastructure failure
    7 authorization / security-policy denial
"""
from __future__ import annotations

from typing import Any


class KernelMemoryError(Exception):
    """Base class. ``code`` is stable; ``details`` is JSON-serialisable context."""

    exit_code: int = 1
    code: str = "KERNEL_MEMORY_ERROR"

    def __init__(self, message: str, *, code: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "exit_code": self.exit_code,
            "details": self.details,
        }


# --- exit code 2: input / schema -------------------------------------------------
class InputError(KernelMemoryError):
    exit_code = 2
    code = "INPUT_ERROR"


class SchemaValidationError(InputError):
    code = "SCHEMA_INVALID"


class CanonicalizationError(InputError):
    code = "CANONICALIZATION_ERROR"


class InvariantViolation(InputError):
    """A cross-record or business invariant failed (deep validation)."""

    code = "INVARIANT_VIOLATION"


# --- exit code 3: reference / idempotency conflicts -------------------------------
class ConflictError(KernelMemoryError):
    exit_code = 3
    code = "CONFLICT"


class IdConflictError(ConflictError):
    """Same record ID, different content."""

    code = "ID_CONFLICT"


class IdempotencyConflictError(ConflictError):
    """Same idempotency key, different request content."""

    code = "IDEMPOTENCY_CONFLICT"


class MissingReferenceError(ConflictError):
    code = "MISSING_REFERENCE"


class ImmutableRecordError(ConflictError):
    code = "IMMUTABLE_RECORD"


# --- exit code 4: incomparable / insufficient evidence ----------------------------
class NotComparableError(KernelMemoryError):
    exit_code = 4
    code = "NOT_COMPARABLE"


class InsufficientEvidenceError(KernelMemoryError):
    exit_code = 4
    code = "INSUFFICIENT_EVIDENCE"


# --- exit code 5: unavailable backend / prerequisite -----------------------------
class PrerequisiteMissingError(KernelMemoryError):
    exit_code = 5
    code = "PREREQUISITE_MISSING"


class BackendUnavailable(PrerequisiteMissingError):
    code = "BACKEND_UNAVAILABLE"


class UnsupportedFormat(PrerequisiteMissingError):
    code = "UNSUPPORTED_FORMAT"


class IncompleteProblemContract(PrerequisiteMissingError):
    """A kernel problem schema/normalizer is not complete enough to register configs."""

    code = "INCOMPLETE_PROBLEM_CONTRACT"


# --- exit code 6: execution infrastructure ---------------------------------------
class ExecutionInfrastructureError(KernelMemoryError):
    exit_code = 6
    code = "EXECUTION_INFRASTRUCTURE_ERROR"


class LeaseLostError(ExecutionInfrastructureError):
    code = "LEASE_LOST"


# --- exit code 7: authorization / security ---------------------------------------
class SecurityPolicyError(KernelMemoryError):
    exit_code = 7
    code = "SECURITY_POLICY_DENIED"


class AuthorizationError(SecurityPolicyError):
    code = "AUTHORIZATION_REQUIRED"


class UnsafePathError(SecurityPolicyError):
    code = "UNSAFE_PATH"


class BudgetExhausted(KernelMemoryError):
    """Not an error in the failure sense; raised to stop loops deterministically."""

    exit_code = 0
    code = "BUDGET_EXHAUSTED"
