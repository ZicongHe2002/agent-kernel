"""Problem schemas and normalizers (specification section 5).

A Config captures *what* to compute. Each kernel provides a problem schema and a
normalizer that fills semantically defined defaults, canonicalises permitted
aliases, validates types, and refuses to guess unknown semantics.

* ``DemoVectorAddProblem`` is the CPU demonstration operator (not an MLA contract).
* ``MlaForwardProblem`` is deliberately incomplete: the real callable, complete
  tensor signature, output contract (O vs O+LSE), scope (full forward vs
  attention-only), reference implementation and tolerances are unknown. It
  refuses to normalise anything until a backend-specific schema derived from the
  real repository is supplied. This is the explicit "cannot be registered" path.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol as TypingProtocol

from .errors import IncompleteProblemContract, InputError, SchemaValidationError
from .hashing import config_hash as compute_config_hash
from .hashing import problem_schema_digest
from .schema import demo_problem_schema, validate_against


class ProblemAdapter(TypingProtocol):
    kernel_id: str
    schema_id: str

    def schema(self) -> dict[str, Any]: ...

    def schema_digest(self) -> str: ...

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]: ...

    def validate(self, problem: dict[str, Any]) -> None: ...

    def config_id_hint(self, problem: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class NormalizedProblem:
    kernel_id: str
    problem_schema_id: str
    problem_schema_digest: str
    problem: dict[str, Any]
    config_hash: str
    config_id_hint: str


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{name} must be an integer, got {value!r}", code="INVALID_PROBLEM")
    if minimum is not None and value < minimum:
        raise InputError(f"{name} must be >= {minimum}, got {value}", code="INVALID_PROBLEM")
    return value


class DemoVectorAddProblem:
    """Normalizer for ``urn:kernel-memory:problem:demo-vector-add:v1``."""

    kernel_id = "demo_vector_add"
    schema_id = "urn:kernel-memory:problem:demo-vector-add:v1"
    normalizer_version = "demo-vector-add-normalizer-v1"
    DTYPE_ALIASES = {"float32": "float32", "f32": "float32", "fp32": "float32"}
    ALLOWED_KEYS = {"n", "dtype", "operation", "outputs"}

    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(demo_problem_schema())

    def schema_digest(self) -> str:
        return problem_schema_digest(demo_problem_schema())

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise InputError("problem must be a JSON object", code="INVALID_PROBLEM")
        unknown = set(raw) - self.ALLOWED_KEYS
        if unknown:
            raise InputError(f"unknown problem fields: {sorted(unknown)}", code="INVALID_PROBLEM")
        if "n" not in raw:
            raise InputError("problem field 'n' is required (no default is semantically defined)", code="INVALID_PROBLEM")
        n = _require_int(raw["n"], "n", minimum=1)
        dtype_raw = raw.get("dtype", "float32")
        if not isinstance(dtype_raw, str) or dtype_raw.lower() not in self.DTYPE_ALIASES:
            raise InputError(f"unsupported dtype {dtype_raw!r}; permitted aliases: {sorted(self.DTYPE_ALIASES)}", code="INVALID_PROBLEM")
        operation = raw.get("operation", "vector_add")
        outputs = raw.get("outputs", ["y"])
        problem = {
            "n": n,
            "dtype": self.DTYPE_ALIASES[dtype_raw.lower()],
            "operation": operation,
            "outputs": list(outputs) if isinstance(outputs, list) else outputs,
        }
        self.validate(problem)
        return problem

    def validate(self, problem: dict[str, Any]) -> None:
        validate_against(demo_problem_schema(), problem, what="demo vector-add problem")

    def config_id_hint(self, problem: dict[str, Any]) -> str:
        dtype = {"float32": "f32"}.get(problem["dtype"], problem["dtype"])
        return f"demo-n{problem['n']}-{dtype}"


class MlaForwardProblem:
    """Placeholder for the real MLA forward problem contract. Not registrable yet."""

    kernel_id = "mla_forward"
    schema_id = "urn:kernel-memory:problem:mla-forward:unresolved"
    status = "incomplete"
    UNRESOLVED = (
        "Actual callable and complete tensor signature",
        "Value/output dimensions and layouts",
        "Every weight/projector shape and precision contract",
        "Causal/mask/length/scale/soft-cap semantics",
        "O-only versus O+LSE contract",
        "Full-forward versus attention-only measurement scope",
        "Trusted reference implementation and approved tolerances",
    )

    def _refuse(self) -> IncompleteProblemContract:
        return IncompleteProblemContract(
            "mla_forward problem contract is incomplete; derive the schema from the real repository "
            "before registering configs. Historical dimensions are context, not an ABI.",
            details={"kernel_id": self.kernel_id, "unresolved": list(self.UNRESOLVED)},
        )

    def schema(self) -> dict[str, Any]:
        raise self._refuse()

    def schema_digest(self) -> str:
        raise self._refuse()

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        raise self._refuse()

    def validate(self, problem: dict[str, Any]) -> None:
        raise self._refuse()

    def config_id_hint(self, problem: dict[str, Any]) -> str:
        raise self._refuse()


class ProblemRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        kernel_id = getattr(adapter, "kernel_id", None)
        if not isinstance(kernel_id, str) or not kernel_id:
            raise InputError("problem adapter must define kernel_id")
        if kernel_id in self._adapters and self._adapters[kernel_id] is not adapter:
            raise InputError(f"a problem adapter for {kernel_id!r} is already registered", code="ADAPTER_CONFLICT")
        self._adapters[kernel_id] = adapter

    def get(self, kernel_id: str) -> Any:
        try:
            return self._adapters[kernel_id]
        except KeyError as exc:
            raise IncompleteProblemContract(
                f"no problem adapter registered for kernel {kernel_id!r}", details={"kernel_id": kernel_id}
            ) from exc

    def kernel_ids(self) -> list[str]:
        return sorted(self._adapters)

    def normalize(self, kernel_id: str, raw: dict[str, Any]) -> NormalizedProblem:
        adapter = self.get(kernel_id)
        problem = adapter.normalize(raw)
        digest = adapter.schema_digest()
        return NormalizedProblem(
            kernel_id=kernel_id,
            problem_schema_id=adapter.schema_id,
            problem_schema_digest=digest,
            problem=problem,
            config_hash=compute_config_hash(
                kernel_id=kernel_id,
                problem_schema_id=adapter.schema_id,
                problem_schema_digest=digest,
                problem=problem,
            ),
            config_id_hint=adapter.config_id_hint(problem),
        )

    def verify_config_payload(self, payload: dict[str, Any]) -> None:
        """Verify a config payload's schema digest, problem validity, and config hash.

        Kernels without a registered adapter are checked for hash self-consistency only;
        their problem semantics cannot be verified and this is reported as a detail.
        """
        kernel_id = payload["kernel_id"]
        adapter = self._adapters.get(kernel_id)
        if adapter is not None and getattr(adapter, "status", "complete") != "incomplete":
            if payload["problem_schema_id"] != adapter.schema_id:
                raise SchemaValidationError(
                    f"config problem_schema_id {payload['problem_schema_id']!r} does not match adapter {adapter.schema_id!r}"
                )
            if payload["problem_schema_digest"] != adapter.schema_digest():
                raise SchemaValidationError("config problem_schema_digest does not match the registered problem schema")
            adapter.validate(payload["problem"])
            if adapter.normalize(payload["problem"]) != payload["problem"]:
                raise SchemaValidationError("config problem is not in normalized form")
        expected = compute_config_hash(
            kernel_id=kernel_id,
            problem_schema_id=payload["problem_schema_id"],
            problem_schema_digest=payload["problem_schema_digest"],
            problem=payload["problem"],
        )
        if payload["config_hash"] != expected:
            raise SchemaValidationError(
                f"config_hash mismatch: recorded {payload['config_hash']}, computed {expected}",
                details={"recorded": payload["config_hash"], "computed": expected},
            )


def default_registry() -> ProblemRegistry:
    registry = ProblemRegistry()
    registry.register(DemoVectorAddProblem())
    registry.register(MlaForwardProblem())
    return registry
