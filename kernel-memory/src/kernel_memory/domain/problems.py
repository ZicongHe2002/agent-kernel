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
from .schema import demo_problem_schema, mla_decode_problem_schema, validate_against


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


class MlaDecodeProblem:
    """Fixed ABI for the project2 MLA decode measurements.

    The adapter deliberately keeps tensor bytes out of a config.  A config stores the
    logical and physical signatures; concrete input/output arrays belong to run artifacts.
    """

    kernel_id = "mla_decode"
    schema_id = "urn:kernel-memory:problem:mla-decode:v1"
    normalizer_version = "mla-decode-v1"
    status = "complete"
    INTERFACE_ALIASES = {
        "contiguous": "contiguous_native",
        "contiguous_native": "contiguous_native",
        "paged": "paged_native",
        "paged_native": "paged_native",
        "dense": "dense_reference",
        "dense_reference": "dense_reference",
    }
    ALLOWED_KEYS = {
        "operation", "interface", "batch", "num_q_heads", "query_tokens_per_request", "latent_dim", "rope_dim",
        "q_dim", "cache_feature_dim", "capacity", "timed_length", "dtype", "score_dtype", "probability_dtype",
        "block_size", "sm_scale", "sliding_window", "soft_cap", "cache_layout", "physical_cache_shape",
        "length_semantics", "input_signature", "output_signature", "input_generation", "page_size",
        "pages_per_block", "block_kv", "decode_batch_size", "transpose_cache", "two_step",
    }

    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(mla_decode_problem_schema())

    def schema_digest(self) -> str:
        return problem_schema_digest(mla_decode_problem_schema())

    @staticmethod
    def _int(value: Any, name: str, *, minimum: int = 0) -> int:
        return _require_int(value, name, minimum=minimum)

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise InputError("MLA decode problem must be a JSON object", code="INVALID_PROBLEM")
        unknown = set(raw) - self.ALLOWED_KEYS
        if unknown:
            raise InputError(f"unknown MLA decode problem fields: {sorted(unknown)}", code="INVALID_PROBLEM")
        missing = self.ALLOWED_KEYS - set(raw)
        if missing:
            raise InputError(f"missing MLA decode problem fields: {sorted(missing)}", code="INVALID_PROBLEM")
        if raw.get("operation") != "mla_decode":
            raise InputError("operation must be 'mla_decode'", code="INVALID_PROBLEM")
        interface_raw = raw.get("interface")
        if not isinstance(interface_raw, str) or interface_raw.lower() not in self.INTERFACE_ALIASES:
            raise InputError(f"unsupported MLA decode interface {interface_raw!r}", code="INVALID_PROBLEM")
        interface = self.INTERFACE_ALIASES[interface_raw.lower()]
        batch = self._int(raw["batch"], "batch", minimum=1)
        heads = self._int(raw["num_q_heads"], "num_q_heads", minimum=1)
        latent = self._int(raw["latent_dim"], "latent_dim", minimum=1)
        rope = self._int(raw["rope_dim"], "rope_dim", minimum=0)
        q_dim = self._int(raw["q_dim"], "q_dim", minimum=1)
        if q_dim != latent + rope:
            raise InputError(f"q_dim must equal latent_dim + rope_dim ({latent + rope}), got {q_dim}", code="INVALID_PROBLEM")
        capacity = self._int(raw["capacity"], "capacity", minimum=1)
        timed_length = self._int(raw["timed_length"], "timed_length", minimum=0)
        if timed_length > capacity:
            raise InputError("timed_length cannot exceed capacity", code="INVALID_PROBLEM")
        block_size = self._int(raw["block_size"], "block_size", minimum=1)
        cache_feature = self._int(raw["cache_feature_dim"], "cache_feature_dim", minimum=1)
        page_size = raw["page_size"]
        pages_per_block = raw["pages_per_block"]
        block_kv = raw["block_kv"]
        decode_batch_size = raw["decode_batch_size"]
        for name, value in (("page_size", page_size), ("pages_per_block", pages_per_block), ("block_kv", block_kv), ("decode_batch_size", decode_batch_size)):
            if value is not None:
                self._int(value, name, minimum=1)
        if interface == "paged_native":
            if page_size is None or pages_per_block is None or block_kv is None or decode_batch_size is None:
                raise InputError("paged_native requires page_size, pages_per_block, block_kv and decode_batch_size", code="INVALID_PROBLEM")
            if page_size % 2:
                raise InputError("paged_native page_size must be even for the packed [page/2, 2, feature] layout", code="INVALID_PROBLEM")
            expected_shape = [batch * ((capacity + page_size - 1) // page_size), page_size // 2, 2, cache_feature]
            expected_layout = "paged"
        elif interface == "contiguous_native":
            expected_shape = [batch, capacity, cache_feature]
            expected_layout = "contiguous"
        else:
            expected_shape = [batch, capacity, cache_feature]
            expected_layout = "dense"
        if raw["cache_layout"] != expected_layout:
            raise InputError(f"cache_layout must be {expected_layout!r} for {interface}", code="INVALID_PROBLEM")
        shape = raw["physical_cache_shape"]
        if not isinstance(shape, list) or shape != expected_shape:
            raise InputError(f"physical_cache_shape must be {expected_shape} for {interface}", code="INVALID_PROBLEM")
        generation = raw["input_generation"]
        if not isinstance(generation, dict) or generation.get("distribution") != "standard_normal" or generation.get("quantization") != "exact_bfloat16":
            raise InputError("input_generation must use standard_normal and exact_bfloat16", code="INVALID_PROBLEM")
        self._int(generation.get("seed"), "input_generation.seed", minimum=0)
        inputs = raw["input_signature"]
        outputs = raw["output_signature"]
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise InputError("input_signature and output_signature must be objects", code="INVALID_PROBLEM")
        expected_inputs = {"q", "new_kv", "cache", "lengths"} if interface != "dense_reference" else {"q", "kv"}
        expected_outputs = {"attention", "updated_cache"} if interface != "dense_reference" else {"output"}
        if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
            raise InputError(f"signature names do not match interface {interface}", code="INVALID_PROBLEM")
        problem = copy.deepcopy(raw)
        # Accept the compact layout labels used by project2's test fixtures,
        # then emit one canonical spelling in the normalized identity.
        paged_layout_aliases = {"paged packed cache", "P64x2F"}
        if isinstance(page_size, int) and page_size > 0 and page_size % 2 == 0:
            paged_layout_aliases.add(f"P{page_size // 2}x2F")
        layout_aliases = {
            "[B,H,q_dim]": {"[B,H,q_dim]", "BHQ"},
            "[B,q_dim]": {"[B,q_dim]", "BQ"},
            "paged packed cache": paged_layout_aliases,
            "[B]": {"[B]", "B"},
            "[B,H,latent_dim]": {"[B,H,latent_dim]", "BHL"},
            "same as cache": {"same as cache", "P64x2F"},
            "[B,capacity,feature]": {"[B,capacity,feature]", "BCF"},
        }
        expected_layouts = {
            "q": "[B,H,q_dim]", "new_kv": "[B,q_dim]", "lengths": "[B]",
            "cache": "paged packed cache" if interface == "paged_native" else "[B,capacity,feature]",
        }
        expected_output_layouts = {
            "attention": "[B,H,latent_dim]",
            "updated_cache": "same as cache",
        }
        for signature, expected in ((problem["input_signature"], expected_layouts), (problem["output_signature"], expected_output_layouts)):
            for name, canonical in expected.items():
                tensor = signature.get(name)
                if isinstance(tensor, dict) and tensor.get("layout") in layout_aliases[canonical]:
                    tensor["layout"] = canonical
        problem.update({
            "operation": "mla_decode", "interface": interface, "batch": batch, "num_q_heads": heads,
            "query_tokens_per_request": 1, "latent_dim": latent, "rope_dim": rope, "q_dim": q_dim,
            "capacity": capacity, "timed_length": timed_length, "cache_feature_dim": cache_feature,
            "dtype": "bfloat16", "score_dtype": "float32", "probability_dtype": "bfloat16",
            "block_size": block_size, "page_size": page_size, "pages_per_block": pages_per_block,
            "block_kv": block_kv, "decode_batch_size": decode_batch_size,
            "length_semantics": "append_then_attend", "cache_layout": expected_layout,
            "physical_cache_shape": expected_shape,
        })
        self.validate(problem)
        return problem

    def validate(self, problem: dict[str, Any]) -> None:
        validate_against(mla_decode_problem_schema(), problem, what="MLA decode problem")
        # JSON Schema can constrain the presence of signature entries, but the
        # dimensions are cross-field values (batch/capacity/etc.).  Keep those
        # ABI checks here so a direct ``validate`` call is just as strict as
        # ``normalize`` and a hand-written bundle cannot smuggle in a different
        # tensor contract.
        interface = problem["interface"]
        batch = problem["batch"]
        heads = problem["num_q_heads"]
        q_dim = problem["q_dim"]
        latent = problem["latent_dim"]
        physical_cache_shape = problem["physical_cache_shape"]

        def require_tensor(
            signature: Any,
            name: str,
            *,
            dtype: str,
            shape: list[int] | None = None,
            layout: str | None = None,
        ) -> None:
            if not isinstance(signature, dict) or name not in signature:
                raise InputError(f"missing MLA decode tensor signature {name!r}", code="INVALID_PROBLEM")
            tensor = signature[name]
            if not isinstance(tensor, dict):
                raise InputError(f"tensor signature {name!r} must be an object", code="INVALID_PROBLEM")
            if tensor.get("dtype") != dtype:
                raise InputError(f"tensor {name!r} must use dtype {dtype!r}", code="INVALID_PROBLEM")
            if shape is not None and tensor.get("shape") != shape:
                raise InputError(f"tensor {name!r} must have shape {shape}", code="INVALID_PROBLEM")
            accepted_layouts = {
                "[B,H,q_dim]": {"[B,H,q_dim]", "BHQ"},
                "[B,q_dim]": {"[B,q_dim]", "BQ"},
                "paged packed cache": {"paged packed cache", "P64x2F"},
                "[B]": {"[B]", "B"},
                "[B,H,latent_dim]": {"[B,H,latent_dim]", "BHL"},
                "same as cache": {"same as cache", "P64x2F"},
                "[B,capacity,feature]": {"[B,capacity,feature]", "BCF"},
            }
            if interface == "paged_native" and isinstance(physical_cache_shape, list) and len(physical_cache_shape) >= 2:
                page_alias = f"P{physical_cache_shape[1]}x2F"
                accepted_layouts["paged packed cache"].add(page_alias)
                accepted_layouts["same as cache"].add(page_alias)
            if layout is not None and tensor.get("layout") not in accepted_layouts.get(layout, {layout}):
                raise InputError(f"tensor {name!r} must use layout {layout!r}", code="INVALID_PROBLEM")

        inputs = problem["input_signature"]
        outputs = problem["output_signature"]
        if interface in {"contiguous_native", "paged_native"}:
            require_tensor(inputs, "q", dtype="bfloat16", shape=[batch, heads, q_dim], layout="[B,H,q_dim]")
            require_tensor(inputs, "new_kv", dtype="bfloat16", shape=[batch, q_dim], layout="[B,q_dim]")
            cache_layout = "paged packed cache" if interface == "paged_native" else "[B,capacity,feature]"
            require_tensor(inputs, "cache", dtype="bfloat16", shape=physical_cache_shape, layout=cache_layout)
            require_tensor(inputs, "lengths", dtype="int32", shape=[batch], layout="[B]")
            require_tensor(outputs, "attention", dtype="bfloat16", shape=[batch, heads, latent], layout="[B,H,latent_dim]")
            require_tensor(outputs, "updated_cache", dtype="bfloat16", shape=physical_cache_shape, layout="same as cache")
        else:
            # The dense reference keeps its own two-input/output signature.  Its
            # buffer dimensions vary with the reference implementation, but the
            # element type remains part of this fixed contract.
            require_tensor(inputs, "q", dtype="bfloat16")
            require_tensor(inputs, "kv", dtype="bfloat16")
            require_tensor(outputs, "output", dtype="bfloat16")

    def config_id_hint(self, problem: dict[str, Any]) -> str:
        interface = str(problem["interface"]).replace("_native", "")
        return f"mla-{interface}-b{problem['batch']}-h{problem['num_q_heads']}-n{problem['capacity']}-l{problem['latent_dim']}-r{problem['rope_dim']}"


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
    registry.register(MlaDecodeProblem())
    registry.register(MlaForwardProblem())
    return registry
