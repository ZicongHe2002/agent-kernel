"""Default adapter registry: the project's kernel and analysis adapters, wired honestly.

Specification section 17. The registry never substitutes one backend for another: a
kernel adapter whose module cannot be imported is *skipped* and the reason recorded in
``UNAVAILABLE`` (backend -> reason), so the runner reports ``BACKEND_UNAVAILABLE`` (exit 5)
instead of quietly executing on a different backend. Constructing the registry imports
no accelerator framework: ``JaxAdapter`` defers ``import jax`` to ``check_environment``,
and the CPU demo adapter's numpy import is guarded inside its own module.

Public API
----------
``default_adapter_registry(backend=None) -> AdapterRegistry``
    ``backend=None`` registers every known kernel adapter (``mock``, ``cpu``, ``jax_tpu``)
    that can be imported plus the analysis adapters (``MockSpillAnalysisAdapter``,
    ``LloAnalysisAdapter``). A named ``backend`` registers only that backend's kernel
    adapter (plus the analysis adapters); an unknown backend yields an empty kernel
    registry, so ``registry.kernel_adapter(backend)`` raises ``BackendUnavailable``.
``available_backends(registry) -> list[str]``  the registry's sorted kernel backends.
``KERNEL_ADAPTERS``  backend -> (module, class name) for the project's kernel adapters.
``UNAVAILABLE``  module-level dict of backends whose adapter module failed to import.
"""
from __future__ import annotations

import importlib
from typing import Any

from ..domain.errors import InvariantViolation
from .analysis import LloAnalysisAdapter, MockSpillAnalysisAdapter
from .base import AdapterRegistry

KERNEL_ADAPTERS: dict[str, tuple[str, str]] = {
    "mock": ("kernel_memory.adapters.mock", "MockAdapter"),
    "cpu": ("kernel_memory.adapters.cpu_demo", "CpuDemoAdapter"),
    "jax_tpu": ("kernel_memory.adapters.jax_tpu", "JaxAdapter"),
}

UNAVAILABLE: dict[str, str] = {}


def _load_kernel_adapter(backend: str) -> Any | None:
    """Instantiate the kernel adapter for ``backend``; ``None`` (and an ``UNAVAILABLE`` entry) when its module is missing."""
    module_name, class_name = KERNEL_ADAPTERS[backend]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        UNAVAILABLE[backend] = f"{module_name} could not be imported: {type(exc).__name__}: {exc}"
        return None
    adapter_cls = getattr(module, class_name, None)
    if adapter_cls is None:
        UNAVAILABLE[backend] = f"{module_name} defines no {class_name}"
        return None
    adapter = adapter_cls()
    if getattr(adapter, "backend", None) != backend:
        raise InvariantViolation(
            f"{module_name}.{class_name} labels itself backend {getattr(adapter, 'backend', None)!r}, expected {backend!r}",
            code="ADAPTER_BACKEND_MISMATCH",
            details={"backend": backend, "adapter_backend": getattr(adapter, "backend", None)},
        )
    UNAVAILABLE.pop(backend, None)
    return adapter


def analysis_adapters() -> list[Any]:
    """The project's analysis adapters (fresh instances)."""
    return [MockSpillAnalysisAdapter(), LloAnalysisAdapter()]


def default_adapter_registry(backend: str | None = None) -> AdapterRegistry:
    """Build the project's adapter registry; see the module docstring for the backend rules."""
    registry = AdapterRegistry()
    if backend is None:
        backends = list(KERNEL_ADAPTERS)
    elif backend in KERNEL_ADAPTERS:
        backends = [backend]
    else:
        backends = []  # unknown backend: nothing registered, so lookups raise BackendUnavailable
    for name in backends:
        adapter = _load_kernel_adapter(name)
        if adapter is not None:
            registry.register_kernel_adapter(adapter)
    for adapter in analysis_adapters():
        registry.register_analysis_adapter(adapter)
    return registry


def available_backends(registry: AdapterRegistry) -> list[str]:
    return registry.backends()
