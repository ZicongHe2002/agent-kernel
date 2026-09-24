"""Default adapter registry: honest backend availability, no accelerator import at construction."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernel_memory.adapters import registry as registry_module
from kernel_memory.adapters.analysis import LloAnalysisAdapter, MockSpillAnalysisAdapter
from kernel_memory.adapters.base import AdapterRegistry, AnalysisAdapter, KernelAdapter
from kernel_memory.adapters.registry import UNAVAILABLE, available_backends, default_adapter_registry
from kernel_memory.domain.errors import BackendUnavailable, InvariantViolation
from kernel_memory.domain.models import Record
from kernel_memory.services import optimize
from kernel_memory.storage import MemoryStore

from conftest import ARTIFACT_ROOT, import_demo_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_zero_arg_registry_has_every_project_backend_and_both_analysis_adapters() -> None:
    registry = default_adapter_registry()
    assert isinstance(registry, AdapterRegistry)
    assert registry.backends() == ["cpu", "jax_tpu", "mock"]
    assert available_backends(registry) == ["cpu", "jax_tpu", "mock"]
    analysis = registry.analysis_adapters()
    assert [a.adapter_id for a in analysis] == ["llo-unsupported", "mock-spill"]
    assert isinstance(analysis[0], LloAnalysisAdapter) and isinstance(analysis[1], MockSpillAnalysisAdapter)
    for adapter in analysis:
        assert isinstance(adapter, AnalysisAdapter)
    for backend in registry.backends():
        adapter = registry.kernel_adapter(backend)
        assert isinstance(adapter, KernelAdapter)
        assert adapter.backend == backend  # never relabelled
    assert registry.kernel_adapter("mock").adapter_id == "mock-v1"
    assert registry.kernel_adapter("cpu").adapter_id == "cpu-demo-v1"
    assert registry.kernel_adapter("jax_tpu").adapter_id == "jax-tpu-v0"
    assert UNAVAILABLE == {}


def test_named_backend_registers_only_that_kernel_adapter() -> None:
    registry = default_adapter_registry("mock")
    assert registry.backends() == ["mock"]
    assert [a.adapter_id for a in registry.analysis_adapters()] == ["llo-unsupported", "mock-spill"]
    with pytest.raises(BackendUnavailable) as info:
        registry.kernel_adapter("cpu")
    assert info.value.details == {"backend": "cpu", "available": ["mock"]}
    assert default_adapter_registry("cpu").backends() == ["cpu"]
    assert default_adapter_registry("jax_tpu").backends() == ["jax_tpu"]


def test_unknown_backend_yields_empty_kernel_registry_and_exit_5() -> None:
    registry = default_adapter_registry("nope")
    assert registry.backends() == []
    assert [a.adapter_id for a in registry.analysis_adapters()] == ["llo-unsupported", "mock-spill"]
    with pytest.raises(BackendUnavailable) as info:
        registry.kernel_adapter("nope")
    assert info.value.exit_code == 5 and info.value.code == "BACKEND_UNAVAILABLE"
    assert info.value.details == {"backend": "nope", "available": []}
    assert "nope" not in UNAVAILABLE  # unknown is not the same as unimportable


def test_unimportable_adapter_module_is_skipped_and_recorded_never_substituted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry_module.KERNEL_ADAPTERS, "cpu", ("kernel_memory.adapters.does_not_exist", "CpuDemoAdapter"))
    registry = default_adapter_registry()
    assert registry.backends() == ["jax_tpu", "mock"]
    assert "cpu" in UNAVAILABLE and "does_not_exist" in UNAVAILABLE["cpu"]
    with pytest.raises(BackendUnavailable):
        registry.kernel_adapter("cpu")
    # A later successful construction clears the stale reason.
    monkeypatch.undo()
    assert default_adapter_registry("cpu").backends() == ["cpu"]
    assert "cpu" not in UNAVAILABLE


def test_module_lacking_the_adapter_class_is_recorded_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry_module.KERNEL_ADAPTERS, "mock", ("kernel_memory.adapters.mock", "NoSuchAdapter"))
    registry = default_adapter_registry("mock")
    assert registry.backends() == []
    assert "NoSuchAdapter" in UNAVAILABLE["mock"]
    monkeypatch.undo()
    default_adapter_registry("mock")
    assert "mock" not in UNAVAILABLE


def test_adapter_labelled_with_another_backend_is_an_invariant_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CPU demo adapter class under the "mock" key would relabel the backend; the registry refuses.
    monkeypatch.setitem(registry_module.KERNEL_ADAPTERS, "mock", ("kernel_memory.adapters.cpu_demo", "CpuDemoAdapter"))
    with pytest.raises(InvariantViolation) as info:
        default_adapter_registry("mock")
    assert info.value.code == "ADAPTER_BACKEND_MISMATCH"


def test_constructing_the_registry_never_imports_jax() -> None:
    if not PYTHON.exists():
        pytest.skip(f"project interpreter {PYTHON} is missing; cannot check a fresh process")
    code = (
        "import sys, kernel_memory.adapters.registry as r\n"
        "reg = r.default_adapter_registry()\n"
        "print(__import__('json').dumps({'jax': 'jax' in sys.modules, 'backends': reg.backends()}))\n"
    )
    proc = subprocess.run(
        [str(PYTHON), "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"jax": False, "backends": ["cpu", "jax_tpu", "mock"]}


def test_services_optimize_default_registry_delegates() -> None:
    registry = optimize.default_adapter_registry("mock")
    assert registry.backends() == ["mock"]
    assert registry.kernel_adapter("mock").adapter_id == "mock-v1"
    assert [a.adapter_id for a in registry.analysis_adapters()] == ["llo-unsupported", "mock-spill"]
    with pytest.raises(BackendUnavailable) as info:
        optimize.default_adapter_registry("nope").kernel_adapter("nope")
    assert info.value.exit_code == 5


# ------------------------------------------------------------------------------ CLI end to end
def test_cli_run_with_mock_backend_publishes_a_trusted_worker_run(
    tmp_path: Path, bundle_records: list[Record], capsys: pytest.CaptureFixture[str]
) -> None:
    from kernel_memory.cli.main import main

    root = tmp_path / "memory"
    store = MemoryStore.init(root)
    import_demo_bundle(store, bundle_records, ARTIFACT_ROOT)

    code = main(["--root", str(root), "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "request-reg-1", "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads(captured.out.strip().splitlines()[-1])
    assert result["run"] is not None
    run = result["run"]
    assert run["record_id"] == "run-request-reg-1-a1"
    assert run["provenance"] == "trusted_worker"
    assert run["execution_status"] == "succeeded" and run["failure_reason"] is None
    assert run["environment_backend"] == "mock"
    assert result["request"]["request_id"] == "request-reg-1"

    reopened = MemoryStore.open(root)
    record = reopened.get("run-request-reg-1-a1")
    assert record is not None and record.payload.provenance == "trusted_worker"
    metrics = {m.name: m for m in record.payload.analysis_metrics}
    # The mock adapter emits no spill report: the static estimate is not collected, never 0.
    assert metrics["register_spill_vmem_static_bytes"].status == "not_collected"
    assert metrics["register_spill_vmem_static_bytes"].value is None


def test_cli_run_with_unknown_backend_is_denied_before_the_registry_is_consulted(
    tmp_path: Path, bundle_records: list[Record], capsys: pytest.CaptureFixture[str]
) -> None:
    """Runner order: authorization (``allow_<backend>_execution``) precedes adapter lookup (exit 7, no Run)."""
    from kernel_memory.cli.main import main

    root = tmp_path / "memory"
    store = MemoryStore.init(root)
    import_demo_bundle(store, bundle_records, ARTIFACT_ROOT)

    code = main(["--root", str(root), "run", "--subject", "baseline-demo", "--backend", "nope", "--request-id", "request-reg-2", "--json"])
    captured = capsys.readouterr()
    assert code == 7
    error = json.loads(captured.err.strip().splitlines()[-1])
    # DESIGN section 7: a denied action is an AuthorizationError (exit 7); the runner names the
    # denied flag in its code rather than using the class default AUTHORIZATION_REQUIRED.
    assert error["exit_code"] == 7
    assert error["error"] == "BACKEND_EXECUTION_NOT_AUTHORIZED"
    assert "allow_nope_execution" in json.dumps(error)
    assert MemoryStore.open(root).get("run-request-reg-2-a1") is None  # a denial is never a Run


def test_cli_run_with_unimportable_adapter_reports_backend_unavailable_exit_5(
    tmp_path: Path, bundle_records: list[Record], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An authorised local backend whose adapter cannot be imported is reported, never substituted (exit 5, no Run)."""
    from kernel_memory.cli.main import main

    root = tmp_path / "memory"
    store = MemoryStore.init(root)
    import_demo_bundle(store, bundle_records, ARTIFACT_ROOT)
    monkeypatch.setitem(registry_module.KERNEL_ADAPTERS, "mock", ("kernel_memory.adapters.does_not_exist", "MockAdapter"))

    code = main(["--root", str(root), "run", "--subject", "baseline-demo", "--backend", "mock", "--request-id", "request-reg-3", "--json"])
    captured = capsys.readouterr()
    assert code == 5, captured.err
    result = json.loads(captured.out.strip().splitlines()[-1])
    assert result["request_id"] == "request-reg-3"
    assert "BACKEND_UNAVAILABLE" in json.dumps(result)
    assert "does_not_exist" in UNAVAILABLE["mock"]
    assert MemoryStore.open(root).get("run-request-reg-3-a1") is None  # an unavailable backend is not an execution
