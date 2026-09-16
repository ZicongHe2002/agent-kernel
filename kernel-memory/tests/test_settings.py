"""Settings loader: safety switches, budgets, credentials-by-env-name only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_memory.domain.errors import AuthorizationError, InputError, PrerequisiteMissingError
from kernel_memory.settings import DEFAULT_PERMISSIONS, default_settings, load_settings, pending_integrations

TEMPLATE = Path(__file__).resolve().parents[1] / "fixtures" / "handoff" / "examples" / "project_settings.template.json"


def test_template_loads_with_safe_defaults() -> None:
    settings = load_settings(TEMPLATE)
    assert settings.template_only is True
    assert settings.memory_root.name == "memory"
    assert settings.allows("allow_local_cpu_tests") is True
    for permission in ("allow_network", "allow_tpu_execution", "allow_remote_write", "allow_model_api_calls", "allow_candidate_code_write"):
        assert settings.allows(permission) is False
    assert settings.budget.max_candidates == 8
    assert settings.budget.max_execution_attempts == 24
    assert settings.github_token_env_name == "GITHUB_TOKEN"


def test_require_denied_permission_is_authorization_error() -> None:
    settings = load_settings(TEMPLATE)
    with pytest.raises(AuthorizationError) as excinfo:
        settings.require("allow_remote_write", action="create a pull request")
    assert excinfo.value.exit_code == 7


def test_unknown_permission_rejected() -> None:
    settings = default_settings()
    with pytest.raises(InputError):
        settings.allows("allow_anything")


def test_credentials_come_from_environment_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"settings_version": "0.2.0", "github_token_env_name": "KM_TEST_TOKEN"}))
    settings = load_settings(path)
    monkeypatch.delenv("KM_TEST_TOKEN", raising=False)
    assert settings.github_token() is None
    with pytest.raises(PrerequisiteMissingError) as excinfo:
        settings.require_github_token()
    assert excinfo.value.exit_code == 5
    monkeypatch.setenv("KM_TEST_TOKEN", "value-from-env")
    assert settings.require_github_token() == "value-from-env"


def test_secret_like_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"settings_version": "0.2.0", "github_token": "ghp_secret"}))
    with pytest.raises(InputError):
        load_settings(path)


def test_unknown_keys_and_bad_permissions_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"settings_version": "0.2.0", "unexpected": 1}))
    with pytest.raises(InputError):
        load_settings(path)
    path.write_text(json.dumps({"settings_version": "0.2.0", "permissions": {"allow_network": "yes"}}))
    with pytest.raises(InputError):
        load_settings(path)
    path.write_text(json.dumps({"settings_version": "0.2.0", "permissions": {"allow_everything": True}}))
    with pytest.raises(InputError):
        load_settings(path)
    path.write_text(json.dumps({"settings_version": "0.2.0", "budget": {"max_candidates": -1}}))
    with pytest.raises(InputError):
        load_settings(path)


def test_relative_memory_root_resolves_against_settings_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"settings_version": "0.2.0", "memory_root": "../store"}))
    settings = load_settings(path)
    assert settings.memory_root == (tmp_path / "store").resolve()


def test_pending_integrations_name_exact_missing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    settings = load_settings(TEMPLATE)
    pending = {item["integration"]: item for item in pending_integrations(settings, tpu_available=False, jax_installed=True)}
    assert "live GitHub collection" in pending
    assert "github_repository" in pending["live GitHub collection"]["missing"]
    assert "TPU device" in pending["TPU execution"]["missing"]
    assert "permissions.allow_tpu_execution" in pending["TPU execution"]["missing"]
    assert "model-driven planner" in pending
    assert pending["LLO analysis"]["status"] == "unsupported"
    assert pending["remote PR write actions"]["status"] == "disabled"


def test_default_permissions_are_safe() -> None:
    assert DEFAULT_PERMISSIONS["allow_remote_write"] is False
    assert DEFAULT_PERMISSIONS["allow_tpu_execution"] is False
    assert DEFAULT_PERMISSIONS["allow_model_api_calls"] is False
    assert DEFAULT_PERMISSIONS["allow_candidate_code_write"] is False
