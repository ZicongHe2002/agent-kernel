"""Project settings: safety switches, external dependencies, budgets.

The settings file mirrors ``examples/project_settings.template.json`` from the handoff.
It never contains secrets: credentials are read from the environment variables *named*
in the file. Every permission defaults to the safe value (false) except local CPU tests
and project-code writes, which the launch prompt authorises.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain.errors import AuthorizationError, InputError, PrerequisiteMissingError
from .domain.jsonio import load_json_file
from .execution.types import Budget

SETTINGS_VERSION = "0.2.0"

DEFAULT_PERMISSIONS: dict[str, bool] = {
    "allow_project_code_write": True,
    "allow_local_cpu_tests": True,
    "allow_network": False,
    "allow_dependency_install": False,
    "allow_candidate_code_write": False,
    "allow_local_candidate_commit": False,
    "allow_remote_write": False,
    "allow_tpu_execution": False,
    "allow_model_api_calls": False,
}

_KNOWN_KEYS = {
    "template_only",
    "settings_version",
    "project_root",
    "memory_root",
    "source_repository",
    "kernel_id",
    "kernel_entrypoint",
    "problem_schema_path",
    "trusted_reference_entrypoint",
    "approved_verifier_path",
    "github_repository",
    "github_token_env_name",
    "model_provider",
    "model_api_key_env_name",
    "llo_adapter",
    "permissions",
    "budget",
    "notes",
}

_SECRET_LIKE = ("token", "secret", "password", "api_key", "apikey", "credential")


@dataclass(frozen=True)
class Settings:
    memory_root: Path
    project_root: Path | None
    source_repository: str | None
    kernel_id: str | None
    kernel_entrypoint: str | None
    problem_schema_path: str | None
    trusted_reference_entrypoint: str | None
    approved_verifier_path: str | None
    github_repository: str | None
    github_token_env_name: str | None
    model_provider: str | None
    model_api_key_env_name: str | None
    llo_adapter: str | None
    permissions: dict[str, bool]
    budget: Budget
    template_only: bool = False
    notes: list[str] = field(default_factory=list)
    source_path: Path | None = None

    # ------------------------------------------------------------------ permissions
    def allows(self, permission: str) -> bool:
        if permission not in DEFAULT_PERMISSIONS:
            raise InputError(f"unknown permission {permission!r}", details={"known": sorted(DEFAULT_PERMISSIONS)})
        return bool(self.permissions.get(permission, DEFAULT_PERMISSIONS[permission]))

    def require(self, permission: str, *, action: str) -> None:
        if not self.allows(permission):
            raise AuthorizationError(
                f"{action} requires permission {permission!r}, which is false in the project settings",
                details={"permission": permission, "action": action},
            )

    # ------------------------------------------------------------------ credentials (never stored)
    def github_token(self) -> str | None:
        """Read the GitHub token from the configured environment variable; never from the settings file."""
        if not self.github_token_env_name:
            return None
        return os.environ.get(self.github_token_env_name) or None

    def require_github_token(self) -> str:
        token = self.github_token()
        if token is None:
            raise PrerequisiteMissingError(
                "GitHub token is not available",
                code="GITHUB_TOKEN_MISSING",
                details={"env_var": self.github_token_env_name},
            )
        return token

    def model_api_key(self) -> str | None:
        if not self.model_api_key_env_name:
            return None
        return os.environ.get(self.model_api_key_env_name) or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_version": SETTINGS_VERSION,
            "template_only": self.template_only,
            "memory_root": str(self.memory_root),
            "project_root": str(self.project_root) if self.project_root else None,
            "source_repository": self.source_repository,
            "kernel_id": self.kernel_id,
            "kernel_entrypoint": self.kernel_entrypoint,
            "problem_schema_path": self.problem_schema_path,
            "trusted_reference_entrypoint": self.trusted_reference_entrypoint,
            "approved_verifier_path": self.approved_verifier_path,
            "github_repository": self.github_repository,
            "github_token_env_name": self.github_token_env_name,
            "model_provider": self.model_provider,
            "model_api_key_env_name": self.model_api_key_env_name,
            "llo_adapter": self.llo_adapter,
            "permissions": dict(self.permissions),
            "budget": self.budget.to_dict(),
            "notes": list(self.notes),
        }


def default_settings(memory_root: Path | str = "./memory") -> Settings:
    return Settings(
        memory_root=Path(memory_root),
        project_root=None,
        source_repository=None,
        kernel_id=None,
        kernel_entrypoint=None,
        problem_schema_path=None,
        trusted_reference_entrypoint=None,
        approved_verifier_path=None,
        github_repository=None,
        github_token_env_name="GITHUB_TOKEN",
        model_provider=None,
        model_api_key_env_name=None,
        llo_adapter=None,
        permissions=dict(DEFAULT_PERMISSIONS),
        budget=Budget(),
    )


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError(f"settings.{key} must be a string or null")
    return value


def load_settings(path: Path | str) -> Settings:
    path = Path(path)
    data = load_json_file(path, max_bytes=1_000_000)
    if not isinstance(data, dict):
        raise InputError("settings file must contain a JSON object")
    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        raise InputError(f"unknown settings keys: {sorted(unknown)}", details={"known": sorted(_KNOWN_KEYS)})
    if data.get("settings_version") not in (None, SETTINGS_VERSION):
        raise InputError(f"unsupported settings_version {data.get('settings_version')!r}")
    for key, value in data.items():
        if isinstance(value, str) and any(marker in key.lower() for marker in _SECRET_LIKE) and not key.endswith("_env_name"):
            raise InputError(f"settings key {key!r} looks like a credential; only environment-variable names belong here")
    permissions = dict(DEFAULT_PERMISSIONS)
    raw_permissions = data.get("permissions", {})
    if not isinstance(raw_permissions, dict):
        raise InputError("settings.permissions must be an object")
    for key, value in raw_permissions.items():
        if key not in DEFAULT_PERMISSIONS:
            raise InputError(f"unknown permission {key!r}", details={"known": sorted(DEFAULT_PERMISSIONS)})
        if not isinstance(value, bool):
            raise InputError(f"permission {key!r} must be a boolean")
        permissions[key] = value
    raw_budget = data.get("budget", {})
    if not isinstance(raw_budget, dict):
        raise InputError("settings.budget must be an object")
    budget_fields = {f for f in Budget.__dataclass_fields__}
    unknown_budget = set(raw_budget) - budget_fields
    if unknown_budget:
        raise InputError(f"unknown budget keys: {sorted(unknown_budget)}")
    budget = Budget(**raw_budget)
    budget.validate()
    memory_root_raw = data.get("memory_root", "./memory")
    if not isinstance(memory_root_raw, str) or not memory_root_raw:
        raise InputError("settings.memory_root must be a non-empty string")
    memory_root = Path(memory_root_raw)
    if not memory_root.is_absolute():
        memory_root = (path.parent / memory_root).resolve()
    project_root_raw = _opt_str(data, "project_root")
    project_root = None
    if project_root_raw and project_root_raw != "RESOLVE_FROM_AUTHORIZED_WORKSPACE":
        project_root = Path(project_root_raw)
    notes = data.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise InputError("settings.notes must be a list of strings")
    return Settings(
        memory_root=memory_root,
        project_root=project_root,
        source_repository=_opt_str(data, "source_repository"),
        kernel_id=_opt_str(data, "kernel_id"),
        kernel_entrypoint=_opt_str(data, "kernel_entrypoint"),
        problem_schema_path=_opt_str(data, "problem_schema_path"),
        trusted_reference_entrypoint=_opt_str(data, "trusted_reference_entrypoint"),
        approved_verifier_path=_opt_str(data, "approved_verifier_path"),
        github_repository=_opt_str(data, "github_repository"),
        github_token_env_name=_opt_str(data, "github_token_env_name") or "GITHUB_TOKEN",
        model_provider=_opt_str(data, "model_provider"),
        model_api_key_env_name=_opt_str(data, "model_api_key_env_name"),
        llo_adapter=_opt_str(data, "llo_adapter"),
        permissions=permissions,
        budget=budget,
        template_only=bool(data.get("template_only", False)),
        notes=list(notes),
        source_path=path,
    )


def pending_integrations(settings: Settings, *, tpu_available: bool = False, jax_installed: bool = False) -> list[dict[str, Any]]:
    """Enumerate integrations that cannot run with the current settings/environment and why."""
    items: list[dict[str, Any]] = []
    if settings.github_repository is None or settings.github_token() is None or not settings.allows("allow_network"):
        items.append(
            {
                "integration": "live GitHub collection",
                "status": "unexecuted",
                "missing": [
                    m
                    for m, present in (
                        ("github_repository", settings.github_repository is not None),
                        (f"env {settings.github_token_env_name}", settings.github_token() is not None),
                        ("permissions.allow_network", settings.allows("allow_network")),
                    )
                    if not present
                ],
            }
        )
    if not tpu_available or not settings.allows("allow_tpu_execution") or settings.kernel_entrypoint is None:
        items.append(
            {
                "integration": "TPU execution",
                "status": "unexecuted",
                "missing": [
                    m
                    for m, present in (
                        ("TPU device", tpu_available),
                        ("jax installed", jax_installed),
                        ("permissions.allow_tpu_execution", settings.allows("allow_tpu_execution")),
                        ("kernel_entrypoint", settings.kernel_entrypoint is not None),
                        ("problem_schema_path", settings.problem_schema_path is not None),
                        ("trusted_reference_entrypoint", settings.trusted_reference_entrypoint is not None),
                        ("approved_verifier_path", settings.approved_verifier_path is not None),
                    )
                    if not present
                ],
            }
        )
    if settings.model_provider is None or settings.model_api_key() is None or not settings.allows("allow_model_api_calls"):
        items.append(
            {
                "integration": "model-driven planner",
                "status": "unexecuted",
                "missing": [
                    m
                    for m, present in (
                        ("model_provider", settings.model_provider is not None),
                        ("model_api_key_env_name + value", settings.model_api_key() is not None),
                        ("permissions.allow_model_api_calls", settings.allows("allow_model_api_calls")),
                    )
                    if not present
                ],
            }
        )
    if settings.llo_adapter is None:
        items.append({"integration": "LLO analysis", "status": "unsupported", "missing": ["llo_adapter (format specification + sample)"]})
    if not settings.allows("allow_remote_write"):
        items.append({"integration": "remote PR write actions", "status": "disabled", "missing": ["permissions.allow_remote_write + explicit authorization"]})
    return items
