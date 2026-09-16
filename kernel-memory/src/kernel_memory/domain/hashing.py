"""Identity hashing (specification sections 5, 6, 12).

All identity hashes are ``sha256:<hex>`` over the RFC 8785 canonical form of a
versioned identity object. The rules here are locked by golden vectors.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .jcs import canonicalize

HASH_VERSION = "jcs-sha256-v1"
SHA256_PREFIX = "sha256:"


def sha256_bytes(data: bytes) -> str:
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def jcs_digest(value: Any) -> str:
    return sha256_bytes(canonicalize(value))


def problem_schema_digest(problem_schema: dict[str, Any]) -> str:
    return jcs_digest(problem_schema)


def config_hash(*, kernel_id: str, problem_schema_id: str, problem_schema_digest: str, problem: dict[str, Any]) -> str:
    return jcs_digest(
        {
            "hash_version": HASH_VERSION,
            "kernel_id": kernel_id,
            "problem_schema_id": problem_schema_id,
            "problem_schema_digest": problem_schema_digest,
            "problem": problem,
        }
    )


def variant_digest(*, source_digest: str, entrypoint: str, implementation_overrides: dict[str, Any], checkout_mode: str) -> str:
    return jcs_digest(
        {
            "source_digest": source_digest,
            "entrypoint": entrypoint,
            "implementation_overrides": implementation_overrides,
            "checkout_mode": checkout_mode,
        }
    )


def snapshot_hash(snapshot: dict[str, Any], hash_field: str) -> str:
    """Hash of a complete environment/protocol/verifier snapshot with its own hash field removed."""
    return jcs_digest({k: v for k, v in snapshot.items() if k != hash_field})


def environment_hash(environment: dict[str, Any]) -> str:
    return snapshot_hash(environment, "environment_hash")


def protocol_hash(protocol: dict[str, Any]) -> str:
    return snapshot_hash(protocol, "protocol_hash")


def verifier_hash(verifier: dict[str, Any]) -> str:
    return snapshot_hash(verifier, "verifier_hash")


def comparison_key(*, config_hash: str, environment_hash: str, protocol_hash: str, verifier_hash: str, checkout_mode: str) -> str:
    return jcs_digest(
        {
            "config_hash": config_hash,
            "environment_hash": environment_hash,
            "protocol_hash": protocol_hash,
            "verifier_hash": verifier_hash,
            "checkout_mode": checkout_mode,
        }
    )


def policy_hash(policy: dict[str, Any]) -> str:
    return jcs_digest(policy)


def artifact_digest(data: bytes) -> str:
    """Artifact digests cover the original bytes, never a re-serialised object."""
    return sha256_bytes(data)


def source_digest(manifest: Iterable[tuple[str, str]]) -> str:
    """Digest of a source manifest: sorted (relative path, sha256-of-bytes) pairs.

    Compute this from the project source actually loaded, not from candidate-provided metadata.
    """
    entries = sorted((str(path), str(digest)) for path, digest in manifest)
    return jcs_digest({"source_manifest_version": 1, "files": [list(e) for e in entries]})


def is_sha256_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(SHA256_PREFIX) and len(value) == 71 and all(
        c in "0123456789abcdef" for c in value[7:]
    )
