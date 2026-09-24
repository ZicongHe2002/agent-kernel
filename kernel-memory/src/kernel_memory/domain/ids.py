"""Identifiers, filesystem slugs, timestamps, and safe path resolution."""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .errors import InputError, UnsafePathError

RECORD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,120}$")
_UNSAFE_CHARS = re.compile(r"[^a-z0-9._-]+")


def validate_record_id(value: str, *, what: str = "record_id") -> str:
    # fullmatch: `$` in re.match would accept a trailing newline ("a\n"), which must be rejected.
    if not isinstance(value, str) or not RECORD_ID_PATTERN.fullmatch(value):
        raise InputError(f"invalid {what}: {value!r}", code="INVALID_ID")
    return value


def slug_for_id(value: str) -> str:
    """Deterministic, case-insensitive-filesystem-safe directory name for an identifier.

    If the identifier is already a safe lower-case slug it is used verbatim. Otherwise the
    lower-cased sanitised form gets an 8-hex suffix derived from the exact original, so
    ``Run-A`` and ``run-a`` (or ``a:b`` and ``a_b``) never collide.
    """
    if not isinstance(value, str) or not value:
        raise InputError("cannot slugify an empty identifier", code="INVALID_ID")
    lowered = value.lower()
    if _SAFE_SLUG.fullmatch(lowered) and lowered == value and ".." not in value:
        return value
    sanitized = _UNSAFE_CHARS.sub("-", lowered).strip("-.") or "id"
    sanitized = sanitized[:100]
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}-{suffix}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str, *, what: str = "created_at") -> datetime:
    """Parse an RFC 3339 timestamp and require timezone awareness."""
    if not isinstance(value, str):
        raise InputError(f"{what} must be a string timestamp", code="INVALID_TIMESTAMP")
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise InputError(f"{what} is not an RFC 3339 timestamp: {value!r}", code="INVALID_TIMESTAMP") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputError(f"{what} must be timezone-aware: {value!r}", code="INVALID_TIMESTAMP")
    return parsed.astimezone(timezone.utc)


def new_uuid_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def short_hash_id(prefix: str, *parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` and refuse traversal, absolute paths, and symlink escapes."""
    if not isinstance(relative, str) or not relative:
        raise UnsafePathError("empty path")
    if "\x00" in relative:
        raise UnsafePathError("path contains NUL byte")
    candidate = Path(relative)
    if candidate.is_absolute() or relative.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", relative):
        raise UnsafePathError(f"absolute paths are not permitted: {relative!r}")
    if any(part == ".." for part in candidate.parts):
        raise UnsafePathError(f"path traversal is not permitted: {relative!r}")
    root_resolved = Path(root).resolve()
    target = (root_resolved / candidate).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafePathError(f"path escapes the permitted root: {relative!r}") from exc
    # Reject symlinks in any component below the root (dangerous symlink escapes).
    probe = root_resolved
    for part in candidate.parts:
        probe = probe / part
        if probe.is_symlink():
            raise UnsafePathError(f"symlinked path component is not permitted: {relative!r}")
    return target


def fsync_directory(path: Path) -> None:
    """Best-effort directory fsync (not supported on every platform/filesystem)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
