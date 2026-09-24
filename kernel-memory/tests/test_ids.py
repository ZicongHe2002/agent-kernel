"""Identifiers, filesystem slugs, timestamps and safe path resolution (DESIGN section 4, spec 19; T28)."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kernel_memory.domain import ids
from kernel_memory.domain.errors import InputError, SecurityPolicyError, UnsafePathError


# --------------------------------------------------------------------------------------
# slug_for_id
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["run-a", "cfg-demo", "a", "a.b_c-d", "0abc", "x" * 121, "run-demo-c-failure"])
def test_slug_for_id_keeps_safe_slugs_verbatim(value: str) -> None:
    assert ids.slug_for_id(value) == value


def test_slug_for_id_is_deterministic() -> None:
    assert ids.slug_for_id("Run-A") == ids.slug_for_id("Run-A")
    assert ids.slug_for_id("a:b") == ids.slug_for_id("a:b")


def test_slug_for_id_case_variants_are_distinct() -> None:
    upper = ids.slug_for_id("Run-A")
    lower = ids.slug_for_id("run-a")
    assert lower == "run-a"
    assert upper != lower
    assert upper.startswith("run-a-")
    assert re.fullmatch(r"run-a-[0-9a-f]{8}", upper)
    # Different case variants get different suffixes as well.
    assert ids.slug_for_id("RUN-A") != upper


def test_slug_for_id_colon_and_underscore_are_distinct() -> None:
    colon = ids.slug_for_id("a:b")
    underscore = ids.slug_for_id("a_b")
    assert underscore == "a_b"
    assert colon != underscore
    assert re.fullmatch(r"a-b-[0-9a-f]{8}", colon)
    assert ids.slug_for_id("a/b") != colon  # different unsafe characters, different suffix


def test_slug_for_id_output_is_always_filesystem_safe() -> None:
    for value in ("Run-A", "a:b", "a/b", "../../etc", "..", "...", "a b c", "\u00e9t\u00e9", "x" * 300, "A" * 300):
        slug = ids.slug_for_id(value)
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*", slug), slug
        assert "/" not in slug and "\\" not in slug
        assert slug not in (".", "..")
        assert len(slug) <= 121


def test_slug_for_id_dotdot_is_never_verbatim() -> None:
    assert ids.slug_for_id("a..b") != "a..b"
    assert ids.slug_for_id("..") != ".."


def test_slug_for_id_rejects_empty_or_non_string() -> None:
    for bad in ("", None, 5):
        with pytest.raises(InputError) as info:
            ids.slug_for_id(bad)  # type: ignore[arg-type]
        assert info.value.code == "INVALID_ID"


def test_slug_for_id_no_collisions_across_many_variants() -> None:
    values = ["run-a", "Run-A", "RUN-A", "run_a", "run:a", "run/a", "run a", "run.a", "run-a-", "-run-a"]
    slugs = [ids.slug_for_id(v) for v in values]
    assert len(set(slugs)) == len(values)


# --------------------------------------------------------------------------------------
# validate_record_id
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["a", "run-demo-a", "A.B_C:d-e", "0", "x" * 256, "urn:kernel-memory:problem:demo-vector-add:v1"])
def test_validate_record_id_accepts_pattern(value: str) -> None:
    assert ids.validate_record_id(value) == value
    assert ids.RECORD_ID_PATTERN.match(value)


@pytest.mark.parametrize("value", ["", "-a", ".a", "_a", ":a", "a b", "a/b", "a\\b", "x" * 257, "\u00e4", "a\n", "../x", None, 5, True])
def test_validate_record_id_rejects_pattern_violations(value: object) -> None:
    with pytest.raises(InputError) as info:
        ids.validate_record_id(value)  # type: ignore[arg-type]
    assert info.value.code == "INVALID_ID"
    assert info.value.exit_code == 2


def test_validate_record_id_names_the_field() -> None:
    with pytest.raises(InputError) as info:
        ids.validate_record_id("bad id", what="config_ref")
    assert "config_ref" in str(info.value)


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------
def test_parse_utc_timestamp_accepts_z_suffix() -> None:
    parsed = ids.parse_utc_timestamp("2026-09-08T00:00:00Z")
    assert parsed == datetime(2026, 9, 8, tzinfo=timezone.utc)
    assert parsed.tzinfo is timezone.utc
    assert ids.parse_utc_timestamp("2026-09-08T00:00:00z") == parsed
    assert ids.parse_utc_timestamp("2026-09-08T00:00:00.250Z").microsecond == 250000


def test_parse_utc_timestamp_accepts_offsets_and_normalises_to_utc() -> None:
    assert ids.parse_utc_timestamp("2026-09-08T02:00:00+02:00") == datetime(2026, 9, 8, tzinfo=timezone.utc)
    assert ids.parse_utc_timestamp("2026-09-07T19:00:00-05:00") == datetime(2026, 9, 8, tzinfo=timezone.utc)
    assert ids.parse_utc_timestamp("2026-09-08T00:00:00+00:00") == datetime(2026, 9, 8, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "value",
    ["2026-09-08T00:00:00", "2026-09-08", "2026-09-08 00:00:00", "2026-09-08T00:00:00.123"],
)
def test_parse_utc_timestamp_rejects_naive(value: str) -> None:
    with pytest.raises(InputError) as info:
        ids.parse_utc_timestamp(value)
    assert info.value.code == "INVALID_TIMESTAMP"
    assert info.value.exit_code == 2
    assert "timezone-aware" in str(info.value)


@pytest.mark.parametrize("value", ["", "yesterday", "2026-13-01T00:00:00Z", "2026-09-08T25:00:00Z", "2026-09-08T00:00:00+25:00", 12345, None])
def test_parse_utc_timestamp_rejects_garbage(value: object) -> None:
    with pytest.raises(InputError) as info:
        ids.parse_utc_timestamp(value)  # type: ignore[arg-type]
    assert info.value.code == "INVALID_TIMESTAMP"


def test_parse_utc_timestamp_names_the_field() -> None:
    with pytest.raises(InputError) as info:
        ids.parse_utc_timestamp("2026-09-08T00:00:00", what="observed_at")
    assert "observed_at" in str(info.value)


def test_utc_now_iso_is_parseable_and_utc() -> None:
    text = ids.utc_now_iso()
    assert text.endswith("Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text)
    parsed = ids.parse_utc_timestamp(text)
    assert parsed.tzinfo is timezone.utc
    assert ids.utc_now().tzinfo is timezone.utc


# --------------------------------------------------------------------------------------
# Derived identifiers
# --------------------------------------------------------------------------------------
def test_short_hash_id_is_deterministic_and_prefixed() -> None:
    a = ids.short_hash_id("run", "cfg-demo", "commit-demo-a")
    b = ids.short_hash_id("run", "cfg-demo", "commit-demo-a")
    assert a == b
    assert re.fullmatch(r"run-[0-9a-f]{16}", a)
    assert ids.validate_record_id(a) == a


def test_short_hash_id_distinguishes_part_boundaries_and_length() -> None:
    assert ids.short_hash_id("p", "ab", "c") != ids.short_hash_id("p", "a", "bc")
    assert ids.short_hash_id("p", "abc") != ids.short_hash_id("p", "ab", "c")
    assert ids.short_hash_id("p", "a", "b") != ids.short_hash_id("p", "b", "a")
    assert ids.short_hash_id("p", "a", "b") != ids.short_hash_id("q", "a", "b")
    short = ids.short_hash_id("p", "a", "b", length=8)
    assert re.fullmatch(r"p-[0-9a-f]{8}", short)
    assert ids.short_hash_id("p", "a", "b").startswith(short)


def test_new_uuid_id_is_prefixed_unique_and_valid() -> None:
    a, b = ids.new_uuid_id("req"), ids.new_uuid_id("req")
    assert a != b
    assert re.fullmatch(r"req-[0-9a-f]{32}", a)
    assert ids.validate_record_id(a) == a


# --------------------------------------------------------------------------------------
# resolve_inside (T28: malicious paths)
# --------------------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "inner").mkdir(parents=True)
    (root / "inner" / "file.txt").write_text("inside\n")
    return root


def test_resolve_inside_accepts_plain_relative_paths(root: Path) -> None:
    target = ids.resolve_inside(root, "inner/file.txt")
    assert target == (root / "inner" / "file.txt").resolve()
    assert target.read_text() == "inside\n"
    assert ids.resolve_inside(root, "./inner/file.txt") == target
    assert ids.resolve_inside(root, "inner/./file.txt") == target
    # Not-yet-existing paths under the root are fine (used for writes).
    assert ids.resolve_inside(root, "new/dir/file.json") == (root.resolve() / "new" / "dir" / "file.json")


def test_t28_resolve_inside_rejects_parent_traversal(root: Path) -> None:
    for rel in ("../x", "inner/../../x", "inner/../inner/file.txt", "..", "a/../../b"):
        with pytest.raises(UnsafePathError) as info:
            ids.resolve_inside(root, rel)
        assert info.value.exit_code == 7
        assert info.value.code == "UNSAFE_PATH"
        assert isinstance(info.value, SecurityPolicyError)


def test_t28_resolve_inside_rejects_absolute_paths(root: Path) -> None:
    for rel in ("/etc/passwd", str(root / "inner" / "file.txt"), "\\x", "C:\\Windows", "D:/x"):
        with pytest.raises(UnsafePathError) as info:
            ids.resolve_inside(root, rel)
        assert info.value.exit_code == 7


def test_t28_resolve_inside_rejects_nul_bytes(root: Path) -> None:
    with pytest.raises(UnsafePathError) as info:
        ids.resolve_inside(root, "inner/file.txt\x00.json")
    assert "NUL" in str(info.value)
    assert info.value.exit_code == 7


def test_t28_resolve_inside_rejects_empty_or_non_string(root: Path) -> None:
    for bad in ("", None, 5, b"inner"):
        with pytest.raises(UnsafePathError):
            ids.resolve_inside(root, bad)  # type: ignore[arg-type]


def test_t28_resolve_inside_rejects_symlink_escaping_the_root(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n")
    link = root / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform without symlinks
        pytest.skip(f"symlinks are not supported on this filesystem: {exc}")
    assert link.is_symlink()
    assert (link / "secret.txt").exists()  # the escape would work without the guard
    with pytest.raises(UnsafePathError) as info:
        ids.resolve_inside(root, "escape/secret.txt")
    assert info.value.exit_code == 7
    with pytest.raises(UnsafePathError):
        ids.resolve_inside(root, "escape")


def test_t28_resolve_inside_rejects_symlink_component_even_when_it_stays_inside(root: Path) -> None:
    link = root / "alias"
    try:
        os.symlink(root / "inner", link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform without symlinks
        pytest.skip(f"symlinks are not supported on this filesystem: {exc}")
    assert (link / "file.txt").read_text() == "inside\n"
    with pytest.raises(UnsafePathError) as info:
        ids.resolve_inside(root, "alias/file.txt")
    assert "symlink" in str(info.value)


def test_t28_resolve_inside_rejects_symlinked_file(root: Path, tmp_path: Path) -> None:
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside\n")
    link = root / "inner" / "link.txt"
    try:
        os.symlink(outside_file, link)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform without symlinks
        pytest.skip(f"symlinks are not supported on this filesystem: {exc}")
    with pytest.raises(UnsafePathError):
        ids.resolve_inside(root, "inner/link.txt")


def test_resolve_inside_root_itself_may_be_reached_through_a_symlinked_root(tmp_path: Path) -> None:
    """A symlink *above* the root (for example macOS /var -> /private/var) is not an escape."""
    real_root = tmp_path / "real"
    (real_root / "d").mkdir(parents=True)
    (real_root / "d" / "f").write_text("f")
    root_link = tmp_path / "rootlink"
    try:
        os.symlink(real_root, root_link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform without symlinks
        pytest.skip(f"symlinks are not supported on this filesystem: {exc}")
    assert ids.resolve_inside(root_link, "d/f") == (real_root / "d" / "f").resolve()


def test_unsafe_path_error_hierarchy() -> None:
    err = UnsafePathError("x")
    assert isinstance(err, SecurityPolicyError)
    assert err.exit_code == 7
    assert err.code == "UNSAFE_PATH"
    assert err.to_dict()["exit_code"] == 7


def test_fsync_directory_is_best_effort(tmp_path: Path) -> None:
    ids.fsync_directory(tmp_path)  # existing directory: no error
    ids.fsync_directory(tmp_path / "does-not-exist")  # missing directory: silently ignored
