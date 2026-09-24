"""Strict JSON/YAML loading and readable JSON writing (DESIGN section 3; spec 3, 5.2, T32)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kernel_memory.domain import jsonio
from kernel_memory.domain.errors import InputError, KernelMemoryError


def _assert_code(excinfo: pytest.ExceptionInfo, code: str) -> None:
    err = excinfo.value
    assert isinstance(err, InputError)
    assert err.code == code
    assert err.exit_code == 2


# --------------------------------------------------------------------------------------
# loads_strict
# --------------------------------------------------------------------------------------
def test_loads_strict_parses_ordinary_json() -> None:
    assert jsonio.loads_strict('{"b": [1, 2.5, null, true], "a": "x"}') == {"b": [1, 2.5, None, True], "a": "x"}
    assert jsonio.loads_strict("[]") == []
    assert jsonio.loads_strict('"s"') == "s"


def test_t32_loads_strict_rejects_duplicate_object_key() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict('{"a": 1, "a": 2}')
    _assert_code(excinfo, "DUPLICATE_JSON_KEY")
    assert "'a'" in str(excinfo.value)


def test_t32_loads_strict_rejects_duplicate_key_in_nested_object() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict('{"outer": {"k": 1, "k": 1}}')
    _assert_code(excinfo, "DUPLICATE_JSON_KEY")


def test_t32_loads_strict_rejects_duplicate_key_even_when_values_equal() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict('{"a": 1, "a": 1}')
    _assert_code(excinfo, "DUPLICATE_JSON_KEY")


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", '{"x": NaN}', "[1, Infinity]"])
def test_t32_loads_strict_rejects_nonfinite_constants(text: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict(text)
    _assert_code(excinfo, "NONFINITE_JSON")


@pytest.mark.parametrize("text", ["", "{", '{"a": }', "{'a': 1}", '{"a": 1,}', "undefined", '{"a": 1} x'])
def test_t32_loads_strict_rejects_invalid_json(text: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict(text)
    _assert_code(excinfo, "INVALID_JSON")


def test_loads_strict_decodes_utf8_bytes() -> None:
    assert jsonio.loads_strict('{"name": "héllo ✓"}'.encode("utf-8")) == {"name": "héllo ✓"}
    assert jsonio.loads_strict(b"[1, 2]") == [1, 2]


def test_t32_loads_strict_rejects_invalid_utf8_bytes() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.loads_strict(b'{"a": "\xff\xfe"}')
    _assert_code(excinfo, "INVALID_UTF8")


def test_loads_strict_errors_are_kernel_memory_errors_with_stable_payload() -> None:
    with pytest.raises(KernelMemoryError) as excinfo:
        jsonio.loads_strict("{")
    payload = excinfo.value.to_dict() if hasattr(excinfo.value, "to_dict") else None
    if payload is not None:
        assert payload["error"] == "INVALID_JSON"
        assert payload["exit_code"] == 2


# --------------------------------------------------------------------------------------
# load_json_file
# --------------------------------------------------------------------------------------
def test_load_json_file_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text('{"k": [1, 2, 3]}', encoding="utf-8")
    assert jsonio.load_json_file(path) == {"k": [1, 2, 3]}
    # str paths are accepted as well as Path objects
    assert jsonio.load_json_file(str(path)) == {"k": [1, 2, 3]}  # type: ignore[arg-type]


def test_load_json_file_rejects_file_above_max_bytes(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_text('{"k": "' + "x" * 100 + '"}', encoding="utf-8")
    size = path.stat().st_size
    with pytest.raises(InputError) as excinfo:
        jsonio.load_json_file(path, max_bytes=size - 1)
    _assert_code(excinfo, "INPUT_TOO_LARGE")
    assert str(size) in str(excinfo.value)
    # exactly at the limit is allowed
    assert jsonio.load_json_file(path, max_bytes=size) == {"k": "x" * 100}


def test_load_json_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_json_file(tmp_path / "does-not-exist.json")
    _assert_code(excinfo, "FILE_NOT_FOUND")
    assert "does-not-exist.json" in str(excinfo.value)


def test_load_json_file_applies_strict_rules(tmp_path: Path) -> None:
    dup = tmp_path / "dup.json"
    dup.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(InputError) as excinfo:
        jsonio.load_json_file(dup)
    _assert_code(excinfo, "DUPLICATE_JSON_KEY")

    bad_utf8 = tmp_path / "bad.json"
    bad_utf8.write_bytes(b'{"a": "\xc3"}')
    with pytest.raises(InputError) as excinfo:
        jsonio.load_json_file(bad_utf8)
    _assert_code(excinfo, "INVALID_UTF8")


def test_load_json_file_default_limit_is_generous() -> None:
    assert jsonio.DEFAULT_MAX_BYTES >= 1_000_000


# --------------------------------------------------------------------------------------
# dumps_readable / dumps_compact
# --------------------------------------------------------------------------------------
def test_dumps_readable_is_stable_sorted_and_newline_terminated() -> None:
    value = {"b": 1, "a": {"z": [3, 2, 1], "y": None}}
    text = jsonio.dumps_readable(value)
    assert text == jsonio.dumps_readable({"a": {"y": None, "z": [3, 2, 1]}, "b": 1})
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"y"') < text.index('"z"')
    assert text.startswith("{\n  ")
    assert json.loads(text) == value


def test_dumps_readable_keeps_non_ascii_unescaped() -> None:
    text = jsonio.dumps_readable({"name": "héllo ✓ 日本"})
    assert "héllo ✓ 日本" in text
    assert "\\u" not in text


def test_dumps_readable_rejects_nonfinite_floats() -> None:
    with pytest.raises(ValueError):
        jsonio.dumps_readable({"x": math.nan})
    with pytest.raises(ValueError):
        jsonio.dumps_readable([math.inf])


def test_dumps_compact_has_no_whitespace_and_sorted_keys() -> None:
    text = jsonio.dumps_compact({"b": [1, 2], "a": {"d": "x y", "c": True}})
    assert text == '{"a":{"c":true,"d":"x y"},"b":[1,2]}'
    # No whitespace outside string literals.
    outside = text.replace("x y", "xy")
    assert " " not in outside and "\n" not in outside and "\t" not in outside


def test_dumps_compact_keeps_non_ascii_and_rejects_nan() -> None:
    assert jsonio.dumps_compact({"k": "é"}) == '{"k":"é"}'
    with pytest.raises(ValueError):
        jsonio.dumps_compact(math.nan)


def test_round_trip_readable_then_strict_is_identity() -> None:
    value = {
        "int": 1,
        "float": 1.5,
        "neg": -0.25,
        "str": "a\"b\\c\né",
        "null": None,
        "bools": [True, False],
        "nested": {"list": [{"k": [1, [2, [3]]]}], "empty": {}, "empty_list": []},
    }
    assert jsonio.loads_strict(jsonio.dumps_readable(value)) == value
    assert jsonio.loads_strict(jsonio.dumps_compact(value)) == value


def test_round_trip_fixture_bundle(bundle_path: Path) -> None:
    bundle = jsonio.load_json_file(bundle_path)
    assert bundle["is_fixture"] is True
    assert jsonio.loads_strict(jsonio.dumps_readable(bundle)) == bundle
    assert jsonio.loads_strict(jsonio.dumps_compact(bundle)) == bundle
    # Byte-level idempotence: re-serialising the parsed readable form yields the same text.
    text = jsonio.dumps_readable(bundle)
    assert jsonio.dumps_readable(jsonio.loads_strict(text)) == text


# --------------------------------------------------------------------------------------
# load_yaml_strict
# --------------------------------------------------------------------------------------
YAML_DOC = """
kernel_id: demo_vector_add
n: 16
ratio: 1.5
enabled: true
nothing: null
tags:
  - fixture
  - not-mla
nested:
  key: "quoted: value"
  list: [1, 2, 3]
"""

JSON_DOC = """
{
  "kernel_id": "demo_vector_add",
  "n": 16,
  "ratio": 1.5,
  "enabled": true,
  "nothing": null,
  "tags": ["fixture", "not-mla"],
  "nested": {"key": "quoted: value", "list": [1, 2, 3]}
}
"""


def test_yaml_plain_mapping_normalizes_to_same_value_as_json() -> None:
    from_yaml = jsonio.load_yaml_strict(YAML_DOC)
    from_json = jsonio.loads_strict(JSON_DOC)
    assert from_yaml == from_json
    assert jsonio.dumps_compact(from_yaml) == jsonio.dumps_compact(from_json)
    # YAML is never a second truth: types are exactly the JSON data model.
    assert type(from_yaml["n"]) is int
    assert type(from_yaml["ratio"]) is float
    assert type(from_yaml["enabled"]) is bool
    assert from_yaml["nothing"] is None


def test_yaml_scalars_and_sequences_normalize() -> None:
    assert jsonio.load_yaml_strict("- 1\n- two\n- 3.0\n") == [1, "two", 3.0]
    assert jsonio.load_yaml_strict("plain\n") == "plain"
    assert jsonio.load_yaml_strict("{}\n") == {}
    assert jsonio.load_yaml_strict("") is None


def test_t32_yaml_duplicate_key_rejected() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict("a: 1\na: 2\n")
    _assert_code(excinfo, "DUPLICATE_YAML_KEY")
    assert "'a'" in str(excinfo.value)


def test_t32_yaml_duplicate_key_rejected_in_nested_mapping() -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict("outer:\n  k: 1\n  k: 1\n")
    _assert_code(excinfo, "DUPLICATE_YAML_KEY")


def test_t32_yaml_python_tag_rejected_and_nothing_executes(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    doc = f'cmd: !!python/object/apply:os.system ["touch {marker}"]\n'
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict(doc)
    _assert_code(excinfo, "INVALID_YAML")
    assert not marker.exists()


@pytest.mark.parametrize(
    "doc",
    [
        "x: !!python/name:os.system\n",
        "x: !custom value\n",
        "x: !!python/tuple [1, 2]\n",
        "!!python/object:builtins.dict {}\n",
        "x: !<tag:example.com,2026:thing> 1\n",
    ],
)
def test_t32_yaml_custom_tags_rejected(doc: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict(doc)
    _assert_code(excinfo, "INVALID_YAML")


@pytest.mark.parametrize("doc", ["1: one\n", "true: yes\n", "null: x\n", "? [a, b]\n: c\n", "1.5: x\n"])
def test_t32_yaml_non_string_mapping_key_rejected(doc: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict(doc)
    _assert_code(excinfo, "INVALID_YAML")


@pytest.mark.parametrize(
    "doc",
    [
        "when: 2026-09-08\n",
        "when: 2026-09-08T00:00:00Z\n",
        "when: 2026-09-08 10:11:12\n",
        "b: !!binary aGVsbG8=\n",
        "s: !!set {a, b}\n",
        "x: .nan\n",
        "x: .inf\n",
    ],
)
def test_t32_yaml_non_json_values_rejected(doc: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict(doc)
    _assert_code(excinfo, "INVALID_YAML")


def test_t32_yaml_quoted_timestamp_stays_a_string() -> None:
    assert jsonio.load_yaml_strict('when: "2026-09-08T00:00:00Z"\n') == {"when": "2026-09-08T00:00:00Z"}


@pytest.mark.parametrize("doc", ["a: [1, 2\n", "a: b: c\n", "\ta: 1\n", "- a\nb: 1\n"])
def test_t32_yaml_syntax_errors_rejected(doc: str) -> None:
    with pytest.raises(InputError) as excinfo:
        jsonio.load_yaml_strict(doc)
    _assert_code(excinfo, "INVALID_YAML")


def test_yaml_fixture_bundle_round_trips_through_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    bundle = jsonio.load_json_file(Path(__file__).resolve().parents[1] / "fixtures" / "handoff" / "examples" / "demo_bundle.json")
    text = yaml.safe_dump(bundle, sort_keys=True, allow_unicode=True, default_flow_style=False)
    assert jsonio.load_yaml_strict(text) == bundle
