"""RFC 8785 (JCS) canonicalisation: golden vectors, Appendix B numbers, strings, key order (spec 5.2, T32)."""
from __future__ import annotations

import json
import random

import pytest

from kernel_memory.domain import hashing
from kernel_memory.domain.errors import CanonicalizationError, InputError
from kernel_memory.domain.jcs import MAX_SAFE_INTEGER, canonical_text, canonicalize, format_number
from kernel_memory.domain.schema import golden_hash_vectors


# --------------------------------------------------------------------------------------
# Golden vectors shipped with the contracts
# --------------------------------------------------------------------------------------
def test_golden_vector_file_has_four_vectors() -> None:
    vectors = golden_hash_vectors()
    assert vectors["hash_version"] == hashing.HASH_VERSION == "jcs-sha256-v1"
    assert [v["name"] for v in vectors["vectors"]] == [
        "demo_problem_schema",
        "config_identity",
        "comparison_identity",
        "promotion_policy",
    ]


@pytest.mark.parametrize("index", range(4))
def test_golden_vector_digest_reproduced(index: int) -> None:
    vector = golden_hash_vectors()["vectors"][index]
    assert hashing.jcs_digest(vector["payload"]) == vector["digest"], vector["name"]


@pytest.mark.parametrize("index", range(4))
def test_golden_vector_digest_changes_when_payload_changes(index: int) -> None:
    vector = golden_hash_vectors()["vectors"][index]
    mutated = json.loads(json.dumps(vector["payload"]))
    mutated["__mutation__"] = 1
    assert hashing.jcs_digest(mutated) != vector["digest"]


# --------------------------------------------------------------------------------------
# RFC 8785 Appendix B: number serialisation (ECMAScript Number::toString)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (295.0, "295"),
        (4.5, "4.5"),
        (2e-3, "0.002"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (1e21, "1e+21"),
        (1e30, "1e+30"),
        (1e-27, "1e-27"),
        (333333333.33333329, "333333333.3333333"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (9007199254740992.0, "9007199254740992"),
    ],
)
def test_format_number_matches_rfc8785_appendix_b(value: float, expected: str) -> None:
    assert format_number(value) == expected
    # The same text must appear inside a canonicalised document.
    assert canonical_text([value]) == f"[{expected}]"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (9007199254740992, "9007199254740992"), (-9007199254740992, "-9007199254740992"), (42, "42")],
)
def test_integers_serialise_as_plain_digits(value: int, expected: str) -> None:
    assert canonical_text(value) == expected


def test_negative_numbers_keep_sign_and_exponent_form() -> None:
    assert format_number(-1e21) == "-1e+21"
    assert format_number(-0.000001) == "-0.000001"
    assert format_number(-4.5) == "-4.5"


def test_integer_valued_float_and_int_canonicalise_identically() -> None:
    assert canonicalize({"n": 16}) == canonicalize({"n": 16.0})
    assert hashing.jcs_digest({"n": 100}) == hashing.jcs_digest({"n": 100.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(CanonicalizationError):
        format_number(value)
    with pytest.raises(CanonicalizationError):
        canonicalize({"x": value})


@pytest.mark.parametrize("value", [MAX_SAFE_INTEGER + 1, -(MAX_SAFE_INTEGER + 1), 2**63, 10**30])
def test_integers_beyond_2_pow_53_are_rejected_not_rounded(value: int) -> None:
    with pytest.raises(CanonicalizationError) as info:
        canonicalize(value)
    assert info.value.exit_code == 2
    with pytest.raises(CanonicalizationError):
        canonicalize({"n": [value]})


def test_max_safe_integer_boundary_is_accepted() -> None:
    assert canonical_text(MAX_SAFE_INTEGER) == "9007199254740992"
    assert canonical_text(-MAX_SAFE_INTEGER) == "-9007199254740992"


def test_canonicalization_error_is_an_input_error_with_exit_code_2() -> None:
    err = CanonicalizationError("x")
    assert isinstance(err, InputError)
    assert err.exit_code == 2
    assert err.code == "CANONICALIZATION_ERROR"


# --------------------------------------------------------------------------------------
# RFC 8785 string escaping (section 3.2.2.2)
# --------------------------------------------------------------------------------------
def test_rfc8785_string_sample_escaping() -> None:
    # The RFC sample: euro sign, dollar, U+000F, newline, quotes, backslash, forward slash.
    sample = chr(0x20AC) + "$" + chr(0x0F) + "\n" + "A'B" + '"' + "\\" + "\\" + '"' + "/"
    text = canonical_text(sample)
    assert text.startswith('"') and text.endswith('"')
    assert text == '"' + "\u20ac$\\u000f\\nA'B\\\"\\\\\\\\\\\"/" + '"'
    body = text[1:-1]
    assert "\\u000f" in body  # lower-case \u escape for control characters
    assert "\\n" in body  # newline uses the short escape
    assert "\\\"" in body  # quote escaped
    assert "\\\\" in body  # backslash escaped
    assert "\\/" not in body and body.endswith("/")  # forward slash is NOT escaped
    assert "\u20ac" in body and "\\u20ac" not in body  # non-ASCII emitted literally


def test_control_characters_use_short_or_lowercase_u_escapes() -> None:
    assert canonical_text("\b\f\n\r\t") == '"\\b\\f\\n\\r\\t"'
    assert canonical_text("\x00\x1f") == '"\\u0000\\u001f"'
    # U+007F and above are emitted literally (only < U+0020 is escaped).
    assert canonical_text("\x7f") == '"\x7f"'


def test_canonical_bytes_are_utf8() -> None:
    assert canonicalize("\u20ac") == b'"\xe2\x82\xac"'
    assert canonicalize(chr(0x1F602)) == b'"\xf0\x9f\x98\x82"'


# --------------------------------------------------------------------------------------
# RFC 8785 key ordering (UTF-16 code units, section 3.2.3)
# --------------------------------------------------------------------------------------
RFC_ORDERED_KEYS = [
    "\n",
    "\r",
    "1",
    "</script>",
    chr(0x80),
    chr(0xF6),  # o with diaeresis
    chr(0x20AC),  # euro sign
    chr(0x1F602),  # face with tears of joy (surrogate pair in UTF-16)
    chr(0xFB33),  # Hebrew dalet with dagesh (sorts after the surrogate pair in UTF-16)
]


def test_keys_sorted_by_utf16_code_units_not_code_points() -> None:
    # Code-point order would put U+FB33 before U+1F602; UTF-16 order puts it after (D83D < FB33).
    assert sorted(RFC_ORDERED_KEYS) != RFC_ORDERED_KEYS
    rng = random.Random(8785)
    for _ in range(5):
        shuffled_keys = list(RFC_ORDERED_KEYS)
        rng.shuffle(shuffled_keys)
        obj = {k: i for i, k in enumerate(shuffled_keys)}
        text = canonical_text(obj)
        # json.loads preserves member order, so the parsed key list is the serialised order.
        assert list(json.loads(text)) == RFC_ORDERED_KEYS


def test_rfc8785_key_order_sample_exact_text() -> None:
    obj = {k: None for k in reversed(RFC_ORDERED_KEYS)}
    expected = (
        '{"\\n":null,"\\r":null,"1":null,"</script>":null,"\x80":null,"\xf6":null,'
        '"\u20ac":null,"\U0001F602":null,"\ufb33":null}'
    )
    assert canonical_text(obj) == expected


def test_canonical_bytes_independent_of_insertion_order() -> None:
    a = {"z": 1, "a": {"y": [1, 2, {"q": True, "b": None}], "b": "x"}, "m": 3.5}
    b = {"m": 3.5, "a": {"b": "x", "y": [1, 2, {"b": None, "q": True}]}, "z": 1}
    assert list(a) != list(b)
    assert canonicalize(a) == canonicalize(b)
    assert hashing.jcs_digest(a) == hashing.jcs_digest(b)


def test_array_order_is_significant() -> None:
    assert canonicalize([1, 2]) != canonicalize([2, 1])


def test_no_whitespace_and_literals() -> None:
    assert canonical_text({"a": [True, False, None, ""]}) == '{"a":[true,false,null,""]}'
    assert canonical_text({}) == "{}"
    assert canonical_text([]) == "[]"


def test_nested_structure_matches_rfc8785_style_output() -> None:
    doc = {
        "numbers": [333333333.33333329, 1e30, 4.5, 0.002, 1e-27],
        "string": "\u20ac$\x0f\nA'B\"\\\\\"/",
        "literals": [None, True, False],
    }
    assert canonical_text(doc) == (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"\u20ac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )


# --------------------------------------------------------------------------------------
# Rejections (T32)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", [1, 2.5, None, True, ("a",)])
def test_non_string_keys_are_rejected(key: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize({key: "x"})  # type: ignore[dict-item]


def test_non_json_types_are_rejected() -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize({"x": {1, 2}})  # set
    with pytest.raises(CanonicalizationError):
        canonicalize({"x": b"bytes"})
    with pytest.raises(CanonicalizationError):
        canonicalize(object())


def test_booleans_are_not_numbers() -> None:
    assert canonical_text(True) == "true"
    assert canonicalize({"n": True}) != canonicalize({"n": 1})
    assert hashing.jcs_digest({"n": True}) != hashing.jcs_digest({"n": 1})


def test_tuples_serialise_like_arrays() -> None:
    assert canonicalize((1, "a")) == canonicalize([1, "a"])


def test_jcs_digest_is_sha256_prefixed_hex() -> None:
    digest = hashing.jcs_digest({"a": 1})
    assert hashing.is_sha256_ref(digest)
    assert digest == hashing.sha256_bytes(canonicalize({"a": 1}))


def test_sort_keys_json_dumps_is_not_jcs_for_unicode_keys() -> None:
    """Document why the project does not use the json.dumps(sort_keys=True) shortcut."""
    obj = {chr(0xFB33): 1, chr(0x1F602): 2}
    shortcut = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    # Python sorts str by code point: U+FB33 (64307) < U+1F602 (128514).
    assert list(json.loads(shortcut)) == [chr(0xFB33), chr(0x1F602)]
    # JCS sorts by UTF-16 code units: D83D DE02 < FB33.
    assert list(json.loads(canonical_text(obj))) == [chr(0x1F602), chr(0xFB33)]
    assert shortcut != canonical_text(obj)
