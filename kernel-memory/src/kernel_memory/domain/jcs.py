"""RFC 8785 JSON Canonicalization Scheme (JCS).

This is a complete implementation for the JSON data model as used by this
project, not the ``json.dumps(sort_keys=True)`` shortcut:

* Object members are sorted by the UTF-16 code units of their keys.
* Strings are serialised like ECMAScript ``JSON.stringify``: only ``"``, ``\\``
  and control characters below U+0020 are escaped (``\\b \\f \\n \\r \\t`` or
  lower-case ``\\u00xx``); everything else is emitted literally as UTF-8. Lone
  surrogates are escaped as ``\\udxxx`` (well-formed JSON.stringify behaviour).
* Numbers follow ECMAScript ``Number.prototype.toString``: shortest round-trip
  digits, ``1e+21``, ``1e-7``, ``0.000001``, ``-0`` becomes ``0``.
* ``NaN``/``Infinity`` are rejected. Integers outside the IEEE-754 exactly
  representable range (|n| > 2**53) are rejected instead of silently rounded.
* No whitespace.

The golden vectors in ``contracts/hash_vectors.json`` and the RFC 8785
Appendix B examples are locked by ``tests/test_jcs.py``.
"""
from __future__ import annotations

import math
from typing import Any

from .errors import CanonicalizationError

MAX_SAFE_INTEGER = 2**53

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _serialize_string(value: str) -> str:
    out: list[str] = ['"']
    for ch in value:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
            continue
        cp = ord(ch)
        if cp < 0x20 or 0xD800 <= cp <= 0xDFFF:
            out.append("\\u%04x" % cp)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def format_number(value: float) -> str:
    """ECMAScript Number::toString(10) for a finite double."""
    if isinstance(value, bool):  # pragma: no cover - guarded by caller
        raise CanonicalizationError("booleans are not numbers")
    if not math.isfinite(value):
        raise CanonicalizationError("NaN and Infinity cannot be canonicalised")
    if value == 0:
        return "0"
    text = repr(float(value))  # shortest round-trip representation
    sign = ""
    if text.startswith("-"):
        sign = "-"
        text = text[1:]
    if "e" in text:
        mantissa, exp_text = text.split("e")
        exponent = int(exp_text)
    else:
        mantissa, exponent = text, 0
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
    else:
        int_part, frac_part = mantissa, ""
    digits = int_part + frac_part
    e10 = exponent - len(frac_part)
    while len(digits) > 1 and digits.endswith("0"):
        digits = digits[:-1]
        e10 += 1
    digits = digits.lstrip("0")
    if not digits:  # pragma: no cover - zero handled above
        return "0"
    k = len(digits)
    n = k + e10
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        exp_sign = "+" if e > 0 else "-"
        mant = digits if k == 1 else digits[0] + "." + digits[1:]
        body = f"{mant}e{exp_sign}{abs(e)}"
    return sign + body


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", errors="surrogatepass")


def _serialize(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} exceeds the exactly representable IEEE-754 range (2**53)"
            )
        out.append(str(value))
    elif isinstance(value, float):
        out.append(format_number(value))
    elif isinstance(value, str):
        out.append(_serialize_string(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        first = True
        for item in value:
            if not first:
                out.append(",")
            first = False
            _serialize(item, out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalizationError(f"object keys must be strings, got {type(key).__name__}")
        out.append("{")
        first = True
        for key in sorted(value, key=_utf16_key):
            if not first:
                out.append(",")
            first = False
            out.append(_serialize_string(key))
            out.append(":")
            _serialize(value[key], out)
        out.append("}")
    else:
        raise CanonicalizationError(f"type {type(value).__name__} is not JSON-serialisable")


def canonicalize(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 encoding of ``value``."""
    out: list[str] = []
    _serialize(value, out)
    return "".join(out).encode("utf-8", errors="surrogatepass")


def canonical_text(value: Any) -> str:
    return canonicalize(value).decode("utf-8", errors="surrogatepass")
