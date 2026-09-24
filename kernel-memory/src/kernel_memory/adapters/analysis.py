"""Analysis adapters: turn stored evidence artifacts into schema-valid Metric dicts.

Specification sections 13 and 17. Observations are not explanations: an analysis
adapter reports *what a parser read from an artifact* with the parser's identity and
version, and never invents a value. Every metric it emits carries a status:

* ``observed``        a value was read; ``source_artifact_ref`` names the artifact it came from
* ``not_collected``   no relevant artifact / the artifact lacked the field; ``value`` is null
* ``unsupported``     an adapter recognised the artifact but has no parser for its format
* ``parse_error``     the artifact was malformed or the adapter failed; ``value`` is null
* ``not_applicable``  reserved for metrics that do not apply to a backend

Uncollected values are ``null``, never ``0`` (section 13). The runner
(``execution/runner.py``) calls ``analyze_artifacts(adapters, pairs)`` after a
succeeded execution and additionally discards any observed metric whose
``source_artifact_ref`` is not part of the run.

Public API
----------
``KNOWN_METRICS``  name -> {unit, kind, scope, definition}; ``tuple(KNOWN_METRICS)`` is
    the runner's ``KNOWN_METRICS``.
``metric_dict(name, *, status, value=None, source_artifact_ref=None, parser_id=None,
              parser_version=None, note="", declared=KNOWN_METRICS) -> dict``
    Build a Metric that validates against the record schema; enforces
    ``status != observed -> value is None`` and ``observed -> value and source present``.
``MockSpillAnalysisAdapter``  parses the fixture ``mock-spill-v1`` JSON report.
``LloAnalysisAdapter``       recognises ``llo_dump`` artifacts and honestly refuses
    (``UnsupportedFormat``): no LLO format specification or sample was supplied.
``analyze_artifacts(adapters, pairs, *, declared_metrics=KNOWN_METRICS) -> AnalysisReport``
    Dispatches ``(ArtifactRef, bytes)`` pairs to the adapters that accept them and
    always returns one metric per declared name; never raises for adapter failures.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..domain.errors import InputError, KernelMemoryError, UnsupportedFormat
from ..domain.jsonio import loads_strict
from ..domain.models import ArtifactRef, to_json
from ..domain.schema import validate_nested
from .base import AnalysisReport, ArtifactBlob

SPILL_METRIC = "register_spill_vmem_static_bytes"

KNOWN_METRICS: dict[str, dict[str, str]] = {
    SPILL_METRIC: {
        "unit": "bytes",
        "kind": "static_estimate",
        "scope": "compiled_kernel",
        "definition": (
            "Static compiler estimate of register spill bytes to VMEM; "
            "a static estimate, not runtime HBM traffic."
        ),
    }
}

METRIC_STATUSES: tuple[str, ...] = ("observed", "not_collected", "unsupported", "parse_error", "not_applicable")
MOCK_SPILL_FORMAT = "mock-spill-v1"
MOCK_ANALYSIS_KIND = "mock_analysis"
LLO_DUMP_KIND = "llo_dump"
_JSON_MEDIA_TYPES: frozenset[str] = frozenset({"application/json", "text/json"})


def _is_json_media_type(media_type: Any) -> bool:
    if not isinstance(media_type, str):
        return False
    base = media_type.split(";", 1)[0].strip().lower()
    return base in _JSON_MEDIA_TYPES or base.endswith("+json")


def metric_dict(
    name: str,
    *,
    status: str,
    value: int | float | None = None,
    source_artifact_ref: str | None = None,
    parser_id: str | None = None,
    parser_version: str | None = None,
    note: str = "",
    declared: Mapping[str, Mapping[str, str]] = KNOWN_METRICS,
) -> dict[str, Any]:
    """Build a schema-valid Metric dict for a declared metric name.

    ``note`` is appended to the metric's semantic definition (e.g. why a value is missing).
    """
    if not isinstance(name, str) or not name:
        raise InputError("metric name must be a non-empty string", code="INVALID_METRIC")
    spec = declared.get(name)
    if spec is None:
        raise InputError(f"unknown metric {name!r}", code="UNKNOWN_METRIC", details={"known": sorted(declared)})
    if status not in METRIC_STATUSES:
        raise InputError(f"invalid metric status {status!r}", code="INVALID_METRIC", details={"allowed": list(METRIC_STATUSES)})
    if status == "observed":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(f"observed metric {name!r} needs a numeric value", code="INVALID_METRIC")
        if not source_artifact_ref:
            raise InputError(f"observed metric {name!r} needs a source_artifact_ref", code="INVALID_METRIC")
    elif value is not None:
        raise InputError(
            f"metric {name!r} with status {status!r} must have a null value (uncollected values are never numbers)",
            code="INVALID_METRIC",
        )
    definition = str(spec["definition"])
    if note:
        definition = f"{definition} {note}".strip()
    metric = {
        "name": name,
        "status": status,
        "value": value,
        "unit": str(spec["unit"]),
        "kind": str(spec["kind"]),
        "scope": str(spec["scope"]),
        "source_artifact_ref": source_artifact_ref,
        "parser_id": parser_id,
        "parser_version": parser_version,
        "definition": definition,
    }
    validate_nested("Metric", metric)
    return metric


def _manifest_of(ref: Any) -> dict[str, Any]:
    if isinstance(ref, ArtifactRef):
        return to_json(ref)
    if isinstance(ref, dict):
        return dict(ref)
    raise InputError(f"artifact reference must be an ArtifactRef or dict, got {type(ref).__name__}", code="INVALID_ARTIFACT_REF")


def _artifact_id_of(ref: Any) -> str:
    manifest = _manifest_of(ref)
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise InputError("artifact reference lacks an artifact_id", code="INVALID_ARTIFACT_REF")
    return artifact_id


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------
class MockSpillAnalysisAdapter:
    """Parses the fixture ``mock-spill-v1`` JSON report into the static spill estimate.

    The report is a JSON object ``{"format": "mock-spill-v1", "register_spill_vmem_static_bytes": <int>=0>}``.
    A recognised artifact with another ``format`` raises ``UnsupportedFormat``; malformed JSON or an
    invalid value becomes ``parse_error``; a missing key becomes ``not_collected``. ``0`` is reported
    only when the report literally says ``0``.
    """

    adapter_id = "mock-spill"
    parser_version = "1"
    metrics: tuple[str, ...] = (SPILL_METRIC,)
    conclusion = "spill observed; execution status is unaffected"

    def accepts(self, artifact_manifest: dict) -> bool:
        if not isinstance(artifact_manifest, dict):
            return False
        if artifact_manifest.get("kind") == MOCK_ANALYSIS_KIND:
            return True
        artifact_id = artifact_manifest.get("artifact_id")
        return (
            isinstance(artifact_id, str)
            and artifact_id.endswith("spill-report")
            and _is_json_media_type(artifact_manifest.get("media_type"))
        )

    def _metric(self, status: str, artifact_id: str, *, value: int | None = None, note: str = "") -> dict[str, Any]:
        return metric_dict(
            SPILL_METRIC,
            status=status,
            value=value,
            source_artifact_ref=artifact_id if status == "observed" else None,
            parser_id=self.adapter_id,
            parser_version=self.parser_version,
            note=note,
        )

    def _parse_one(self, artifact_id: str, blob: bytes) -> dict[str, Any]:
        try:
            payload = loads_strict(blob)
        except KernelMemoryError as exc:
            return self._metric("parse_error", artifact_id, note=f"Parse error in artifact {artifact_id!r}: {exc.message}")
        if not isinstance(payload, dict):
            return self._metric(
                "parse_error", artifact_id, note=f"Parse error in artifact {artifact_id!r}: report must be a JSON object, got {type(payload).__name__}."
            )
        fmt = payload.get("format")
        if fmt != MOCK_SPILL_FORMAT:
            raise UnsupportedFormat(
                f"artifact {artifact_id!r} has format {fmt!r}; adapter {self.adapter_id!r} parses only {MOCK_SPILL_FORMAT!r}",
                details={"artifact_id": artifact_id, "format": fmt, "expected_format": MOCK_SPILL_FORMAT},
            )
        if SPILL_METRIC not in payload:
            return self._metric(
                "not_collected", artifact_id, note=f"Not collected: artifact {artifact_id!r} ({MOCK_SPILL_FORMAT}) has no {SPILL_METRIC!r} field."
            )
        value = payload[SPILL_METRIC]
        if isinstance(value, bool) or not isinstance(value, int):
            return self._metric(
                "parse_error",
                artifact_id,
                note=f"Parse error in artifact {artifact_id!r}: {SPILL_METRIC!r} must be a non-negative integer byte count, got {type(value).__name__} {value!r}.",
            )
        if value < 0:
            return self._metric(
                "parse_error", artifact_id, note=f"Parse error in artifact {artifact_id!r}: {SPILL_METRIC!r} is negative ({value})."
            )
        return self._metric("observed", artifact_id, value=value)

    def parse(self, artifacts: list[tuple[Any, bytes]]) -> AnalysisReport:
        parsed: list[dict[str, Any]] = []
        for ref, blob in sorted(artifacts, key=lambda pair: _artifact_id_of(pair[0])):
            if not isinstance(blob, (bytes, bytearray)):
                raise InputError("artifact payload must be bytes", code="INVALID_ARTIFACT_BYTES")
            parsed.append(self._parse_one(_artifact_id_of(ref), bytes(blob)))
        # An observed value from any accepted artifact takes precedence over a status-only reading.
        observed = [m for m in parsed if m["status"] == "observed"]
        others = [m for m in parsed if m["status"] != "observed"]
        return AnalysisReport(metrics=observed + others, conclusion=self.conclusion if observed else None, artifacts=[])


class LloAnalysisAdapter:
    """Recognises LLO dumps and refuses honestly: no LLO format specification or sample was supplied."""

    adapter_id = "llo-unsupported"
    parser_version = "0"
    metrics: tuple[str, ...] = (SPILL_METRIC,)
    message = "no LLO format specification or sample was supplied; cannot parse"

    def accepts(self, artifact_manifest: dict) -> bool:
        return isinstance(artifact_manifest, dict) and artifact_manifest.get("kind") == LLO_DUMP_KIND

    def parse(self, artifacts: list[tuple[Any, bytes]]) -> AnalysisReport:
        artifact_ids = sorted(_artifact_id_of(ref) for ref, _ in artifacts)
        raise UnsupportedFormat(self.message, details={"artifact_ids": artifact_ids, "kind": LLO_DUMP_KIND, "adapter_id": self.adapter_id})


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------
def _adapter_metric_names(adapter: Any, declared_metrics: Mapping[str, Mapping[str, str]]) -> list[str]:
    names = getattr(adapter, "metrics", None)
    if names is None:
        return list(declared_metrics)
    return [n for n in names if n in declared_metrics]


def _normalize_report(result: Any) -> tuple[list[Any], str | None, list[ArtifactBlob]]:
    if isinstance(result, AnalysisReport):
        return list(result.metrics or []), result.conclusion, [a for a in (result.artifacts or []) if isinstance(a, ArtifactBlob)]
    if isinstance(result, dict):
        conclusion = result.get("conclusion")
        return (
            list(result.get("metrics") or []),
            conclusion if isinstance(conclusion, str) else None,
            [a for a in (result.get("artifacts") or []) if isinstance(a, ArtifactBlob)],
        )
    if isinstance(result, list):
        return list(result), None, []
    raise InputError(f"analysis adapter returned {type(result).__name__}, expected AnalysisReport", code="INVALID_ANALYSIS_REPORT")


def _accepted_pairs(adapter: Any, manifests: list[tuple[dict[str, Any], Any, bytes]]) -> list[tuple[Any, bytes]]:
    accepted: list[tuple[Any, bytes]] = []
    for manifest, ref, blob in manifests:
        if bool(adapter.accepts(dict(manifest))):
            accepted.append((ref, blob))
    return accepted


def analyze_artifacts(
    adapters: list[Any],
    pairs: list[tuple[Any, bytes]],
    *,
    declared_metrics: Mapping[str, Mapping[str, str]] = KNOWN_METRICS,
) -> AnalysisReport:
    """Run every accepting adapter over ``pairs`` and return one metric per declared name.

    * adapters are processed in ``adapter_id`` order; an adapter that accepts no artifact is skipped;
    * a metric produced by an adapter is kept as produced (first producer wins per name);
    * ``UnsupportedFormat`` from an adapter marks the metrics it would have produced ``unsupported``,
      any other exception marks them ``parse_error`` (both with a null value); processing continues;
    * declared metrics nobody produced are ``not_collected`` ("no relevant artifact").
    """
    manifests: list[tuple[dict[str, Any], Any, bytes]] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise InputError("analysis pairs must be (ArtifactRef, bytes) tuples", code="INVALID_ANALYSIS_INPUT")
        ref, blob = pair
        if not isinstance(blob, (bytes, bytearray)):
            raise InputError("artifact payload must be bytes", code="INVALID_ARTIFACT_BYTES")
        manifests.append((_manifest_of(ref), ref, bytes(blob)))

    ordered = sorted(adapters or [], key=lambda a: str(getattr(a, "adapter_id", "")))
    produced: dict[str, dict[str, Any]] = {}
    placeholders: dict[str, dict[str, Any]] = {}
    conclusions: list[str] = []
    extra_artifacts: list[ArtifactBlob] = []

    for adapter in ordered:
        adapter_id = str(getattr(adapter, "adapter_id", "") or type(adapter).__name__)
        parser_version = getattr(adapter, "parser_version", None)
        parser_version = str(parser_version) if parser_version is not None else None
        names = _adapter_metric_names(adapter, declared_metrics)
        try:
            accepted = _accepted_pairs(adapter, manifests)
            if not accepted:
                continue
            accepted_ids = sorted(_artifact_id_of(ref) for ref, _ in accepted)
            metrics_raw, conclusion, artifacts = _normalize_report(adapter.parse(accepted))
        except UnsupportedFormat as exc:
            note = f"Unsupported: adapter {adapter_id!r} (parser version {parser_version}) cannot parse artifact(s) {accepted_ids}: {exc.message}"
            for name in names:
                placeholders.setdefault(
                    name, metric_dict(name, status="unsupported", parser_id=adapter_id, parser_version=parser_version, note=note, declared=declared_metrics)
                )
            continue
        except Exception as exc:  # an adapter failure is a status, never a value and never fatal
            note = f"Parse error: adapter {adapter_id!r} (parser version {parser_version}) failed with {type(exc).__name__}: {exc}"
            for name in names:
                placeholders.setdefault(
                    name, metric_dict(name, status="parse_error", parser_id=adapter_id, parser_version=parser_version, note=note, declared=declared_metrics)
                )
            continue
        for raw in metrics_raw:
            metric = dict(raw) if isinstance(raw, dict) else to_json(raw)
            name = metric.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in produced:
                continue
            try:
                validate_nested("Metric", metric)
                if metric["status"] == "observed" and metric["value"] is None:
                    raise InputError("observed metric without a value")
                if metric["status"] != "observed" and metric["value"] is not None:
                    raise InputError("non-observed metric carries a value")
            except KernelMemoryError as exc:
                note = f"Parse error: adapter {adapter_id!r} produced an invalid metric: {exc.message}"
                if name in declared_metrics:
                    placeholders.setdefault(
                        name, metric_dict(name, status="parse_error", parser_id=adapter_id, parser_version=parser_version, note=note, declared=declared_metrics)
                    )
                continue
            produced[name] = metric
        if conclusion:
            conclusions.append(conclusion)
        extra_artifacts.extend(artifacts)

    metrics: list[dict[str, Any]] = []
    for name in declared_metrics:
        if name in produced:
            metrics.append(produced[name])
        elif name in placeholders:
            metrics.append(placeholders[name])
        else:
            metrics.append(metric_dict(name, status="not_collected", note="Not collected: no relevant artifact.", declared=declared_metrics))
    for name in sorted(n for n in produced if n not in declared_metrics):
        metrics.append(produced[name])
    return AnalysisReport(metrics=metrics, conclusion="; ".join(conclusions) if conclusions else None, artifacts=extra_artifacts)
