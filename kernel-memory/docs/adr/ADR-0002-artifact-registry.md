# ADR-0002: Artifact registry keyed by artifact_id alongside the content-addressed blob store

**Status:** accepted · **Date:** 2026-09-08

## Context

The record schema stores full `Artifact` descriptors (id, kind, uri, sha256, size, media type, retention,
availability) only inside `run.payload.artifacts[]`. Other fields reference artifacts *by id* without a
descriptor: `commit.diff_artifact_ref`, `run.correctness.report_artifact_ref`, `run.timing.samples_artifact_ref`,
`metric.source_artifact_ref`. A commit's diff artifact therefore had no schema-defined place for its descriptor.
The handoff validator also requires that the same artifact_id never appears with two different descriptors.

## Decision

* Blobs live in `artifacts/sha256/<prefix>/<digest>` (digest of the original bytes).
* Descriptors live in `artifacts/registry/<artifact-slug>.json` as `Artifact` objects. The store registers every
  run artifact on publication and refuses conflicting descriptors for one artifact_id (`ARTIFACT_CONFLICT`).
  Collectors that store diffs register a descriptor the same way and then reference it from
  `commit.diff_artifact_ref`.
* Imported records keep their original `uri` verbatim (provenance, immutability). Evidence is resolved through
  `sha256` in the blob store, never through the URI. Missing/corrupt blobs are diagnostics, not silent nulls.

## Consequences

* `diff_artifact_ref` and in-run references are resolvable and validated without extending the P0 schema.
* Re-importing the same bundle is byte-for-byte idempotent.
* Reinterpretation of an artifact (new parser version) creates Annotations/profile Runs; the registry entry itself
  is immutable.
