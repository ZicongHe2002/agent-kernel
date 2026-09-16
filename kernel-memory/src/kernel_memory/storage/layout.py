"""Physical layout of the authoritative store (specification section 4).

    memory/
      manifest.json
      kernels/<kernel-slug>/
        kernel.json
        annotations/<annotation-slug>.json          # annotations whose target is the kernel
        configs/<config-slug>/
          config.json
          trajectory.json                          # generated view
          memory_records.jsonl                     # generated view
          attempt/<pr-slug>/
            pr.json
            snapshots/<snapshot-slug>.json
            commits/<commit-slug>/
              commit.json
              runs/<run-slug>/run.json
          baselines/<baseline-slug>/
            baseline.json
            runs/<run-slug>/run.json
          relations/<relation-slug>.json
          decisions/<decision-slug>.json
          annotations/<annotation-slug>.json
      artifacts/sha256/<2-char prefix>/<64-hex digest>
      artifacts/registry/<artifact-slug>.json      # ArtifactRef descriptors by artifact_id
      requests/<request-slug>/request.json
      requests/<request-slug>/events/<sequence>-<event-slug>.json
      journal/records.jsonl                        # publication journal (integrity evidence)
      .runtime/                                    # lock, pending manifests, staging (not facts)
      .cache/index.sqlite                          # optional, rebuildable

Slugs are derived from record identifiers with ``slug_for_id``; identities remain the
record IDs inside each file, never the paths.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable

from ..domain.errors import MissingReferenceError
from ..domain.ids import slug_for_id
from ..domain.models import Record

VIEW_FILE_NAMES = frozenset({"trajectory.json", "memory_records.jsonl", "context.json"})
KERNELS_DIR = "kernels"
ARTIFACTS_DIR = "artifacts"
REQUESTS_DIR = "requests"
JOURNAL_DIR = "journal"
RUNTIME_DIR = ".runtime"
CACHE_DIR = ".cache"
MANIFEST_FILE = "manifest.json"

Resolver = Callable[[str], Record | None]
KernelResolver = Callable[[str], Record | None]


def kernel_dir(kernel_id: str) -> PurePosixPath:
    return PurePosixPath(KERNELS_DIR) / slug_for_id(kernel_id)


def _resolve(resolver: Resolver, record_id: str, *allowed: str) -> Record:
    record = resolver(record_id)
    if record is None:
        raise MissingReferenceError(f"referenced record {record_id!r} does not exist", details={"record_id": record_id})
    if allowed and record.record_type not in allowed:
        raise MissingReferenceError(
            f"referenced record {record_id!r} has type {record.record_type!r}, expected one of {list(allowed)}",
            details={"record_id": record_id, "record_type": record.record_type, "expected": list(allowed)},
        )
    return record


def config_dir_for(record: Record, resolver: Resolver, kernel_resolver: KernelResolver) -> PurePosixPath:
    """Directory of the config that owns ``record`` (walks parents as needed)."""
    t = record.record_type
    p = record.payload
    if t == "config":
        kernel = kernel_resolver(p.kernel_id)
        if kernel is None:
            raise MissingReferenceError(
                f"config {record.record_id!r} refers to unknown kernel_id {p.kernel_id!r}",
                details={"record_id": record.record_id, "kernel_id": p.kernel_id},
            )
        return kernel_dir(p.kernel_id) / "configs" / slug_for_id(record.record_id)
    if t in ("pr", "baseline", "relation", "decision"):
        return config_dir_for(_resolve(resolver, p.config_ref, "config"), resolver, kernel_resolver)
    if t == "run":
        return config_dir_for(_resolve(resolver, p.config_ref, "config"), resolver, kernel_resolver)
    if t == "commit":
        return config_dir_for(_resolve(resolver, p.pr_ref, "pr"), resolver, kernel_resolver)
    if t == "pr_snapshot":
        return config_dir_for(_resolve(resolver, p.pr_ref, "pr"), resolver, kernel_resolver)
    if t == "annotation":
        target = _resolve(resolver, p.target_ref)
        if target.record_type == "kernel":
            raise MissingReferenceError("kernel annotations have no config directory")
        return config_dir_for(target, resolver, kernel_resolver)
    raise MissingReferenceError(f"record type {t!r} has no config directory")


def pr_dir_for(pr: Record, resolver: Resolver, kernel_resolver: KernelResolver) -> PurePosixPath:
    return config_dir_for(pr, resolver, kernel_resolver) / "attempt" / slug_for_id(pr.record_id)


def subject_dir_for(subject: Record, resolver: Resolver, kernel_resolver: KernelResolver) -> PurePosixPath:
    if subject.record_type == "commit":
        pr = _resolve(resolver, subject.payload.pr_ref, "pr")
        return pr_dir_for(pr, resolver, kernel_resolver) / "commits" / slug_for_id(subject.record_id)
    if subject.record_type == "baseline":
        return config_dir_for(subject, resolver, kernel_resolver) / "baselines" / slug_for_id(subject.record_id)
    raise MissingReferenceError(
        f"run subject {subject.record_id!r} must be a commit or baseline, got {subject.record_type!r}"
    )


def record_relpath(record: Record, resolver: Resolver, kernel_resolver: KernelResolver) -> PurePosixPath:
    """Authoritative file path (relative to the store root) for a record."""
    t = record.record_type
    p = record.payload
    slug = slug_for_id(record.record_id)
    if t == "kernel":
        return kernel_dir(p.kernel_id) / "kernel.json"
    if t == "config":
        return config_dir_for(record, resolver, kernel_resolver) / "config.json"
    if t == "pr":
        return pr_dir_for(record, resolver, kernel_resolver) / "pr.json"
    if t == "pr_snapshot":
        pr = _resolve(resolver, p.pr_ref, "pr")
        return pr_dir_for(pr, resolver, kernel_resolver) / "snapshots" / f"{slug}.json"
    if t == "commit":
        pr = _resolve(resolver, p.pr_ref, "pr")
        return pr_dir_for(pr, resolver, kernel_resolver) / "commits" / slug / "commit.json"
    if t == "baseline":
        return config_dir_for(record, resolver, kernel_resolver) / "baselines" / slug / "baseline.json"
    if t == "run":
        subject = _resolve(resolver, p.subject_ref, "commit", "baseline")
        return subject_dir_for(subject, resolver, kernel_resolver) / "runs" / slug / "run.json"
    if t == "relation":
        return config_dir_for(record, resolver, kernel_resolver) / "relations" / f"{slug}.json"
    if t == "decision":
        return config_dir_for(record, resolver, kernel_resolver) / "decisions" / f"{slug}.json"
    if t == "annotation":
        target = _resolve(resolver, p.target_ref)
        if target.record_type == "kernel":
            return kernel_dir(target.payload.kernel_id) / "annotations" / f"{slug}.json"
        return config_dir_for(target, resolver, kernel_resolver) / "annotations" / f"{slug}.json"
    raise MissingReferenceError(f"unknown record type {t!r}")


def artifact_relpath(sha256_ref: str) -> PurePosixPath:
    digest = sha256_ref.split(":", 1)[1]
    return PurePosixPath(ARTIFACTS_DIR) / "sha256" / digest[:2] / digest


def artifact_registry_relpath(artifact_id: str) -> PurePosixPath:
    return PurePosixPath(ARTIFACTS_DIR) / "registry" / f"{slug_for_id(artifact_id)}.json"


def request_dir(request_id: str) -> PurePosixPath:
    return PurePosixPath(REQUESTS_DIR) / slug_for_id(request_id)


def is_record_file(relpath: PurePosixPath) -> bool:
    """Files under kernels/ that hold authoritative records (excludes views, temp files)."""
    if not relpath.parts or relpath.parts[0] != KERNELS_DIR:
        return False
    name = relpath.name
    if name in VIEW_FILE_NAMES or name.startswith(".") or not name.endswith(".json"):
        return False
    return True
