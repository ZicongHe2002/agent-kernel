"""Physical layout of the authoritative store (layout_version 2, ADR-0004).

    memory/
      manifest.json
      kernels/<kernel-slug>/
        kernel.json
        annotations/<annotation-slug>.json            # annotations whose target is the kernel
        trajectory/                                    # GENERATED views for the whole kernel (deletable)
          trajectory.json                              #   kernel-level view document
          memory_records.jsonl                         #   compact lines: kernel, algorithms, every shape
          shapes/<config-slug>/trajectory.json         #   per-shape view document
          shapes/<config-slug>/memory_records.jsonl
        <algorithm-slug>/                              # slug of payload.algorithm_id (whiteboard: algorithm_N)
          algorithm.json                               #   the record; method_summary is authored data inside it
          annotations/<annotation-slug>.json           #   annotations whose target is the algorithm
          <config-slug>/                               #   slug of the config record id (whiteboard: shape_N)
            config.json
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
      artifacts/registry/<artifact-slug>.json          # ArtifactRef descriptors by artifact_id
      requests/<request-slug>/request.json
      requests/<request-slug>/events/<sequence>-<event-slug>.json
      journal/records.jsonl                            # publication journal (integrity evidence)
      .runtime/                                        # lock, pending manifests, staging (not facts)
      .cache/index.sqlite                              # optional, rebuildable

Algorithm and shape directories sit directly under the kernel directory (the user's whiteboard),
so a few names are reserved: an algorithm id whose slug is ``trajectory``, ``annotations`` or
``kernel.json`` and a config id whose slug is ``annotations`` or ``algorithm.json`` are rejected
(``RESERVED_SLUG``). Slugs are derived from identifiers with ``slug_for_id``; identities remain
the record IDs inside each file, never the paths. Everything under ``kernels/<k>/trajectory/`` is
a derived view and is never treated as a record.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable

from ..domain.errors import InputError, MissingReferenceError
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
TRAJECTORY_DIR = "trajectory"
SHAPE_VIEWS_DIR = "shapes"
ANNOTATIONS_DIR = "annotations"
KERNEL_FILE = "kernel.json"
ALGORITHM_FILE = "algorithm.json"
CONFIG_FILE = "config.json"
RESERVED_ALGORITHM_SLUGS = frozenset({TRAJECTORY_DIR, ANNOTATIONS_DIR, KERNEL_FILE})
RESERVED_SHAPE_SLUGS = frozenset({ANNOTATIONS_DIR, ALGORITHM_FILE})

Resolver = Callable[[str], Record | None]
KernelResolver = Callable[[str], Record | None]


def kernel_dir(kernel_id: str) -> PurePosixPath:
    return PurePosixPath(KERNELS_DIR) / slug_for_id(kernel_id)


def kernel_view_dir(kernel_id: str) -> PurePosixPath:
    return kernel_dir(kernel_id) / TRAJECTORY_DIR


def shape_view_dir(kernel_id: str, config_record_id: str) -> PurePosixPath:
    return kernel_view_dir(kernel_id) / SHAPE_VIEWS_DIR / slug_for_id(config_record_id)


def algorithm_slug(algorithm_id: str) -> str:
    slug = slug_for_id(algorithm_id)
    if slug in RESERVED_ALGORITHM_SLUGS:
        raise InputError(
            f"algorithm_id {algorithm_id!r} is reserved as a directory name under the kernel ({sorted(RESERVED_ALGORITHM_SLUGS)})",
            code="RESERVED_SLUG",
            details={"algorithm_id": algorithm_id, "reserved": sorted(RESERVED_ALGORITHM_SLUGS)},
        )
    return slug


def shape_slug(config_record_id: str) -> str:
    slug = slug_for_id(config_record_id)
    if slug in RESERVED_SHAPE_SLUGS:
        raise InputError(
            f"config record id {config_record_id!r} is reserved as a directory name under an algorithm ({sorted(RESERVED_SHAPE_SLUGS)})",
            code="RESERVED_SLUG",
            details={"record_id": config_record_id, "reserved": sorted(RESERVED_SHAPE_SLUGS)},
        )
    return slug


def algorithm_dir(kernel_id: str, algorithm_id: str) -> PurePosixPath:
    return kernel_dir(kernel_id) / algorithm_slug(algorithm_id)


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


def _require_kernel(kernel_id: str, record: Record, kernel_resolver: KernelResolver) -> Record:
    kernel = kernel_resolver(kernel_id)
    if kernel is None:
        raise MissingReferenceError(
            f"{record.record_type} {record.record_id!r} refers to unknown kernel_id {kernel_id!r}",
            details={"record_id": record.record_id, "kernel_id": kernel_id},
        )
    return kernel


def algorithm_dir_for(algorithm: Record, kernel_resolver: KernelResolver) -> PurePosixPath:
    if algorithm.record_type != "algorithm":
        raise MissingReferenceError(f"record {algorithm.record_id!r} is not an algorithm")
    p = algorithm.payload
    _require_kernel(p.kernel_id, algorithm, kernel_resolver)
    return algorithm_dir(p.kernel_id, p.algorithm_id)


def config_dir_for(record: Record, resolver: Resolver, kernel_resolver: KernelResolver) -> PurePosixPath:
    """Directory of the config (shape) that owns ``record`` (walks parents as needed)."""
    t = record.record_type
    p = record.payload
    if t == "config":
        _require_kernel(p.kernel_id, record, kernel_resolver)
        algorithm = _resolve(resolver, p.algorithm_ref, "algorithm")
        if algorithm.payload.kernel_id != p.kernel_id:
            raise MissingReferenceError(
                f"config {record.record_id!r} belongs to kernel {p.kernel_id!r} but its algorithm "
                f"{algorithm.record_id!r} belongs to kernel {algorithm.payload.kernel_id!r}",
                code="ALGORITHM_KERNEL_MISMATCH",
                details={"record_id": record.record_id, "algorithm_ref": algorithm.record_id},
            )
        return algorithm_dir_for(algorithm, kernel_resolver) / shape_slug(record.record_id)
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
        if target.record_type in ("kernel", "algorithm"):
            raise MissingReferenceError(f"{target.record_type} annotations have no config directory")
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
        return kernel_dir(p.kernel_id) / KERNEL_FILE
    if t == "algorithm":
        return algorithm_dir_for(record, kernel_resolver) / ALGORITHM_FILE
    if t == "config":
        return config_dir_for(record, resolver, kernel_resolver) / CONFIG_FILE
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
            return kernel_dir(target.payload.kernel_id) / ANNOTATIONS_DIR / f"{slug}.json"
        if target.record_type == "algorithm":
            return algorithm_dir_for(target, kernel_resolver) / ANNOTATIONS_DIR / f"{slug}.json"
        return config_dir_for(target, resolver, kernel_resolver) / ANNOTATIONS_DIR / f"{slug}.json"
    raise MissingReferenceError(f"unknown record type {t!r}")


def artifact_relpath(sha256_ref: str) -> PurePosixPath:
    digest = sha256_ref.split(":", 1)[1]
    return PurePosixPath(ARTIFACTS_DIR) / "sha256" / digest[:2] / digest


def artifact_registry_relpath(artifact_id: str) -> PurePosixPath:
    return PurePosixPath(ARTIFACTS_DIR) / "registry" / f"{slug_for_id(artifact_id)}.json"


def request_dir(request_id: str) -> PurePosixPath:
    return PurePosixPath(REQUESTS_DIR) / slug_for_id(request_id)


def is_view_path(relpath: PurePosixPath) -> bool:
    """True for anything under ``kernels/<kernel>/trajectory/`` (generated views, never records)."""
    parts = relpath.parts
    return len(parts) >= 4 and parts[0] == KERNELS_DIR and parts[2] == TRAJECTORY_DIR


def is_record_file(relpath: PurePosixPath) -> bool:
    """Files under kernels/ that hold authoritative records (excludes the trajectory view tree and temp files).

    The rule is directory-based: a record whose slug happens to be ``trajectory`` inside a shape
    directory is still a record; only ``kernels/<kernel>/trajectory/**`` is excluded.
    """
    if not relpath.parts or relpath.parts[0] != KERNELS_DIR:
        return False
    if is_view_path(relpath):
        return False
    name = relpath.name
    if name.startswith(".") or not name.endswith(".json"):
        return False
    return True
