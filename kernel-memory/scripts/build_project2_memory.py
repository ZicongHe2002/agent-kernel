#!/usr/bin/env python3
"""Build the compact, method/shape-oriented Memory output from Project2.

Input is the Project2 directory. This fixed import recipe reads the two decode
matrices, the single-decode rerun, Result6 forward and the original TPU forward
baseline. Historical measurements are imported, not executed again on this host.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from kernel_memory.domain.models import Record
from kernel_memory.services.importer import import_bundle
from kernel_memory.services.output_memory import export_output_memory, verify_output_memory
from kernel_memory.services.validation import deep_validate
from kernel_memory.storage import MemoryStore

DECODE_DATASETS = (
    "result/decode/result-decode-2",
    "result/decode/result-decode-3",
    "result/decode/mla_decode_single/rerun",
)
FORWARD_DATASETS = (
    "result/baseline_unoptimized_tpu_20260907",
    "result/result_6_tpu",
)


def merge_records(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Reuse common identities without changing any measurement or payload."""
    merged: dict[str, dict[str, Any]] = {}
    for records in groups:
        for record in records:
            previous = merged.get(record["record_id"])
            if previous is None:
                merged[record["record_id"]] = record
            elif previous != record:
                if (previous["record_type"] in {"kernel", "algorithm", "config"}
                        and previous["payload"] == record["payload"]):
                    if record["created_at"] < previous["created_at"]:
                        merged[record["record_id"]] = record
                else:
                    raise ValueError(f"conflicting input records: {record['record_id']}")
    # Validate wire contracts before any store or output publication.
    for record in merged.values():
        Record.from_dict(record)
    return sorted(merged.values(), key=lambda item: (item["record_type"], item["record_id"]))


def build_memory(project2: Path, output: Path) -> dict[str, Any]:
    import import_project2_decode_matrix as decode
    import import_project2_mla_fwd as forward

    project2, output = Path(project2).resolve(), Path(output).resolve()
    if output == project2 or output.is_relative_to(project2):
        raise ValueError("output must be outside the source Project2 directory")
    # Fail on incomplete input rather than silently publishing an empty method.
    for relative in (*DECODE_DATASETS, *FORWARD_DATASETS):
        if not (project2 / relative).is_dir():
            raise FileNotFoundError(f"required input dataset missing: {relative}")
    with tempfile.TemporaryDirectory(prefix="kmem-project2-") as directory:
        staging = Path(directory)
        groups = []
        for index, relative in enumerate(DECODE_DATASETS):
            path = decode.build_matrix_bundle(project2 / relative, staging / f"decode-{index}",
                                              summary_only=True, measured_only=False)
            groups.append(json.loads(path.read_text(encoding="utf-8"))["records"])
        for relative in FORWARD_DATASETS:
            groups.append(forward.build_records(project2 / relative))
        records = merge_records(groups)
        bundle = staging / "bundle.json"
        bundle.write_text(json.dumps({"bundle_version": "0.3.0", "is_fixture": False,
                          "description": "Project2 fixed historical import recipe v1",
                          "records": records}, ensure_ascii=False) + "\n", encoding="utf-8")
        store = MemoryStore.init(staging / "store")
        imported = import_bundle(store, bundle)
        report = deep_validate(store)
        if not report.ok:
            raise ValueError(json.dumps(report.to_dict(), ensure_ascii=False))
        exported = export_output_memory(store, output)
        verified = verify_output_memory(output)
        if not verified["ok"]:
            raise ValueError(json.dumps(verified, ensure_ascii=False))
        return {"output": str(output), "input_recipe": "project2-v1",
                "input_datasets": list(DECODE_DATASETS + FORWARD_DATASETS),
                "record_counts": dict(sorted(Counter(r["record_type"] for r in records).items())),
                "records_imported": imported.records_published,
                "validation": {"ok": report.ok, "errors": len(report.errors),
                               "warning_counts": dict(sorted(Counter(i.code for i in report.warnings).items()))},
                "export": {key: value for key, value in exported.items() if key != "files"},
                "output_verified": verified["ok"],
                "measurements_reexecuted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project2", type=Path, help="source Project2 directory")
    parser.add_argument("output", type=Path, help="final compact Memory directory")
    args = parser.parse_args(argv)
    print(json.dumps(build_memory(args.project2, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
