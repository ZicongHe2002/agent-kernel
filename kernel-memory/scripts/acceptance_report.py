#!/usr/bin/env python3
"""Generate docs/ACCEPTANCE.md from an actual pytest run.

Maps every test named ``test_tNN_*`` to acceptance scenario TNN (specification section 22),
records the real outcome of each test (passed / failed / skipped with reason / error) from a
JUnit XML produced by pytest, and lists scenarios with no executed test as NOT COVERED.
A skipped test is reported as not executed, never as a pass.

Usage (from the project root):
    .venv/bin/python -m pytest tests -q --junitxml .demo/junit.xml
    .venv/bin/python scripts/acceptance_report.py --junit .demo/junit.xml --out docs/ACCEPTANCE.md
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

SCENARIOS = {
    "T01": "Register equivalent normalized configs twice → same hash, no duplicate problem",
    "T02": "Change O-only to O+LSE → new Config; no mixed ranking",
    "T03": "Change tiling only → same Config, new variant/commit",
    "T04": "Several changes in one commit → group-only attribution by default",
    "T05": "Collect three commits, execute one → others remain untested",
    "T06": "PR2 branches from a non-head commit of PR1 → graph identifies the actual origin",
    "T07": "Same source commit in multiple PRs → separate memberships, no moved results",
    "T08": "Head OID differs from actual merge OID → persist tested source; separate comparison groups",
    "T09": "More than 250 commits, shallow history, incomplete pagination → partial coverage or verified fallback",
    "T10": "Force push → preserve history, append snapshot",
    "T11": "Replay an idempotent request → no duplicate execution/Run",
    "T12": "Same ID, different content → explicit conflict",
    "T13": "Rerun the same commit intentionally → new request/Run; old data unchanged",
    "T14": "Compilation failure → no invented correctness/latency",
    "T15": "Execution succeeds, output is wrong → succeeded + fail; no promotion",
    "T16": "Nonzero spill with correct execution → not automatically execution failed",
    "T17": "Missing LLO or parser failure → null and status, not zero",
    "T18": "Different environment/protocol/verifier/scope → NOT_COMPARABLE with differences",
    "T19": "Fixture or unverified import → excluded from production best by default",
    "T20": "Samples disagree with median/p90 → deep validation rejects",
    "T21": "One small improvement or noisy data → provisional/inconclusive, not confirmed",
    "T22": "Three valid pairs meet policy → append policy/evidence-backed Decision",
    "T23": "Change verifier or tolerance → new hash; Agent cannot weaken silently",
    "T24": "Delete trajectory/cache → deterministic reconstruction",
    "T25": "Crash mid-write or after Run publication → recover without repeating measurement",
    "T26": "Late result from expired worker → fencing prevents overwrite",
    "T27": "Missing/corrupt artifact → diagnostic; relevant confirmation blocked",
    "T28": "Malicious path or log prompt injection → deny escalation, do not execute data instructions",
    "T29": "No TPU/token/model API → core works; integrations explicitly unexecuted",
    "T30": "Budget exhaustion, cancellation, restart → finite stop, preserved accounting/history",
    "T31": "Migrated short SHA or ambiguous spill semantics → unresolved/unverified, not guessed",
    "T32": "Misspelled property, duplicate key, NaN, boolean dimension → reject invalid input",
}

# Scenarios whose full scope cannot be executed on this host; the note is appended verbatim.
PARTIAL_NOTES = {
    "T02": "hash gate executed with a test-local O vs O+LSE problem adapter; the real MLA output contract is pending real source",
    "T09": "offline HTTP fixtures and a local git clone executed; live GitHub collection unexecuted (no token/repository)",
    "T29": "executed on a CPU-only host: TPU adapter reports BackendUnavailable, model planner reports MODEL_PROVIDER_NOT_CONFIGURED",
    "T17": "exercised with the mock spill format; no real LLO format/sample exists, LloAnalysisAdapter reports unsupported",
}

TEST_ID = re.compile(r"^test_(t\d\d)(?:_|$)", re.IGNORECASE)


def parse_junit(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    cases = root.iter("testcase")
    results: dict[str, dict] = {}
    for tc in cases:
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        module = classname.split(".")[-1] if classname else "?"
        outcome, detail = "passed", ""
        for child in tc:
            if child.tag == "skipped":
                outcome = "skipped"
                detail = child.get("message") or (child.text or "").strip()
            elif child.tag in ("failure", "error"):
                outcome = child.tag
                detail = (child.get("message") or "")[:200]
        results[f"{module}::{name}"] = {"outcome": outcome, "detail": detail, "module": module, "name": name}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--command", default=".venv/bin/python -m pytest tests -q --junitxml .demo/junit.xml")
    args = ap.parse_args()
    results = parse_junit(args.junit)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for key, r in results.items():
        base = r["name"].split("[", 1)[0]
        m = TEST_ID.match(base)
        if m:
            by_scenario[m.group(1).upper()].append({**r, "key": key})
    counts = defaultdict(int)
    for r in results.values():
        counts[r["outcome"]] += 1
    lines = [
        "# Acceptance scenarios T01–T32 → executed tests",
        "",
        f"Generated {date.today().isoformat()} by `scripts/acceptance_report.py` from an actual run of:",
        "",
        f"```bash\n{args.command}\n```",
        "",
        f"Totals: **{counts['passed']} passed, {counts['failure'] + counts['error']} failed, {counts['skipped']} skipped** "
        f"({len(results)} tests). A skipped test is *not executed* and is never counted as a pass.",
        "",
        "Status legend: `executed-pass` (all mapped tests passed), `FAILED` (at least one mapped test failed), "
        "`skipped` (mapped tests exist but were skipped, reason shown), `partial` (executed here but part of the "
        "scenario's scope needs an external prerequisite, note shown), `NOT COVERED` (no test carries this id).",
        "",
        "| Id | Scenario | Tests (module::function) | Status |",
        "|---|---|---|---|",
    ]
    not_covered = []
    for sid, title in SCENARIOS.items():
        tests = sorted(by_scenario.get(sid, []), key=lambda r: r["key"])
        if not tests:
            not_covered.append(sid)
            lines.append(f"| {sid} | {title} | — | **NOT COVERED** |")
            continue
        outcomes = {t["outcome"] for t in tests}
        if outcomes & {"failure", "error"}:
            status = "**FAILED**: " + "; ".join(f"{t['key']}" for t in tests if t["outcome"] in ("failure", "error"))
        elif outcomes == {"skipped"}:
            status = "skipped: " + "; ".join(t["detail"] or "no reason" for t in tests)
        else:
            status = "executed-pass"
            skipped = [t for t in tests if t["outcome"] == "skipped"]
            if skipped:
                status += " (" + "; ".join(f"{t['key']} skipped: {t['detail'] or 'no reason'}" for t in skipped) + ")"
        if sid in PARTIAL_NOTES and not outcomes & {"failure", "error"}:
            status = f"partial — {PARTIAL_NOTES[sid]}; tests {status}"
        names = "<br>".join(f"`{t['key']}`" for t in tests)
        lines.append(f"| {sid} | {title} | {names} | {status} |")
    lines += ["", "## Skipped tests (not executed)", ""]
    skipped = [r for r in results.values() if r["outcome"] == "skipped"]
    if skipped:
        for r in sorted(skipped, key=lambda r: (r["module"], r["name"])):
            lines.append(f"* `{r['module']}::{r['name']}` — {r['detail'] or 'no reason given'}")
    else:
        lines.append("None. Every collected test executed.")
    lines += ["", "## Integrations not executed on this host", "",
              "* Live GitHub collection (no repository/token; `permissions.allow_network` false) — client tested with offline HTTP fixtures only.",
              "* TPU execution (no TPU device, no MLA source/ABI, no `allow_tpu_execution`) — `JaxAdapter` reports `BackendUnavailable` here.",
              "* Model-driven planner (no provider credentials, `allow_model_api_calls` false) — `UnavailableModelPlanner` reports `MODEL_PROVIDER_NOT_CONFIGURED`; MockPlanner exercises orchestration only.",
              "* LLO analysis (no format specification or sample) — `LloAnalysisAdapter` reports `UnsupportedFormat`.",
              "* v0.1 migration of real data (no real v0.1 dataset exists here) — exercised on a synthetic input only.",
              ""]
    if not_covered:
        lines += [f"Scenarios without a mapped test: {', '.join(not_covered)}.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(SCENARIOS) - len(not_covered)}/{len(SCENARIOS)} scenarios covered; totals {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
