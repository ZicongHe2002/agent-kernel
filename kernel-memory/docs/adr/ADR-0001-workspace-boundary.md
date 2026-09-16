# ADR-0001: Workspace boundary and project root

**Status:** accepted · **Date:** 2026-09-08

## Context

The authorized workspace is `…/company-project/kernel-agent/`. It is not a Git repository. It contains the
read-only handoff package `kernel_memory_ai_handoff/` (specification, contracts, fixtures, validator). No
existing project, `AGENTS.md`, dependency declaration, or test suite was present to integrate into. Sibling
directories under `company-project/` (`math-model`, `project3`, a `.docx`, a `.zip`) belong to other tasks
and were neither read nor modified.

## Decision

* Create the project as `kernel-agent/kernel-memory/` (per the launch prompt: "otherwise create a separate
  `kernel-memory` directory within the clearly authorized workspace").
* Treat `kernel_memory_ai_handoff/` as immutable input. Contracts are copied verbatim into
  `src/kernel_memory/contracts/`; fixtures into `fixtures/handoff/examples/`. The copies are the ones tests use.
* Use a project-local virtual environment `.venv` (Python 3.11.8) with pinned dependencies recorded in
  `requirements.lock.txt`. Nothing is installed globally; global Git configuration is untouched.
* No commits/pushes/remote PR operations are performed; the project directory is not a Git repository unless the
  user initialises one.

## Consequences

* Handoff checksums (`MANIFEST.sha256`) remain valid; the handoff validator can still be run against the original.
* The MLA repository, TPU hardware, GitHub token, model credentials, and LLO samples were not present; the
  corresponding integrations are implemented as explicit, tested unavailable/unsupported paths (see
  `docs/IMPLEMENTATION_STATUS.md`).
