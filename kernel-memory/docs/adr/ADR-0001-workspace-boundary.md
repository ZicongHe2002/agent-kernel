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

## Amendment 2026-09-24

The sections above are kept as written on 2026-09-08; the following facts have changed since.

* **Workspace path and version control.** The project was relocated from `…/company-project/kernel-agent/`
  to `…/company-project/agent-kernel/`. That directory **is** a Git repository (`ZicongHe2002/agent-kernel`,
  branch `main`); the project lives at `agent-kernel/kernel-memory/`. The statements "It is not a Git repository"
  and "the project directory is not a Git repository unless the user initialises one" no longer hold.
* **Who commits.** Commits, pushes, and remote PR operations are performed only by the user. Agents working in this
  repository may edit files and stage nothing on their own; `git add`, `git commit`, and `git rm` remain user actions.
* **What is tracked.** The virtual environment and caches had been committed with the initial import. They are being
  removed from the index (staged deletions, awaiting the user's commit) and are ignored by the root `.gitignore`:
  `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.DS_Store`, `kernel-memory/.demo/`, `*.sqlite`.
  Source, contracts, fixtures, tests, scripts, and docs stay tracked.
* **Environment.** The venv was recreated in place (Python 3.11.8; pins in `requirements.lock.txt`). On this macOS
  host the editable-install `.pth` file is repeatedly re-flagged hidden, which Python 3.11.4+ then skips;
  `chflags nohidden .venv/lib/python3.11/site-packages/*.pth` clears it, and a `.pth`-independent fallback
  (a `site-packages/kernel_memory` symlink to `src/kernel_memory`, plus `tests/conftest.py` adding `src/` to
  `sys.path`) keeps `kmem` and the test suite importable regardless. Nothing is installed globally.
* **Unchanged.** `kernel_memory_ai_handoff/` remains immutable input; its `MANIFEST.sha256` must keep validating
  (`shasum -a 256 -c MANIFEST.sha256` from inside that directory), so nothing is written there and pytest is never
  run inside it. Sibling directories under `company-project/` are still neither read nor modified. The MLA source,
  TPU hardware, GitHub token, model credentials, and LLO samples are still absent; see
  `docs/IMPLEMENTATION_STATUS.md` for the current pending-integration list.
