# agent-kernel
kernel自动优化

Repository for the kernel-optimization agent. The implemented component is **Kernel Memory**
(`kernel_memory`, CLI `kmem`): an authoritative, file-based memory that stores, validates, retrieves,
compares and reconstructs kernel-optimization history along the hierarchy
`kernel → config → attempt (collection of PRs) → PR → commit → run`.

## Layout

| Path | Role |
|---|---|
| `kernel-memory/` | The project: `src/kernel_memory/` (package), `tests/`, `fixtures/handoff/` (copies of the synthetic contracts and demo bundle), `scripts/` (demos, acceptance report), `docs/`. |
| `kernel_memory_ai_handoff/` | Read-only input: the specification, machine contracts, synthetic examples and the handoff validator. Nothing is written there; its `MANIFEST.sha256` must stay valid (`cd kernel_memory_ai_handoff && shasum -a 256 -c MANIFEST.sha256`). |
| `LICENSE`, `.gitignore` | Repository metadata. |

## Quick start

All commands run from `kernel-memory/`. Python 3.11 is required; the pinned versions used here are in
`kernel-memory/requirements.lock.txt`.

```bash
cd kernel-memory
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
# macOS only, if `.venv/bin/python -c "import kernel_memory"` fails (hidden .pth file):
chflags nohidden .venv/lib/python3.11/site-packages/*.pth
.venv/bin/python -m pytest tests -q
bash scripts/demo_p0.sh                       # offline demo on the synthetic bundle
REPS=50 WARMUP=10 bash scripts/demo_cpu.sh    # real CPU execution of the demo vector-add kernel
```

The demos write only under `kernel-memory/.demo/` (git-ignored). `.venv/bin/kmem --help` lists the CLI.

## Documentation

| Document | Content |
|---|---|
| [kernel-memory/docs/README.md](kernel-memory/docs/README.md) | Documentation index. |
| [kernel-memory/docs/IMPLEMENTATION_STATUS.md](kernel-memory/docs/IMPLEMENTATION_STATUS.md) | What is done, what was actually executed, what is pending and why. Read first when resuming. |
| [kernel-memory/docs/ACCEPTANCE.md](kernel-memory/docs/ACCEPTANCE.md) | Acceptance scenarios T01–T32 mapped to the tests that were executed, with status. |
| [kernel-memory/docs/DESIGN.md](kernel-memory/docs/DESIGN.md) | Architecture, module map, shared APIs, identifier and testing conventions. |
| [kernel-memory/docs/CONFIGURATION.md](kernel-memory/docs/CONFIGURATION.md) | Settings, GitHub / TPU / model / LLO configuration, permissions, budgets. |
| [kernel-memory/docs/RECOVERY.md](kernel-memory/docs/RECOVERY.md) | Store integrity model, publication protocol, `kmem recover`, `kmem validate --deep`. |
| [kernel-memory/docs/MIGRATION.md](kernel-memory/docs/MIGRATION.md) | v0.1 → v0.2 migration: input shape, mapping rules, what is never inferred, verification. |
| [kernel-memory/docs/adr/](kernel-memory/docs/adr/) | Architecture decision records. |

## Boundaries

* All fixtures are synthetic. The demo bundle's timings (100/90/88 µs) exist only to test arithmetic and
  display; they are not CPU, GPU or TPU measurements, and fixture records never become production results.
* No TPU execution, live GitHub collection, model-driven planning, or LLO parsing has been executed.
  Each integration is implemented as an explicit, tested unavailable/unauthorized path until its real
  prerequisite (hardware, token, credentials, format sample, authorization) exists; see
  `kernel-memory/docs/IMPLEMENTATION_STATUS.md`.
* The CPU demo measures vector addition on the local host to prove execution integration. It says
  nothing about MLA or TPU performance.

## Repository hygiene

* `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.DS_Store`, `kernel-memory/.demo/` and `*.sqlite`
  are git-ignored. Do not commit a virtual environment or demo stores.
* Never run `pip` outside `kernel-memory/.venv`, and never run `pytest` inside `kernel_memory_ai_handoff/`
  (it would write `__pycache__` next to `MANIFEST.sha256`).
* Commits and pushes are made by the repository owner; see `kernel-memory/docs/adr/ADR-0001-workspace-boundary.md`.
