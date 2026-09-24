# Kernel Memory (`kernel_memory`, CLI `kmem`)

Authoritative, file-based Memory for a kernel-optimization agent. It stores, validates,
retrieves, compares, and reconstructs kernel-optimization history using the hierarchy

```text
Memory
└── name / kernel_id
    └── config                       # one fixed computational problem
        ├── trajectory               # generated, no exclusive facts, rebuildable
        └── attempt                  # the collection of all PRs for this config
            └── PR
                └── commit
                    ├── changes + summary
                    └── run
                        ├── Result (execution status, correctness, timing)
                        ├── Analysis (metrics with provenance)
                        └── Artifact references (content-addressed evidence)
```

`attempt` is the collection, not a single PR. Untested commits stay untested. Fixtures never
enter production rankings. See `docs/` for the design, ADRs, status, and operating guides.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"      # pinned versions used here: requirements.lock.txt
.venv/bin/python -m pytest tests -q
```

macOS note: if `import kernel_memory` fails inside the venv, run
`chflags nohidden .venv/lib/python3.11/site-packages/*.pth` (Python skips hidden `.pth` files), or add the
`.pth`-independent fallback `ln -s ../../../../src/kernel_memory .venv/lib/python3.11/site-packages/kernel_memory`.

## Demos

Both scripts run from this directory with `.venv`, need no network, and write only under `.demo/`
(git-ignored; the target store is deleted and recreated on every run).

* `bash scripts/demo_p0.sh [STORE]` — offline P0 walk-through on the synthetic handoff bundle:
  `init`, `import-bundle --allow-fixture` twice (the second import is idempotent), `validate --deep`,
  deterministic `trajectory --rebuild` / `--verify`, `query` (untested commits, tiling changes),
  `compare run-demo-a run-demo-baseline` (fixture arithmetic: speedup 100/90, 10 % reduction),
  `decide --dry-run` (blocked: `FIXTURE_NOT_ELIGIBLE`), `export-context`, `status`. Default store
  `.demo/p0-memory`. Every number in it is a fixture value, not a measurement.
* `REPS=50 WARMUP=10 bash scripts/demo_cpu.sh [STORE]` — real execution of the demo vector-add
  kernel on this host's CPU: registers kernel, config and baseline, executes and measures the baseline
  and a runtime-override variant (`provenance=trusted_worker`), replays a request id without executing
  again, records a deliberately wrong candidate as `succeeded` with correctness `fail`, compares,
  decides with one pair (`inconclusive`), shows the `jax_tpu` backend refusing explicitly, deep-validates,
  rebuilds the trajectory, and runs the MockPlanner under a small budget. `REPS` and `WARMUP` set the
  benchmark repetitions and warm-up iterations (defaults 50 and 10). Default store `.demo/cpu-memory`;
  protocol, verifier, pairs and budget files go to `.demo/cpu-demo-files/`. This demonstrates
  execution integration on CPU only; it says nothing about MLA or TPU performance.

## Documentation

`docs/README.md` (index), `docs/IMPLEMENTATION_STATUS.md` (what is done, what was executed, what is
pending), `docs/ACCEPTANCE.md` (acceptance scenarios T01–T32 mapped to the tests that were actually
executed, with their status), `docs/DESIGN.md`, `docs/CONFIGURATION.md`, `docs/RECOVERY.md`,
`docs/MIGRATION.md`, `docs/adr/`.
