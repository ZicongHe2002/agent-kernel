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
`chflags nohidden .venv/lib/python3.11/site-packages/*.pth` (Python skips hidden `.pth` files).

Full documentation: `docs/README.md` (index), `docs/IMPLEMENTATION_STATUS.md` (what is done,
what was executed, what is pending).
