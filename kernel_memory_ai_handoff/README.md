# Kernel Memory / Optimization Agent — AI Implementation Handoff

**v0.2.0 · 2026-09-08**

## How to use (translated from Chinese)

This is an input package for a coding AI to implement the project, **not a completed Kernel Memory system**. The primary hierarchy follows the user's latest agreement: `name → config → trajectory / attempt (PR collection) → PR → commit → run`. The older PDF is historical reference only; its numbers are not measured data.

**Recommended procedure:** Place this directory in a workspace the coding AI can read, open `prompts/BUILD.zh-CN.md`, and send the task content below the separator as a message to the AI. Have the AI read `docs/IMPLEMENTATION.zh-CN.md` and the shared `contracts/` and `examples/`. Normally, there is no need to upload both language versions together.

| File | Purpose |
|---|---|
| `docs/IMPLEMENTATION.zh-CN.md` | Complete implementation specification, translated from Chinese: 25 numbered sections and 32 acceptance scenarios |
| `docs/IMPLEMENTATION.en.md` | Complete English specification corresponding to the original Chinese version |
| `prompts/BUILD.zh-CN.md` | Launch prompt, translated from Chinese: requires the AI to implement, test, and report honestly |
| `prompts/BUILD.en.md` | Equivalent English launch prompt |
| `contracts/record.schema.json` | JSON Schema for P0 records: 10 record types and shared nested types |
| `contracts/demo_problem.schema.json` | Demo-only vector-add problem contract, not an MLA ABI |
| `contracts/hash_vectors.json` | Four identity-hash golden vectors; inputs are deliberately restricted to a simple JSON subset |
| `examples/demo_bundle.json` | 18 synthetic records, two PR contexts, and four synthetic Runs |
| `examples/artifacts/` | Synthetic samples, correctness reports, and analysis evidence, with checksums |
| `examples/mla_config.draft.json` | Historical parameters and unresolved items; cannot be registered as a complete Config |
| `examples/project_settings.template.json` | Starting configuration for safety switches, external dependencies, and budgets |
| `tools/validate_handoff.py` | Validates this handoff package; not a production implementation |
| `tools/test_handoff_negative_cases.py` | Reproduces 12 negative checks; not the future project's complete 32-scenario acceptance suite |
| `HANDOFF_VALIDATION.json` | Results from the handoff validation actually performed for this delivery |
| `HANDOFF_NEGATIVE_TESTS.json` | Negative-test results for the handoff validator |
| `MANIFEST.sha256` | File-content checksums, excluding the manifest itself |

You can first check the handoff package in a Python environment that already has the `jsonschema` dependency:

```bash
python tools/validate_handoff.py
python tools/test_handoff_negative_cases.py
```

If the dependency is missing, the validator exits explicitly; it neither installs automatically nor pretends validation passed. Installation must follow the actual workspace permissions, preferably using a project virtual environment.

**Three boundaries:** Fixtures are entirely fabricated data and cannot enter production rankings; this package has not performed real CPU/TPU benchmarks or live GitHub collection; the documented `kmem` commands are interfaces the AI must implement, not currently installed software.

Implementation order: the P0 Memory core first, then P1 read-only PR collection and a trusted Runner, followed by the P2 budgeted optimization loop. The AI must inspect the actual project for repository paths, the real MLA callable and complete ABI, TPU availability, LLO format, credentials, and write permissions, without guessing or fabricating them. Missing external prerequisites must not block core code that can be completed offline.

## English: how to use

This is an implementation-input package for a coding AI, **not a completed Kernel Memory product**. It follows the latest hierarchy: `name → config → trajectory / attempt (PR collection) → PR → commit → run`. The historical PDF is reference only; its numbers are not measured results.

Place this directory in an accessible workspace, then send the task content below the separator in `prompts/BUILD.en.md`. Ask the AI to read `docs/IMPLEMENTATION.en.md` and the shared `contracts/` and `examples/`. One language version is sufficient.

The full specifications have matching sections 0–24 and 32 acceptance scenarios. The package includes strict P0 schemas, four hash vectors, 18 synthetic records, seven evidence artifacts, a deliberately incomplete MLA draft, and a handoff-only validation tool. The tool does not implement the future product or measure hardware.

Run `python tools/validate_handoff.py` in an environment with `jsonschema` to verify the package. Missing dependencies are reported explicitly rather than installed automatically. Full production JCS support still needs a tested implementation; the checker intentionally handles only the restricted golden-vector domain.

Build P0 first, then read-only collection and trusted execution in P1, then budgeted orchestration in P2. Identify actual repository/ABI/hardware/credentials instead of inventing them. Report unexecuted integrations separately from passed local tests.

**No real CPU/TPU benchmark, live GitHub collection, model optimization, or real LLO parser validation is represented by this handoff.**
