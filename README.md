# Agent-CR (Interface-First v1)

This repository now contains a new `agent_cr/` package that implements an interface-first C/R architecture for agent sandboxes.

Legacy benchmark scripts remain in `legacy/` and are unchanged.

## What Is Implemented

- Modular contracts for:
  - Scheduler, executor, policies
  - Runtime adapters (generic + docker/runc dry-run stubs)
  - Process/filesystem workers (checkpoint + restore)
  - Checkpoint manager (local filesystem implementation)
  - Sandbox inspector, interceptor hooks, sandbox lifecycle manager
  - Telemetry event/metric sink
- Versioned checkpoint manifest (`v1`) with integrity hash validation.
- Deterministic dry-run behavior for runtime operations (`executed=False`) so orchestration can be tested end-to-end.
- Sync executor API backed by thread pool for concurrent jobs.

## Quickstart

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Run microbench:

```bash
python3 benchmarks/bench_agent_cr_micro.py --iters 1000 --storage-iters 200 --executor-jobs 64
```

Run simulated E2E benchmark:

```bash
python3 benchmarks/bench_agent_cr_e2e.py --sandboxes 8 --iters 20
```

## Package Layout

- `agent_cr/` - core package
- `benchmarks/` - new v1 interface/simulated benchmarks
- `tests/` - unit + contract + simulated e2e tests
- `legacy/` - prior runtime benchmark scripts
