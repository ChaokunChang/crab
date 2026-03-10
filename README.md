# Agent-CR (v2 Runtime Path)

This repository contains the `agent_cr/` package for checkpoint/restore of agent sandboxes.

Legacy benchmark scripts remain in `legacy/` and are unchanged.

## What Is Implemented

- Modular contracts for:
  - Scheduler, executor, policies
  - Runtime adapters (`runc` + CRIU for process state, Docker kept as a compatibility stub)
  - Process/filesystem workers (checkpoint + restore)
  - Checkpoint manager (local filesystem implementation)
  - eBPF-centered sandbox inspector, interceptor hooks, real `runc` sandbox lifecycle manager
  - Telemetry event/metric sink
- Versioned checkpoint manifest (`v1`) with integrity hash validation.
- Real `runc checkpoint` / `runc restore` execution with CRIU image directories captured as artifacts.
- Real ZFS snapshot / rollback execution for filesystem checkpoint and restore.
- In-memory eBPF event collector for tests plus an inspector API that derives change signals from eBPF events.
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
- `benchmarks/` - benchmarks
- `tests/` - unit + contract + simulated e2e tests
- `legacy/` - prior runtime benchmark scripts
