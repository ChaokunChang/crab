# Agent-CR

`agent_cr` is a Python library for checkpointing and restoring agent sandboxes.

The codebase currently supports two practical modes:

- A lightweight `docker` path used by unit and simulated integration tests. The adapter reports planned checkpoint/restore commands but does not execute a real container runtime.
- A real `runc` path that executes `runc`/CRIU for process state and ZFS snapshot or rollback for filesystem state.

## What The Library Provides

- `AgentCRSystem` as the top-level coordinator for checkpointing, manual restore, and recovery.
- `build_default_system(...)` for assembling the default in-process scheduler, executor, storage, telemetry, request tracking, and sandbox manager.
- `CRScheduler` plus policy implementations for:
  - Default checkpointing
  - Fault tolerance
  - Spot preemption
  - Tree search
- `CRExecutor`, `DefaultCWorker`, and `DefaultRWorker` for checkpoint and restore execution.
- Runtime adapters:
  - `RuncRuntimeAdapter` for real process and filesystem operations
  - `DockerRuntimeAdapter` as a compatibility/testing stub
- Sandbox managers:
  - `RuncSandboxManager`
  - `InMemorySandboxManager`
- Checkpoint storage:
  - `LocalCheckpointManager`
  - retention wrappers `KeepAllCheckpointManager`, `LatestOnlyCheckpointManager`, and `DeleteAfterRestoreCheckpointManager`
- Request interception and tracking:
  - `AgentCRRequestInterceptor`
  - `AgentCRRequestInterceptorServer`
  - `InMemoryRequestStateStore`
  - `SandboxResponseGateRegistry`
- Inspectors and telemetry:
  - `EBPFSandboxInspector`
  - `RequestAwareSandboxInspector`
  - `InMemoryTelemetrySink`

For Host inspector manual usage and real Docker validation, See `agent_cr/host_inspector/README.md`.

## Main Entry Point

```python
from pathlib import Path

from agent_cr import SchedulerConfig, StorageConfig, build_default_system

system = build_default_system(
    storage_root=Path("tmp/agent-cr"),
    runtime="docker",  # or "runc"
    storage_config=StorageConfig(root_dir=Path("tmp/agent-cr")),
    scheduler_config=SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
    ),
    enforce_restore_checkpoint_validation=False,
)
```

The default builder creates:

- `CRScheduler`
- `CRExecutor`
- `LocalCheckpointManager`
- `RequestAwareSandboxInspector`
- `SandboxResponseGateRegistry`
- `InMemoryRequestStateStore`
- `InMemoryTelemetrySink` by default

Use direct `AgentCRSystem(...)` construction when you need custom runtime paths, real-host wiring, retention wrappers, or a non-default scheduler policy.

## Checkpoint, Restore, And Recovery

### Manual Checkpoint

- `checkpoint_once(...)` pauses the sandbox, runs the checkpoint workers, records completion, and resumes when appropriate.
- `checkpoint_if_due(...)` asks the scheduler/policy whether a checkpoint should run and only executes when the decision says yes.

### Manual Restore

- `restore_once(...)` prepares the sandbox, resolves the restorable manifest, and runs filesystem restore before process restore.
- On success it marks the sandbox as restored and notifies storage.

### Recovery

- `notify_fault(...)` queues a fault recovery event.
- `notify_preemption(...)` records preemption metadata, queues a recovery event, and can trigger a fresh checkpoint before restore depending on policy.
- The recovery loop either restores a checkpoint or falls back to the configured `relaunch_handler`.

### Restore Validation Toggle

`_validate_restore_checkpoint(...)` still exists, but it is not enforced by default.

- Default: `enforce_restore_checkpoint_validation=False`
- When enabled:
  - `restore_once(...)` fails with a validation error if a live-request checkpoint no longer matches the pending response gate.
  - Recovery skips invalid live-request checkpoints and keeps scanning older checkpoints.

This is useful when you want strict coupling between restore eligibility and a still-pending intercepted LLM request. Leave it disabled if you want recovery to prefer availability over that validation.

## Runtime Notes

### `docker` mode

- Good for fast tests and local API-level validation.
- Does not perform a real runtime checkpoint or restore.
- Uses `InMemorySandboxManager`.

### `runc` mode

- Executes real runtime operations through `RuncRuntimeAdapter`.
- Uses `RuncSandboxManager`.
- Real-host scenarios require `docker`, `runc`, `criu`, `zfs`, and for the benchmark harness also `zpool`.

## Repository Layout

- `agent_cr/`: library code
- `tests/`: unit and integration tests
- `benchmarks/`: real-host and microbenchmark entrypoints
- `integrations/`: benchmark agents, LLM services, and sandbox integrations
- `docs/`: architecture notes
- `legacy/`: older scripts kept outside the current implementation path

## Tests

Run the full test suite:

```bash
python3 -m unittest discover -s tests -v
```

Focused modules used most often for restore and recovery work:

```bash
python3 -m unittest tests.test_system_integration tests.test_interceptor -v
```

## Benchmarks

Microbench:

```bash
python3 benchmarks/bench_agent_cr_micro.py --iters 1000 --storage-iters 200 --executor-jobs 64
```

Sandboxed simulated-agent benchmark:

```bash
python3 -m benchmarks.run --config benchmarks/examples/e2e.manual.yaml
```

Real-host recovery benchmarks:

```bash
python3 -m benchmarks.run --config benchmarks/examples/fault.manual.yaml
python3 -m benchmarks.run --config benchmarks/examples/fault.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/spot.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/tree.auto.yaml
```

Each benchmark run is now driven by a YAML config. The top-level fields are:

```yaml
scenario: fault | spot | tree | e2e
mode: manual | auto
provider: openai | anthropic
agent: simulated | iflow
llm_service: simulated | manual | simulated_for_iflow | iflow_trace_replay
task_dataset: path/to/tasks.jsonl
sandboxes: 1
iterations: 5
output: logs/tmp/out.csv
log_file: logs/tmp/out.log
log_file_mode: append | write
benchmark_root: logs/tmp/benchmark-runs
zpool_size: 10G
zpool_name: agentcrbench-cache
zpool_image: logs/tmp/bench.zpool.img
reuse_zpool: false
log_level: info
transfer_delay_ms: 0.0
work_dir_host_root: logs/tmp
scenario_options: {}
```

Logging notes:

- `log_file` sends benchmark logs to a file instead of stderr/stdout.
- `log_file_mode` controls the Python `FileHandler` mode.
- Default `log_file_mode: append` preserves existing log history.
- Use `log_file_mode: write` when you want each benchmark run to start with a fresh log file.
- `benchmark_root` places each run under a timestamped subdirectory rooted at the configured path. If omitted, benchmarks use a temporary directory. `AGENTCR_BENCH_DIR` is still accepted as a fallback for older workflows.
- `benchmark.run` now logs an explicit start marker and end marker for each run, and the final summary/artifact paths are logged as well as printed.
- `zpool_size` controls the backing file size for ephemeral benchmark zpools.
- `reuse_zpool: true` keeps the zpool across runs instead of recreating it every time.
- When reusing a pool, set both `zpool_name` and `zpool_image` to stable values. Each run still destroys and recreates the `pool/agent-cr` dataset so the benchmark starts clean.

The per-scenario knobs live under `scenario_options`:

```bash
python3 -m benchmarks.run --config benchmarks/examples/fault.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/spot.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/tree.auto.yaml
```

The real-host benchmarks allocate temporary runtime state, create a ZFS pool, build the simulated agent image, and launch `runc` sandboxes through the shared harness in [benchmarks/real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py).
