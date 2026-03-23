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

## Telemetry

Telemetry stays JSONL-based and is designed to be lightweight by default.

- The default shape is still a stream of `event` and `metric` records.
- New lifecycle instrumentation now emits correlated `*.start`, `*.finish`, and `*.duration_ms` records for:
  - benchmark tasks and verification
  - interceptor request handling and response-gate waits
  - benchmark LLM service handling
  - scheduler evaluation
  - executor queueing and job execution
  - checkpoint/restore flows and process/filesystem sub-steps
  - `runc` and ZFS command execution
- Correlation attributes are attached where available, including `run_id`, `sandbox_id`, `task_id`, `request_id`, `job_id`, `checkpoint_id`, `event_type`, and `component`.

`TelemetryConfig` now supports a few knobs:

```python
from pathlib import Path

from agent_cr import TelemetryConfig, build_default_system

system = build_default_system(
    storage_root=Path("tmp/agent-cr"),
    telemetry_config=TelemetryConfig(
        enabled=True,
        jsonl_path=Path("tmp/agent-cr/telemetry.jsonl"),
        keep_in_memory_copy=True,
        detail_level="basic",  # or "detailed"
        capture_command_output=False,
        max_text_attribute_bytes=2048,
    ),
)
```

Notes:

- `detail_level="basic"` is the default and is intended for normal benchmark runs.
- `detail_level="detailed"` preserves the same event/metric model but allows richer attributes for deeper analysis.
- `capture_command_output=False` avoids storing command stdout/stderr unless you explicitly want it.

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
max_workers: 32  # legacy fallback for all phases when phase_workers is omitted
phase_workers:
  build: 16
  prepare: 16
  run: 32
  verification: 8
iterations: 5
output: logs/tmp/out.csv
log_file: logs/tmp/out.log
log_file_mode: append | write
benchmark_root: logs/tmp/benchmark-runs
telemetry_output: logs/tmp/out.telemetry.jsonl  # legacy top-level form, still supported
telemetry:
  output: logs/tmp/out.telemetry.jsonl
  detail_level: basic | detailed
  capture_command_output: false
  max_text_attribute_bytes: 2048
zpool_size: 10G
zpool_name: agentcrbench-cache
zpool_image: logs/tmp/bench.zpool.img
reuse_zpool: false
log_level: info
transfer_delay_ms: 0.0
work_dir_host_root: logs/tmp
scenario_options: {}
llm_service_options: {}  # merged into per-task llm_service_config
```

Benchmark runs now use a four-phase pipeline:

- `build`: shared image/materialization work
- `prepare`: bundle/rootfs/workdir/network setup before `runc` starts
- `run`: sandbox start plus scenario workload/recovery logic
- `verification`: post-run validation

The runner enforces two barriers:

- all sandboxes must finish `build` and `prepare` before any sandbox enters `run`
- all sandboxes must finish `run` before any sandbox enters `verification`

`phase_workers` lets each phase use a different concurrency limit. When `phase_workers` is omitted, or when a phase key is missing, that phase falls back to `max_workers` and then to `sandboxes`.

Example:

```yaml
sandboxes: 100
max_workers: 32
phase_workers:
  build: 16
  prepare: 16
  run: 32
  verification: 8
```

This prepares all 100 sandboxes first, then starts the run phase with at most 32 concurrent run workers, and only starts verification after every run-phase task is complete.

Benchmark artifacts now include both:

- a CSV output with benchmark-level outcome fields such as `success_ratio` and `lost_actions`
- a telemetry JSONL stream used for system timing analysis and summary reconstruction

The benchmark runner is now telemetry-first for timing summaries:

- if the telemetry file for the run contains the required timing metrics, printed summaries are computed from telemetry
- if not, the runner falls back to row-based summary computation for compatibility
- benchmark-derived outcome metrics such as `success_ratio_avg` and `lost_actions_avg_avg` are still preserved

This removes the old replay-mode `avg of avg` summary issue for timing metrics like checkpoint, restore, and recovery latency.

Telemetry analysis and visualization:

```bash
python3 -m agent_cr.telemetry_analysis.report \
  --input logs/iflow.fault.auto.minimax_hard_14tasks.debug.telemetry.jsonl \
  --output-dir logs/iflow.fault.auto.minimax_hard_14tasks.telemetry_report
```

The analyzer is streaming-oriented and is intended for large JSONL files. It does not load the full telemetry file into memory. The generated output directory contains:

- `report.html`: self-contained visual report with hotspot charts, task latency charts, LLM/checkpoint breakdown views, and lifecycle-gap diagnostics
- `summary.json`: machine-readable aggregate report
- `operation_summary.csv`: per-operation counts and latency quantiles
- `task_summary.csv`: per-task benchmark and recovery metrics
- `slow_operations.csv`: slowest recorded operations with correlation identifiers
- `lifecycle_gaps.csv`: operations where `*.start` and `*.finish` counts do not match

Logging notes:

- `log_file` sends benchmark logs to a file instead of stderr/stdout.
- `log_file_mode` controls the Python `FileHandler` mode.
- Default `log_file_mode: append` preserves existing log history.
- Use `log_file_mode: write` when you want each benchmark run to start with a fresh log file.
- `benchmark_root` places each run under a timestamped subdirectory rooted at the configured path. If omitted, benchmarks use a temporary directory. `AGENTCR_BENCH_DIR` is still accepted as a fallback for older workflows.
- `benchmark.run` now logs an explicit start marker and end marker for each run, and the final summary/artifact paths are logged as well as printed.
- Benchmark YAML supports a nested `telemetry:` block for telemetry output and detail controls.
- `telemetry.output` sets the JSONL artifact path. If omitted, the runner defaults to `<output>.telemetry.jsonl` or `<config>.telemetry.jsonl`.
- `telemetry.detail_level` accepts `basic` or `detailed`.
- `telemetry.capture_command_output` is `false` by default to avoid storing command stdout/stderr in normal runs.
- `telemetry.max_text_attribute_bytes` bounds long text attributes when detailed capture is enabled.
- The legacy top-level `telemetry_output` field is still accepted for compatibility, but `telemetry.output` is the preferred YAML form.
- `phase_workers` overrides concurrency per benchmark phase. Missing phase keys fall back to `max_workers`.
- Phase telemetry now emits distinct phase-qualified records such as `benchmark.phase.build.*`, `benchmark.phase.prepare.*`, `benchmark.phase.run.*`, `benchmark.phase.verification.*`, and `benchmark.phase.<phase>.item.*` so JSONL output shows phase timing and configured concurrency.
- `zpool_size` controls the backing file size for ephemeral benchmark zpools.
- `reuse_zpool: true` keeps the zpool across runs instead of recreating it every time.
- When reusing a pool, set both `zpool_name` and `zpool_image` to stable values. Each run still destroys and recreates the `pool/agent-cr` dataset so the benchmark starts clean.

### LLM Service Options

`llm_service_options` is a top-level YAML block whose keys are merged into each task's `llm_service_config` (dataset-level values take precedence). This is useful for controlling replay behavior globally without editing the dataset JSONL.

For `iflow_trace_replay`, the following options control how response delays are simulated:

- `response_delay_policy`: selects how the replay service simulates LLM response latency.
  - `fixed` (default): uses a constant delay of `response_delay_ms` milliseconds (default 250ms).
  - `trace_replay`: uses the actual request-to-response timestamps recorded in the trajectory, scaled by `response_delay_scaling_factor`.
- `response_delay_ms`: constant delay in milliseconds, used by the `fixed` policy and as a fallback when `trace_replay` timestamps are unavailable (default 250).
- `response_delay_scaling_factor`: multiplier applied to trace-derived delays when `response_delay_policy` is `trace_replay` (default 1.0). A value of 0.5 replays at 2× speed; 2.0 replays at half speed.

Example YAML:

```yaml
llm_service_options:
  response_delay_policy: trace_replay
  response_delay_scaling_factor: 1.0
```

The per-scenario knobs live under `scenario_options`:

```bash
python3 -m benchmarks.run --config benchmarks/examples/fault.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/spot.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/tree.auto.yaml
```

The real-host benchmarks allocate temporary runtime state, create a ZFS pool, build the simulated agent image, and launch `runc` sandboxes through the shared harness in [benchmarks/real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py).
