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

Two core tuning knobs changed recently:

- `SchedulerConfig.inspect_without_pause` now exists and defaults to `False`. The default path still pauses before inspecting, which is the safer option when live inspection of a running sandbox may be stale or unsafe.
- `ExecutorConfig` now supports separate `checkpoint_workers` and `restore_workers`. When either is omitted, it falls back to the legacy `max_workers` value.

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
- Correlation attributes are attached where available, including `run_id`, `sandbox_id`, `task_run_id`, `task_id`, `request_id`, `job_id`, `checkpoint_id`, `event_type`, and `component`.

`TelemetryConfig` now supports a few knobs:

```python
from pathlib import Path

from agent_cr import TelemetryConfig, build_default_system

system = build_default_system(
    storage_root=Path("tmp/agent-cr"),
    telemetry_config=TelemetryConfig(
        enabled=True,
        jsonl_path=Path("tmp/agent-cr/telemetry.jsonl"),
        keep_in_memory_copy=False,
        detail_level="basic",  # or "detailed"
        capture_command_output=False,
        max_text_attribute_bytes=2048,
        writer_mode="async",
        queue_capacity=16384,
        batch_max_records=256,
        flush_interval_ms=50,
        overflow_policy="drop_new",
        serializer="auto",
    ),
)
```

Notes:

- `detail_level="basic"` is the default and is intended for normal benchmark runs.
- `detail_level="detailed"` preserves the same event/metric model but allows richer attributes for deeper analysis.
- `capture_command_output=False` avoids storing command stdout/stderr unless you explicitly want it.
- With `jsonl_path` set, `keep_in_memory_copy` now defaults to `False` unless you explicitly enable it.
- `writer_mode="async"` is the default JSONL mode and keeps telemetry writes off the request/checkpoint hot path with a bounded queue.
- `overflow_policy="drop_new"` is the default async backpressure policy. Dropped telemetry is summarized at shutdown.
- JSONL writes are batched and still file-locked, so the benchmark harness and benchmark LLM router subprocess can safely share one telemetry file.

## Checkpoint, Restore, And Recovery

### Manual Checkpoint

- `checkpoint_once(...)` pauses the sandbox, runs the checkpoint workers, records completion, and resumes when appropriate.
- `checkpoint_if_due(...)` asks the scheduler/policy whether a checkpoint should run and only executes when the decision says yes.

### Incremental Process Checkpoints (opt-in)

Enable with `SchedulerConfig.incremental_process_enabled=True` (or the matching benchmark YAML field). The setting applies to every scheduler policy — default, fault tolerance, spot preemption, and tree search.

When enabled, each chain participant runs a CRIU pre-dump + final-dump pair:

1. `runc checkpoint --pre-dump --image-path <ckpt>/pre_dump [--parent-path ../<prev>/pre_dump]` — writes only memory pages dirtied since the parent pre-dump (or every page if this is an anchor). Process keeps running, soft-dirty page tracking stays armed.
2. `runc checkpoint --image-path <ckpt>/process --parent-path ../pre_dump --leave-running=<bool>` — full process tree state plus the small delta accumulated between the pre-dump and freeze. This is the restorable artifact.

Anchors and chain nodes use the same shape; the only difference is that an anchor's pre-dump has no `--parent-path`. A new anchor is emitted automatically every `full_process_checkpoint_interval` checkpoints, or earlier if `max_process_chain_length` is hit. The very first checkpoint per sandbox is always an anchor.

Both knobs cap chain length but they exist for different reasons:

- `full_process_checkpoint_interval` (default `8`) is the **policy knob**. Tune it to balance restore cost (longer chains = more pre-dump dirs to walk on restore, more chained pages to apply) against per-checkpoint savings (longer chains = fewer expensive anchor dumps).
- `max_process_chain_length` (default `16`) is the **safety cap**. It exists for cases where the in-memory state store mis-tracks the chain length — a missed `record_process_checkpoint`, a corrupted counter after a recovery, a future bug that bypasses the interval check. Without it, any such bug would let chains grow unbounded; the cap puts a hard ceiling on blast radius regardless of policy logic. In normal operation the interval always trips first and the cap is dead code; it is load-bearing only when something is wrong.

Whichever check fires first forces a fresh anchor.

What counts toward the chain-length counter (`InMemorySchedulerStateStore._process_chain_length`):

- **Successful chain nodes** — increment by 1.
- **Successful anchors** — reset the counter to 0.
- **Skipped checkpoints** (scheduler decided `should_checkpoint=False`, or policy returned early) — do not count. `mark_checkpoint_complete` is never called.
- **Filesystem-only checkpoints** (`checkpoint_process=False`) — do not count. The system passes `process_checkpoint_id=None` and the chain-length update is gated on a non-`None` process id.
- **Failed process checkpoints** — do not count. `mark_checkpoint_complete` is only called when `result.status == "succeeded"`.

So both knobs measure only **successful, process-bearing** chain nodes since the last anchor. The decision rule is `next_chain_length = current + 1; if next_chain_length >= full_process_checkpoint_interval or next_chain_length >= max_process_chain_length, emit a fresh anchor`.

Restore is unchanged — CRIU walks `parent` symlinks each pre-dump dir already contains. The system additionally validates the chain on disk before invoking restore: missing intermediate pre-dump directories surface as `FileNotFoundError` naming the missing checkpoint id rather than producing a corrupted sandbox.

Retention is chain-aware:

- `LocalCheckpointManager.delete_checkpoint(...)` rejects deleting a parent that has live descendants unless `cascade=True` is passed. `delete_all_checkpoints(...)` iterates leaves first so the check never trips.
- `LatestOnlyCheckpointManager` extends "latest" to mean the entire `parent_checkpoint_id` chain — every ancestor of a protected checkpoint stays on disk so a mid-chain restore is always satisfiable. Old fulls and completed chains are still evicted as soon as the active chain advances past them.
- `KeepAllCheckpointManager` and `DeleteAfterRestoreCheckpointManager` keep their existing semantics (the latter already iterates newest-first on restore-complete, so the chain check passes naturally).

`CheckpointManifest` gains `parent_checkpoint_id` and `process_kind ∈ {full, incremental}`. These fields are only written into canonical JSON when non-default, so legacy v1 manifests still validate against their stored integrity hashes.

A self-contained example pair lives at `benchmarks/examples/terminus/terminus.nofault.auto.incremental_demo.{baseline,incremental}.yaml`. On a curated 8-task subset of the terminus replay dataset, enabling incremental cut total process-dump bytes by 99%, the freeze-blocking final dump's mean latency by 44%, and end-to-end checkpoint flow p90 by 35%, while wall-clock and `success_ratio` stayed identical to the baseline. Memory-heavy workloads (statistical models, ML training) benefit most; filesystem-bound workloads see no change because the filesystem path was already ZFS-incremental.

### Manual Restore

- `restore_once(...)` prepares the sandbox, resolves the restorable manifest, and runs filesystem restore before process restore.
- On success it marks the sandbox as restored and notifies storage.

### Recovery

- `notify_fault(...)` queues a fault recovery event.
- `notify_preemption(...)` records preemption metadata, queues a recovery event, and can trigger a fresh checkpoint before restore depending on policy.
- The recovery loop restores a checkpoint when one is available; the relaunch fallback is opt-in.

### Relaunch Fallback Toggle

`AgentCRSystem.relaunch_on_restore_failure` is `False` by default.

- When `False` (the default): a recovery restore failure surfaces as a hard error in the recovery record (`status="failed"`) and `relaunch_handler` is not invoked. The same applies when no restorable checkpoint is available: status becomes `"no_checkpoint"`.
- When `True`: recovery falls back to `relaunch_handler` after a restore failure (or when no checkpoint exists), which preserves the previous availability behavior.

The relaunch path is intentionally off by default because it can mask real bugs — a corrupt checkpoint, a broken restore plumbing change, or a misconfigured baseline policy all silently degrade to "relaunched" otherwise. Pair this default with `scheduler.checkpoint_full_baseline_on_first_checkpoint=true` to make every recovery use a complete checkpoint and treat any restore failure as a regression to investigate.

The benchmark harness exposes the same opt-in via the top-level `relaunch_on_restore_failure: true | false` YAML field, threaded through to the underlying `AgentCRSystem`.

### Restore Validation Toggle

`_validate_restore_checkpoint(...)` still exists, but it is not enforced by default.

- Default: `enforce_restore_checkpoint_validation=False`
- When enabled:
  - `restore_once(...)` fails with a validation error if a live-request checkpoint no longer matches the pending response gate.
  - Recovery skips invalid live-request checkpoints and keeps scanning older checkpoints.

This is useful when you want strict coupling between restore eligibility and a still-pending intercepted LLM request. Leave it disabled if you want recovery to prefer availability over that validation.

### Response Gating Guarantees

When response gating is enabled:

- pending interceptor responses are tracked per sandbox and per request generation, not just by sandbox
- overlapping requests from the same sandbox stay isolated from each other during coordination and recovery
- a buffered response is released only after that exact request generation has finished coordination and any submitted checkpoint work for it has completed or been skipped

For Claude Code replay workloads, response gating is now request-kind aware:

- `main_loop` Anthropic `POST /v1/messages` requests remain fully gated and checkpoint-visible
- auxiliary `helper` requests and `count_tokens` probes are tagged in interceptor metadata but bypass response gating
- live-request checkpoint metadata is only captured for gated requests, so auxiliary Claude traffic no longer creates misleading checkpoint windows or inflated `llm.gate_wait_ms`

That same request-generation metadata is also stored in live-request checkpoints, so restore-time response release can target the matching buffered response without accidentally releasing a newer one.

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
python3 -m benchmarks.run --config benchmarks/examples/mini_swe/mini_swe.spec.auto.10tasks.debug.yaml
python3 -m benchmarks.run --config benchmarks/examples/terminus/terminus.fault.auto.10tasks.debug.yaml
python3 -m benchmarks.run --config benchmarks/examples/terminus/terminus.spec.auto.10tasks.debug.yaml
```

Each benchmark run is now driven by a YAML config. The top-level fields are:

```yaml
scenario: fault | spot | tree | e2e | spec
mode: manual | auto
provider: openai | anthropic
agent: simulated | iflow | mini_swe | claude_code | terminus
llm_service: simulated | manual | simulated_for_iflow | iflow_trace_replay | mini_swe_trace_replay | mini_swe_spec_trace_replay | claude_code_trace_replay | terminus_trace_replay | terminus_spec_trace_replay
task_dataset: path/to/tasks.jsonl
sandboxes: 1
max_workers: 32  # legacy fallback for all phases when phase_workers is omitted
phase_workers:
  setup: 16
  run: 32
  verification: 8
rootfs_reuse:
  enabled: true
phase_merging:
  setup_and_run: false
  setup_and_run_executor_pool: separate | shared
sandbox_resource_limits:
  cpus: null           # int OCI CPU-quota cap (enforced via cgroup cpu.max)
  memory_bytes: null   # int cgroup memory.limit
  pids_limit: null     # int cgroup pids.limit
  cpu_period_us: 100000
executor:
  checkpoint_workers: 32
  restore_workers: 32
  coordination_workers: 8
  composite_step_workers: 16
  checkpoint_queue_size: 10000
  checkpoint_scheduling_policy: fifo | reactive
  reactive_checkpoint_urgent_quota: 4
  max_retries: 0
  retry_backoff_seconds: 0.05
scheduler:
  policy: scenario_default | no_checkpointing
  min_checkpoint_interval_seconds: 0.0
  force_checkpoint_after_seconds: 0.0
  require_change_signal: true
  checkpoint_full_baseline_on_first_checkpoint: false
  prefer_checkpoint_during_llm_request: true
  require_llm_request_for_checkpoint: false
  inspect_without_pause: false
  incremental_process_enabled: false
  full_process_checkpoint_interval: 8
  max_process_chain_length: 16
llm_server:
  launch_mode: process
host_inspector:
  launch_mode: process
  log_level: INFO       # DEBUG | INFO | WARNING | ERROR | CRITICAL
  log_file: false       # when true, writes to <benchmark_run_root>/host-inspector.log
iterations: 5  # spec requires exactly 0 because it replays full traces
output: logs/tmp/out.csv
log_file: logs/tmp/out.log
log_file_mode: append | write
verification:
  enabled: true
benchmark_root_home: logs/tmp/benchmark-runs
benchmark_run_name: null  # defaults to a timestamp such as 20260416_010203
clear_benchmark_root_after_run: false
storage_planes:
  runtime_root: /mnt/agent-cr-runtime
  storage_root: /mnt/agent-cr-runtime/storage
  agent_host_root: /mnt/agent-cr-runtime/agents
runtime_command_timeout_seconds: 60.0
runtime_zfs_prepare_timeout_seconds: 300.0
telemetry_output: logs/tmp/out.telemetry.jsonl  # legacy top-level form, still supported
telemetry:
  output: logs/tmp/out.telemetry.jsonl
  file_mode: append | write
  detail_level: basic | detailed
  capture_command_output: false
  max_text_attribute_bytes: 2048
  keep_in_memory_copy: false
  writer_mode: async | sync
  queue_capacity: 16384
  batch_max_records: 256
  flush_interval_ms: 50
  overflow_policy: drop_new | block
  serializer: auto | stdlib | orjson
  report:
    enabled: true
    output_dir: logs/tmp/out.telemetry.report
    top_k: 25
    log_scale_charts: false
    export_svg: true
monitoring:
  enabled: true
  sample_interval_ms: 1000
  include_host: true
  include_sandboxes: true
zpool_size: 10G
zpool_name: agentcrbench-cache
zpool_image: logs/tmp/bench.zpool.img
reuse_zpool: false
image_cache_root: logs/tmp/image-cache
log_level: info
transfer_delay_ms: 0.0
work_dir_host_root: logs/tmp
max_agent_timeout_scale: 1.0
max_test_timeout_scale: 1.0
relaunch_on_restore_failure: false  # opt-in fallback to relaunch_handler when a recovery restore fails
scenario_options: {}
llm_service_options: {}  # merged into per-task llm_service_config
```

For the fault scenario, `scenario_options` also supports:

- `delete_filesystem_checkpoints: false` by default. Keeping filesystem checkpoints avoids synchronous old-snapshot deletion in the run-phase checkpoint hot path. Set it to `true` only when you explicitly want more aggressive retention cleanup.
- Latest-only retention cleanup now runs asynchronously in the background. Checkpoint completion only schedules the cleanup work instead of waiting for old-snapshot deletion inline.

For the speculative-execution scenario (`scenario: spec`):

- `agent` must be `mini_swe` or `terminus`
- `llm_service` must be the matching `<agent>_spec_trace_replay` (i.e. `mini_swe_spec_trace_replay` or `terminus_spec_trace_replay`)
- `iterations` must be exactly `0`
- `scenario_options.acceptance_rate` controls how often the draft command is accepted. Legacy alias: `accept_rate`.
- `scenario_options.draft_response_delay_scaling_factor` controls how much faster the draft replay stream is than the oracle replay stream. Legacy alias: `speculative_delay_scaling_factor`.
- `scenario_options.mismatch_policy` currently supports only `preserve_command_class`.
- `scenario_options.enable_fork_reuse` defaults to `false`. When enabled, a finalized fork whose active and fork sandboxes both report `state_unchanged` and whose draft exec had completed by oracle-finish time is cached for the next turn, so the next speculative step skips a ZFS clone + CRIU restore. The cache is invalidated on any sandbox restore/recovery and is per-sandbox, so task boundaries always miss.
- `scenario_options.eager_fork_cleanup_on_reject` defaults to `false`. When enabled, rejected speculative turns tear the fork down immediately rather than letting the draft exec finish in the background — bounding the hidden CPU penalty to a short drain window instead of the full draft-exec tail. Forks are never reused after a reject in this mode.
- Three fork-prep optimizations cut speculative `fork_restore_ms` further; all default `false` and are independently composable. They build on `scheduler.incremental_process_enabled=true` (chain ancestors must exist to be shareable):
  - `scenario_options.enable_fork_chain_sharing` (Phase B) — replaces per-fork `shutil.copytree` of every chain ancestor's CRIU image with relative symlinks (`Runtime.link_ancestor_pre_dump`) and pin-refcounts the chain (`Storage.pin_chain`) so retention can't prune ancestors mid-fork. On promote, the fork inlines its borrowed bytes before the source is destroyed. **Recommended default-on**: cleanest single optimization, ~−40 % mean fork latency and ~−3 % wall-clock with no observed cost.
  - `scenario_options.enable_lazy_restore` (Phase D) — plumbs `runc restore --lazy-pages` and spawns the `criu lazy-pages` daemon ahead of restore; pages stream in via userfaultfd. Requires runc 1.3+ with `--lazy-pages` support, CRIU 4.x, and kernel userfaultfd (`unprivileged_userfaultfd=1` or root + `CAP_SYS_PTRACE`). **Recommended default-on** for spec scenarios when the platform supports it: −19 % mean fork latency at −1.5 % wall-clock.
  - `scenario_options.enable_background_prefork` (Phase A) — speculatively warms a fork after each successful checkpoint via a dedicated executor; `ensure_fork()` returns from cache. Pair with `prefork_max_concurrent_global` (recommended `2`) and `prefork_min_interval_seconds` (recommended `2.0`) to keep background warming from saturating disk I/O. **Opt-in only**: even with the throttle, A's I/O contention costs ~+8 % wall-clock on the 8-task spec subset and a single-sandbox smoke confirms the cost is intrinsic (+17 % wall-clock with one sandbox), not from cross-sandbox queueing. Use when per-turn `fork_restore_ms` jitter matters more than aggregate wall-clock; skip when the workload is I/O-bound on its own or wall-clock is the optimization target.
- A self-contained 6-variant comparison set lives at [`benchmarks/examples/terminus/terminus.spec.auto.incremental_demo.{baseline,chain_sharing,prefork,lazy,b_plus_d,all_opts}.yaml`](/root/workspace/agent-cr/benchmarks/examples/terminus) with helper [`incremental_demo_compare.py`](/root/workspace/agent-cr/benchmarks/examples/terminus/incremental_demo_compare.py). On the 8-task subset, B+D combined cuts `fork_restore_ms` mean by 18 %; all_opts (B+A+D throttled) cuts p99 from 2.85 s to 724 ms at +5.7 % wall-clock cost.
- [docs/speculative-execution-benchmark.md](/root/workspace/agent-cr/docs/speculative-execution-benchmark.md) documents the execution model, telemetry, fork-prep optimizations, and tracked example/evaluation configs.
- [docs/incremental-fork-restore-analysis.md](/root/workspace/agent-cr/docs/incremental-fork-restore-analysis.md) walks through the 6-variant benchmark in detail: per-task wall-clock decomposition, single-sandbox smoke confirmation, why each optimization succeeds or fails, and when to enable which.

Benchmark runs now use a three-phase pipeline:

- `setup`: shared image/materialization work, bundle/rootfs/workdir/network setup, sandbox launch, and any readiness that must complete before the benchmark task starts
- `run`: scenario workload/recovery logic after sandbox setup has completed
- `verification`: post-run validation when `verification.enabled` is `true`

By default, the runner enforces hard barriers:

- all sandboxes must finish `setup` before any sandbox enters `run`
- all sandboxes must finish `run` before any sandbox enters `verification`

`phase_merging.setup_and_run: true` relaxes only the first barrier for per-sandbox scenario flows (`fault`, `spot`, `tree`, and replay-style `e2e`). In that mode, each sandbox can enter `run` as soon as its own `setup` finishes. The verification barrier remains unchanged, and cohort-style non-replay `e2e` still uses the old setup barrier.

`phase_merging.setup_and_run_executor_pool` controls how merged setup/run work is scheduled. `separate` keeps today's behavior with independent setup and run executor pools. `shared` submits one combined `setup+run` task per sandbox to a single executor pool sized by `min(phase_workers.setup, phase_workers.run)`.

If `verification.enabled: false`, the verification phase is skipped entirely.

`phase_workers` lets each phase use a different concurrency limit. When `phase_workers` is omitted, or when a phase key is missing, that phase falls back to `max_workers` and then to `sandboxes`.

Example:

```yaml
sandboxes: 100
max_workers: 32
phase_workers:
  setup: 16
  run: 32
  verification: 8
```

This fully sets up all 100 sandboxes first, then starts the run phase with at most 32 concurrent run workers, and only starts verification after every run-phase task is complete.

`rootfs_reuse.enabled` defaults to `true`. In real-host ZFS-backed benchmarks, the harness now materializes a shared base rootfs per normalized recipe, snapshots it once, and clones that snapshot for each sandbox instead of copying a full rootfs into every sandbox dataset. For compose-backed tasks, the recipe is anchored by `docker_compose_file`, `service_name`, `agent_type`, and normalized rootfs materialization inputs. When `reuse_zpool: true`, compose-backed shared rootfs bases persist in the reused pool across benchmark runs; otherwise reuse is limited to the current benchmark run.

The benchmark YAML also exposes run-phase tuning for the core Agent-CR system:

- `executor.checkpoint_workers` and `executor.restore_workers` split checkpoint and restore concurrency. If omitted, both inherit the benchmark's effective `max_workers`.
- `executor.coordination_workers` bounds live-request coordination threads, and `executor.composite_step_workers` bounds the shared worker pool used for parallel process/filesystem checkpoint sub-steps.
- `executor.checkpoint_queue_size`, `executor.max_retries`, and `executor.retry_backoff_seconds` tune checkpoint admission and retry behavior.
- `scheduler` overrides only `SchedulerConfig` fields. The scenario still chooses the policy class (`default`, `fault-tolerance`, `spot-preemption`, or `tree-search`), and the YAML block merges onto that scenario-owned default field by field.
- `scheduler.policy` defaults to `scenario_default`. Set it to `no_checkpointing` only when you explicitly want a scenario to run without checkpoint capture.
- `scheduler.checkpoint_full_baseline_on_first_checkpoint` forces the first checkpoint for a sandbox to include both process and filesystem state even when the scheduler would otherwise choose a narrower scope.
- `scheduler.inspect_without_pause` is opt-in and defaults to `false`. Turning it on allows the scheduler to inspect before pausing, but that is intentionally not the default because live inspection of a running sandbox can be risky depending on the inspector.
- `scheduler.incremental_process_enabled` is opt-in and defaults to `false`. When enabled, every chain participant becomes a `runc checkpoint --pre-dump` followed by a final `runc checkpoint --parent-path`, so non-anchor checkpoints write only memory pages dirtied since the parent pre-dump. Restore is unchanged — CRIU walks the chain through the `parent` symlink it writes itself. Requires a runtime that advertises `supports_incremental_process=True` (the `runc` runtime does; the in-memory test runtime does not). When the runtime does not support it, the worker logs a warning and falls back to a single full dump.
- `scheduler.full_process_checkpoint_interval` defaults to `8`. After this many checkpoints in a chain (anchor counted as #1), the next checkpoint becomes a fresh anchor that resets the chain. Bound it small enough to keep restore-time chain walks short and to limit blast radius if any pre-dump dir is lost.
- `scheduler.max_process_chain_length` defaults to `16`. Hard safety cap. Whichever of `full_process_checkpoint_interval` and `max_process_chain_length` triggers first wins; the cap exists so a state-store mis-track can't grow chains indefinitely.
- `llm_server.launch_mode` defaults to `process`, which runs the benchmark LLM router in a separate process to reduce thread pressure in the main benchmark process. `thread` remains available for tests and debugging.
- `host_inspector.launch_mode` also defaults to `process`, which keeps the host inspector and its filesystem-monitor threads out of the main benchmark process.
- `host_inspector.log_level` defaults to `INFO`. Set to `DEBUG` to log every eBPF filesystem event and every register/status/reset call — useful for diagnosing missed filesystem-change signals.
- `host_inspector.log_file` defaults to `false`. Set to `true` to write host-inspector logs to `<benchmark_run_root>/host-inspector.log` in addition to stderr. Enable both `log_level: DEBUG` and `log_file: true` when diagnosing checkpoint-related bugs such as stale forks or missed writes.
- The benchmark request path still goes through HTTP on `localhost`; only the router launch mode changed. Benchmark timings therefore still include interceptor-to-router transport overhead.
- `storage_planes` lets you move run-phase host writes off the benchmark artifact root. This is especially important when `zpool_image` is a file vdev: keeping CRIU checkpoint images, storage manifests/artifacts, exported rootfs state, and agent host directories on a different filesystem/device avoids sibling host writes contending with the ZFS backing file.
- `storage_planes` is fully opt-in. If you omit it, the harness preserves the legacy layout under the benchmark run root.
- `runtime_command_timeout_seconds` and `runtime_zfs_prepare_timeout_seconds` tune the runtime command and ZFS materialization timeouts used by the real-host harness.
- `image_cache_root` overrides the host directory used for cached benchmark images.

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
python3 -m benchmarks.telemetry_analysis.report \
  --input logs/iflow.fault.auto.minimax_hard_14tasks.debug.telemetry.jsonl \
  --output-dir logs/iflow.fault.auto.minimax_hard_14tasks.telemetry_report \
  --figure-window-seconds 10
```

The analyzer is streaming-oriented and is intended for large JSONL files. It does not load the full telemetry file into memory. The generated output directory contains:

- `report.html`: self-contained visual report with hotspot charts, checkpoint analysis, restore analysis, resource-usage views, and lifecycle-gap diagnostics
- `summary.json`: machine-readable aggregate report
- `operation_summary.csv`: per-operation counts and latency quantiles
- `task_summary.csv`: per-task benchmark and recovery metrics
- `slow_operations.csv`: slowest recorded operations with correlation identifiers
- `lifecycle_gaps.csv`: operations where `*.start` and `*.finish` counts do not match
- `checkpoint_analysis.csv`, `checkpoint_per_task.csv`, `checkpoint_load.csv`
- `restore_analysis.csv`, `restore_per_task.csv`, `restore_load.csv`
- `resource_summary.csv`, `resource_samples.csv`
- standalone SVG figure files for hotspot, checkpoint, restore, and resource charts
- when speculative-execution metrics are present, extra SVGs and an HTML section for saved time, agent-loop penalty, hidden reject cost, net gain, and accept rate
- when speculative runs emit fork-reuse attributes, an additional `spec_fork_reuse.csv` and a `Fork Reuse` subsection inside the speculative report, including a funnel (total → finalized → state-unchanged → cache-eligible → reused), gap attributions (state-unchanged → cache-eligible, cache-eligible → reused), and an accept/reject × state-unchanged/changed outcome matrix

Report CLI notes:

- `--figure-window-seconds <seconds>` averages checkpoint and restore line charts within fixed-size time windows. This applies to load-over-time and latency-over-time figures and helps smooth dense traces from large runs.
- Omit `--figure-window-seconds` or pass `0` to keep the raw point-by-point checkpoint/restore figures.

Logging notes:

- `log_file` sends benchmark logs to a file instead of stderr/stdout.
- `log_file_mode` controls the Python `FileHandler` mode.
- Default `log_file_mode: append` preserves existing log history.
- Use `log_file_mode: write` when you want each benchmark run to start with a fresh log file.
- `verification.enabled` defaults to `true`. Set it to `false` to skip the benchmark verification phase and omit verification fields from output rows.
- `benchmark_root_home` is the parent directory for benchmark runs. `benchmark_run_name` is the run directory under that parent; when omitted, it defaults to a timestamp such as `20260416_010203`. The stale `benchmark_root` key is still accepted as an alias for `benchmark_root_home`. If no home is configured, benchmarks use a temporary directory; an explicit `benchmark_run_name` requires a home from config or `AGENTCR_BENCH_DIR`. `AGENTCR_BENCH_DIR` is still accepted as a fallback for older workflows.
- `clear_benchmark_root_after_run` defaults to `false`. When enabled, the runner deletes the resolved per-run benchmark directory after postprocessing completes. Tempdir-backed runs keep their existing automatic cleanup behavior.
- The resolved benchmark run root is still the benchmark artifact root and the default home for runtime bundles, runtime checkpoint images, sandbox metadata, checkpoint storage, exported image rootfs, and agent host state. Use `storage_planes` only when you want to move some of that hot write traffic elsewhere.
- `output`, `log_file`, `telemetry.output`, and `telemetry.report.output_dir` still write to their configured paths. After the run, existing files/directories are also copied by basename into the resolved benchmark run root.
- `benchmark.run` now logs an explicit start marker and end marker for each run, and the final summary/artifact paths are logged as well as printed.
- Benchmark YAML supports a nested `telemetry:` block for telemetry output and detail controls.
- `telemetry.output` sets the JSONL artifact path. If omitted, the runner defaults to `<output>.telemetry.jsonl` or `<config>.telemetry.jsonl`.
- `telemetry.file_mode` defaults to `append`. Use `write` to remove any existing telemetry JSONL once at run start before the harness and router append fresh records.
- `telemetry.detail_level` accepts `basic` or `detailed`.
- `telemetry.capture_command_output` is `false` by default to avoid storing command stdout/stderr in normal runs.
- `telemetry.max_text_attribute_bytes` bounds long text attributes when detailed capture is enabled.
- `telemetry.keep_in_memory_copy` defaults to `false` for benchmark runs when JSONL output is enabled.
- `telemetry.writer_mode`, `telemetry.queue_capacity`, `telemetry.batch_max_records`, `telemetry.flush_interval_ms`, `telemetry.overflow_policy`, and `telemetry.serializer` control the async JSONL writer.
- `telemetry.report.enabled` controls automatic report generation after the benchmark run.
- `telemetry.report.output_dir` overrides the default `<telemetry_output>.report` directory.
- `telemetry.report.top_k`, `telemetry.report.log_scale_charts`, and `telemetry.report.export_svg` tune report output.
- The legacy top-level `telemetry_output` field is still accepted for compatibility, but `telemetry.output` is the preferred YAML form.
- The legacy top-level `telemetry_file_mode` field is also accepted for compatibility, but `telemetry.file_mode` is the preferred YAML form.
- In `llm_server.launch_mode: process`, the router subprocess writes telemetry with the same `run_id` into the same JSONL file as the harness.
- `monitoring.enabled` turns host and per-sandbox resource sampling on or off during the run.
- `monitoring.sample_interval_ms` controls the sampling cadence.
- `monitoring.include_host` and `monitoring.include_sandboxes` control host and per-sandbox monitoring coverage.
- `phase_workers` overrides concurrency per benchmark phase. Missing phase keys fall back to `max_workers`.
- `max_agent_timeout_scale` and `max_test_timeout_scale` scale per-task `task_config.options.max_agent_timeout_sec` and `task_config.options.max_test_timeout_sec` when those values are present in the dataset. Both default to `1.0`.
- `rootfs_reuse.enabled` defaults to `true`. Set it to `false` to restore the older per-sandbox rootfs materialization path.
- `sandbox_resource_limits` applies OCI `linux.resources` cgroup limits to every launched sandbox. `cpus` maps to a CPU quota (`cpu_period_us * cpus`), `memory_bytes` maps to `memory.limit`, and `pids_limit` maps to `pids.limit`. When `cpus` is set, the launcher also injects `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`, `LOKY_MAX_CPU_COUNT`, and `DJANGO_TEST_PROCESSES` into the container (and the SWE-bench verify step) so process/thread-pool sizing follows the cgroup quota rather than `os.cpu_count()`. Omitting the block preserves the pre-limit behavior.
- `phase_merging.setup_and_run` defaults to `false`. Set it to `true` to pipeline setup directly into run for eligible per-sandbox scenarios.
- `phase_merging.setup_and_run_executor_pool` defaults to `separate`. Set it to `shared` to use one executor pool for merged `setup+run` work.
- Phase telemetry now emits distinct phase-qualified records such as `benchmark.phase.setup.*`, `benchmark.phase.run.*`, `benchmark.phase.verification.*`, and `benchmark.phase.<phase>.item.*` so JSONL output shows phase timing and configured concurrency.
- The telemetry HTML report now includes a `Turn Analysis` section with stats, CDFs, and over-time charts for `llm_response_time`, `pure_llm_time`, `action_time`, and `turn_time`.
- `Turn Analysis` keeps the aggregate `all` view and now also breaks those same metrics out by `request_kind` when telemetry includes it, so Claude runs can separate `main_loop`, `helper`, and `count_tokens` behavior.
- For speculative-execution runs, the telemetry HTML report distinguishes visible `penalty` from `hidden reject cost`: `penalty` is the delay that still blocks the agent loop, while hidden cost is fork-side work that continued after the oracle path advanced.
- `Overhead Analysis` now includes a dedicated `llm.gate_wait` CDF figure in addition to the existing summary tables and latency charts.
- `llm_response_time` is the observed interceptor-side latency for a request, while `pure_llm_time` is the underlying `llm.service.request` duration from the LLM service itself.
- `zpool_size` controls the backing file size for ephemeral benchmark zpools.
- `reuse_zpool: true` keeps the zpool across runs instead of recreating it every time.
- When reusing a pool, set both `zpool_name` and `zpool_image` to stable values. Each run still destroys and recreates the `pool/agent-cr` dataset so the benchmark starts clean, but compose-backed shared rootfs cache datasets created by `rootfs_reuse.enabled: true` are reused until the pool itself is destroyed.

### LLM Service Options

`llm_service_options` is a top-level YAML block whose keys are merged into each task's `llm_service_config` (dataset-level values take precedence). This is useful for controlling replay behavior globally without editing the dataset JSONL.

For replay-backed services such as `iflow_trace_replay`, `mini_swe_trace_replay`, `mini_swe_spec_trace_replay`, and `claude_code_trace_replay`, the following options control how response delays are simulated:

- `response_delay_policy`: selects how the replay service simulates LLM response latency.
  - `fixed` (default): uses a constant delay of `response_delay_ms` milliseconds.
  - `trace_replay`: uses the actual request-to-response timestamps recorded in the trajectory, scaled by `response_delay_scaling_factor`.
- `response_delay_ms`: constant delay in milliseconds, used by the `fixed` policy and as a fallback when `trace_replay` timestamps are unavailable. The default is service-specific: `250` for `iflow_trace_replay`, `0` for `mini_swe_trace_replay`, and `0` for `claude_code_trace_replay`.
- `response_delay_scaling_factor`: multiplier applied to trace-derived delays when `response_delay_policy` is `trace_replay` (default 1.0). A value of 0.5 replays at 2× speed; 2.0 replays at half speed.
- `minimal_delay`: lower clamp in milliseconds applied after the policy-specific delay and `response_delay_scaling_factor` are resolved (default `0`).
- `maximal_delay`: upper clamp in milliseconds applied after the policy-specific delay and `response_delay_scaling_factor` are resolved (default `1e9`).

`mini_swe_spec_trace_replay` also supports speculative controls in its effective `llm_service_config`:

- `acceptance_rate` or legacy `accept_rate`: probability that the draft replay response keeps the oracle command
- `draft_response_delay_scaling_factor` or legacy `speculative_delay_scaling_factor`: extra multiplier applied only to draft replay latency after the base replay delay is computed
- `mismatch_policy`: currently `preserve_command_class`

Example YAML:

```yaml
max_agent_timeout_scale: 1.0
max_test_timeout_scale: 1.0

llm_service_options:
  response_delay_policy: trace_replay
  response_delay_ms: 250
  response_delay_scaling_factor: 1.0
  minimal_delay: 0
  maximal_delay: 1000000000
```

The per-scenario knobs live under `scenario_options`:

```bash
python3 -m benchmarks.run --config benchmarks/examples/fault.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/spot.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/tree.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/mini_swe/mini_swe.spec.auto.10tasks.debug.yaml
```

The real-host benchmarks allocate temporary runtime state, create a ZFS pool, build the simulated agent image, and launch `runc` sandboxes through the shared harness in [benchmarks/real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py).
