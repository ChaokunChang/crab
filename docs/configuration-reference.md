# Configuration Reference

This document catalogs all configuration parameters used by agent-cr, organized by subsystem. Each entry lists the parameter name, default value, how to specify it, and where it is consumed.

---

## 1. Scheduler Configuration (`SchedulerConfig`)

Defined in `agent_cr/config.py`. Passed to `CRScheduler` (and its policies) via `build_default_system()` or direct construction.

| Parameter | Type | Default | Description | Used by |
|---|---|---|---|---|
| `min_checkpoint_interval_seconds` | `float` | `30.0` | Minimum seconds between checkpoints. The scheduler skips checkpoints if less time has elapsed. Must be ≥ 0. | `CheckpointingPolicy.evaluate()` in `scheduler.py` |
| `force_checkpoint_after_seconds` | `float` | `600.0` | Force a checkpoint after this many seconds regardless of change signals. Must be ≥ 0. | `CheckpointingPolicy.evaluate()` in `scheduler.py` |
| `require_change_signal` | `bool` | `True` | When `True`, skip checkpoints unless the inspector reports a process or filesystem change. | `CheckpointingPolicy.evaluate()` in `scheduler.py` |
| `prefer_checkpoint_during_llm_request` | `bool` | `True` | Prefer checkpointing when an LLM request is in flight (to capture the request window). | `CheckpointingPolicy.evaluate()` in `scheduler.py` |
| `require_llm_request_for_checkpoint` | `bool` | `False` | When `True`, only checkpoint while an LLM request is in flight. | `CheckpointingPolicy.evaluate()` in `scheduler.py` |
| `inspect_without_pause` | `bool` | `False` | When `True`, `CRScheduler.query_checkpoint()` first inspects the live sandbox and only pauses later if it decides a non-`leave_running` checkpoint needs quiescing. Default `False` keeps the safer pause-before-inspect path. | `CRScheduler.query_checkpoint()` in `scheduler.py` |

**How to specify:** Pass a `SchedulerConfig` dataclass to `build_default_system(scheduler_config=...)` or to `CRScheduler(config=...)`.

---

## 2. Executor Configuration (`ExecutorConfig`)

Defined in `agent_cr/config.py`. Controls the checkpoint/restore job executor thread pool.

| Parameter | Type | Default | Description | Used by |
|---|---|---|---|---|
| `max_workers` | `int` | `4` | Legacy fallback worker count used when `checkpoint_workers` and/or `restore_workers` are omitted. Must be ≥ 1. | `ExecutorConfig.resolved_checkpoint_workers`, `ExecutorConfig.resolved_restore_workers` in `config.py` |
| `checkpoint_workers` | `int \| None` | `None` | Dedicated checkpoint worker count. When `None`, falls back to `max_workers`. Must be ≥ 1 when provided. | `CRExecutor.__init__()` in `executor.py` |
| `restore_workers` | `int \| None` | `None` | Dedicated restore worker count. When `None`, falls back to `max_workers`. Must be ≥ 1 when provided. | `CRExecutor.__init__()` in `executor.py` |
| `coordination_workers` | `int \| None` | `None` | Worker count for `AgentCRSystem` live-request coordination. When omitted, resolves to `min(8, resolved_checkpoint_workers)`. | `AgentCRSystem.start()` in `system.py` |
| `composite_step_workers` | `int \| None` | `None` | Shared worker count for parallel process/filesystem checkpoint sub-steps. When omitted, resolves to `max(2, min(2 * resolved_checkpoint_workers, 16))`. | `DefaultCWorker.__init__()` in `workers/composite.py` |
| `max_checkpoint_queue_size` | `int` | `10000` | Maximum number of pending checkpoint jobs admitted before new submissions are rejected. Must be ≥ 1. | `CRExecutor.__init__()` in `executor.py` |
| `checkpoint_scheduling_policy` | `"fifo" \| "reactive"` | `"fifo"` | Queue discipline for checkpoint jobs. `fifo` preserves the historical submission order. `reactive` uses normal/urgent queues and promotes live-request checkpoints once their response has already returned. | `CRExecutor.submit_checkpoint()`, `CRExecutor.notify_live_response_ready()` in `executor.py` |
| `reactive_checkpoint_urgent_quota` | `int` | `4` | When `checkpoint_scheduling_policy="reactive"` and both queues are non-empty, the executor serves at most this many urgent jobs before forcing one normal job to avoid starvation. Must be ≥ 1. | `CRExecutor._take_next_checkpoint_item_locked()` in `executor.py` |
| `max_retries` | `int` | `0` | Number of retry attempts for failed checkpoint/restore jobs. Must be ≥ 0. | `CRExecutor._execute_checkpoint()`, `_execute_restore()` in `executor.py` |
| `retry_backoff_seconds` | `float` | `0.05` | Linear backoff base between retries: sleeps `retry_backoff_seconds * (attempt + 1)`. Must be ≥ 0. | `CRExecutor._execute_checkpoint()`, `_execute_restore()` in `executor.py` |

**How to specify:** Pass an `ExecutorConfig` dataclass to `build_default_system(executor_config=...)` or to `CRExecutor(config=...)`.

---

## 3. Storage Configuration (`StorageConfig`)

Defined in `agent_cr/config.py`. Configures the local checkpoint storage layout.

| Parameter | Type | Default | Description | Used by |
|---|---|---|---|---|
| `root_dir` | `Path` | *(required)* | Root directory for checkpoint data (manifests + artifacts). | `LocalCheckpointManager.__init__()` in `storage/local.py` |
| `manifests_dirname` | `str` | `"manifests"` | Subdirectory name under `root_dir` for manifest JSON files. Must be non-empty. | `LocalCheckpointManager.__init__()` |
| `artifacts_dirname` | `str` | `"artifacts"` | Subdirectory name under `root_dir` for artifact blobs. Must be non-empty. | `LocalCheckpointManager.__init__()` |

**How to specify:** Pass a `StorageConfig` dataclass to `build_default_system(storage_config=...)` or to `LocalCheckpointManager(config=...)`.

---

## 4. Telemetry Configuration (`TelemetryConfig`)

Defined in `agent_cr/config.py`. Controls the telemetry subsystem.

| Parameter | Type | Default | Description | Used by |
|---|---|---|---|---|
| `enabled` | `bool` | `True` | Whether telemetry is active. When `False`, a `NoopTelemetrySink` is used. | `build_default_system()` in `system.py` |
| `jsonl_path` | `Path \| None` | `None` | If set, telemetry events/metrics are appended to this JSONL file. JSONL writes are batched and asynchronous by default, and file locks still keep multi-process appends safe. | `build_default_system()` in `system.py`; `JsonlTelemetrySink` / `AsyncJsonlTelemetrySink` in `telemetry.py` |
| `keep_in_memory_copy` | `bool \| None` | `None` | Keep a copy of all telemetry in memory (`InMemoryTelemetrySink`). When `None`, in-memory capture stays enabled only when JSONL output is not configured; with `jsonl_path` set, it defaults to `False`. | `build_default_system()` in `system.py` |
| `detail_level` | `str` | `"basic"` | Telemetry detail level. `"basic"` or `"detailed"`. When `"detailed"`, command stdout/stderr is captured in telemetry. | `ConfiguredTelemetrySink` in `telemetry.py`; `RuncRuntime._command_finish_attributes()` |
| `capture_command_output` | `bool` | `False` | Capture stdout/stderr of runtime commands in telemetry events regardless of `detail_level`. | `ConfiguredTelemetrySink` in `telemetry.py`; `RuncRuntime._command_finish_attributes()` |
| `max_text_attribute_bytes` | `int` | `2048` | Maximum byte length for text attributes in telemetry. Longer values are truncated. Minimum enforced: 32. | `ConfiguredTelemetrySink`, `telemetry_truncate_value()` in `telemetry.py` |
| `writer_mode` | `str` | `"async"` | Telemetry JSONL writer mode. `async` uses a bounded queue and background writer; `sync` writes on the caller thread. | `build_configured_telemetry_sink()` in `telemetry.py` |
| `queue_capacity` | `int` | `16384` | Maximum queued telemetry records in async mode. Must be ≥ 1. | `AsyncJsonlTelemetrySink` in `telemetry.py` |
| `batch_max_records` | `int` | `256` | Maximum JSONL records written per async batch. Must be ≥ 1. | `AsyncJsonlTelemetrySink` in `telemetry.py` |
| `flush_interval_ms` | `int` | `50` | Async telemetry flush interval in milliseconds. Must be ≥ 1. | `AsyncJsonlTelemetrySink` in `telemetry.py` |
| `overflow_policy` | `str` | `"drop_new"` | Async overflow behavior. `drop_new` keeps the hot path non-blocking and emits a drop summary later; `block` applies backpressure. | `AsyncJsonlTelemetrySink` in `telemetry.py` |
| `serializer` | `str` | `"auto"` | JSON serializer choice. `auto` prefers `orjson` when installed and falls back to stdlib JSON. | `JsonlTelemetrySink` in `telemetry.py` |

**How to specify:** Pass a `TelemetryConfig` dataclass to `build_default_system(telemetry_config=...)`. For benchmarks, these are also configurable via the YAML `telemetry:` block.

---

## 5. Runc Runtime Configuration

### 5a. Runtime Paths (`RuncRuntimePaths`)

Defined in `agent_cr/runtime/runc.py`. Filesystem paths for the runc runtime.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state_root` | `Path` | `/run/agent-cr/runc` | Root for runc state directory (`--root` flag). |
| `bundle_root` | `Path` | `/var/lib/agent-cr/bundles` | Root directory for sandbox OCI bundles. |
| `checkpoint_root` | `Path` | `/var/lib/agent-cr/checkpoints` | Root directory for CRIU checkpoint images. |
| `metadata_root` | `Path` | `/var/lib/agent-cr/sandbox-metadata` | Root directory for persisted sandbox description JSON files. |
| `zfs_dataset_prefix` | `str` | `"agentcr/sandboxes"` | ZFS dataset prefix for sandbox filesystems. |

**How to specify:** Pass a `RuncRuntimePaths` to `RuncRuntime(paths=...)`.

### 5b. Checkpoint Options (`RuncCheckpointOptions`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tcp_established` | `bool` | `True` | Pass `--tcp-established` to CRIU during checkpoint. |
| `shell_job` | `bool` | `True` | Pass `--shell-job` to CRIU during checkpoint. |
| `tcp_skip_in_flight` | `bool` | `True` | Pass `--tcp-skip-in-flight` to CRIU during checkpoint. |
| `extra_args` | `tuple[str, ...]` | `()` | Additional CLI args passed to `runc checkpoint`. |

### 5c. Restore Options (`RuncRestoreOptions`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detach` | `bool` | `True` | Pass `-d` (detach) to `runc restore`. |
| `tcp_established` | `bool` | `True` | Pass `--tcp-established` to CRIU during restore. |
| `shell_job` | `bool` | `True` | Pass `--shell-job` to CRIU during restore. |
| `extra_args` | `tuple[str, ...]` | `()` | Additional CLI args passed to `runc restore`. |

### 5d. Runtime Binaries

| Parameter | Type | Default | Description | Where |
|---|---|---|---|---|
| `runtime_bin` | `str` | `"runc"` | Path or name of the runc binary. | `RuncRuntime.__init__()` |
| `zfs_bin` | `str` | `"zfs"` | Path or name of the zfs binary. | `RuncRuntime.__init__()`, `LocalCheckpointManager.__init__()` |

**How to specify:** Pass `RuncRuntimeOptions` (which bundles `RuncCheckpointOptions` and `RuncRestoreOptions`) to `build_default_system(runc_runtime_options=...)` or to `RuncRuntime(options=...)`.

---

## 6. Request Interceptor Server Configuration

Defined in `agent_cr/interceptor.py` via `AgentCRRequestInterceptorServer.__init__()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `upstream_url` | `str` | *(required)* | Base URL of the upstream LLM service. |
| `host` | `str` | `"127.0.0.1"` | Host to bind the interceptor HTTP server. |
| `port` | `int` | `0` | Port to bind. `0` selects an ephemeral port. |
| `upstream_timeout_seconds` | `float` | `3600.0` | Timeout (seconds) for forwarding requests to the upstream LLM service. |
| `max_workers` | `int \| None` | `None` | Maximum worker threads for the interceptor HTTP server. The server now uses a bounded pooled HTTP worker model instead of unbounded request threads. |

**How to specify:** Constructor arguments to `AgentCRRequestInterceptorServer(...)`.

---

## 7. Host Inspector Configuration

### 7a. Host Inspector Daemon (`HostInspectorDaemon`)

Defined in `agent_cr/host_inspector/server.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `process_poll_interval_s` | `float` | `1.0` | Accepted for compatibility; process change detection is on-demand. |

### 7b. Host Inspector Server (`HostInspectorServer`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"127.0.0.1"` | Host to bind. |
| `port` | `int` | `0` | Port to bind. `0` selects an ephemeral port. |
| `max_workers` | `int \| None` | `None` | Maximum worker threads for the host-inspector HTTP server. |

### 7c. Host Inspector CLI (`agent_cr/host_inspector/__main__.py` and `server.py:main()`)

| CLI Flag | Default | Description |
|---|---|---|
| `--host` | `"127.0.0.1"` | Bind host. |
| `--port` | `9782` | Bind port. |
| `--process-poll-interval` | `1.0` | Process poll interval (compatibility). |
| `--helper-path` | `None` (auto-detected) | Path to the eBPF filesystem monitor helper binary. |
| `--runc-state-root` | `None` | Override runc state root for the runtime resolver. |
| `--log-level` | `"INFO"` | Python logging level. |
| `--max-workers` | `32` | Maximum worker threads for the host-inspector HTTP server. |

### 7d. Host Inspector Service Client (`HostInspectorServiceClient`)

Defined in `agent_cr/remote_inspector.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | *(required)* | Base URL of the host inspector HTTP service. |
| `timeout_s` | `float` | `5.0` | HTTP request timeout for inspector API calls. |

### 7e. Host Inspector Watch CLI (`agent_cr/host_inspector/watch.py`)

| CLI Flag | Default | Description |
|---|---|---|
| `--base-url` | *(required)* | Host inspector base URL. |
| `--interval` | `1.0` | Polling interval in seconds. |
| `--iterations` | `0` | Number of iterations (`0` = forever). |

---

## 8. AgentCRSystem Configuration

Defined in `agent_cr/system.py`. Constructor parameters on the `AgentCRSystem` dataclass.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recovery_delay_seconds` | `float` | `0.0` | Delay (seconds) before restoring a sandbox during recovery (used to simulate transfer delay). |
| `enforce_restore_checkpoint_validation` | `bool` | `False` | When `True`, validates that a restore checkpoint exists and is restorable before attempting restore. |

### Internal Constants (not user-configurable, but notable)

| Constant | Value | Description | Location |
|---|---|---|---|
| `_RESTORE_RUNTIME_READY_ATTEMPTS` | `10` | Number of retries to check if sandbox is running after restore. | `system.py:58` |
| `_RESTORE_RUNTIME_READY_DELAY_S` | `0.1` | Delay between runtime-ready checks after restore. | `system.py:59` |
| `_HOST_INSPECTOR_REGISTER_ATTEMPTS` | `3` | Number of retries for registering a sandbox with the host inspector. | `runtime/runc.py:21` |
| `_HOST_INSPECTOR_REGISTER_RETRY_DELAY_S` | `0.2` | Delay between host inspector registration retries. | `runtime/runc.py:22` |

---

## 9. Checkpoint Retention Policies

Defined in `agent_cr/storage/policies.py`. These wrap a `CheckpointManager` delegate.

| Policy Class | Behavior |
|---|---|
| `KeepAllCheckpointManager` | Retains all checkpoints. No automatic deletion. |
| `LatestOnlyCheckpointManager` | After each checkpoint, prunes all unprotected checkpoints (keeps latest, latest-with-process, latest-with-filesystem, and pinned checkpoints). |
| `DeleteAfterRestoreCheckpointManager` | After a restore completes, deletes all checkpoints for that sandbox. |

**How to specify:** Wrap a `LocalCheckpointManager` instance, e.g. `LatestOnlyCheckpointManager(base_storage)`. In benchmarks, configured via `checkpoint_manager_factory` in `HarnessSettings`.

---

## 10. Simulated LLM Service Configuration

Defined in `integrations/llm_services/simulated/service.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `"127.0.0.1"` | Bind host (CLI mode). |
| `--port` | `int` | *(required)* | Bind port (CLI mode). |
| `--response-delay-ms` | `int` | `500` | Artificial delay (ms) added to each simulated LLM response. |
| `response_delay_ms` | `int` | `0` | When used via `handle_request()` or `serve()` programmatically. |

---

## 11. Benchmark Configuration (`BenchmarkConfig`)

Defined in `benchmarks/config.py`. Loaded from YAML via `load_config()`.

### Top-level YAML Fields

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `scenario` | `str` | *(required)* | Scenario type: `e2e`, `fault`, `spot`, `tree`, `spec`. |
| `mode` | `str` | *(required)* | Execution mode: `manual` or `auto`. |
| `provider` | `str` | `"openai"` | LLM provider: `openai` or `anthropic`. |
| `agent` | `str` | `"simulated"` | Agent type: `simulated`, `iflow`, `mini_swe`, or `claude_code`. |
| `llm_service` | `str \| null` | `null` | LLM service type override: `simulated`, `manual`, `simulated_for_iflow`, `iflow_trace_replay`, `mini_swe_trace_replay`, `mini_swe_spec_trace_replay`, or `claude_code_trace_replay`. |
| `task_dataset` | `path \| null` | `null` | Path to a task dataset file (relative to config file). |
| `sandboxes` | `int` | `1` | Number of concurrent sandboxes. Must be > 0. |
| `max_workers` | `int \| null` | `null` | Max worker threads. Defaults to `sandboxes` count. |
| `phase_workers` | `mapping \| null` | `null` | Per-phase worker overrides for `setup`, `run`, and `verification`. Missing keys fall back to `max_workers`. |
| `rootfs_reuse` | `mapping` | `{ enabled: true }` | Controls shared-rootfs reuse for benchmark sandbox provisioning. |
| `phase_merging` | `mapping` | `{ setup_and_run: false, setup_and_run_executor_pool: separate }` | Controls whether eligible scenarios can pipeline setup directly into run and how merged setup/run work is scheduled. |
| `iterations` | `int` | scenario-dependent | Number of iterations. Defaults: e2e=5, fault=3, spot=3, tree=1, spec=0. `spec` is full-trace replay only and therefore requires `iterations: 0`. |
| `output` | `path \| null` | `null` | Path for CSV output file. |
| `log_file` | `path \| null` | `null` | Path for log file. |
| `log_file_mode` | `str` | `"append"` | Log file open mode: `append` or `write`. |
| `log_level` | `str` | `"info"` | Python log level: `debug`, `info`, `warning`, `error`, `critical`. |
| `benchmark_root_home` | `path \| null` | `null` | Parent directory for benchmark run data. Falls back to `AGENTCR_BENCH_DIR` env var, then a temp directory. |
| `benchmark_run_name` | `str \| null` | `null` | Directory name for this run under `benchmark_root_home`. Defaults to a timestamp such as `20260416_010203` when a benchmark root home is used. Requires `benchmark_root_home`, `benchmark_root`, or `AGENTCR_BENCH_DIR`. |
| `benchmark_root` | `path \| null` | `null` | Deprecated alias for `benchmark_root_home`, kept for older configs. |
| `clear_benchmark_root_after_run` | `bool` | `false` | Delete the resolved per-run benchmark directory after postprocessing completes. Tempdir-backed benchmark roots keep their existing automatic cleanup behavior. |
| `storage_planes` | `mapping` | `{}` | Optional benchmark storage-plane overrides for runtime state, checkpoint storage, and agent host directories. |
| `llm_service_options` | `mapping` | `{}` | Mapping merged into each task record's `llm_service_config`. Dataset-level values still take precedence. |
| `max_agent_timeout_scale` | `float` | `1.0` | Multiplies dataset-provided `task_config.options.max_agent_timeout_sec` values before task launch. Missing dataset values stay unset. |
| `max_test_timeout_scale` | `float` | `1.0` | Multiplies dataset-provided `task_config.options.max_test_timeout_sec` values before task launch. Missing dataset values stay unset. |

When `output`, `log_file`, `telemetry.output`, or `telemetry.report.output_dir` are configured outside the resolved benchmark run root, the runner keeps writing those configured paths and also copies each existing artifact by basename into `<benchmark_root_home>/<benchmark_run_name>/` after the run.

---

## 12. Telemetry Report CLI (`benchmarks.telemetry_analysis.report`)

Standalone telemetry reports are generated by the report CLI in `benchmarks/telemetry_analysis/report.py`.

| CLI Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Telemetry JSONL input path. |
| `--output-dir` | *(required)* | Directory where the HTML, JSON, CSV, and SVG report artifacts are written. |
| `--run-id` | dominant run in file | Optional `run_id` filter. |
| `--top-k` | `25` | Top-K operations/outliers to retain in the summaries. |
| `--exclude-failed-tasks` | `false` | Exclude sandboxes whose `benchmark.task.success_ratio` is `0`. |
| `--log-scale-charts` | `false` | Use log scaling for bar-chart widths. |
| `--figure-window-seconds` | `0` | Average time-series charts within fixed-size time windows. Applies to load, latency, and Turn Analysis figures. `0` disables aggregation. |
| `--no-export-svg` | `false` | Skip standalone SVG export. |
| `zpool_size` | `str` | `"10G"` | Size of the ZFS pool image file (passed to `truncate -s`). |
| `zpool_name` | `str \| null` | `null` | Explicit ZFS pool name. Auto-generated if `null`. |
| `zpool_image` | `path \| null` | `null` | Path to the ZFS pool image file. |
| `reuse_zpool` | `bool` | `false` | If `true`, keep the ZFS pool across benchmark runs. |
| `image_cache_root` | `path \| null` | `null` | Cache directory for Docker images. Default: `.cache/agent-cr/images`. |
| `transfer_delay_ms` | `float` | `0.0` | Simulated transfer delay (ms) applied as `recovery_delay_seconds` during auto-recovery. |
| `work_dir_host_root` | `path \| null` | `null` | Host-side working directory root for sandboxes. |
| `executor` | `mapping` | `{}` | Benchmark-only executor overrides. Resolves to `ExecutorConfig` using the benchmark's effective worker count as the fallback. |
| `scheduler` | `mapping` | `{}` | Benchmark-only `SchedulerConfig` field overrides. Merged onto the scenario's scheduler defaults; does not choose the scheduler policy class. |
| `llm_server` | `mapping` | `{ launch_mode: process }` | Benchmark LLM router launch settings. Controls whether the router runs in a subprocess or a thread. |
| `host_inspector` | `mapping` | `{ launch_mode: process }` | Host-inspector launch settings. Controls whether the host inspector runs in a subprocess or in the benchmark process. |

### Phase Worker YAML Block (`phase_workers:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `phase_workers.setup` | `int \| null` | `max_workers` | Worker cap for the setup phase. Must be > 0 when provided. |
| `phase_workers.run` | `int \| null` | `max_workers` | Worker cap for the run phase. Must be > 0 when provided. |
| `phase_workers.verification` | `int \| null` | `max_workers` | Worker cap for the verification phase. Must be > 0 when provided. |

By default, the benchmark harness executes all runs in three phases with hard barriers between them:

- all sandboxes finish `setup` before `run`
- all sandboxes finish `run` before `verification`

When `phase_merging.setup_and_run: true` is enabled, only the setup-to-run barrier is relaxed, and only for eligible per-sandbox scenario flows:

- `fault`, `spot`, `tree`, and replay-backed `e2e` can start each sandbox's `run` work immediately after that sandbox's `setup` finishes
- cohort-style non-replay `e2e` still keeps the full setup barrier
- `verification` still waits for all run work to finish in every mode

`phase_merging.setup_and_run_executor_pool` controls the executor topology for merged setup/run flows:

- `separate` keeps independent setup and run executor pools
- `shared` uses one executor pool and submits a combined `setup+run` task per sandbox
- in `shared` mode, the pool size is `min(phase_workers.setup, phase_workers.run)` so the single pool stays within both configured phase limits

### Rootfs Reuse YAML Block (`rootfs_reuse:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `rootfs_reuse.enabled` | `bool` | `true` | Enables shared ZFS rootfs reuse during benchmark setup. The harness materializes one shared base rootfs per normalized recipe, snapshots it once, and clones that snapshot into each sandbox dataset instead of copying the full rootfs for every sandbox. |

Notes:

- For compose-backed tasks, the reuse key is recipe-based and anchored by resolved `docker_compose_file`, `service_name`, `agent_type`, and normalized `rootfs_init_dirs` / `rootfs_copy_paths`.
- When `reuse_zpool: true`, compose-backed shared rootfs bases persist in the reused pool across benchmark runs.
- For non-compose or no-dataset runs, shared rootfs reuse is scoped to the current benchmark invocation.
- Set `rootfs_reuse.enabled: false` to restore the older per-sandbox rootfs materialization path.

### Sandbox Resource Limits YAML Block (`sandbox_resource_limits:`)

Optional OCI `linux.resources` cgroup limits applied to every sandbox created by the harness. All fields are optional; omitting a field leaves that resource uncapped.

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `sandbox_resource_limits.cpus` | `int \| null` | `null` | CPU quota cap expressed as whole vCPUs. When set, the OCI spec receives `resources.cpu.quota = cpu_period_us * cpus` and `resources.cpu.period = cpu_period_us`. The launcher also injects `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`, `LOKY_MAX_CPU_COUNT`, and `DJANGO_TEST_PROCESSES` into the sandbox environment (and the SWE-bench verify step) so pool sizing honors the cgroup quota rather than `os.cpu_count()`. |
| `sandbox_resource_limits.memory_bytes` | `int \| null` | `null` | Maps to `resources.memory.limit` in bytes. Must be positive when set. |
| `sandbox_resource_limits.pids_limit` | `int \| null` | `null` | Maps to `resources.pids.limit`. Must be positive when set. |
| `sandbox_resource_limits.cpu_period_us` | `int` | `100000` | CFS period in microseconds used to translate `cpus` into a quota. Change only if you need a non-default CFS period. |

Notes:

- These fields are enforced through cgroups. The env-var injection is independent of the cgroup cap because the sandbox launcher drops the cgroup namespace; joblib/loky's `_cpu_count_cgroup` then sees the host root `cpu.max` and falls through to `os.cpu_count()` unless `LOKY_MAX_CPU_COUNT` is set.
- The block is defined by `SandboxResourceLimitsConfig` in `benchmarks/config.py` and applied in `integrations/sandboxes/runtime/bundle.py` (`_apply_resource_limits`, `concurrency_env_for_cpu_limit`).

### Phase Merging YAML Block (`phase_merging:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `phase_merging.setup_and_run` | `bool` | `false` | Lets eligible per-sandbox scenarios start a sandbox's run work immediately after its own setup finishes, instead of waiting for all setups to complete. Verification still waits for all run work to finish. |
| `phase_merging.setup_and_run_executor_pool` | `str` | `"separate"` | Executor topology for merged setup/run flows. `separate` keeps the current independent setup and run pools. `shared` uses one pool and submits combined `setup+run` tasks per sandbox. |

### Storage Plane YAML Block (`storage_planes:`)

These paths control the hottest host-side write locations used by the real-host harness. They are especially useful when `zpool_image` is a file-backed vdev and you want checkpoint/process/storage writes to land on a different filesystem or device.

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `storage_planes.runtime_root` | `path \| null` | `null` | Root for runtime bundles, CRIU checkpoint images, sandbox metadata, and exported image rootfs. When omitted, the harness preserves the legacy layout and uses the benchmark run root. |
| `storage_planes.storage_root` | `path \| null` | `null` | Root for checkpoint manifests and artifact storage. When omitted, the harness preserves the legacy layout and uses `<benchmark_run_root>/storage`. |
| `storage_planes.agent_host_root` | `path \| null` | `null` | Root for agent host-side state and per-sandbox host directories. When omitted, the harness preserves the legacy layout and uses the benchmark run root. |

Notes:

- The resolved benchmark run root remains the benchmark artifact root for CSV output, logs, and default telemetry/report artifacts.
- `storage_planes` is opt-in. Omitting it preserves the pre-separation directory layout.
- To reduce interference with file-backed ZFS pools, point `storage_planes.*` at a different filesystem or block device than `zpool_image`.
- If you also use `work_dir_host_root`, that path is independent and may need the same treatment for very write-heavy tasks.

### Executor YAML Block (`executor:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `executor.checkpoint_workers` | `int \| null` | `null` | Checkpoint worker count. Falls back to the benchmark's effective `max_workers` when omitted. |
| `executor.restore_workers` | `int \| null` | `null` | Restore worker count. Falls back to the benchmark's effective `max_workers` when omitted. |
| `executor.coordination_workers` | `int \| null` | `null` | Live-request coordination worker count. When omitted, resolves to `min(8, resolved_checkpoint_workers)`. |
| `executor.composite_step_workers` | `int \| null` | `null` | Shared worker count for parallel process/filesystem checkpoint sub-steps. When omitted, resolves to `max(2, min(2 * resolved_checkpoint_workers, 16))`. |
| `executor.checkpoint_queue_size` | `int` | `10000` | Max pending checkpoint jobs before new submissions are rejected. Must be > 0. |
| `executor.checkpoint_scheduling_policy` | `"fifo" \| "reactive"` | `"fifo"` | Checkpoint queue discipline. `fifo` preserves the historical behavior; `reactive` promotes checkpoints whose overlapping LLM response has already returned. |
| `executor.reactive_checkpoint_urgent_quota` | `int` | `4` | When `executor.checkpoint_scheduling_policy: reactive`, serve at most this many urgent jobs in a row before forcing one normal job if both queues are non-empty. |
| `executor.max_retries` | `int` | `0` | Retry attempts for checkpoint/restore jobs. Must be ≥ 0. |
| `executor.retry_backoff_seconds` | `float` | `0.05` | Linear retry backoff base in seconds. Must be ≥ 0. |

### Scheduler YAML Block (`scheduler:`)

The scenario still owns the scheduler policy class. For example, `fault` auto mode still uses the fault-tolerance policy and `tree` still uses the tree-search policy. The YAML block below only overrides `SchedulerConfig` fields used by that scenario-selected policy.

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `scheduler.min_checkpoint_interval_seconds` | `float \| null` | `null` | Override for `SchedulerConfig.min_checkpoint_interval_seconds`. |
| `scheduler.force_checkpoint_after_seconds` | `float \| null` | `null` | Override for `SchedulerConfig.force_checkpoint_after_seconds`. |
| `scheduler.require_change_signal` | `bool \| null` | `null` | Override for `SchedulerConfig.require_change_signal`. |
| `scheduler.prefer_checkpoint_during_llm_request` | `bool \| null` | `null` | Override for `SchedulerConfig.prefer_checkpoint_during_llm_request`. |
| `scheduler.require_llm_request_for_checkpoint` | `bool \| null` | `null` | Override for `SchedulerConfig.require_llm_request_for_checkpoint`. |
| `scheduler.inspect_without_pause` | `bool \| null` | `null` | Override for `SchedulerConfig.inspect_without_pause`. The safe default remains `false`. |

### Turn Analysis Metrics

The telemetry report's `Turn Analysis` section derives four timing views from request telemetry:

- `llm_response_time`: interceptor-side request latency from `llm.interceptor_total_ms`, with `llm.request_total_ms` as a fallback alias.
- `pure_llm_time`: raw service-side request duration from `llm.service.request.duration_ms`.
- `action_time`: elapsed time from one request finishing to the next request starting within the same sandbox.
- `turn_time`: `llm_response_time + action_time`.

Each metric is reported with summary statistics, a CDF, and a time-series chart that honors `--figure-window-seconds`.

When request telemetry includes `request_kind`, the report keeps the aggregate `all` view and also emits per-kind rows and figures. This is particularly useful for Claude Code replay, where `main_loop`, `helper`, and `count_tokens` requests have different checkpoint and gating behavior.

The `Overhead Analysis` section also includes a dedicated `llm.gate_wait` CDF so long-tail checkpoint delay is easier to separate from the median case.

### LLM Server YAML Block (`llm_server:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `llm_server.launch_mode` | `str` | `"process"` | Launch the benchmark LLM router in a subprocess (`process`) or a thread in the benchmark process (`thread`). The request path still uses HTTP over `localhost` in both modes. |

### Host Inspector YAML Block (`host_inspector:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `host_inspector.launch_mode` | `str` | `"process"` | Launch the host inspector in a subprocess (`process`) or in a thread inside the benchmark process (`thread`). `process` is the default to reduce main-process thread pressure. |

### Telemetry YAML Block (`telemetry:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `telemetry.output` (or top-level `telemetry_output`) | `path \| null` | `null` | Path for telemetry JSONL output. Defaults to `<benchmark_run_root>/telemetry.jsonl`. |
| `telemetry.detail_level` | `str` | `"basic"` | `"basic"` or `"detailed"`. |
| `telemetry.capture_command_output` | `bool` | `false` | Capture runtime command stdout/stderr in telemetry. |
| `telemetry.max_text_attribute_bytes` | `int` | `2048` | Maximum byte length per text attribute in telemetry. Must be > 0. |
| `telemetry.keep_in_memory_copy` | `bool \| null` | `null` | Keep a copy of benchmark telemetry in memory. When omitted and JSONL output is configured, the benchmark defaults this to `false`. |
| `telemetry.writer_mode` | `str` | `"async"` | Telemetry writer mode: `async` or `sync`. |
| `telemetry.queue_capacity` | `int` | `16384` | Async telemetry queue capacity. |
| `telemetry.batch_max_records` | `int` | `256` | Maximum JSONL records per async write batch. |
| `telemetry.flush_interval_ms` | `int` | `50` | Async telemetry flush interval in milliseconds. |
| `telemetry.overflow_policy` | `str` | `"drop_new"` | Async telemetry overflow behavior. |
| `telemetry.serializer` | `str` | `"auto"` | JSON serializer for telemetry output. `auto` prefers `orjson` when available. |

### LLM Service Options (`llm_service_options:`)

These keys are merged into each task record's `llm_service_config` before the benchmark launches. Dataset-provided `llm_service_config` values still win when both sides define the same key.

For replay-backed LLM services such as `iflow_trace_replay`, `mini_swe_trace_replay`, `mini_swe_spec_trace_replay`, and `claude_code_trace_replay`:

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `llm_service_options.response_delay_policy` | `str` | `"fixed"` | Replay delay policy. `fixed` uses `response_delay_ms`; `trace_replay` uses trace-derived request-to-response delays. |
| `llm_service_options.response_delay_ms` | `float` | `250.0` | Fixed replay delay in milliseconds. Also used as a fallback when `trace_replay` cannot recover a trace delay. |
| `llm_service_options.response_delay_scaling_factor` | `float` | `1.0` | Multiplier applied to trace-derived delays when `response_delay_policy` is `trace_replay`. |
| `llm_service_options.minimal_delay` | `float` | `0.0` | Lower clamp in milliseconds applied after the replay policy chooses a delay and after any scaling factor is applied. |
| `llm_service_options.maximal_delay` | `float` | `1e9` | Upper clamp in milliseconds applied after the replay policy chooses a delay and after any scaling factor is applied. Must be greater than or equal to `minimal_delay`. |

`mini_swe_spec_trace_replay` also understands additional speculative controls when they appear in the effective `llm_service_config`:

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `acceptance_rate` | `float` | `0.5` | Probability that the draft response matches the oracle command. Legacy alias: `accept_rate`. |
| `draft_response_delay_scaling_factor` | `float` | `0.5` | Extra scaling factor applied only to draft replay latency after `response_delay_scaling_factor`. Legacy alias: `speculative_delay_scaling_factor`. |
| `mismatch_policy` | `str` | `"preserve_command_class"` | Policy used when mutating rejected draft responses. Current supported value: `preserve_command_class`. |

### Scenario Options (`scenario_options:`)

#### Fault Scenario (`benchmarks/scenarios/fault.py`)

| Key | Type | Default | Description |
|---|---|---|---|
| `injection_rate` | `float` | `0.5` | Probability of injecting a fault per iteration. Range: [0.0, 1.0]. |
| `first_forced_event_chunk` | `int` | `0` | First chunk/iteration at which fault injection is allowed. Must be ≤ `iterations`. |
| `delete_filesystem_checkpoints` | `bool` | `false` | Whether the fault-scenario retention policy is allowed to delete older filesystem checkpoints. The default keeps filesystem checkpoints so ZFS snapshot destruction does not run inline on the run-phase hot path. |

#### Spot Preemption Scenario (`benchmarks/scenarios/spot.py`)

| Key | Type | Default | Description |
|---|---|---|---|
| `injection_rate` | `float` | `0.5` | Probability of injecting a preemption per iteration. Range: [0.0, 1.0]. |
| `first_forced_event_chunk` | `int` | `0` | First chunk/iteration at which preemption injection is allowed. |
| `grace_period_seconds` | `float` | `60.0` | Grace period (seconds) before preemption takes effect. Must be > 0. |

#### Tree Search Scenario (`benchmarks/scenarios/tree.py`)

| Key | Type | Default | Description |
|---|---|---|---|
| `source_steps` | `int` | `6` | Number of steps in the source (trunk) run before branching. Must be > 0. |
| `branch_points` | `int` | `2` | Number of branch points to create. Must be ≥ 0. |
| `fork_steps` | `int` | `3` | Number of steps to run on each forked branch. Must be ≥ 0. |
| `replay_mode` | `str` | `"sequential"` | How to replay branches: `"sequential"` or `"concurrent"`. |

#### Speculative Execution Scenario (`benchmarks/scenarios/spec.py`)

These keys are merged into each task row's `llm_service_config` for `scenario: spec`.

| Key | Type | Default | Description |
|---|---|---|---|
| `acceptance_rate` | `float` | `0.5` | Probability that the draft replay response matches the oracle command. Legacy alias: `accept_rate`. |
| `draft_response_delay_scaling_factor` | `float` | `0.5` | Extra multiplier applied only to draft replay latency after the normal replay delay policy is resolved. Legacy alias: `speculative_delay_scaling_factor`. |
| `mismatch_policy` | `str` | `"preserve_command_class"` | Draft-mismatch mutation policy. Current supported value: `preserve_command_class`. |
| `enable_fork_reuse` | `bool` | `false` | When `true`, the speculative controller caches a finalized fork whose active and fork sandboxes both report `state_unchanged` and whose draft exec completed by oracle-finish time. The next turn consumes that cached fork instead of a fresh ZFS clone + CRIU restore. The cache is invalidated on any sandbox restore/recovery and is per-sandbox, so task boundaries always miss. Wired through `BenchmarkConfig.scenario_options["enable_fork_reuse"]` → `RealHostScenarioHarness(fork_reuse_enabled=...)` → `_SpeculativeSandboxController`. |
| `eager_fork_cleanup_on_reject` | `bool` | `false` | When `true`, rejected speculative turns destroy the fork immediately (instead of letting the draft exec keep running on the fork until it finishes on its own). Destroying the fork terminates the `runc exec` subprocess, which bounds the hidden CPU penalty to the drain window (`_SPEC_FUTURE_DRAIN_TIMEOUT_S`, currently 5s) rather than the full draft-exec tail. The rejected fork is never cached for reuse in this mode; combining with `enable_fork_reuse=true` still caches forks on accepts. Wired through `BenchmarkConfig.scenario_options["eager_fork_cleanup_on_reject"]` → `RealHostScenarioHarness(eager_fork_cleanup_on_reject=...)` → `_SpeculativeSandboxController.eager_cleanup_on_reject`. |

Additional `spec` scenario constraints:

- `agent` must be `mini_swe`
- `llm_service` must be `mini_swe_spec_trace_replay`
- `iterations` must be exactly `0`

---

## 12. Benchmark Microbenchmark CLI

Defined in `benchmarks/bench_agent_cr_micro.py`.

| CLI Flag | Default | Description |
|---|---|---|
| `--iters` | `1000` | Number of scheduler/inspector microbenchmark iterations. |
| `--storage-iters` | `200` | Number of storage round-trip iterations. |
| `--executor-jobs` | `64` | Number of checkpoint jobs to submit in the executor benchmark. |
| `--runtime` | `"docker"` | Runtime to use: `docker` or `runc`. |
| `--out` | `""` | Output CSV path for results. |

---

## 13. Network Configuration

Defined in `integrations/sandboxes/runtime/network.py`.

| Configuration | Default | Description |
|---|---|---|
| Benchmark network CIDR | `10.250.0.0/24` (auto-selected from `10.250.0.0/16`) | A /24 network is selected that doesn't conflict with existing host routes. |
| `AGENT_CR_BENCHMARK_NETWORK_CIDR` env var | *(none)* | Override the benchmark network CIDR explicitly. Must be a /24 IPv4 network. |

---

## 14. Runtime Resolver Configuration

Defined in `agent_cr/host_inspector/runtime_resolver.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `docker_bin` | `str` | `"docker"` | Path/name of the Docker binary. |
| `runc_bin` | `str` | `"runc"` | Path/name of the runc binary. |
| `runc_state_root` | `str \| Path \| None` | `None` | Override runc state root (passed as `--root` to runc). |

---

## 15. Environment Variables

| Variable | Used in | Description |
|---|---|---|
| `AGENTCR_BENCH_DIR` | `benchmarks/real_host_scenario_base.py` | Override the benchmark data root directory. Set to `tmpdir` or `tmp` to use a temp directory. |
| `AGENT_CR_BENCHMARK_NETWORK_CIDR` | `integrations/sandboxes/runtime/network.py` | Explicit /24 network CIDR for benchmark sandbox networking. |

---

## 16. Scheduler Policies

These are not configurations themselves but determine how `SchedulerConfig` values are interpreted:

| Policy Class | Config Used | Behavior Summary |
|---|---|---|
| `CheckpointingPolicy` (default) | All `SchedulerConfig` fields | Change-driven checkpointing with min/force intervals and optional LLM-request preference. |
| `FaultToleranceCheckpointingPolicy` | Inherits from default | Same as default but sets `leave_running=True` so the sandbox continues executing during checkpoint. |
| `SpotPreemptionCheckpointingPolicy` | Uses `SchedulerConfig` (constructor) | Checkpoint only when `preemption_notice` metadata is present and `preemption_grace_remaining_seconds > 0`. |
| `TreeSearchCheckpointingPolicy` | None from config | Checkpoint when `tree_search_step` metadata is present. Always `leave_running=True`. |

---

## Summary of Default Values at a Glance

| Config | Key Parameter | Default |
|---|---|---|
| Scheduler | `min_checkpoint_interval_seconds` | 30.0 s |
| Scheduler | `force_checkpoint_after_seconds` | 600.0 s |
| Executor | `max_workers` | 4 |
| Executor | `max_retries` | 0 |
| Executor | `retry_backoff_seconds` | 0.05 s |
| Telemetry | `detail_level` | `"basic"` |
| Telemetry | `max_text_attribute_bytes` | 2048 |
| Interceptor | `upstream_timeout_seconds` | 3600.0 s |
| Host Inspector Client | `timeout_s` | 5.0 s |
| Host Inspector Server | `port` | 9782 (CLI) / 0 (programmatic) |
| Recovery | `recovery_delay_seconds` | 0.0 s |
| Benchmark | `zpool_size` | `"10G"` |
| Benchmark | `transfer_delay_ms` | 0.0 ms |
| Benchmark | `iterations` (fault/spot) | 3 |
| Benchmark | `iterations` (e2e) | 5 |
| Benchmark | `iterations` (tree) | 1 |
| Simulated LLM | `response_delay_ms` | 500 ms (CLI) / 0 (programmatic) |
| Runc Paths | `state_root` | `/run/agent-cr/runc` |
| Runc Paths | `zfs_dataset_prefix` | `agentcr/sandboxes` |
