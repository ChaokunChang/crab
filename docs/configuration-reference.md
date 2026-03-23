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

**How to specify:** Pass a `SchedulerConfig` dataclass to `build_default_system(scheduler_config=...)` or to `CRScheduler(config=...)`.

---

## 2. Executor Configuration (`ExecutorConfig`)

Defined in `agent_cr/config.py`. Controls the checkpoint/restore job executor thread pool.

| Parameter | Type | Default | Description | Used by |
|---|---|---|---|---|
| `max_workers` | `int` | `4` | Number of restore worker threads in the `ThreadPoolExecutor`. Must be ≥ 1. | `CRExecutor.__init__()` in `executor.py` |
| `max_checkpoint_queue_size` | `int` | `10000` | Maximum number of pending checkpoint jobs in the queue before new submissions are rejected. Must be ≥ 1. | `CRExecutor.__init__()` in `executor.py` |
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
| `jsonl_path` | `Path \| None` | `None` | If set, telemetry events/metrics are appended to this JSONL file. | `build_default_system()` in `system.py`; `JsonlTelemetrySink` in `telemetry.py` |
| `keep_in_memory_copy` | `bool` | `True` | Keep a copy of all telemetry events/metrics in memory (`InMemoryTelemetrySink`). | `build_default_system()` in `system.py` |
| `detail_level` | `str` | `"basic"` | Telemetry detail level. `"basic"` or `"detailed"`. When `"detailed"`, command stdout/stderr is captured in telemetry. | `ConfiguredTelemetrySink` in `telemetry.py`; `RuncRuntime._command_finish_attributes()` |
| `capture_command_output` | `bool` | `False` | Capture stdout/stderr of runtime commands in telemetry events regardless of `detail_level`. | `ConfiguredTelemetrySink` in `telemetry.py`; `RuncRuntime._command_finish_attributes()` |
| `max_text_attribute_bytes` | `int` | `2048` | Maximum byte length for text attributes in telemetry. Longer values are truncated. Minimum enforced: 32. | `ConfiguredTelemetrySink`, `telemetry_truncate_value()` in `telemetry.py` |

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

### 7c. Host Inspector CLI (`agent_cr/host_inspector/__main__.py` and `server.py:main()`)

| CLI Flag | Default | Description |
|---|---|---|
| `--host` | `"127.0.0.1"` | Bind host. |
| `--port` | `9782` | Bind port. |
| `--process-poll-interval` | `1.0` | Process poll interval (compatibility). |
| `--helper-path` | `None` (auto-detected) | Path to the eBPF filesystem monitor helper binary. |
| `--runc-state-root` | `None` | Override runc state root for the runtime resolver. |
| `--log-level` | `"INFO"` | Python logging level. |

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
| `scenario` | `str` | *(required)* | Scenario type: `e2e`, `fault`, `spot`, `tree`. |
| `mode` | `str` | *(required)* | Execution mode: `manual` or `auto`. |
| `provider` | `str` | `"openai"` | LLM provider: `openai` or `anthropic`. |
| `agent` | `str` | `"simulated"` | Agent type: `simulated` or `iflow`. |
| `llm_service` | `str \| null` | `null` | LLM service type override: `simulated`, `manual`, `iflow_trace_replay`, `simulated_for_iflow`. |
| `task_dataset` | `path \| null` | `null` | Path to a task dataset file (relative to config file). |
| `sandboxes` | `int` | `1` | Number of concurrent sandboxes. Must be > 0. |
| `max_workers` | `int \| null` | `null` | Max worker threads. Defaults to `sandboxes` count. |
| `iterations` | `int` | scenario-dependent | Number of iterations. Defaults: e2e=5, fault=3, spot=3, tree=1. |
| `output` | `path \| null` | `null` | Path for CSV output file. |
| `log_file` | `path \| null` | `null` | Path for log file. |
| `log_file_mode` | `str` | `"append"` | Log file open mode: `append` or `write`. |
| `log_level` | `str` | `"info"` | Python log level: `debug`, `info`, `warning`, `error`, `critical`. |
| `benchmark_root` | `path \| null` | `null` | Root directory for benchmark run data. Falls back to `AGENTCR_BENCH_DIR` env var, then a temp directory. |
| `zpool_size` | `str` | `"10G"` | Size of the ZFS pool image file (passed to `truncate -s`). |
| `zpool_name` | `str \| null` | `null` | Explicit ZFS pool name. Auto-generated if `null`. |
| `zpool_image` | `path \| null` | `null` | Path to the ZFS pool image file. |
| `reuse_zpool` | `bool` | `false` | If `true`, keep the ZFS pool across benchmark runs. |
| `image_cache_root` | `path \| null` | `null` | Cache directory for Docker images. Default: `.cache/agent-cr/images`. |
| `transfer_delay_ms` | `float` | `0.0` | Simulated transfer delay (ms) applied as `recovery_delay_seconds` during auto-recovery. |
| `work_dir_host_root` | `path \| null` | `null` | Host-side working directory root for sandboxes. |

### Telemetry YAML Block (`telemetry:`)

| YAML Key | Type | Default | Description |
|---|---|---|---|
| `telemetry.output` (or top-level `telemetry_output`) | `path \| null` | `null` | Path for telemetry JSONL output. Defaults to `<benchmark_root>/telemetry.jsonl`. |
| `telemetry.detail_level` | `str` | `"basic"` | `"basic"` or `"detailed"`. |
| `telemetry.capture_command_output` | `bool` | `false` | Capture runtime command stdout/stderr in telemetry. |
| `telemetry.max_text_attribute_bytes` | `int` | `2048` | Maximum byte length per text attribute in telemetry. Must be > 0. |

### Scenario Options (`scenario_options:`)

#### Fault Scenario (`benchmarks/scenarios/fault.py`)

| Key | Type | Default | Description |
|---|---|---|---|
| `injection_rate` | `float` | `0.5` | Probability of injecting a fault per iteration. Range: [0.0, 1.0]. |
| `first_forced_event_chunk` | `int` | `0` | First chunk/iteration at which fault injection is allowed. Must be ≤ `iterations`. |

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
