# Telemetry Reference

This document describes the current telemetry emitted by Crab and the benchmark harness.
It is intended to be the canonical reference for:

- performance analysis
- latency breakdown analysis
- bottleneck diagnosis
- anomaly and long-tail inspection
- future analysis/reporting code

The telemetry stream is JSONL-based and is designed to be lightweight by default.

JSONL telemetry now uses an async buffered writer by default. The hot path enqueues records into a bounded queue, and a background writer flushes compact JSONL batches to disk. When the queue overflows in default `overflow_policy="drop_new"` mode, the writer emits a shutdown summary event (`telemetry.writer.dropped`) so analysis can detect lossy runs.

In benchmark `llm_server.launch_mode: process`, both the main benchmark process and the benchmark LLM router subprocess can write to the same JSONL file. Batch writes are still protected by a file lock so records stay intact across processes.

## Record Format

Each JSONL line is one record with this high-level shape:

```json
{
  "timestamp": "2026-03-23T01:45:06.408568+08:00",
  "kind": "event" | "metric",
  "name": "llm.crab_delay_ms",
  "value": 340.7377160037868,
  "attributes": {
    "run_id": "...",
    "sandbox_id": "fault-3",
    "task_id": "flood-monitoring-basic",
    "request_id": "...",
    "checkpoint_id": "...",
    "job_id": "...",
    "component": "interceptor"
  }
}
```

Notes:

- `event` records have no `value`.
- `metric` records always have numeric `value`.
- `attributes` is the primary join surface for analysis code.
- `timestamp` is wall-clock time in ISO-8601 format.

## Naming Convention

The preferred pattern for timed operations is:

- `foo.bar.start`
- `foo.bar.finish`
- `foo.bar.duration_ms`

These are produced by the shared `start_operation(...)` helper.

Important implications:

- `*.start` and `*.finish` are lifecycle markers.
- `*.duration_ms` is the measured wall-clock latency for that operation instance.
- `*.duration_ms` is usually the preferred source for latency analysis.
- `*.start` and `*.finish` are useful for count-matching, failure diagnosis, and chronology.
- `*.duration_ms` now excludes the helper's own telemetry emit cost. In `detail_level="basic"`, operation `*.start` and `*.finish` events are suppressed while `*.duration_ms` remains.

Some older `*_ms` metrics still exist for compatibility. When both exist, analysis code should prefer the newer `*.duration_ms` form.

## Common Correlation Attributes

Not every record contains every key, but these are the standard correlation fields:

| Attribute | Meaning |
| --- | --- |
| `run_id` | Logical benchmark run identifier. Use this first when a file may contain multiple runs. |
| `sandbox_id` | Sandbox/container identity. Primary join key for per-sandbox timelines. |
| `task_run_id` | Stable logical benchmark task-run identity. In speculative runs, multiple promoted sandboxes can share one `task_run_id`. |
| `task_id` | Logical benchmark task identity. Often attached by benchmark-level telemetry. |
| `request_id` | Logical LLM request identity. Used to connect interceptor-side and service-side records. |
| `request_generation` | Monotonic per-sandbox request generation used to distinguish overlapping gated requests from the same sandbox. |
| `checkpoint_id` | Checkpoint identity. Used to join checkpoint, restore, and recovery records. |
| `job_id` | Executor job identity. Used for queue wait and execution analysis. |
| `event_type` | Benchmark/recovery event type such as `fault` or `preemption`. |
| `component` | Producing subsystem, such as `interceptor`, `llm_service`, `scheduler`, `runtime`, `checkpoint`, `restore`, `recovery`, `benchmark`, or `system`. |
| `request_kind` | Optional classifier for request families such as Claude Code `main_loop`, `helper`, and `count_tokens`. |
| `pair_id` | Speculative-execution pair identifier used to correlate draft and oracle work for one agent turn. |
| `status` | Usually `succeeded`, `failed`, or `skipped` on finish/duration records. |
| `operation` | Runtime command operation name for generic command telemetry. |

## Metric Families

## 1. Interceptor And LLM Path

These metrics describe the request path from the sandbox, through the interceptor, to the benchmark LLM service, and back to the sandbox.

For Claude Code replay workloads, interceptor telemetry now distinguishes auxiliary traffic from scheduler-visible turns:

- `request_kind="main_loop"` for replay-turn Anthropic `POST /v1/messages` requests
- `request_kind="helper"` for auxiliary helper-model `POST /v1/messages` requests
- `request_kind="count_tokens"` for `POST /v1/messages/count_tokens`

Only gated requests emit non-zero `llm.gate_wait_ms` or `interceptor.response_gate.wait.duration_ms`. In current Claude replay runs, that means `main_loop` requests remain gate-visible while `helper` and `count_tokens` requests bypass the response gate.

### Lifecycle Events

| Name | Meaning |
| --- | --- |
| `interceptor.request.received` | Interceptor accepted a request from the sandbox. |
| `interceptor.request.forward.start` | Forwarding to the upstream LLM service started. |
| `interceptor.request.forward.finish` | Forwarding call returned from upstream. |
| `interceptor.request.upstream_response_received` | Upstream response bytes were available to the interceptor. |
| `interceptor.response_gate.wait.start` | Interceptor began waiting on the response gate. |
| `interceptor.response_gate.wait.finish` | Response gate wait completed. |
| `interceptor.response.released` | Response was released back to the sandbox after the matching request generation finished coordination. |
| `llm.service.request.start` | Benchmark LLM router/service started handling the request. |
| `llm.service.request.finish` | Benchmark LLM router/service finished handling the request. |
| `request.start` | Legacy hook-level request-start marker. |
| `request.finish` | Legacy hook-level request-finish marker. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `interceptor.request.forward.duration_ms` | Total time spent in the interceptor’s upstream transport call. |
| `llm.upstream_latency_ms` | Time spent in the interceptor server’s HTTP forwarding call to the upstream LLM endpoint. |
| `llm.service.request.duration_ms` | Time spent executing the benchmark LLM service handler. |
| `interceptor.response_gate.wait.duration_ms` | Response-gate wait duration measured by the shared operation helper. This now excludes the helper's own emit overhead. |
| `llm.gate_wait_ms` | Explicit blocking wait duration inside `wait_for_release(...)`. |
| `llm.crab_delay_ms` | Delay from “upstream response received” to “response released to agent”, sampled before telemetry writes so it no longer absorbs telemetry self-overhead. |
| `llm.interceptor_total_ms` | Total interceptor-side request latency from request handling start to release back to sandbox. |
| `llm.request_total_ms` | Legacy alias for interceptor total latency. Prefer `llm.interceptor_total_ms`. |

### Timing Relationship

The intended request chronology is:

1. `interceptor.request.received`
2. `interceptor.request.forward.start`
3. `llm.service.request.start`
4. `llm.service.request.finish`
5. `interceptor.request.forward.finish`
6. `interceptor.request.upstream_response_received`
7. `interceptor.response_gate.wait.start`
8. `interceptor.response_gate.wait.finish`
9. `interceptor.response.released`

The main latency relationships are:

- `llm.service.request.duration_ms` is the service handler time.
- `llm.upstream_latency_ms` includes upstream HTTP transport and therefore is usually greater than or equal to `llm.service.request.duration_ms`.
- `interceptor.request.forward.duration_ms` wraps the full forward call seen by the interceptor and therefore is usually greater than or equal to `llm.upstream_latency_ms`.
- `llm.crab_delay_ms` measures the delay introduced after the upstream response arrives but before the sandbox receives it.
- `llm.gate_wait_ms` is the explicit blocking portion of that delay.
- `interceptor.response_gate.wait.duration_ms` and `llm.gate_wait_ms` should now be very close; large persistent gaps usually indicate real wrapper work or clocking issues rather than telemetry overhead.
- `llm.interceptor_total_ms` is the end-to-end interceptor latency and therefore is usually the largest of the LLM path timing metrics.

When `response_gate_enabled=false` in request attributes, expect both response-gate metrics to be absent or zero for that request. This is the intended behavior for auxiliary Claude helper and token-count traffic.

For overlapping requests from the same sandbox, correlate by both `request_id` and `request_generation` when available. Response-gate release is generation-specific.

In practice, analysis code should treat:

- `llm.service.request.duration_ms` as service compute
- `llm.crab_delay_ms` as Crab-added response delay
- `llm.interceptor_total_ms` as sandbox-observed request latency at the interceptor boundary

## 2. Scheduler And Coordination

These metrics explain why checkpointing decisions were made and how much scheduler/executor overhead was incurred.

### Events

| Name | Meaning |
| --- | --- |
| `scheduler.evaluate` | Scheduler decision record with `should_checkpoint`, `reason`, and scope flags. |
| `scheduler.checkpoint_complete` | Scheduler state store updated after checkpoint completion. |
| `interceptor.state_changed` | Interceptor signaled that request state changed for a sandbox. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `scheduler.evaluate.duration_ms` | Time spent evaluating a scheduler decision. |
| `executor.job_queue_wait_ms` | Time from job submission to start of execution by the executor worker pool. This now reflects real worker contention instead of a forced single-checkpoint-thread bottleneck. |
| `executor.job.duration_ms` | Total execution time of a checkpoint or restore job. |
| `executor.job_duration_ms` | Legacy alias for `executor.job.duration_ms`. |
| `recovery.queue_wait_ms` | Time from recovery event enqueue to the start of recovery handling. This captures queueing before checkpoint selection and restore submission. |
| `recovery.restore.duration_ms` | Recovery-time wrapper around `restore_once(...)`. This lets reports separate restore execution from checkpoint selection and post-restore release/bookkeeping. |

### Executor Lifecycle

| Event | Meaning |
| --- | --- |
| `executor.job_submitted` | Job entered the executor queue. |
| `executor.job_dequeued` | Job left the queue and started execution. |
| `executor.job.start` | Executor started running the job body. |
| `executor.job.finish` | Executor finished the job body. |
| `executor.job_finished` | Legacy completion event with outcome details. |

## 3. Checkpoint Flow

Checkpoint telemetry exists at multiple layers. This is intentional.

### Layers

| Layer | Signal |
| --- | --- |
| System-level coordination | `component=system`, operation name `checkpoint.flow` |
| Composite checkpoint worker | `component=checkpoint`, operation name `checkpoint.flow` |
| Runtime/process/filesystem implementation | `component=runtime`, operation names like `sandbox.checkpoint_process` and `sandbox.checkpoint_filesystem` |

Because layers are nested, their durations are not additive. They are different views of the same end-to-end checkpoint.

### Core Lifecycle

| Name | Meaning |
| --- | --- |
| `checkpoint.flow.start` / `finish` / `duration_ms` | System-level or worker-level checkpoint flow. Use `component` to disambiguate. |
| `checkpoint.process.start` / `finish` / `duration_ms` | Process checkpoint worker step. |
| `checkpoint.filesystem.start` / `finish` / `duration_ms` | Filesystem checkpoint worker step. |
| `checkpoint.persist_artifacts.start` / `finish` / `duration_ms` | Artifact persistence time. |
| `checkpoint.persist_manifest.start` / `finish` / `duration_ms` | Manifest persistence time. |
| `checkpoint.write_manifest.start` / `finish` / `duration_ms` | Manifest file write only. |
| `checkpoint.retention_policy.start` / `finish` / `duration_ms` | Retention scheduling time after the manifest is written. Actual pruning can continue asynchronously in the background. |
| `checkpoint.captured_live_request` | Checkpoint metadata recorded an in-flight LLM request. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `checkpoint.flow.duration_ms` | Preferred checkpoint flow duration. Use `component` to distinguish system vs worker. |
| `checkpoint.process.duration_ms` | Process checkpoint step duration. |
| `checkpoint.filesystem.duration_ms` | Filesystem checkpoint step duration. |
| `checkpoint.persist_artifacts.duration_ms` | Artifact persistence duration. |
| `checkpoint.persist_manifest.duration_ms` | Manifest persistence duration. |
| `checkpoint.write_manifest.duration_ms` | Manifest file write duration, without retention cleanup. |
| `checkpoint.retention_policy.duration_ms` | Retention scheduling duration on the checkpoint hot path. It no longer includes background pruning/deletion work. |
| `checkpoint.total_ms` | Legacy alias for worker checkpoint total duration. Prefer `checkpoint.flow.duration_ms`. |
| `checkpoint.process_ms` | Legacy alias for `checkpoint.process.duration_ms`. |
| `checkpoint.filesystem_ms` | Legacy alias for `checkpoint.filesystem.duration_ms`. |
| `checkpoint.persist_artifacts_ms` | Legacy alias for `checkpoint.persist_artifacts.duration_ms`. |
| `checkpoint.write_manifest_ms` | Legacy alias for `checkpoint.write_manifest.duration_ms`. |
| `checkpoint.retention_policy_ms` | Legacy alias for `checkpoint.retention_policy.duration_ms`. |
| `checkpoint.persist_manifest_ms` | Legacy alias for `checkpoint.persist_manifest.duration_ms`. |

### Important Relationship

When process and filesystem checkpointing run in parallel, the sum of:

- `checkpoint.process.duration_ms`
- `checkpoint.filesystem.duration_ms`

can exceed:

- `checkpoint.flow.duration_ms`

This is expected and does not indicate a telemetry bug.

## 4. Restore And Recovery Flow

Restore and recovery telemetry are related but distinct:

- `restore.*` is the actual restore execution
- `recovery.*` is the higher-level policy/recovery orchestration around checkpoint selection and response release

### Restore

| Name | Meaning |
| --- | --- |
| `restore.flow.start` / `finish` / `duration_ms` | Restore flow duration. Use `component` to distinguish system vs worker. |
| `restore.resolve_manifest.start` / `finish` / `duration_ms` | Time to load and resolve a restorable manifest. |
| `restore.filesystem.start` / `finish` / `duration_ms` | Filesystem restore step. |
| `restore.process.start` / `finish` / `duration_ms` | Process restore step. |

Legacy aliases:

- `restore.resolve_manifest_ms`
- `restore.filesystem_ms`
- `restore.process_ms`
- `restore.total_ms`

Prefer the `*.duration_ms` names.

### Recovery

| Name | Meaning |
| --- | --- |
| `recovery.total.start` / `finish` / `duration_ms` | End-to-end recovery handling for one recovery event. |
| `recovery.started` | Recovery loop began handling a recovery event. |
| `recovery.finished` | Recovery loop finished handling a recovery event. |
| `recovery.select_checkpoint.start` / `finish` / `duration_ms` | Time spent choosing a checkpoint for recovery. |
| `recovery.response_release.start` / `finish` / `duration_ms` | Time spent trying to release a buffered response after restore. |
| `recovery.response_released` | A buffered response was actually released. |
| `recovery.event_received` | Fault/preemption event entered the recovery queue. |
| `recovery.checkpoint_skipped_stale_request` | Candidate checkpoint was rejected because its live request no longer matched the pending request or request generation. |
| `recovery.no_satisfiable_checkpoint` | No acceptable recovery checkpoint was found. |

### Important Relationship

`recovery.total.duration_ms` can include:

- checkpoint selection time
- optional delay before restore
- `restore.flow.duration_ms`
- response release work
- relaunch fallback behavior

Therefore:

- use `restore.flow.duration_ms` for actual restore cost
- use `recovery.total.duration_ms` for end-to-end recovery overhead

## 5. Runtime Command And Sandbox Layer

The runtime layer emits telemetry for both:

- specific operations such as `sandbox.runtime_pause.duration_ms`
- generic command telemetry via `sandbox.command` and `sandbox.command_duration_ms`

### Operation-Specific Runtime Metrics

Examples include:

- `sandbox.bundle_spec.duration_ms`
- `sandbox.runtime_create.duration_ms`
- `sandbox.runtime_start.duration_ms`
- `sandbox.runtime_pause.duration_ms`
- `sandbox.runtime_resume.duration_ms`
- `sandbox.runtime_delete.duration_ms`
- `sandbox.runtime_exec.duration_ms`
- `sandbox.runtime_state.duration_ms`
- `sandbox.checkpoint_process.duration_ms`
- `sandbox.restore_process.duration_ms`
- `sandbox.checkpoint_filesystem.duration_ms`
- `sandbox.restore_filesystem.duration_ms`
- `sandbox.zfs_create.duration_ms`
- `sandbox.zfs_destroy.duration_ms`
- `sandbox.zfs_clone.duration_ms`
- `sandbox.zfs_get_origin.duration_ms`

These are the preferred runtime cost signals.

### Generic Runtime Command Telemetry

| Name | Meaning |
| --- | --- |
| `sandbox.command` | Generic command record with `operation`, `command`, `returncode`, `success`, and optionally `stdout`/`stderr`. |
| `sandbox.command_duration_ms` | Generic command duration metric for the same command. |

Use the generic command telemetry for:

- inspecting exact argv
- grouping by `attributes.operation`
- debugging runtime failures

Use the operation-specific `sandbox.*.duration_ms` metrics for headline latency reporting.

Important:

- `sandbox.command_duration_ms` duplicates operation-specific runtime timing at a more generic level.
- Analysis code should not sum both indiscriminately.

## 6. Benchmark Harness And Scenario Metrics

Benchmark telemetry is the preferred source for benchmark-level summary metrics when available.

When `llm_server.launch_mode` is `process`, the benchmark LLM router emits its telemetry with the same `run_id` into the same JSONL stream as the harness.

### Phase Lifecycle

The benchmark harness now runs three explicit phases: `setup`, `run`, and `verification`.

| Name | Meaning |
| --- | --- |
| `benchmark.phase.setup.start` / `finish` / `duration_ms` | Whole-run timing for the `setup` phase. Carries `phase`, `phase_scope="run"`, `sandbox_count`, and `configured_max_workers`. |
| `benchmark.phase.run.start` / `finish` / `duration_ms` | Whole-run timing for the `run` phase. Carries `phase`, `phase_scope="run"`, `sandbox_count`, and `configured_max_workers`. |
| `benchmark.phase.verification.start` / `finish` / `duration_ms` | Whole-run timing for the `verification` phase. Carries `phase`, `phase_scope="run"`, `sandbox_count`, and `configured_max_workers`. |
| `benchmark.phase.<phase>.item.start` / `finish` / `duration_ms` | Per-sandbox timing inside a phase. Carries `phase`, `phase_scope="sandbox"`, plus benchmark sandbox/task attributes. |
| `benchmark.phase.<phase>.configured_max_workers` | Configured worker cap emitted once per phase. Carries `phase` and `phase_scope="run"`. |

### Task Lifecycle

| Name | Meaning |
| --- | --- |
| `benchmark.task.start` | Benchmark submitted the task future during the `run` phase. |
| `benchmark.task.finish` | Task future completed. |
| `benchmark.task.duration_ms` | Task end-to-end latency from benchmark task start to task future completion. |
| `benchmark.task.ready` | Sandbox task reported ready after launch or recovery. |
| `benchmark.task.verify.start` / `finish` / `duration_ms` | Verification script execution during the `verification` phase. |
| `benchmark.task.success_ratio` | Benchmark success ratio emitted as telemetry. |

### Scenario Row Metrics

These come from benchmark scenario logic and are benchmark-level outcome/breakdown metrics, not core system timings:

| Metric | Meaning |
| --- | --- |
| `benchmark.checkpoint_ms` | Benchmark-observed checkpoint latency. |
| `benchmark.restore_ms` | Benchmark-observed restore latency. |
| `benchmark.recovery_ms` | Benchmark-observed recovery handler latency. |
| `benchmark.readiness_ms` | Time from recovery completion to task ready again. |
| `benchmark.end_to_end_recovery_ms` | Event-to-ready recovery latency. |
| `benchmark.workload_resume_ms` | Time until useful workload progress resumes. |
| `benchmark.migration_ms` | Spot/preemption migration time. |
| `benchmark.budget_slack_ms` | Remaining slack relative to preemption grace budget. |
| `benchmark.checkpoint_batch_ms` | Batch checkpoint timing in e2e scenarios. |
| `benchmark.restore_batch_ms` | Batch restore timing in e2e scenarios. |
| `benchmark.replay_progress_ms` | Tree/replay progress interval timing. |
| `benchmark.fanout_ms` | Tree-search fanout overhead. |
| `benchmark.lost_actions` | Lost work/actions after recovery. |

These metrics often carry:

- `task_id`
- `task_run_id`
- `sandbox_id`
- `iteration`
- `event_type`
- `event_injected`
- `recovery_status`

That metadata is important for replay/fault/preemption analysis.

### Speculative Execution Metrics

The `spec` scenario emits additional per-turn metrics from the `mini_swe` agent:

| Metric/Event | Meaning |
| --- | --- |
| `spec.turn.finish` | Per-turn speculative completion event. Carries `pair_id`, `accepted`, `draft_first`, `oracle_first`, `fork_created`, `fork_reused`, `current_state_changed`, `fork_state_changed`, `fork_finalized`, and `reuse_candidate`. `fork_finalized=1` indicates the turn reached `_finalize_speculative_fork` (turns that short-circuit on oracle-first, draft-failure, or no-fork-available emit `fork_finalized=0`). `reuse_candidate=1` indicates the reuse gate accepted the turn — i.e. the draft exec had completed by the time the oracle result arrived, so the fork's post-action state is well-defined and it is eligible for caching when `scenario_options.enable_fork_reuse` is `true`. |
| `benchmark.spec.saved_ms` | Estimated time saved by an accepted speculative turn. |
| `benchmark.spec.penalty_ms` | Agent-loop-visible delay from a rejected speculative turn. |
| `benchmark.spec.hidden_penalty_ms` | Rejected speculative work that continued on the fork after the oracle path was already unblocked. |
| `benchmark.spec.net_gain_ms` | `saved_ms - penalty_ms` for the turn. |
| `benchmark.spec.fork_restore_ms` | Time spent waiting for a speculative fork restore to become usable. |
| `benchmark.spec.speculative_exec_ms` | Time spent executing the speculative draft command on the fork. |
| `benchmark.spec.accept_rate` | Running accepted/(accepted+rejected) ratio as observed by the agent. |

At the scenario-row level, the benchmark also emits task summaries that may appear in `task_summary.csv` and the HTML report:

- `spec_total_turns`
- `spec_accept_count`
- `spec_reject_count`
- `spec_accept_rate`
- `spec_saved_ms`
- `spec_penalty_ms`
- `spec_hidden_penalty_ms`
- `spec_net_gain_ms`
- `spec_fork_create_count`
- `spec_fork_reuse_count`

`spec_net_gain_ms` is computed as `spec_saved_ms - spec_penalty_ms`, so it reflects visible loop impact rather than hidden background fork cost.

When these metrics are present, the telemetry report adds a `Speculative Execution Summary` section with saved-time, agent-loop-penalty, hidden-reject-cost, net-gain, and accept-rate charts.
Task summary tables and task-level charts now group speculative promotions by `task_run_id`, so `spec-0`, `spec-0-spec-1`, and later accepted descendants appear as one logical task run in the report.

When `spec.turn.finish` events carry `fork_finalized` and `reuse_candidate`, the report also renders a `Fork Reuse` subsection with a `total turns → finalized → state-unchanged → cache-eligible → reused` funnel, two gap rows (`state-unchanged → cache-eligible`, `cache-eligible → reused`), a fork-source / reuse-rate summary, and an accept/reject × state-unchanged/state-changed outcome matrix. The same 13 metrics are exported as `spec_fork_reuse.csv`. Under `scenario_options.enable_fork_reuse: false`, `forks_reused` is always `0` by design — every cache-eligible turn falls into the `cache-eligible → reused` gap. See [speculative-execution-benchmark.md](/root/workspace/crab/docs/speculative-execution-benchmark.md#fork-reuse) for the full caching contract.

## 7. Image And Build Metrics

These are emitted by the real-host image/materialization path.

| Metric/Event | Meaning |
| --- | --- |
| `image.inspect_ms` / `image.inspect` | Docker image inspect cost and event. |
| `image.inspect_defaults_ms` | Time to inspect runtime defaults from image config. |
| `image.build_ms` / `image.build` | Docker build cost and event. |
| `image.build_cache_hit` | Image build was skipped because the image already existed. |
| `image.defaults_cache_hit` / `image.defaults_cache_miss` | Runtime-defaults cache outcome. |
| `image.cache_lock_wait_ms` | Time waiting for the export/cache lock. |
| `image.export_ms` | Docker export/rootfs materialization cost. |
| `image.export_cache_hit` / `image.export_cache_miss` | Rootfs export cache outcome. |
| `compose.build_cache_hit` | Compose translation used an already-built image. |

## Metric Relationships And Analysis Rules

## 1. Prefer `*.duration_ms`

When both exist:

- prefer `foo.duration_ms`
- treat legacy `foo_ms` as compatibility aliases

Recommended alias handling:

| Preferred | Legacy alias |
| --- | --- |
| `llm.interceptor_total_ms` | `llm.request_total_ms` |
| `executor.job.duration_ms` | `executor.job_duration_ms` |
| `checkpoint.flow.duration_ms` | `checkpoint.total_ms` |
| `checkpoint.process.duration_ms` | `checkpoint.process_ms` |
| `checkpoint.filesystem.duration_ms` | `checkpoint.filesystem_ms` |
| `checkpoint.persist_artifacts.duration_ms` | `checkpoint.persist_artifacts_ms` |
| `checkpoint.persist_manifest.duration_ms` | `checkpoint.persist_manifest_ms` |
| `restore.resolve_manifest.duration_ms` | `restore.resolve_manifest_ms` |

## 2. Use `component` To Disambiguate Multi-Layer Metrics

Some operation names intentionally exist at more than one layer:

- `checkpoint.flow.duration_ms`
- `restore.flow.duration_ms`

Disambiguate them by `component`:

- `component=system` means higher-level orchestration
- `component=checkpoint` or `component=restore` means composite worker execution

## 3. Avoid Double Counting Nested Metrics

Examples:

- `sandbox.command_duration_ms` duplicates command-level runtime timing already present in `sandbox.*.duration_ms`
- `checkpoint.flow.duration_ms` contains substeps such as `checkpoint.process.duration_ms`
- `restore.flow.duration_ms` contains substeps such as `restore.filesystem.duration_ms`
- `executor.job.duration_ms` contains worker execution, which itself contains checkpoint/restore substeps

Therefore:

- do not sum all durations to estimate wall-clock time
- use one layer at a time depending on the question

Examples:

- “How expensive is restore end-to-end?” use `restore.flow.duration_ms`
- “Which substep dominates restore?” use `restore.resolve_manifest.duration_ms`, `restore.filesystem.duration_ms`, `restore.process.duration_ms`
- “Which runtime command is hottest?” use `sandbox.*.duration_ms` or group `sandbox.command_duration_ms` by `operation`

## 4. Filter By `run_id`

Analysis code should always filter by `run_id` first when reading a telemetry file that may contain multiple runs.

## 5. Join By The Smallest Sufficient Key

Use:

- `request_id` for one LLM request
- `job_id` for one executor job
- `checkpoint_id` for one checkpoint/restore unit
- `sandbox_id` for one sandbox timeline
- `task_id` for one logical benchmark task

Do not use `sandbox_id` alone when a finer-grained key exists.

## 6. Exclude Failed Task Runs

When `--exclude-failed-tasks` is passed to the report CLI (or `exclude_failed_tasks=True` is passed to `analyze_telemetry_file`), the analyzer performs a first-pass scan to identify sandboxes where `benchmark.task.success_ratio` equals `0`. All telemetry records from those sandboxes are then excluded from analysis.

Filtering is at the **sandbox level**: if the same task is solved by multiple sandboxes (e.g., using different trajectories or models), only the sandboxes that actually failed are excluded. Sandboxes that succeeded on the same task are retained.

The analysis result includes:

- `exclude_failed_tasks` — whether filtering was enabled.
- `excluded_sandbox_task_pairs` — list of `(sandbox_id, task_id)` pairs that were excluded.

These fields appear in the HTML report summary table and in `summary.json`.

CLI usage:

```bash
python3 -m benchmarks.telemetry_analysis.report \
    --input telemetry.jsonl \
    --output-dir report/ \
    --exclude-failed-tasks \
    --figure-window-seconds 10
```

When `--figure-window-seconds` is set, checkpoint and restore line charts in the HTML/SVG report are averaged within each fixed-size time window. This applies to both load-over-time and latency-over-time figures.

## 7. Lifecycle Count Matching

For operations emitted via `start_operation(...)`, analysis code can validate telemetry consistency with:

- `foo.start` count
- `foo.finish` count

Any systematic mismatch suggests:

- a crash or exception between start and finish
- a bug in instrumentation coverage
- a truncated telemetry file

## 8. Report-Specific Notes

The HTML telemetry report keeps its existing aggregate sections and now adds two request-level views that are especially useful for Claude replay diagnosis:

- `Turn Analysis` now includes aggregate `all` rows plus per-`request_kind` rows and time-series figures when `request_kind` is present in telemetry.
- `Overhead Analysis` now includes a dedicated `llm.gate_wait` CDF figure, which helps distinguish a mostly-zero distribution with a long tail from a uniformly elevated gate cost.

CSV exports now mirror those views:

- `turn_analysis.csv`, `turn_timeseries.csv`, and `turn_cdf.csv` include a `request_kind` column
- `overhead_cdf.csv` exports the CDF points used by the overhead plots

## 9. Recommended Queries

For future analysis/reporting code, the most useful first-pass queries are:

1. Top operations by cumulative `*.duration_ms`
2. Top operations by invocation count
3. Top operations by P95/P99 latency
4. Task-level `benchmark.task.duration_ms`
5. Verification cost via `benchmark.task.verify.duration_ms`
6. LLM path breakdown:
   - `llm.service.request.duration_ms`
   - `interceptor.request.forward.duration_ms`
   - `llm.crab_delay_ms`
   - `llm.gate_wait_ms`
   - `llm.interceptor_total_ms`
7. Recovery breakdown:
   - `recovery.total.duration_ms`
   - `recovery.select_checkpoint.duration_ms`
   - `restore.flow.duration_ms`
   - `recovery.response_release.duration_ms`
8. Checkpoint breakdown:
   - `checkpoint.flow.duration_ms`
   - `checkpoint.process.duration_ms`
   - `checkpoint.filesystem.duration_ms`
   - `checkpoint.persist_artifacts.duration_ms`
   - `checkpoint.persist_manifest.duration_ms`
9. Benchmark outcome metrics:
   - `benchmark.task.success_ratio`
   - `benchmark.lost_actions`

## 10. Scope And Maintenance Note

This document describes the telemetry currently emitted by:

- core Crab runtime/scheduler/executor/system modules
- the interceptor and benchmark LLM service router
- the real-host benchmark harness and scenario code
- image/build/export helpers

If telemetry names or meanings change, this document should be updated together with the implementation so that future analysis code can continue to rely on it.
