# Speculative Execution Benchmark

This document describes the `spec` benchmark scenario for speculative execution experiments with Agent-CR.

## Purpose

The `spec` scenario measures whether speculative execution can reduce turn latency for replayed `mini_swe` tasks.

For each agent turn:

- the agent sends an `oracle` request to the normal replay stream
- the agent sends a `draft` request to a smaller, faster replay stream
- if the draft decision arrives first, the agent may execute it on a forked sandbox restored from the latest checkpoint
- when the oracle decision arrives, the agent either accepts the speculative work or discards it and runs the oracle command on the active sandbox

The benchmark is meant to quantify both:

- saved time from accepted speculative turns
- wasted time from rejected speculative turns

## Scope And Current Limits

Version 1 is intentionally narrow:

- only `agent: mini_swe` is supported
- only `llm_service: mini_swe_spec_trace_replay` is supported
- only full-trace replay is supported, so `iterations` must be exactly `0`
- `iflow`, `claude_code`, and agent-in-sandbox flows are not supported in this scenario

These constraints are enforced during config loading and again when the scenario is prepared.

## Execution Model

The current implementation works like this:

1. The agent issues paired draft and oracle requests with the same speculative pair id.
2. Agent-CR treats the pair as one checkpoint coordination unit, so only one checkpoint can be scheduled for that pair.
3. If the oracle response wins the race, the turn degrades to normal oracle execution.
4. If the draft response wins and a speculative fork is available, the draft command runs on the fork.
5. When the oracle response arrives, the agent compares the normalized command strings.
6. Matching commands accept the speculative turn and promote the fork.
7. Mismatched commands reject the speculative turn and execute the oracle command on the active sandbox.

Important details:

- command equality is based on normalized command text, not THOUGHT text
- submission turns are never draft-mutated
- fork reuse is gated on `scenario_options.enable_fork_reuse`; see [Fork Reuse](#fork-reuse) below for the full caching contract
- accepted fork promotion also copies exact LLM replay state onto the promoted sandbox id so the replay cursor stays consistent
- speculative fork creation is a consumer of existing checkpoints only; it must not create checkpoints itself

## Fork Reuse

Creating a speculative fork costs a ZFS clone plus a CRIU restore per turn. When the previous turn left both the active sandbox and its fork in identical state, that work is redundant — the next speculative step could run on the already-restored fork instead of cloning a fresh one.

### Caching contract

The controller caches a fork from one turn for use by the next turn only when all of the following hold:

1. `scenario_options.enable_fork_reuse` is `true`.
2. The turn reached the finalize path (`_finalize_speculative_fork` was called). Turns that short-circuit — oracle-first, draft-failure, no-fork-available, agent-timeout — bypass the state check and are not cache candidates.
3. The inspector reports `current_state_changed=False` for the active sandbox *and* `fork_state_changed=False` for the fork.
4. The reuse gate (`reuse_candidate=True`) accepts the turn. This is the controller's assertion that the fork's post-action state is well-defined — specifically, that the draft exec had completed by the time the oracle result arrived. Rejected turns whose draft was still mid-flight have `reuse_candidate=False`, so the fork snapshot cannot be trusted and is never cached even if both sides look unchanged.
5. The same sandbox is still alive. Any `restore_once(...)` or recovery event invalidates the per-sandbox cache, so cached forks never survive a restart.

When all five conditions hold, the fork handle is stashed on `_SpeculativeSandboxController._cached_fork[sandbox_id]`. The next speculative turn reads that slot before falling back to the normal fork-creation path, so the ZFS clone and CRIU restore are skipped.

### Telemetry attributes

Each `spec.turn.finish` event now carries:

- `fork_finalized` (`1` when the turn reached the finalize path, `0` otherwise)
- `reuse_candidate` (`1` when the reuse gate accepted the turn, `0` otherwise — always `0` when `fork_finalized=0`)
- `current_state_changed` / `fork_state_changed` (pre-existing; the inspector-derived state diff)
- `fork_created` / `fork_reused` (pre-existing; whether the next turn opened a fresh fork or consumed a cached one)

### Report funnel and gap attribution

When these attributes are present, the telemetry report's speculative-execution section grows a **Fork Reuse** subsection with:

1. A funnel: `total turns → finalized turns → state-unchanged → cache-eligible (state-unchanged ∧ reuse_candidate) → reused`.
2. Two gap rows that explain where reuse "leaked":
   - `state-unchanged → cache-eligible`: rejected turns whose draft hadn't finished by oracle-finish time, plus retryable-failure finalize paths. The state snapshot looks clean but the reuse gate refuses to cache because the fork's post-action state isn't defined.
   - `cache-eligible → reused`: cached forks the next turn never consumed — dominated by task boundaries (the last turn of each task caches a fork nobody reads) and cache invalidations from sandbox restore/recovery.
3. A fork-source table (`forks_created` vs `forks_reused`) and a reuse-rate summary.
4. An outcome matrix crossing accept/reject with state-unchanged/state-changed for finalized turns only.

A machine-readable `spec_fork_reuse.csv` mirrors the same counts (13 metrics including both `gap_*` rows).

When `enable_fork_reuse` is `false`, the same funnel still renders from telemetry, but `forks_reused` is always `0` — every cache-eligible turn shows up in the `cache-eligible → reused` gap. That's the intended read for ablation runs.

## Configuration

The main checked-in example is [mini_swe.spec.auto.10tasks.debug.yaml](/root/workspace/agent-cr/benchmarks/examples/mini_swe.spec.auto.10tasks.debug.yaml).

Minimal shape:

```yaml
scenario: spec
mode: auto
provider: openai
agent: mini_swe
llm_service: mini_swe_spec_trace_replay
iterations: 0
task_dataset: /root/workspace/agent-cr/results/datasets/...
scenario_options:
  acceptance_rate: 0.5
  draft_response_delay_scaling_factor: 0.5
  mismatch_policy: preserve_command_class
  enable_fork_reuse: false  # set true to cache & reuse forks across turns
llm_service_options:
  response_delay_policy: trace_replay
  response_delay_scaling_factor: 1.0
```

### Scenario Options

For `scenario: spec`, `scenario_options` are merged into each task row's `llm_service_config`.

- `acceptance_rate`
  - default `0.5`
  - legacy alias: `accept_rate`
  - probability that the draft replay response matches the oracle command
- `draft_response_delay_scaling_factor`
  - default `0.5`
  - legacy alias: `speculative_delay_scaling_factor`
  - additional multiplier applied only to draft replay latency
- `mismatch_policy`
  - currently only `preserve_command_class`
  - rejected draft turns mutate to a different command while trying to stay in the same command bucket
- `enable_fork_reuse`
  - default `false`
  - gates speculative fork caching across turns (see [Fork Reuse](#fork-reuse))
  - when disabled, the controller behaves as before: every speculative turn creates a fresh fork
  - wired through `BenchmarkConfig.scenario_options["enable_fork_reuse"]` → `RealHostScenarioHarness(fork_reuse_enabled=...)` → `_SpeculativeSandboxController`
- `eager_fork_cleanup_on_reject`
  - default `false`
  - when `true`, rejected speculative turns destroy the fork immediately instead of letting the draft exec keep running in the background. Tearing down the fork stops the `runc exec` subprocess from continuing to burn host CPU after the oracle has already moved on; the main thread drains the speculative future for up to `_SPEC_FUTURE_DRAIN_TIMEOUT_S` before returning.
  - the rejected fork is never cached for reuse in this mode (its post-action state is undefined once the sandbox is destroyed), so combining this with `enable_fork_reuse=true` only disables reuse on rejects — accepts still populate the reuse cache normally.
  - `benchmark.spec.hidden_penalty_ms` now reports the bounded cleanup window rather than the unbounded background exec tail. Raw `speculative_exec_ms` stays unchanged.
  - wired through `BenchmarkConfig.scenario_options["eager_fork_cleanup_on_reject"]` → `RealHostScenarioHarness(eager_fork_cleanup_on_reject=...)` → `_SpeculativeSandboxController.eager_cleanup_on_reject`

### LLM Service Options

`mini_swe_spec_trace_replay` also supports the normal replay-delay knobs under `llm_service_options`:

- `response_delay_policy`
- `response_delay_ms`
- `response_delay_scaling_factor`
- `minimal_delay`
- `maximal_delay`

For draft requests, the effective delay is:

1. compute the base replay delay from `response_delay_policy`
2. apply `response_delay_scaling_factor`
3. apply `draft_response_delay_scaling_factor`
4. clamp with `minimal_delay` and `maximal_delay`

## Telemetry And Reporting

The speculative path adds per-turn and per-task telemetry.

Per-turn:

- event: `spec.turn.finish`
- attributes include `pair_id`, `accepted`, `draft_first`, `oracle_first`, `fork_created`, `fork_reused`, `current_state_changed`, `fork_state_changed`, `fork_finalized`, and `reuse_candidate`

Per-turn metrics:

- `benchmark.spec.saved_ms`
- `benchmark.spec.penalty_ms`
- `benchmark.spec.hidden_penalty_ms`
- `benchmark.spec.net_gain_ms`
- `benchmark.spec.fork_restore_ms`
- `benchmark.spec.speculative_exec_ms`
- `benchmark.spec.accept_rate`

Per-task row fields and summary metrics:

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

Metric semantics:

- `benchmark.spec.saved_ms` / `spec_saved_ms`: time saved on accepted speculative turns.
- `benchmark.spec.penalty_ms` / `spec_penalty_ms`: agent-loop-visible delay from rejected speculation. This is the part that remains on the critical path.
- `benchmark.spec.hidden_penalty_ms` / `spec_hidden_penalty_ms`: rejected speculative work that continued on the fork after the oracle path had already moved on.
- `benchmark.spec.net_gain_ms` / `spec_net_gain_ms`: `saved_ms - penalty_ms`, so net gain uses visible penalty rather than hidden background work.

When these metrics are present, the telemetry HTML report adds a speculative-execution section with saved-time, agent-loop-penalty, hidden-reject-cost, net-gain, and accept-rate charts. When the fork-reuse attributes (`fork_finalized`, `reuse_candidate`) are present, the same section also renders the [Fork Reuse](#fork-reuse) funnel, gap attribution, fork-source, and outcome-matrix tables and exports `spec_fork_reuse.csv`.
Task-level report rows and charts are grouped by stable `task_run_id`, so promoted speculative sandboxes from one benchmark task run are merged together instead of appearing as separate task rows.

## Stale-Fork Diagnosis Notes

This section records the diagnosis from the `django__django-10973` replicated Mini-SWE run where some speculative sandboxes finished with empty submissions.

Observed failure shape:

- The failing sandboxes reached the same command boundary where the oracle command rewrote `django/db/backends/postgresql/client.py` via shell redirection, e.g. `cat > /testbed/django/db/backends/postgresql/client.py << 'EOF' … EOF`.
- The active sandbox completed that command, but an immediate post-command state check sometimes reported `current_state_changed=False`.
- In failing runs, the next speculative fork was cloned from the previous filesystem checkpoint, so later accepted speculative work inherited a stale filesystem and the final submission could be empty or invalid.
- In passing runs, the scheduler caught the filesystem change shortly after the command and stored a filesystem checkpoint before the next useful fork clone.

### Root cause: racy post-hoc `fd_kind`

Redirected writes can be hard to classify after process exit. Shell heredoc redirection turns a single command like `cat > file << 'EOF'` into a sequence of syscalls that racily reuses the fd slot:

1. `openat("/testbed/.../client.py", O_WRONLY|O_CREAT|O_TRUNC, 0644)` — returns the real regular-file fd.
2. `dup2(fd, 1)` — duplicates it onto fd 1.
3. `close(fd)` — closes the original fd.
4. The shell creates the heredoc pipe; the kernel reuses the just-freed fd slot for the pipe read end.
5. `write(1, …)` writes the heredoc body to the real file; eventually fd 1 is also closed.

The host-inspector C helper resolves `fd_kind` by `stat("/proc/<pid>/fd/<fd>")` **after** the `openat` has returned. Between the syscall return and the stat call, steps 2–4 can execute, so the helper frequently sees a fifo on the original fd slot and reports `fd_kind=fifo` for a write that actually landed on a regular file. The server's `_is_countable_fs_event` treated fifo/socket/char fds as non-persistent and silently dropped the event, leaving the openat as the only surviving evidence. If that single openat event was lost to ringbuf overflow or to the same race, the inspector reported `filesystem_changed=False` even though the write had succeeded.

### Fix

`agent_cr/host_inspector/server.py` now treats the BPF-captured absolute path as authoritative for mutating-open syscalls:

- If the syscall is `open`/`openat`/`openat2`/`creat`, the flags include `O_CREAT`, `O_TRUNC`, or `O_TMPFILE`, and the captured path is an absolute path outside of `/dev/`, `/proc/`, and `/sys/`, the event is latched as a filesystem change regardless of the post-hoc `fd_kind` value.
- If the path is empty or points at a pseudo filesystem, the previous `fd_kind`-based filter still applies — so mutating opens of `/dev/null`, `/proc/self/...`, etc. are still ignored.
- Write/metadata-fd syscalls continue to rely on `fd_kind` because no path is captured for them; they need userspace's fd-to-path mapping to classify. The heredoc case is now reliably caught by the open event alone.

Regression tests live in `tests/test_host_inspector_server.py`:

- `test_mutating_open_on_real_path_survives_racy_fd_kind` — parametrized over racy `fd_kind=fifo|socket`, checks that an openat to a `/testbed` path still latches `filesystem_changed=True`.
- `test_mutating_open_on_dev_null_still_dropped` — guards that pseudo-fs paths remain ignored.

### Event-delivery latency (unchanged)

Event delivery is still asynchronous — kernel tracepoint → BPF ring buffer → C helper → stdout pipe → Python reader thread → `HostInspectorDaemon._handle_fs_event()`. A command can finish and the Mini-SWE speculative finalizer can inspect the sandbox before that pipeline has updated `filesystem_changed`. The response gate provides the normal checkpoint boundary: a gated draft/oracle response is not released until Agent-CR has run checkpoint coordination for that request generation and any submitted checkpoint has completed or been skipped.

## Evaluation Sweep

The tracked evaluation sweep now uses:

- [spec.0.0.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.0.yaml)
- [spec.0.1.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.1.yaml)
- [spec.0.2.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.2.yaml)
- [spec.0.3.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.3.yaml)
- [spec.0.4.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.4.yaml)
- [spec.0.5.yaml](/root/workspace/agent-cr/benchmarks/evaluation/spec.0.5.yaml)

Those files sweep `acceptance_rate` from `0.0` to `0.5` while keeping the draft delay scaling fixed at `0.5`.
