# Claude Code Integration

This document records the current Claude Code integration in `agent-cr`, the decisions we made while stabilizing replay, and the cases we intentionally do not support yet.

See also:

- [agent-integration-notes.md](/root/workspace/agent-cr/docs/agent-integration-notes.md) for the more general checkpoint/restore lessons that came out of the Claude integration work.

## Overview

Claude Code is integrated as an agent-in-sandbox agent:

- the Claude Code CLI runs inside the sandbox
- its Anthropic API calls are intercepted on the host
- the host LLM router serves replayed Anthropic Messages responses from recorded Claude traces
- Claude still executes tools inside the sandbox, so benchmark verification checks the real filesystem and process state

The replay path is therefore:

1. sandbox Claude Code sends `POST /v1/messages`
2. interceptor pauses/releases around the request lifecycle
3. `claude_code_trace_replay` returns the next recorded Anthropic response as SSE
4. Claude executes the returned tools locally in the sandbox

## Current Decisions

### Trace source

We hard-cut to:

- `results/traces/tbench-claude-code-claude-opus4.6-trajectories/manifest.json`

For each manifest row, the canonical trace is:

- `results_path/<single task dir>/agent/trajectory.json`

We do not use the sibling `*-traj.json` files in that directory tree. They are lossy convenience artifacts and are not the source of truth.

### Replay modes

We use two practical replay modes:

- non-strict replay:
  - intended for broad trace coverage
  - supports `Task`, `TaskOutput`, and `TaskStop`
  - loads recorded subagent sidechains from `agent/sessions/projects/**/subagents/agent-*.jsonl`
- strict replay:
  - intended for benchmark-quality dataset generation
  - still excludes `Task`, `TaskOutput`, and `TaskStop`
  - only accepts traces whose API-visible tool inputs are reconstructable from the recorded files

Important decision:

- `strict replayable` means structurally reproducible, not semantically guaranteed to pass verification

### Benchmark dataset policy

The benchmark dataset is generated from successful manifest rows with:

- `--strict-replayable`
- `--deduplicate`

That gives one strict row per task. This is still not enough by itself, because some strict traces are dirty in a semantic sense even though their inputs are reconstructable.

Because of that, the generator now allows narrow task-specific trial preferences for empirically validated traces. Today this is used for `mailman`, where the shortest strict trace was reproducibly bad but two longer strict traces passed real verification.

### Version-pinned Claude binary

The sandbox Claude Code binary is pinned to the trace's recorded `agent_version`.

Resolution order is:

1. `AGENT_CR_CLAUDE_CODE_BINARY`
2. `AGENT_CR_CLAUDE_CODE_VERSION`
3. dataset-provided `trace_agent_version`
4. fallback only when no replay version is available

Current traces use Claude Code `2.1.34`, so the sandbox must not silently drift to a newer cached binary.

### No-fault benchmark policy

The no-fault Claude benchmark still uses the `fault` scenario driver, but with:

- `injection_rate: 0.0`
- `first_forced_event_chunk: 0`

Decision:

- no-fault no longer disables checkpointing implicitly
- the default checkpoint policy is determined by `scenario`
- if we want a no-checkpoint no-fault run, we must set:
  - `scheduler.policy: no_checkpointing`

This makes the behavior explicit in YAML instead of inferring it from fault-injection settings.

### Why verification uses `uv`

Claude does not have a special verifier. Claude and `iflow` both go through the same shared benchmark verification path in:

- [core.py](/root/workspace/agent-cr/benchmarks/core.py)
- [real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py)

At a high level, verification is still simple:

1. wait for the task to finish
2. run `/tests/run-tests.sh` inside the sandbox
3. record pass/fail from the exit code or task-specific grading

What makes it look complicated is the bootstrap around `run-tests.sh`, not the grading logic itself.

Why the bootstrap exists:

- many TerminalBench tasks ship `run-tests.sh` scripts that explicitly do:
  - `apt-get update`
  - `apt-get install ...`
  - `curl ... astral.sh/uv ...`
  - `uv venv ...`
  - `uv pip install ...`
  - `uv run pytest ...`
- so the verifier has to make sure that a usable `uv` and Python test environment exist inside the sandbox before the task-authored verifier script runs

Concrete example:

- [run-tests.sh](/root/workspace/agent-cr/results/original-tasks/torch-tensor-parallelism/run-tests.sh)

Decision:

- we keep the benchmark contract as "run the task's own `run-tests.sh`"
- we do not rewrite every task verifier into a host-specific harness
- instead, we provide a small `uv` compatibility shim so these task-authored scripts behave consistently inside benchmark sandboxes

Why `iflow` may have felt simpler:

- `iflow` uses the same verifier
- the difference was mostly observational, not architectural
- earlier `iflow` runs happened not to expose as many verifier edge cases
- Claude runs hit more dependency-heavy and `pytest`-sensitive tasks, which forced us to harden the shared verifier

## What We Implemented

### Agent and sandbox support

Implemented in:

- [claude_code.py](/root/workspace/agent-cr/integrations/agents/claude_code.py)
- [harness.py](/root/workspace/agent-cr/integrations/sandboxes/claude_code/harness.py)

Key behaviors:

- launches Claude Code inside the sandbox with the required wrapper semantics
- preserves `exec` and `> /dev/null 2>&1` behavior where required
- sets the Anthropic base URL to the host router correctly
- pins the Claude binary version to the trace
- handles replay progress and restore/reactivation for replay workloads
- keeps Claude home/state and debug artifacts on mounted host state, while avoiding long-lived wrapper stdio redirection to mutable mounted log files
- mounts the full Claude home root on host state instead of copying only `/root/.claude`, so Claude temp files like `.claude.json.tmp.*` stay restore-safe too

### Replay service

Implemented in:

- [request_classification.py](/root/workspace/agent-cr/integrations/llm_services/claude_code_trace_replay/request_classification.py)
- [service.py](/root/workspace/agent-cr/integrations/llm_services/claude_code_trace_replay/service.py)

Key behaviors:

- parses the ATIF `agent/trajectory.json` format
- merges recorded agent steps into actual Anthropic response turns
- serves SSE in Anthropic Messages format
- classifies Claude requests into `main_loop`, `helper`, `count_tokens`, and defensive `other`
- treats only `main_loop` requests as replay turn boundaries
- returns synthetic `count_tokens` responses without consuming replay turns
- handles helper-model side requests without consuming main trace turns
- supports non-strict `Task` sidechains via recorded subagent JSONL files
- rewrites recorded background task IDs to live task IDs
- remaps dynamic filenames when the task regenerates randomized asset names
- rewrites brittle Git hash literals when runtime commit IDs differ
- applies narrow runtime-specific normalization for dirty traces, including the Mailman/Postfix `relay_domains` conflict

### Request taxonomy and turn boundaries

For the replayable Claude benchmark corpus, the request classifier uses the following model:

- `main_loop`:
  - `POST /v1/messages`
  - model family matches the replay trace's main model family, which is currently Opus in the benchmark dataset
  - these are the user-visible assistant turns recorded in `agent/trajectory.json`
  - example:
    - request path: `POST /v1/messages`
    - request model: `claude-opus-4-6`
    - typical effect: returns the next replayed assistant turn, for example an Opus response that emits a `Bash` tool call
- `helper`:
  - `POST /v1/messages`
  - model family differs from the replay trace's main model family
  - in the replayable benchmark dataset, these are the fast Haiku-family helper calls Claude Code uses around local tool execution
  - example:
    - request path: `POST /v1/messages`
    - request model: `claude-haiku-4-5-20251001`
    - typical effect: returns a short helper response around local tool execution and does not consume the next replay turn
- `count_tokens`:
  - `POST /v1/messages/count_tokens`
  - token-count probes used by the Claude client
  - example:
    - request path: `POST /v1/messages/count_tokens`
    - request payload: the current Claude conversation is sent for token estimation
    - typical effect: returns a synthetic token-count response such as `{"input_tokens": ...}` and does not consume replay progress
- `other`:
  - any unexpected path we handle defensively without letting it advance replay progress
  - example:
    - request path: anything outside the replayed Anthropic Messages endpoints, such as `/v1/models`
    - typical effect: classified defensively as non-turn traffic and never allowed to advance `trace_cursor`

Important consequences:

- `trace_cursor` tracks committed `main_loop` turns only
- helper and `count_tokens` traffic never advances `trace_cursor`
- benchmark checkpoint metadata keeps reading `snapshot().trace_cursor`, so `benchmark_trace_cursor` now means main-loop replay progress, not all Claude API requests
- replay restore still keys off consumed main-loop responses only

Tool blocks such as `TodoWrite`, `WebFetch`, and `WebSearch` are still part of a containing `main_loop` turn. They are not separate request kinds.

### Response gating and checkpoint capture

Claude Code replay now uses the same request taxonomy in the live interceptor path as in the replay service.

Implemented in:

- [interceptor.py](/root/workspace/agent-cr/agent_cr/interceptor.py)
- [system.py](/root/workspace/agent-cr/agent_cr/system.py)
- [real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py)

Behavior:

- only `main_loop` Claude requests arm the response gate
- `helper` and `count_tokens` requests still flow through the interceptor and telemetry, but bypass response-gate blocking
- live-request checkpoint metadata is only captured for gated requests, so auxiliary Claude traffic does not create a checkpoint/restore dependency
- replay snapshots expose diagnostic counters such as `main_loop_request_count`, `helper_request_count`, and `count_tokens_request_count` without changing restore semantics

This split matters because the replayable benchmark dataset records main-loop progress in `trajectory.json`, while helper and token-count traffic is auxiliary client behavior. Treating all Claude requests as turn boundaries caused unnecessary checkpoint scheduling, large gate waits, and misleading latency analysis.

### Runtime and benchmark fixes

Implemented across:

- [real_host_scenario_base.py](/root/workspace/agent-cr/benchmarks/real_host_scenario_base.py)
- [fault.py](/root/workspace/agent-cr/benchmarks/scenarios/fault.py)
- [runc.py](/root/workspace/agent-cr/agent_cr/runtime/runc.py)

Key fixes:

- no-fault checkpoint behavior is now controlled explicitly by `scheduler.policy` instead of being inferred from `injection_rate: 0.0`
- verification bootstraps a working `uv` shim inside the sandbox
- verification no longer races a still-paused replay container after the final intercepted request
- Postfix spool ownership is repaired in copied rootfs bundles so Mailman/Postfix can actually start
- runtime/ZFS timeouts are configurable for larger benchmark runs
- mixed restore no longer fails on wrapper-owned mounted log fds, because the long-lived Claude wrapper now sends live `stdout` / `stderr` to `/dev/null` instead of a mutable mounted output log

### Checkpoint-safe mounted I/O

One Claude-specific bug turned into a useful general integration lesson.

Original behavior:

- Claude state was correctly mounted under the host work directory
- but the long-lived shell wrapper also redirected its live `stdout` / `stderr` to a mounted `claude_code.output.log`

Why that broke mixed restore:

- mounted state is excluded from rootfs-diff monitoring, which is good
- but the restored baseline process still brings back its open file descriptors
- if the mounted output log has grown on the newer filesystem checkpoint, CRIU sees that the restored fd no longer matches the file metadata it checkpointed
- the result was restore failures such as `bad size` on `opt/claude-code-logs/claude_code.output.log`

What we changed:

- keep Claude's mounted host state
- mount the full Claude `HOME` on host state instead of only mounting `.claude`
- keep Claude's agent-native `--debug-file` on the mounted logs directory
- stop redirecting the wrapper's live `stdout` / `stderr` to a mutable mounted file
- send the wrapper's `stdout` / `stderr` to `/dev/null` instead

Why this is the right split:

- mounted state is still the right place for agent home, sessions, structured logs, and completion markers
- wrapper-level stdio is low-value compared with successful restore
- this matches the safer pattern already used by `iflow`

This fix is documented more generally in [agent-integration-notes.md](/root/workspace/agent-cr/docs/agent-integration-notes.md).

### Dataset generation

Implemented in:

- [generate_claude_code_replay_dataset.py](/root/workspace/agent-cr/benchmarks/generate_claude_code_replay_dataset.py)

Behavior:

- reads the new manifest-driven bundle
- keeps all successful traces by default
- round-robins multi-success tasks in the full replay dataset
- supports `--strict-replayable`
- supports `--deduplicate`
- supports `--exclude-task`
- supports `--exclude-trial-id`
- supports `--include-trial-id`
- emits `trace_trial_id`, `trace_task_checksum`, `trace_result`, `trace_total_steps`, and `trace_agent_version`
- currently includes a narrow preferred-trial override for `mailman`
- excludes `git-multibranch` and `hf-model-inference` by default because they are not currently benchmark-valid
- excludes a few empirically bad duplicate trials by default:
  - `41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8` (`mailman`)
  - `b45538f5-0862-45f0-9758-f97e8fb29037` (`mailman`)
  - `8b56ce95-9f8b-4020-839d-5d1cc6dc9a10` (`qemu-alpine-ssh`)
  - `d19326ec-e1ff-476b-9b76-83103b1c8694` (`qemu-startup`)
  - `4437cac5-d01b-43b8-ac37-f1d6f26cea89` (`fix-git`)

Useful commands:

```bash
python3 benchmarks/generate_claude_code_replay_dataset.py \
  --out results/datasets/claude_code_replay.jsonl

python3 benchmarks/generate_claude_code_replay_dataset.py \
  --strict-replayable \
  --deduplicate \
  --out results/datasets/claude_code_replay_benchmark.jsonl

python3 benchmarks/generate_claude_code_replay_dataset.py \
  --strict-replayable \
  --deduplicate \
  --exclude-task some-other-task \
  --out results/datasets/claude_code_replay_benchmark.jsonl

python3 benchmarks/generate_claude_code_replay_dataset.py \
  --strict-replayable \
  --include-trial-id 8286d88c-726a-4eeb-91da-f95cc6082630 \
  --include-trial-id 99db6d5b-18ef-4743-9956-51e5cc7f670a \
  --out results/datasets/claude_code_replay_benchmark_fix.jsonl
```

## Replayability Definitions

We use three different notions of "good" trace:

- replayable:
  - the runtime can emit a plausible response stream
  - this may still skip tools or rely on degraded behavior
- strict replayable:
  - all important API-visible tool inputs can be reconstructed from the trace files
  - this is a structural property
- benchmark-valid:
  - the replayed task actually passes the benchmark verifier in a real sandbox
  - this is an empirical property

Decision:

- strict replayable is necessary for the benchmark dataset
- strict replayable is not sufficient for the benchmark dataset

The `mailman` task is the clearest example. Multiple strict traces existed, but only some of them passed real verification, so the benchmark dataset now prefers empirically validated trials.

## Unsupported Or Partial Cases

### 1. Legacy Claude trace bundles

Not supported:

- the old dirty Claude trace bundles
- the sibling `*-traj.json` files in the new bundle

Why not:

- they are lossy
- older bundles contain placeholder-heavy tool inputs
- they caused frequent replay/verification divergence
- we have a better canonical source now: `results_path/.../agent/trajectory.json`

### 2. Strict-mode `Task`, `TaskOutput`, `TaskStop`

Current status:

- supported in non-strict replay
- intentionally excluded by `--strict-replayable`

Why not in strict mode:

- they depend on subagent/client-side behavior that is not part of the plain Anthropic Messages tool surface
- even with recorded sidechain files, we do not currently treat them as benchmark-grade deterministic API replay

### 3. Full emulation of purely client-side Claude tools

Current status:

- these are skipped rather than fully emulated:
  - `AskUserQuestion`
  - `EnterPlanMode`
  - `ExitPlanMode`
  - `TodoWrite`
  - `WebFetch`
  - `WebSearch`

Why not:

- Claude Code handles them client-side
- the trace does not expose a stable, complete server-side schema for their behavior
- some are informational or planning-only rather than task-state mutations
- emulating them faithfully would mean reimplementing Claude Code client behavior, not just replaying Anthropic API traffic

### 4. Automatic semantic validation of every successful trace

Not supported yet.

Current status:

- the generator knows structural strictness
- the generator does not automatically benchmark every candidate trajectory before selecting one
- we currently use small empirical overrides where needed

Why not yet:

- full semantic validation requires launching real benchmark runs for many candidate traces
- that is expensive and slow compared with structural filtering
- we only added narrow overrides where failures were reproduced and clearly attributable to bad trace choice

Practical consequence:

- `manifest result contains Pass` is not enough
- `strict replayable` is not enough
- the benchmark dataset is curated, not purely derived from manifest pass/fail

### 5. `git-multibranch`

Not supported yet as a benchmark-valid Claude replay task.

Current status:

- the trace is strict-replayable
- the live benchmark still fails reproducibly
- the dataset generator now excludes it by default
- the observed failure is not a placeholder issue

Why not yet:

- the failure is semantic and runtime-specific rather than structural
- in the live verifier, `git clone git@localhost:/git/project` fails with:
  - `fatal: protocol error: bad line length character: Welc`
- after that, the HTTPS endpoints return `404 Not Found` instead of deployed branch content
- this suggests the replayed setup is not producing a benchmark-valid Git-over-SSH service / deployment pipeline, even though the recorded tool inputs are reconstructable
- unlike `mailman`, we do not yet have a clearly validated alternative strict trial or a narrow, justified normalization that fixes the task without overfitting

Decision for now:

- document `git-multibranch` as a known unsupported benchmark case
- exclude it from generated Claude datasets unless explicitly opted back in via code changes

### 6. `hf-model-inference`

Not supported yet as a benchmark-valid Claude replay task.

Current status:

- the trace is structurally replayable enough to run
- the live benchmark still does not pass reliably under recovery
- the dataset generator now excludes it by default

Why not yet:

- the task depends on bringing up a long-lived local inference service with heavyweight model setup
- the failure mode is not a simple missing placeholder or single replay cursor bug
- we do not yet have a narrow fix or validated replacement trial that consistently survives checkpoint/restore in benchmark conditions

Decision for now:

- treat `hf-model-inference` as a known unsupported benchmark case
- exclude it from generated Claude datasets unless explicitly opted back in via code changes

### 7. Automatic binary download without a configured source

Partially supported.

Current status:

- version pinning is supported
- automatic download only works if `AGENT_CR_CLAUDE_CODE_BINARY_URL_TEMPLATE` is configured

Why not beyond that:

- we do not hardcode a vendor download URL in the repo
- environments may have different internal artifact mirrors or security policies

## Current Known Decision Points

- Use only the full `agent/trajectory.json` traces from the Opus 4.6 bundle.
- Keep strict mode conservative.
- Keep non-strict mode broader and capable of replaying Task-family traces.
- Treat benchmark dataset selection as a curation problem, not only a parser problem.
- Prefer narrow empirical trial overrides over broad task-specific heuristics.
- Do not claim that a manifest "Pass" trace is benchmark-safe until it has been validated in the real harness.

## Current Benchmark Dataset State

As of the current integration state:

- the strict deduplicated benchmark dataset contains 51 rows for 51 tasks
- the `mailman` benchmark row now points at empirically passing trial `46c2e780-0160-4acf-a1e6-668cc5ca506b`
- the strict duplicated benchmark dataset contains 167 rows after filtering the known bad duplicate trials above
- in that duplicated benchmark dataset, replay progress comes from the recorded main Opus turns; helper-model traffic and `count_tokens` probes are treated as auxiliary and do not contribute to benchmark turn boundaries

This should be treated as the current curated benchmark set, not as a mechanically perfect projection of all manifest-pass traces.
