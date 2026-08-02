# Authoring a custom one-task debug benchmark

Use this when you want to verify or smoke-test a specific runtime
behaviour — an inspector signal, a scheduler decision, the shape of a
checkpoint chain — without depending on the full terminal-bench
dataset. You drive the agent through a hand-crafted trajectory you
control, disable test verification, and inspect the debug logs.

The `bg-server-state-*` artifacts in this directory are the canonical
worked example (verifying that a long-running tmux-descendant Python
HTTP server trips `process_changed` after the bash-narrowed tmux
ignore rule, commit `a0e8e2d`). Its 7-turn trajectory deliberately
exercises every scheduler change-signal outcome — five turns with the
server alive driving full checkpoints (process-only and process+fs),
one bash-builtin `>` redirect with no tracked pid driving an fs-only
checkpoint (`process_kind=full` but `process_artifacts=0`), and one
pure-stdout `echo` driving a skip (`reason=no_change_signal`, no
manifest written). Treat them as a reference layout for both the file
shape and for designing trajectories that exercise specific code
paths.

## How the pieces fit

```
   benchmark YAML  (task_dataset:)
        │
        ▼
   dataset JSONL  (task_root, docker_compose_file, llm_service_config.trace_path)
        │
        ├─► task dir   (Dockerfile, docker-compose.yaml, task.yaml, run-tests.sh)
        └─► trajectory.json  (ATIF-v1.6 user + agent steps with tool_calls)
```

Two path-resolution rules to remember:

- Paths inside the **YAML** (e.g. `task_dataset`) resolve relative to
  the YAML's directory — see `_resolve_optional_path` in
  `benchmarks/config.py`.
- Paths inside the **dataset JSONL** (`task_root`,
  `docker_compose_file`, `llm_service_config.trace_path`) resolve
  relative to the JSONL's directory — see `load_task_dataset` in
  `benchmarks/core.py:52`.

Absolute paths are accepted everywhere but make the artifacts
non-portable across clones, so prefer relative paths and colocate the
YAML + JSONL + task dir under `benchmarks/examples/debug/` (the
project-level `data/` tree is gitignored, so do **not** stage anything
there).

## Step-by-step

### 1. Pick a `task_id` and a base image

Choose a unique `task_id` (it ends up in the Docker image name as
`crab-termnius-<task_id>`). Pick a small base image — usually
`ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624`, already
cached on every benchmark host — and add only what your trajectory
actually needs.

### 2. Lay out the task directory

Under `benchmarks/examples/debug/<task_id>/`:

- `Dockerfile` — single layer on top of the base image is usually
  enough.
- `docker-compose.yaml` — copy from any cached task (e.g.
  `results/original-tasks/analyze-access-logs/docker-compose.yaml`)
  and don't change the env-var references; they're populated by the
  benchmark harness at runtime.
- `task.yaml` — required fields: `instruction`, `parser_name`,
  `max_agent_timeout_sec`, `max_test_timeout_sec`,
  `run_tests_in_same_shell`. The rest are metadata.
- `run-tests.sh` — required to exist, but can be `exit 0` when you
  set `verification_enabled: false` in the YAML.

### 3. Write the trajectory

`trajectory.json` is ATIF-v1.6:

```json
{
  "schema_version": "ATIF-v1.6",
  "session_id": "<unique>",
  "agent": {"name": "terminus-2", "version": "2.0.0",
            "model_name": "synthetic/<task_id>",
            "extra": {"parser": "json", "temperature": 0.0}},
  "steps": [
    { "step_id": 1, "timestamp": "...", "source": "user",  "message": "<prompt>" },
    { "step_id": 2, "timestamp": "...", "source": "agent", "message": "Analysis: ...\nPlan: ...",
      "tool_calls": [
        {"tool_call_id": "t1_1", "function_name": "bash_command",
         "arguments": {"keystrokes": "...\n", "duration": 2.0}}
      ]
    },
    ...
    { "step_id": N, "timestamp": "...", "source": "agent", "message": "...",
      "tool_calls": [{"tool_call_id": "tn_1", "function_name": "mark_task_complete", "arguments": {}}]
    }
  ],
  "final_metrics": {}
}
```

Rules the parser at
`integrations/llm_services/terminus_trace_replay/service.py` enforces:

- Exactly one initial `source=user` step — its `message` becomes the
  agent's prompt.
- Every subsequent `source=agent` step becomes one assistant turn.
  Its `message` is split on `Plan:` into the `analysis` / `plan`
  fields of the JSON response terminus expects.
- Each `tool_calls` entry with `function_name=bash_command` becomes
  one shell command (each must end with `\n` — those are real
  keystrokes sent to tmux).
- `function_name=mark_task_complete` sets `task_complete: true` in
  the assistant response, which terminus reads to end the loop.
- Timestamps drive the response-delay simulator when
  `response_delay_policy: trace_replay`. 1–5 s deltas keep smoke runs
  fast; large deltas make the benchmark wait. Validate by calling
  `parse_replay_trace` from a python -c one-liner before you run.

For best signal-to-noise, put **one command per agent step** so each
turn boundary is a clean inspector poll. Multiple commands in one
turn just collapse into one shell burst between two polls.

### 4. Author the dataset JSONL

One row per task, but for a debug smoke you usually want exactly one.
Minimum fields (see `BenchmarkTaskRecord` in `benchmarks/support.py`):

```json
{"agent_type":"terminus","docker_compose_file":"<task_id>/docker-compose.yaml",
 "llm_service_config":{"trace_path":"<task_id>/trajectory.json"},
 "llm_service_type":"terminus_trace_replay","service_name":"client",
 "task_config":{"options":{"max_agent_timeout_sec":180.0,"max_test_timeout_sec":30.0,
                           "run_tests_in_same_shell":false,"task_id":"<task_id>"}},
 "task_description":{"prompt":"<same as trajectory step-1 message>"},
 "task_id":"<task_id>","task_root":"<task_id>",
 "trace_replay_progress_count":<N-1>,"trace_response_count":<N-1>,"trace_malformed_line_count":0}
```

Validate it loads:

```bash
python3 -c "from benchmarks.core import load_task_dataset; from pathlib import Path; \
  r=load_task_dataset(Path('benchmarks/examples/debug/<task_id>-dataset.jsonl'))[0]; \
  print(r.task_id, r.docker_compose_file, r.llm_service_config['trace_path'])"
```

### 5. Author the benchmark YAML

Start by copying the closest existing config (for terminus,
`terminus.e2e.bgserver.smoke.yaml` in this directory) and change:

- `task_dataset:` — point at your JSONL.
- `sandboxes`, `max_workers`, every `phase_workers.*` and
  `executor.*_workers` — set to 1 for a single-task smoke.
- `verification_enabled: false` — skip test runs (so the noop
  `run-tests.sh` is fine).
- `iterations: 1`, `scenario_options.injection_rate: 0.0` — baseline
  fault scenario with zero fault injection.
- `log_level: debug` AND `host_inspector.log_level: DEBUG` AND
  `host_inspector.log_file: true` — without all three you miss the
  per-poll inspector metadata.
- `log_file`, `output`, `telemetry.output` — name them after your
  task so they don't collide with other runs.

### 6. Run and inspect

```bash
python3 -m benchmarks.run --config benchmarks/examples/debug/<your>.yaml
```

The last line of stdout prints the run directory:

```
artifacts replicated to %s <benchmark_root>/<TIMESTAMP>
```

Inside that directory:

| File | What's in it |
|---|---|
| `host-inspector.log` | per-poll inspector metadata at DEBUG: `tracked_pids`, `ignored_pids`, `fs_ignored_pids`, `dirty_pids`, `baseline_pids`, `process_changed`, `filesystem_changed`, `live_dirty` |
| `<your-log-file>.log` | scheduler decisions: `should_checkpoint`, `reason`, `observed_process_changed`, `observed_filesystem_changed`, `checkpoint_process`, `checkpoint_filesystem` |
| `storage/manifests/<sandbox-id>/*.json` | one manifest per checkpoint; key fields are `process_kind`, `parent_checkpoint_id`, `process_artifacts`, `filesystem_artifacts`, `metadata.reason` |
| `<your-telemetry>.telemetry.jsonl` | structured telemetry stream — fine-grained but verbose; use the report directory below for charts |

### 7. Useful one-liners

```bash
# 1) Inspector classifications across the run:
grep -E "ignore_process_rules|tracked_pids|ignored_pids" <run>/host-inspector.log | head

# 2) Per-poll process_changed / fs_changed / live_dirty:
grep "status sandbox=" <run>/host-inspector.log | head

# 3) Scheduler decisions:
grep "Scheduler selected" <your-log-file>.log

# 4) Checkpoint chain shape (the headline signal for an A/B comparison):
for m in $(ls -tr <run>/storage/manifests/*/*.json); do
  python3 -c "
import json; d=json.load(open('$m'))
print('%s created=%s process_kind=%s proc=%d fs=%d reason=%s' % (
  d['checkpoint_id'][:12], d['created_at'][11:23],
  d.get('process_kind','full'), len(d['process_artifacts']),
  len(d['filesystem_artifacts']), d.get('metadata',{}).get('reason','?')))"
done
```

## Pattern: branch A/B comparison

When you're verifying a fix that changes inspector or scheduler
behaviour, run the same trajectory on both branches and diff the
manifest chain. The `.master.yaml` next to the main YAML in this
directory shows the recipe:

1. Make a second copy of the YAML pointing the same `task_dataset`,
   but with `log_file`, `output`, `telemetry.output` renamed (e.g.
   `…smoke.master.*`) so the runs don't overwrite each other's
   artifacts.
2. `git stash && git checkout master`, run the `.master.yaml`,
   capture its run directory.
3. `git checkout <your-branch> && git stash pop`, run the main YAML.
4. Diff the two `storage/manifests/.../*.json` chains. Differences
   in `process_kind`, `proc_artifacts`/`fs_artifacts` counts, and the
   number of manifests are the cleanest signal of behavioural change.

## Gotchas worth remembering

- `data/` is gitignored. Put committed scratch artifacts under
  `benchmarks/examples/debug/`, not under `data/custom-tasks/`.
- `iterations` for `scenario: fault | spot` controls event-injection
  chunking, not replay count; `scenario: spec | tree` ignores it.
  For a no-fault baseline, use `scenario: fault` with
  `injection_rate: 0.0` and `first_forced_event_chunk: 0`.
- The first checkpoint is always full because of
  `checkpoint_full_baseline_on_first_checkpoint: true`. Don't read
  the baseline as evidence of the rule under test.
- For terminus, the pane bash is pre-started before the first LLM
  request — so the baseline checkpoint captures tmux + bash. If your
  trajectory depends on processes that start later, you cannot
  recover them by restoring the baseline image alone.
- ZFS cleanup ordering when iterating: stop runc containers first,
  then destroy datasets. The harness does this for you on a clean
  exit; if you `Ctrl-C` mid-run you may need to clean manually.
- Don't run the full unittest suite to verify; specific modules
  (`tests.test_host_inspector_server`, etc.) are enough and the full
  suite has known flaky tests.

## Where to grow this guide

Things this doc does not yet cover and that future contributors should
add as they hit them: speculation/spec replay scenarios, tree-search
scenarios, fault injection knobs (`scenario_options.injection_rate`
and the forced-chunk dial), the iflow / mini_swe / claude_code agent
variants, custom telemetry attributes, building tasks that need
multi-container compose setups (sidecars, real network services).
