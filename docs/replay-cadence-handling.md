# Replay-Cadence Handling

When agent-cr replays a recorded Terminus trace, the replay's per-turn
wall-clock can drift from the original capture in either direction. The
trace is just a sequence of recorded LLM responses — neither it nor the
trace-replay LLM service know how long each shell command will actually
take on the replay host. The replay agent re-issues those responses
verbatim, but the foreground commands they trigger run at the replay
host's resource budget, not the capture host's.

This doc covers two opposite-direction drift problems and how
agent-cr handles them.

---

## The two drift modes

### Slower replay — pane is busy when the trace's next turn arrives

In the capture, `make all` finished in 5 wait turns. In the replay (e.g.
under contention from a parallel sandbox or with weaker IO) it needs 7
wait turns. When the trace's first non-wait turn after the build (say
`make install`) arrives, the build is still grinding. Two failure modes
follow if we naively replay:

1. **Promotion of a doomed fork.** Speculative execution forked the
   sandbox at an earlier checkpoint and ran the next predicted batch on
   the fork. If the fork's prediction matches the oracle's response and
   we promote, `runc delete -f` on the active container kills the
   in-flight `make`/`gcc`/`ld` tree mid-run, and the fork's resumed
   `make` (started later from the older checkpoint) is necessarily
   behind the active's actual progress. Long enough builds never reach
   `make install`; verification then misses `/tmp/CompCert/ccomp`,
   `/usr/local/bin/povray`, etc.
2. **Keystroke queueing.** Even without fork promotion, sending the
   trace's next keystrokes (`make install\n`) into a busy pane just
   queues them behind the still-running `make all`. Bash buffers them,
   the next turn's observation captures stale output, and every
   subsequent turn ends up further behind.

### Faster replay — pane drained before the trace's wait turns end

Inverse situation: in the capture, `make build` took 5 wait turns. On
the replay host (more CPU, faster disks) it finishes in 3. Turns N+4
and N+5 are still recorded in the trace as `keystrokes=""` with
`duration=60`. The agent dutifully sleeps 60s × 2 doing nothing useful
— pure slack wall-time, no correctness issue.

---

## Mechanisms

### 1. Drain guards (slower replay)

Two `is_pane_idle()` checks on the active sandbox's tmux pane gate the
spec controller's fork lifecycle:

- **Pre-fork active-busy guard** (`integrations/agents/terminus.py`,
  `_spec_perform_step`): right after the draft+oracle race, before
  submitting the fork-restore future, probe `pane_current_command`. If
  it isn't a shell, skip fork creation entirely, wait for the active's
  pane to drain via `wait_for_quiescence`, then run oracle on the
  now-idle active. This bypasses CRIU restore + the runc-exec storm of
  a fork that would have been killed at promotion anyway.
- **Post-match active-busy guard** (same function, after `commands_match`
  branch): a narrow race where the active became busy between the
  pre-fork probe and oracle's return. Drains the same way, runs oracle
  on the active, drops the fork.

Both guards have a timeout (`_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S = 1800
s`). On timeout we fall through to running oracle on the still-busy
active — bash queues the new keystrokes behind the in-flight command
exactly as before the guards existed. So a stuck foreground command
never hangs the run; it just costs the timeout window before falling
back.

### 2. Fast-forward (faster replay)

When the trace's next batch is a pure-wait turn (all
`keystrokes=""` / whitespace-only) and the active pane is already
idle and stable, skip the recorded `min_timeout_sec` sleep instead of
dozing through it.

Implementation in `terminus.py:_maybe_fast_forward_idle_wait`. The probe
is conservative — three checks must all pass:

1. `is_pane_idle()` — `pane_current_command` is a shell name.
2. `capture_pane()` snapshot is unchanged across a `_FAST_FORWARD_SETTLE_S`
   (0.3 s) settle window — absorbs any final flushes from a just-
   finished foreground command.
3. `is_pane_idle()` again — no new fg command started during the settle
   window.

Any False answer falls through to the regular `send_keys` sleep — same
behavior as before this knob existed. The skip emits
`terminus.fast_forward.skip` and credits the saved ms to
`benchmark.terminus.fast_forward_saved_ms`.

The check applies in both replay modes: `terminus_trace_replay`
(baseline single-turn) and `terminus_spec_trace_replay` (speculative).
It runs inside `_execute_commands`, which is the convergence point for
all paths that reach the active sandbox's pane.

---

## Configuration

| YAML key | Default | What it controls |
|---|---|---|
| `scenario_options.fast_forward_idle_waits` | `true` | Enable §2 (fast-forward). Set to `false` to measure unmoderated trace cadence. |

The drain guards (§1) are unconditionally on — they're correctness, not
optimization. The pre-fork guard saves CRIU restore work; the post-match
guard prevents the doomed-promotion failure mode. There's no kill-switch
because there's no scenario where running them does harm.

`_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S` and `_FAST_FORWARD_SETTLE_S` are
constants in `integrations/agents/terminus.py`; tweak them if a
particular trace set needs different timing, but the defaults are what
the spec_friendly and compcert benchmarks were tuned against.

---

## Telemetry

Both mechanisms emit per-event telemetry, aggregated per-sandbox in the
report's "Replay Cadence Handling" section.

### Events

| Event | Attributes | When it fires |
|---|---|---|
| `terminus.fast_forward.skip` | `sandbox_id`, `command_count`, `intended_sleep_ms`, `saved_ms` | Each empty-wait turn that the fast-forward probe successfully skipped. |
| `terminus.guard.pre_fork_drain` | `sandbox_id`, `busy_cmd`, `drain_wait_ms`, `drain_resolved` | Each time the pre-fork active-busy guard fired (pane was busy at draft+oracle race finish). `drain_resolved=1` if the pane drained before `_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S`. |
| `terminus.guard.post_match_drain` | `sandbox_id`, `busy_cmd`, `drain_wait_ms`, `drain_resolved` | Each time the post-match guard fired (rare race window). |

### Metrics

| Metric | Unit | Notes |
|---|---|---|
| `benchmark.terminus.fast_forward_saved_ms` | ms | Per-skip wall-time saved (`intended_sleep_ms - settle window`). |
| `benchmark.terminus.guard_drain_ms` | ms | Per-drain wait time. Includes successful drains and timeouts. |

### Report section

"Replay Cadence Handling" appears between Restore Analysis and Resource
Usage when any of the three event types fired. It shows:

- **Run-wide totals** by mechanism: events, total wall-time moved,
  average per event, and timeouts (drain guards only).
- **Per-sandbox breakdown** with the same columns. Use this to spot
  build-heavy sandboxes that dominate either side: a compcert sandbox
  typically accounts for the bulk of both pre-fork drain time (slow
  replay) and fast-forward savings (fast replay) within the same run.

CSV mirror: `replay_cadence_per_sandbox.csv` in the report directory.

---

## How to read the numbers

- `fast_forward_saved_ms` total close to `fast_forward_intended_sleep_ms`
  total → fast-forward is doing what it should. The settle window is
  the only overhead.
- High `pre_fork_drain_count` clustered on a few sandboxes → those
  sandboxes have build-heavy traces; expect them to dominate end-to-end
  wall time too.
- `pre_fork_drain_timeouts > 0` → a foreground command exceeded the 30
  min drain budget. Investigate: either a genuine hang or a build
  legitimately longer than the timeout. The agent recovered by queuing
  keystrokes behind the still-running command (same as before guards
  existed), so the run isn't broken — but those tasks may verify with
  partial state.
- `post_match_drain_count` should be much smaller than
  `pre_fork_drain_count`: pre-fork is the primary defense; post-match
  only catches the narrow race where the active became busy *after* the
  pre-fork probe ran. If post-match is firing often, the
  draft+oracle round-trip is long enough that build state shifted under
  the guard — worth a closer look.
