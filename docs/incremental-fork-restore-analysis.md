# Detailed analysis: Incremental fork/restore optimizations on terminus spec scenario

This document explains the 6-variant benchmark results for the fork-prep
optimizations (B = chain-ancestor sharing, A = background pre-fork,
D = lazy-pages restore) introduced in PR #26. It answers four questions:

1. Where do the wall-clock differences come from per variant?
2. Which optimizations actually save time on the agent's critical path?
3. Why does prefork (A) regress wall-clock even with the throttle?
4. When should each optimization be enabled?

All numbers are from the 8-task `terminus_replay_spec_friendly.incremental_demo`
subset, 8 sandboxes, ZFS-backed, host with 32 CPUs / 200G zpool. Configs:
`benchmarks/examples/terminus/terminus.spec.auto.incremental_demo.*.yaml`.
Single-sandbox confirmation (rstan-to-pystan only) used to isolate
contention vs. intrinsic cost.

## 1. Headline summary

| variant | wall-clock | mean fr_ms | nz_p99 | nz_max | accept_rate | total saved |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2 268 s | 146 ms | 3 781 ms | 7 013 ms | 6.2 % | 444 s |
| **chain_sharing (B)** | **2 193 s (−3.3 %)** | **88 ms** | 2 084 ms | 2 474 ms | 6.2 % | 445 s |
| lazy (D) | 2 235 s (−1.5 %) | 118 ms | 2 930 ms | 3 481 ms | 6.2 % | 443 s |
| b_plus_d | 2 275 s (+0.3 %) | 120 ms | 2 838 ms | 3 845 ms | 6.2 % | 443 s |
| prefork (A, throttled) | 2 451 s (+8.1 %) | 44 ms | 1 140 ms | 1 837 ms | 5.3 % | 368 s |
| **all_opts** (B+A+D, throttled) | 2 398 s (+5.7 %) | **41 ms** | **812 ms** | **1 099 ms** | 4.8 % | 309 s |

`fr_ms` = `benchmark.spec.fork_restore_ms` per spec turn. `nz_*` percentiles
exclude the (majority) of turns where the oracle won the race and `fr_ms` is 0.

## 2. Where do the wall-clock differences come from?

### Decomposition of task wall-clock

For every variant, each task's `task_completion_ms` decomposes as:

```
task_completion_ms = spec_exec_sum_ms + fork_restore_sum_ms + non_spec_time_ms
```

where `spec_exec_sum_ms` is the time spent inside speculative-execution
turns (oracle and draft commands), `fork_restore_sum_ms` is the
agent-loop wait for fork prep, and `non_spec_time_ms` covers regular
oracle-only commands, agent setup, and verification.

For rstan-to-pystan (the most fork-active task, contributing 50 % of
total saved time):

| variant | task_completion (s) | spec_exec (s) | fork_restore (s) | non_spec (s) |
|---|---:|---:|---:|---:|
| baseline | 1 864 | 552 | 16 | **1 296** |
| chain_sharing | 1 872 | 559 | 10 | 1 303 |
| lazy | 1 878 | 550 | 13 | 1 315 |
| b_plus_d | 1 881 | 554 | 9 | 1 318 |
| **prefork** | **2 201** | 581 | 5 | **1 615 (+319)** |
| **all_opts** | **2 112** | 495 | 4 | **1 612 (+316)** |

The +337 s prefork regression on rstan is **almost entirely in `non_spec_time`**
(+319 s). spec_exec is roughly unchanged. fork_restore actually drops by 11 s.

`non_spec_time` is the active sandbox's regular work — non-speculative
`runc exec` calls, R compilation, library installs, file I/O. **Background
prefork activity (clone + restore + daemon spawn + dataset destroy) is
saturating the same disk and CPU the active sandbox is using.**

### Confirmation: single-sandbox smoke

To rule out cross-sandbox contention as the cause, we ran rstan-to-pystan
in isolation with `sandboxes: 1` (configs:
`terminus.spec.auto.incremental_demo.rstan_smoke.{baseline,prefork}.yaml`):

| metric | smoke baseline | smoke prefork | Δ |
|---|---:|---:|---:|
| wall-clock | 1 850 s | 2 172 s | **+322 s (+17.4 %)** |
| spec_exec_sum | 555 s | 558 s | +3 s (noise) |
| fork_restore wait | 8.2 s | 3.4 s | −4.8 s saved |
| accepts | 7 | 7 | same |
| saved | 233 s | 234 s | same |
| preforks attempted | 0 | 16 (8 useful + 8 wasted) | – |

**The +322 s regression survives single-sandbox isolation.** Prefork's cost
is intrinsic to running clone/restore I/O on the same host as the active
workload, not to cross-sandbox queueing.

## 3. Why does each optimization succeed or fail?

### B (chain_sharing) — the only pure win

What B replaces: a `shutil.copytree` of every chain ancestor's CRIU
pre-dump dir into the fork's runtime tree (the leaf's `parent` symlink
is otherwise dereferenced by copytree, inlining ~thousands of small
`pages-N.img` / `pagemap-N.img` files per ancestor). What B does instead:
one `os.symlink` per ancestor + a chain pin in the storage layer so
retention can't prune the source.

Measured impact on the 8-task subset: 9 forks shared, average 2.2
ancestors per fork, ~61 KB of file bytes avoided. **The win isn't bytes
— it's syscalls.** Each ancestor pre-dump dir contains ~50–200 small
files; copytree means thousands of `mkdir`/`open`/`copy_file_range`/`close`
calls. The 13-second reduction in `fork_restore_sum` (33.2 s → 20.1 s
across all sandboxes) maps cleanly onto 9 shared forks × ~1.5 s saved
per shared fork.

B touches no kernel, no daemon, no cross-process state. It runs on the
fork-prep critical path only when a fork is created and contributes
zero overhead at any other time. **−3.3 % wall-clock** + **−40 % mean
fork_restore_ms** with **zero correctness risk**.

### D (lazy-pages) — modest pure win

D plumbs `runc restore --lazy-pages` and spawns a `criu lazy-pages` daemon
ahead of restore. The restored process's pages are mapped lazily via
userfaultfd; the kernel pulls in pages on demand from the daemon, which
reads them from the on-disk image.

Measured per-restore subprocess time: median 164 ms (baseline) vs 165 ms
(lazy) — **statistically identical**. So D doesn't make individual
restores faster. What it does do is make page faults *concurrent with
process execution* rather than blocking — the agent's draft `runc exec`
can start running immediately after restore returns, before all pages
are loaded. For short-lived discarded forks, pages that are never
touched are never read.

Measured impact: −19 % mean `fork_restore_ms` (146 → 118), −1.5 % wall
clock. The daemon-spawn + socket-poll overhead is < 50 ms per restore
(within the median's noise floor). Accept rate, saved time, and per-task
outcomes are identical to baseline. **No measurable downside**.

### A (prefork) — saves agent-loop wait, costs much more elsewhere

What A does: `_SpeculativeSandboxController` warms a fork after each
successful checkpoint, in a dedicated executor thread, so `ensure_fork()`
returns from cache instead of running clone+restore on the agent's
critical path.

The cache hits are real and substantial. On rstan-to-pystan in the
8-sandbox run, prefork shifted the fork-creation distribution from
12c+1r (baseline) to 6c+7r (prefork) and 7c+5r (all_opts). On the
single-sandbox smoke, the shift was 12c+1r → 5c+8r. **Half of all
fork-using turns are now cache hits.**

But the cost surfaces in three places:

1. **I/O contention with active sandbox** (the largest cost, +319 s on
   rstan). The prefork worker continuously runs `zfs clone`, copies
   CRIU images, spawns CRIU daemons, and destroys datasets. Each of
   these competes with the active sandbox's regular `runc exec` calls
   for disk bandwidth. On the smoke run with only 1 sandbox, this still
   produced +322 s — confirming the contention is host-resource-level,
   not cross-sandbox-queue-level.

2. **Lost speculation savings on mcmc-sampling-stan** (−77 s):
   - baseline: 5 finalized forks, 2 accepted, 79 s saved
   - prefork: 1 finalized fork, 0 accepted, 0 s saved

   By the time the agent calls `ensure_fork()`, the cached prefork's
   base checkpoint is often *older* than the latest checkpoint (active
   sandbox advanced state in the interim). `ensure_fork()` correctly
   discards the stale cache and falls through to synchronous clone, but
   the synchronous fork lands on a different state-snapshot than the
   baseline did, and the speculation outcomes diverge.

3. **Throttle ineffectiveness on the steady-state cost.** The
   `prefork_max_concurrent_global=2` cap eliminates outliers (max
   `fork_restore_ms` 23.4 s → 1.8 s) but only saves 39 s wall-clock per
   run. Continuous low-grade I/O from 1 prefork-in-flight is still
   enough to slow active commands.

Net result on the 8-task run: **agents wait 24 s less for forks, but
the active workload runs ~200 s slower and 77 s of speculation savings
are lost.** Net wall-clock: **+183 s (8.1 %)**. Even per-turn fork
latency wins (best p99 of 1 140 ms vs baseline 3 781 ms) don't make up
for the steady-state I/O cost.

### B + A + D combined — best percentiles, inherits A's cost

Combining all three gives the best `fork_restore_ms` percentiles
(p99 = 812 ms vs 3 781 ms baseline, an 4.7× reduction) because A's
cache hits compound with B's chain-sharing on the cache-miss path and
D's lazy mode further trims restore time. But the bulk of A's
wall-clock cost remains: **+130 s vs baseline (5.7 %)**, attributable
almost entirely to A.

The combination is the right choice when you care about per-turn
fork latency *jitter* more than aggregate wall-clock — for example,
interactive workloads or low-`acceptance_rate` runs where every fork
that wins the race is a real save.

## 4. Are the results expected?

| Variant | Predicted | Actual | Match? |
|---|---|---|---|
| B | "Pure win, replaces copytree with symlink" | −40 % mean, −3.3 % wall, 0 cost | ✓ |
| D | "Modest win, lazy mode trims restore tail" | −19 % mean, −1.5 % wall, 0 cost | ✓ |
| A | "Cache hits cut p99; throttle prevents starvation" | p99 −58 %, but +8.1 % wall-clock | **partial** |
| All | "Wins compound on percentiles, A's cost inherited" | best p99/p95, +5.7 % wall | ✓ |

The A surprise: the throttle DOES prevent the catastrophic 23 s outliers,
but background prefork I/O contention is a steady-state problem, not an
outlier problem. Capping concurrent prefork to 2 reduces peak burst but
doesn't reduce *average* I/O pressure: with 8 sandboxes each completing
checkpoints every ~10–30 seconds, the throttle is rarely the limiting
factor — most of the time only 1 prefork is in flight, and that's
enough to slow active sandboxes' commands.

## 5. When to enable which

- **B (chain_sharing)** — recommended **default-on** for any run with
  `incremental_process_enabled=true`. Faster forks, faster end-to-end,
  zero observed cost. The chain pin also makes the existing retention
  policy correct in the presence of forks (would otherwise allow
  ancestor pruning under multi-fork load — currently latent).

- **D (lazy_restore)** — recommended **default-on** for spec scenarios.
  Modest pure latency win. Only constraint is platform: needs runc 1.3+
  with `--lazy-pages`, CRIU 4.x, and userfaultfd kernel support
  (`/proc/sys/vm/unprivileged_userfaultfd=1` or root + `CAP_SYS_PTRACE`).
  Falls back gracefully when daemon spawn fails (logs warning, restore
  fails clearly rather than silently).

- **A (background_prefork)** — keep **opt-in**. Useful when:
  - Per-turn latency jitter matters more than aggregate wall-clock
    (interactive use, demos, latency-sensitive measurements)
  - Hardware has I/O headroom: separate disk for ZFS dataset and
    container working dir, or NVMe with enough bandwidth that 1
    concurrent prefork doesn't saturate
  - High-acceptance-rate workloads where every cache-hit shaves 100s of ms
    off the spec turn that would otherwise have been wasted on copy

  Skip A when:
  - The workload is I/O-bound on its own (rstan-to-pystan: R compilation,
    package installs, dpkg)
  - Wall-clock throughput is the optimization target
  - Acceptance rate is low (most tasks here: 0 % accepts mean every
    prefork is overhead with no compensating speculation gain)

- **All-opts** — only when latency percentiles are the success metric
  AND you accept the wall-clock cost. The combined run is the best
  configuration for benchmark publication of "fork prep latency under
  speculative execution"; it is **not** the best configuration for
  "minimize task completion time".

## 6. Open follow-ups

- **A is fundamentally limited by host I/O headroom.** Plausible fixes:
  separate I/O cgroup for the prefork worker (give it a low blkio.weight),
  or migrate prefork to a separate NUMA node / dedicated disk. Both are
  significant infra changes; not in scope for this PR.
- **A wastes ~50 % of preforks** (8 useful / 16 attempted on the smoke
  run). Could be reduced by predicting which checkpoints will produce a
  fork that wins the race — e.g., skip prefork when oracle is consistently
  faster for this task.
- **B's chain pin is also load-bearing for D's safety.** If retention
  pruned a chain ancestor while a lazy-pages daemon was still serving
  that ancestor's pages, the fork would SIGBUS. We added the pin for B
  but it's a prerequisite for D too. Documenting this here so the
  reasoning isn't lost.
