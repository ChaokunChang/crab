# Lazy-restore safety contract

This document describes the runtime-side safety pin that protects the
`enable_lazy_restore` (Phase D) optimization from a `SIGBUS`-on-fault
class of failures. It is the resolution of an open follow-up flagged
in the original optimization PR.

## The bug we are protecting against

Phase D plumbs `runc restore --lazy-pages`. With the flag set, `runc
restore` returns once metadata + a small eager set of pages is mapped;
the rest of the restored process's memory is mapped but unpopulated.
The kernel resolves later page faults through `userfaultfd`, which
forwards the fault to a `criu lazy-pages` daemon. The daemon reads the
requested page from the on-disk CRIU image set and writes it back
through the `userfaultfd`. The restored process is **already running**
during this entire dance.

The image set for a fork in chain-sharing mode is **shared with the
source sandbox**. Phase B replaces per-fork `shutil.copytree` of every
chain ancestor's pre-dump dir with a relative symlink into the source's
runtime tree (`<source>/<ckpt>/pre_dump/`). The fork's leaf has its own
copy of the leaf bytes, but its `pre_dump/parent` symlink is preserved
and resolves into the source. CRIU follows that symlink chain when
serving page faults.

If the source's runtime tree is pruned out from under the daemon —
either:

- `LatestOnlyCheckpointManager` evicts a chain ancestor whose pages the
  fork still needs, or
- the source sandbox itself is destroyed (`destroy_sandbox_dataset`)
  before the fork's daemon finishes,

— the daemon's next read fails. The `userfaultfd` protocol's response
to "page source disappeared mid-fault" is a process-level fatal signal
on the restored process: **`SIGBUS`, not a clean restore failure**.
This is materially worse than a synchronous restore failure because:

1. The fork was already running. Its bash, agent loop, in-progress
   exec, every thread — all get the signal at an arbitrary userspace
   address, well after the harness moved on.
2. There is no way for the harness to clean up the in-progress turn,
   roll back state changes, or surface the failure to the agent loop
   gracefully. Restored containers that SIGBUS leave the spec-pair in
   limbo.
3. The signal is delivered asynchronously to whichever thread happens
   to touch the missing page first; reproduction is racy.

## The fix

Three load-bearing additions, all keeping their existing call patterns:

### 1. Runtime-side daemon registry

`RuncRuntime` maintains `_lazy_pages_daemons: dict[int,
_LazyPagesDaemonHandle]` indexed by daemon PID. The handle records the
sandbox/checkpoint identity plus the resolved on-disk `image_path` and
`work_path` the daemon depends on.

- `_register_lazy_pages_daemon` is called inside `_spawn_lazy_pages_daemon`
  immediately after the lazy-pages socket appears (i.e., the daemon is
  proven ready to serve faults; before this, no fork can fault).
- `_unregister_lazy_pages_daemon` is called inside
  `reap_lazy_pages_daemon`, on every exit path.

The registry is guarded by a dedicated `Lock` (`_lazy_pages_lock`); it
is read by `runtime_image_path_in_use` from arbitrary threads (the
storage retention background dispatcher).

### 2. Runtime predicate

`Runtime.runtime_image_path_in_use(path: Path) -> bool` is a contract
method on the abstract `Runtime` interface (default `False` for
runtimes without lazy-pages support, e.g. `InMemoryRuntime`).

The runc implementation:

- Snapshots the live daemons (`_live_lazy_pages_daemons`), pruning any
  whose PID is no longer alive — this matters because CRIU's
  `lazy-pages` exits *on its own* once all faults have been served, not
  via our explicit reap. Without lazy pruning of dead PIDs, retention
  would block forever waiting for a process that has already died.
- Resolves the input path (so a symlinked checkpoint dir is compared
  against the daemon's resolved `image_path`).
- Returns True when the input path is the daemon's image_path, an
  ancestor of it, or a descendant. All three matter:
  - **Equal** → exact image dir is being read.
  - **Ancestor** → e.g., the parent checkpoint dir (`<sandbox>/<ckpt>/`)
    is being pruned; `rmtree` would walk into the daemon's image dir.
  - **Descendant** → e.g., a nested pre-dump dir within the image dir
    is targeted; pruning it still breaks the daemon's reads.

### 3. Storage prune deferral

`LocalCheckpointManager.__init__` accepts an optional
`runtime_image_path_in_use: Callable[[Path], bool]` and exposes
`set_runtime_image_path_in_use` for late binding (the benchmark harness
constructs the storage manager before the runtime is fully wired).

`_delete_process_runtime_paths` consults the predicate before calling
`_remove_runtime_dir`. When the predicate returns True, the call is
skipped with a logged warning:

> Deferring runtime checkpoint dir prune; in use by an active runtime
> operation (lazy-pages daemon serving userfault faults from this image
> source). path=...

The next time retention runs (the next checkpoint completion triggers a
prune for `LatestOnlyCheckpointManager`), the daemon may have exited;
the predicate then returns False and the dir is reclaimed cleanly.

If the predicate raises (programming error, broken callback), the
exception is logged and the prune proceeds. The fail-open behavior is
deliberate: an indefinitely-deferred prune leaks storage, while a rare
SIGBUS is observable and recoverable. Both are bugs but the latter is
easier to detect.

## Wiring

Two construction sites consume the predicate:

- `agent_cr/system.py::build_default_system` — passes
  `runtime_impl.runtime_image_path_in_use` to `LocalCheckpointManager`
  when constructing the default storage. If the caller supplies their
  own checkpoint manager (e.g., a retention wrapper around a custom
  storage backend), the system attempts to late-bind via
  `setattr`-style discovery: any manager that exposes
  `set_runtime_image_path_in_use` is configured; others are left
  alone (this preserves the contract for the LocalCheckpointManager
  case while not failing on third-party managers).
- `benchmarks/real_host_scenario_base.py::RealHostScenarioHarness._build_system`
  — same callable is passed to `LocalCheckpointManager` directly at
  construction time. The harness always uses `LocalCheckpointManager`
  as the base and wraps it in a retention policy
  (`LatestOnlyCheckpointManager` / `KeepAllCheckpointManager`); the
  base manager is what actually owns `_delete_process_runtime_paths`,
  so the predicate-on-base wiring is sufficient.

## Tests

Three test groups in `tests/test_runc_runtime_checkpoint.py`:

- `test_runtime_image_path_in_use_default_returns_false` — empty
  registry → predicate is uniformly `False`. Establishes that the new
  code path is inert when D is not used.
- `test_runtime_image_path_in_use_after_register_and_reap` — spawns a
  real subprocess, registers it as a daemon handle, verifies the
  predicate fires on (a) the daemon's exact image_path, (b) an
  ancestor (the checkpoint dir whose `rmtree` would break it), and (c)
  is False on an unrelated sibling. Reaping the daemon clears the
  registry; the predicate goes back to `False`.
- `test_runtime_image_path_in_use_drops_dead_daemons` — registers a
  PID that has already exited and confirms the predicate prunes the
  stale entry rather than guarding indefinitely.

Two tests in `tests/test_storage.py`:

- `test_runtime_image_path_in_use_defers_prune` — installs a stub
  predicate that returns `True` on the first call and `False` on the
  second. The first prune leaves the directory on disk; the second
  reclaims it. This is the exact retention-pass semantics described
  above.
- `test_runtime_image_path_in_use_predicate_exception_does_not_wedge_retention`
  — installs a predicate that raises. The exception is swallowed and
  the prune proceeds.

A latent-binding test:

- `test_runtime_image_path_in_use_late_bind` — confirms a manager
  constructed without a predicate accepts one via the
  `set_runtime_image_path_in_use` setter.

## What this does NOT cover

- **Cross-host lazy restore.** If a fork's lazy-pages daemon is reading
  page bytes from a `criu page-server` on a different host, the
  predicate returns False (the local image_path doesn't exist) and the
  remote host's storage is not protected. We do not have cross-host
  lazy restore today; if/when that ships, the safety pin needs to be
  extended to either (a) the page-server side, or (b) a shared lock
  service that both ends consult.
- **Failed-spawn races.** If `criu lazy-pages` spawns and dies before
  the socket appears, `_spawn_lazy_pages_daemon` returns `None` without
  registering. The restore then proceeds with `--lazy-pages` and
  promptly fails; no daemon is alive, no SIGBUS scenario is possible,
  no protection is needed.
- **Daemon liveness check on the page-fault path.** The kernel may
  page-fault between our liveness check and the daemon actually being
  available. CRIU's daemon is robust to this — it serves faults
  synchronously after socket bind — so we treat socket appearance as
  the readiness signal.

## Why this is now a separate PR (not folded into the original)

The original PR (#26, merged) shipped the fork-prep optimizations and
explicitly flagged this issue as an open follow-up. Implementing the
fix needed:

- A new contract method on `Runtime` (touches the abstract base and
  every implementation).
- A new optional kwarg on `LocalCheckpointManager` plus a setter for
  late binding.
- Wiring at the two system-construction sites.
- Five new tests (three runtime, two storage).

This is a self-contained change with its own review surface; folding
it into a 2 600-line optimization PR would have buried both. Shipping
it now removes the implicit dependency between B and D and closes the
documented latent footgun.
