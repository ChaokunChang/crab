# Crab daemon

Crab runs as a long-lived host service — the equivalent of
`dockerd` — that owns the runc state, ZFS datasets, host inspector,
LLM interceptor + forwarder, and the sandbox network bridge. SDK
clients (and the `crab` CLI) connect to it over a Unix-domain
socket.

There is no in-process engine fallback. Starting two daemons (or
mixing a daemon with another in-process engine) on the same host is
not supported — they would race on the same paths, ports, and
network namespaces.

## Architecture

```
   ┌─────────────────────────────┐   HTTP/JSON over     ┌─────────────────────┐
   │  SDK process                │   Unix domain socket │  crabd (daemon)  │
   │                             │ ───────────────────▶ │                     │
   │  Sandbox(...) ── Engine     │                      │  Engine (in-proc)   │
   │    │  .runtime ─ Runtime    │ ◀─────────────────── │   ├─ RuncRuntime    │
   │    │            Proxy       │                      │   ├─ HostInspector  │
   │    └─ agent.bind(sbx)       │                      │   ├─ Interceptor    │
   │                             │                      │   ├─ Forwarder      │
   └─────────────────────────────┘                      │   └─ NetworkBridge  │
                                                        └─────────────────────┘
                       crab CLI (same protocol)
```

The daemon does not host the agent — your SDK process keeps running
the Python `Agent` code (so `IFlowAgent.execute(...)` and friends
keep their normal control flow). It just routes every operation that
touches OS state (`runc create`, `runc exec`, ZFS dataset
creation, host inspector registration, interceptor upstream
registration, network lease allocation) through the daemon's API.

## Lifecycle

The daemon is independent of any SDK process:

- `Sandbox(...)` asks the daemon to launch a sandbox.
- `sbx.kill()` asks the daemon to destroy that sandbox.
- The daemon stays up across SDK process exits.
- The daemon only stops on `crab daemon stop`, `SIGTERM`/`SIGINT`,
  or `POST /shutdown`.

## Quick start

Terminal 1 — start the daemon in the foreground with an engine config:

```bash
crab daemon start --foreground \
  --config examples/sdk/configs/iflow_replay_engine.runc.yaml
```

Detached:

```bash
crab daemon start \
  --config examples/sdk/configs/iflow_replay_engine.runc.yaml \
  --log-file /var/log/crabd.log
```

Terminal 2 — verify it's reachable:

```bash
crab daemon status
crab info
crab sandbox ls
```

Terminal 3 — run any SDK script. `Sandbox(...)` will connect to the
running daemon automatically:

```bash
sudo --preserve-env=PYTHONPATH PYTHONPATH=. \
  python3 examples/sdk/01_basic_sandbox.py
```

For the no-API-key iFlow workflow, start its replay router and pass the task
assets and trace to `examples/sdk/02_iflow_replay.py`; see
[sdk-iflow-replay.md](sdk-iflow-replay.md).

Stop the daemon when done:

```bash
crab daemon stop
```

## Socket location

By default the daemon listens on:

- `/run/crab/crab.sock` when started as root.
- `$XDG_RUNTIME_DIR/crab/crab.sock` (or `~/.cache/crab/crab.sock`)
  otherwise.

Override with `--socket /path/to/socket` on `crab daemon start` and
either `--socket` on subsequent CLI calls or the
`CRAB_DAEMON_SOCKET` environment variable for the SDK.

The socket file is created with permissions `0600`; only the user who
started the daemon can talk to it. There is no separate auth in v1.

## `crab` CLI reference

After `pip install -e .` (or `pip install .`), the `crab` and
`crabd` console scripts are on `$PATH`. The same subcommands are
available via `python -m crab.cli ...` and `python -m crab.daemon ...`
without installing.

```
crab [--socket PATH] [--timeout SECONDS] [--json] <subcommand>

  daemon start [--config FILE] [--foreground] [--log-file PATH] [--pid-file PATH]
  daemon stop  [--timeout SECONDS]
  daemon status

  info

  sandbox run IMAGE [--name NAME] [--work-dir PATH] [-e KEY=VALUE]... \
              [--network] [--detach | --rm] [-- CMD [ARG ...]]
  sandbox ls
  sandbox stop|pause|resume SANDBOX_ID
  sandbox rm SANDBOX_ID [SANDBOX_ID ...]
  sandbox exec SANDBOX_ID [--cwd PATH] [--user USER] [--timeout SECONDS] \
               -- CMD [ARG ...]

  checkpoint ls SANDBOX_ID
  checkpoint create SANDBOX_ID [--leave-running | --no-leave-running]
  checkpoint rm SANDBOX_ID CHECKPOINT_ID [--cascade]

  restore SANDBOX_ID CHECKPOINT_ID
```

`--json` switches `info`, `daemon status`, and `sandbox ls` / `checkpoint ls`
to raw JSON output (useful for scripting).

### Sandbox lifecycle verbs

- `run`     — create a sandbox from `IMAGE` (mirrors `docker run`).
  Without a trailing `-- CMD`, the sandbox is created and its id is
  printed. With `-- CMD`, the command is exec'd and its output is
  streamed back; `--rm` destroys the sandbox after the command exits.
  `--detach` is equivalent to "create and exit". Use `-e KEY=VALUE` to
  set env, `--work-dir` to bind a host directory at `/work`, and
  `--network` to allocate a sandbox network namespace.
- `stop`    — graceful stop via `runtime.stop()`. The runc container
  exits; the bundle and ZFS dataset stay so the sandbox can be
  restored from a checkpoint or destroyed later with `rm`.
- `pause`   — `runtime.pause()` (cgroup freeze). All processes inside
  the sandbox are suspended; resume with `resume`.
- `resume`  — `runtime.resume()`.
- `rm`      — destroy: `runtime.stop()` + `runtime.delete()` +
  network/upstream cleanup. The per-sandbox ZFS dataset is destroyed
  here too. Checkpoint manifests under `storage_root` are NOT pruned
  by `rm`; use `crab checkpoint rm` to clear those.
- `exec`    — run a command in an already-running sandbox; output is
  buffered and printed at the end (streaming is a follow-up).

### Checkpoint verbs and `restore`

Checkpoints are stored as `(sandbox_id, checkpoint_id)` pairs under
`storage_root`; checkpoint ids are unique per-sandbox, so the CLI
takes both. `create` captures process + filesystem in one shot; the
top-level `restore` verb rolls the sandbox back to the chosen
checkpoint (top-level because it acts on a sandbox using a checkpoint
— same shape as `git restore`).

```
crab checkpoint create $SBX               # → prints ckpt-<id>
crab checkpoint ls $SBX                   # → table of checkpoints
crab checkpoint rm $SBX $CKPT             # delete a single checkpoint
crab checkpoint rm $SBX $CKPT --cascade   # also delete descendant checkpoints

crab restore $SBX $CKPT                   # roll filesystem + process back
```

## SDK usage

```python
from crab import Engine, Sandbox

# Connects to the default socket; raises FileNotFoundError with a clear
# message if the daemon is not running.
engine = Engine.connect()

# Or point at a non-default socket explicitly:
engine = Engine.connect("/tmp/crab-dev.sock")

with engine:
    sbx = Sandbox(image="ubuntu:22.04", engine=engine)
    ...
    sbx.kill()
```

`Sandbox(...)` without an explicit `engine=` argument calls
`get_default_engine()`, which connects to the default socket lazily on
first use.

## What `crab daemon stop` destroys

For every sandbox the daemon tracked this session, stop calls
`runtime.stop()` + `runtime.delete()` — which kills the runc container
and destroys the per-sandbox ZFS child dataset — and then shuts down
`engine.stop()`, which tears down:

- the host inspector subprocess,
- the LLM interceptor HTTP server,
- the LLM forwarder HTTP server,
- the sandbox network bridge and per-sandbox veth pairs,
- the daemon Unix socket file (and PID file, if configured).

**Not destroyed by stop**:

- The runc state root directory (just empty after deletes).
- The ZFS pool and its parent dataset (only *children* under
  `zfs_dataset_prefix` are destroyed; the prefix dataset and pool stay).
- Checkpoint manifests and bundle skeletons under `storage_root` /
  `runtime_root/bundles/<sid>/` for sandboxes that were deleted —
  `runtime.delete` removes the live runc + ZFS dataset but leaves the
  manifest store and bundle dir untouched.
- The image rootfs cache under `image_cache_root`.

If the daemon dies ungracefully (kill -9, OOM, crash), any sandboxes
it tracked in memory are NOT cleaned up; their runc records and ZFS
datasets persist on disk. Use `runc list` / `zfs list` to inspect, and
`runc delete` / `zfs destroy` to manually clean up — daemon state
rehydration is a follow-up.

## v1 scope and limitations

Routed through the daemon (CLI and SDK proxy):

- `Sandbox(...)` (launch via `runtime.launch`)
- `sbx.commands.run(...)` / `sbx.files.read|write|exists`
- `sbx.kill()`
- `agent.bind(sbx, ...)` (LLM upstream registration + host-inspector
  filter installation)
- Sandbox network leases
- Sandbox lifecycle: `stop`, `pause`, `resume`, `rm`, `exec`
- Checkpoints: `create`, `ls`, `rm`, `restore`

Not yet routed in v1 (will land in a follow-up):

- Streaming exec — `crab sandbox exec` buffers stdout/stderr and
  prints at the end. No PTY allocation either.
- Daemon state recovery on restart — a v1 restart starts with an
  empty sandbox registry. Existing runc/ZFS state on disk is
  inspectable via `runc list` / `zfs list` but is not re-tracked by
  the daemon until the operator either reuses the same sandbox names
  (which will fail with EEXIST) or cleans up.
- Multi-user / multi-host daemons — v1 is single-user Unix-socket
  only; no remote API, no per-user namespaces.
