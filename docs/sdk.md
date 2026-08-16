# Crab SDK

The Python SDK talks to the same long-running daemon as the `crab` CLI. The
daemon owns runc, CRIU, ZFS datasets, checkpoint storage, and host inspection;
the SDK process owns your application and agent control flow.

Install Crab and start the daemon before creating a `Sandbox`. For the manual
checkpoint examples below, the default config is sufficient:

```bash
sudo ./scripts/install-ubuntu.sh
sudo crab daemon start --config /etc/crab/config.yaml
```

Because the installed v0 daemon and socket are root-owned, run SDK examples as
root as well, or explicitly arrange a supported socket/permission model.

## Launch, checkpoint, and restore

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        sandbox.commands.run("echo before > /root/state.txt")
        checkpoint_id = sandbox.checkpoint()

        sandbox.commands.run("echo after > /root/state.txt")
        sandbox.restore(checkpoint_id)

        assert sandbox.files.read("/root/state.txt").strip() == "before"
    finally:
        sandbox.kill()
```

`Engine.connect()` uses `/run/crab/crab.sock` for a root daemon. Pass a socket
path explicitly, or set `CRAB_DAEMON_SOCKET`, when the daemon uses another
location.

## Sandbox API

Create a sandbox from an image:

```python
sandbox = Sandbox(
    image="ubuntu:22.04",
    name="my-sandbox",       # optional; generated when omitted
    env={"MODE": "test"},   # environment inherited by sandbox commands
    network=False,           # request a network namespace explicitly
    engine=engine,
)
```

The main operations are:

```python
result = sandbox.commands.run(
    ["sh", "-lc", "uname -a"],
    cwd="/root",
    timeout=30,
    check=True,
)

sandbox.files.write("/root/note.txt", "hello\n")
text = sandbox.files.read("/root/note.txt")
exists = sandbox.files.exists("/root/note.txt")

checkpoint_id = sandbox.checkpoint()
checkpoints = sandbox.checkpoints.list()
sandbox.restore(checkpoint_id)
sandbox.checkpoints.delete(checkpoint_id)

sandbox.pause()
sandbox.resume()
sandbox.kill()
```

`commands.run()` currently buffers output and does not allocate a PTY. File
helpers are intended for text-sized files; use sandbox commands for binary or
streaming transfers.

### Host work directories are not rolled back

`Sandbox(work_dir="./repo")` bind-mounts that host directory at `/work`.
Checkpoint restore does not revert it. This is useful when files must survive
a sandbox rollback, but it is the wrong choice when the repository itself must
be transactional.

For rollback of project files, place the checkout inside the sandbox's
ZFS-backed root filesystem, for example under `/root/project`.

## Inspect existing state

The SDK can reconnect to a sandbox that the same live daemon is already
tracking:

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox.connect("my-sandbox", engine=engine)
    print(sandbox.checkpoints.list())
```

The daemon does not currently rebuild its in-memory sandbox registry after a
restart, so this is not restart recovery.

## Agent integrations

Crab exposes a small `Agent` contract and two built-in profiles:

| Profile | Protocol | Execution model |
| --- | --- | --- |
| `ClaudeCodeAgent` | Anthropic | Installs and invokes Claude Code inside the sandbox |
| `IFlowAgent` | OpenAI-compatible | Uses the packaged iFlow runtime inside the sandbox |

Both built-ins require a sandbox network namespace and the LLM interceptor.
The safe default config disables those features because the no-key rollback
smoke test does not need them. Start an agent-oriented config instead, such as
the iFlow replay config:

```bash
sudo crab daemon start \
  --config examples/sdk/configs/iflow_replay_engine.runc.yaml
```

Then bind the agent to a networked sandbox and register its real or replay
upstream:

```python
from crab import Engine, Sandbox
from crab.agents_builtin.iflow import IFlowAgent

with Engine.connect() as engine:
    sandbox = Sandbox(
        image="crab-iflow-bench:workspace",
        network=True,
        engine=engine,
    )
    try:
        agent = IFlowAgent().bind(
            sandbox,
            llm_url="http://127.0.0.1:18080",
        )
        result = agent.run("Solve the task")
    finally:
        sandbox.kill()
```

For a complete no-API-key workflow, use the
[iFlow trace replay example](sdk-iflow-replay.md). To integrate another agent,
see [Bring your own agent](byo-agent.md).

## How LLM interception is routed

When an agent is bound with `llm_url=...`, Crab registers that upstream for
the sandbox and exposes the interceptor URL through provider-specific base URL
environment variables.

- In-sandbox agents are attributed through the sandbox network lease and
  source IP mapping.
- Host-driven agents must send `X-Agent-Sandbox-Id: <sandbox-id>` explicitly;
  setting `CRAB_SANDBOX_ID` alone does not modify arbitrary HTTP clients.

For example, a host-side Anthropic client can pass a default header:

```python
import anthropic

client = anthropic.Anthropic(
    base_url=agent.llm_base_url,
    default_headers={"X-Agent-Sandbox-Id": str(sandbox.sandbox_id)},
)
```

API keys remain the responsibility of the upstream client or agent. Avoid
putting them in Crab config files, telemetry, or command output.

## Templates

`DockerComposeTemplate` translates one Compose service into a runc sandbox and
can materialize Terminal-Bench task assets:

```python
from crab import Sandbox
from crab.templates import DockerComposeTemplate

template = DockerComposeTemplate(
    compose_file="/path/to/docker-compose.yaml",
    service_name="client",
    task_root="/path/to/task-root",
)
sandbox = Sandbox(template=template, engine=engine)
```

This is a translation layer, not a general Docker Compose runtime. Multi-
service behavior should be validated for each task.

## Current limitations

- `Sandbox.fork(count, lazy=False)` clones a running sandbox via
  checkpoint+restore: each fork is an independent, running sandbox sharing
  the parent's state at fork time (incremental chain sharing applies when
  available). `lazy=True` restores with CRIU lazy-pages for a faster
  return. Forks share the parent's `work_dir` host mount. Works both with
  a local in-process engine and against the daemon (`crab sandbox fork`
  from the CLI).
- `Sandbox.begin(label=None)` opens a snapshot-based transaction
  (`with sandbox.begin() as txn:` commits on clean exit, aborts on
  exception; `txn.exec/commit/abort`). Begin takes an adaptive base
  checkpoint and arms observation staging; abort restores the base and
  drops staged LLM responses (gated callers get a 409); commit delivers
  them and drops a freshly-taken base. Weak isolation: txn actions run in
  place. Auto-checkpoints are suppressed while a txn is open. Works both
  with a local in-process engine and against the daemon (`crab txn
  begin/commit/abort/status` from the CLI).
- `Sandbox.begin(isolation="fork")` opens a fork-backed transaction
  (strong isolation): begin forks the sandbox and `txn.exec` runs in the
  fork while the source keeps serving its pre-txn state; commit promotes
  the fork's whole filesystem *and process* state back onto the source's
  unchanged identity (`commit(force=True)` overrides the dirty-source
  gate and discards source-side writes made during the txn); abort just
  destroys the fork — the source is never restored.
- `Sandbox.changeset(since=None)` returns the changed rootfs paths
  (added/modified/removed/renamed) relative to a base checkpoint's
  filesystem snapshot; `since=None` diffs against the sandbox's fork
  point. Raw truth from the CoW backend (`zfs diff` / `btrfs send`).
  Works both locally and against the daemon (`crab sandbox changeset`).
- `Sandbox.merge(fork, policy="fail_fast", ignore_prefixes=None,
  merger=None)` three-way merges a fork's filesystem changes back into
  the source: fork-side changes apply where the source did not touch the
  same path since the fork point; conflicts resolve per policy
  (`fail_fast` aborts before any write, `prefer_fork`, `prefer_source`,
  `text_merge` runs a line-based diff3). Returns a `MergeReport`
  (applied/conflicted/skipped, `rolled_back`); apply failures raise
  `MergeError` carrying the report after a path-level rollback from the
  pre-merge snapshot. Works both locally and against the daemon
  (`crab sandbox merge`); custom `merger` hooks are local-only.
- `Sandbox.merge_processes(fork, strategy="auto", ...)` is the process
  half of consolidation (C4). `auto` probes this sandbox's live
  processes: with background processes running, the fork's journaled
  execs are **replayed** here verbatim and every command's outcome is
  diffed against the recorded returncode/stdout digest (deviations are
  counted; `stop_on_deviation=True` aborts at the first one; the fork
  stays alive); with none, the fork is **promoted** wholesale onto this
  sandbox's identity (PR-C4.2 — `policy`/`observations`/`lazy_pages`/
  `force` steer it). Returns a ProcessMergeReport. Works both locally
  and against the daemon (`crab sandbox merge-processes`).
- `Sandbox.consolidate_observations(fork, policy="append",
  summarizer=None)` adopts a fork's journal history into this sandbox's
  journal as `kind="observation"` records with provenance (fork id,
  origin seq/kind/txn): `append` copies every qualifying record,
  `dedupe` skips execs the source produced identically itself since the
  fork point, `none` copies nothing (combine with a local-only
  `summarizer` callable for a digest entry). Fork-txn commits adopt the
  fork's history automatically; `Sandbox.merge(fork,
  observations="append"|"dedupe")` opts a merge in and nests the
  resulting report under `MergeReport.observations`. Works both locally
  and against the daemon (`crab sandbox consolidate`).
- `Sandbox.actions(kind=None, limit=None)` reads the per-sandbox action
  journal: every exec attempt (argv, cwd, env, exit status, timing;
  stdout/stderr as size+sha256 only) plus lifecycle markers
  (launch/checkpoint/restore/fork/destroy) and adopted fork history
  (`kind="observation"`, C3). Journals are JSONL files under
  `{storage_root}/journal/` and record env values verbatim — treat them
  with the same care as checkpoint images. Works both locally and
  against the daemon (`crab sandbox actions`); disable recording with
  `EngineConfig(enable_action_journal=False)`.
- Daemon restart rehydration is not implemented.
- Exec output is buffered; streaming and PTY support are not implemented.
- `resources`, `timeout`, and `labels` constructor arguments are currently
  advisory metadata, not enforced resource limits or lifecycle policies.
- Only OpenAI-compatible and Anthropic LLM base-URL conventions are built in.
