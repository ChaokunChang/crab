# Crab

English | [简体中文](README.zh-CN.md)

Crab gives AI-agent sandboxes recoverable savepoints. It observes agent turns,
decides whether process state, filesystem state, or both need to be saved, and
coordinates checkpoint/restore with the host runtime.

Crab is currently a v0 version for technical preview. The v0 backend uses:

- `runc` to run sandboxes;
- CRIU to checkpoint and restore process state;
- ZFS to snapshot and roll back the sandbox root filesystem;
- an eBPF host inspector to detect recovery-relevant changes.

Crab is the control and coordination layer. Additional checkpoint substrates
are planned, but v0 deliberately ships one complete, tested backend instead of
silent best-effort fallbacks.

## See it work

On an Ubuntu 24.04/26.04 x86-64 host:

```bash
git clone https://github.com/open-agent-infra/crab.git
cd crab
sudo ./scripts/install-ubuntu.sh
sudo ./scripts/smoke-rollback.sh
```

The installer creates a dedicated sparse-file ZFS pool named `crab`; it never
selects an arbitrary existing pool or repartitions a disk. The smoke test then:

1. launches an Ubuntu sandbox;
2. creates a file and a long-running process;
3. takes a full checkpoint;
4. mutates the file and kills the process;
5. restores the checkpoint and verifies both states were recovered.

See [Installation](docs/installation.md) for installer options and
[Getting started](docs/getting-started.md) for the same flow step by step.

## Manual checkpoint and restore

Start the daemon and create a sandbox:

```bash
sudo crab daemon start --config /etc/crab/config.yaml
SBX=$(sudo crab sandbox run --detach ubuntu:22.04)
```

Create some state and save it:

```bash
sudo crab sandbox exec "$SBX" -- sh -lc 'echo before > /root/state.txt'
CKPT=$(sudo crab checkpoint create "$SBX")
sudo crab checkpoint ls "$SBX"
```

Mutate and roll back:

```bash
sudo crab sandbox exec "$SBX" -- sh -lc 'echo after > /root/state.txt'
sudo crab restore "$SBX" "$CKPT"
sudo crab sandbox exec "$SBX" -- cat /root/state.txt
# before
```

Clean up:

```bash
sudo crab sandbox rm "$SBX"
sudo crab daemon stop
```

## SDK

The SDK talks to the same long-running daemon as the CLI:

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        sandbox.commands.run("echo before > /root/state.txt")
        checkpoint = sandbox.checkpoint(label="before-change")
        sandbox.commands.run("echo after > /root/state.txt")
        sandbox.restore(checkpoint)
    finally:
        sandbox.kill()
```

Crab also exposes an `Agent` integration contract and built-in Claude Code and
iFlow profiles. See the [SDK guide](docs/sdk.md),
[bring-your-own-agent guide](docs/byo-agent.md), and
[iFlow trace replay example](docs/sdk-iflow-replay.md).

The installed smoke-test config disables sandbox networking and LLM
interception. Agent integrations that make in-sandbox LLM calls must use an
agent-oriented config; see [Configuration](docs/configuration-reference.md).

## What is included in a checkpoint?

The v0 full checkpoint covers process state and the sandbox's ZFS-backed root
filesystem. Host bind mounts are outside that snapshot boundary.

In particular, `Sandbox(work_dir="./repo")` and `crab sandbox run --work-dir`
mount a host directory at `/work`; restoring a Crab checkpoint does **not**
roll that host directory back. Clone or copy a repository into the sandbox
root filesystem when you want Crab to protect it.

Crab also cannot undo side effects outside the sandbox, such as GitHub pushes,
cloud API calls, payments, or writes to an external database.

## v0 scope

Supported:

- Ubuntu 24.04/26.04 on x86-64;
- a root-owned, single-user daemon;
- manual checkpoint inspection and restore through CLI or SDK;
- semantics-aware automatic checkpointing through the LLM interceptor;
- one host using the runc + CRIU + ZFS backend.

Not yet supported:

- macOS, Windows, rootless operation, or multi-user access;
- filesystem-only fallback when CRIU or ZFS is unavailable;
- daemon restart rehydration for existing sandboxes;
- transparent rollback of host bind mounts or external side effects;
- a stable public substrate plugin API.

## Documentation

The [documentation index](docs/README.md) separates user guides, operator
reference, architecture, agent integration, and research/benchmark notes.

The research design and evaluation are described in the paper:
[Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](https://arxiv.org/abs/2604.28138).

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
tests, documentation expectations, and change hygiene.

## Development

Install the Python package from the repository and run the dependency-light
SDK, checkpoint transport, image, and replay tests:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest -v \
  tests.test_remote_engine_checkpoint \
  tests.test_image_runtime \
  tests.test_sdk_sandbox \
  tests.test_iflow_trace_replay
```

Real-host tests additionally require the dependencies installed by
`scripts/install-ubuntu.sh`. The historical benchmark suite also has optional
SWE-bench dependencies. Benchmark datasets and recorded traces are not part of
the normal installation path.

## License

Crab is released under the [MIT License](LICENSE).
