<p align="center">
  <img src="assets/crab-logo.png" alt="Crab logo" width="180">
</p>

<h1 align="center">Crab</h1>

<p align="center"><strong>Give long-running AI agents real savepoints—not just chat history.</strong></p>

<p align="center">
  <a href="https://github.com/open-agent-infra/crab/actions/workflows/ci.yml"><img src="https://github.com/open-agent-infra/crab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://arxiv.org/abs/2604.28138"><img src="https://img.shields.io/badge/arXiv-2604.28138-b31b1b.svg" alt="arXiv: 2604.28138"></a>
</p>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

AI agents do more than exchange messages. They edit repositories, install
packages, start background services, compile code, and accumulate state across
hundreds of tool calls. When that workflow crashes—or the agent takes a bad
turn—replaying the conversation does not reconstruct the environment it was
working in.

Crab makes the entire agent sandbox recoverable. It observes each agent turn,
determines whether that turn changed filesystem state, process state, both, or
nothing recovery-relevant, and creates the smallest useful checkpoint at the
right time.

The result is a workflow you can resume, roll back, branch, or migrate without
starting the agent over.

## Why Crab?

Existing recovery approaches force an uncomfortable choice:

| Approach | What it preserves | What breaks |
| --- | --- | --- |
| Application/framework checkpoints (for example Claude Code or LangGraph persistence) | Conversation history and sometimes Git or file changes | Installed packages, background processes, in-memory services, and shell side effects are missing after recovery |
| Restart and replay | Eventually rebuilds the environment | Repeats every prior model turn and tool call, wasting time, tokens, and external capacity |
| Full container/VM checkpoint after every turn (for example Docker/runc or Firecracker-based sandboxes) | Complete execution state | Treats every turn as equally stateful and creates prohibitive checkpoint traffic under co-location |
| **Crab** | **Adaptive filesystem and process state at agent-turn boundaries** | **Preserves complete recovery points while skipping unnecessary checkpoint work** |

Crab does not replace Claude Code, Codex, LangGraph, Docker, or a cloud sandbox.
It adds the recovery layer between an agent workflow and its execution
substrate, so existing agents gain complete, efficient savepoints.

The underlying problem is an **agent–OS semantic gap**:

- the agent framework sees model turns and tool calls, but not their complete
  operating-system effects;
- the operating system sees processes and file activity, but does not know
  which agent turn they belong to or whether they matter for recovery.

Crab bridges the two layers. It correlates turn boundaries with OS-visible
effects, chooses no checkpoint, filesystem-only, process-only, or full state,
and overlaps checkpoint work with the time the agent is already waiting for
the next LLM response. At host scale, it also schedules checkpoint traffic
across co-located sandboxes instead of letting them overwhelm shared storage.

Crab does this without changing the agent's tool loop or teaching CRIU, ZFS,
or another checkpoint backend about agents.

## What does this change for an agent workflow?

| Without Crab | With Crab |
| --- | --- |
| A crash restarts a long task from the beginning | Restore the latest complete sandbox checkpoint |
| A bad command triggers a fragile sequence of cleanup commands | Roll the sandbox back to a known-good state |
| Every-turn full snapshots create heavy I/O | Stateless turns are skipped; stateful turns save only the required granularity |
| A rollout branch re-executes its shared prefix | Start new exploration from an intermediate sandbox state |
| Spot preemption discards in-flight work | Checkpoint and resume the sandbox on replacement capacity |

For individual developers, Crab provides CLI and SDK savepoints around coding
and shell agents. For agent platforms, it provides the policy and coordination
layer between agent semantics and the underlying checkpoint substrate.

## Results

In the [Crab paper](https://arxiv.org/abs/2604.28138), evaluated with Claude
Code, iFlow CLI, and SWE-agent on Terminal-Bench and SWE-Bench workloads, Crab:

- raised recovery correctness from **8% for chat-only recovery to 100%**;
- skipped up to **87% of checkpoint work** because most turns changed no
  recovery-relevant state;
- stayed within **1.9% of fault-free, checkpoint-free execution time**, even
  under dense sandbox co-location;
- reduced wall-clock time by up to **29%** and rollback tokens by **36%** when
  rollback was exposed directly to the agent as a tool;
- reduced redundant tokens in branched RL rollouts by **40.0–64.2%** through
  intermediate-state reuse.

## How it works

Crab combines three views that no single layer has on its own:

1. The **Coordinator** identifies agent-turn boundaries and uses LLM wait time
   as a window for asynchronous checkpointing.
2. The eBPF-based **Inspector** observes recovery-relevant process and
   filesystem effects and selects checkpoint granularity.
3. The host-scoped **C/R Engine** schedules, creates, tracks, and restores
   checkpoints across sandboxes using existing runtime and storage backends.

The v0 backend uses `runc` for sandbox lifecycle, CRIU for process state, ZFS
for filesystem snapshots, and an eBPF host inspector. Crab is the control and
coordination layer: its job is to decide **when** to checkpoint and **what
granularity** is required, rather than baking those decisions into a particular
C/R implementation.

## See it work

On an Ubuntu 24.04/26.04 x86-64 host:

```bash
git clone https://github.com/open-agent-infra/crab.git
cd crab
sudo ./scripts/install-ubuntu.sh
sudo ./scripts/smoke-rollback.sh
```

The installer creates a dedicated sparse-file ZFS pool named `crab`; it never
selects an arbitrary existing pool or repartitions a disk. The smoke demo needs
no model API key. It launches an Ubuntu sandbox, saves live filesystem and
process state, corrupts both, restores the checkpoint, and verifies that the
original state is back.

See [Installation](docs/installation.md) for installer options and
[Getting started](docs/getting-started.md) for the same flow step by step.

## Manual checkpoint and rollback

Start the daemon and create a sandbox:

```bash
sudo crab daemon start --config /etc/crab/config.yaml
SBX=$(sudo crab sandbox run --detach ubuntu:22.04)
```

Create state and save it:

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

Crab exposes an `Agent` integration contract and built-in Claude Code and iFlow
profiles. See the [SDK guide](docs/sdk.md),
[bring-your-own-agent guide](docs/byo-agent.md), and the
[real iFlow trace replay](docs/sdk-iflow-replay.md), which demonstrates
semantics-aware checkpoints without a model API key.

## Checkpoint boundary

A v0 full checkpoint covers process state and the sandbox's ZFS-backed root
filesystem. Host bind mounts are outside that snapshot boundary.

In particular, `Sandbox(work_dir="./repo")` and `crab sandbox run --work-dir`
mount a host directory at `/work`; restoring a Crab checkpoint does not roll
that host directory back. Clone or copy a repository into the sandbox root
filesystem when you want Crab to protect it.

Crab also cannot undo side effects outside the sandbox, such as GitHub pushes,
cloud API calls, payments, or writes to an external database.

## v0 technical preview

Today, Crab supports Ubuntu 24.04/26.04 on x86-64, a root-owned single-user
daemon, manual CLI/SDK restore, semantics-aware automatic checkpointing, and
the runc + CRIU + ZFS backend.

The next layer of product work includes additional checkpoint substrates,
rootless and multi-user operation, daemon restart rehydration, and a packaged
agent-facing rollback tool/skill. The current backend is deliberately complete
and testable rather than a silent best-effort fallback.

## Documentation

The [documentation index](docs/README.md) covers installation, CLI and SDK use,
agent integration, configuration, architecture, and telemetry. Research design
and evaluation details are in the
[paper](https://arxiv.org/abs/2604.28138).

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
testing, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the runtime
archives included with the replay example.

## Citation

If Crab is useful in your work, please cite the arXiv paper:

```bibtex
@misc{wu2026crabsemanticsawarecheckpointrestoreruntime,
  title={Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes},
  author={Tianyuan Wu and Chaokun Chang and Lunxi Cao and Wei Gao and Wei Wang},
  year={2026},
  eprint={2604.28138},
  archivePrefix={arXiv},
  primaryClass={cs.OS},
  url={https://arxiv.org/abs/2604.28138},
}
```

## Development

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
`scripts/install-ubuntu.sh`. Benchmark datasets and recorded traces are not
part of the normal installation path.

## License

Crab is released under the [MIT License](LICENSE). Bundled third-party
components retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
