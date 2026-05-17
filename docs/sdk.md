# Agent-CR SDK

This document describes the user-facing SDK introduced in `agent_cr/` —
the E2B-style entry point for creating sandboxes, attaching agents, running
tasks, and using checkpoint/restore.

Operators who run the underlying engine on a host should also read
[architecture.md](architecture.md). Most users will not touch the engine
directly.

## Quick start

```python
from agent_cr import Sandbox
from agent_cr.agents_builtin.claude_code import ClaudeCodeAgent

sbx = Sandbox(
    image="ubuntu:22.04",
    work_dir="./repo",
)
agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")

result = agent.run("Fix the failing tests in /work/repo")
print(result.output, result.exit_code)

# Same sandbox; another task. Agent state and /work contents persist.
agent.run("Now bump the version to 1.2.0 and update CHANGELOG.md")

# Manual inspection between tasks.
sbx.commands.run("git diff").stdout

# Checkpoint, restore, kill.
ckpt = sbx.checkpoint(label="post-tests")
sbx.restore(ckpt)
sbx.kill()
```

The SDK ships with these built-in agent profiles (registered automatically
when `agent_cr` is imported):

| Name           | Protocol  | Notes                                          |
| -------------- | --------- | ---------------------------------------------- |
| `claude-code`  | anthropic | Installs `@anthropic-ai/claude-code` if needed |
| `iflow`        | openai    | Uses the existing cached iFlow runtime assets  |

More built-ins (`terminus`, `mini-swe`) will be added in follow-up PRs. The
harness-side versions of those agents are unchanged and continue to work via
`integrations/agents/...`.

## Concepts

### Sandbox

A `Sandbox` is a long-lived isolated environment with a runtime, a working
directory, image/template, env, resources, and labels. Agents are attached
after the sandbox exists. Sandbox lifetime is **decoupled** from task lifetime
— you can run many `agent.run(...)` calls, poke around manually with
`commands.run(...)`, and checkpoint at any point.

```python
sbx = Sandbox(
    image="ubuntu:22.04",    # any docker tag, or a Dockerfile path
    template=None,           # optional template such as DockerComposeTemplate
    work_dir="./repo",       # bound to /work inside the sandbox
    env={"ANTHROPIC_API_KEY": "..."},
    resources={"cpus": 4, "memory": "8Gi"},
    labels={"exp": "fizz-42"},
)
```

Sub-namespaces:

- `sbx.commands.run(cmd, env=..., cwd=..., timeout=..., check=False)` — exec
  inside the sandbox. Accepts a shell string or `argv=[...]`.
- `sbx.files.read(path)` / `sbx.files.write(path, content)` / `sbx.files.exists(path)`.
- `sbx.checkpoints.list()` / `sbx.checkpoints.delete(id, cascade=False)`.

Lifecycle:

- `sbx.pause()` / `sbx.resume()`
- `sbx.checkpoint(label=...)` / `sbx.restore(ckpt_id)`
- `sbx.get_host(port)` — host-reachable URL for a port exposed inside the
  sandbox.
- `sbx.kill()` or use as a context manager.

### Task

`agent.run(task)` is synchronous. It blocks until the agent invocation
finishes and returns a `TaskResult` with `exit_code`, `output`, and free-form
`extra` data the agent fills in.

```python
result = agent.run("Refactor utils.py", timeout=600)
print(result.exit_code, result.output, result.extra)
```

Use `agent.run_async(task)` only when you intentionally want a background
host-side task handle:

```python
task = agent.run_async("Refactor utils.py")
print(task.done())
result = task.wait(timeout=600)
print(result.exit_code, result.output, result.extra)
```

### Agent

An agent is a profile that describes how to install and invoke a coding
assistant inside the sandbox. The contract is small:

```python
from agent_cr import Agent, TaskResult

class MyAgent(Agent):
    name = "my-agent"
    llm_protocol = "anthropic"   # "openai" | "anthropic"

    def install(self, sbx):                              # optional
        sbx.commands.run("pipx install my-agent", check=True)

    def execute(self, sbx, task):                        # required
        result = sbx.commands.run(f"my-agent -p {task!r}", check=True)
        return TaskResult(exit_code=result.returncode, output=result.stdout)
```

See [byo-agent.md](byo-agent.md) for a deeper guide including the on-host
agent pattern (where `execute()` drives the sandbox via a host-side loop).

Bind the profile to a sandbox:

```python
agent = MyAgent().bind(sbx, llm_url="https://api.openai.com")
agent.run("Solve the task")
```

### Templates

Templates describe a prepared sandbox shape without exposing engine-level
runc/ZFS details on `Sandbox(...)`. For compose-backed task images:

```python
from agent_cr import Sandbox
from agent_cr.agents_builtin.iflow import IFlowAgent
from agent_cr.templates import DockerComposeTemplate

template = DockerComposeTemplate(
    compose_file="/path/to/docker-compose.yaml",
    service_name="client",
    task_root="/path/to/task-root",
)

sbx = Sandbox(template=template)
agent = IFlowAgent().bind(sbx, llm_url="http://replay-or-llm")
agent.run("Solve the task")
```

The template reuses the same compose-to-runc translator as the benchmark
harness, including `/tests/run-tests.sh` materialization when `task_root` is
provided.

### Engine

The Engine is the runtime manager that owns the underlying runc/docker
runtime, the checkpoint store, and the LLM interceptor. Most users do not
touch it directly — `Sandbox(...)` lazily starts an in-process engine via
`get_default_engine()`.

Operators who deploy Agent-CR on a host will eventually start the engine as
a daemon and have `Sandbox(...)` connect to it. That mode is reserved
(`Engine.connect()` raises today) and will land in a follow-up PR. The SDK
shape is forward-compatible.

```python
from agent_cr import Engine, EngineConfig

with Engine.start(EngineConfig(runtime="runc")) as engine:
    sbx = Sandbox(engine=engine)
    agent = ClaudeCodeAgent().bind(sbx, llm_url="...")
    ...

with Engine.start("examples/sdk/configs/iflow_replay_engine.runc.yaml") as engine:
    sbx = Sandbox(engine=engine)
    agent = IFlowAgent().bind(sbx, llm_url="http://127.0.0.1:18080")
    ...
```

`EngineConfig` knobs:

| Field                  | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `runtime`              | `"docker"` (in-memory, for tests) or `"runc"` (production) |
| `storage_root`         | Checkpoint storage. Defaults to a tempdir.                 |
| `interceptor_host/port`| Where the LLM interceptor binds. `0` picks a free port.    |
| `enable_interceptor`   | Disable when no agents talk to an LLM.                     |
| `agent_worker_threads` | Max concurrent `agent.run()` invocations.                  |
| `telemetry_config`     | Optional JSONL telemetry sink for engine/runtime events.   |
| `log_file/log_level`   | Optional SDK engine log file configuration.                |

Telemetry, scheduler policy, and retention are sysadmin-side concerns and
are not exposed as user-facing options. Operators tune them in
`SchedulerConfig` / `TelemetryConfig` / `StorageConfig` passed to
`EngineConfig` if needed. `EngineConfig.from_file(path)` and
`Engine.start(path)` accept an engine-only YAML file for the in-process engine
until daemon config lands.

## LLM interception

The Engine wires two services that sit between an agent and its real LLM
upstream — the **same architecture used by the benchmark harness**:

```
sandbox / on-host agent
        │  LLM call
        ▼
AgentCRRequestInterceptorServer   ← unchanged from harness; gates requests
        │                            for semantic-aware checkpoint/restore
        ▼  single upstream URL
SdkLLMForwarder                   ← reads X-Agent-Sandbox-Id, dispatches
        │                            to the registered per-sandbox URL
        ▼
real LLM (api.anthropic.com / api.openai.com / ...)
```

The interceptor sees one upstream URL (the forwarder). The forwarder reads
`X-Agent-Sandbox-Id` (set by the interceptor's resolver) and forwards raw
bytes to the per-sandbox upstream URL the SDK registered when the sandbox
was bound to an agent. The interceptor itself is the same class the harness uses;
the SDK adds the forwarder, the harness uses the benchmark router for the
same role.

When you pass `llm_url=...` to `agent.bind(sbx, ...)`, the engine:

1. Registers the upstream URL under this sandbox in the forwarder.
2. Sets `ANTHROPIC_BASE_URL` (or `OPENAI_BASE_URL`, depending on the
   agent's `llm_protocol`) inside the sandbox env to the interceptor URL.
3. Sets the same env vars in the host-side `agent.run()` thread context so
   on-host agents using vendor SDKs (anthropic, openai) pick up the
   interceptor URL with zero code changes.
4. Stamps every outbound LLM call with `X-Agent-Sandbox-Id` so the
   forwarder can route to the right upstream — sandbox identity flows
   through whether the agent runs in-sandbox (interceptor IP-resolves) or
   on-host (interceptor sees an explicit header).

You do not need to write any LLM-related plumbing in your agent code. If
your agent SDK does not read env vars, construct the client explicitly:

```python
client = anthropic.Anthropic(base_url=sbx.llm_base_url)
```

## Running the examples

The repo has no `setup.py`, so run examples from the repo root with
`PYTHONPATH` pointing at it:

```bash
cd /root/workspace/acr-deploy/agent-cr
PYTHONPATH=. python3 examples/sdk/02_byo_agent.py
```

The `02_byo_agent.py` example runs against the in-memory runtime and
exercises the full SDK shape (install, run, follow-up task). The
`05_iflow_runc.py` example runs a real `runc` sandbox, invokes iFlow through
the SDK, and verifies checkpoint/restore. The
`06_iflow_replay_dataset_runc.py` example runs the first two iFlow replay
dataset rows as two concurrent runc sandboxes and verifies each task with
`/tests/run-tests.sh`; it expects the existing benchmark LLM router to be
running as an external replay service. See
[sdk-iflow-replay.md](sdk-iflow-replay.md).

## Known limitations (first cut)

- `Engine.connect()` is reserved; daemon mode lands in a follow-up.
- `Sandbox.fork()` is reserved; the underlying CRIU+ZFS machinery exists
  but its integration with the SDK launch path needs follow-up wiring.
- Bare-image launch via `Sandbox(image="ubuntu:22.04")` now prepares an OCI
  bundle/rootfs and runs through the real `runc` backend when the engine is
  configured with `runtime="runc"`.
- Only `openai` and `anthropic` LLM protocols are supported. Adding more is
  a matter of registering env-var lists in `agent_cr.agent.llm_env_vars_for`.
- Built-in terminus/mini-swe profiles are not yet available in the SDK. The
  harness path for those agents is unchanged.
