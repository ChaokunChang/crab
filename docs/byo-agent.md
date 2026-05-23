# Bring Your Own Agent

This guide shows how to integrate a custom agent into Agent-CR through the
SDK. The minimal contract is intentionally small — you declare which LLM
protocol your agent speaks and provide `install()` + `execute()` methods.

For the broader SDK surface, see [sdk.md](sdk.md).

## The contract

```python
from agent_cr import Agent, TaskResult

class MyAgent(Agent):
    name = "my-agent"            # used for logs and registry lookup
    llm_protocol = "anthropic"   # "openai" or "anthropic"
    version = "1.0"              # optional, used for image caching keys

    def install(self, sbx):
        """Run once when the agent is bound to a sandbox.

        Use this to install your agent's CLI, copy assets, warm caches, etc.
        Strict failure semantics: raising here aborts agent attachment and
        propagates the exception to `agent.bind(sbx, ...)`.
        """

    def execute(self, sbx, task: str) -> TaskResult:
        """Execute a single task. Required.

        Runs in the caller's thread. For in-sandbox agents the body typically
        does one `sbx.commands.run(...)` driving the agent CLI inside the
        sandbox. For on-host agents the body runs its own loop.
        """

    def on_restore(self, sbx):
        """Optional. Called after a checkpoint restore."""

    def request_stop(self):
        """Optional. Cooperative cancellation hook."""
```

That's the whole contract.

## In-sandbox agents

This is the common case: the agent is installed in the sandbox and its CLI
self-drives the LLM/tool loop inside.

```python
import shlex
from agent_cr import Agent, TaskResult, register_agent


class CodexAgent(Agent):
    name = "codex"
    llm_protocol = "openai"

    def install(self, sbx):
        sbx.commands.run("pipx install codex-cli", check=True)

    def execute(self, sbx, task):
        result = sbx.commands.run(
            argv=["codex", "-p", task],
            env=self.command_env(),
            check=False,
        )
        return TaskResult(
            exit_code=result.returncode,
            output=result.stdout,
            extra={"stderr": result.stderr},
        )


register_agent("codex", CodexAgent)
```

User-side:

```python
from agent_cr import Sandbox

sbx = Sandbox(image="ubuntu:22.04")
agent = CodexAgent().bind(sbx, llm_url="https://api.openai.com")
print(agent.run("Fix the failing tests"))
```

`self.command_env()` supplies `OPENAI_BASE_URL` and `AGENT_CR_SANDBOX_ID` to
that one sandbox command, so the codex CLI's outbound LLM traffic is tagged
with this sandbox's id and forwarded to `https://api.openai.com`.

## On-host agents

For agents that orchestrate the sandbox from outside (think
`terminus`/`mini-swe`-style tmux drivers), the `execute()` body itself is the
LLM/tool loop and issues many small `sbx.commands.run(...)` calls.

```python
import anthropic
from agent_cr import Agent, TaskResult


class HostDrivenAgent(Agent):
    name = "host-driven-example"
    llm_protocol = "anthropic"

    def install(self, sbx):
        sbx.commands.run("apt-get update && apt-get install -y tmux", check=True)
        sbx.commands.run("tmux new-session -d -s work", check=True)

    def execute(self, sbx, task):
        # Agent.run has set ANTHROPIC_BASE_URL in this process while execute()
        # is running, so the anthropic SDK picks up the interceptor URL.
        # The interceptor tags the request with this sandbox's id and forwards
        # to the llm_url passed to bind().
        client = anthropic.Anthropic()

        history = [{"role": "user", "content": task}]
        for step in range(20):
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=history,
            )
            text = response.content[0].text
            if "TASK_DONE" in text:
                return TaskResult(exit_code=0, output=text)

            # The model emits a shell command; run it in the sandbox.
            cmd = self._extract_command(text)
            sandbox_result = sbx.commands.run(cmd, check=False)
            history.append({"role": "assistant", "content": text})
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"Exit code: {sandbox_result.returncode}\n"
                        f"Output:\n{sandbox_result.stdout[:2000]}"
                    ),
                }
            )

        return TaskResult(exit_code=1, output="step budget exhausted")

    def _extract_command(self, text):
        # Strip a fenced block, parse, return a single command. Skipped here
        # for brevity.
        ...
```

User-side is identical:

```python
sbx = Sandbox(image="ubuntu:22.04")
agent = HostDrivenAgent().bind(sbx, llm_url="https://api.anthropic.com")
agent.run("Set up a Postgres container and run the migration")
```

The user doesn't know or care whether the agent runs in-sandbox or on-host.

## Multi-task lifecycle

Both styles support multiple sequential tasks on the same sandbox. The
agent's in-sandbox state (sessions, history, files in `/work`) persists
between tasks because the sandbox is not torn down between `run()` calls.

```python
sbx = Sandbox(image="ubuntu:22.04")
agent = MyAgent().bind(sbx, llm_url=...)
agent.run("Set up the project")
agent.run("Now add a CHANGELOG entry")
agent.run("Bump the version")
```

## Registering and resolving

Three ways to point the SDK at your agent:

```python
# Instance — what you'd write in a quick experiment.
agent = MyAgent().bind(sbx)

# Registered name — best when you want CLI parity (e.g. `agentcr run --agent foo`).
register_agent("my-agent", MyAgent)
agent = resolve_agent("my-agent").bind(sbx)

# Import path — like Harbor's --agent-import-path. Useful for plugin layouts.
agent = resolve_agent("my_pkg.agents:MyAgent").bind(sbx)
```

## What the engine does for you

When `agent.install()` or `agent.execute()` is called:

1. The agent exposes the provider-specific interceptor URL as
   `agent.llm_base_url`.
2. It exports the right env vars (`ANTHROPIC_BASE_URL` /
   `OPENAI_BASE_URL` / `*_API_BASE`) while `agent.run()` invokes
   `agent.execute(...)` so vendor SDKs pick them up.
3. `agent.command_env(...)` returns those values for in-sandbox CLIs that
   need them in `sbx.commands.run(..., env=...)`.
4. It stamps every outbound LLM call with `X-Agent-Sandbox-Id` so the
   per-sandbox forwarder routes to the right real upstream — the same
   header-and-IP pattern the benchmark harness uses to scale to many
   parallel sandboxes through one interceptor.
5. It captures the request through the C/R scheduler — so checkpoints can
   be taken at semantically meaningful turn boundaries without you wiring
   anything.

You do not have to think about any of this.
