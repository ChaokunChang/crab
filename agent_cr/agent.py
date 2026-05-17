"""Agent contract for the user-facing SDK.

This module defines the abstraction users implement when bringing their own
agent into an Agent-CR sandbox. It intentionally has a narrower contract than
the internal `integrations.agents.BaseAgent` (which is one-shot, task-baked-
into-the-bundle, and tied to the benchmark harness flow). The `Agent` here is
multi-task per sandbox: `install()` runs once when the agent is bound to a
sandbox, and `run(task)` runs each task synchronously.

User-facing attachment is deliberately separate from sandbox construction:

    sbx = Sandbox(image="ubuntu:22.04")
    agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")
    result = agent.run("Fix the failing tests")

`run()` blocks until the agent invocation finishes and returns a `TaskResult`.
Callers that explicitly want background execution can use `run_async()`.
"""
from __future__ import annotations

import importlib
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Literal

if TYPE_CHECKING:
    from .sandbox import Sandbox


LLMProtocol = Literal["openai", "anthropic"]

# Environment variables the engine injects into both the sandbox process and
# the host-side agent run context. The agent does not need to know about these
# explicitly — it just reads its protocol's standard env vars and gets the
# interceptor URL.
_OPENAI_ENV_VARS: tuple[str, ...] = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
_ANTHROPIC_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE")


def llm_env_vars_for(protocol: LLMProtocol) -> tuple[str, ...]:
    if protocol == "openai":
        return _OPENAI_ENV_VARS
    if protocol == "anthropic":
        return _ANTHROPIC_ENV_VARS
    raise ValueError(f"unsupported llm_protocol: {protocol!r}")


@dataclass
class TaskResult:
    """Outcome of a single `agent.run(task)` invocation.

    Built-in adapters fill `exit_code` and `output` from the underlying CLI.
    User agents can populate `extra` with structured data they want to surface
    (e.g. parsed trajectory metadata).
    """

    exit_code: int = 0
    output: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class Task:
    """Handle returned by `agent.run_async(...)`.

    Mirrors a future plus a result. The agent's body runs in an engine-managed
    worker thread; `.wait()` blocks until that thread returns.
    """

    def __init__(self, future: Future[TaskResult], description: str) -> None:
        self._future = future
        self._description = description
        self._cancelled = threading.Event()

    @property
    def description(self) -> str:
        return self._description

    def done(self) -> bool:
        return self._future.done()

    def wait(self, timeout: float | None = None) -> TaskResult:
        try:
            return self._future.result(timeout=timeout)
        except FutureTimeoutError:
            raise TimeoutError(f"task timed out after {timeout}s")

    @property
    def result(self) -> TaskResult:
        if not self._future.done():
            raise RuntimeError("task is not finished; call .wait() first")
        return self._future.result()

    def cancel(self) -> None:
        """Request cancellation. The agent's `request_stop()` is invoked; the
        worker thread observes the flag on its next polling cycle."""
        self._cancelled.set()

    def cancellation_requested(self) -> bool:
        return self._cancelled.is_set()


class Agent:
    """Base class for user agents.

    Subclasses set `name` and `llm_protocol`, and override `install()` and
    `execute()`. The optional `on_restore()` hook fires after a checkpoint
    restore so the agent can re-sync any in-memory state.

    The engine guarantees, before calling `install()` or `execute()`:
      - `sbx.llm_base_url` is set to the interceptor URL for this sandbox.
      - Each env var named in `llm_env_vars_for(self.llm_protocol)` is set in
        both the sandbox's env and the process env of the engine worker that
        invokes `execute()`. So agents using standard provider SDKs (anthropic,
        openai) pick up the right base URL with zero code changes.
      - Outbound LLM calls carry an `X-Agent-Sandbox-Id` header that lets the
        interceptor route per-sandbox upstream URLs without IP magic.
    """

    name: ClassVar[str] = ""
    llm_protocol: ClassVar[LLMProtocol] = "openai"
    version: ClassVar[str] = ""

    # Profile metadata used by the engine. Builtin agents can override these
    # to express their image / bundle requirements.
    default_image: ClassVar[str | None] = None
    """Default container image; falls back to a base ubuntu image if unset."""

    requires_network_namespace: ClassVar[bool] = False
    """Whether this agent runs in-sandbox and needs a distinct network
    identity for multi-sandbox LLM routing. Host-side agents can leave this
    disabled and set the sandbox id header themselves."""

    def bind(
        self,
        sbx: "Sandbox",
        *,
        llm_url: str | None = None,
        install: bool = True,
    ) -> "Agent":
        """Bind this agent instance to an already-created sandbox.

        `install()` runs during binding by default. The same agent instance is
        returned so user code can write:

            agent = IFlowAgent().bind(sbx, llm_url="http://127.0.0.1:18080")
            result = agent.run(task)
        """
        sbx.attach_agent(self, llm_url=llm_url, install=install)
        return self

    def _set_bound_sandbox(self, sbx: "Sandbox") -> None:
        bound = getattr(self, "_bound_sandbox", None)
        if bound is not None and bound is not sbx:
            raise RuntimeError(
                f"agent {type(self).__name__} is already bound to sandbox {bound.sandbox_id}"
            )
        self._bound_sandbox = sbx

    @property
    def sandbox(self) -> "Sandbox":
        sbx = getattr(self, "_bound_sandbox", None)
        if sbx is None:
            raise RuntimeError("agent is not bound to a sandbox; call agent.bind(sbx) first")
        return sbx

    def install(self, sbx: "Sandbox") -> None:
        """Run once when the agent is bound to a sandbox. Use this to
        install CLIs, copy assets, or warm caches. The default is a no-op."""

    def run(self, task: str, *, timeout: float | None = None) -> TaskResult:
        """Execute one task synchronously and return its result."""
        return self.sandbox._run_agent_task_sync(task, timeout=timeout)

    def run_async(self, task: str) -> Task:
        """Start one task in the engine worker pool and return a task handle.

        The public SDK defaults to synchronous `run()` because the underlying
        agent CLI invocation usually runs to completion. This method remains
        for callers that intentionally want host-side concurrency.
        """
        return self.sandbox._submit_agent_task(task)

    def stop(self) -> None:
        """Request cooperative cancellation of the current task, if any."""
        self.sandbox._stop_agent_task()

    def status(self) -> dict[str, object]:
        """Return the current task status for this agent's sandbox."""
        return self.sandbox._agent_status()

    def execute(self, sbx: "Sandbox", task: str) -> TaskResult:
        """Agent implementation hook. Required. Return a `TaskResult`.

        The body of this method runs in the engine's process (i.e. on the
        host). For in-sandbox agents, it typically does one
        `sbx.commands.run(...)` that drives the agent CLI inside the sandbox
        until the task is finished. For on-host agents, it implements its
        own loop that issues many `sbx.commands.run(...)` calls.
        """
        raise NotImplementedError

    def on_restore(self, sbx: "Sandbox") -> None:
        """Called after a checkpoint restore. Default no-op."""

    def collect_results(self, sbx: "Sandbox") -> dict[str, object]:
        """Optional post-task hook. Default returns empty dict."""
        return {}

    def request_stop(self) -> None:
        """Cooperative cancellation. Default no-op."""


# ---------------------------------------------------------------------------
# Registry — built-in profiles + user-registered + import-path resolution
# ---------------------------------------------------------------------------


_AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_agent(name: str, factory: type[Agent] | Callable[[], Agent]) -> None:
    """Register a user agent under a name.

    `factory` can be either an `Agent` subclass (instantiated with no args) or
    a callable returning an Agent instance. Re-registering an existing name
    overwrites the previous binding.
    """
    if not name:
        raise ValueError("agent name must be non-empty")
    if isinstance(factory, type):
        if not issubclass(factory, Agent):
            raise TypeError(f"{factory!r} is not an Agent subclass")
        cls = factory

        def _make() -> Agent:
            return cls()

        registered: Callable[[], Agent] = _make
    elif callable(factory):
        registered = factory
    else:
        raise TypeError(f"factory must be an Agent subclass or callable, got {factory!r}")
    with _REGISTRY_LOCK:
        _AGENT_REGISTRY[name] = registered


def list_agents() -> list[str]:
    with _REGISTRY_LOCK:
        return sorted(_AGENT_REGISTRY.keys())


def resolve_agent(spec: str | Agent | type[Agent]) -> Agent:
    """Resolve an agent spec to an Agent instance.

    Accepts:
      - `Agent` instance — returned as-is.
      - `Agent` subclass — instantiated with no args.
      - registered name (e.g. "claude-code") — instantiated via factory.
      - import path "pkg.mod:Class" — imported and instantiated.
    """
    if isinstance(spec, Agent):
        return spec
    if isinstance(spec, type):
        if not issubclass(spec, Agent):
            raise TypeError(f"{spec!r} is not an Agent subclass")
        return spec()
    if isinstance(spec, str):
        if ":" in spec and "/" not in spec:
            # import-path form
            module_path, _, attr = spec.partition(":")
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                raise ValueError(f"could not import agent module {module_path!r}: {exc}") from exc
            obj = getattr(module, attr, None)
            if obj is None:
                raise ValueError(f"attribute {attr!r} not found in module {module_path}")
            return resolve_agent(obj)
        with _REGISTRY_LOCK:
            factory = _AGENT_REGISTRY.get(spec)
        if factory is None:
            raise KeyError(
                f"unknown agent {spec!r}; known agents: {sorted(_AGENT_REGISTRY.keys())}"
            )
        return factory()
    raise TypeError(f"unsupported agent spec: {spec!r}")


__all__ = [
    "Agent",
    "LLMProtocol",
    "Task",
    "TaskResult",
    "list_agents",
    "llm_env_vars_for",
    "register_agent",
    "resolve_agent",
]
