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
"""
from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Literal

if TYPE_CHECKING:
    from .sandbox import Sandbox


LLMProtocol = Literal["openai", "anthropic"]

# Provider environment variables Agent-CR sets while an agent is running, or
# returns from Agent.command_env(...) for in-sandbox CLI invocations.
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


class Agent:
    """Base class for user agents.

    Subclasses set `name` and `llm_protocol`, and override `install()` and
    `execute()`. The optional `on_restore()` hook fires after a checkpoint
    restore so the agent can re-sync any in-memory state.

    The engine guarantees, before calling `install()` or `execute()`:
      - `self.llm_base_url` is set to the interceptor URL for this sandbox.
      - Each env var named in `llm_env_vars_for(self.llm_protocol)` is set in
        the process env while `run()` invokes `execute()`. Agents that execute
        an in-sandbox CLI should pass `env=self.command_env(...)` to
        `sbx.commands.run(...)`.
      - Outbound LLM calls carry an `X-Agent-Sandbox-Id` header that lets the
        interceptor route per-sandbox upstream URLs without IP magic.
    """

    name: ClassVar[str] = ""
    llm_protocol: ClassVar[LLMProtocol] = "openai"
    version: ClassVar[str] = ""

    # Advisory metadata for templates/docs. Sandbox images are still selected
    # by Sandbox(...), not by binding an agent.
    default_image: ClassVar[str | None] = None
    """Suggested container image for this agent, if it has one."""

    requires_network_namespace: ClassVar[bool] = False
    """Whether this agent runs in-sandbox and needs a distinct network
    identity for multi-sandbox LLM routing. Host-side agents can leave this
    disabled and set the sandbox id header themselves."""

    # Host-inspector filters this agent contributes when it binds to a
    # sandbox. The Sandbox keeps its own default rules (idle init) and any
    # user-supplied rules separate, so binding/rebinding only touches
    # these. Default to empty tuples for agents that don't need filtering.
    HOST_INSPECTOR_IGNORE_PROCESS_RULES: ClassVar[tuple[dict[str, object], ...]] = ()
    HOST_INSPECTOR_IGNORED_PATH_PREFIXES: ClassVar[tuple[str, ...]] = ()

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
        if self.requires_network_namespace and not sbx.has_network_namespace:
            raise RuntimeError(
                "this agent requires a sandbox network namespace; create the "
                "sandbox with network=True or use an Engine with sandbox networking enabled"
            )
        previous_bound = getattr(self, "_bound_sandbox", None)
        previous_llm_url = getattr(self, "_llm_url", None)
        self._bind_sandbox(sbx)
        self._bind_llm_url(llm_url)
        registered_upstream = False
        try:
            if self._llm_url:
                sbx.engine.register_upstream(sbx.sandbox_id, self._llm_url)
                registered_upstream = True
            if install and not bool(getattr(self, "_installed", False)):
                sbx._install_attached_agent(self)
                self._installed = True
            # Push this agent's host-inspector filters to the sandbox. This
            # always REPLACES the previously installed agent-contributed
            # filters (so rebinding the same agent or binding a different
            # one never compounds the rule set), but leaves the sandbox's
            # default and user-supplied filters alone.
            sbx.add_host_inspector_filters(
                ignore_process_rules=[
                    dict(rule) for rule in type(self).HOST_INSPECTOR_IGNORE_PROCESS_RULES
                ],
                ignored_path_prefixes=list(type(self).HOST_INSPECTOR_IGNORED_PATH_PREFIXES),
            )
        except Exception:
            if registered_upstream:
                sbx.engine.unregister_upstream(sbx.sandbox_id)
            if previous_bound is None:
                try:
                    delattr(self, "_bound_sandbox")
                except AttributeError:
                    pass
            else:
                self._bound_sandbox = previous_bound
            self._llm_url = previous_llm_url
            raise
        return self

    def _bind_sandbox(self, sbx: "Sandbox") -> None:
        bound = getattr(self, "_bound_sandbox", None)
        if bound is not None and bound is not sbx:
            raise RuntimeError(
                f"agent {type(self).__name__} is already bound to sandbox {bound.sandbox_id}"
            )
        self._bound_sandbox = sbx

    def _bind_llm_url(self, llm_url: str | None) -> None:
        if llm_url:
            self._llm_url = llm_url.rstrip("/")
        elif not hasattr(self, "_llm_url"):
            self._llm_url = None

    @property
    def sandbox(self) -> "Sandbox":
        sbx = getattr(self, "_bound_sandbox", None)
        if sbx is None:
            raise RuntimeError("agent is not bound to a sandbox; call agent.bind(sbx) first")
        return sbx

    @property
    def upstream_url(self) -> str | None:
        """Real LLM/replay upstream URL registered for this agent."""
        return getattr(self, "_llm_url", None)

    @property
    def llm_base_url(self) -> str | None:
        """Provider base URL agents should use for intercepted LLM traffic."""
        base = self.sandbox.engine.interceptor_base_url
        if base is None:
            return None
        if self.llm_protocol == "openai":
            return f"{base}/v1"
        return base

    @property
    def openai_base_url(self) -> str | None:
        base = self.sandbox.engine.interceptor_base_url
        return None if base is None else f"{base}/v1"

    def command_env(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """Environment values this agent wants injected into sandbox commands."""
        env: dict[str, str] = {}
        base_url = self.llm_base_url
        if base_url:
            for var in llm_env_vars_for(self.llm_protocol):
                env[var] = base_url
            env["AGENT_CR_SANDBOX_ID"] = str(self.sandbox.sandbox_id)
        if overrides:
            env.update(overrides)
        return env

    def install(self, sbx: "Sandbox") -> None:
        """Run once when the agent is bound to a sandbox. Use this to
        install CLIs, copy assets, or warm caches. The default is a no-op."""

    def run(self, task: str) -> TaskResult:
        """Execute one task synchronously and return its result."""
        sbx = self.sandbox
        prior_env: dict[str, str | None] = {}
        env_vars = llm_env_vars_for(self.llm_protocol)
        base_url = self.llm_base_url
        try:
            if base_url:
                for var in env_vars:
                    prior_env[var] = os.environ.get(var)
                    os.environ[var] = base_url
                prior_env["AGENT_CR_SANDBOX_ID"] = os.environ.get("AGENT_CR_SANDBOX_ID")
                os.environ["AGENT_CR_SANDBOX_ID"] = str(sbx.sandbox_id)
            result = self.execute(sbx, task)
            if result is None:
                result = TaskResult()
            elif not isinstance(result, TaskResult):
                raise TypeError(
                    f"agent.execute must return TaskResult or None, got {type(result).__name__}"
                )
            extra = self.collect_results(sbx)
            if extra:
                result.extra.update(extra)
            return result
        finally:
            for var, prior in prior_env.items():
                if prior is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = prior

    def stop(self) -> None:
        """Request cooperative cancellation of the current task, if any."""
        self.request_stop()

    def execute(self, sbx: "Sandbox", task: str) -> TaskResult:
        """Agent implementation hook. Required. Return a `TaskResult`.

        The body of this method runs where the caller invoked `agent.run()`.
        For in-sandbox agents, it typically does one
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
    "TaskResult",
    "list_agents",
    "llm_env_vars_for",
    "register_agent",
    "resolve_agent",
]
