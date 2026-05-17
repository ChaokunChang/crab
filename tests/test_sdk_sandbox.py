"""SDK unit tests — covers Sandbox, Agent, Engine, and the registry.

These tests use the in-memory runtime exclusively so they run anywhere
without runc/CRIU/ZFS. They validate the SDK contract and the wiring
between Sandbox → Engine → AgentCRSystem → interceptor.
"""
from __future__ import annotations

import unittest

from agent_cr.agent import (
    Agent,
    Task,
    TaskResult,
    list_agents,
    register_agent,
    resolve_agent,
)
from agent_cr.engine import Engine, EngineConfig, shutdown_default_engine
from agent_cr.sandbox import Sandbox


class _EchoAgent(Agent):
    name = "test-echo"
    llm_protocol = "openai"

    def __init__(self) -> None:
        self.installed = False
        self.tasks_run: list[str] = []
        self.restored = 0

    def install(self, sbx):
        self.installed = True

    def execute(self, sbx, task):
        self.tasks_run.append(task)
        return TaskResult(exit_code=0, output=f"echoed: {task}", extra={"len": len(task)})

    def on_restore(self, sbx):
        self.restored += 1


class _FailingInstallAgent(Agent):
    name = "test-fail"
    llm_protocol = "openai"

    def install(self, sbx):
        raise RuntimeError("install must fail strictly")

    def execute(self, sbx, task):
        return TaskResult(output="should not be reached")


class TestSandboxLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine.start(EngineConfig(runtime="docker"))

    def tearDown(self) -> None:
        self.engine.stop()
        shutdown_default_engine()

    def test_sandbox_launches_and_kills(self) -> None:
        sbx = Sandbox(image="ubuntu:22.04", engine=self.engine)
        try:
            self.assertIsNotNone(sbx.sandbox_id)
            self.assertFalse(sbx.closed)
        finally:
            sbx.kill()
        self.assertTrue(sbx.closed)

    def test_sandbox_as_context_manager(self) -> None:
        with Sandbox(image="ubuntu:22.04", engine=self.engine) as sbx:
            sandbox_id = sbx.sandbox_id
            self.assertFalse(sbx.closed)
        self.assertTrue(sbx.closed)

    def test_llm_url_is_registered_when_agent_binds(self) -> None:
        sbx = Sandbox(image="ubuntu:22.04", engine=self.engine)
        agent = _EchoAgent()
        try:
            agent.bind(sbx, llm_url="https://api.example")
            self.assertEqual(
                self.engine._lookup_upstream(sbx.sandbox_id),
                "https://api.example",
            )
        finally:
            sbx.kill()
        # After kill, upstream is unregistered.
        self.assertIsNone(self.engine._lookup_upstream(sbx.sandbox_id))


class TestAgentNamespace(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine.start(EngineConfig(runtime="docker"))

    def tearDown(self) -> None:
        self.engine.stop()
        shutdown_default_engine()

    def test_install_runs_once(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            self.assertTrue(agent.installed)
        finally:
            sbx.kill()

    def test_agent_bind_attaches_to_existing_sandbox(self) -> None:
        sbx = Sandbox(image="ubuntu:22.04", engine=self.engine)
        profile = _EchoAgent()
        try:
            agent = profile.bind(sbx, llm_url="https://api.example")
            self.assertIs(agent, profile)
            self.assertTrue(profile.installed)
            self.assertEqual(self.engine._lookup_upstream(sbx.sandbox_id), "https://api.example")
            result = agent.run("bound task")
            self.assertEqual(result.output, "echoed: bound task")
            self.assertEqual(agent.status()["state"], "done")
        finally:
            sbx.kill()

    def test_attach_agent_method(self) -> None:
        sbx = Sandbox(image="ubuntu:22.04", engine=self.engine)
        profile = _EchoAgent()
        try:
            attached = sbx.attach_agent(profile, llm_url="https://api.example")
            self.assertIs(attached, profile)
            self.assertTrue(profile.installed)
            self.assertIs(profile.sandbox, sbx)
            self.assertEqual(profile.run("attached method").output, "echoed: attached method")
        finally:
            sbx.kill()

    def test_install_failure_is_reported_by_bind(self) -> None:
        sbx = Sandbox(engine=self.engine)
        with self.assertRaises(RuntimeError):
            _FailingInstallAgent().bind(sbx)
        self.assertFalse(sbx.closed)
        sbx.kill()

    def test_multi_task_on_one_sandbox(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            r1 = agent.run("first")
            self.assertEqual(r1.exit_code, 0)
            self.assertEqual(r1.output, "echoed: first")

            r2 = agent.run("second")
            self.assertEqual(r2.output, "echoed: second")

            self.assertEqual(agent.tasks_run, ["first", "second"])
        finally:
            sbx.kill()

    def test_concurrent_task_rejected(self) -> None:
        import threading

        class _SlowAgent(Agent):
            name = "test-slow"
            llm_protocol = "openai"

            def __init__(self) -> None:
                self.proceed = threading.Event()

            def execute(self, sbx, task):
                self.proceed.wait(timeout=5.0)
                return TaskResult(output="done")

        agent = _SlowAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            t1 = agent.run_async("task one")
            with self.assertRaises(RuntimeError):
                agent.run_async("task two")
            agent.proceed.set()
            t1.wait()
        finally:
            sbx.kill()

    def test_no_agent_attached(self) -> None:
        sbx = Sandbox(image="ubuntu:22.04", engine=self.engine)
        agent = _EchoAgent()
        try:
            with self.assertRaises(RuntimeError):
                agent.run("nope")
            self.assertEqual(sbx._agent_status(), {"state": "idle"})
        finally:
            sbx.kill()

    def test_status_transitions(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            self.assertEqual(agent.status()["state"], "idle")
            agent.run("hello")
            self.assertEqual(agent.status()["state"], "done")
        finally:
            sbx.kill()


class TestAgentRegistry(unittest.TestCase):
    def test_builtins_registered_at_import(self) -> None:
        names = list_agents()
        self.assertIn("claude-code", names)
        self.assertIn("claude_code", names)
        self.assertIn("iflow", names)

    def test_register_and_resolve_by_name(self) -> None:
        register_agent("test-reg-1", _EchoAgent)
        agent = resolve_agent("test-reg-1")
        self.assertIsInstance(agent, _EchoAgent)

    def test_resolve_by_class_or_instance(self) -> None:
        self.assertIsInstance(resolve_agent(_EchoAgent), _EchoAgent)
        inst = _EchoAgent()
        self.assertIs(resolve_agent(inst), inst)

    def test_resolve_by_import_path(self) -> None:
        spec = "agent_cr.agents_builtin.claude_code:ClaudeCodeAgent"
        agent = resolve_agent(spec)
        self.assertEqual(agent.name, "claude-code")
        self.assertEqual(agent.llm_protocol, "anthropic")

    def test_unknown_name_raises(self) -> None:
        with self.assertRaises(KeyError):
            resolve_agent("does-not-exist")

    def test_invalid_factory_rejected(self) -> None:
        with self.assertRaises(TypeError):
            register_agent("bad", 42)  # type: ignore[arg-type]

    def test_blank_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            register_agent("", _EchoAgent)


class TestTaskHandle(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine.start(EngineConfig(runtime="docker"))

    def tearDown(self) -> None:
        self.engine.stop()
        shutdown_default_engine()

    def test_task_returns_result(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            t = agent.run_async("payload")
            self.assertIsInstance(t, Task)
            r = t.wait()
            self.assertIsInstance(r, TaskResult)
            self.assertEqual(r.output, "echoed: payload")
            self.assertEqual(r.extra["len"], 7)
        finally:
            sbx.kill()

    def test_task_result_property_after_wait(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            t = agent.run_async("x")
            t.wait()
            self.assertTrue(t.done())
            self.assertEqual(t.result.output, "echoed: x")
        finally:
            sbx.kill()

    def test_task_result_property_before_done_raises(self) -> None:
        agent = _EchoAgent()
        sbx = Sandbox(engine=self.engine)
        try:
            agent.bind(sbx)
            t = agent.run_async("y")
            # The task may not yet have started; result access must raise.
            if not t.done():
                with self.assertRaises(RuntimeError):
                    _ = t.result
            t.wait()
        finally:
            sbx.kill()


class TestEngineLifecycle(unittest.TestCase):
    def test_engine_starts_and_stops(self) -> None:
        engine = Engine.start(EngineConfig(runtime="docker"))
        self.assertTrue(engine.started)
        self.assertIsNotNone(engine.interceptor_base_url)
        engine.stop()
        self.assertFalse(engine.started)

    def test_engine_context_manager(self) -> None:
        with Engine.start(EngineConfig(runtime="docker")) as engine:
            self.assertTrue(engine.started)
        self.assertFalse(engine.started)

    def test_engine_connect_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            Engine.connect()

    def test_engine_disables_interceptor(self) -> None:
        engine = Engine.start(EngineConfig(runtime="docker", enable_interceptor=False))
        try:
            self.assertIsNone(engine.interceptor_base_url)
        finally:
            engine.stop()


if __name__ == "__main__":
    unittest.main()
