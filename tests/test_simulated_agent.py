from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from integrations.sandboxes.simulated.agent_cli import AgentRuntime, parse_anthropic_tool_calls, parse_openai_tool_calls
from integrations.llm_services.simulated.service import (
    SimulatedLLMState,
    build_anthropic_response,
    build_openai_response,
    handle_request,
)


class SimulatedAgentTests(unittest.TestCase):
    def test_openai_response_contains_tool_calls(self) -> None:
        payload = build_openai_response("append_journal", 3)
        tool_calls = parse_openai_tool_calls(payload)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "append_journal")
        self.assertEqual(tool_calls[0]["input"]["filename"], "journal.log")

    def test_anthropic_response_contains_tool_use_block(self) -> None:
        payload = build_anthropic_response("spawn_probe", 4)
        tool_calls = parse_anthropic_tool_calls(payload)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "spawn_probe")
        self.assertEqual(tool_calls[0]["input"]["message"], "probe 4")

    def test_service_serves_both_provider_endpoints(self) -> None:
        state = SimulatedLLMState()
        openai_payload = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-1"},
            payload={
                "model": "simulated-openai",
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_workdir",
                            "description": "x",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            state=state,
        )
        self.assertEqual(
            openai_payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_workdir",
        )

        anthropic_payload = handle_request(
            path="/v1/messages",
            headers={"X-Agent-Sandbox-Id": "sbx-2"},
            payload={
                "model": "simulated-anthropic",
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "name": "fetch_proxy_health",
                        "description": "x",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
            },
            state=state,
        )
        self.assertEqual(anthropic_payload["content"][0]["name"], "fetch_proxy_health")

    def test_runtime_run_cycle_records_fetch_errors_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim_agent_") as tmp:
            runtime = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:1",
                sandbox_id="sbx-1",
                work_dir=Path(tmp),
                poll_interval_s=0.0,
                status_port=0,
            )

            calls = iter(
                [
                    TimeoutError("socket dropped"),
                    [{"name": "show_pwd", "input": {}}],
                ]
            )

            def fake_fetch() -> list[dict[str, object]]:
                item = next(calls)
                if isinstance(item, Exception):
                    raise item
                return item

            runtime.fetch_tool_calls = fake_fetch  # type: ignore[method-assign]

            runtime.run_cycle()
            self.assertEqual(runtime.state["request_errors"], 1)
            self.assertEqual(runtime.state["total_actions"], 0)
            self.assertEqual(runtime.state["last_error"]["stage"], "fetch_tool_calls")

            runtime.run_cycle()
            self.assertEqual(runtime.state["request_errors"], 1)
            self.assertEqual(runtime.state["total_actions"], 1)
            self.assertTrue(runtime.state["last_tool_result"]["cwd"].startswith(str(Path(tmp))))

    def test_runtime_reloads_persisted_state_on_relaunch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim_agent_") as tmp:
            work_dir = Path(tmp)
            runtime = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:1",
                sandbox_id="sbx-1",
                work_dir=work_dir,
                poll_interval_s=0.0,
                status_port=0,
            )
            runtime.memory_notes.append("remembered")
            runtime.state["total_actions"] = 7
            runtime.state["completed_requests"] = 3
            runtime.persist_state()
            first_runtime_id = runtime.state["runtime_id"]

            reloaded = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:1",
                sandbox_id="sbx-1",
                work_dir=work_dir,
                poll_interval_s=0.0,
                status_port=0,
            )

            self.assertEqual(reloaded.state["total_actions"], 7)
            self.assertEqual(reloaded.state["completed_requests"], 3)
            self.assertEqual(reloaded.memory_notes, ["remembered"])
            self.assertNotEqual(reloaded.state["runtime_id"], first_runtime_id)

    def test_runtime_writes_info_warning_and_debug_logs_to_work_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim_agent_") as tmp:
            runtime = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:1",
                sandbox_id="sbx-logs",
                work_dir=Path(tmp),
                poll_interval_s=0.0,
                status_port=0,
            )

            calls = iter(
                [
                    TimeoutError("socket dropped"),
                    [{"name": "show_pwd", "input": {}}],
                ]
            )

            def fake_fetch() -> list[dict[str, object]]:
                item = next(calls)
                if isinstance(item, Exception):
                    raise item
                return item

            runtime.fetch_tool_calls = fake_fetch  # type: ignore[method-assign]

            runtime.run_cycle()
            runtime.run_cycle()
            for handler in runtime.logger.handlers:
                handler.flush()

            log_text = runtime.log_path.read_text(encoding="utf-8")
            self.assertIn("INFO agent runtime started", log_text)
            self.assertIn("DEBUG starting agent cycle", log_text)
            self.assertIn("WARNING runtime error stage=fetch_tool_calls", log_text)
            self.assertIn("INFO running tool name=show_pwd", log_text)
            self.assertIn("DEBUG tool result name=show_pwd", log_text)

    def test_runtime_normalizes_v1_llm_base_for_chat_requests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim_agent_") as tmp:
            runtime = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:9000/v1",
                sandbox_id="sbx-v1",
                work_dir=Path(tmp),
                poll_interval_s=0.0,
                status_port=0,
            )

            self.assertEqual(
                runtime._build_url("/v1/chat/completions"),
                "http://127.0.0.1:9000/v1/chat/completions",
            )

    def test_runtime_normalizes_v1_llm_base_for_proxy_health_requests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim_agent_") as tmp:
            runtime = AgentRuntime(
                provider="openai",
                llm_base_url="http://127.0.0.1:9000/v1",
                sandbox_id="sbx-v1",
                work_dir=Path(tmp),
                poll_interval_s=0.0,
                status_port=0,
            )

            self.assertEqual(
                runtime._build_url("/healthz"),
                "http://127.0.0.1:9000/healthz",
            )


if __name__ == "__main__":
    unittest.main()
