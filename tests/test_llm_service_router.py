from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
import sys

from integrations.llm_services.router import BenchmarkLLMRouter, BenchmarkLLMRouterClient, serve_benchmark_llm_router


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_json(url: str, *, timeout_s: float = 10.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


class _SnapshotProbeState:
    instances: dict[str, "_SnapshotProbeState"] = {}

    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = llm_service_config or {}
        self.name = str(config.get("name", "probe"))
        self.snapshot_calls: list[bool] = []
        type(self).instances[self.name] = self

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        _ = (path, headers, payload)
        return {}

    def snapshot(self, *, include_events: bool = True) -> dict[str, object]:
        self.snapshot_calls.append(include_events)
        return {"name": self.name, "events": ["included"] if include_events else []}

    def reset(self) -> None:
        return

    def restore(self, *, consumed_response_count: int) -> None:
        _ = consumed_response_count
        return


class _ExplodingSnapshotState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        _ = llm_service_config

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        _ = (path, headers, payload)
        return {}

    def snapshot(self, *, include_events: bool = True) -> dict[str, object]:
        raise AssertionError(f"unexpected snapshot(include_events={include_events}) on unrelated sandbox")

    def reset(self) -> None:
        return

    def restore(self, *, consumed_response_count: int) -> None:
        _ = consumed_response_count
        return


class BenchmarkLLMRouterTests(unittest.TestCase):
    def test_router_dispatches_to_simulated_service_for_registered_sandbox(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-sim", llm_service_type="simulated")

        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-sim"},
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
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_workdir",
        )

    def test_router_dispatches_to_iflow_simulated_service_for_registered_sandbox(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-iflow", llm_service_type="simulated_for_iflow")

        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
            payload={
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_shell_command",
                            "description": "x",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "run_shell_command",
        )

    def test_router_control_request_targets_registered_manual_service(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-manual", llm_service_type="manual")

        router.handle_control_request(
            path="/control/run_shell_command",
            payload={"sandbox_id": "sbx-manual", "command": 'sh -lc "echo hi"'},
        )
        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-manual"},
            payload={},
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "run_shell_command",
        )

    def test_router_dispatches_to_iflow_trace_replay_and_deduplicates_duplicate_tool_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "first"}],
                                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [
                                                    {
                                                        "id": "call-first",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": json.dumps(
                                                                {"command": 'sh -lc "echo first >/tmp/first.txt"'},
                                                                sort_keys=True,
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "second"}],
                                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [
                                                    {
                                                        "id": "call-second",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": json.dumps(
                                                                {"command": 'sh -lc "echo second >/tmp/second.txt"'},
                                                                sort_keys=True,
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            router = BenchmarkLLMRouter()
            router.register_sandbox(
                sandbox_id="sbx-replay",
                llm_service_type="iflow_trace_replay",
                llm_service_config={"trace_path": str(trace_path)},
            )

            first = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            second = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            replayed_second = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            router.reset_sandbox("sbx-replay")
            replayed_first = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        first_command = json.loads(first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        second_command = json.loads(second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        replayed_command = json.loads(replayed_second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        replayed_first_command = json.loads(
            replayed_first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["command"]
        self.assertEqual(first_command, 'sh -lc "echo first >/tmp/first.txt"')
        self.assertEqual(second_command, 'sh -lc "echo second >/tmp/second.txt"')
        self.assertIn("/dev/null", replayed_command)
        self.assertEqual(replayed_first_command, 'sh -lc "echo first >/tmp/first.txt"')

    def test_router_dispatches_to_iflow_trace_replay_and_preserves_duplicate_system_setup_tool_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            original_command = "apt-get install -y python3-pip && pip3 install --break-system-packages pyarrow"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "install deps"}],
                                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [
                                                    {
                                                        "id": "call-install",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": json.dumps(
                                                                {"command": original_command},
                                                                sort_keys=True,
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            router = BenchmarkLLMRouter()
            router.register_sandbox(
                sandbox_id="sbx-replay",
                llm_service_type="iflow_trace_replay",
                llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0},
            )

            original = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "install deps"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "install deps"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        original_args = json.loads(original["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        duplicate_args = json.loads(duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(original, duplicate)
        self.assertEqual(original_args["command"], original_command)
        self.assertEqual(duplicate_args["command"], original_command)

    def test_router_single_sandbox_snapshot_does_not_touch_other_registered_services(self) -> None:
        _SnapshotProbeState.instances.clear()
        router = BenchmarkLLMRouter(
            registry={
                "probe": _SnapshotProbeState,
                "explode": _ExplodingSnapshotState,
            }
        )
        router.register_sandbox(
            sandbox_id="sbx-keep",
            llm_service_type="probe",
            llm_service_config={"name": "keep"},
        )
        router.register_sandbox(
            sandbox_id="sbx-ignore",
            llm_service_type="explode",
        )

        snapshot = router.snapshot("sbx-keep", include_events=False)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["llm_service_type"], "probe")
        self.assertEqual(snapshot["state"]["name"], "keep")
        self.assertEqual(_SnapshotProbeState.instances["keep"].snapshot_calls, [False])

    def test_router_client_single_sandbox_snapshot_omits_replay_events_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "first"}],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {"choices": [{"message": {"content": "first"}}]},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            server = serve_benchmark_llm_router(host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            self.addCleanup(thread.join, 5.0)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            _wait_for_json(f"{base_url}/healthz")

            server.benchmark_llm_router.register_sandbox(
                sandbox_id="sbx-replay",
                llm_service_type="iflow_trace_replay",
                llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0},
            )
            server.benchmark_llm_router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [],
                },
            )

            direct_snapshot = server.benchmark_llm_router.snapshot("sbx-replay")
            client = BenchmarkLLMRouterClient(base_url)
            state = client.snapshot("sbx-replay")

        assert direct_snapshot is not None
        assert state is not None
        self.assertIn("events", direct_snapshot["state"])
        self.assertNotIn("events", state["state"])
        self.assertEqual(state["state"]["consumed_response_count"], 1)
        self.assertEqual(state["state"]["trace_cursor"], 1)

    def test_router_client_controls_thread_server(self) -> None:
        server = serve_benchmark_llm_router(host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5.0)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        _wait_for_json(f"{base_url}/healthz")
        client = BenchmarkLLMRouterClient(base_url)

        client.register_sandbox(sandbox_id="sbx-client", llm_service_type="simulated")
        snapshot = client.snapshot()
        assert snapshot is not None
        self.assertIn("sbx-client", snapshot)
        client.reset_sandbox("sbx-client")
        client.unregister_sandbox("sbx-client")
        self.assertIsNone(client.snapshot("sbx-client"))

    def test_router_cli_starts_process_and_serves_control_endpoints(self) -> None:
        port = _find_free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "integrations.llm_services.router",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_json(f"{base_url}/healthz")
            client = BenchmarkLLMRouterClient(base_url)
            client.register_sandbox(sandbox_id="sbx-process", llm_service_type="simulated")
            state = client.snapshot("sbx-process")
            self.assertIsNotNone(state)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5.0)
            if process.stderr is not None:
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
