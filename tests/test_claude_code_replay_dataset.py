from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.generate_claude_code_replay_dataset import generate_dataset


def _trace_payload(
    *,
    bash_command: str = "echo ok",
    model_name: str = "claude-opus-4-6",
    task_tool: bool = False,
) -> dict[str, object]:
    tool_name = "Task" if task_tool else "Bash"
    tool_arguments: dict[str, object]
    if task_tool:
        tool_arguments = {
            "description": "Create helper outputs",
            "subagent_type": "general-purpose",
            "prompt": "Create /app/attack.py",
        }
        observation = {"results": [{"source_call_id": "toolu_1", "content": "subagent finished"}]}
        tool_metadata = {}
    else:
        tool_arguments = {"command": bash_command, "description": "Run command"}
        observation = {"results": [{"source_call_id": "toolu_1", "content": "ok"}]}
        tool_metadata = {"tool_result_metadata": {"tool_use_result": {"stdout": "ok"}}}
    return {
        "schema_version": "ATIF-v1.2",
        "session_id": "session-1",
        "agent": {
            "name": "claude-code",
            "version": "2.1.34",
            "model_name": model_name,
            "extra": {},
        },
        "steps": [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Working on it",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": model_name,
                "extra": {},
            },
            {
                "source": "agent",
                "message": f"Executed {tool_name} toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": model_name,
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": tool_name,
                        "arguments": tool_arguments,
                    }
                ],
                "observation": observation,
                "extra": tool_metadata,
            },
        ],
        "final_metrics": {"total_steps": 3, "extra": {}},
    }


class ClaudeCodeReplayDatasetTests(unittest.TestCase):
    def _write_task_assets(self, tasks_root: Path, task_id: str) -> None:
        task_root = tasks_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "task.yaml").write_text(
            "\n".join(
                [
                    "instruction: |-",
                    f"  Solve {task_id}",
                    "max_agent_timeout_sec: 900",
                    "max_test_timeout_sec: 180",
                    "run_tests_in_same_shell: false",
                ]
            ),
            encoding="utf-8",
        )
        (task_root / "docker-compose.yaml").write_text(
            "\n".join(
                [
                    "services:",
                    "  client:",
                    "    image: example:latest",
                ]
            ),
            encoding="utf-8",
        )
        (task_root / "run-tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def _write_results_trace(
        self,
        traces_root: Path,
        *,
        trial_id: str,
        task_dir_name: str,
        payload: dict[str, object],
    ) -> str:
        results_path = f"{trial_id}-results"
        agent_root = traces_root / results_path / task_dir_name / "agent"
        agent_root.mkdir(parents=True, exist_ok=True)
        (agent_root / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")
        return results_path

    def test_generate_dataset_uses_manifest_pass_rows_and_round_robins_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")
            self._write_task_assets(tasks_root, "beta")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-1",
                        "task_name": "alpha",
                        "task_checksum": "chk-a1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-1",
                            task_dir_name="alpha__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "alpha-2",
                        "task_name": "alpha",
                        "task_checksum": "chk-a2",
                        "result": "✓ Pass",
                        "total_steps": 6,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-2",
                            task_dir_name="alpha__two",
                            payload=_trace_payload(bash_command="$aa"),
                        ),
                    },
                    {
                        "trial_id": "beta-1",
                        "task_name": "beta",
                        "task_checksum": "chk-b1",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="beta-1",
                            task_dir_name="beta__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "gamma-1",
                        "task_name": "headless-terminal",
                        "task_checksum": "chk-g1",
                        "result": "✓ Pass",
                        "total_steps": 8,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="gamma-1",
                            task_dir_name="gamma__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "beta-fail",
                        "task_name": "beta",
                        "task_checksum": "chk-bf",
                        "result": "✗ Fail",
                        "total_steps": 9,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="beta-fail",
                            task_dir_name="beta__fail",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual([row["task_id"] for row in rows], ["alpha", "beta", "alpha"])
        self.assertEqual(rows[0]["trace_trial_id"], "alpha-1")
        self.assertEqual(rows[2]["trace_trial_id"], "alpha-2")
        self.assertEqual(rows[0]["trace_agent_version"], "2.1.34")
        self.assertEqual(rows[0]["task_config"]["options"]["trace_agent_version"], "2.1.34")
        self.assertTrue(rows[0]["llm_service_config"]["trace_path"].endswith("agent/trajectory.json"))

    def test_generate_dataset_excludes_default_unsupported_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")
            self._write_task_assets(tasks_root, "git-multibranch")
            self._write_task_assets(tasks_root, "hf-model-inference")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-1",
                        "task_name": "alpha",
                        "task_checksum": "chk-a1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-1",
                            task_dir_name="alpha__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "git-1",
                        "task_name": "git-multibranch",
                        "task_checksum": "chk-g1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="git-1",
                            task_dir_name="git_multibranch__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "hf-1",
                        "task_name": "hf-model-inference",
                        "task_checksum": "chk-h1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="hf-1",
                            task_dir_name="hf_model_inference__one",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual([row["task_id"] for row in rows], ["alpha"])

    def test_generate_dataset_excludes_default_bad_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "mailman")
            self._write_task_assets(tasks_root, "qemu-alpine-ssh")
            self._write_task_assets(tasks_root, "qemu-startup")
            self._write_task_assets(tasks_root, "fix-git")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "46c2e780-0160-4acf-a1e6-668cc5ca506b",
                        "task_name": "mailman",
                        "task_checksum": "chk-mailman-good",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="46c2e780-0160-4acf-a1e6-668cc5ca506b",
                            task_dir_name="mailman__good",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8",
                        "task_name": "mailman",
                        "task_checksum": "chk-mailman-bad",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8",
                            task_dir_name="mailman__bad",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "1a8b08c0-fd6e-46bb-98d0-ebf7bb43fe2c",
                        "task_name": "qemu-alpine-ssh",
                        "task_checksum": "chk-qemu-good",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="1a8b08c0-fd6e-46bb-98d0-ebf7bb43fe2c",
                            task_dir_name="qemu__good",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "8b56ce95-9f8b-4020-839d-5d1cc6dc9a10",
                        "task_name": "qemu-alpine-ssh",
                        "task_checksum": "chk-qemu-bad",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="8b56ce95-9f8b-4020-839d-5d1cc6dc9a10",
                            task_dir_name="qemu__bad",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "a80f3594-1090-4c08-8764-d2731d91b683",
                        "task_name": "qemu-startup",
                        "task_checksum": "chk-qemu-start-good",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="a80f3594-1090-4c08-8764-d2731d91b683",
                            task_dir_name="qemu_startup__good",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "d19326ec-e1ff-476b-9b76-83103b1c8694",
                        "task_name": "qemu-startup",
                        "task_checksum": "chk-qemu-start-bad",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="d19326ec-e1ff-476b-9b76-83103b1c8694",
                            task_dir_name="qemu_startup__bad",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "b7f7e39a-36ef-497e-930a-5d38a870fec0",
                        "task_name": "fix-git",
                        "task_checksum": "chk-fix-git-good",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="b7f7e39a-36ef-497e-930a-5d38a870fec0",
                            task_dir_name="fix_git__good",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "4437cac5-d01b-43b8-ac37-f1d6f26cea89",
                        "task_name": "fix-git",
                        "task_checksum": "chk-fix-git-bad",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="4437cac5-d01b-43b8-ac37-f1d6f26cea89",
                            task_dir_name="fix_git__bad",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual(
            [row["trace_trial_id"] for row in rows],
            [
                "b7f7e39a-36ef-497e-930a-5d38a870fec0",
                "46c2e780-0160-4acf-a1e6-668cc5ca506b",
                "1a8b08c0-fd6e-46bb-98d0-ebf7bb43fe2c",
                "a80f3594-1090-4c08-8764-d2731d91b683",
            ],
        )

    def test_generate_dataset_excludes_additional_tasks_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")
            self._write_task_assets(tasks_root, "beta")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-1",
                        "task_name": "alpha",
                        "task_checksum": "chk-a1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-1",
                            task_dir_name="alpha__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "beta-1",
                        "task_name": "beta",
                        "task_checksum": "chk-b1",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="beta-1",
                            task_dir_name="beta__one",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                exclude_tasks={"beta"},
            )

        self.assertEqual([row["task_id"] for row in rows], ["alpha"])

    def test_generate_dataset_can_exclude_and_include_specific_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-1",
                        "task_name": "alpha",
                        "task_checksum": "chk-a1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-1",
                            task_dir_name="alpha__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "alpha-2",
                        "task_name": "alpha",
                        "task_checksum": "chk-a2",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-2",
                            task_dir_name="alpha__two",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            excluded_rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                exclude_trial_ids={"alpha-1"},
            )
            included_rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                include_trial_ids={"alpha-2"},
            )

        self.assertEqual([row["trace_trial_id"] for row in excluded_rows], ["alpha-2"])
        self.assertEqual([row["trace_trial_id"] for row in included_rows], ["alpha-2"])

    def test_generate_dataset_strict_replayable_and_deduplicate_pick_best_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")
            self._write_task_assets(tasks_root, "beta")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-0",
                        "task_name": "alpha",
                        "task_checksum": "chk-a0",
                        "result": "✓ Pass",
                        "total_steps": 1,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-0",
                            task_dir_name="alpha__zero",
                            payload=_trace_payload(task_tool=True),
                        ),
                    },
                    {
                        "trial_id": "alpha-1",
                        "task_name": "alpha",
                        "task_checksum": "chk-a1",
                        "result": "✓ Pass",
                        "total_steps": 5,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-1",
                            task_dir_name="alpha__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "alpha-2",
                        "task_name": "alpha",
                        "task_checksum": "chk-a2",
                        "result": "✓ Pass",
                        "total_steps": 3,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="alpha-2",
                            task_dir_name="alpha__two",
                            payload=_trace_payload(bash_command="$aa"),
                        ),
                    },
                    {
                        "trial_id": "beta-1",
                        "task_name": "beta",
                        "task_checksum": "chk-b1",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="beta-1",
                            task_dir_name="beta__one",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            strict_rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                strict_replayable=True,
            )
            dedup_rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                deduplicate=True,
            )

        self.assertEqual([row["trace_trial_id"] for row in strict_rows], ["alpha-1", "beta-1"])
        self.assertEqual(len(dedup_rows), 2)
        self.assertEqual(dedup_rows[0]["trace_trial_id"], "alpha-1")
        self.assertEqual(dedup_rows[1]["trace_trial_id"], "beta-1")

    def test_generate_dataset_skips_missing_and_ambiguous_results_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "alpha")

            good_results = self._write_results_trace(
                traces_root,
                trial_id="alpha-good",
                task_dir_name="alpha__good",
                payload=_trace_payload(),
            )
            ambiguous_root = traces_root / "alpha-ambiguous-results"
            (ambiguous_root / "one" / "agent").mkdir(parents=True, exist_ok=True)
            (ambiguous_root / "two" / "agent").mkdir(parents=True, exist_ok=True)
            (ambiguous_root / "one" / "agent" / "trajectory.json").write_text(
                json.dumps(_trace_payload()),
                encoding="utf-8",
            )
            (ambiguous_root / "two" / "agent" / "trajectory.json").write_text(
                json.dumps(_trace_payload()),
                encoding="utf-8",
            )

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "alpha-good",
                        "task_name": "alpha",
                        "task_checksum": "chk-good",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": good_results,
                    },
                    {
                        "trial_id": "alpha-missing",
                        "task_name": "alpha",
                        "task_checksum": "chk-missing",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": "alpha-missing-results",
                    },
                    {
                        "trial_id": "alpha-ambiguous",
                        "task_name": "alpha",
                        "task_checksum": "chk-ambiguous",
                        "result": "✓ Pass",
                        "total_steps": 4,
                        "agent_version": "2.1.34",
                        "results_path": "alpha-ambiguous-results",
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual([row["trace_trial_id"] for row in rows], ["alpha-good"])

    def test_generate_dataset_deduplicate_prefers_empirically_validated_mailman_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            traces_root.mkdir(parents=True, exist_ok=True)
            output_path = root / "dataset.jsonl"

            self._write_task_assets(tasks_root, "mailman")

            manifest = {
                "trajectories": [
                    {
                        "trial_id": "41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8",
                        "task_name": "mailman",
                        "task_checksum": "chk-mailman-a",
                        "result": "✓ Pass",
                        "total_steps": 43,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8",
                            task_dir_name="mailman__one",
                            payload=_trace_payload(),
                        ),
                    },
                    {
                        "trial_id": "46c2e780-0160-4acf-a1e6-668cc5ca506b",
                        "task_name": "mailman",
                        "task_checksum": "chk-mailman-b",
                        "result": "✓ Pass",
                        "total_steps": 59,
                        "agent_version": "2.1.34",
                        "results_path": self._write_results_trace(
                            traces_root,
                            trial_id="46c2e780-0160-4acf-a1e6-668cc5ca506b",
                            task_dir_name="mailman__two",
                            payload=_trace_payload(),
                        ),
                    },
                ]
            }
            (traces_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
                deduplicate=True,
            )

        self.assertEqual([row["trace_trial_id"] for row in rows], ["46c2e780-0160-4acf-a1e6-668cc5ca506b"])


if __name__ == "__main__":
    unittest.main()
