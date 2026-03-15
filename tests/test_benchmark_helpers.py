from __future__ import annotations

import ipaddress
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_cr import (
    ArtifactKind,
    ArtifactReference,
    CheckpointId,
    CheckpointManifest,
    EBPFEventKind,
    SandboxId,
    SchedulerConfig,
)
from integrations.agents import BaseAgent, IFlowAgent, SandboxHandle, TaskConfig, TaskDescription
from benchmarks.bench_agent_cr_sandbox_e2e import run_benchmark
from agent_cr.models import utc_now
from benchmarks.bench_tree_search import choose_replay_steps
from benchmarks.real_host_scenario_base import (
    BenchmarkTaskRecord,
    RealHostScenarioHarness,
    TreeSearchCheckpointRecord,
    build_tree_search_checkpoint_index,
    bounded_probability,
    compute_summary,
    parse_ipv4_route_networks,
    resolve_checkpoint_copy_plan,
    resolve_work_dir_host_path,
    select_benchmark_network,
    select_injected_indices,
    total_actions,
    write_bundle_config,
)


class BenchmarkHelperTests(unittest.TestCase):
    def _tree_search_manifest(
        self,
        checkpoint_id: str,
        *,
        step: int | None,
    ) -> CheckpointManifest:
        metadata: dict[str, object] = {}
        if step is not None:
            metadata["tree_search_step"] = step
        return CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId(checkpoint_id),
            sandbox_id=SandboxId("sbx"),
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata=metadata,
        ).with_integrity()

    def test_choose_replay_steps_is_deterministic(self) -> None:
        self.assertEqual(choose_replay_steps(6, 2), [1, 3])
        self.assertEqual(choose_replay_steps(4, 10), [1, 2, 3])

    def test_build_tree_search_checkpoint_index_collects_steps(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=2),
            self._tree_search_manifest("ckpt-3", step=3),
        ]

        self.assertEqual(
            build_tree_search_checkpoint_index(manifests, initial_steps=3, require_complete=True),
            {
                1: TreeSearchCheckpointRecord(CheckpointId("ckpt-1"), replay_actions=1),
                2: TreeSearchCheckpointRecord(CheckpointId("ckpt-2"), replay_actions=2),
                3: TreeSearchCheckpointRecord(CheckpointId("ckpt-3"), replay_actions=3),
            },
        )

    def test_build_tree_search_checkpoint_index_rejects_duplicates(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=1),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate tree-search checkpoint for step 1"):
            build_tree_search_checkpoint_index(manifests, initial_steps=2)

    def test_build_tree_search_checkpoint_index_rejects_missing_steps_when_required(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-3", step=3),
        ]

        with self.assertRaisesRegex(ValueError, r"missing tree-search checkpoints for steps \[2\]"):
            build_tree_search_checkpoint_index(manifests, initial_steps=3, require_complete=True)

    def test_build_tree_search_checkpoint_index_ignores_extra_trailing_steps(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=2),
            self._tree_search_manifest("ckpt-3", step=3),
            self._tree_search_manifest("ckpt-4", step=4),
        ]

        self.assertEqual(
            build_tree_search_checkpoint_index(manifests, initial_steps=3),
            {
                1: TreeSearchCheckpointRecord(CheckpointId("ckpt-1"), replay_actions=1),
                2: TreeSearchCheckpointRecord(CheckpointId("ckpt-2"), replay_actions=2),
                3: TreeSearchCheckpointRecord(CheckpointId("ckpt-3"), replay_actions=3),
            },
        )

    def test_compute_summary_averages_metrics(self) -> None:
        rows = [
            {"checkpoint_ms": 10.0, "restore_ms": 20.0},
            {"checkpoint_ms": 30.0, "restore_ms": 40.0},
        ]
        self.assertEqual(
            compute_summary(rows, ["checkpoint_ms", "restore_ms"]),
            {"checkpoint_ms": 20.0, "restore_ms": 30.0},
        )

    def test_parse_ipv4_route_networks_ignores_default_and_host_routes(self) -> None:
        self.assertEqual(
            parse_ipv4_route_networks(
                "\n".join(
                    [
                        "default via 172.24.95.253 dev eth0",
                        "10.250.0.0/24 dev acb0 proto kernel scope link src 10.250.0.1",
                        "172.24.95.253 dev eth0 scope link src 172.24.82.236",
                    ]
                )
            ),
            [ipaddress.ip_network("10.250.0.0/24")],
        )

    def test_select_benchmark_network_skips_occupied_routes(self) -> None:
        bridge_ip, network_cidr = select_benchmark_network(
            existing_routes="\n".join(
                [
                    "10.250.0.0/24 dev acb0 proto kernel scope link src 10.250.0.1",
                    "10.250.1.0/24 dev acb1 proto kernel scope link src 10.250.1.1",
                ]
            )
        )
        self.assertEqual((bridge_ip, network_cidr), ("10.250.2.1", "10.250.2.0/24"))

    def test_total_actions_reads_payload(self) -> None:
        self.assertEqual(total_actions({"total_actions": 7}), 7)

    def test_select_injected_indices_honors_first_forced_iteration(self) -> None:
        import random

        rng = random.Random(0)
        self.assertEqual(
            select_injected_indices(3, iteration=1, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        self.assertEqual(
            select_injected_indices(3, iteration=2, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        forced = select_injected_indices(3, iteration=3, rate=0.0, first_forced_iteration=3, rng=rng)
        self.assertEqual(forced, [0])

    def test_resolve_checkpoint_copy_plan_only_includes_required_ancestors(self) -> None:
        sid = SandboxId("sbx")
        checkpoint_order = [CheckpointId("ckpt-1"), CheckpointId("ckpt-2"), CheckpointId("ckpt-3")]
        process_ref = ArtifactReference(
            kind=ArtifactKind.PROCESS,
            name="process_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/process/process_checkpoint.json",
            size_bytes=1,
            sha256="0" * 64,
            metadata={},
        )
        filesystem_ref = ArtifactReference(
            kind=ArtifactKind.FILESYSTEM,
            name="filesystem_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/filesystem/filesystem_checkpoint.json",
            size_bytes=1,
            sha256="1" * 64,
            metadata={},
        )
        manifests = {
            CheckpointId("ckpt-1"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[filesystem_ref],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-2"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-3"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
        }

        self.assertEqual(
            resolve_checkpoint_copy_plan(checkpoint_order, manifests, CheckpointId("ckpt-3")),
            [
                (CheckpointId("ckpt-1"), False, True),
                (CheckpointId("ckpt-3"), True, False),
            ],
        )

    def test_bounded_probability_rejects_invalid_values(self) -> None:
        self.assertEqual(bounded_probability("0.3"), 0.3)
        with self.assertRaises(Exception):
            bounded_probability("3.0")

    def test_resolve_work_dir_host_path_uses_per_sandbox_subdirectory(self) -> None:
        root = Path("/tmp/bench-workdirs")
        self.assertEqual(
            resolve_work_dir_host_path(root, "sandbox-1"),
            root / "sandbox-1",
        )
        self.assertIsNone(resolve_work_dir_host_path(None, "sandbox-1"))

    def test_resolve_work_dir_host_path_makes_relative_roots_absolute(self) -> None:
        self.assertEqual(
            resolve_work_dir_host_path(Path("logs/tmp"), "sandbox-1"),
            (Path.cwd() / "logs/tmp").resolve() / "sandbox-1",
        )

    def test_clone_host_work_dir_copies_source_contents_for_fork(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            work_dir_host_root=Path("/tmp/placeholder"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            harness.work_dir_host_root = Path(tmp)
            source_dir = resolve_work_dir_host_path(harness.work_dir_host_root, "source-box")
            target_dir = resolve_work_dir_host_path(harness.work_dir_host_root, "fork-box")
            assert source_dir is not None
            assert target_dir is not None
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "agent_cli.log").write_text("from-source\n", encoding="utf-8")
            (source_dir / "nested").mkdir()
            (source_dir / "nested" / "state.json").write_text('{"step": 1}\n', encoding="utf-8")
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

            harness._clone_host_work_dir(SandboxId("source-box"), SandboxId("fork-box"))

            self.assertEqual((target_dir / "agent_cli.log").read_text(encoding="utf-8"), "from-source\n")
            self.assertEqual((target_dir / "nested" / "state.json").read_text(encoding="utf-8"), '{"step": 1}\n')
            self.assertFalse((target_dir / "stale.txt").exists())

    def test_write_bundle_config_adds_work_dir_bind_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [{"type": "network"}, {"type": "pid"}], "seccomp": {"defaultAction": "SCMP_ACT_ERRNO"}},
                        "mounts": [{"destination": "/proc", "source": "proc", "type": "proc"}],
                        "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": True},
                    }
                ),
                encoding="utf-8",
            )

            work_dir_host_path = bundle_dir / "host-work" / "sandbox-1"
            write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="http://127.0.0.1:9000/v1",
                provider="openai",
                sandbox_name="sandbox-1",
                status_port=9001,
                cgroup_path="agent-cr/test/sandbox-1",
                work_dir_host_path=work_dir_host_path,
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIn("AGENT_CR_LLM_BASE_URL=http://127.0.0.1:9000/v1", payload["process"]["env"])
            work_mounts = [mount for mount in payload["mounts"] if mount["destination"] == "/work"]
            self.assertEqual(
                work_mounts,
                [
                    {
                        "destination": "/work",
                        "source": str(work_dir_host_path),
                        "type": "bind",
                        "options": ["rbind", "rw"],
                    }
                ],
            )
            self.assertTrue(work_dir_host_path.is_dir())
            self.assertEqual(payload["process"]["cwd"], "/work")

    def test_sandbox_handle_status_url_uses_status_host(self) -> None:
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-status"),
            bundle_dir=Path("/tmp/sbx-status"),
            status_port=8123,
            status_host="10.250.0.22",
            last_status={},
        )

        self.assertEqual(handle.status_url, "http://10.250.0.22:8123/status")

    def test_allocate_benchmark_network_lease_assigns_unique_guest_ips(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            ),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path("/tmp")

        with patch("benchmarks.real_host_scenario_base.subprocess.run") as run:
            first = harness._allocate_benchmark_network_lease(SandboxId("sbx-a"))
            second = harness._allocate_benchmark_network_lease(SandboxId("sbx-b"))

        self.assertNotEqual(first.guest_ip, second.guest_ip)
        self.assertEqual(first.namespace_path.name, first.namespace_name)
        self.assertEqual(second.namespace_path.name, second.namespace_name)
        self.assertTrue(run.called)

    def test_release_benchmark_network_lease_cleans_up_ip_mapping(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            ),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path("/tmp")

        with patch("benchmarks.real_host_scenario_base.subprocess.run"):
            lease = harness._allocate_benchmark_network_lease(SandboxId("sbx-a"))
            harness._benchmark_ip_to_sandbox[lease.guest_ip] = SandboxId("sbx-a")
            harness._release_benchmark_network_lease(SandboxId("sbx-a"))

        self.assertNotIn(lease.guest_ip, harness._benchmark_ip_to_sandbox)
        self.assertNotIn(SandboxId("sbx-a"), harness._benchmark_network_leases)

    def test_resolve_interceptor_sandbox_id_uses_registered_guest_ip_for_any_benchmark(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            ),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness._benchmark_ip_to_sandbox["10.250.0.42"] = SandboxId("spot-0")

        self.assertEqual(
            harness.resolve_interceptor_sandbox_id("10.250.0.42", {}, b""),
            "spot-0",
        )
        self.assertIsNone(harness.resolve_interceptor_sandbox_id("10.250.0.43", {}, b""))

    def test_launch_sandbox_and_task_records_task_metadata(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-launch"),
            bundle_dir=Path("/tmp/sbx-launch"),
            status_port=8123,
            last_status={},
        )
        task_description = TaskDescription("solve a task")
        task_config = TaskConfig(minimum_actions=3)

        with patch.object(harness, "launch_sandbox", return_value=handle) as launch_sandbox:
            with patch.object(harness, "launch_task") as launch_task:
                result = harness.launch_sandbox_and_task(
                    "sbx-launch",
                    agent_type="simulated",
                    llm_service_type=None,
                    task_description=task_description,
                    task_config=task_config,
                )

        self.assertIs(result, handle)
        launch_sandbox.assert_called_once_with("sbx-launch", agent_type="simulated", llm_service_type=None)
        launch_task.assert_called_once_with("simulated", task_description, task_config, "sbx-launch")

    def test_launch_task_creates_task_run_and_future(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-agent"),
            bundle_dir=Path("/tmp/sbx-agent"),
            status_port=8123,
            last_status={},
        )
        harness._sandbox_by_id[handle.sandbox_id] = handle
        mock_task_run = Mock()
        mock_future = Mock()
        task_description = TaskDescription("progress")
        task_config = TaskConfig(minimum_actions=2)

        with patch.object(harness, "build_task_run", return_value=mock_task_run) as build_task_run:
            with patch.object(harness._task_executor, "submit", return_value=mock_future) as submit:
                returned = harness.launch_task("simulated", task_description, task_config, "sbx-agent")

        self.assertIs(returned, mock_task_run)
        self.assertEqual(handle.agent_type, "simulated")
        self.assertEqual(handle.task_description, task_description)
        self.assertEqual(handle.task_config, task_config)
        self.assertIs(handle.task_run, mock_task_run)
        self.assertIs(handle.task_future, mock_future)
        build_task_run.assert_called_once_with("simulated", handle, task_description, task_config)
        submit.assert_called_once_with(mock_task_run.perform_task)

    def test_build_task_run_passes_explicit_runtime_inputs(self) -> None:
        class _RecordingAgent(BaseAgent):
            agent_type = "recording"

            def perform_task(self) -> None:
                return None

        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path("/tmp/harness-root")
        harness.runtime_state_root = Path("/tmp/runtime-root")
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-explicit"),
            bundle_dir=Path("/tmp/sbx-explicit"),
            status_port=8123,
            last_status={},
            llm_base_url="http://127.0.0.1:43123/v1",
        )

        with patch.object(harness, "get_agent_class", return_value=_RecordingAgent):
            agent = harness.build_task_run("recording", sandbox, TaskDescription("go"), TaskConfig())

        self.assertEqual(agent.runtime_state_root, Path("/tmp/runtime-root"))
        self.assertEqual(agent.agent_host_dir, Path("/tmp/harness-root/recording/sbx-explicit"))
        self.assertEqual(agent.llm_base_url, "http://127.0.0.1:43123/v1")

    def test_relaunch_sandbox_creates_fresh_task_run(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-relaunch"),
            bundle_dir=Path("/tmp/sbx-relaunch"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(minimum_actions=0),
        )
        harness.sandbox_manager = SimpleNamespace(
            describe=lambda sandbox_id: SimpleNamespace(metadata={"zfs_dataset": "", "bundle_path": "/tmp/bundle"}),
            launch=Mock(),
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.runtime_state_root = Path("/tmp/runtime")

        fake_task_run = SimpleNamespace(
            wait_for_task_ready=Mock(),
            poll_status=Mock(return_value={"total_actions": 0}),
        )
        with patch.object(harness, "launch_task", side_effect=lambda *args, **kwargs: setattr(sandbox, "task_run", fake_task_run)):
            harness.relaunch_sandbox(sandbox)

        self.assertIs(sandbox.task_run, fake_task_run)
        fake_task_run.wait_for_task_ready.assert_called_once()
        fake_task_run.poll_status.assert_called_once()
        self.assertEqual(sandbox.llm_base_url, "http://10.250.0.1:43123/v1")

    def test_launch_sandbox_from_docker_compose_file_translates_supported_service(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="agent_cr_compose_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        launch_mock = Mock()
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=launch_mock))
        config_dir = harness.root / "bundle"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-compose"),
            bundle_dir=config_dir,
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        compose_file = harness.root / "compose.yaml"
        compose_file.write_text(
            """
services:
  app:
    image: alpine:3.20
    command: "echo hello"
    working_dir: /work
    environment:
      HELLO: world
""".strip(),
            encoding="utf-8",
        )
        env_file = harness.root / ".env"
        env_file.write_text("", encoding="utf-8")

        with patch.object(harness, "_allocate_benchmark_network_lease", return_value=SimpleNamespace(guest_ip="10.250.0.2")):
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)):
                with patch("benchmarks.real_host_scenario_base.export_docker_image_rootfs", return_value=harness.root / "rootfs"):
                    result = harness.launch_sandbox_from_docker_compose_file(
                        compose_file,
                        env_file,
                        sandbox_name="sbx-compose",
                        service_name="app",
                    )

        self.assertEqual(result.launch_source, "compose")
        launch_mock.assert_called_once()
        metadata = launch_mock.call_args.args[1]
        self.assertEqual(metadata["compose_service_name"], "app")

    def test_launch_sandbox_uses_host_network_for_simulated_agents(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="agent_cr_launch_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=Mock()))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=Mock()))
        sandbox_image = SimpleNamespace(exported_rootfs=harness.root / "rootfs")
        sandbox_image.exported_rootfs.mkdir(parents=True, exist_ok=True)

        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-sim"),
            bundle_dir=harness.root / "bundle",
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=[]),
            extra_launch_metadata=Mock(return_value={}),
            wait_for_task_ready=Mock(),
        )

        with patch.object(harness, "_allocate_benchmark_network_lease") as allocate_network:
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)) as prepare_handle:
                with patch.object(harness, "build_task_run", return_value=task_run), patch.object(
                    harness,
                    "ensure_sandbox_image",
                    return_value=sandbox_image,
                ):
                    result = harness.launch_sandbox("sbx-sim", agent_type="simulated")

        self.assertIs(result, handle)
        allocate_network.assert_not_called()
        prepare_handle.assert_called_once_with(
            "sbx-sim",
            interceptor_host="127.0.0.1",
            network_lease=None,
            agent_type="simulated",
            llm_service_type="simulated",
        )

    def test_prepare_sandbox_handle_sets_llm_base_url(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="agent_cr_handle_test_"))
        harness.interceptor = SimpleNamespace(port=43123)
        register_sandbox = Mock()
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=register_sandbox))

        with patch("benchmarks.real_host_scenario_base.subprocess.run"), patch(
            "benchmarks.real_host_scenario_base.write_bundle_config"
        ), patch("benchmarks.real_host_scenario_base.find_free_port", return_value=8123):
            handle, _ = harness._prepare_sandbox_handle(
                "sbx-url",
                interceptor_host="10.250.0.1",
                network_lease=None,
                agent_type="iflow",
                llm_service_type="simulated_for_iflow",
            )

        self.assertEqual(handle.llm_base_url, "http://10.250.0.1:43123/v1")
        register_sandbox.assert_called_once_with(
            sandbox_id="sbx-url",
            llm_service_type="simulated_for_iflow",
        )

    def test_launch_sandbox_uses_benchmark_network_for_iflow_agents(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="agent_cr_launch_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=Mock()))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=Mock()))
        sandbox_image = SimpleNamespace(exported_rootfs=harness.root / "rootfs")
        sandbox_image.exported_rootfs.mkdir(parents=True, exist_ok=True)

        lease = SimpleNamespace(guest_ip="10.250.0.2", namespace_path=Path("/var/run/netns/test-iflow"))
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow"),
            bundle_dir=harness.root / "bundle-iflow",
            status_port=8124,
            status_host=lease.guest_ip,
            last_status={},
        )
        task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=[]),
            extra_launch_metadata=Mock(return_value={}),
            wait_for_task_ready=Mock(),
        )

        with patch.object(harness, "_allocate_benchmark_network_lease", return_value=lease) as allocate_network:
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)) as prepare_handle:
                with patch.object(harness, "build_task_run", return_value=task_run), patch.object(
                    harness,
                    "ensure_sandbox_image",
                    return_value=sandbox_image,
                ):
                    result = harness.launch_sandbox("sbx-iflow", agent_type="iflow")

        self.assertIs(result, handle)
        allocate_network.assert_called_once_with(SandboxId("sbx-iflow"))
        prepare_handle.assert_called_once_with(
            "sbx-iflow",
            interceptor_host=harness.benchmark_bridge_ip,
            network_lease=lease,
            agent_type="iflow",
            llm_service_type="simulated_for_iflow",
        )
        self.assertEqual(harness._benchmark_ip_to_sandbox[lease.guest_ip], SandboxId("sbx-iflow"))

    def test_launch_sandbox_from_docker_compose_file_rejects_unsupported_features(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_file = root / "compose.yaml"
            compose_file.write_text(
                """
services:
  app:
    image: alpine
    depends_on:
      - db
""".strip(),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported compose features"):
                harness.launch_sandbox_from_docker_compose_file(
                    compose_file,
                    env_file,
                    sandbox_name="sbx-compose",
                    service_name="app",
                )

    def test_iflow_agent_uses_foreground_runc_exec(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow"),
            bundle_dir=Path("/tmp/sbx-iflow"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(
            sandbox,
            TaskDescription("do work"),
            TaskConfig(options={"FOO": "bar"}),
            runtime_state_root=Path("/tmp/runtime"),
        )
        with patch("integrations.agents.iflow.subprocess.run") as run:
            agent.perform_task()

        argv = run.call_args.args[0]
        self.assertEqual(argv[:4], ["runc", "--root", "/tmp/runtime", "exec"])
        self.assertNotIn("-d", argv)
        self.assertIn("sbx-iflow", argv)

    def test_iflow_configure_bundle_redirects_idle_init_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "cgroupsPath": ""},
                        "mounts": [],
                        "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": False},
                    }
                ),
                encoding="utf-8",
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-config"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": "/tmp/logs",
                    }
                },
            )
            agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig())

            agent.configure_bundle()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["process"]["args"][:2], ["/bin/sh", "-lc"])
            self.assertIn("exec >/dev/null 2>&1", payload["process"]["args"][2])

    def test_iflow_prepare_sandbox_uses_extended_benchmark_session_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-prepare"),
                bundle_dir=Path(tmp) / "bundle",
                status_port=8123,
                last_status={},
            )
            runtime = SimpleNamespace(
                root=Path(tmp) / "runtime",
                mounted_entrypoint="/opt/iflow-runtime/entry.js",
                ignore_process_rules=[],
            )
            state = SimpleNamespace(
                iflow_home=Path(tmp) / ".iflow",
                npm_home=Path(tmp) / ".npm",
                logs_dir=Path(tmp) / "logs",
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                agent_host_dir=Path(tmp) / "iflow" / "sbx-iflow-prepare",
                llm_base_url="http://10.250.9.1:4567/v1",
            )

            with patch("integrations.agents.iflow.prepare_iflow_runtime", return_value=runtime), patch(
                "integrations.agents.iflow.prepare_iflow_state",
                return_value=state,
            ) as prepare_state:
                agent.prepare_sandbox()

        self.assertEqual(prepare_state.call_args.kwargs["max_session_turns"], 4096)
        self.assertEqual(prepare_state.call_args.kwargs["base_url"], "http://10.250.9.1:4567/v1")

    def test_iflow_agent_exposes_synthetic_progress_status(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-progress"),
            bundle_dir=Path("/tmp/sbx-iflow-progress"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(
            sandbox,
            TaskDescription("do work"),
            TaskConfig(options={"action_tick_seconds": 0.01}),
            runtime_state_root=Path("/tmp/runtime"),
        )

        def _sleeping_run(*args, **kwargs):
            _ = (args, kwargs)
            time.sleep(0.2)
            return SimpleNamespace(returncode=0)

        with patch("integrations.agents.iflow.subprocess.run", side_effect=_sleeping_run):
            worker = threading.Thread(target=agent.perform_task)
            worker.start()
            payload = agent.wait_for_progress(minimum_actions=2)
            delta_payload = agent.wait_for_action_delta(delta=1)
            worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(int(payload["total_actions"]), 2)
        self.assertGreaterEqual(int(delta_payload["total_actions"]), int(payload["total_actions"]) + 1)
        self.assertEqual(sandbox.last_status, delta_payload)
        final_payload = agent.poll_status()
        self.assertEqual(final_payload["state"], "finished")
        self.assertGreaterEqual(int(final_payload["total_actions"]), int(delta_payload["total_actions"]))
        self.assertEqual(sandbox.last_status, delta_payload)
        self.assertEqual(len(agent._recorded_activity_events()), 6)
        self.assertEqual(
            {event.kind for event in agent._recorded_activity_events()},
            {EBPFEventKind.FILE_WRITE, EBPFEventKind.PROCESS_EXEC, EBPFEventKind.NETWORK_EGRESS},
        )

    def test_iflow_agent_uses_one_second_default_action_tick(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-default-tick"),
            bundle_dir=Path("/tmp/sbx-iflow-default-tick"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        self.assertEqual(agent._tick_seconds, 1.0)

    def test_iflow_agent_on_restore_complete_resumes_synthetic_progress(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-restore"),
            bundle_dir=Path("/tmp/sbx-iflow-restore"),
            status_port=8123,
            last_status={"total_actions": 4},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        agent._started_at_monotonic = time.monotonic() - 0.8
        agent._finished_at_monotonic = time.monotonic()

        baseline = agent._synthetic_action_count()
        agent.on_restore_complete()

        self.assertEqual(agent._task_state(), "running")
        self.assertGreaterEqual(agent._synthetic_action_count(), baseline)

    def test_iflow_poll_status_is_non_mutating_until_status_helpers_run(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-poll"),
            bundle_dir=Path("/tmp/sbx-iflow-poll"),
            status_port=8123,
            last_status={"total_actions": 1},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig())
        agent._started_at_monotonic = time.monotonic() - 0.05

        polled = agent.poll_status()

        self.assertGreaterEqual(int(polled["total_actions"]), 0)
        self.assertEqual(sandbox.last_status, {"total_actions": 1})
        agent._set_status(polled)
        self.assertEqual(sandbox.last_status, polled)

    def test_clone_checkpoint_to_fork_preserves_llm_base_url(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        source = SandboxHandle(
            sandbox_id=SandboxId("sbx-source"),
            bundle_dir=Path("/tmp/sbx-source"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_service_type="simulated_for_iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        target = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/sbx-fork"),
            status_port=8124,
            last_status={},
        )
        harness.root = Path("/tmp")
        harness.runtime = object()
        harness.sandbox_manager = SimpleNamespace(_items={}, _persist=Mock())
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.storage = SimpleNamespace(
            put_manifest=Mock(),
            get_artifact=Mock(),
            put_artifact=Mock(),
        )

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=source.sandbox_id,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()

        with patch.object(harness, "_agent_requires_benchmark_network", return_value=False), patch.object(
            harness, "_prepare_sandbox_handle", return_value=(target, None)
        ), patch.object(harness, "list_checkpoint_manifests", return_value=[manifest]), patch.object(
            harness, "build_task_run", return_value=Mock()
        ), patch("benchmarks.real_host_scenario_base.resolve_checkpoint_copy_plan", return_value=[(CheckpointId("ckpt-1"), False, True)]), patch(
            "benchmarks.real_host_scenario_base.subprocess.run"
        ):
            forked = harness.clone_checkpoint_to_fork(source, CheckpointId("ckpt-1"), "sbx-fork")

        self.assertIs(forked, target)
        self.assertEqual(target.llm_base_url, "http://10.250.0.1:43123/v1")

    def test_restore_once_notifies_task_run_after_successful_restore(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-restore"),
            bundle_dir=Path("/tmp/sbx-restore"),
            status_port=8123,
            last_status={},
            task_run=SimpleNamespace(on_restore_complete=Mock()),
        )
        harness.system = SimpleNamespace(
            restore_once=Mock(
                return_value=SimpleNamespace(
                    status=SimpleNamespace(value="succeeded"),
                    checkpoint_id=CheckpointId("ckpt-1"),
                )
            )
        )

        result = harness.restore_once(sandbox, CheckpointId("ckpt-1"))

        sandbox.task_run.on_restore_complete.assert_called_once()
        self.assertEqual(result.checkpoint_id, CheckpointId("ckpt-1"))

    def test_wait_for_recovery_notifies_task_run_after_successful_restore(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        observed_at = utc_now()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-recovery"),
            bundle_dir=Path("/tmp/sbx-recovery"),
            status_port=8123,
            last_status={},
            task_run=SimpleNamespace(on_restore_complete=Mock()),
        )
        harness.system = SimpleNamespace(
            get_last_recovery_record=Mock(
                return_value=SimpleNamespace(
                    event_type="fault",
                    started_at=observed_at,
                    status="restored",
                    checkpoint_id=CheckpointId("ckpt-1"),
                )
            )
        )

        record = harness.wait_for_recovery(
            sandbox,
            event_type="fault",
            observed_after=observed_at,
            timeout_s=0.1,
        )

        sandbox.task_run.on_restore_complete.assert_called_once()
        self.assertEqual(record.checkpoint_id, CheckpointId("ckpt-1"))

    def test_load_dataset_normalizes_relative_paths_and_cycles_rows(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "agent_type": "simulated",
                                "llm_service_type": "manual",
                                "task_description": "task-a",
                                "task_config": {"minimum_actions": 1},
                                "docker_compose_file": "compose.yaml",
                                "env_file": "task.env",
                            }
                        ),
                        json.dumps(
                            {
                                "agent_type": "iflow",
                                "task_description": {"prompt": "task-b"},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            records = harness.load_dataset(dataset_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].llm_service_type, "manual")
            self.assertEqual(records[0].docker_compose_file, (root / "compose.yaml").resolve())
            self.assertEqual(records[0].env_file, (root / "task.env").resolve())
            selected = harness.select_task_record(
                records,
                sandbox_index=3,
                default_agent_type="simulated",
                default_llm_service_type="simulated",
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )
            self.assertEqual(selected.agent_type, "iflow")
            self.assertEqual(selected.task_description.prompt, "task-b")
            self.assertIsNone(selected.llm_service_type)

    def test_resolve_llm_service_type_defaults_from_agent(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )

        self.assertEqual(
            harness.resolve_llm_service_type(agent_type="simulated", llm_service_type=None),
            "simulated",
        )
        self.assertEqual(
            harness.resolve_llm_service_type(agent_type="iflow", llm_service_type=None),
            "simulated_for_iflow",
        )

    def test_resolve_llm_service_type_rejects_anthropic_for_manual_only_services(self) -> None:
        harness = RealHostScenarioHarness(
            provider="anthropic",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )

        with self.assertRaisesRegex(ValueError, "only supports provider=openai"):
            harness.resolve_llm_service_type(agent_type="simulated", llm_service_type="manual")

    def test_launch_task_record_routes_dataset_compose_rows(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        record = BenchmarkTaskRecord(
            agent_type="simulated",
            task_description=TaskDescription("compose"),
            task_config=TaskConfig(),
            docker_compose_file=Path("/tmp/compose.yaml"),
            env_file=Path("/tmp/task.env"),
        )
        with patch.object(harness, "launch_sandbox_from_docker_compose_file", return_value="compose-handle") as launch_compose:
            result = harness.launch_task_record("sbx-compose", record)

        self.assertEqual(result, "compose-handle")
        launch_compose.assert_called_once()

    def test_e2e_benchmark_uses_shared_harness_launch_flow(self) -> None:
        task_run = SimpleNamespace(
            wait_for_progress=Mock(return_value={"total_actions": 6}),
        )
        harness = SimpleNamespace(
            load_dataset=lambda path: [],
            select_task_record=lambda *args, **kwargs: BenchmarkTaskRecord(
                agent_type="simulated",
                task_description=TaskDescription("task"),
                task_config=TaskConfig(),
            ),
            launch_task_record=Mock(
                side_effect=lambda name, record: SandboxHandle(
                    sandbox_id=SandboxId(name),
                    bundle_dir=Path("/tmp") / name,
                    status_port=8123,
                    last_status={
                        "total_actions": 6,
                        "filesystem_actions": 1,
                        "process_actions": 1,
                        "network_actions": 1,
                    },
                    task_run=task_run,
                )
            ),
            request_state_store=None,
            checkpoint_if_due=Mock(return_value=None),
            restore_once=Mock(),
            get_sandbox_handle=lambda sandbox_id: SandboxHandle(
                sandbox_id=SandboxId(sandbox_id),
                bundle_dir=Path("/tmp") / sandbox_id,
                status_port=8123,
                last_status={},
            ),
            set_snapshot_metadata=Mock(),
        )
        rows = run_benchmark(
            SimpleNamespace(
                sandboxes=1,
                iters=1,
                provider="openai",
                agent_type="simulated",
                llm_service_type="simulated",
                dataset=None,
            ),
            harness,
        )
        self.assertEqual(len(rows), 1)
        harness.launch_task_record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
