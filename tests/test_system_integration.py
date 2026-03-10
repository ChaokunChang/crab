from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import (
    AgentCRSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    DefaultCWorker,
    DefaultHeuristicPolicy,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    ExecutorConfig,
    InMemoryEBPFEventCollector,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    PolicyConfig,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
)
from agent_cr.models import JobStatus, utc_now
from agent_cr.runtime import CommandRunner


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class SystemIntegrationTests(unittest.TestCase):
    def test_runc_system_checkpoint_restore_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime_paths = RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                zfs_dataset_prefix="pool/agent-cr",
            )
            sandbox_paths = RuncSandboxManagerPaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/agent-cr",
            )

            runtime = RuncRuntimeAdapter(command_runner=runner, paths=runtime_paths)
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            checkpoint_worker = DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
            )
            restore_worker = DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
            )
            executor = CRExecutor(ExecutorConfig(max_workers=1), checkpoint_worker, restore_worker, telemetry)
            scheduler = CRScheduler(
                SchedulerConfig(),
                DefaultHeuristicPolicy(
                    PolicyConfig(
                        min_checkpoint_interval_seconds=0.0,
                        force_checkpoint_after_seconds=0.0,
                        require_change_signal=True,
                    )
                ),
                inspector,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(command_runner=runner, paths=sandbox_paths)
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                sandbox_manager=sandbox_manager,
                telemetry=telemetry,
            )

            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-int",
                    "bundle_path": str(root / "bundles" / "sbx-int"),
                },
            )
            self.assertEqual(sandbox_id, SandboxId("sbx-int"))

            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
            collector.record(
                EBPFEvent(
                    sandbox_id=sandbox_id,
                    kind=EBPFEventKind.FILE_WRITE,
                    observed_at=utc_now(),
                    metadata={"path": "/workspace/file.txt"},
                )
            )

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            self.assertIsNotNone(checkpoint_result.manifest)
            self.assertTrue(len(checkpoint_result.manifest.process_artifacts) >= 1)
            self.assertTrue(len(checkpoint_result.manifest.filesystem_artifacts) >= 1)

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)

            description = system.sandbox_manager.describe(sandbox_id)
            self.assertEqual(description.status, "running")

            system.sandbox_manager.stop(sandbox_id)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            system.sandbox_manager.delete(sandbox_id)

            expected_commands = [
                ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-int' / 'rootfs'}", "pool/agent-cr/sbx-int"),
                ("runc", "--root", str(root / "runtime-state"), "run", "-d", "--bundle", str(root / "bundles" / "sbx-int"), "sbx-int"),
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "checkpoint",
                    "sbx-int",
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--leave-running=false",
                    "--tcp-established",
                ),
                ("zfs", "snapshot", f"pool/agent-cr/sbx-int@{checkpoint_result.checkpoint_id}"),
                ("zfs", "rollback", "-r", f"pool/agent-cr/sbx-int@{checkpoint_result.checkpoint_id}"),
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-int"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--tcp-established",
                    "sbx-int",
                ),
                ("runc", "--root", str(root / "runtime-state"), "kill", "sbx-int", "TERM"),
                ("runc", "--root", str(root / "runtime-state"), "delete", "-f", "sbx-int"),
                ("zfs", "destroy", "-r", "pool/agent-cr/sbx-int"),
            ]
            self.assertEqual(runner.commands, expected_commands)

            event_names = [name for name, _ in telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            executor.shutdown()


if __name__ == "__main__":
    unittest.main()
