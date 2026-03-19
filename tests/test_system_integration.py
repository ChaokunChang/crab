from __future__ import annotations

import json
import threading
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
    DeleteAfterRestoreCheckpointManager,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    ExecutorConfig,
    FailureCode,
    FaultToleranceCheckpointingPolicy,
    CheckpointJob,
    JobId,
    InMemoryEBPFEventCollector,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RequestContext,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    SpotPreemptionCheckpointingPolicy,
    StorageConfig,
)
from agent_cr.models import JobStatus, RecoveryEvent, utc_now
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request
from agent_cr.runtime import CommandRunner

RuncRuntimeAdapter = RuncRuntime
RuncSandboxManager = RuncRuntime
RuncSandboxManagerPaths = RuncRuntimePaths


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


class FailingRestoreRunner(FakeCommandRunner):
    def run(self, command: list[str], *, cwd: Path | None = None):
        result = super().run(command, cwd=cwd)
        if "restore" in command:
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 1, "stdout": "", "stderr": "restore failed"},
            )()
        return result


class SystemIntegrationTests(unittest.TestCase):
    def _wait_for_record(self, system: AgentCRSystem, sandbox_id: SandboxId, expected_event_type: str):
        import time

        deadline = time.time() + 5.0
        while time.time() < deadline:
            record = system.get_last_recovery_record(sandbox_id)
            if record is not None and record.event_type == expected_event_type:
                return record
            time.sleep(0.05)
        self.fail(f"timed out waiting for recovery record {expected_event_type} for {sandbox_id}")

    def _build_runc_system(
        self,
        *,
        root: Path,
        runner: CommandRunner,
        telemetry: InMemoryTelemetrySink,
        inspector: EBPFSandboxInspector,
        request_store: InMemoryRequestStateStore | None = None,
        relaunch_handler=None,
        enforce_restore_checkpoint_validation: bool = False,
    ) -> tuple[AgentCRSystem, CRExecutor]:
        runtime = RuncRuntimeAdapter(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                zfs_dataset_prefix="pool/agent-cr",
            ),
        )
        storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
            ),
            telemetry,
        )
        sandbox_manager = RuncSandboxManager(
            command_runner=runner,
            paths=RuncSandboxManagerPaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/agent-cr",
            ),
        )
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            inspector,
            sandbox_manager,
            InMemorySchedulerStateStore(),
            telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        system = AgentCRSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=sandbox_manager,
            telemetry=telemetry,
            request_state_store=request_store,
            relaunch_handler=relaunch_handler,
            enforce_restore_checkpoint_validation=enforce_restore_checkpoint_validation,
        )
        return system, executor

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
            storage_root = root / "storage"
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
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
            sandbox_manager = RuncSandboxManager(command_runner=runner, paths=sandbox_paths)
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
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
            self.assertGreaterEqual(len(checkpoint_result.manifest.process_artifacts), 1)
            self.assertTrue(len(checkpoint_result.manifest.filesystem_artifacts) >= 1)

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)

            description = system.sandbox_manager.describe(sandbox_id)
            self.assertEqual(description.status, "running")

            system.sandbox_manager.stop(sandbox_id)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            system.sandbox_manager.delete(sandbox_id)

            self.assertEqual(
                runner.commands[:4],
                [
                    ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-int' / 'rootfs'}", "pool/agent-cr/sbx-int"),
                    (
                        "runc",
                        "--root",
                        str(root / "runtime-state"),
                        "create",
                        "--bundle",
                        str(root / "bundles" / "sbx-int"),
                        "sbx-int",
                    ),
                    ("runc", "--root", str(root / "runtime-state"), "start", "sbx-int"),
                    ("runc", "--root", str(root / "runtime-state"), "pause", "sbx-int"),
                ],
            )
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "checkpoint",
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--leave-running=false",
                    "--tcp-established",
                    "--shell-job",
                    "--tcp-skip-in-flight",
                    "sbx-int",
                ),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "snapshot", f"pool/agent-cr/sbx-int@{checkpoint_result.checkpoint_id}"),
                runner.commands,
            )
            self.assertIn(
                ("runc", "--root", str(root / "runtime-state"), "delete", "-f", "sbx-int"),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "rollback", "-r", f"pool/agent-cr/sbx-int@{checkpoint_result.checkpoint_id}"),
                runner.commands,
            )
            self.assertIn(
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
                    "--shell-job",
                    "sbx-int",
                ),
                runner.commands,
            )
            self.assertEqual(
                runner.commands[-3:],
                [
                    (
                        "runc",
                        "--root",
                        str(root / "runtime-state"),
                        "kill",
                        "sbx-int",
                        "TERM",
                    ),
                    ("runc", "--root", str(root / "runtime-state"), "delete", "-f", "sbx-int"),
                    ("zfs", "destroy", "-r", "pool/agent-cr/sbx-int"),
                ],
            )

            event_names = [name for name, _ in telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            executor.shutdown()

    def test_notify_fault_marks_sandbox_not_running_before_recovery_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
            )

            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-faulted",
                    "bundle_path": str(root / "bundles" / "sbx-faulted"),
                },
            )
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

            system.notify_fault(sandbox_id)

            self.assertFalse(inspector.inspect(sandbox_id).is_running)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            executor.shutdown()

    def test_fault_tolerance_policy_resumes_sandbox_after_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(
                    SchedulerConfig(
                        min_checkpoint_interval_seconds=0.0,
                        force_checkpoint_after_seconds=0.0,
                        require_change_signal=True,
                    )
                ),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-ft",
                    "bundle_path": str(root / "bundles" / "sbx-ft"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            result = system.checkpoint_if_due(sandbox_id)

            self.assertIsNotNone(result)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "running")
            self.assertTrue(any("--leave-running=true" in command for command in runner.commands))
            executor.shutdown()

    def test_fault_notification_restores_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_store,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-fault",
                    "bundle_path": str(root / "bundles" / "sbx-auto-fault"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.request_state_store = None
            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-auto-fault"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-auto-fault" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-auto-fault" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "sbx-auto-fault",
                ),
                runner.commands,
            )

    def test_restore_releases_buffered_response_for_matching_live_request_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-live-restore",
                    "bundle_path": str(root / "bundles" / "sbx-live-restore"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()

            response_payload: dict[str, object] = {}
            request_finished = threading.Event()

            def _run_intercept() -> None:
                from agent_cr import AgentCRRequestInterceptor

                interceptor = AgentCRRequestInterceptor(
                    upstream_transport=lambda path, headers, body: (
                        200,
                        [("Content-Type", "application/json")],
                        json.dumps(
                            handle_request(
                                path=path,
                                headers=headers,
                                payload=json.loads(body.decode("utf-8")),
                                state=SimulatedLLMState(),
                            ),
                            sort_keys=True,
                        ).encode("utf-8"),
                    ),
                    request_state_store=request_store,
                    response_gate_registry=system.response_gate_registry,
                )
                _, _, body = interceptor.intercept(
                    path="/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-Agent-Sandbox-Id": str(sandbox_id),
                        "X-Request-Id": "req-live",
                    },
                    body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode("utf-8"),
                )
                response_payload["body"] = json.loads(body.decode("utf-8"))
                request_finished.set()

            thread = threading.Thread(target=_run_intercept)
            thread.start()
            pending = None
            for _ in range(100):
                pending = system.response_gate_registry.get_pending(sandbox_id)
                if pending is not None:
                    break
                request_finished.wait(0.01)
            self.assertIsNotNone(pending)
            assert pending is not None

            checkpoint_job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=True,
                metadata=system._build_checkpoint_metadata(sandbox_id),
            )
            checkpoint_result = executor.run_checkpoint(checkpoint_job)
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)
            self.assertTrue(system._release_checkpoint_response_gate(sandbox_id, checkpoint_result.checkpoint_id))

            thread.join(timeout=2.0)
            self.assertTrue(request_finished.is_set())
            self.assertIn("body", response_payload)
            self.assertEqual(request_store.get(sandbox_id).completed_llm_requests, 1)
            executor.shutdown()

    def test_restore_validation_can_reject_invalid_live_request_checkpoint_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                enforce_restore_checkpoint_validation=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-restore-validate",
                    "bundle_path": str(root / "bundles" / "sbx-restore-validate"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            checkpoint_job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=True,
                metadata={
                    "captures_inflight_llm": True,
                    "captured_request_id": "req-missing",
                },
            )
            checkpoint_result = executor.run_checkpoint(checkpoint_job)
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)

            self.assertEqual(restore_result.status, JobStatus.FAILED)
            self.assertEqual(restore_result.failure_code, FailureCode.VALIDATION_ERROR)
            self.assertIn("no matching interceptor-held request is pending", restore_result.message)
            self.assertFalse(
                any(command[3] == "restore" for command in runner.commands if len(command) > 3),
            )
            executor.shutdown()

    def test_fault_notification_skips_stale_live_request_checkpoint_for_older_safe_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                enforce_restore_checkpoint_validation=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-stale-skip",
                    "bundle_path": str(root / "bundles" / "sbx-stale-skip"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            safe_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(safe_checkpoint)
            assert safe_checkpoint is not None

            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()
            context = RequestContext(
                request_id="req-stale",
                sandbox_id=sandbox_id,
                started_at=utc_now(),
                metadata={"provider": "openai"},
            )
            request_store.mark_request_start(context)
            generation = system.response_gate_registry.arm(sandbox_id, context.request_id)
            self.assertIsNotNone(generation)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            stale_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(stale_checkpoint)
            assert stale_checkpoint is not None
            self.assertNotEqual(safe_checkpoint.checkpoint_id, stale_checkpoint.checkpoint_id)
            request_store.mark_request_end(context)

            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.request_state_store = None
            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertEqual(record.checkpoint_id, safe_checkpoint.checkpoint_id)
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-stale-skip"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-stale-skip" / str(safe_checkpoint.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-stale-skip" / str(safe_checkpoint.checkpoint_id) / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "sbx-stale-skip",
                ),
                runner.commands,
            )

    def test_fault_notification_relaunches_when_no_satisfiable_live_request_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            relaunched: list[tuple[SandboxId, str]] = []
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                relaunch_handler=lambda sandbox_id, event_type: relaunched.append((sandbox_id, event_type)),
                enforce_restore_checkpoint_validation=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-no-live-match",
                    "bundle_path": str(root / "bundles" / "sbx-no-live-match"),
                },
            )
            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()
            context = RequestContext(
                request_id="req-stale-only",
                sandbox_id=sandbox_id,
                started_at=utc_now(),
                metadata={"provider": "openai"},
            )
            request_store.mark_request_start(context)
            system.response_gate_registry.arm(sandbox_id, context.request_id)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            stale_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(stale_checkpoint)
            request_store.mark_request_end(context)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "relaunched")
            self.assertEqual(relaunched, [(sandbox_id, "fault")])
            self.assertFalse(any(command[3] == "restore" for command in runner.commands if len(command) > 3))

    def test_fault_notification_relaunches_when_restore_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FailingRestoreRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            relaunched: list[tuple[SandboxId, str]] = []

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_store,
                relaunch_handler=lambda sandbox_id, event_type: relaunched.append((sandbox_id, event_type)),
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-fallback",
                    "bundle_path": str(root / "bundles" / "sbx-auto-fallback"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "relaunched")
            self.assertEqual(relaunched, [(sandbox_id, "fault")])

    def test_preemption_notification_checkpoints_then_restores(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-spot",
                    "bundle_path": str(root / "bundles" / "sbx-auto-spot"),
                },
            )
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

            system.start()
            try:
                system.notify_preemption(sandbox_id, grace_remaining_seconds=30.0)
                record = self._wait_for_record(system, sandbox_id, "preemption")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertTrue(any("--leave-running=false" in command for command in runner.commands))
            self.assertTrue(any(command[3] == "restore" for command in runner.commands if len(command) > 3))

    def test_preemption_reuses_recent_checkpoint_without_duplicate_pause(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-recent-spot",
                    "bundle_path": str(root / "bundles" / "sbx-recent-spot"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                    metadata={
                        "preemption_notice": True,
                        "preemption_grace_remaining_seconds": 30.0,
                    },
                )
            )

            observed_at = utc_now()
            checkpoint_result = system.checkpoint_once(sandbox_id, leave_running=False)
            self.assertEqual(checkpoint_result.status.value, "succeeded")

            try:
                system._handle_recovery_event(
                    RecoveryEvent(
                        sandbox_id=sandbox_id,
                        event_type="preemption",
                        observed_at=observed_at,
                        grace_remaining_seconds=30.0,
                        reason="preemption",
                    )
                )
            finally:
                executor.shutdown()

            record = system.get_last_recovery_record(sandbox_id)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.status, "restored")
            pause_commands = [command for command in runner.commands if len(command) > 3 and command[3] == "pause"]
            restore_commands = [command for command in runner.commands if len(command) > 3 and command[3] == "restore"]
            self.assertEqual(len(pause_commands), 1)
            self.assertEqual(len(restore_commands), 1)

    def test_preemption_notification_restores_with_delete_after_restore_storage_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            storage = DeleteAfterRestoreCheckpointManager(
                LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            )
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-spot-prune",
                    "bundle_path": str(root / "bundles" / "sbx-auto-spot-prune"),
                },
            )
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

            system.start()
            try:
                system.notify_preemption(sandbox_id, grace_remaining_seconds=30.0)
                record = self._wait_for_record(system, sandbox_id, "preemption")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertEqual(storage.list_checkpoints(sandbox_id), [])
            self.assertTrue(any(command[3] == "restore" for command in runner.commands if len(command) > 3))


if __name__ == "__main__":
    unittest.main()
