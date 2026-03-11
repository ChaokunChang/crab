from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import Event, Lock, Thread

from .config import ExecutorConfig, PolicyConfig, SchedulerConfig, StorageConfig
from .contracts import SandboxInspector
from .executor import CRExecutor
from .ids import JobId
from .inspector import EBPFSandboxInspector
from .interceptor import InMemoryRequestStateStore, RequestAwareSandboxInspector, SandboxResponseGateRegistry
from .models import CheckpointJob, CheckpointResult, RestoreJob, RestoreResult, SandboxId, utc_now
from .policy import DefaultHeuristicPolicy
from .runtime import DockerRuntimeAdapter, RuncRuntimeAdapter
from .sandbox_manager import InMemorySandboxManager, RuncSandboxManager
from .scheduler import CRScheduler, InMemorySchedulerStateStore
from .storage import LocalCheckpointManager
from .telemetry import InMemoryTelemetrySink, NoopTelemetrySink
from .workers import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    DefaultCWorker,
    DefaultRWorker,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentCRSystem:
    scheduler: CRScheduler
    executor: CRExecutor
    storage: LocalCheckpointManager
    inspector: SandboxInspector
    sandbox_manager: InMemorySandboxManager | RuncSandboxManager
    telemetry: InMemoryTelemetrySink | NoopTelemetrySink
    request_state_store: InMemoryRequestStateStore | None = None
    response_gate_registry: SandboxResponseGateRegistry | None = None
    _interceptor_lock: Lock = field(init=False, repr=False)
    _interceptor_pending: set[SandboxId] = field(init=False, repr=False)
    _coordination_lock: Lock = field(init=False, repr=False)
    _active_coordination: set[SandboxId] = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _monitor_thread: Thread | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.response_gate_registry is None:
            self.response_gate_registry = SandboxResponseGateRegistry()
        self._interceptor_lock = Lock()
        self._interceptor_pending = set()
        self._coordination_lock = Lock()
        self._active_coordination = set()
        self._stop_event = Event()

    def start(self) -> None:
        if self.request_state_store is None:
            return
        with self._coordination_lock:
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._stop_event.clear()
            if self.response_gate_registry is not None:
                self.response_gate_registry.enable()
            self._monitor_thread = Thread(target=self._run_monitor_loop, name="agent-cr-system", daemon=True)
            self._monitor_thread.start()
        logger.info("Started AgentCRSystem monitor loop")

    def stop(self) -> None:
        self._stop_event.set()
        if self.response_gate_registry is not None:
            self.response_gate_registry.disable()
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._monitor_thread = None
        logger.info("Stopped AgentCRSystem monitor loop")

    def checkpoint_once(self, sandbox_id: SandboxId) -> CheckpointResult:
        logger.info("Running manual checkpoint for sandbox %s", sandbox_id)
        paused = self._pause_for_manual_checkpoint(sandbox_id)
        result: CheckpointResult | None = None
        job: CheckpointJob | None = None
        try:
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
            )
            result = self.executor.run_checkpoint(job)
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(sandbox_id, result.finished_at)
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
        finally:
            if paused and self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)
            self._release_response_gate(sandbox_id)
        assert result is not None
        logger.info("Manual checkpoint for sandbox %s finished with status=%s", sandbox_id, result.status.value)
        return result

    def checkpoint_if_due(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        try:
            logger.debug("Checking whether sandbox %s is due for checkpoint", sandbox_id)
            result = self._execute_checkpoint_flow(sandbox_id)
            if result is None:
                logger.debug("Sandbox %s is not due for checkpoint", sandbox_id)
                return None
            logger.info("Checkpoint-if-due for sandbox %s finished with status=%s", sandbox_id, result.status.value)
            return result
        finally:
            with self._interceptor_lock:
                self._interceptor_pending.discard(sandbox_id)
            self._release_response_gate(sandbox_id)

    def checkpoint_due_sandboxes(self, sandbox_ids: list[SandboxId]) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        for sandbox_id in sandbox_ids:
            result = self.checkpoint_if_due(sandbox_id)
            if result is not None:
                results.append(result)
        return results

    def restore_once(self, sandbox_id: SandboxId, checkpoint_id) -> RestoreResult:
        logger.info("Running manual restore for sandbox %s checkpoint=%s", sandbox_id, checkpoint_id)
        self.sandbox_manager.prepare_for_restore(sandbox_id)
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
        )
        result = self.executor.run_restore(job)
        if result.status.value == "succeeded":
            self.sandbox_manager.mark_restored(sandbox_id)
        logger.info(
            "Manual restore for sandbox %s checkpoint=%s finished with status=%s",
            sandbox_id,
            checkpoint_id,
            result.status.value,
        )
        return result

    def notify_interceptor_state_change(self, sandbox_id: SandboxId) -> None:
        with self._interceptor_lock:
            self._interceptor_pending.add(sandbox_id)
            pending = sandbox_id in self._interceptor_pending
        logger.debug("Recorded interceptor state change for sandbox %s pending=%s", sandbox_id, pending)
        self.telemetry.emit_event(
            "interceptor.state_changed",
            {
                "sandbox_id": str(sandbox_id),
                "pending": pending,
            },
        )

    def has_pending_interceptor_signal(self, sandbox_id: SandboxId) -> bool:
        with self._interceptor_lock:
            return sandbox_id in self._interceptor_pending

    def _run_monitor_loop(self) -> None:
        assert self.request_state_store is not None
        while not self._stop_event.is_set():
            change = self.request_state_store.wait_for_change(timeout=0.5)
            if change is None or change.event_type != "request_start":
                continue
            self._dispatch_coordination(change.sandbox_id)

    def _dispatch_coordination(self, sandbox_id: SandboxId) -> None:
        with self._coordination_lock:
            if sandbox_id in self._active_coordination:
                return
            self._active_coordination.add(sandbox_id)
        worker = Thread(
            target=self._coordinate_sandbox_request,
            args=(sandbox_id,),
            name=f"agent-cr-coordinate-{sandbox_id}",
            daemon=True,
        )
        worker.start()

    def _coordinate_sandbox_request(self, sandbox_id: SandboxId) -> None:
        try:
            self._execute_checkpoint_flow(sandbox_id)
        except Exception:
            logger.exception("Checkpoint coordination failed for sandbox %s", sandbox_id)
            self._resume_sandbox(sandbox_id)
        finally:
            self._release_response_gate(sandbox_id)
            with self._coordination_lock:
                self._active_coordination.discard(sandbox_id)
            with self._interceptor_lock:
                self._interceptor_pending.discard(sandbox_id)

    def _execute_checkpoint_flow(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        decision = self.scheduler.query_checkpoint(sandbox_id)
        if not decision.should_checkpoint:
            return None

        job = CheckpointJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason=decision.reason,
            checkpoint_process=decision.checkpoint_process,
            checkpoint_filesystem=decision.checkpoint_filesystem,
            metadata={"policy": decision.policy_name, **decision.metadata},
        )
        result: CheckpointResult | None = None
        try:
            result = self.executor.submit_checkpoint(job).result()
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(sandbox_id, result.finished_at)
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
            return result
        finally:
            if self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)

    def _pause_for_manual_checkpoint(self, sandbox_id: SandboxId) -> bool:
        try:
            self.sandbox_manager.pause(sandbox_id)
            return True
        except Exception:
            logger.exception("Failed to pause sandbox %s for manual checkpoint", sandbox_id)
            return False

    def _resume_sandbox(self, sandbox_id: SandboxId) -> None:
        try:
            description = self.sandbox_manager.describe(sandbox_id)
        except Exception:
            return
        if description.status != "paused":
            return
        try:
            self.sandbox_manager.resume(sandbox_id)
        except Exception:
            logger.exception("Failed to resume sandbox %s", sandbox_id)

    def _release_response_gate(self, sandbox_id: SandboxId) -> None:
        if self.response_gate_registry is not None:
            self.response_gate_registry.release(sandbox_id)

    def _should_resume_after_checkpoint(
        self,
        job: CheckpointJob | None,
        result: CheckpointResult | None,
    ) -> bool:
        if job is None or not job.checkpoint_process:
            return True
        if result is None:
            return True
        for status in result.operation_statuses:
            if status.metadata.get("phase") == "process_checkpoint" and bool(status.executed):
                return False
        return True


def build_default_system(
    *,
    storage_root: str | Path,
    runtime: str = "runc",
    scheduler_config: SchedulerConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    storage_config: StorageConfig | None = None,
    policy_config: PolicyConfig | None = None,
    use_in_memory_telemetry: bool = True,
    request_state_store: InMemoryRequestStateStore | None = None,
) -> AgentCRSystem:
    logger.info("Building default agent-cr system with runtime=%s storage_root=%s", runtime, storage_root)
    scheduler_cfg = scheduler_config or SchedulerConfig()
    executor_cfg = executor_config or ExecutorConfig()
    store_cfg = storage_config or StorageConfig(root_dir=Path(storage_root))
    policy_cfg = policy_config or PolicyConfig()

    if runtime == "docker":
        adapter = DockerRuntimeAdapter()
        sandbox_manager = InMemorySandboxManager()
    elif runtime == "runc":
        adapter = RuncRuntimeAdapter()
        sandbox_manager = RuncSandboxManager()
    else:
        raise ValueError(f"unsupported runtime adapter: {runtime}")

    telemetry = InMemoryTelemetrySink() if use_in_memory_telemetry else NoopTelemetrySink()
    storage = LocalCheckpointManager(store_cfg)

    process_c = AdapterProcessCWorker(adapter)
    process_r = AdapterProcessRWorker(adapter)
    fs_c = AdapterFileSystemCWorker(adapter)
    fs_r = AdapterFileSystemRWorker(adapter)

    c_worker = DefaultCWorker(process_c, fs_c, storage, adapter)
    r_worker = DefaultRWorker(process_r, fs_r, storage)

    executor = CRExecutor(executor_cfg, c_worker, r_worker, telemetry)
    request_store = request_state_store or InMemoryRequestStateStore()
    response_gate_registry = SandboxResponseGateRegistry()
    inspector = RequestAwareSandboxInspector(EBPFSandboxInspector(), request_store)
    scheduler = CRScheduler(
        scheduler_cfg,
        DefaultHeuristicPolicy(policy_cfg),
        inspector,
        sandbox_manager,
        InMemorySchedulerStateStore(),
        telemetry,
    )

    logger.debug(
        "Constructed agent-cr components runtime=%s telemetry=%s",
        runtime,
        type(telemetry).__name__,
    )
    return AgentCRSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=inspector,
        sandbox_manager=sandbox_manager,
        telemetry=telemetry,
        request_state_store=request_store,
        response_gate_registry=response_gate_registry,
    )
