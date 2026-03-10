from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread

from .config import ExecutorConfig, PolicyConfig, SchedulerConfig, StorageConfig
from .contracts import SandboxInspector
from .executor import CRExecutor
from .ids import JobId
from .inspector import EBPFSandboxInspector
from .interceptor import InMemoryRequestStateStore, RequestAwareSandboxInspector
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


@dataclass
class AgentCRSystem:
    scheduler: CRScheduler
    executor: CRExecutor
    storage: LocalCheckpointManager
    inspector: SandboxInspector
    sandbox_manager: InMemorySandboxManager | RuncSandboxManager
    telemetry: InMemoryTelemetrySink | NoopTelemetrySink
    request_state_store: InMemoryRequestStateStore | None = None
    _interceptor_lock: Lock = field(init=False, repr=False)
    _interceptor_pending: set[SandboxId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._interceptor_lock = Lock()
        self._interceptor_pending = set()

    def checkpoint_once(self, sandbox_id: SandboxId) -> CheckpointResult:
        job = CheckpointJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason="manual",
        )
        result = self.executor.run_checkpoint(job)
        if result.status.value == "succeeded":
            self.scheduler.mark_checkpoint_complete(sandbox_id, result.finished_at)
        return result

    def checkpoint_if_due(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        job = self.scheduler.poll_and_schedule(sandbox_id)
        if job is None:
            return None
        result = self.executor.run_checkpoint(job)
        if result.status.value == "succeeded":
            self.scheduler.mark_checkpoint_complete(sandbox_id, result.finished_at)
        return result

    def checkpoint_due_sandboxes(self, sandbox_ids: list[SandboxId]) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        for sandbox_id in sandbox_ids:
            result = self.checkpoint_if_due(sandbox_id)
            if result is not None:
                results.append(result)
        return results

    def restore_once(self, sandbox_id: SandboxId, checkpoint_id) -> RestoreResult:
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
        )
        return self.executor.run_restore(job)

    def notify_interceptor_state_change(self, sandbox_id: SandboxId) -> None:
        with self._interceptor_lock:
            if sandbox_id in self._interceptor_pending:
                return
            self._interceptor_pending.add(sandbox_id)
        Thread(target=self._checkpoint_from_interceptor, args=(sandbox_id,), daemon=True).start()

    def _checkpoint_from_interceptor(self, sandbox_id: SandboxId) -> None:
        try:
            result = self.checkpoint_if_due(sandbox_id)
            self.telemetry.emit_event(
                "interceptor.scheduler_notified",
                {
                    "sandbox_id": str(sandbox_id),
                    "scheduled": result is not None,
                    "status": "" if result is None else result.status.value,
                },
            )
        except Exception as exc:
            self.telemetry.emit_event(
                "interceptor.scheduler_error",
                {
                    "sandbox_id": str(sandbox_id),
                    "error": str(exc),
                },
            )
        finally:
            with self._interceptor_lock:
                self._interceptor_pending.discard(sandbox_id)


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
    scheduler_cfg = scheduler_config or SchedulerConfig()
    executor_cfg = executor_config or ExecutorConfig()
    store_cfg = storage_config or StorageConfig(root_dir=Path(storage_root))
    policy_cfg = policy_config or PolicyConfig()

    if runtime == "docker":
        adapter = DockerRuntimeAdapter()
    elif runtime == "runc":
        adapter = RuncRuntimeAdapter()
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
    inspector = RequestAwareSandboxInspector(EBPFSandboxInspector(), request_store)
    scheduler = CRScheduler(
        scheduler_cfg,
        DefaultHeuristicPolicy(policy_cfg),
        inspector,
        InMemorySchedulerStateStore(),
        telemetry,
    )
    sandbox_manager = InMemorySandboxManager() if runtime == "docker" else RuncSandboxManager()

    return AgentCRSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=inspector,
        sandbox_manager=sandbox_manager,
        telemetry=telemetry,
        request_state_store=request_store,
    )
