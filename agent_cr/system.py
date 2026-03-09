from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ExecutorConfig, PolicyConfig, SchedulerConfig, StorageConfig
from .executor import CRExecutor
from .ids import JobId
from .inspector import InMemorySandboxInspector
from .models import CheckpointJob, CheckpointResult, RestoreJob, RestoreResult, SandboxSnapshot, utc_now
from .policy import DefaultHeuristicPolicy
from .runtime import DockerRuntimeAdapter, RuncRuntimeAdapter
from .sandbox_manager import InMemorySandboxManager
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
    inspector: InMemorySandboxInspector
    sandbox_manager: InMemorySandboxManager
    telemetry: InMemoryTelemetrySink | NoopTelemetrySink

    def checkpoint_once(self, sandbox_id) -> CheckpointResult:
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

    def restore_once(self, sandbox_id, checkpoint_id) -> RestoreResult:
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
        )
        return self.executor.run_restore(job)


def build_default_system(
    *,
    storage_root: str | Path,
    runtime: str = "docker",
    scheduler_config: SchedulerConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    storage_config: StorageConfig | None = None,
    policy_config: PolicyConfig | None = None,
    use_in_memory_telemetry: bool = True,
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
    inspector = InMemorySandboxInspector()
    scheduler = CRScheduler(
        scheduler_cfg,
        DefaultHeuristicPolicy(policy_cfg),
        inspector,
        InMemorySchedulerStateStore(),
        telemetry,
    )
    sandbox_manager = InMemorySandboxManager()

    return AgentCRSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=inspector,
        sandbox_manager=sandbox_manager,
        telemetry=telemetry,
    )
