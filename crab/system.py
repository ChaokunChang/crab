from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable

from .config import ExecutorConfig, SchedulerConfig, StorageConfig, TelemetryConfig
from .contracts import CheckpointManager, Runtime, SandboxInspector, TelemetrySink
from .executor import CRExecutor
from .ids import CheckpointId, JobId
from .inspector import EBPFSandboxInspector
from .remote_inspector import HostInspectorServiceClient, RemoteSandboxInspector
from .interceptor import InMemoryRequestStateStore, RequestAwareSandboxInspector, SandboxResponseGateRegistry
from .models import (
    ArtifactPayload,
    ChangesetEntry,
    ChangesetResult,
    CheckpointManifest,
    CheckpointJob,
    CheckpointResult,
    FailureCode,
    JobStatus,
    RecoveryEvent,
    RecoveryRecord,
    RestoreJob,
    RestoreResult,
    SandboxId,
    SandboxSnapshot,
    utc_now,
)
from . import forking
from .journal import ActionJournal
from .txn import (
    TxnAbortError,
    TxnAbortResult,
    TxnActiveError,
    TxnCommitResult,
    TxnDescription,
    TxnError,
    TxnMismatchError,
    new_txn_id,
)
from .runtime import InMemoryRuntime, RuncRuntime, RuncRuntimeOptions
from .scheduler import CRScheduler, InMemorySchedulerStateStore, SchedulerPolicy
from .storage import LocalCheckpointManager
from .telemetry import (
    CompositeTelemetrySink,
    ConfiguredTelemetrySink,
    NoopTelemetrySink,
    build_configured_telemetry_sink,
    start_operation,
)
from .workers import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    DefaultCWorker,
    DefaultRWorker,
)
from .workers.composite import resolve_restore_manifest

logger = logging.getLogger(__name__)

_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"
_CAPTURED_REQUEST_ID = "captured_request_id"
_CAPTURED_REQUEST_GENERATION = "captured_request_generation"
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"
_CAPTURED_REQUEST_GROUP_KIND = "captured_request_group_kind"
_CAPTURED_REQUEST_GROUP_ID = "captured_request_group_id"
_CAPTURED_REQUEST_IDS = "captured_request_ids"
_CAPTURED_REQUEST_GROUP_STARTED_AT = "captured_request_group_started_at"
_RESTORE_RUNTIME_READY_ATTEMPTS = 10
_RESTORE_RUNTIME_READY_DELAY_S = 0.1


def _checkpoint_scope(job: CheckpointJob) -> str:
    if job.checkpoint_process and job.checkpoint_filesystem:
        return "full"
    if job.checkpoint_process:
        return "process_only"
    if job.checkpoint_filesystem:
        return "filesystem_only"
    return "none"


def _checkpoint_guard_from_inspector(inspector: SandboxInspector) -> Callable[[CheckpointJob], tuple[bool, str | None]]:
    def guard(job: CheckpointJob) -> tuple[bool, str | None]:
        try:
            snapshot = inspector.inspect(job.sandbox_id)
        except Exception:
            return True, None
        if snapshot.is_running:
            return True, None
        return False, "sandbox_not_running"

    return guard


@dataclass
class CrabSystem:
    scheduler: CRScheduler
    executor: CRExecutor
    storage: CheckpointManager
    inspector: SandboxInspector
    runtime: Runtime
    telemetry: TelemetrySink
    request_state_store: InMemoryRequestStateStore | None = None
    response_gate_registry: SandboxResponseGateRegistry | None = None
    journal: ActionJournal | None = None
    relaunch_handler: Callable[[SandboxId, str, bool], None] | None = None
    extra_checkpoint_metadata_provider: Callable[[SandboxId], dict[str, object]] | None = None
    restore_metadata_handler: Callable[[SandboxId, CheckpointManifest], None] | None = None
    recovery_delay_seconds: float = 0.0
    enforce_restore_checkpoint_validation: bool = False
    # When False (the default), a restore failure during recovery surfaces as a
    # hard error rather than silently falling back to `relaunch_handler`. The
    # relaunch path is intended as an availability backstop, but it tends to
    # mask real bugs (corrupt checkpoints, broken restore plumbing) — especially
    # for callers that set `checkpoint_full_baseline_on_first_checkpoint=true`
    # and therefore expect every recovery to use a complete checkpoint. Set to
    # True to opt back into relaunch on restore failure.
    relaunch_on_restore_failure: bool = False
    _interceptor_lock: Lock = field(init=False, repr=False)
    _interceptor_pending: set[SandboxId] = field(init=False, repr=False)
    _coordination_lock: Lock = field(init=False, repr=False)
    _active_coordination: set[SandboxId] = field(init=False, repr=False)
    _recovery_lock: Lock = field(init=False, repr=False)
    _recovery_queue: Queue[RecoveryEvent | None] = field(init=False, repr=False)
    _recovery_records: dict[SandboxId, RecoveryRecord] = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _monitor_thread: Thread | None = field(init=False, repr=False, default=None)
    _recovery_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)
    _recovery_futures: list[Future[None]] = field(init=False, repr=False, default_factory=list)
    _recovery_worker_count: int = field(init=False, repr=False, default=0)
    _coordination_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)
    _fork_lock: Lock = field(init=False, repr=False)
    _fork_chain_pins: dict[SandboxId, tuple[SandboxId, CheckpointId]] = field(init=False, repr=False)
    _fork_children: dict[SandboxId, set[SandboxId]] = field(init=False, repr=False)
    _txn_lock: Lock = field(init=False, repr=False)
    _active_txns: dict[SandboxId, TxnDescription | None] = field(init=False, repr=False)

    @property
    def sandbox_manager(self) -> Runtime:
        return self.runtime

    @sandbox_manager.setter
    def sandbox_manager(self, value: Runtime) -> None:
        self.runtime = value

    def __post_init__(self) -> None:
        if self.response_gate_registry is None:
            self.response_gate_registry = SandboxResponseGateRegistry()
        self._interceptor_lock = Lock()
        self._interceptor_pending = set()
        self._coordination_lock = Lock()
        self._active_coordination = set()
        self._recovery_lock = Lock()
        self._recovery_queue = Queue()
        self._recovery_records = {}
        self._stop_event = Event()
        self._fork_lock = Lock()
        self._fork_chain_pins = {}
        self._fork_children = {}
        self._txn_lock = Lock()
        self._active_txns = {}

    def start(self) -> None:
        with self._coordination_lock:
            request_running = self._monitor_thread is not None and self._monitor_thread.is_alive()
            recovery_running = self._recovery_pool is not None and any(
                not future.done() for future in self._recovery_futures
            )
            if request_running and recovery_running:
                return
            self._stop_event.clear()
            if self.response_gate_registry is not None:
                self.response_gate_registry.enable()
            if self._coordination_pool is None:
                self._coordination_pool = ThreadPoolExecutor(
                    max_workers=self.executor.config.resolved_coordination_workers,
                    thread_name_prefix="crab-coordinate",
                )
            if self.request_state_store is not None and not request_running:
                self._monitor_thread = Thread(target=self._run_monitor_loop, name="crab-system", daemon=True)
                self._monitor_thread.start()
            if not recovery_running:
                recovery_workers = self.executor.config.resolved_restore_workers
                self._recovery_worker_count = recovery_workers
                self._recovery_pool = ThreadPoolExecutor(
                    max_workers=recovery_workers,
                    thread_name_prefix="crab-recovery",
                )
                self._recovery_futures = [
                    self._recovery_pool.submit(self._run_recovery_loop)
                    for _ in range(recovery_workers)
                ]
        logger.info("Started CrabSystem background loops")

    def stop(self) -> None:
        self._stop_event.set()
        if self.response_gate_registry is not None:
            self.response_gate_registry.disable()
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        for _ in range(self._recovery_worker_count):
            self._recovery_queue.put(None)
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._monitor_thread = None
        recovery_pool = self._recovery_pool
        self._recovery_pool = None
        self._recovery_futures = []
        self._recovery_worker_count = 0
        if recovery_pool is not None:
            recovery_pool.shutdown(wait=True, cancel_futures=False)
        coordination_pool = self._coordination_pool
        self._coordination_pool = None
        if coordination_pool is not None:
            coordination_pool.shutdown(wait=True, cancel_futures=False)
        self.telemetry.flush()
        logger.info("Stopped CrabSystem background loops")

    def _telemetry_attrs(
        self,
        sandbox_id: SandboxId,
        *,
        component: str,
        checkpoint_id: CheckpointId | None = None,
        job_id: JobId | None = None,
        event_type: str | None = None,
        request_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "component": component,
            "sandbox_id": str(sandbox_id),
        }
        if checkpoint_id is not None:
            attributes["checkpoint_id"] = str(checkpoint_id)
        if job_id is not None:
            attributes["job_id"] = str(job_id)
        if event_type is not None:
            attributes["event_type"] = event_type
        if request_id is not None:
            attributes["request_id"] = request_id
        if extra:
            attributes.update(extra)
        return attributes

    def _journal_lifecycle(
        self,
        sandbox_id: SandboxId,
        event: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Best-effort lifecycle marker in the action journal (B1)."""
        journal = self.journal
        if journal is None:
            return
        try:
            journal.record_lifecycle(sandbox_id, event, metadata=metadata)
        except Exception:
            logger.exception(
                "Journal lifecycle record failed sandbox=%s event=%s", sandbox_id, event
            )

    def checkpoint_once(self, sandbox_id: SandboxId, leave_running: bool=False) -> CheckpointResult:
        logger.info("Running manual checkpoint for sandbox %s", sandbox_id)
        operation = start_operation(
            self.telemetry,
            "checkpoint.flow",
            self._telemetry_attrs(sandbox_id, component="system", extra={"reason": "manual"}),
        )
        pending_request = self._next_pending_live_request(sandbox_id)
        paused = self._pause_for_manual_checkpoint(sandbox_id)
        result: CheckpointResult | None = None
        job: CheckpointJob | None = None
        try:
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=leave_running,
                metadata=self._build_checkpoint_metadata(sandbox_id, pending_request=pending_request),
            )
            result = self.executor.run_checkpoint(job)
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": "manual",
                        "leave_running": bool(leave_running),
                    },
                )
        finally:
            if paused and self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)
            self._release_response_gate(sandbox_id, pending_request)
            self._refresh_interceptor_pending_state(sandbox_id)
        finish_attrs: dict[str, object] = {}
        if result is not None:
            finish_attrs["checkpoint_id"] = str(result.checkpoint_id)
            finish_attrs["failure_code"] = result.failure_code.value
        if job is not None:
            finish_attrs["job_id"] = str(job.job_id)
            finish_attrs["checkpoint_scope"] = _checkpoint_scope(job)
        operation.finish(
            status="failed" if result is None else result.status.value,
            attributes=finish_attrs,
        )
        assert result is not None
        logger.info("Manual checkpoint for sandbox %s finished with status=%s", sandbox_id, result.status.value)
        return result

    def checkpoint_if_due(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        if self._txn_active(sandbox_id):
            logger.debug("Skipping checkpoint-if-due; txn active sandbox=%s", sandbox_id)
            return None
        pending_request = self._next_pending_live_request(sandbox_id)
        try:
            logger.debug("Checking whether sandbox %s is due for checkpoint", sandbox_id)
            result = self._execute_checkpoint_flow(sandbox_id, pending_request=pending_request)
            if result is None:
                logger.debug("Sandbox %s is not due for checkpoint", sandbox_id)
                return None
            logger.info("Checkpoint-if-due for sandbox %s finished with status=%s", sandbox_id, result.status.value)
            return result
        finally:
            self._release_response_gate(sandbox_id, pending_request)
            self._refresh_interceptor_pending_state(sandbox_id)

    def checkpoint_due_sandboxes(self, sandbox_ids: list[SandboxId]) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        for sandbox_id in sandbox_ids:
            result = self.checkpoint_if_due(sandbox_id)
            if result is not None:
                results.append(result)
        return results

    def restore_once(
        self,
        sandbox_id: SandboxId,
        checkpoint_id,
        *,
        restore_metadata: dict[str, object] | None = None,
    ) -> RestoreResult:
        logger.info("Running manual restore for sandbox %s checkpoint=%s", sandbox_id, checkpoint_id)
        started = utc_now()
        restore_checkpoint_id = CheckpointId(str(checkpoint_id))
        operation = start_operation(
            self.telemetry,
            "restore.flow",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                checkpoint_id=restore_checkpoint_id,
                extra={"reason": "manual"},
            ),
        )
        restore_message = (
            self._validate_restore_checkpoint(sandbox_id, restore_checkpoint_id)
            if self.enforce_restore_checkpoint_validation
            else None
        )
        if restore_message is not None:
            logger.warning(
                "Skipping restore for sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                restore_checkpoint_id,
                restore_message,
            )
            failed_result = RestoreResult(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                checkpoint_id=restore_checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=FailureCode.VALIDATION_ERROR,
                message=restore_message,
            )
            operation.finish(
                status=failed_result.status.value,
                attributes={
                    "job_id": str(failed_result.job_id),
                    "failure_code": failed_result.failure_code.value,
                },
            )
            return failed_result
        restore_manifest = None
        if self.restore_metadata_handler is not None:
            restore_manifest = self._resolve_restore_manifest(sandbox_id, restore_checkpoint_id)
            if restore_manifest is not None:
                # Preload replay/router state before the restored process can
                # resume making requests. This avoids a narrow race where the
                # process issues its first post-restore request before the
                # replay service cursor is rewound to the checkpoint position.
                self.restore_metadata_handler(sandbox_id, restore_manifest)
        self.runtime.prepare_for_restore(sandbox_id)
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=restore_checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
            metadata=dict(restore_metadata or {}),
        )
        result = self.executor.run_restore(job)
        if result.status.value == "succeeded":
            runtime_state = self._wait_for_runtime_running(sandbox_id)
            if runtime_state is None:
                self.runtime.sync_runtime_state(sandbox_id, is_running=False)
                message = f"restore completed but sandbox {sandbox_id} is not running"
                logger.warning(
                    "Restore reported success but sandbox is not running sandbox=%s checkpoint=%s",
                    sandbox_id,
                    restore_checkpoint_id,
                )
                result = replace(
                    result,
                    status=JobStatus.FAILED,
                    finished_at=utc_now(),
                    failure_code=FailureCode.RUNTIME_ERROR,
                    message=message,
                )
            else:
                self.runtime.mark_restored(sandbox_id)
                self._mark_sandbox_running(sandbox_id)
                self.storage.handle_restore_complete(sandbox_id, result.checkpoint_id)
                self._journal_lifecycle(
                    sandbox_id,
                    "restore",
                    metadata={"checkpoint_id": str(restore_checkpoint_id)},
                )
        logger.info(
            "Manual restore for sandbox %s checkpoint=%s finished with status=%s",
            sandbox_id,
            restore_checkpoint_id,
            result.status.value,
        )
        operation.finish(
            status=result.status.value,
            attributes={
                "job_id": str(result.job_id),
                "failure_code": result.failure_code.value,
            },
        )
        return result

    def fork_once(
        self,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        *,
        checkpoint_id: CheckpointId | None = None,
        target_rootfs_path: Path,
        bundle_root: Path | None = None,
        checkpoint_root: Path | None = None,
    ) -> forking.ForkResult:
        """Clone a source sandbox's checkpoint state onto a new sandbox id.

        Takes a fresh checkpoint when ``checkpoint_id`` is None, clones the
        filesystem via the runtime's provider, copies manifests/artifacts
        with path rewrites, and applies incremental chain sharing (pin +
        ancestor symlinks) when available. The fork is left *stopped*;
        callers restore it (Engine.fork_sandbox does, optionally lazily).
        Mechanics sunk from the benchmark harness's clone_checkpoint_to_fork.
        """
        paths = getattr(self.runtime, "paths", None)
        if bundle_root is None:
            bundle_root = None if paths is None else paths.bundle_root
        if checkpoint_root is None:
            checkpoint_root = None if paths is None else paths.checkpoint_root
        if bundle_root is None or checkpoint_root is None:
            raise ValueError("fork_once requires bundle_root/checkpoint_root (runtime exposes no paths)")

        operation = start_operation(
            self.telemetry,
            "fork.flow",
            self._telemetry_attrs(
                source_sandbox_id,
                component="system",
                extra={"target_sandbox_id": str(target_sandbox_id)},
            ),
        )
        try:
            if checkpoint_id is None:
                checkpoint_result = self.checkpoint_once(source_sandbox_id, leave_running=True)
                if checkpoint_result.status.value != "succeeded" or checkpoint_result.checkpoint_id is None:
                    raise RuntimeError(
                        f"fork checkpoint failed for sandbox {source_sandbox_id}: "
                        f"status={checkpoint_result.status.value}"
                    )
                checkpoint_id = checkpoint_result.checkpoint_id

            manifests = {
                cid: self.storage.get_manifest(source_sandbox_id, cid)
                for cid in self.storage.list_checkpoints(source_sandbox_id)
            }
            if checkpoint_id not in manifests:
                raise ValueError(f"checkpoint {checkpoint_id} not found for sandbox {source_sandbox_id}")
            checkpoint_order = list(manifests.keys())
            copy_plan = forking.resolve_checkpoint_copy_plan(checkpoint_order, manifests, checkpoint_id)
            filesystem_checkpoint_id = next(
                copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem
            )

            target_rootfs_path.mkdir(parents=True, exist_ok=True)
            target_dataset = self.runtime.clone_filesystem_snapshot(
                source_sandbox_id,
                filesystem_checkpoint_id,
                target_sandbox_id,
                target_rootfs_path=target_rootfs_path,
            )
            # Restore flows (prepare_for_restore/mark_restored) require a
            # runtime description; forks were never launched, so adopt one.
            self.runtime.adopt_sandbox_description(
                target_sandbox_id,
                runtime_name=self.runtime.name,
                status="stopped",
                metadata={
                    "sandbox_id": str(target_sandbox_id),
                    "bundle_path": str(bundle_root / str(target_sandbox_id)),
                    "rootfs_path": str(target_rootfs_path),
                    "zfs_dataset": target_dataset,
                    "forked_from": str(source_sandbox_id),
                },
            )

            # Chain sharing: only meaningful when the runtime supports
            # incremental process checkpoints and the leaf has ancestors.
            chain_sharing_active = False
            leaf = manifests.get(checkpoint_id)
            try:
                supports_incremental = bool(self.runtime.capabilities().supports_incremental_process)
            except Exception:
                supports_incremental = False
            if supports_incremental and leaf is not None and leaf.parent_checkpoint_id is not None:
                pin_chain = getattr(self.storage, "pin_chain", None)
                if callable(pin_chain) and pin_chain(source_sandbox_id, checkpoint_id):
                    with self._fork_lock:
                        self._fork_chain_pins[target_sandbox_id] = (source_sandbox_id, checkpoint_id)
                    chain_sharing_active = True
                else:
                    logger.warning(
                        "Fork chain-sharing pin unavailable; using copy mode source=%s target=%s checkpoint=%s",
                        source_sandbox_id,
                        target_sandbox_id,
                        checkpoint_id,
                    )

            chain_links = 0
            chain_bytes_saved = 0
            for copy_id, copy_process, copy_filesystem in copy_plan:
                source_manifest = manifests[copy_id]
                is_leaf = copy_id == checkpoint_id
                link_this_entry = chain_sharing_active and copy_process and not is_leaf
                process_refs = []
                filesystem_refs = []
                if copy_process:
                    if link_this_entry:
                        self.runtime.link_ancestor_pre_dump(source_sandbox_id, target_sandbox_id, copy_id)
                    for reference in source_manifest.process_artifacts:
                        payload = self.storage.get_artifact(source_sandbox_id, copy_id, reference)
                        if link_this_entry:
                            rewritten = forking.rewrite_process_artifact_linked(
                                payload,
                                target_sandbox_id=target_sandbox_id,
                                checkpoint_id=copy_id,
                                bundle_root=bundle_root,
                                checkpoint_root=checkpoint_root,
                            )
                            chain_links += 1
                            chain_bytes_saved += int(reference.size_bytes or 0)
                        else:
                            rewritten = forking.rewrite_process_artifact(
                                payload,
                                source_sandbox_id=source_sandbox_id,
                                target_sandbox_id=target_sandbox_id,
                                checkpoint_id=copy_id,
                                bundle_root=bundle_root,
                                checkpoint_root=checkpoint_root,
                                preserve_symlinks=chain_sharing_active and is_leaf,
                            )
                        process_refs.append(
                            self.storage.put_artifact(
                                target_sandbox_id,
                                copy_id,
                                ArtifactPayload(
                                    kind=reference.kind,
                                    name=reference.name,
                                    data=rewritten,
                                    metadata=dict(reference.metadata),
                                ),
                            )
                        )
                if copy_filesystem:
                    fork_fs_metadata = self.runtime.filesystem_checkpoint_metadata(target_sandbox_id, copy_id)
                    for reference in source_manifest.filesystem_artifacts:
                        payload = self.storage.get_artifact(source_sandbox_id, copy_id, reference)
                        filesystem_refs.append(
                            self.storage.put_artifact(
                                target_sandbox_id,
                                copy_id,
                                ArtifactPayload(
                                    kind=reference.kind,
                                    name=reference.name,
                                    data=forking.rewrite_filesystem_artifact(
                                        payload,
                                        target_sandbox_id=target_sandbox_id,
                                        checkpoint_id=copy_id,
                                        filesystem_metadata=fork_fs_metadata,
                                    ),
                                    metadata=dict(reference.metadata),
                                ),
                            )
                        )
                manifest = CheckpointManifest(
                    schema_version=source_manifest.schema_version,
                    checkpoint_id=source_manifest.checkpoint_id,
                    sandbox_id=target_sandbox_id,
                    created_at=source_manifest.created_at,
                    runtime_name=source_manifest.runtime_name,
                    runtime_version=source_manifest.runtime_version,
                    process_artifacts=process_refs,
                    filesystem_artifacts=filesystem_refs,
                    metadata=dict(source_manifest.metadata),
                ).with_integrity()
                self.storage.put_manifest(manifest)

            if chain_sharing_active:
                # Plant ancestor symlinks for the whole parent chain so
                # CRIU's chain walk during restore resolves into the
                # source's bytes without copying. Walk via manifests because
                # the copy plan short-circuits when the leaf carries both
                # process+filesystem artifacts (the common incremental case).
                try:
                    cursor_id = manifests[checkpoint_id].parent_checkpoint_id
                    seen: set[CheckpointId] = set()
                    while cursor_id is not None and cursor_id not in seen:
                        seen.add(cursor_id)
                        self.runtime.link_ancestor_pre_dump(source_sandbox_id, target_sandbox_id, cursor_id)
                        chain_links += 1
                        parent_manifest = manifests.get(cursor_id)
                        if parent_manifest is None:
                            break
                        chain_bytes_saved += sum(
                            int(ref.size_bytes or 0) for ref in parent_manifest.process_artifacts
                        )
                        cursor_id = parent_manifest.parent_checkpoint_id
                except Exception:
                    logger.exception(
                        "Failed to plant ancestor symlinks for chain sharing source=%s target=%s checkpoint=%s",
                        source_sandbox_id,
                        target_sandbox_id,
                        checkpoint_id,
                    )

            with self._fork_lock:
                self._fork_children.setdefault(source_sandbox_id, set()).add(target_sandbox_id)

            inherited_checkpoint_at = manifests[checkpoint_id].created_at
            upsert = getattr(self.inspector, "upsert_snapshot", None)
            if callable(upsert):
                upsert(
                    SandboxSnapshot(
                        sandbox_id=target_sandbox_id,
                        runtime_name=self.runtime.name,
                        is_running=False,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                        last_checkpoint_at=inherited_checkpoint_at,
                    )
                )
            self.scheduler.mark_checkpoint_complete(target_sandbox_id, inherited_checkpoint_at)

            result = forking.ForkResult(
                source_sandbox_id=source_sandbox_id,
                target_sandbox_id=target_sandbox_id,
                checkpoint_id=checkpoint_id,
                filesystem_checkpoint_id=filesystem_checkpoint_id,
                chain_shared=chain_sharing_active,
                chain_links=chain_links,
                chain_bytes_saved=chain_bytes_saved,
            )
            operation.finish(
                status="succeeded",
                attributes={
                    "checkpoint_id": str(checkpoint_id),
                    "filesystem_checkpoint_id": str(filesystem_checkpoint_id),
                    "chain_shared": chain_sharing_active,
                    "chain_links": chain_links,
                },
            )
            self._journal_lifecycle(
                source_sandbox_id,
                "fork_source",
                metadata={
                    "target_sandbox_id": str(target_sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "chain_shared": chain_sharing_active,
                },
            )
            self._journal_lifecycle(
                target_sandbox_id,
                "fork_created",
                metadata={
                    "source_sandbox_id": str(source_sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                },
            )
            logger.info(
                "Forked sandbox source=%s target=%s checkpoint=%s chain_shared=%s links=%d bytes_saved=%d",
                source_sandbox_id,
                target_sandbox_id,
                checkpoint_id,
                chain_sharing_active,
                chain_links,
                chain_bytes_saved,
            )
            return result
        except Exception:
            operation.finish(status="failed")
            raise

    def release_fork(self, target_sandbox_id: SandboxId) -> None:
        """Reverse fork_once's bookkeeping when a fork is destroyed."""
        with self._fork_lock:
            pin = self._fork_chain_pins.pop(target_sandbox_id, None)
            for children in self._fork_children.values():
                children.discard(target_sandbox_id)
        if pin is None:
            return
        source_sandbox_id, leaf_checkpoint_id = pin
        unpin_chain = getattr(self.storage, "unpin_chain", None)
        if not callable(unpin_chain):
            return
        try:
            unpin_chain(source_sandbox_id, leaf_checkpoint_id)
        except Exception:
            logger.exception(
                "Failed to unpin chain for fork=%s source=%s leaf=%s",
                target_sandbox_id,
                source_sandbox_id,
                leaf_checkpoint_id,
            )

    def prepare_source_destroy(self, source_sandbox_id: SandboxId) -> None:
        """Before destroying a sandbox that has live forks, detach them:
        promote one fork's filesystem clone so the source's dataset loses
        its dependents, and replace chain-shared symlinks with real bytes
        (storage artifacts and runtime pre-dump trees)."""
        self._journal_lifecycle(source_sandbox_id, "destroy")
        with self._fork_lock:
            live_forks = sorted(self._fork_children.get(source_sandbox_id, set()))
            pinned_forks = [fork for fork, (source, _) in self._fork_chain_pins.items() if source == source_sandbox_id]
        if not live_forks and not pinned_forks:
            return
        if live_forks:
            # Promoting one clone re-parents the source's snapshots (and any
            # sibling clones) onto it; destroying the source then succeeds.
            # Backends without clone-origin dependencies no-op here.
            try:
                self.runtime.promote_filesystem_dataset(live_forks[0])
            except Exception:
                logger.exception(
                    "Failed to promote fork filesystem before source destroy source=%s fork=%s",
                    source_sandbox_id,
                    live_forks[0],
                )
        if pinned_forks:
            materialize = getattr(self.storage, "materialize_linked_artifacts", None)
            if callable(materialize):
                try:
                    materialize(source_sandbox_id)
                except Exception:
                    logger.exception("Failed to materialize linked storage artifacts for source=%s", source_sandbox_id)
            for fork_id in pinned_forks:
                try:
                    self.runtime.materialize_linked_pre_dumps(fork_id)
                except Exception:
                    logger.exception("Failed to materialize linked pre-dumps for fork=%s", fork_id)
        with self._fork_lock:
            self._fork_children.pop(source_sandbox_id, None)

    def _wait_for_runtime_running(self, sandbox_id: SandboxId):
        for attempt in range(_RESTORE_RUNTIME_READY_ATTEMPTS):
            try:
                runtime_state = self.runtime.inspect_runtime(sandbox_id)
            except Exception:
                runtime_state = None
            if runtime_state is not None and runtime_state.is_running:
                return runtime_state
            if attempt + 1 < _RESTORE_RUNTIME_READY_ATTEMPTS:
                time.sleep(_RESTORE_RUNTIME_READY_DELAY_S)
        return None

    def notify_fault(self, sandbox_id: SandboxId, *, reason: str = "fault") -> None:
        logger.info("Received fault notification for sandbox=%s reason=%s", sandbox_id, reason)
        self._mark_sandbox_not_running(sandbox_id)
        event = RecoveryEvent(
            sandbox_id=sandbox_id,
            event_type="fault",
            observed_at=utc_now(),
            reason=reason,
        )
        self._recovery_queue.put(event)
        self.telemetry.emit_event(
            "recovery.event_received",
            {"sandbox_id": str(sandbox_id), "event_type": "fault", "reason": reason},
        )

    def notify_preemption(self, sandbox_id: SandboxId, *, grace_remaining_seconds: float) -> None:
        logger.info(
            "Received preemption notification for sandbox=%s grace_remaining_seconds=%.3f",
            sandbox_id,
            grace_remaining_seconds,
        )
        self._merge_snapshot_metadata(
            sandbox_id,
            preemption_notice=True,
            preemption_grace_remaining_seconds=grace_remaining_seconds,
        )
        event = RecoveryEvent(
            sandbox_id=sandbox_id,
            event_type="preemption",
            observed_at=utc_now(),
            grace_remaining_seconds=grace_remaining_seconds,
            reason="preemption",
        )
        self._recovery_queue.put(event)
        self.telemetry.emit_event(
            "recovery.event_received",
            {
                "sandbox_id": str(sandbox_id),
                "event_type": "preemption",
                "grace_remaining_seconds": grace_remaining_seconds,
            },
        )

    def get_last_recovery_record(self, sandbox_id: SandboxId) -> RecoveryRecord | None:
        with self._recovery_lock:
            return self._recovery_records.get(sandbox_id)

    def notify_interceptor_state_change(self, sandbox_id: SandboxId) -> None:
        # Reconcile against the actual gate state instead of blindly adding.
        # The monitor loop (auto_cr only) does the same reconciliation on its
        # own cadence; in manual mode where the monitor doesn't run, this is
        # the *only* path that prunes the set, so an unconditional add would
        # leak the sandbox forever and make has_pending_interceptor_signal
        # return True permanently — which deadlocks wait_for_task_completion's
        # replay-complete short-circuit.
        self._refresh_interceptor_pending_state(sandbox_id)
        with self._interceptor_lock:
            pending = sandbox_id in self._interceptor_pending
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        logger.debug("Recorded interceptor state change for sandbox %s pending=%s", sandbox_id, pending)
        self.telemetry.emit_event(
            "interceptor.state_changed",
            {
                "sandbox_id": str(sandbox_id),
                "pending": pending,
            },
        )

    # ----- transactions (B2) -------------------------------------------
    # Snapshot-based, weak isolation: actions run in place; abort rewinds
    # to the base checkpoint; observation staging keeps gated responses
    # from escaping an uncommitted txn. One active txn per sandbox.

    def _txn_active(self, sandbox_id: SandboxId) -> bool:
        with self._txn_lock:
            return sandbox_id in self._active_txns

    def begin_txn(self, sandbox_id: SandboxId, *, label: str | None = None) -> TxnDescription:
        txn_id = new_txn_id()
        with self._txn_lock:
            existing = self._active_txns.get(sandbox_id)
            if existing is not None:
                raise TxnActiveError(
                    f"transaction already active for {sandbox_id}: {existing.txn_id}"
                )
            if sandbox_id in self._active_txns:
                raise TxnActiveError(f"transaction begin already in flight for {sandbox_id}")
            # Reservation: suppresses auto-checkpoints and locks out
            # concurrent begins while the base checkpoint runs.
            self._active_txns[sandbox_id] = None
        try:
            base_checkpoint_id: CheckpointId | None = None
            base_was_fresh = False
            changed = True
            try:
                snapshot = self.inspector.inspect(sandbox_id)
                changed = bool(snapshot.process_changed or snapshot.filesystem_changed)
            except Exception:
                changed = True
            if not changed:
                base_checkpoint_id = self._latest_full_checkpoint_id(sandbox_id)
            if base_checkpoint_id is None:
                result = self.checkpoint_once(sandbox_id, leave_running=True)
                if result.status.value != "succeeded":
                    raise TxnError(
                        f"txn base checkpoint failed for {sandbox_id}: "
                        f"status={result.status.value} message={result.message}"
                    )
                base_checkpoint_id = result.checkpoint_id
                base_was_fresh = True
            self.begin_observation_staging(sandbox_id)
            journal = self.journal
            if journal is not None:
                try:
                    journal.set_active_txn(sandbox_id, txn_id)
                except Exception:
                    logger.exception("Failed to set active txn on journal sandbox=%s", sandbox_id)
            description = TxnDescription(
                txn_id=txn_id,
                sandbox_id=str(sandbox_id),
                base_checkpoint_id=str(base_checkpoint_id),
                base_was_fresh=base_was_fresh,
                started_at=utc_now().isoformat(),
                label=label,
            )
            with self._txn_lock:
                self._active_txns[sandbox_id] = description
            self._journal_lifecycle(
                sandbox_id,
                "txn_begin",
                metadata={
                    "txn_id": txn_id,
                    "base_checkpoint_id": str(base_checkpoint_id),
                    "base_was_fresh": base_was_fresh,
                    **({"label": label} if label else {}),
                },
            )
            self.telemetry.emit_event(
                "txn.begin",
                self._telemetry_attrs(
                    sandbox_id,
                    component="system",
                    extra={
                        "txn_id": txn_id,
                        "base_checkpoint_id": str(base_checkpoint_id),
                        "base_was_fresh": base_was_fresh,
                    },
                ),
            )
            logger.info(
                "Began txn %s for sandbox %s base=%s fresh=%s",
                txn_id,
                sandbox_id,
                base_checkpoint_id,
                base_was_fresh,
            )
            return description
        except Exception:
            with self._txn_lock:
                self._active_txns.pop(sandbox_id, None)
            raise

    def commit_txn(self, sandbox_id: SandboxId, txn_id: str) -> TxnCommitResult:
        active = self._require_txn(sandbox_id, txn_id)
        registry = self.response_gate_registry
        if registry is not None:
            # Anything still pending (armed but never checkpoint-released)
            # moves into the staged buffer first.
            registry.release(sandbox_id)
        released = self.release_staged_observations(sandbox_id)
        self.end_observation_staging(sandbox_id)
        base_dropped = False
        if active.base_was_fresh and active.base_checkpoint_id is not None:
            try:
                self.storage.delete_checkpoint(
                    sandbox_id, CheckpointId(active.base_checkpoint_id), cascade=False
                )
                base_dropped = True
            except Exception:
                logger.warning(
                    "Keeping txn base checkpoint after commit (delete failed) sandbox=%s ckpt=%s",
                    sandbox_id,
                    active.base_checkpoint_id,
                    exc_info=True,
                )
        self._journal_lifecycle(
            sandbox_id,
            "txn_commit",
            metadata={
                "txn_id": active.txn_id,
                "released": released,
                "base_dropped": base_dropped,
            },
        )
        self._clear_txn(sandbox_id)
        self.telemetry.emit_event(
            "txn.commit",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={"txn_id": active.txn_id, "released": released, "base_dropped": base_dropped},
            ),
        )
        logger.info(
            "Committed txn %s for sandbox %s released=%d base_dropped=%s",
            active.txn_id,
            sandbox_id,
            released,
            base_dropped,
        )
        return TxnCommitResult(
            txn_id=active.txn_id,
            released_observations=released,
            base_dropped=base_dropped,
        )

    def abort_txn(self, sandbox_id: SandboxId, txn_id: str) -> TxnAbortResult:
        active = self._require_txn(sandbox_id, txn_id)
        registry = self.response_gate_registry
        if registry is not None:
            registry.release(sandbox_id)
        discarded = self.discard_staged_observations(sandbox_id)
        assert active.base_checkpoint_id is not None
        restore = self.restore_once(sandbox_id, CheckpointId(active.base_checkpoint_id))
        if restore.status.value != "succeeded":
            # Txn stays open: observations are already dropped (idempotent),
            # the caller may retry abort.
            raise TxnAbortError(
                f"txn abort restore failed for {sandbox_id}: "
                f"status={restore.status.value} message={restore.message}",
                restore_result=restore,
            )
        self.end_observation_staging(sandbox_id)
        self._journal_lifecycle(
            sandbox_id,
            "txn_abort",
            metadata={
                "txn_id": active.txn_id,
                "discarded": discarded,
                "restored_checkpoint_id": str(active.base_checkpoint_id),
            },
        )
        self._clear_txn(sandbox_id)
        self.telemetry.emit_event(
            "txn.abort",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={"txn_id": active.txn_id, "discarded": discarded},
            ),
        )
        logger.info(
            "Aborted txn %s for sandbox %s discarded=%d restored=%s",
            active.txn_id,
            sandbox_id,
            discarded,
            active.base_checkpoint_id,
        )
        return TxnAbortResult(
            txn_id=active.txn_id,
            discarded_observations=discarded,
            restored_checkpoint_id=active.base_checkpoint_id,
        )

    def current_txn(self, sandbox_id: SandboxId) -> TxnDescription | None:
        with self._txn_lock:
            return self._active_txns.get(sandbox_id)

    def release_txn(self, sandbox_id: SandboxId) -> None:
        """Teardown hook (sandbox kill with an open txn): drop staged
        observations and disarm — no restore, the sandbox is dying."""
        with self._txn_lock:
            active = self._active_txns.pop(sandbox_id, None)
        if active is None:
            return
        try:
            registry = self.response_gate_registry
            if registry is not None:
                registry.release(sandbox_id)
                registry.discard_staged(sandbox_id)
                registry.end_staging(sandbox_id)
        except Exception:
            logger.exception("Txn teardown staging cleanup failed sandbox=%s", sandbox_id)
        journal = self.journal
        if journal is not None:
            try:
                journal.set_active_txn(sandbox_id, None)
            except Exception:
                logger.debug("Failed to clear journal txn on teardown", exc_info=True)
        logger.info("Released open txn %s during sandbox teardown %s", active.txn_id, sandbox_id)

    def _require_txn(self, sandbox_id: SandboxId, txn_id: str) -> TxnDescription:
        with self._txn_lock:
            active = self._active_txns.get(sandbox_id)
        if active is None:
            raise TxnMismatchError(f"no active transaction for {sandbox_id}")
        if active.txn_id != str(txn_id):
            raise TxnMismatchError(
                f"txn mismatch for {sandbox_id}: active={active.txn_id} given={txn_id}"
            )
        return active

    def _clear_txn(self, sandbox_id: SandboxId) -> None:
        journal = self.journal
        if journal is not None:
            try:
                journal.set_active_txn(sandbox_id, None)
            except Exception:
                logger.debug("Failed to clear journal txn", exc_info=True)
        with self._txn_lock:
            self._active_txns.pop(sandbox_id, None)

    def _latest_full_checkpoint_id(self, sandbox_id: SandboxId) -> CheckpointId | None:
        """Newest checkpoint carrying both process and filesystem
        artifacts — the only safe reuse target for a txn base."""
        try:
            checkpoint_ids = self.storage.list_checkpoints(sandbox_id)
        except Exception:
            return None
        for checkpoint_id in reversed(list(checkpoint_ids)):
            try:
                manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
            except Exception:
                continue
            if getattr(manifest, "process_artifacts", None) and getattr(
                manifest, "filesystem_artifacts", None
            ):
                return checkpoint_id
        return None

    # ----- filesystem changesets (C1) ----------------------------------
    # The backend diff (zfs diff / btrfs send) is the source of truth;
    # the inspector gate is only a fast path that may skip the diff when
    # it can prove the answer is "nothing changed".

    def _latest_filesystem_checkpoint_id(self, sandbox_id: SandboxId) -> CheckpointId | None:
        """Newest checkpoint carrying filesystem artifacts — the boundary
        where the inspector's filesystem cursor was last reset."""
        try:
            checkpoint_ids = self.storage.list_checkpoints(sandbox_id)
        except Exception:
            return None
        for checkpoint_id in reversed(list(checkpoint_ids)):
            try:
                manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
            except Exception:
                continue
            if getattr(manifest, "filesystem_artifacts", None):
                return checkpoint_id
        return None

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        use_inspector_gate: bool = True,
    ) -> ChangesetResult:
        """Changed rootfs paths of ``sandbox_id`` relative to
        ``checkpoint_id``'s filesystem snapshot (C1). The inspector gate
        may skip the backend diff only when it proves nothing touched
        the filesystem since the last filesystem checkpoint AND that
        checkpoint is the requested base; anything less falls through to
        the authoritative diff."""
        skipped_by_gate = False
        if use_inspector_gate:
            filesystem_changed = True
            try:
                snapshot = self.inspector.inspect(sandbox_id)
                filesystem_changed = bool(snapshot.filesystem_changed)
            except Exception:
                filesystem_changed = True
            if not filesystem_changed and self._latest_filesystem_checkpoint_id(sandbox_id) == checkpoint_id:
                skipped_by_gate = True
        if skipped_by_gate:
            entries: tuple[ChangesetEntry, ...] = ()
        else:
            entries = tuple(self.runtime.changeset_since(sandbox_id, checkpoint_id))
        result = ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=checkpoint_id,
            entries=entries,
            skipped_by_gate=skipped_by_gate,
        )
        self._journal_lifecycle(
            sandbox_id,
            "changeset",
            metadata={
                "base_checkpoint_id": str(checkpoint_id),
                "entry_count": len(entries),
                "skipped_by_gate": skipped_by_gate,
            },
        )
        self.telemetry.emit_event(
            "changeset.computed",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                checkpoint_id=checkpoint_id,
                extra={
                    "entry_count": len(entries),
                    "skipped_by_gate": skipped_by_gate,
                },
            ),
        )
        logger.info(
            "Computed changeset sandbox=%s base=%s entries=%d skipped_by_gate=%s",
            sandbox_id,
            checkpoint_id,
            len(entries),
            skipped_by_gate,
        )
        return result

    def fork_changeset(self, target_sandbox_id: SandboxId) -> ChangesetResult:
        """Changeset of a fork relative to its fork point (the source
        checkpoint snapshot materialized on the fork's own dataset at
        clone time)."""
        checkpoint_id = self._fork_point_checkpoint_id(target_sandbox_id)
        if checkpoint_id is None:
            raise ValueError(
                f"no fork_created journal marker for {target_sandbox_id}; "
                "fork_changeset only works for sandboxes created by fork_once "
                "with the action journal enabled"
            )
        return self.changeset_since(target_sandbox_id, checkpoint_id)

    def _fork_point_checkpoint_id(self, target_sandbox_id: SandboxId) -> CheckpointId | None:
        journal = self.journal
        if journal is None:
            return None
        try:
            records = journal.entries(target_sandbox_id, kind="lifecycle")
        except Exception:
            logger.exception("Failed to read journal for fork point sandbox=%s", target_sandbox_id)
            return None
        for record in reversed(records):
            if record.payload.get("event") != "fork_created":
                continue
            metadata = record.payload.get("metadata") or {}
            raw = metadata.get("checkpoint_id")
            if raw:
                return CheckpointId(str(raw))
        return None

    # ----- observation staging (B1) -----------------------------------
    # Thin facade over the response-gate registry's staging extension so
    # the B2 transaction API has one system-level surface to drive. Each
    # transition also lands in the action journal for the C3/C4 audit
    # trail.

    def begin_observation_staging(self, sandbox_id: SandboxId) -> None:
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        registry.begin_staging(sandbox_id)
        self._journal_lifecycle(sandbox_id, "staging_begin")

    def release_staged_observations(self, sandbox_id: SandboxId) -> int:
        """Commit path: deliver everything staged."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        released = registry.release_staged(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_commit", metadata={"released": released}
        )
        return released

    def discard_staged_observations(self, sandbox_id: SandboxId) -> int:
        """Abort path: drop everything staged (callers get 409)."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        discarded = registry.discard_staged(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_abort", metadata={"discarded": discarded}
        )
        return discarded

    def end_observation_staging(self, sandbox_id: SandboxId) -> int:
        """Disarm; leftovers are delivered (fail-open)."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        leftover = registry.end_staging(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_end", metadata={"delivered_leftover": leftover}
        )
        return leftover

    def notify_live_response_ready(
        self,
        sandbox_id: SandboxId,
        request_id: str,
        generation: int | None = None,
    ) -> None:
        self.executor.notify_live_response_ready(
            sandbox_id,
            request_id,
            generation=generation,
        )

    def has_pending_interceptor_signal(self, sandbox_id: SandboxId) -> bool:
        with self._interceptor_lock:
            return sandbox_id in self._interceptor_pending

    def _run_monitor_loop(self) -> None:
        assert self.request_state_store is not None
        while not self._stop_event.is_set():
            change = self.request_state_store.wait_for_change(timeout=0.5)
            if change is not None:
                self._refresh_interceptor_pending_state(change.sandbox_id)
                if change.event_type == "request_start":
                    coord_decision = self._should_coordinate_live_request(change.sandbox_id, change.request_id)
                    logger.debug(
                        "DIAG.monitor.request_start sandbox=%s request_id=%s should_coord=%s",
                        change.sandbox_id,
                        "" if change.request_id is None else change.request_id,
                        coord_decision,
                    )
                    if coord_decision:
                        self._dispatch_coordination(change.sandbox_id)
                    else:
                        logger.debug(
                            "Skipping stale request_start coordination sandbox=%s request_id=%s",
                            change.sandbox_id,
                            "" if change.request_id is None else change.request_id,
                        )
                else:
                    logger.debug(
                        "DIAG.monitor.event sandbox=%s event_type=%s request_id=%s",
                        change.sandbox_id,
                        change.event_type,
                        "" if change.request_id is None else change.request_id,
                    )
            self._dispatch_pending_coordination()

    def _should_coordinate_live_request(self, sandbox_id: SandboxId, request_id: str | None) -> bool:
        if self._txn_active(sandbox_id):
            logger.debug("DIAG.coord.check.txn_active sandbox=%s", sandbox_id)
            return False
        if self.request_state_store is None or self.response_gate_registry is None:
            logger.debug(
                "DIAG.coord.check.no_store sandbox=%s",
                sandbox_id,
            )
            return False
        request_state = self.request_state_store.get(sandbox_id)
        if not request_state.llm_request_in_flight:
            logger.debug(
                "DIAG.coord.check.no_in_flight sandbox=%s request_id=%s active_llm_requests=%d",
                sandbox_id,
                "" if request_id is None else request_id,
                request_state.active_llm_requests,
            )
            return False
        if request_id is None:
            oldest = self.response_gate_registry.get_oldest_pending(sandbox_id)
            logger.debug(
                "DIAG.coord.check.no_request_id sandbox=%s oldest_pending=%s",
                sandbox_id,
                "" if oldest is None else oldest.request_id,
            )
            return oldest is not None
        found = self.response_gate_registry.find_pending_request(sandbox_id, request_id)
        logger.debug(
            "DIAG.coord.check.find_pending sandbox=%s request_id=%s found=%s",
            sandbox_id,
            request_id,
            "" if found is None else found.generation,
        )
        return found is not None

    def _run_recovery_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._recovery_queue.get(timeout=0.5)
            except Empty:
                continue
            if event is None:
                return
            logger.info(
                "Recovery loop dequeued event sandbox=%s event_type=%s reason=%s",
                event.sandbox_id,
                event.event_type,
                event.reason,
            )
            self._handle_recovery_event(event)

    def _dispatch_coordination(self, sandbox_id: SandboxId) -> None:
        with self._coordination_lock:
            if sandbox_id in self._active_coordination:
                logger.debug(
                    "DIAG.coord.dispatch.skipped_already_active sandbox=%s",
                    sandbox_id,
                )
                return
            self._active_coordination.add(sandbox_id)
            pool = self._coordination_pool
        if pool is None:
            with self._coordination_lock:
                self._active_coordination.discard(sandbox_id)
            raise RuntimeError("coordination pool is not running")
        logger.info("DIAG.coord.dispatch.submitted sandbox=%s", sandbox_id)
        pool.submit(self._coordinate_sandbox_request, sandbox_id)

    def _coordinate_sandbox_request(self, sandbox_id: SandboxId) -> None:
        iteration = 0
        try:
            while not self._stop_event.is_set() and self._should_coordinate_any_pending_request(sandbox_id):
                iteration += 1
                pending_request = self._next_pending_live_request(sandbox_id)
                if pending_request is None:
                    logger.debug(
                        "DIAG.coord.loop.no_pending sandbox=%s iter=%d",
                        sandbox_id,
                        iteration,
                    )
                    break
                logger.debug(
                    "DIAG.coord.loop.execute sandbox=%s iter=%d request_id=%s generation=%s",
                    sandbox_id,
                    iteration,
                    pending_request.request_id,
                    pending_request.generation,
                )
                try:
                    self._execute_checkpoint_flow(sandbox_id, pending_request=pending_request)
                except Exception:
                    logger.exception(
                        "Checkpoint coordination failed for sandbox %s request_id=%s generation=%s",
                        sandbox_id,
                        pending_request.request_id,
                        pending_request.generation,
                    )
                    self._resume_sandbox(sandbox_id)
                finally:
                    self._release_response_gate(sandbox_id, pending_request)
                    self._refresh_interceptor_pending_state(sandbox_id)
        finally:
            should_redispatch = False
            with self._coordination_lock:
                self._active_coordination.discard(sandbox_id)
            self._refresh_interceptor_pending_state(sandbox_id)
            if self._should_coordinate_any_pending_request(sandbox_id):
                should_redispatch = True
                self._dispatch_coordination(sandbox_id)
            logger.info(
                "DIAG.coord.loop.exit sandbox=%s iters=%d redispatched=%s",
                sandbox_id,
                iteration,
                should_redispatch,
            )

    def _handle_recovery_event(self, event: RecoveryEvent) -> None:
        if not self._acquire_coordination(event.sandbox_id):
            return
        operation = start_operation(
            self.telemetry,
            "recovery.total",
            self._telemetry_attrs(
                event.sandbox_id,
                component="recovery",
                event_type=event.event_type,
                extra={"reason": event.reason},
            ),
        )
        started = utc_now()
        checkpoint_id = None
        restore_manifest: CheckpointManifest | None = None
        pinned_restore_ids: list[CheckpointId] = []
        status = "failed"
        message = None
        try:
            logger.info(
                "Handling recovery event sandbox=%s event_type=%s reason=%s",
                event.sandbox_id,
                event.event_type,
                event.reason,
            )
            queue_wait_ms = max(0.0, (started - event.received_at).total_seconds() * 1000.0)
            self.telemetry.emit_metric(
                "recovery.queue_wait_ms",
                queue_wait_ms,
                self._telemetry_attrs(
                    event.sandbox_id,
                    component="recovery",
                    event_type=event.event_type,
                    extra={"reason": event.reason},
                ),
            )
            self.telemetry.emit_event(
                "recovery.started",
                {
                    "sandbox_id": str(event.sandbox_id),
                    "event_type": event.event_type,
                },
            )
            if event.event_type == "preemption":
                selection_operation = start_operation(
                    self.telemetry,
                    "recovery.select_checkpoint",
                    self._telemetry_attrs(event.sandbox_id, component="recovery", event_type=event.event_type),
                )
                checkpoint_id = self._select_recovery_checkpoint_after(
                    event.sandbox_id,
                    observed_after=event.observed_at,
                )
                selection_operation.finish(status="succeeded", attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)})
                if checkpoint_id is not None:
                    logger.info(
                        "Reusing checkpoint already captured after preemption notice sandbox=%s checkpoint=%s",
                        event.sandbox_id,
                        checkpoint_id,
                    )
                else:
                    logger.info("Triggering preemption checkpoint flow for sandbox=%s", event.sandbox_id)
                    self._drain_active_runtime_execs(event.sandbox_id)
                    try:
                        checkpoint_result = self._execute_checkpoint_flow(event.sandbox_id)
                    except Exception:
                        checkpoint_id = self._select_recovery_checkpoint_after(
                            event.sandbox_id,
                            observed_after=event.observed_at,
                        )
                        if checkpoint_id is None:
                            raise
                        logger.warning(
                            "Preemption checkpoint flow failed after a recent checkpoint was captured; continuing with recovery sandbox=%s checkpoint=%s",
                            event.sandbox_id,
                            checkpoint_id,
                        )
                    else:
                        if checkpoint_result is not None and checkpoint_result.status.value == "succeeded":
                            checkpoint_id = checkpoint_result.checkpoint_id
                            logger.info(
                                "Preemption checkpoint completed sandbox=%s checkpoint=%s",
                                event.sandbox_id,
                                checkpoint_id,
                            )
                        elif checkpoint_result is not None:
                            message = checkpoint_result.message
                            logger.warning(
                                "Preemption checkpoint failed sandbox=%s message=%s",
                                event.sandbox_id,
                                checkpoint_result.message,
                            )
            if checkpoint_id is None:
                selection_operation = start_operation(
                    self.telemetry,
                    "recovery.select_checkpoint",
                    self._telemetry_attrs(event.sandbox_id, component="recovery", event_type=event.event_type),
                )
                checkpoint_id = self._select_recovery_checkpoint(event.sandbox_id)
                selection_operation.finish(status="succeeded", attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)})
                logger.info(
                    "Resolved latest checkpoint for recovery sandbox=%s checkpoint=%s",
                    event.sandbox_id,
                    "" if checkpoint_id is None else checkpoint_id,
                )
            if checkpoint_id is not None:
                self.telemetry.emit_event(
                    "recovery.checkpoint_resolved",
                    {
                        "sandbox_id": str(event.sandbox_id),
                        "event_type": event.event_type,
                        "checkpoint_id": str(checkpoint_id),
                    },
                )
            else:
                self.telemetry.emit_event(
                    "recovery.checkpoint_missing",
                    {
                        "sandbox_id": str(event.sandbox_id),
                        "event_type": event.event_type,
                    },
                )
            if checkpoint_id is not None:
                if self.recovery_delay_seconds > 0:
                    logger.info(
                        "Sleeping before restore sandbox=%s delay_seconds=%.3f",
                        event.sandbox_id,
                        self.recovery_delay_seconds,
                    )
                    time.sleep(self.recovery_delay_seconds)
                logger.info(
                    "Starting recovery restore sandbox=%s checkpoint=%s",
                    event.sandbox_id,
                    checkpoint_id,
                )
                restore_manifest, pinned_restore_ids = self._pin_restore_checkpoints(event.sandbox_id, checkpoint_id)
                if restore_manifest is None:
                    raise FileNotFoundError(f"manifest not found: selected checkpoint {checkpoint_id}")
                restore_operation = start_operation(
                    self.telemetry,
                    "recovery.restore",
                    self._telemetry_attrs(
                        event.sandbox_id,
                        component="recovery",
                        event_type=event.event_type,
                        checkpoint_id=checkpoint_id,
                    ),
                )
                try:
                    restore_result = self.restore_once(event.sandbox_id, checkpoint_id)
                except Exception:
                    restore_operation.finish(
                        status="failed",
                        attributes={"checkpoint_id": str(checkpoint_id)},
                    )
                    raise
                restore_operation.finish(
                    status=restore_result.status.value,
                    attributes={
                        "checkpoint_id": str(checkpoint_id),
                        "failure_code": restore_result.failure_code.value,
                        "job_id": str(restore_result.job_id),
                    },
                )
                if restore_result.status.value == "succeeded":
                    self._release_checkpoint_response_gate(
                        event.sandbox_id,
                        checkpoint_id,
                        manifest=restore_manifest,
                    )
                    status = "restored"
                    logger.info(
                        "Recovery restore succeeded sandbox=%s checkpoint=%s",
                        event.sandbox_id,
                        checkpoint_id,
                    )
                elif (
                    self.relaunch_handler is not None
                    and self.relaunch_on_restore_failure
                ):
                    logger.warning(
                        "Recovery restore failed; invoking relaunch handler sandbox=%s checkpoint=%s message=%s",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
                    self.relaunch_handler(
                        event.sandbox_id,
                        event.event_type,
                        True,
                    )
                    status = "relaunched"
                    message = "restore_failed_relaunch_handler_invoked"
                else:
                    # Restore failed and either no relaunch handler is wired or
                    # the relaunch fallback is opted out via
                    # relaunch_on_restore_failure=False (the default). Surface
                    # this as a hard error so latent bugs in checkpoint capture
                    # / restore are not masked by a silent relaunch.
                    logger.error(
                        "Recovery restore failed sandbox=%s checkpoint=%s message=%s "
                        "(set relaunch_on_restore_failure=True to fall back to relaunch_handler)",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
                    raise RuntimeError(
                        f"recovery restore failed sandbox={event.sandbox_id} "
                        f"checkpoint={checkpoint_id} message={restore_result.message}"
                    )
            elif (
                self.relaunch_handler is not None
                and self.relaunch_on_restore_failure
            ):
                logger.info("No checkpoint available; invoking relaunch handler for sandbox=%s", event.sandbox_id)
                self.relaunch_handler(
                    event.sandbox_id,
                    event.event_type,
                    False,
                )
                status = "relaunched"
                message = "relaunch_handler_invoked"
            else:
                status = "no_checkpoint"
                message = "no restorable checkpoint available"
                logger.warning(
                    "No checkpoint available for sandbox=%s and relaunch fallback is disabled "
                    "(set relaunch_on_restore_failure=True to fall back to relaunch_handler)",
                    event.sandbox_id,
                )
            if event.event_type == "preemption":
                self._clear_snapshot_metadata(
                    event.sandbox_id,
                    "preemption_notice",
                    "preemption_grace_remaining_seconds",
                )
        except Exception as exc:
            logger.exception("Recovery handling failed for sandbox %s event=%s", event.sandbox_id, event.event_type)
            status = "failed"
            message = str(exc)
        finally:
            finished = utc_now()
            logger.info(
                "Finished recovery event sandbox=%s event_type=%s status=%s checkpoint=%s message=%s",
                event.sandbox_id,
                event.event_type,
                status,
                "" if checkpoint_id is None else checkpoint_id,
                "" if message is None else message,
            )
            record = RecoveryRecord(
                sandbox_id=event.sandbox_id,
                event_type=event.event_type,
                started_at=started,
                finished_at=finished,
                status=status,
                checkpoint_id=checkpoint_id,
                message=message,
            )
            with self._recovery_lock:
                self._recovery_records[event.sandbox_id] = record
            self.telemetry.emit_event(
                "recovery.finished",
                {
                    "sandbox_id": str(event.sandbox_id),
                    "event_type": event.event_type,
                    "status": status,
                    "checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id),
                },
            )
            operation.finish(
                status=status,
                attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)},
            )
            self._release_coordination(event.sandbox_id)
            self._refresh_interceptor_pending_state(event.sandbox_id)
            if self._should_coordinate_any_pending_request(event.sandbox_id):
                self._dispatch_coordination(event.sandbox_id)
            if pinned_restore_ids:
                self._unpin_restore_checkpoints(event.sandbox_id, pinned_restore_ids)

    def _drain_active_runtime_execs(self, sandbox_id: SandboxId) -> None:
        """Ask the runtime to terminate any in-flight `runc exec`
        subprocesses for `sandbox_id`. Called before the spot
        preemption checkpoint flow so CRIU doesn't trip over half-
        stream unix-socket connections wired by runc-exec stdio."""
        cancel = getattr(self.runtime, "cancel_active_execs", None)
        if not callable(cancel):
            return
        try:
            cancelled = cancel(sandbox_id, timeout_s=2.0)
        except Exception:
            logger.debug(
                "Failed to drain active runtime execs sandbox=%s",
                sandbox_id,
                exc_info=True,
            )
            return
        if cancelled:
            logger.info(
                "Drained %d in-flight runtime exec(s) before preemption checkpoint sandbox=%s",
                cancelled,
                sandbox_id,
            )

    def _execute_checkpoint_flow(
        self,
        sandbox_id: SandboxId,
        *,
        pending_request: PendingSandboxResponse | None = None,
    ) -> CheckpointResult | None:
        if self._txn_active(sandbox_id):
            # Auto-checkpoints are suppressed inside a transaction: they
            # would pollute the retention chain with doomed states and
            # stage gated responses prematurely. Manual checkpoint_once
            # remains allowed (explicit user intent).
            logger.debug("Skipping scheduled checkpoint; txn active sandbox=%s", sandbox_id)
            return None
        operation = start_operation(
            self.telemetry,
            "checkpoint.flow",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                request_id=None if pending_request is None else pending_request.request_id,
                extra=(
                    None
                    if pending_request is None
                    else {"request_generation": pending_request.generation}
                ),
            ),
        )
        decision = self.scheduler.query_checkpoint(sandbox_id)
        if not decision.should_checkpoint:
            operation.finish(status="skipped", attributes={"reason": decision.reason})
            return None
        checkpoint_metadata = self._build_checkpoint_metadata(sandbox_id, pending_request=pending_request)

        job = CheckpointJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason=decision.reason,
            checkpoint_process=decision.checkpoint_process,
            checkpoint_filesystem=decision.checkpoint_filesystem,
            leave_running=decision.leave_running,
            is_incremental_process=decision.is_incremental_process,
            parent_process_checkpoint_id=decision.parent_process_checkpoint_id,
            produce_pre_dump=decision.produce_pre_dump,
            metadata={"policy": decision.policy_name, **decision.metadata, **checkpoint_metadata},
        )
        result: CheckpointResult | None = None
        flow_exception: BaseException | None = None
        try:
            result = self.executor.submit_checkpoint(job).result()
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": decision.reason,
                        "leave_running": bool(job.leave_running),
                    },
                )
            return result
        except BaseException as exc:
            flow_exception = exc
            raise
        finally:
            # Telemetry must be symmetric: every flow.start needs a
            # matching flow.finish, even if post-checkpoint bookkeeping
            # (e.g. daemon-side reset) raised. An orphan flow.start is
            # counted as a failure by the report tooling.
            if result is not None:
                operation.finish(
                    status=result.status.value,
                    attributes={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": decision.reason,
                        "job_id": str(job.job_id),
                        "checkpoint_scope": _checkpoint_scope(job),
                        "failure_code": result.failure_code.value,
                    },
                )
            else:
                operation.finish(
                    status="failed",
                    attributes={
                        "reason": decision.reason,
                        "job_id": str(job.job_id),
                        "checkpoint_scope": _checkpoint_scope(job),
                        "error": type(flow_exception).__name__ if flow_exception is not None else "unknown",
                    },
                )
            if self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)

    def _latest_checkpoint_id(self, sandbox_id: SandboxId):
        checkpoints = self.storage.list_checkpoints(sandbox_id)
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _build_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        *,
        pending_request: PendingSandboxResponse | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {_CAPTURES_INFLIGHT_LLM: False}
        if self.extra_checkpoint_metadata_provider is not None:
            try:
                metadata.update(self.extra_checkpoint_metadata_provider(sandbox_id))
            except Exception:
                logger.exception("Failed to collect extra checkpoint metadata for sandbox=%s", sandbox_id)
        if self.request_state_store is None or self.response_gate_registry is None:
            return metadata
        pending = pending_request or self.response_gate_registry.get_oldest_pending(sandbox_id)
        if pending is None:
            return metadata
        request_context = self.request_state_store.get_request_context(sandbox_id, pending.request_id)
        if request_context is None:
            return metadata
        if not bool(request_context.metadata.get("response_gate_enabled", True)):
            logger.debug(
                "Skipping live-request checkpoint capture for auxiliary request sandbox=%s request_id=%s kind=%s",
                sandbox_id,
                pending.request_id,
                request_context.metadata.get("request_kind"),
            )
            return metadata
        metadata[_CAPTURES_INFLIGHT_LLM] = True
        metadata[_CAPTURED_REQUEST_ID] = pending.request_id
        metadata[_CAPTURED_REQUEST_GENERATION] = pending.generation
        provider = request_context.metadata.get("provider")
        if provider is not None:
            metadata[_CAPTURED_REQUEST_PROVIDER] = str(provider)
        metadata[_CAPTURED_REQUEST_STARTED_AT] = request_context.started_at.isoformat()
        if pending.request_ids:
            metadata[_CAPTURED_REQUEST_IDS] = list(pending.request_ids)
        if pending.request_group_kind and pending.request_group_id:
            metadata[_CAPTURED_REQUEST_GROUP_KIND] = pending.request_group_kind
            metadata[_CAPTURED_REQUEST_GROUP_ID] = pending.request_group_id
            contexts = self.request_state_store.get_request_contexts_for_group(
                sandbox_id,
                request_group_kind=pending.request_group_kind,
                request_group_id=pending.request_group_id,
            )
            if contexts:
                group_started_at = min(context.started_at for context in contexts)
                metadata[_CAPTURED_REQUEST_GROUP_STARTED_AT] = group_started_at.isoformat()
                metadata[_CAPTURED_REQUEST_IDS] = [context.request_id for context in contexts]
        logger.info(
            "Checkpoint captured live request sandbox=%s request_id=%s generation=%s group_kind=%s group_id=%s",
            sandbox_id,
            pending.request_id,
            pending.generation,
            pending.request_group_kind,
            pending.request_group_id,
        )
        self.telemetry.emit_event(
            "checkpoint.captured_live_request",
            {
                "sandbox_id": str(sandbox_id),
                "request_id": pending.request_id,
                "request_generation": pending.generation,
                "request_group_kind": pending.request_group_kind or "",
                "request_group_id": pending.request_group_id or "",
            },
        )
        return metadata

    def _validate_restore_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            return None
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        if captured_group_kind and captured_group_id:
            pending = self._find_captured_pending_request(sandbox_id, manifest)
            if pending is None:
                return (
                    f"checkpoint {checkpoint_id} captured live request group "
                    f"{captured_group_kind}:{captured_group_id} but no matching interceptor-held group is pending"
                )
            if pending.request_group_kind != captured_group_kind or pending.request_group_id != captured_group_id:
                return (
                    f"checkpoint {checkpoint_id} captured live request group "
                    f"{captured_group_kind}:{captured_group_id} but current pending group is "
                    f"{pending.request_group_kind}:{pending.request_group_id}"
                )
            return None
        if not captured_request_id:
            return f"checkpoint {checkpoint_id} advertises live-request restore without captured_request_id"
        pending = self._find_captured_pending_request(sandbox_id, manifest)
        if pending is None:
            return (
                f"checkpoint {checkpoint_id} captured live request {captured_request_id} "
                "but no matching interceptor-held request is pending"
            )
        if pending.request_id != captured_request_id:
            return (
                f"checkpoint {checkpoint_id} captured live request {captured_request_id} "
                f"but current pending request is {pending.request_id}"
            )
        return None

    def _select_recovery_checkpoint(self, sandbox_id: SandboxId) -> CheckpointId | None:
        checkpoints = list(reversed(self.storage.list_checkpoints(sandbox_id)))
        for checkpoint_id in checkpoints:
            validation_message = (
                self._validate_restore_checkpoint(sandbox_id, checkpoint_id)
                if self.enforce_restore_checkpoint_validation
                else None
            )
            if validation_message is None:
                return checkpoint_id
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
            if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
                logger.warning(
                    "Skipping checkpoint after validation failure sandbox=%s checkpoint=%s message=%s",
                    sandbox_id,
                    checkpoint_id,
                    validation_message,
                )
                continue
            logger.warning(
                "Skipping stale live-request checkpoint sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                checkpoint_id,
                validation_message,
            )
            self.telemetry.emit_event(
                "recovery.checkpoint_skipped_stale_request",
                {
                    "sandbox_id": str(sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "message": validation_message,
                },
            )
        logger.warning("No restorable checkpoint available for sandbox=%s", sandbox_id)
        self.telemetry.emit_event(
            "recovery.no_satisfiable_checkpoint",
            {"sandbox_id": str(sandbox_id)},
        )
        return None

    def _select_recovery_checkpoint_after(
        self,
        sandbox_id: SandboxId,
        *,
        observed_after,
    ) -> CheckpointId | None:
        checkpoints = list(reversed(self.storage.list_checkpoints(sandbox_id)))
        for checkpoint_id in checkpoints:
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
            if manifest.created_at < observed_after:
                break
            validation_message = (
                self._validate_restore_checkpoint(sandbox_id, checkpoint_id)
                if self.enforce_restore_checkpoint_validation
                else None
            )
            if validation_message is None:
                return checkpoint_id
            logger.warning(
                "Skipping recent checkpoint after validation failure sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                checkpoint_id,
                validation_message,
            )
        return None

    def _resolve_restore_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId):
        manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
        return resolve_restore_manifest(self.storage, manifest)

    def _pin_restore_checkpoints(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[CheckpointManifest | None, list[CheckpointId]]:
        manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        pin_checkpoint = getattr(self.storage, "pin_checkpoint", None)
        if not callable(pin_checkpoint):
            return manifest, []
        if not pin_checkpoint(sandbox_id, checkpoint_id):
            return None, []
        pinned_ids = [checkpoint_id]
        for metadata_key in ("process_restore_checkpoint_id", "filesystem_restore_checkpoint_id"):
            raw_value = manifest.metadata.get(metadata_key)
            if raw_value is None:
                continue
            candidate = CheckpointId(str(raw_value))
            if candidate in pinned_ids:
                continue
            if pin_checkpoint(sandbox_id, candidate):
                pinned_ids.append(candidate)
        return manifest, pinned_ids

    def _unpin_restore_checkpoints(self, sandbox_id: SandboxId, checkpoint_ids: list[CheckpointId]) -> None:
        unpin_checkpoint = getattr(self.storage, "unpin_checkpoint", None)
        if not callable(unpin_checkpoint):
            return
        for checkpoint_id in reversed(checkpoint_ids):
            unpin_checkpoint(sandbox_id, checkpoint_id)

    def _release_checkpoint_response_gate(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        manifest: CheckpointManifest | None = None,
    ) -> bool:
        release_operation = start_operation(
            self.telemetry,
            "recovery.response_release",
            self._telemetry_attrs(sandbox_id, component="recovery", checkpoint_id=checkpoint_id),
        )
        if self.response_gate_registry is None:
            release_operation.finish(status="skipped")
            return False
        if manifest is None:
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            release_operation.finish(status="skipped")
            return False
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            release_operation.finish(status="skipped")
            return False
        request_ids = manifest.metadata.get(_CAPTURED_REQUEST_IDS, [])
        if not isinstance(request_ids, list) or not request_ids:
            request_ids = [captured_request_id]
        for request_id in request_ids:
            self.executor.clear_live_response_ready(sandbox_id, str(request_id))
        pending = self._find_captured_pending_request(sandbox_id, manifest)
        if pending is None:
            release_operation.finish(status="skipped")
            return False
        if captured_group_kind and captured_group_id:
            released = self.response_gate_registry.release_pending(
                sandbox_id,
                request_id=pending.request_id,
                generation=pending.generation,
            )
        else:
            released = self.response_gate_registry.release_pending(
                sandbox_id,
                request_id=captured_request_id,
                generation=pending.generation,
            )
        if released:
            logger.info(
                "Released buffered response to restored sandbox=%s request_id=%s checkpoint=%s",
                sandbox_id,
                captured_request_id,
                checkpoint_id,
            )
            self.telemetry.emit_event(
                "recovery.response_released",
                {
                    "sandbox_id": str(sandbox_id),
                    "request_id": captured_request_id,
                    "checkpoint_id": str(checkpoint_id),
                },
            )
        release_operation.finish(
            status="succeeded" if released else "failed",
            attributes={"request_id": captured_request_id},
        )
        return released

    def _merge_snapshot_metadata(self, sandbox_id: SandboxId, **metadata: object) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is None:
            return
        snapshot = self.inspector.inspect(sandbox_id)
        upsert(
            snapshot.__class__(
                sandbox_id=snapshot.sandbox_id,
                runtime_name=snapshot.runtime_name,
                is_running=snapshot.is_running,
                process_changed=snapshot.process_changed,
                filesystem_changed=snapshot.filesystem_changed,
                observed_at=utc_now(),
                last_checkpoint_at=snapshot.last_checkpoint_at,
                metadata={**snapshot.metadata, **metadata},
            )
        )

    def _clear_snapshot_metadata(self, sandbox_id: SandboxId, *keys: str) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is None:
            return
        snapshot = self.inspector.inspect(sandbox_id)
        metadata = dict(snapshot.metadata)
        for key in keys:
            metadata.pop(key, None)
        upsert(
            snapshot.__class__(
                sandbox_id=snapshot.sandbox_id,
                runtime_name=snapshot.runtime_name,
                is_running=snapshot.is_running,
                process_changed=snapshot.process_changed,
                filesystem_changed=snapshot.filesystem_changed,
                observed_at=utc_now(),
                last_checkpoint_at=snapshot.last_checkpoint_at,
                metadata=metadata,
            )
        )

    def _acquire_coordination(self, sandbox_id: SandboxId) -> bool:
        while not self._stop_event.is_set():
            with self._coordination_lock:
                if sandbox_id not in self._active_coordination:
                    self._active_coordination.add(sandbox_id)
                    return True
            self._stop_event.wait(0.05)
        return False

    def _release_coordination(self, sandbox_id: SandboxId) -> None:
        with self._coordination_lock:
            self._active_coordination.discard(sandbox_id)

    def _pause_for_manual_checkpoint(self, sandbox_id: SandboxId) -> bool:
        try:
            self.runtime.pause(sandbox_id)
            return True
        except Exception:
            logger.exception("Failed to pause sandbox %s for manual checkpoint", sandbox_id)
            return False

    def _resume_sandbox(self, sandbox_id: SandboxId) -> None:
        try:
            description = self.runtime.describe(sandbox_id)
        except Exception:
            return
        if description.status != "paused":
            return
        try:
            self.runtime.resume(sandbox_id)
        except Exception:
            logger.exception("Failed to resume sandbox %s", sandbox_id)
            return
        self._mark_sandbox_running(sandbox_id)

    def quiesce_for_verification(
        self,
        sandbox_id: SandboxId,
        *,
        drain_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        # Terminal transition from run phase to verification phase. Enforces:
        #   1. scheduler no longer issues checkpoint decisions for this sandbox
        #   2. no executor jobs remain pending or running
        #   3. the container is not paused
        # Any of these held at verify time produce the "cannot exec in a paused
        # container" race we saw in 20260420_123846 spec-91-spec-81.
        self.scheduler.deactivate_sandbox(sandbox_id)
        deadline = time.monotonic() + max(0.0, float(drain_timeout_seconds))
        while self.executor.has_active_job(sandbox_id):
            if time.monotonic() >= deadline:
                logger.warning(
                    "quiesce_for_verification timed out draining executor for sandbox %s; proceeding",
                    sandbox_id,
                )
                break
            time.sleep(poll_interval_seconds)
        self._resume_sandbox(sandbox_id)

    def _mark_sandbox_not_running(self, sandbox_id: SandboxId) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is not None:
            try:
                snapshot = self.inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            if snapshot is not None and snapshot.is_running:
                upsert(
                    replace(
                        snapshot,
                        is_running=False,
                        observed_at=utc_now(),
                    )
                )
        try:
            self.runtime.sync_runtime_state(sandbox_id, is_running=False)
        except Exception:
            logger.debug("Failed to sync runtime state for faulted sandbox %s", sandbox_id, exc_info=True)

    def _mark_sandbox_running(self, sandbox_id: SandboxId) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is not None:
            try:
                snapshot = self.inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            if snapshot is not None:
                upsert(
                    replace(
                        snapshot,
                        is_running=True,
                        observed_at=utc_now(),
                    )
                )
        try:
            self.runtime.sync_runtime_state(sandbox_id, is_running=True)
        except Exception:
            logger.debug("Failed to sync runtime state for running sandbox %s", sandbox_id, exc_info=True)

    def _release_response_gate(
        self,
        sandbox_id: SandboxId,
        pending_request: PendingSandboxResponse | None = None,
    ) -> None:
        if pending_request is None or not pending_request.request_ids:
            self.executor.clear_live_response_ready(
                sandbox_id,
                None if pending_request is None else pending_request.request_id,
            )
        else:
            for request_id in pending_request.request_ids:
                self.executor.clear_live_response_ready(sandbox_id, request_id)
        if self.response_gate_registry is None:
            return
        if pending_request is None:
            self.response_gate_registry.release(sandbox_id)
            return
        self.response_gate_registry.release_pending(
            sandbox_id,
            request_id=pending_request.request_id,
            generation=pending_request.generation,
        )

    def _dispatch_pending_coordination(self) -> None:
        with self._interceptor_lock:
            sandbox_ids = list(self._interceptor_pending)
        for sandbox_id in sandbox_ids:
            if self._should_coordinate_any_pending_request(sandbox_id):
                self._dispatch_coordination(sandbox_id)
                continue
            if not self._has_pending_response_gate(sandbox_id):
                self._refresh_interceptor_pending_state(sandbox_id)

    def _should_coordinate_any_pending_request(self, sandbox_id: SandboxId) -> bool:
        if self._txn_active(sandbox_id):
            # Inside a txn the coordination loop must not run at all: its
            # finally-release would prematurely move armed responses into
            # the staged buffer (losing per-request release granularity)
            # while the actual checkpoint flow is suppressed anyway.
            logger.debug("DIAG.coord.any.txn_active sandbox=%s", sandbox_id)
            return False
        if self.request_state_store is None or self.response_gate_registry is None:
            return False
        if not self.request_state_store.get(sandbox_id).llm_request_in_flight:
            return False
        return self.response_gate_registry.get_oldest_pending(sandbox_id) is not None

    def _next_pending_live_request(self, sandbox_id: SandboxId) -> PendingSandboxResponse | None:
        if self.response_gate_registry is None:
            return None
        return self.response_gate_registry.get_oldest_pending(sandbox_id)

    def _has_pending_response_gate(self, sandbox_id: SandboxId) -> bool:
        return self.response_gate_registry is not None and self.response_gate_registry.get_pending(sandbox_id) is not None

    def _refresh_interceptor_pending_state(self, sandbox_id: SandboxId) -> None:
        with self._interceptor_lock:
            if self._has_pending_response_gate(sandbox_id):
                self._interceptor_pending.add(sandbox_id)
            else:
                self._interceptor_pending.discard(sandbox_id)

    def _find_captured_pending_request(
        self,
        sandbox_id: SandboxId,
        manifest: CheckpointManifest,
    ) -> PendingSandboxResponse | None:
        if self.response_gate_registry is None:
            return None
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        if captured_group_kind and captured_group_id:
            return self.response_gate_registry.find_pending_group(
                sandbox_id,
                request_group_kind=captured_group_kind,
                request_group_id=captured_group_id,
            )
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            return None
        raw_generation = manifest.metadata.get(_CAPTURED_REQUEST_GENERATION)
        if raw_generation is not None:
            try:
                generation = int(raw_generation)
            except (TypeError, ValueError):
                generation = None
            else:
                return self.response_gate_registry.get_pending_generation(sandbox_id, generation)
        return self.response_gate_registry.find_pending_request(sandbox_id, captured_request_id)

    def _should_resume_after_checkpoint(
        self,
        job: CheckpointJob | None,
        result: CheckpointResult | None,
    ) -> bool:
        if job is None:
            return True
        if result is None or result.status.value != "succeeded":
            return True
        if isinstance(self.runtime, InMemoryRuntime):
            return True
        return job.leave_running


def build_default_system(
    *,
    storage_root: str | Path,
    runtime: str = "runc",
    scheduler_config: SchedulerConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    storage_config: StorageConfig | None = None,
    runc_runtime_options: RuncRuntimeOptions | None = None,
    use_in_memory_telemetry: bool = True,
    telemetry_config: TelemetryConfig | None = None,
    request_state_store: InMemoryRequestStateStore | None = None,
    host_inspector_url: str | None = None,
    scheduler_policy: SchedulerPolicy | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    relaunch_handler: Callable[[SandboxId, str, bool], None] | None = None,
    enforce_restore_checkpoint_validation: bool = False,
    relaunch_on_restore_failure: bool = False,
) -> CrabSystem:
    logger.info("Building default crab system with runtime=%s storage_root=%s", runtime, storage_root)
    scheduler_cfg = scheduler_config or SchedulerConfig()
    executor_cfg = executor_config or ExecutorConfig()
    store_cfg = storage_config or StorageConfig(root_dir=Path(storage_root))

    host_inspector_client = (
        HostInspectorServiceClient(host_inspector_url) if host_inspector_url is not None else None
    )

    telemetry_cfg = telemetry_config or TelemetryConfig(enabled=True)
    telemetry = build_configured_telemetry_sink(
        telemetry_cfg,
        keep_in_memory_fallback=use_in_memory_telemetry,
    )

    if runtime == "docker":
        runtime_impl = InMemoryRuntime(name="docker", host_inspector_client=host_inspector_client)
    elif runtime == "runc":
        runtime_impl = RuncRuntime(
            host_inspector_client=host_inspector_client,
            telemetry=telemetry,
            options=runc_runtime_options,
        )
    else:
        raise ValueError(f"unsupported runtime: {runtime}")
    storage = checkpoint_manager or LocalCheckpointManager(
        store_cfg,
        runtime_image_path_in_use=runtime_impl.runtime_image_path_in_use,
        destroy_filesystem_ref=runtime_impl.destroy_filesystem_ref,
    )
    if checkpoint_manager is None:
        # When the caller supplied their own manager (likely wrapped in a
        # retention policy), late-bind the safety predicate so it can
        # still defer pruning a runtime tree with an active lazy-pages
        # daemon. Caller-supplied managers may not be ``LocalCheckpointManager``;
        # ``setattr``-style installation would be wrong on those, so we
        # only call the setter when the manager exposes it.
        pass
    setter = getattr(storage, "set_runtime_image_path_in_use", None)
    if callable(setter):
        setter(runtime_impl.runtime_image_path_in_use)
    fs_ref_setter = getattr(storage, "set_destroy_filesystem_ref", None)
    if callable(fs_ref_setter):
        fs_ref_setter(runtime_impl.destroy_filesystem_ref)
    request_store = request_state_store or InMemoryRequestStateStore()
    response_gate_registry = SandboxResponseGateRegistry()
    base_inspector: SandboxInspector
    if host_inspector_client is not None:
        base_inspector = RemoteSandboxInspector(host_inspector_client)
    else:
        base_inspector = EBPFSandboxInspector()
    inspector = RequestAwareSandboxInspector(base_inspector, request_store)

    process_c = AdapterProcessCWorker(runtime_impl)
    process_r = AdapterProcessRWorker(runtime_impl)
    fs_c = AdapterFileSystemCWorker(runtime_impl)
    fs_r = AdapterFileSystemRWorker(runtime_impl)

    c_worker = DefaultCWorker(
        process_c,
        fs_c,
        storage,
        runtime_impl,
        checkpoint_guard=_checkpoint_guard_from_inspector(inspector),
        telemetry=telemetry,
        step_workers=executor_cfg.resolved_composite_step_workers,
    )
    r_worker = DefaultRWorker(
        process_r, fs_r, storage, telemetry=telemetry, runtime=runtime_impl
    )

    executor = CRExecutor(executor_cfg, c_worker, r_worker, telemetry)
    scheduler = CRScheduler(
        scheduler_cfg,
        inspector,
        runtime_impl,
        InMemorySchedulerStateStore(),
        telemetry,
        scheduler_policy,
    )

    logger.debug(
        "Constructed crab components runtime=%s telemetry=%s",
        runtime,
        type(telemetry).__name__,
    )
    return CrabSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=inspector,
        runtime=runtime_impl,
        telemetry=telemetry,
        request_state_store=request_store,
        response_gate_registry=response_gate_registry,
        relaunch_handler=relaunch_handler,
        recovery_delay_seconds=0.0,
        enforce_restore_checkpoint_validation=enforce_restore_checkpoint_validation,
        relaunch_on_restore_failure=relaunch_on_restore_failure,
    )
