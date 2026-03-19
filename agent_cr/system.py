from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    utc_now,
)
from .runtime import InMemoryRuntime, RuncRuntime, RuncRuntimeOptions
from .scheduler import CRScheduler, InMemorySchedulerStateStore, SchedulerPolicy
from .storage import LocalCheckpointManager
from .telemetry import CompositeTelemetrySink, InMemoryTelemetrySink, JsonlTelemetrySink, NoopTelemetrySink
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
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"


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
class AgentCRSystem:
    scheduler: CRScheduler
    executor: CRExecutor
    storage: CheckpointManager
    inspector: SandboxInspector
    runtime: Runtime
    telemetry: TelemetrySink
    request_state_store: InMemoryRequestStateStore | None = None
    response_gate_registry: SandboxResponseGateRegistry | None = None
    relaunch_handler: Callable[[SandboxId, str], None] | None = None
    extra_checkpoint_metadata_provider: Callable[[SandboxId], dict[str, object]] | None = None
    restore_metadata_handler: Callable[[SandboxId, CheckpointManifest], None] | None = None
    recovery_delay_seconds: float = 0.0
    enforce_restore_checkpoint_validation: bool = False
    _interceptor_lock: Lock = field(init=False, repr=False)
    _interceptor_pending: set[SandboxId] = field(init=False, repr=False)
    _coordination_lock: Lock = field(init=False, repr=False)
    _active_coordination: set[SandboxId] = field(init=False, repr=False)
    _recovery_lock: Lock = field(init=False, repr=False)
    _recovery_queue: Queue[RecoveryEvent | None] = field(init=False, repr=False)
    _recovery_records: dict[SandboxId, RecoveryRecord] = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _monitor_thread: Thread | None = field(init=False, repr=False, default=None)
    _recovery_thread: Thread | None = field(init=False, repr=False, default=None)

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

    def start(self) -> None:
        with self._coordination_lock:
            request_running = self._monitor_thread is not None and self._monitor_thread.is_alive()
            recovery_running = self._recovery_thread is not None and self._recovery_thread.is_alive()
            if request_running and recovery_running:
                return
            self._stop_event.clear()
            if self.response_gate_registry is not None:
                self.response_gate_registry.enable()
            if self.request_state_store is not None and not request_running:
                self._monitor_thread = Thread(target=self._run_monitor_loop, name="agent-cr-system", daemon=True)
                self._monitor_thread.start()
            if not recovery_running:
                self._recovery_thread = Thread(target=self._run_recovery_loop, name="agent-cr-recovery", daemon=True)
                self._recovery_thread.start()
        logger.info("Started AgentCRSystem background loops")

    def stop(self) -> None:
        self._stop_event.set()
        if self.response_gate_registry is not None:
            self.response_gate_registry.disable()
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        self._recovery_queue.put(None)
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._monitor_thread = None
        recovery_thread = self._recovery_thread
        if recovery_thread is not None:
            recovery_thread.join(timeout=5.0)
        self._recovery_thread = None
        logger.info("Stopped AgentCRSystem background loops")

    def checkpoint_once(self, sandbox_id: SandboxId, leave_running: bool=False) -> CheckpointResult:
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
                leave_running=leave_running,
                metadata=self._build_checkpoint_metadata(sandbox_id),
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
        started = utc_now()
        restore_checkpoint_id = CheckpointId(str(checkpoint_id))
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
            return RestoreResult(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                checkpoint_id=restore_checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=FailureCode.VALIDATION_ERROR,
                message=restore_message,
            )
        self.runtime.prepare_for_restore(sandbox_id)
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=restore_checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
        )
        result = self.executor.run_restore(job)
        if result.status.value == "succeeded":
            restore_manifest = None
            if self.restore_metadata_handler is not None:
                restore_manifest = self._resolve_restore_manifest(sandbox_id, restore_checkpoint_id)
            self.runtime.mark_restored(sandbox_id)
            self.storage.handle_restore_complete(sandbox_id, result.checkpoint_id)
            if self.restore_metadata_handler is not None and restore_manifest is not None:
                self.restore_metadata_handler(sandbox_id, restore_manifest)
        logger.info(
            "Manual restore for sandbox %s checkpoint=%s finished with status=%s",
            sandbox_id,
            restore_checkpoint_id,
            result.status.value,
        )
        return result

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

    def _handle_recovery_event(self, event: RecoveryEvent) -> None:
        if not self._acquire_coordination(event.sandbox_id):
            return
        started = utc_now()
        checkpoint_id = None
        restore_manifest: CheckpointManifest | None = None
        status = "failed"
        message = None
        try:
            logger.info(
                "Handling recovery event sandbox=%s event_type=%s reason=%s",
                event.sandbox_id,
                event.event_type,
                event.reason,
            )
            self.telemetry.emit_event(
                "recovery.started",
                {
                    "sandbox_id": str(event.sandbox_id),
                    "event_type": event.event_type,
                },
            )
            if event.event_type == "preemption":
                checkpoint_id = self._select_recovery_checkpoint_after(
                    event.sandbox_id,
                    observed_after=event.observed_at,
                )
                if checkpoint_id is not None:
                    logger.info(
                        "Reusing checkpoint already captured after preemption notice sandbox=%s checkpoint=%s",
                        event.sandbox_id,
                        checkpoint_id,
                    )
                else:
                    logger.info("Triggering preemption checkpoint flow for sandbox=%s", event.sandbox_id)
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
                checkpoint_id = self._select_recovery_checkpoint(event.sandbox_id)
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
                restore_manifest = self._resolve_restore_manifest(event.sandbox_id, checkpoint_id)
                restore_result = self.restore_once(event.sandbox_id, checkpoint_id)
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
                elif self.relaunch_handler is not None:
                    logger.warning(
                        "Recovery restore failed; invoking relaunch handler sandbox=%s checkpoint=%s message=%s",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
                    self.relaunch_handler(event.sandbox_id, event.event_type)
                    status = "relaunched"
                    message = "restore_failed_relaunch_handler_invoked"
                else:
                    status = "restore_failed"
                    message = restore_result.message
                    logger.warning(
                        "Recovery restore failed sandbox=%s checkpoint=%s message=%s",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
            elif self.relaunch_handler is not None:
                logger.info("No checkpoint available; invoking relaunch handler for sandbox=%s", event.sandbox_id)
                self.relaunch_handler(event.sandbox_id, event.event_type)
                status = "relaunched"
                message = "relaunch_handler_invoked"
            else:
                status = "no_checkpoint"
                message = "no restorable checkpoint available"
                logger.warning("No checkpoint available for sandbox=%s and no relaunch handler configured", event.sandbox_id)
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
            with self._interceptor_lock:
                self._interceptor_pending.discard(event.sandbox_id)
            self._release_coordination(event.sandbox_id)

    def _execute_checkpoint_flow(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        decision = self.scheduler.query_checkpoint(sandbox_id)
        if not decision.should_checkpoint:
            return None
        checkpoint_metadata = self._build_checkpoint_metadata(sandbox_id)

        job = CheckpointJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason=decision.reason,
            checkpoint_process=decision.checkpoint_process,
            checkpoint_filesystem=decision.checkpoint_filesystem,
            leave_running=decision.leave_running,
            metadata={"policy": decision.policy_name, **decision.metadata, **checkpoint_metadata},
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

    def _latest_checkpoint_id(self, sandbox_id: SandboxId):
        checkpoints = self.storage.list_checkpoints(sandbox_id)
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _build_checkpoint_metadata(self, sandbox_id: SandboxId) -> dict[str, object]:
        metadata: dict[str, object] = {_CAPTURES_INFLIGHT_LLM: False}
        if self.extra_checkpoint_metadata_provider is not None:
            try:
                metadata.update(self.extra_checkpoint_metadata_provider(sandbox_id))
            except Exception:
                logger.exception("Failed to collect extra checkpoint metadata for sandbox=%s", sandbox_id)
        if self.request_state_store is None or self.response_gate_registry is None:
            return metadata
        request_state = self.request_state_store.get(sandbox_id)
        pending = self.response_gate_registry.get_pending(sandbox_id)
        if not request_state.llm_request_in_flight or pending is None:
            return metadata
        if request_state.last_request_id != pending.request_id:
            return metadata
        metadata[_CAPTURES_INFLIGHT_LLM] = True
        metadata[_CAPTURED_REQUEST_ID] = pending.request_id
        if request_state.last_llm_provider is not None:
            metadata[_CAPTURED_REQUEST_PROVIDER] = request_state.last_llm_provider
        if request_state.last_llm_request_started_at is not None:
            metadata[_CAPTURED_REQUEST_STARTED_AT] = request_state.last_llm_request_started_at.isoformat()
        logger.info(
            "Checkpoint captured live request sandbox=%s request_id=%s",
            sandbox_id,
            pending.request_id,
        )
        self.telemetry.emit_event(
            "checkpoint.captured_live_request",
            {
                "sandbox_id": str(sandbox_id),
                "request_id": pending.request_id,
            },
        )
        return metadata

    def _validate_restore_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            return None
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            return f"checkpoint {checkpoint_id} advertises live-request restore without captured_request_id"
        pending = None if self.response_gate_registry is None else self.response_gate_registry.get_pending(sandbox_id)
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

    def _release_checkpoint_response_gate(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        manifest: CheckpointManifest | None = None,
    ) -> bool:
        if self.response_gate_registry is None:
            return False
        if manifest is None:
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            return False
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            return False
        pending = self.response_gate_registry.get_pending(sandbox_id)
        if pending is None or pending.request_id != captured_request_id:
            return False
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
            snapshot = self.inspector.inspect(sandbox_id)
        except Exception:
            snapshot = None
        if snapshot is not None and not snapshot.is_running:
            return
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

    def _release_response_gate(self, sandbox_id: SandboxId) -> None:
        if self.response_gate_registry is not None:
            self.response_gate_registry.release(sandbox_id)

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
    relaunch_handler: Callable[[SandboxId, str], None] | None = None,
    enforce_restore_checkpoint_validation: bool = False,
) -> AgentCRSystem:
    logger.info("Building default agent-cr system with runtime=%s storage_root=%s", runtime, storage_root)
    scheduler_cfg = scheduler_config or SchedulerConfig()
    executor_cfg = executor_config or ExecutorConfig()
    store_cfg = storage_config or StorageConfig(root_dir=Path(storage_root))

    host_inspector_client = (
        HostInspectorServiceClient(host_inspector_url) if host_inspector_url is not None else None
    )

    telemetry_cfg = telemetry_config or TelemetryConfig(enabled=True, keep_in_memory_copy=use_in_memory_telemetry)
    if not telemetry_cfg.enabled:
        telemetry = NoopTelemetrySink()
    else:
        sinks: list[TelemetrySink] = []
        if telemetry_cfg.keep_in_memory_copy:
            sinks.append(InMemoryTelemetrySink())
        if telemetry_cfg.jsonl_path is not None:
            sinks.append(JsonlTelemetrySink(Path(telemetry_cfg.jsonl_path)))
        if not sinks:
            sinks.append(NoopTelemetrySink())
        telemetry = sinks[0] if len(sinks) == 1 else CompositeTelemetrySink(sinks)

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
    storage = checkpoint_manager or LocalCheckpointManager(store_cfg)
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
    )
    r_worker = DefaultRWorker(process_r, fs_r, storage, telemetry=telemetry)

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
        "Constructed agent-cr components runtime=%s telemetry=%s",
        runtime,
        type(telemetry).__name__,
    )
    return AgentCRSystem(
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
    )
