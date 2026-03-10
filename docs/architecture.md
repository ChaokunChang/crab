# Agent-CR Current Architecture

This document describes the current `agent_cr` implementation with a real runtime path for `runc`/CRIU, ZFS-backed filesystem state, and an eBPF-centered inspector.

## High-Level Component Diagram

```mermaid
flowchart TD
    Client[Library Client] -->|build_default_system| System[AgentCRSystem]

    System --> Scheduler[CRScheduler]
    System --> Executor[CRExecutor]
    System --> Storage[LocalCheckpointManager]
    System --> Inspector[EBPFSandboxInspector]
    System --> SandboxMgr[RuncSandboxManager / InMemory fallback]
    System --> Telemetry[InMemoryTelemetrySink / NoopTelemetrySink]

    Scheduler --> Policy[DefaultHeuristicPolicy]
    Scheduler --> State[InMemorySchedulerStateStore]
    Scheduler --> Inspector
    Scheduler --> Telemetry

    Executor --> CWorker[DefaultCWorker]
    Executor --> RWorker[DefaultRWorker]
    Executor --> Telemetry

    CWorker --> ProcessC[AdapterProcessCWorker]
    CWorker --> FsC[AdapterFileSystemCWorker]
    CWorker --> Storage

    RWorker --> ProcessR[AdapterProcessRWorker]
    RWorker --> FsR[AdapterFileSystemRWorker]
    RWorker --> Storage

    ProcessC --> Runtime[SandboxRuntimeAdapter]
    FsC --> Runtime
    ProcessR --> Runtime
    FsR --> Runtime

    Runtime --> Docker[DockerRuntimeAdapter stub]
    Runtime --> Runc[RuncRuntimeAdapter + CRIU/ZFS]
```

## Checkpoint and Restore Flow

### Checkpoint

1. `CRScheduler` gets `SandboxSnapshot` from `SandboxInspector`.
2. `CRPolicy` (`DefaultHeuristicPolicy`) returns `ScheduleDecision`.
3. If `should_checkpoint=True`, a `CheckpointJob` is queued in `InMemorySchedulerStateStore`.
4. `CRExecutor.run_checkpoint(...)` dispatches to `DefaultCWorker` using thread-pool workers.
5. `DefaultCWorker` calls:
   - `AdapterProcessCWorker`, which runs `runc checkpoint` and archives the CRIU image directory.
   - `AdapterFileSystemCWorker`, which runs `zfs snapshot` and stores the snapshot metadata artifact.
6. `LocalCheckpointManager` stores artifacts and the `CheckpointManifest` (`v1` + integrity hash).
8. `CheckpointResult` is returned; scheduler can mark checkpoint complete.

### Restore

1. `CRExecutor.run_restore(...)` dispatches to `DefaultRWorker`.
2. `DefaultRWorker` loads manifest from `LocalCheckpointManager`.
3. It restores filesystem state first through `AdapterFileSystemRWorker` using `zfs rollback`.
4. It then restores process state through `AdapterProcessRWorker` by extracting the CRIU artifact and running `runc restore`.
5. `RestoreResult` is returned with standardized status and telemetry.

## Data and Contracts

- IDs: `SandboxId`, `CheckpointId`, `JobId`.
- Job/Result types: `CheckpointJob`, `RestoreJob`, `CheckpointResult`, `RestoreResult`, `JobRecord`.
- Manifest: `CheckpointManifest` with:
  - `schema_version = v1`
  - runtime metadata
  - process/filesystem artifact references
  - integrity (`manifest_sha256`)
- Contracts are defined in `agent_cr/contracts.py` for all extension points.

## Runtime Behavior

- `RuncRuntimeAdapter` executes real commands for process and filesystem state.
- `DockerRuntimeAdapter` remains a compatibility stub and returns planned commands only.
- Local storage (`LocalCheckpointManager`) is fully implemented.
- Scheduler/executor remain in-process.
- The default sandbox manager for `runc` is `RuncSandboxManager`, which drives `runc run|kill|delete` and ZFS dataset lifecycle.
- Inspector state is derived from eBPF events via `EBPFSandboxInspector`.

## Module Map

- System assembly: `agent_cr/system.py`
- Scheduling/state: `agent_cr/scheduler.py`
- Execution: `agent_cr/executor.py`
- Workers: `agent_cr/workers/`
- Runtime adapters: `agent_cr/runtime/`
- Storage: `agent_cr/storage/`
- Policies: `agent_cr/policy/`
- Domain models/config/ids: `agent_cr/models.py`, `agent_cr/config.py`, `agent_cr/ids.py`
- Telemetry/interceptor/sandbox lifecycle/inspector:
  - `agent_cr/telemetry.py`
  - `agent_cr/interceptor.py`
  - `agent_cr/sandbox_manager.py`
  - `agent_cr/inspector.py`

## Notes

- `legacy/` remains separate and untouched by this architecture.
- Benchmarks for this architecture are in `benchmarks/`.
- Tests for this architecture are in `tests/`.
