# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Tests
```bash
# Full test suite
python3 -m unittest discover -s tests -v

# Specific test modules
python3 -m unittest tests.test_system_integration tests.test_interceptor -v

# Single test class or method
python3 -m unittest tests.test_system_integration.TestSystemIntegration.test_name -v
```

Never run Full test suite, it is very slow, and there are some known issues.

### Benchmarks
```bash
# Microbenchmark
python3 benchmarks/bench_agent_cr_micro.py --iters 1000 --storage-iters 200 --executor-jobs 64

# YAML-driven scenarios
python3 -m benchmarks.run --config benchmarks/examples/e2e.manual.yaml
python3 -m benchmarks.run --config benchmarks/examples/fault.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/spot.auto.yaml
python3 -m benchmarks.run --config benchmarks/examples/tree.auto.yaml
```

## Architecture

`agent-cr` is a Python library for fault-tolerant agent execution via process and filesystem checkpointing. It coordinates CRIU (process snapshots), runc (container runtime), and ZFS (filesystem snapshots) to checkpoint and restore agent sandboxes.

### Core System (`agent_cr/`)

`AgentCRSystem` (`system.py`) is the top-level orchestrator. It composes:

- **`CRScheduler`** (`scheduler.py`) — decides *when* to checkpoint based on policy (FaultTolerance, SpotPreemption, TreeSearch, Default)
- **`CRExecutor`** (`executor.py`) — executes checkpoint/restore jobs via worker threads
- **`CheckpointManager`** (`storage/local.py`) — stores checkpoint manifests and data on disk; wrapped by retention policies (`storage/policies.py`: KeepAll, LatestOnly, DeleteAfterRestore)
- **`Runtime`** (`runtime/`) — abstracts container operations; `InMemoryRuntime` for testing, `RuncRuntime` (`runtime/runc.py`) for production (real CRIU + ZFS)
- **`SandboxInspector`** (`inspector.py`, `remote_inspector.py`) — detects process/filesystem changes in sandboxes; production uses eBPF via `host_inspector/`
- **`RequestInterceptor`** (`interceptor.py`) — intercepts in-flight LLM requests so checkpoints can be coordinated with request boundaries
- **`SandboxResponseGateRegistry`** / **`InMemoryRequestStateStore`** (`interceptor.py`) — buffers responses and tracks request lifecycle

### Checkpoint/Restore Flow

1. **Checkpoint**: Pause sandbox → CRIU image (process) + ZFS snapshot (filesystem) → store `CheckpointManifest`
2. **Restore**: Load manifest → ZFS rollback (filesystem) → `runc restore` (process) → resume
3. **Recovery**: Fault/preemption detected → select best checkpoint → restore → continue

Workers are split into `ProcessCheckpointWorker` / `FilesystemCheckpointWorker` (and restore equivalents), composed by `CompositeWorker` (`workers/composite.py`).

### Runtime Modes

- **`docker` / in-memory** (testing): logs commands but doesn't execute; allows unit tests without real containers
- **`runc`** (production): real CRIU + ZFS; requires host setup (ZFS pool, runc, CRIU installed)

### Benchmark Harness (`benchmarks/`)

YAML-driven scenarios in `benchmarks/examples/`. Key fields:
```yaml
scenario: e2e | fault | spot | tree
mode: manual | auto
provider: openai | anthropic
agent: simulated | iflow
llm_service: simulated | manual | iflow_trace_replay
sandboxes: <count>
benchmark_root: path/to/benchmark-runs
reuse_zpool: bool   # keep ZFS pool across runs
```

`RealHostScenarioBase` (`benchmarks/real_host_scenario_base.py`) is the shared harness for scenarios that run against a real host with actual runc + ZFS.

### Integrations (`integrations/`)

- **`agents/`**: `SimulatedAgent`, `iFlowAgent` — implement the agent contract
- **`llm_services/`**: `SimulatedLLMService`, `ManualLLMService`, `iFlowTraceReplayService` — injectable LLM backends
- **`sandboxes/`**: runtime launcher, network setup, Docker Compose integration

### Key Data Models (`agent_cr/models.py`)

`CheckpointManifest`, `RestoreJob`, `SandboxSnapshot`, `CheckpointResult` — the core data structures passed between components.

### Configuration (`agent_cr/config.py`)

`SchedulerConfig`, `ExecutorConfig`, `StorageConfig`, `TelemetryConfig` — dataclasses configuring each subsystem.
