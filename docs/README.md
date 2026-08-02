# Crab documentation

Start with these documents if you want to run Crab:

- [Installation](installation.md): supported host, installer behavior, ZFS
  pool options, and dependency checks.
- [Getting started](getting-started.md): launch a sandbox, create and inspect
  checkpoints, mutate state, and restore.
- [SDK](sdk.md): create sandboxes and attach agents from Python.
- [Bring your own agent](byo-agent.md): implement the small `Agent` contract.
- [Daemon and CLI](daemon.md): socket, lifecycle, CLI commands, and current
  operational limitations.
- [Configuration reference](configuration-reference.md): scheduler, executor,
  retention, runtime, and telemetry settings.

Agent examples:

- [iFlow replay](sdk-iflow-replay.md): run a recorded iFlow trace without a
  live model API key.
- [Claude Code replay internals](claude-code-integration.md): implementation
  and benchmark-specific integration decisions.
- [Agent integration notes](agent-integration-notes.md): checkpoint-safe agent
  process and I/O patterns learned from integrations.

Architecture and operator internals:

- [Architecture](architecture.md)
- [Telemetry](telemetry.md)
- [Replay cadence handling](replay-cadence-handling.md)
- [Lazy restore safety contract](lazy-restore-safety-contract.md)

Research and benchmark notes:

- [Speculative execution benchmark](speculative-execution-benchmark.md)
- [Incremental fork/restore analysis](incremental-fork-restore-analysis.md)

The research notes document experiments and implementation decisions. They are
not required for the v0 installation or the manual checkpoint/restore flow.
