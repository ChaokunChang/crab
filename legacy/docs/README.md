# Archived documentation

These documents describe earlier benchmark campaigns, implementation PRs, or
an exhaustive internal configuration surface. They are kept for archaeology
and research reproducibility, not as instructions for the current Crab v0.

For current user documentation, start at [`../../docs/README.md`](../../docs/README.md).

- `claude-code-integration.md`: benchmark-specific Claude Code replay work.
- `configuration-reference-full.md`: the former mixed SDK/benchmark/internal
  configuration catalog.
- `incremental-fork-restore-analysis.md`: results from a specific speculative
  execution experiment.
- `lazy-restore-safety-contract.md`: design and PR notes for lazy restore.
- `replay-cadence-handling.md`: Terminus replay timing notes.
- `speculative-execution-benchmark.md`: historical speculative benchmark
  setup and results.
- `repository-agent-notes.md`: superseded repository instructions for coding
  agents; commands in it may refer to removed benchmark files.

Archived documents may contain stale paths, defaults, commands, and measured
results. Do not copy their configuration into a production deployment without
checking the current code and [`config/crab.yaml`](../../config/crab.yaml).
