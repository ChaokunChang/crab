# Agent Integration Notes

This document captures integration lessons that are not specific to a single agent, but tend to matter when an agent runs inside a sandbox and we rely on checkpoint/restore.

## Checkpoint-safe wrapper I/O

If an agent wrapper must survive checkpoint/restore, especially mixed restore where we intentionally combine:

- an older process checkpoint
- with a newer filesystem checkpoint

then the wrapper should avoid keeping long-lived file descriptors open against mutable mounted files.

### Why this matters

Mounted host directories are the right place for agent state that we do not want to count as sandbox-rootfs churn. They are excluded from rootfs monitoring and avoid unnecessary filesystem checkpoints.

However, there is an important exception:

- a long-lived process can still hold an open fd to a mounted file
- if that file changes after the checkpoint is taken
- then an older process restore may fail because CRIU expects the original file metadata / size for that fd

So "mounted" is good for state placement, but "open mutable fd on mounted file" is dangerous for process restore.

### Safe pattern

Prefer this split:

- persistent agent state:
  - mount it from the host work directory
  - examples: agent home, caches, session files, structured debug files, completion markers
- live wrapper `stdout` / `stderr`:
  - send to `/dev/null`
  - or another sink that is not expected to drift under a restored process

If the agent provides its own debug-file option that opens/closes files internally, that is usually safer than redirecting the wrapper's process-level `stdout` / `stderr` to a mounted log file.

### Good defaults for future integrations

When integrating a new agent, treat these as the default rules:

1. Put agent home/state under a mounted host directory when that state should not force rootfs checkpoints.
2. Do not bind long-lived wrapper `stdout` / `stderr` to a mutable mounted file.
3. If logs are needed, prefer:
   - agent-native debug logging APIs
   - host-side log capture
   - `/dev/null` for wrapper stdio if the logs are low value
4. Completion markers are fine on the mounted host directory if they are only written at task end and are not kept open.
5. If the wrapper process itself is long-lived and mostly orchestration, add an inspector ignore rule so it does not create permanent process-change noise between LLM request windows.

### Concrete examples

- `iflow` already follows the safe stdio pattern:
  - the long-lived wrapper writes task markers on the mounted host state
  - wrapper `stdout` / `stderr` go to `/dev/null`
- `claude_code` originally redirected wrapper `stdout` / `stderr` to a mounted `claude_code.output.log`
  - this broke mixed restore because the restored baseline process still had fds for that file while the file had already grown on the newer mounted filesystem
  - the fix was to keep Claude's `--debug-file` on the mounted host state, but move wrapper stdio back to `/dev/null`

### Rule of thumb

If a file is both:

- mutable after checkpoint
- and held open by the process you may later restore

assume it is restore-sensitive until proven otherwise.
