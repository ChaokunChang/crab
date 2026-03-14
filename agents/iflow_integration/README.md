# `iflow-cli` Integration

This folder contains the maintained real-agent harness for running `iflow-cli`
inside an `agent-cr` sandbox and validating host-inspector behavior while
filtering out `iflow`'s own long-lived runtime noise.

## Prerequisites

- root
- `docker`
- `runc`
- `criu`
- `zfs`
- `ip`
- build dependencies for `agent_cr/host_inspector/bpf`
- vendored `iflow` artifacts under `agents/iflow_integration/cache`

Required cache files:

- `node-v22.18.0-linux-x64.tar.xz`
- `iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz`

## What Lives Here

- `Dockerfile`: base image with the OS packages needed by `iflow`
- `install_iflow.sh`: maintained installer owned by `agent-cr`
- `image.py`: helper to build/export the fixture rootfs
- `service.py`: scripted OpenAI-compatible responder that emits native `iflow` tool calls
- `harness.py`: mounted-runtime preparation, bind-mount/state setup, bundle config helpers, and bridge-netns setup

Set `AGENT_CR_IFLOW_CACHE_DIR` only if you want to override the repo-default cache location.

## Build The Rootfs

```bash
python3 -m agents.iflow_integration build-rootfs \
  --tag agent-cr-iflow-agent:latest \
  --output-dir /tmp/agent-cr-iflow-rootfs
```

This builds the maintained image and exports its rootfs.

## Run The Scripted LLM Service

```bash
python3 -m agents.iflow_integration serve-scripted-llm \
  --port 8091 \
  --idle-delay-ms 2000
```

The service speaks OpenAI-compatible `POST /v1/chat/completions`.

## Automated Test

```bash
python3 -m unittest \
  tests.test_iflow_integration_real.IFlowRealIntegrationTests.test_real_iflow_cli_host_inspector_matrix
```

The test does the full flow:

- starts the host inspector
- starts the host-side interceptor
- creates a bridged network namespace so the sandbox reaches `172.17.0.1:<interceptor_port>`
- launches a real `runc` sandbox
- prepares a host-side `iflow` runtime from the cached Node and `iflow` tarballs
- bind-mounts that runtime read-only into `/opt/iflow-runtime`
- bind-mounts `iflow` private writable state outside the container rootfs:
  - `/root/.iflow`
  - `/root/.npm`
  - `/opt/iflow-logs`
- registers generic host-inspector ignore rules so host-inspector excludes the long-lived `iflow` Node processes
- establishes an explicit inspector baseline after `iflow` startup reaches its first LLM wait
- validates transient-process, filesystem-write, and detached-daemon phases
- attempts a real checkpoint during the final-response wait window
- writes an observation report under the test temp directory

## Manual Notes

Inside the sandbox, `iflow` is configured to use:

- `baseUrl=http://172.17.0.1:<interceptor_port>/v1`
- `selectedAuthType=openai-compatible`
- a bind-mounted runtime rooted at `/opt/iflow-runtime`

No in-container proxy is used. In the automated test path, the host-side
interceptor is the only LLM-facing proxy layer. In the manual path below, the
sandbox talks directly to the manual LLM server.

The default runtime mode is mounted runtime, not in-container install. The
runtime is prepared on the host, then mounted into the sandbox read-only.

The default ignore rule shape is generic and process-identity based:

- executable basename `node`
- cmdline contains `/opt/iflow-runtime/node/bin/node`
- cmdline contains `@iflow-ai/iflow-cli/bundle/`

That excludes the real `iflow` runtime from tracked process and filesystem
status, while tool child processes like `/bin/sh` or `python3` are still tracked.

## Manual Interactive Flow

The package also supports a manual control loop where a real `iflow` sandbox
waits on a local OpenAI-compatible server, and you enqueue `run_shell_command`
tool calls by hand.

### Quickstart: 2-Terminal Wrapper

Terminal 1:

```bash
cd /root/workspace/agent-cr
sudo agents/iflow_integration/start-manual-demo.sh
```

That single command:

- builds the host-inspector helper
- starts the host inspector
- starts the manual OpenAI-compatible LLM server
- launches the real `iflow` sandbox
- prints the terminal-2 commands you need
- cleans up the sandbox, netns, work root, and background services on `Ctrl-C`

Terminal 2 then only needs the printed `reset`, `watch`, and `enqueue` commands.

If you want to preserve the work tree for debugging, start it with `KEEP_WORK_ROOT=1`.

The full manual commands are below if you want to run each piece separately.

### 1. Start Host Inspector

Use the same `runc` state root that the manual sandbox launcher will use:

```bash
export WORK_ROOT=/tmp/iflow-manual
export HOST_INSPECTOR_URL=http://127.0.0.1:9782

python3 -m agent_cr.host_inspector \
  --port 9782 \
  --runc-state-root "$WORK_ROOT/runtime-state"
```

### 2. Start The Manual LLM Server

```bash
export MANUAL_LLM_URL=http://127.0.0.1:8091

python3 -m agents.iflow_integration serve-manual-llm \
  --host 0.0.0.0 \
  --port 8091
```

This server exposes:

- `POST /v1/chat/completions`: used by `iflow`
- `POST /control/run_shell_command`: enqueue a manual `run_shell_command` tool call
- `POST /control/final_response`: enqueue a final assistant message
- `GET /control/state`: inspect queued responses and request history

### 3. Start The Interceptor

Route sandbox requests through the interceptor so sandbox identity is resolved
from the bridged sandbox IP instead of a hidden default.

```bash
export INTERCEPTOR_URL=http://127.0.0.1:8092

python3 -m agents.iflow_integration serve-manual-interceptor \
  --host 0.0.0.0 \
  --port 8092 \
  --upstream-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --sandbox-ip 172.17.0.240
```

### 4. Launch The Real `iflow` Sandbox

```bash
python3 -m agents.iflow_integration launch-manual-sandbox \
  --work-root "$WORK_ROOT" \
  --sandbox-id sbx-iflow-manual \
  --host-inspector-url "$HOST_INSPECTOR_URL" \
  --llm-base-url http://172.17.0.1:8092/v1 \
  --sandbox-ip 172.17.0.240
```

The command prints a JSON summary containing:

- sandbox id
- `logs_dir`
- `runtime_state_root`
- mounted-state paths
- ignore rules used for the long-lived `iflow` Node processes

At this point, `iflow` is running and blocked on the manual LLM server, waiting
for the next completion response.

### 5. Reset And Watch Inspector Status

Reset once before each manual tool call:

```bash
curl -s -X POST "$HOST_INSPECTOR_URL/reset" \
  -H 'Content-Type: application/json' \
  -d '{"sandbox_id":"sbx-iflow-manual"}'
```

Then watch it:

```bash
python3 -m agent_cr.host_inspector.watch \
  --base-url "$HOST_INSPECTOR_URL" \
  --interval 0.5 \
  sbx-iflow-manual
```

### 6. Manually Send A `run_shell_command` Tool Call

Example: transient no-op command

```bash
python3 -m agents.iflow_integration enqueue-run-shell-command \
  --base-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --command 'sh -lc "printf noop >/dev/null"'
```

Example: sticky filesystem write

```bash
python3 -m agents.iflow_integration enqueue-run-shell-command \
  --base-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --command 'sh -lc "mkdir -p /work/iflow-probe && printf iflow-artifact >/work/iflow-probe/artifact.txt"'
```

Example: detached daemon

```bash
python3 -m agents.iflow_integration enqueue-run-shell-command \
  --base-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --command 'sh -lc "mkdir -p /work/iflow-probe && python3 -m http.server 8123 >/work/iflow-probe/http.log 2>&1 & echo $! >/work/iflow-probe/http.pid"'
```

After each tool call:

- `iflow` executes the tool
- it sends a new `/v1/chat/completions` request
- the interceptor forwards that request with the resolved sandbox id
- the manual server blocks again, waiting for your next queued response
- `watch.py` shows when `process_changed` or `filesystem_changed` flips

You can inspect the manual server queue and history at any time:

```bash
python3 -m agents.iflow_integration manual-llm-state --base-url "$MANUAL_LLM_URL"
```

### 7. Manually Checkpoint, Restore, And Enter The Sandbox

Once the sandbox is running, you can trigger checkpoint/restore directly from the
same `WORK_ROOT` used to launch it.

Create a long-running counter workload:

```bash
python3 -m agents.iflow_integration enqueue-run-shell-command \
  --base-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --command 'bash -lc '"'"'mkdir -p /work/iflow-observe; count=0; while [ "$count" -lt 200 ]; do count=$((count + 1)); printf "%s\n" "$count" >/work/iflow-observe/counter.txt; sleep 1; done'"'"''
```

Checkpoint it:

```bash
python3 -m agents.iflow_integration checkpoint-manual-sandbox \
  --work-root "$WORK_ROOT"
```

List available checkpoints:

```bash
python3 -m agents.iflow_integration list-manual-checkpoints \
  --work-root "$WORK_ROOT"
```

Restore the latest checkpoint:

```bash
python3 -m agents.iflow_integration restore-manual-sandbox \
  --work-root "$WORK_ROOT"
```

Or restore a specific checkpoint id:

```bash
python3 -m agents.iflow_integration restore-manual-sandbox \
  --work-root "$WORK_ROOT" \
  --checkpoint-id ckpt-...
```

Open an interactive shell inside the restored sandbox:

```bash
python3 -m agents.iflow_integration manual-shell \
  --work-root "$WORK_ROOT"
```

Then inspect the counter from inside:

```bash
cat /work/iflow-observe/counter.txt
watch -n 1 cat /work/iflow-observe/counter.txt
```

The wrapper uses `runc exec` under the session's saved `runtime_state_root`, so
you do not need to remember the raw `runc --root ... exec ...` form.

### 8. Stop The Agent Cleanly

Tell `iflow` to stop:

```bash
python3 -m agents.iflow_integration enqueue-final-response \
  --base-url "$MANUAL_LLM_URL" \
  --sandbox-id sbx-iflow-manual \
  --content 'The session is complete. Summarize briefly and stop.'
```

Then delete the sandbox and clean up the temporary ZFS pool and netns:

```bash
python3 -m agents.iflow_integration stop-manual-sandbox \
  --work-root "$WORK_ROOT"
```

## Logs And Observation Report

The real integration test records:

- scripted LLM request/response history
- host-inspector snapshots per phase
- `recent_fs_events`, `tracked_pids`, `ignored_pids`, `current_pids`, and `dirty_pids` from inspector status
- current `/proc` identities for tracked live pids at each sampled phase
- mounted runtime strategy and mounted state paths
- `/opt/iflow-logs/iflow.stdout`
- `/opt/iflow-logs/iflow.stderr`
- a file tree snapshot from the sandbox rootfs

The consolidated report is written as `iflow_observation.json` in the test temp directory.

## Interpreting Results

- `idle_wait`: expected clean after the explicit reset. If anything changes here, inspect the report to see whether it came from tracked tool/helper processes or from a matcher gap.
- `transient_process`: expected `process_changed=False` and `filesystem_changed=False`.
- `filesystem_write`: expected `filesystem_changed=True` and `process_changed=False`.
- `detached_daemon`: expected `process_changed=True`. `filesystem_changed=True` is also expected because pid/log files are written.

For the manual control loop, these commands are covered by the real integration tests:

- `sh -lc "ls -la /work >/dev/null"`: expected `filesystem_changed=False`
- `sh -lc "cat /etc/hostname >/dev/null"`: expected `filesystem_changed=False`
- `sh -lc "touch 1.txt"`: expected `filesystem_changed=True`
- `sh -lc "echo 123 > foobat.txt"`: expected `filesystem_changed=True`

Net-zero temp churn is reconciled at file granularity, so files or directories
that are created after reset and removed again before inspection should not keep
`filesystem_changed=True`.

## Checkpoint Notes

The harness now blocks `io_uring_setup`, `io_uring_enter`, and
`io_uring_register` at container start with a minimal seccomp profile. That is
what makes the real `iflow` Node 22 runtime checkpointable on this host; the
older `UV_USE_IO_URING=0` env var alone was not sufficient.

If you need to compare behavior with and without that block, set:

```bash
AGENT_CR_IFLOW_BLOCK_IO_URING=0
```

For alternate runtime experiments, you can also point the harness at a
different host-side Node runtime with:

```bash
AGENT_CR_IFLOW_NODE_RUNTIME_DIR=/path/to/node-runtime
```

If the real agent behaves differently, inspect the generated report before changing inspector semantics.
